// =====================================================================
// model.js – Das Llama-Modell: Vorwärts, Rückwärts, Lernschritt
// =====================================================================
//
// Diese Datei kennt keine Zahlen, nur die Reihenfolge der Operationen.
// Gerechnet wird im Backend: backend-cpu.js (JavaScript) oder
// backend-webgpu.js (Grafikkarte). Beide bekommen exakt dieselben Befehle.
//
// Ablauf für einen Batch aus B Textstücken mit je T Tokens (N = B·T):
//
//   Tokens ─► Embedding ─► x
//   für jede Schicht:
//     x ─► RMSNorm ─► Q, K, V ─► RoPE ─► Attention ─► W_o ─► + x   (Residual)
//     x ─► RMSNorm ─► gate, up ─► SwiGLU ─► down       ─► + x   (Residual)
//   x ─► RMSNorm ─► · Embeddingᵀ ─► Logits ─► Cross-Entropy ─► Loss
//
// Rückwärts läuft alles in umgekehrter Reihenfolge; jede Operation hat
// ihr handgeschriebenes Gegenstück (…Bwd).

import { tensorLayout } from './config.js';
import { makeRng } from '../rng.js';

export function createLlama(cfg, backend, { batch }) {
  const L = cfg.nLayers, T = cfg.ctx, B = batch, N = B * T;
  const { dim: d, nHeads: H, nKvHeads: KV, headDim: hd, ffnDim: F, vocabSize: V } = cfg;
  const qd = H * hd, kd = KV * hd;
  const layout = tensorLayout(cfg);
  const be = backend;

  // Parameter, Gradienten und die beiden Adam-Momente: je ein großer Buffer.
  const params = be.alloc(layout.total);
  const grads = be.alloc(layout.total);
  const adamM = be.alloc(layout.total);
  const adamV = be.alloc(layout.total);
  const P = (name) => { const t = layout.byName[name]; return be.view(params, t.offset, t.size); };
  const G = (name) => { const t = layout.byName[name]; return be.view(grads, t.offset, t.size); };

  // Zwischenergebnisse, die der Rückwärtslauf braucht (pro Schicht).
  const xs = Array.from({ length: L + 1 }, () => be.alloc(N * d));   // Residual-Strom vor jeder Schicht
  const acts = Array.from({ length: L }, () => ({
    a: be.alloc(N * d), r1: be.alloc(N),
    q: be.alloc(N * qd), k: be.alloc(N * kd), v: be.alloc(N * kd),
    att: be.alloc(B * H * T * T), o: be.alloc(N * qd),
    xMid: be.alloc(N * d), m: be.alloc(N * d), r2: be.alloc(N),
    g: be.alloc(N * F), u: be.alloc(N * F), h: be.alloc(N * F),
  }));
  const f = be.alloc(N * d), r3 = be.alloc(N), logits = be.alloc(N * V);
  // Arbeitsspeicher für den Rückwärtslauf
  const tmp = be.alloc(N * d), dx = be.alloc(N * d), dm = be.alloc(N * d);
  const dq = be.alloc(N * qd), dk = be.alloc(N * kd), dv = be.alloc(N * kd), dO = be.alloc(N * qd);
  const dS = be.alloc(B * H * T * T);
  const dg = be.alloc(N * F), du = be.alloc(N * F), dh = be.alloc(N * F);
  const tokens = be.allocInt(N), targets = be.allocInt(N);
  const lossBuf = be.alloc(1), normBuf = be.alloc(1);

  // -------------------------------------------------------------------
  //#region llama-forward
  function forward() {
    be.embed(xs[0], P('token_embd.weight'), tokens, N, d);
    for (let l = 0; l < L; l++) {
      const A = acts[l], p = (n) => P(`blk.${l}.${n}.weight`);
      // Attention-Hälfte
      be.rmsnorm(A.a, A.r1, xs[l], p('attn_norm'), N, d, cfg.normEps);
      be.matmul(A.q, A.a, p('attn_q'), N, qd, d, false, true);
      be.matmul(A.k, A.a, p('attn_k'), N, kd, d, false, true);
      be.matmul(A.v, A.a, p('attn_v'), N, kd, d, false, true);
      be.rope(A.q, N, T, H, hd, cfg.ropeBase);
      be.rope(A.k, N, T, KV, hd, cfg.ropeBase);
      be.attention(A.o, A.att, A.q, A.k, A.v, B, T, H, KV, hd);
      be.matmul(tmp, A.o, p('attn_output'), N, d, qd, false, true);
      be.copy(A.xMid, xs[l], N * d);
      be.add(A.xMid, tmp, N * d);
      // MLP-Hälfte (SwiGLU)
      be.rmsnorm(A.m, A.r2, A.xMid, p('ffn_norm'), N, d, cfg.normEps);
      be.matmul(A.g, A.m, p('ffn_gate'), N, F, d, false, true);
      be.matmul(A.u, A.m, p('ffn_up'), N, F, d, false, true);
      be.swiglu(A.h, A.g, A.u, N * F);
      be.matmul(tmp, A.h, p('ffn_down'), N, d, F, false, true);
      be.copy(xs[l + 1], A.xMid, N * d);
      be.add(xs[l + 1], tmp, N * d);
    }
    be.rmsnorm(f, r3, xs[L], P('output_norm.weight'), N, d, cfg.normEps);
    // Weight Tying: Die Ausgabeschicht ist die transponierte Embedding-Matrix.
    be.matmul(logits, f, P('token_embd.weight'), N, V, d, false, true);
    be.crossEntropy(logits, targets, N, V, lossBuf);   // logits enthält danach dLoss/dLogits
  }
  //#endregion

  // -------------------------------------------------------------------
  //#region llama-backward
  function backward() {
    const E = P('token_embd.weight'), gE = G('token_embd.weight');
    be.matmul(tmp, logits, E, N, d, V, false, false);       // dF = dLogits · E
    be.matmul(gE, logits, f, V, d, N, true, false);         // dE = dLogitsᵀ · F
    be.zero(dx, N * d);
    be.rmsnormBwd(dx, G('output_norm.weight'), tmp, xs[L], P('output_norm.weight'), r3, N, d);
    for (let l = L - 1; l >= 0; l--) {
      const A = acts[l], p = (n) => P(`blk.${l}.${n}.weight`), g = (n) => G(`blk.${l}.${n}.weight`);
      // MLP-Hälfte rückwärts.  dx ist hier dLoss/dx_{l+1} = dLoss/dxMid über den Residual-Pfad.
      be.matmul(g('ffn_down'), dx, A.h, d, F, N, true, false);
      be.matmul(dh, dx, p('ffn_down'), N, F, d, false, false);
      be.swigluBwd(dg, du, dh, A.g, A.u, N * F);
      be.matmul(g('ffn_gate'), dg, A.m, F, d, N, true, false);
      be.matmul(g('ffn_up'), du, A.m, F, d, N, true, false);
      be.matmul(dm, dg, p('ffn_gate'), N, d, F, false, false);
      be.matmul(dm, du, p('ffn_up'), N, d, F, false, false, 1);
      be.rmsnormBwd(dx, g('ffn_norm'), dm, A.xMid, p('ffn_norm'), A.r2, N, d);
      // Attention-Hälfte rückwärts
      be.matmul(g('attn_output'), dx, A.o, d, qd, N, true, false);
      be.matmul(dO, dx, p('attn_output'), N, qd, d, false, false);
      be.attentionBwd(dq, dk, dv, dS, dO, A.att, A.q, A.k, A.v, B, T, H, KV, hd);
      be.rope(dq, N, T, H, hd, cfg.ropeBase, true);
      be.rope(dk, N, T, KV, hd, cfg.ropeBase, true);
      be.matmul(g('attn_q'), dq, A.a, qd, d, N, true, false);
      be.matmul(g('attn_k'), dk, A.a, kd, d, N, true, false);
      be.matmul(g('attn_v'), dv, A.a, kd, d, N, true, false);
      be.matmul(dm, dq, p('attn_q'), N, d, qd, false, false);
      be.matmul(dm, dk, p('attn_k'), N, d, kd, false, false, 1);
      be.matmul(dm, dv, p('attn_v'), N, d, kd, false, false, 1);
      be.rmsnormBwd(dx, g('attn_norm'), dm, xs[l], p('attn_norm'), A.r1, N, d);
    }
    be.embedBwd(gE, dx, tokens, N, d);                      // Embedding bekommt beide Anteile
  }
  //#endregion

  //#region llama-step
  // Ein Lernschritt: vorwärts, rückwärts, Gradienten begrenzen, AdamW.
  async function trainStep(hp) {
    forward();
    backward();
    be.sumSquares(grads, layout.total, normBuf);
    for (const t of layout.tensors) {
      const decay = t.kind === 'matrix' || t.kind === 'residual' ? hp.wd : 0;
      be.adamw(be.view(params, t.offset, t.size), be.view(grads, t.offset, t.size),
        be.view(adamM, t.offset, t.size), be.view(adamV, t.offset, t.size), t.size,
        { lr: hp.lr, beta1: 0.9, beta2: 0.95, eps: 1e-8, wd: decay, t: hp.t, maxNorm: hp.maxNorm ?? 1 }, normBuf);
    }
    await be.flush();
    const [loss] = await be.read(lossBuf);
    return loss;
  }
  //#endregion

  // Startwerte: Normalverteilung mit σ = 0,02; die Projektionen zurück in den
  // Residual-Strom kleiner (σ / √(2L)), Norm-Gewichte = 1.
  function initialParams(seed = 1) {
    const rng = makeRng(seed);
    const out = new Float32Array(layout.total);
    for (const t of layout.tensors) {
      const std = t.kind === 'residual' ? 0.02 / Math.sqrt(2 * L) : 0.02;
      for (let i = 0; i < t.size; i++) out[t.offset + i] = t.kind === 'norm' ? 1 : rng.gauss(std);
    }
    return out;
  }

  return {
    cfg, layout, batch: B, params, grads,
    setParams: (arr) => be.write(params, arr),
    setOptState: (m, v) => { be.write(adamM, m); be.write(adamV, v); },
    readParams: () => be.read(params),
    readGrads: () => be.read(grads),
    readOptState: async () => ({ m: await be.read(adamM), v: await be.read(adamV) }),
    // inputs/outputs: Int32Array mit je N Token-IDs (Eingabe und um 1 verschobenes Ziel)
    setBatch: (inp, tgt) => { be.write(tokens, inp); be.write(targets, tgt); },
    initialParams,
    trainStep,
    // Nur Loss + Gradienten, ohne Update (für Tests und Validierung).
    async lossAndGrads() { forward(); backward(); await be.flush(); return (await be.read(lossBuf))[0]; },
    async loss() { forward(); await be.flush(); return (await be.read(lossBuf))[0]; },
  };
}
