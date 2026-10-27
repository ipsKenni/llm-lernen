// =====================================================================
// infer.js – Text erzeugen: ein Token nach dem anderen, mit KV-Cache
// =====================================================================
//
// Beim Training sieht das Modell ganze Textstücke auf einmal. Beim
// Generieren kommt immer nur EIN neues Token dazu. Damit wir nicht für
// jedes Token den ganzen Text neu durchrechnen, merken wir uns Keys und
// Values aller bisherigen Positionen: den KV-Cache. Genau so arbeiten
// llama.cpp und Ollama – und deshalb muss diese Datei exakt dasselbe
// rechnen wie dort, sonst kommt in Ollama etwas anderes heraus.

import { tensorLayout } from './config.js';

export function createInference(cfg, params) {
  const layout = tensorLayout(cfg);
  const { dim: d, nHeads: H, nKvHeads: KV, headDim: hd, ffnDim: F, vocabSize: V, nLayers: L, ctx } = cfg;
  const qd = H * hd, kd = KV * hd, group = H / KV;
  const W = (name) => { const t = layout.byName[name]; return params.subarray(t.offset, t.offset + t.size); };
  const w = Array.from({ length: L }, (_, l) => Object.fromEntries(
    ['attn_norm', 'attn_q', 'attn_k', 'attn_v', 'attn_output', 'ffn_norm', 'ffn_gate', 'ffn_up', 'ffn_down']
      .map((n) => [n, W(`blk.${l}.${n}.weight`)]),
  ));
  const E = W('token_embd.weight'), outNorm = W('output_norm.weight');
  const kCache = Array.from({ length: L }, () => new Float32Array(ctx * kd));
  const vCache = Array.from({ length: L }, () => new Float32Array(ctx * kd));
  const x = new Float32Array(d), xn = new Float32Array(d), q = new Float32Array(qd);
  const att = new Float32Array(ctx), o = new Float32Array(qd), tmp = new Float32Array(d);
  const g = new Float32Array(F), u = new Float32Array(F), logits = new Float32Array(V);

  const matvec = (out, M, v, rows, cols) => {
    for (let r = 0; r < rows; r++) {
      let s = 0;
      const o2 = r * cols;
      for (let c = 0; c < cols; c++) s += M[o2 + c] * v[c];
      out[r] = s;
    }
  };
  const rmsnorm = (out, v, weight) => {
    let ss = 0;
    for (let i = 0; i < d; i++) ss += v[i] * v[i];
    const r = 1 / Math.sqrt(ss / d + cfg.normEps);
    for (let i = 0; i < d; i++) out[i] = weight[i] * v[i] * r;
  };
  const rope = (vec, offset, heads, pos) => {
    for (let h = 0; h < heads; h++) {
      for (let i = 0; i < hd / 2; i++) {
        const theta = pos * Math.pow(cfg.ropeBase, (-2 * i) / hd);
        const c = Math.cos(theta), s = Math.sin(theta);
        const j = offset + h * hd + 2 * i;
        const a = vec[j], b = vec[j + 1];
        vec[j] = a * c - b * s;
        vec[j + 1] = a * s + b * c;
      }
    }
  };

  //#region infer-step
  // Ein Token an Position pos verarbeiten → Logits fürs nächste Token.
  // trace (optional) sammelt die Attention-Gewichte: trace.attn[Schicht][Kopf][pos].
  function step(token, pos, trace = null) {
    x.set(E.subarray(token * d, token * d + d));
    for (let l = 0; l < L; l++) {
      const p = w[l];
      rmsnorm(xn, x, p.attn_norm);
      matvec(q, p.attn_q, xn, qd, d);
      const kc = kCache[l], vc = vCache[l];
      matvec(kc.subarray(pos * kd, pos * kd + kd), p.attn_k, xn, kd, d);
      matvec(vc.subarray(pos * kd, pos * kd + kd), p.attn_v, xn, kd, d);
      rope(q, 0, H, pos);
      rope(kc, pos * kd, KV, pos);
      for (let h = 0; h < H; h++) {
        const kh = Math.floor(h / group);
        let max = -Infinity;
        for (let t = 0; t <= pos; t++) {
          let s = 0;
          for (let i = 0; i < hd; i++) s += q[h * hd + i] * kc[t * kd + kh * hd + i];
          att[t] = s / Math.sqrt(hd);
          if (att[t] > max) max = att[t];
        }
        let sum = 0;
        for (let t = 0; t <= pos; t++) { att[t] = Math.exp(att[t] - max); sum += att[t]; }
        if (trace) trace.attn[l][h][pos] = Array.from({ length: pos + 1 }, (_, t) => att[t] / sum);
        for (let i = 0; i < hd; i++) {
          let acc = 0;
          for (let t = 0; t <= pos; t++) acc += att[t] * vc[t * kd + kh * hd + i];
          o[h * hd + i] = acc / sum;
        }
      }
      matvec(tmp, p.attn_output, o, d, qd);
      for (let i = 0; i < d; i++) x[i] += tmp[i];
      rmsnorm(xn, x, p.ffn_norm);
      matvec(g, p.ffn_gate, xn, F, d);
      matvec(u, p.ffn_up, xn, F, d);
      for (let i = 0; i < F; i++) g[i] = (g[i] / (1 + Math.exp(-g[i]))) * u[i];
      matvec(tmp, p.ffn_down, g, d, F);
      for (let i = 0; i < d; i++) x[i] += tmp[i];
    }
    rmsnorm(xn, x, outNorm);
    matvec(logits, E, xn, V, d);
    return logits;
  }
  //#endregion

  return { step, ctx };
}

//#region infer-sample
// Aus den Logits ein Token ziehen.
//   temperature: < 1 vorsichtiger, > 1 wilder, 0 = immer das wahrscheinlichste
//   topK:        nur die k besten Kandidaten zulassen
//   topP:        nur so viele Kandidaten, bis zusammen p Wahrscheinlichkeit erreicht ist
export function sample(logits, { temperature = 0.8, topK = 40, topP = 0.95 } = {}, random = Math.random) {
  let best = 0;
  for (let i = 1; i < logits.length; i++) if (logits[i] > logits[best]) best = i;
  if (temperature <= 0) return best;
  let ids = Array.from(logits.keys()).sort((a, b) => logits[b] - logits[a]);
  if (topK > 0) ids = ids.slice(0, topK);
  const max = logits[ids[0]];
  let probs = ids.map((i) => Math.exp((logits[i] - max) / temperature));
  const sum = probs.reduce((a, b) => a + b, 0);
  probs = probs.map((p) => p / sum);
  let cum = 0, cut = probs.length;
  for (let i = 0; i < probs.length; i++) { cum += probs[i]; if (cum >= topP) { cut = i + 1; break; } }
  const kept = probs.slice(0, cut), keptSum = kept.reduce((a, b) => a + b, 0);
  let r = random() * keptSum;
  for (let i = 0; i < cut; i++) { r -= kept[i]; if (r <= 0) return ids[i]; }
  return ids[cut - 1];
}
//#endregion

// Text fortsetzen. Wie in Ollama beginnt jede Eingabe mit dem BOS-Token.
// onToken(id) wird für jedes neue Token aufgerufen; false bricht ab.
export function generate(cfg, params, tok, prompt, { maxTokens = 200, onToken, ...samplingOpts } = {}, random = Math.random) {
  const inf = createInference(cfg, params);
  const ids = [tok.eot, ...tok.encode(prompt)].slice(-cfg.ctx);
  let logits;
  ids.forEach((id, pos) => { logits = inf.step(id, pos); });
  const out = [];
  for (let n = 0; n < maxTokens && ids.length < cfg.ctx; n++) {
    const next = sample(logits, samplingOpts, random);
    if (next === tok.eot) break;
    out.push(next);
    ids.push(next);
    if (onToken && onToken(next) === false) break;
    logits = inf.step(next, ids.length - 1);
  }
  return out;
}
