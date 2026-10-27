// Teil 2: Llama-Modell, Tokenizer und Export
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createCpuBackend } from '../public/js/llama/backend-cpu.js';
import { createLlama } from '../public/js/llama/model.js';
import { tensorLayout, countParams } from '../public/js/llama/config.js';
import { trainBPE, makeTokenizer } from '../public/js/llama/bpe.js';
import { makeRng } from '../public/js/rng.js';

const TINY = { vocabSize: 11, dim: 8, nLayers: 2, nHeads: 2, nKvHeads: 1, headDim: 4, ffnDim: 12, ctx: 5, ropeBase: 10000, normEps: 1e-5 };

function randomBatch(cfg, B, seed) {
  const rng = makeRng(seed);
  const n = B * cfg.ctx;
  const inp = Int32Array.from({ length: n }, () => rng.int(cfg.vocabSize));
  const tgt = Int32Array.from({ length: n }, () => rng.int(cfg.vocabSize));
  return { inp, tgt };
}

test('Gradient-Check: ganzes Llama-Modell (Float64, numerisch vs. analytisch)', async () => {
  const be = createCpuBackend(Float64Array);
  const model = createLlama(TINY, be, { batch: 2 });
  const p0 = Float64Array.from(model.initialParams(3), (x) => x * 10);  // größere Werte → aussagekräftiger
  // Norm-Gewichte nicht alle 1 lassen, sonst sieht man Fehler dort nicht.
  const rng = makeRng(9);
  for (const t of model.layout.tensors) if (t.kind === 'norm') for (let i = 0; i < t.size; i++) p0[t.offset + i] = 0.5 + rng.random();
  model.setParams(p0);
  const { inp, tgt } = randomBatch(TINY, 2, 5);
  model.setBatch(inp, tgt);
  await model.lossAndGrads();
  const grads = await model.readGrads();
  const h = 1e-5;
  let worst = 0, checked = 0;
  for (const t of model.layout.tensors) {
    for (let s = 0; s < 6; s++) {
      const i = t.offset + rng.int(t.size);
      const p = p0.slice();
      p[i] = p0[i] + h; model.setParams(p); const lp = await model.loss();
      p[i] = p0[i] - h; model.setParams(p); const lm = await model.loss();
      const num = (lp - lm) / (2 * h);
      const err = Math.abs(num - grads[i]) / Math.max(1e-6, Math.abs(num) + Math.abs(grads[i]));
      if (err > worst) worst = err;
      assert.ok(err < 1e-5, `${t.name}[${i - t.offset}]: analytisch ${grads[i]} vs. numerisch ${num}`);
      checked++;
    }
  }
  console.log(`  ${checked} Gradienten geprüft, größter relativer Fehler ${worst.toExponential(2)}`);
});

test('Parameter-Layout: Größen und GGUF-Namen', () => {
  const layout = tensorLayout(TINY);
  assert.equal(layout.tensors[0].name, 'token_embd.weight');
  assert.equal(layout.byName['blk.1.attn_k.weight'].shape.join('x'), '4x8');
  assert.ok(layout.tensors.every((t) => t.offset % 64 === 0));
  assert.equal(countParams(TINY), 11 * 8 + 2 * (8 + 8 * 8 + 4 * 8 * 2 + 8 * 8 + 8 + 3 * 12 * 8) + 8);
});

test('BPE: deterministisch, verlustfrei, beliebige Bytes darstellbar', () => {
  const docs = readFileSync(new URL('../public/data/maerchen.txt', import.meta.url), 'utf8').split('\n<|endoftext|>\n');
  const a = trainBPE(docs, 600), b = trainBPE(docs, 600);
  assert.deepEqual(a.merges, b.merges);
  assert.equal(a.vocabSize, 600);
  for (const s of ['Es war einmal ein König.', 'Grüße 🤖 aus Köln!\n\tTab', 'ÄÖÜß 123 ...']) assert.equal(a.decode(a.encode(s)), s);
  const re = makeTokenizer({ tokens: a.tokens, merges: a.merges });
  assert.deepEqual(re.encode(docs[3]), a.encode(docs[3]));
});

test('Llama lernt auf Namen (CPU)', async () => {
  const names = readFileSync(new URL('../public/data/names.txt', import.meta.url), 'utf8').split('\n').filter(Boolean);
  const tok = trainBPE(names, 256 + 64);
  const stream = [];
  for (const n of names) stream.push(tok.eot, ...tok.encode(n));
  const cfg = { vocabSize: tok.vocabSize, dim: 64, nLayers: 2, nHeads: 1, nKvHeads: 1, headDim: 64, ffnDim: 128, ctx: 32, ropeBase: 10000, normEps: 1e-5 };
  const B = 4;
  const model = createLlama(cfg, createCpuBackend(), { batch: B });
  model.setParams(model.initialParams(1));
  const rng = makeRng(2);
  const losses = [];
  const t0 = performance.now();
  for (let s = 1; s <= 60; s++) {
    const inp = new Int32Array(B * cfg.ctx), tgt = new Int32Array(B * cfg.ctx);
    for (let b = 0; b < B; b++) {
      const start = rng.int(stream.length - cfg.ctx - 1);
      for (let t = 0; t < cfg.ctx; t++) { inp[b * cfg.ctx + t] = stream[start + t]; tgt[b * cfg.ctx + t] = stream[start + t + 1]; }
    }
    model.setBatch(inp, tgt);
    losses.push(await model.trainStep({ lr: 3e-3, wd: 0.1, t: s }));
  }
  const ms = (performance.now() - t0) / 60;
  console.log(`  Loss ${losses[0].toFixed(3)} → ${losses.at(-1).toFixed(3)} (${ms.toFixed(0)} ms/Schritt, ${countParams(cfg)} Parameter)`);
  assert.ok(losses[0] > 5.5);
  assert.ok(losses.slice(-10).reduce((a, b) => a + b) / 10 < 4);
});

test('Inferenz mit KV-Cache rechnet exakt wie der Trainings-Forward', async () => {
  const { createInference } = await import('../public/js/llama/infer.js');
  const cfg = { ...TINY, ctx: 7 };
  const be = createCpuBackend(Float64Array);
  const model = createLlama(cfg, be, { batch: 1 });
  const p = Float64Array.from(model.initialParams(4), (x) => x * 8);
  model.setParams(p);
  const { inp, tgt } = randomBatch(cfg, 1, 11);
  model.setBatch(inp, tgt);
  const trainLoss = await model.loss();
  const inf = createInference(cfg, Float32Array.from(p));
  let sum = 0;
  for (let t = 0; t < cfg.ctx; t++) {
    const logits = inf.step(inp[t], t);
    const max = Math.max(...logits);
    const lse = Math.log(logits.reduce((s, l) => s + Math.exp(l - max), 0)) + max;
    sum += lse - logits[tgt[t]];
  }
  const inferLoss = sum / cfg.ctx;
  console.log(`  Loss Training ${trainLoss.toFixed(6)} | Inferenz ${inferLoss.toFixed(6)}`);
  assert.ok(Math.abs(trainLoss - inferLoss) < 1e-4);
});

test('GGUF: Datei lässt sich wieder einlesen, Metadaten und Tensoren stimmen', async () => {
  const { writeGGUF, readGGUF } = await import('../public/js/llama/gguf.js');
  const names = ['anna', 'otto', 'emma', 'jürgen'];
  const tok = trainBPE(names, 260);
  const cfg = { ...TINY, vocabSize: tok.vocabSize };
  const params = createLlama(cfg, createCpuBackend(), { batch: 1 }).initialParams(1);
  const bytes = writeGGUF(cfg, params, tok, 'test');
  const g = readGGUF(bytes);
  assert.equal(g.magic, 'GGUF');
  assert.equal(g.version, 3);
  assert.equal(g.meta['general.architecture'], 'llama');
  assert.equal(g.meta['tokenizer.ggml.tokens'].length, tok.vocabSize);
  assert.equal(g.meta['tokenizer.ggml.tokens'][tok.eot], '<|endoftext|>');
  assert.deepEqual(g.tensors.find((t) => t.name === 'blk.0.attn_k.weight').dims, [8, 4]);
  const layout = tensorLayout(cfg);
  for (const t of g.tensors) {
    const src = layout.byName[t.name];
    const data = new Float32Array(bytes.buffer.slice(bytes.byteOffset + g.dataStart + t.offset, bytes.byteOffset + g.dataStart + t.offset + src.size * 4));
    assert.deepEqual(data, params.subarray(src.offset, src.offset + src.size), t.name);
  }
});

test('F16: bekannte Werte, Rundung, Roundtrip', async () => {
  const { toHalf, fromHalf } = await import('../public/js/llama/gguf.js');
  const known = [[1, 0x3c00], [-2, 0xc000], [65504, 0x7bff], [0.1, 0x2e66], [1 / 3, 0x3555], [2 ** -24, 0x0001], [0, 0], [1e6, 0x7c00]];
  for (const [x, h] of known) assert.equal(toHalf(x), h, `toHalf(${x})`);
  for (let h = 0; h < 0x7c00; h += 37) assert.equal(toHalf(fromHalf(h)), h);   // alle Stichproben exakt darstellbar
  assert.equal(toHalf(1 + 2 ** -11), 0x3c00);          // genau in der Mitte → zur geraden Zahl
  assert.equal(toHalf(1 + 3 * 2 ** -11), 0x3c02);
});

test('Q8_0: Fehler etwa ein halber Schritt, Writer-Größen stimmen', async () => {
  const { encodeTensor, decodeTensor, writeGGUF, readGGUF, roundTripParams } = await import('../public/js/llama/gguf.js');
  const rng = makeRng(4);
  const x = Float32Array.from({ length: 32 * 50 }, () => rng.gauss(0.05));
  const bytes = encodeTensor(x, 'q8_0');
  assert.equal(bytes.length, 50 * 34);
  const y = decodeTensor(bytes, x.length, 'q8_0');
  for (let b = 0; b < 50; b++) {
    let amax = 0;
    for (let j = 0; j < 32; j++) amax = Math.max(amax, Math.abs(x[b * 32 + j]));
    // ½ Schritt Rundung plus der f16-Rundung des Faktors d selbst (127 · 2⁻¹¹ ≈ 0,06 Schritte)
    for (let j = 0; j < 32; j++) assert.ok(Math.abs(x[b * 32 + j] - y[b * 32 + j]) <= (amax / 127) * 0.57 + 1e-7);
  }
  const tok = trainBPE(['anna', 'otto', 'emma'], 260);
  const cfg = { vocabSize: tok.vocabSize, dim: 32, nLayers: 1, nHeads: 1, nKvHeads: 1, headDim: 32, ffnDim: 64, ctx: 8, ropeBase: 10000, normEps: 1e-5 };
  const params = createLlama(cfg, createCpuBackend(), { batch: 1 }).initialParams(1);
  const f32 = writeGGUF(cfg, params, tok, 't', 'f32').length;
  const f16 = writeGGUF(cfg, params, tok, 't', 'f16');
  const q8 = writeGGUF(cfg, params, tok, 't', 'q8_0');
  assert.ok(f16.length < f32 && q8.length < f16.length);
  const g = readGGUF(q8);
  assert.equal(g.meta['general.file_type'], 7);
  assert.equal(g.tensors.find((t) => t.name === 'blk.0.attn_norm.weight').type, 0);   // Normen bleiben F32
  assert.equal(g.tensors.find((t) => t.name === 'blk.0.ffn_up.weight').type, 8);
  const rt = roundTripParams(cfg, params, 'q8_0');
  let worst = 0;
  for (let i = 0; i < params.length; i++) worst = Math.max(worst, Math.abs(rt[i] - params[i]));
  console.log(`  Dateigrößen: F32 ${f32} B, F16 ${f16.length} B, Q8_0 ${q8.length} B; größter Rundungsfehler ${worst.toExponential(2)}`);
});
