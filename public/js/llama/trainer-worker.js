// =====================================================================
// trainer-worker.js – Teil 2: Tokenizer, Training und Export im Hintergrund
// =====================================================================

import { trainBPE, makeTokenizer } from './bpe.js';
import { makeConfig, PRESETS, countParams } from './config.js';
import { createCpuBackend } from './backend-cpu.js';
import { createWebGpuBackend, ENGINE_VERSION } from './backend-webgpu.js';
import { createLlama } from './model.js';
import { DATASETS, splitDocs, buildStreams, sampleBatch, learningRate } from './data.js';
import { generate } from './infer.js';
import { writeGGUF, readGGUF, makeModelfile } from './gguf.js';
import { saveCheckpoint, loadCheckpoint, deleteCheckpoint } from './checkpoint.js';
import { makeRng } from '../rng.js';

const CHECKPOINT = 'aktuell';
const S = {
  dataset: null, uploadText: '', tok: null, streams: null, docsCount: 0, chars: 0,
  preset: null, cfg: null, backend: null, model: null, rng: makeRng(1),
  step: 0, total: 1000, maxLr: 1e-3, running: false, history: [], evals: [], samples: [],
  paramsCache: null, paramsStep: -1,
};

const post = (msg, transfer) => postMessage(msg, transfer || []);

async function loadText(datasetKey, uploadText) {
  const ds = DATASETS[datasetKey];
  if (!ds.file) return uploadText;
  const res = await fetch(new URL(`../../data/${ds.file}`, import.meta.url));
  if (!res.ok) throw new Error(`Datensatz ${ds.file} konnte nicht geladen werden (HTTP ${res.status}).`);
  return res.text();
}

async function tokenize({ dataset, text, vocabSize }) {
  S.running = false;
  const raw = await loadText(dataset, text);
  const docs = splitDocs(dataset, raw);
  if (docs.length === 0) throw new Error('Der Text ist leer.');
  let batch = [];
  const tok = trainBPE(docs, vocabSize, {
    onMerge(m) {
      batch.push({ rank: m.rank, a: m.a, b: m.b, id: m.id, count: m.count });
      if (batch.length >= 64) { post({ type: 'merges', items: batch }); batch = []; }
    },
  });
  if (batch.length) post({ type: 'merges', items: batch });
  Object.assign(S, { dataset, uploadText: dataset === 'upload' ? raw : '', tok, docsCount: docs.length, chars: docs.reduce((s, d) => s + d.length, 0) });
  S.streams = buildStreams(docs, tok);
  S.model = null;
  S.step = 0;
  return {
    tokens: tok.tokens, merges: tok.merges, docs: docs.length, chars: S.chars,
    trainTokens: S.streams.train.length, valTokens: S.streams.val.length,
  };
}

async function setup({ preset, prefer = 'auto' }) {
  if (!S.tok) throw new Error('Zuerst den Tokenizer trainieren.');
  S.running = false;
  const p = PRESETS[preset];
  S.cfg = makeConfig(p, S.tok.vocabSize);
  S.preset = preset;
  S.maxLr = p.lr;
  if (S.backend?.destroy) S.backend.destroy();
  S.backend = prefer === 'cpu' ? null : await createWebGpuBackend().catch(() => null);
  if (!S.backend) S.backend = createCpuBackend();
  S.model = createLlama(S.cfg, S.backend, { batch: p.batch });
  S.model.setParams(S.model.initialParams(1));
  S.tuning = (await S.backend.autotune?.(S.model.matmulShapes())) ?? null;
  Object.assign(S, { step: 0, history: [], evals: [], samples: [], paramsCache: null, paramsStep: -1, rng: makeRng(7) });
  return describe();
}

function describe() {
  return {
    backend: S.backend.kind, adapter: S.backend.adapterInfo || null, preset: S.preset, cfg: S.cfg,
    nParams: countParams(S.cfg), batch: S.model.batch, step: S.step, total: S.total,
    dataset: S.dataset, history: S.history, evals: S.evals, samples: S.samples,
    tokenizer: { vocab: S.tok.vocabSize }, trainTokens: S.streams?.train.length ?? 0,
  };
}

async function currentParams() {
  if (S.paramsStep !== S.step || !S.paramsCache) {
    S.paramsCache = await S.model.readParams();
    S.paramsStep = S.step;
  }
  return S.paramsCache;
}

async function evaluate() {
  const { cfg, model } = S;
  const B = model.batch;
  let sum = 0;
  const rng = makeRng(99);   // immer dieselben Validierungs-Ausschnitte → vergleichbar
  for (let i = 0; i < 4; i++) {
    const { inp, tgt } = sampleBatch(S.streams.val, B, cfg.ctx, rng);
    model.setBatch(inp, tgt);
    sum += await model.loss();
  }
  const valLoss = sum / 4;
  const params = await currentParams();
  const prompt = DATASETS[S.dataset].prompt;
  const ids = generate(cfg, params, S.tok, prompt, { maxTokens: 60, temperature: 0.8, topK: 40, topP: 0.95 }, S.rng.random);
  const sample = { step: S.step, prompt, text: S.tok.decode(ids) };
  S.evals.push({ step: S.step, valLoss });
  S.samples.push(sample);
  post({ type: 'eval', step: S.step, valLoss, sample });
}

async function save() {
  if (!S.model) return;
  const params = await currentParams();
  const { m, v } = await S.model.readOptState();
  await saveCheckpoint(CHECKPOINT, {
    version: 1, savedAt: Date.now(), dataset: S.dataset, uploadText: S.uploadText, preset: S.preset, cfg: S.cfg,
    tokenizer: { tokens: S.tok.tokens, merges: S.tok.merges }, params, m, v,
    step: S.step, total: S.total, maxLr: S.maxLr, history: S.history, evals: S.evals, samples: S.samples,
  });
  post({ type: 'saved', step: S.step });
}

async function restore() {
  const c = await loadCheckpoint(CHECKPOINT);
  if (!c || c.version !== 1) return null;
  S.tok = makeTokenizer(c.tokenizer);
  const raw = await loadText(c.dataset, c.uploadText);
  const docs = splitDocs(c.dataset, raw);
  Object.assign(S, { dataset: c.dataset, uploadText: c.uploadText, docsCount: docs.length, chars: docs.reduce((s, d) => s + d.length, 0) });
  S.streams = buildStreams(docs, S.tok);
  await setup({ preset: c.preset });
  S.model.setParams(c.params);
  S.model.setOptState(c.m, c.v);
  Object.assign(S, { step: c.step, total: c.total, maxLr: c.maxLr, history: c.history, evals: c.evals, samples: c.samples, paramsCache: null });
  return { ...describe(), tokens: S.tok.tokens, merges: S.tok.merges, docs: docs.length, chars: S.chars, trainTokens: S.streams.train.length, valTokens: S.streams.val.length };
}

//#region llama-train-loop
async function trainLoop() {
  const { cfg, model } = S;
  const B = model.batch;
  const evalEvery = Math.max(25, Math.round(S.total / 40));
  let t0 = performance.now(), stepsSince = 0, lastSave = performance.now();
  // Bis zu zwei Schritte gleichzeitig unterwegs: Während die GPU rechnet,
  // bereitet JavaScript schon den nächsten Batch vor (Pipelining).
  const pending = [];
  const settle = async () => {
    const loss = await pending.shift();
    if (!Number.isFinite(loss)) throw new Error('Der Loss ist explodiert (NaN). Versuch es mit einem kleineren Modell oder starte neu.');
    S.history.push(loss);
  };
  while (S.running && S.step < S.total) {
    const chunkStart = performance.now();
    // Mehrere Schritte am Stück, dann kurz Luft holen (Nachrichten annehmen).
    while (S.running && S.step < S.total && performance.now() - chunkStart < 120) {
      const { inp, tgt } = sampleBatch(S.streams.train, B, cfg.ctx, S.rng);
      model.setBatch(inp, tgt);
      const lr = learningRate(S.step, S.total, S.maxLr);
      pending.push(model.queueStep({ lr, wd: 0.1, t: S.step + 1, maxNorm: 1 }));
      S.step += 1;
      stepsSince += 1;
      if (pending.length >= 2) await settle();
      if (S.step % evalEvery === 0 || S.step === S.total) {
        while (pending.length) await settle();
        await evaluate();
      }
    }
    while (pending.length) await settle();
    const elapsed = (performance.now() - t0) / 1000;
    post({
      type: 'progress', step: S.step, total: S.total, losses: S.history.slice(S.history.length - stepsSince),
      lr: learningRate(S.step, S.total, S.maxLr), stepsPerSec: stepsSince / elapsed, tokensPerSec: (stepsSince * B * cfg.ctx) / elapsed,
    });
    t0 = performance.now();
    stepsSince = 0;
    if (performance.now() - lastSave > 60000) { await save(); lastSave = performance.now(); }
    await new Promise((r) => setTimeout(r, 0));
  }
  if (S.step >= S.total) post({ type: 'done', step: S.step });
  S.running = false;
  post({ type: 'state', running: false });
  await save();
}
//#endregion

//#region llama-benchmark
// Misst echte Trainingsschritte mit Zufallsdaten. Jede Größe bekommt ihr eigenes
// GPU-Device, das danach wieder freigegeben wird.
// Zeitbudget: pro Größe höchstens ~10 s messen; ist ein Schritt schon länger als
// 15 s, sind die größeren Modelle hier praktisch nicht trainierbar → überspringen.
const BENCH_BUDGET_MS = 10000;
const BENCH_GIVE_UP_MS = 15000;

async function measure(cfg, p, be, { legacy = false } = {}) {
  be.pushOomScope?.();
  const model = createLlama(cfg, be, { batch: p.batch, fused: !legacy });
  model.setParams(model.initialParams(1));
  const err = await be.popOomScope?.();
  if (err) throw new Error('nicht genug Grafikspeicher');
  if (legacy) be.setMatmulVariant?.({ NT: 0, NN: 0, TN: 0 });
  else await be.autotune?.(model.matmulShapes());
  const rng = makeRng(5);
  const fake = Int32Array.from({ length: 4096 }, () => rng.int(cfg.vocabSize));
  const hp = (t) => ({ lr: 1e-4, wd: 0.1, t, maxNorm: 1 });
  const feed = () => { const { inp, tgt } = sampleBatch(fake, p.batch, cfg.ctx, rng); model.setBatch(inp, tgt); };
  // Aufwärmen: Shader kompilieren, Speicher anlegen
  const w0 = performance.now();
  feed(); await model.trainStep(hp(1));
  if (be.kind !== 'cpu') { feed(); await model.trainStep(hp(2)); }
  const warmMs = performance.now() - w0;
  const t0 = performance.now();
  let n = 0;
  const pending = [];
  while (n < 10 && (n < 2 || performance.now() - t0 < BENCH_BUDGET_MS)) {
    feed();
    if (legacy) await model.trainStep(hp(3 + n));          // Engine 1: auf jeden Schritt warten
    else { pending.push(model.queueStep(hp(3 + n))); if (pending.length >= 2) await pending.shift(); }
    n += 1;
    if (performance.now() - t0 > BENCH_GIVE_UP_MS) break;
  }
  await Promise.all(pending);
  const sec = (performance.now() - t0) / 1000 / n;
  return { msPerStep: sec * 1000, tokensPerSec: (p.batch * cfg.ctx) / sec, minutesPer1000: (sec * 1000) / 60, steps: n, warmMs };
}

async function benchmark({ presets, prefer = 'auto', vocab = 2048 }) {
  const results = [];
  let adapter = null, kind = 'cpu', gflops = null, tooSlow = false, compare = null;
  const newBackend = async () => (prefer === 'cpu' ? null : await createWebGpuBackend().catch(() => null)) || createCpuBackend();
  for (const key of presets) {
    const p = PRESETS[key];
    const cfg = makeConfig(p, vocab);
    const entry = { preset: key, params: countParams(cfg), batch: p.batch, ctx: p.ctx };
    if (tooSlow) {
      entry.error = 'übersprungen, schon die kleinere Größe war sehr langsam';
      results.push(entry);
      post({ type: 'benchmark', entry });
      continue;
    }
    const be = await newBackend();
    kind = be.kind;
    adapter = be.adapterInfo || null;
    try {
      if (gflops === null) gflops = await matmulGflops(be);
      const m = await measure(cfg, p, be);
      Object.assign(entry, m);
      if (be.kind !== 'cpu') entry.variant = be.matmulVariant;
      if (m.msPerStep > BENCH_GIVE_UP_MS || m.warmMs > 4 * BENCH_GIVE_UP_MS) tooSlow = true;
    } catch (e) {
      entry.error = e.message;
    } finally {
      be.destroy?.();
    }
    results.push(entry);
    post({ type: 'benchmark', entry });
  }
  // Vergleich mit Engine 1 (ohne Pipelining, Fusion und Autotuning) an der
  // größten Größe, die noch flott läuft.
  const pick = [...results].reverse().find((e) => !e.error && e.msPerStep < 3000);
  if (pick && kind === 'webgpu') {
    const be = await newBackend();
    try {
      const p = PRESETS[pick.preset];
      const old = await measure(makeConfig(p, vocab), p, be, { legacy: true });
      compare = { preset: pick.preset, oldMs: old.msPerStep, newMs: pick.msPerStep, speedup: old.msPerStep / pick.msPerStep };
    } catch (e) {
      compare = { preset: pick.preset, error: e.message };
    } finally {
      be.destroy?.();
    }
  }
  return { kind, adapter, gflops, results, compare, engine: ENGINE_VERSION };
}

// Reine Rechenleistung: C = A · Bᵀ mit 1024×1024-Matrizen (CPU: 256×256).
async function matmulGflops(be) {
  const n = be.kind === 'cpu' ? 256 : 1024, reps = be.kind === 'cpu' ? 2 : 20;
  const A = be.alloc(n * n), B = be.alloc(n * n), C = be.alloc(n * n);
  const rng = makeRng(1);
  const data = Float32Array.from({ length: n * n }, () => rng.random() - 0.5);
  be.write(A, data); be.write(B, data);
  be.matmul(C, A, B, n, n, n, false, true);
  await be.read(be.view(C, 0, 1));
  const t0 = performance.now();
  for (let r = 0; r < reps; r++) be.matmul(C, A, B, n, n, n, false, true);
  await be.read(be.view(C, 0, 1));
  return (2 * n * n * n * reps) / ((performance.now() - t0) / 1000) / 1e9;
}
//#endregion

async function exportModel({ name }) {
  const params = await currentParams();
  const bytes = writeGGUF(S.cfg, params, S.tok, name);
  const info = readGGUF(bytes);
  const prompt = DATASETS[S.dataset].prompt || S.tok.decode(S.streams.train.slice(1, 4));
  const greedy = S.tok.decode(generate(S.cfg, params, S.tok, prompt, { maxTokens: 40, temperature: 0 }));
  const meta = Object.entries(info.meta).map(([k, v]) => [k, Array.isArray(v) ? `${v.length} Einträge` : String(v)]);
  return { gguf: bytes, modelfile: makeModelfile(`${name}.gguf`, S.cfg), meta, tensors: info.tensors.length, prompt, greedy };
}

onmessage = async (e) => {
  const msg = e.data;
  const reply = (data, transfer) => post({ type: 'result', id: msg.id, ...data }, transfer);
  try {
    switch (msg.type) {
      case 'tokenize': reply(await tokenize(msg)); break;
      case 'setup': S.total = msg.total ?? S.total; reply(await setup(msg)); break;
      case 'restore': reply({ state: await restore() }); break;
      case 'describe': reply({ state: S.model ? describe() : null }); break;
      case 'train':
        if (!S.model || S.running) break;
        if (msg.total) S.total = Math.max(msg.total, S.step + 1);
        S.running = true;
        post({ type: 'state', running: true });
        trainLoop().catch((err) => { S.running = false; post({ type: 'error', message: err.message }); post({ type: 'state', running: false }); });
        break;
      case 'pause': S.running = false; break;
      case 'generate': {
        const params = await currentParams();
        let text = '';
        const ids = [];
        generate(S.cfg, params, S.tok, msg.prompt, {
          maxTokens: msg.maxTokens, temperature: msg.temperature, topK: msg.topK, topP: msg.topP,
          onToken(id) { ids.push(id); if (ids.length % 8 === 0) { text = S.tok.decode(ids); post({ type: 'token', id: msg.id, text }); } },
        });
        reply({ text: S.tok.decode(ids), tokens: ids.map((id) => S.tok.display(id)) });
        break;
      }
      case 'export': { const r = await exportModel(msg); reply(r, [r.gguf.buffer]); break; }
      case 'benchmark':
        if (S.running) throw new Error('Während des Trainings kann nicht gemessen werden. Pausiere zuerst.');
        reply(await benchmark(msg));
        break;
      case 'reset': S.running = false; await deleteCheckpoint(CHECKPOINT); reply({}); break;
      default: break;
    }
  } catch (err) {
    reply({ error: err.message });
  }
};
