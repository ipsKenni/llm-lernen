// =====================================================================
// train-worker.js – Das GPT trainiert in einem eigenen Thread
// =====================================================================
// Ein Web Worker rechnet parallel zur Seite. So bleibt alles flüssig,
// auch wenn gerade hunderttausende Value-Objekte pro Sekunde entstehen.

import { buildTokenizer, tokenizerFromItos } from './tokenizer.js';
import { makeRng } from './rng.js';
import { createGPT, createAdam, trainStep, generate, inspect, exportWeights, importWeights } from './transformer.js';

let state = null;  // { cfg, words, wordSet, tok, model, opt, rng, step }
let running = false;
const SAMPLE_EVERY = 100;

function setup(cfg, words, itos = null) {
  const tok = itos ? tokenizerFromItos(itos) : buildTokenizer(words);
  const rng = makeRng(cfg.seed);
  const model = createGPT(cfg, tok.vocabSize, rng);
  state = {
    cfg, words, tok, model, rng,
    wordSet: new Set(words),
    encoded: words.map((w) => tok.encode(w)),
    opt: createAdam(model.params.length),
    step: 0,
  };
}

function postSamples() {
  const names = Array.from({ length: 8 }, () => sampleOne(0.6));
  postMessage({ type: 'samples', step: state.step, names });
}

function sampleOne(temperature, prefix = '') {
  const ids = generate(state.model, state.rng, { temperature, prefix: state.tok.encode(prefix) });
  const name = state.tok.decode(ids);
  return { name, known: state.wordSet.has(name) };
}

function oneStep() {
  const { cfg, encoded, rng } = state;
  const batch = Array.from({ length: cfg.batchSize }, () => encoded[rng.int(encoded.length)]);
  // Lernrate fällt linear auf 0: am Anfang große Schritte, am Ende feinjustieren.
  const lr = cfg.lr * (1 - state.step / cfg.steps);
  const loss = trainStep(state.model, state.opt, batch, lr);
  state.step += 1;
  return loss;
}

// Trainiert in Häppchen von ~50 ms und meldet sich dazwischen, damit
// Pause-Nachrichten ankommen und die Kurve live wächst.
function loop() {
  if (!running) return;
  const t0 = performance.now();
  const losses = [];
  while (performance.now() - t0 < 50 && state.step < state.cfg.steps) {
    losses.push(oneStep());
    if (state.step % SAMPLE_EVERY === 0) postSamples();
  }
  const elapsed = performance.now() - t0;
  postMessage({ type: 'progress', step: state.step, losses, stepsPerSec: (losses.length / elapsed) * 1000 });
  if (state.step >= state.cfg.steps) {
    running = false;
    if (state.step % SAMPLE_EVERY !== 0) postSamples();
    postMessage({ type: 'done', step: state.step });
    return;
  }
  setTimeout(loop, 0);
}

onmessage = (e) => {
  const msg = e.data;
  const reply = (data) => postMessage({ type: 'result', id: msg.id, ...data });
  switch (msg.type) {
    case 'init':
      running = false;
      setup(msg.cfg, msg.words);
      postMessage({ type: 'ready', nParams: state.model.params.length, itos: state.tok.itos, step: 0 });
      break;
    case 'train':
      if (!state || running || state.step >= state.cfg.steps) break;
      running = true;
      postMessage({ type: 'running', running: true });
      loop();
      break;
    case 'pause':
      running = false;
      postMessage({ type: 'running', running: false });
      break;
    case 'step':
      if (!state || running || state.step >= state.cfg.steps) break;
      postMessage({ type: 'progress', step: state.step + 1, losses: [oneStep()], stepsPerSec: 0 });
      if (state.step % SAMPLE_EVERY === 0) postSamples();
      break;
    case 'generate':
      reply({ names: Array.from({ length: msg.n }, () => sampleOne(msg.temperature, msg.prefix ?? '')) });
      break;
    case 'inspect': {
      const prefix = [...msg.prefix].filter((c) => state.tok.stoi.has(c)).join('');
      reply({ prefix, ...inspect(state.model, state.tok.encode(prefix), msg.temperature) });
      break;
    }
    case 'export':
      reply({ data: { ...exportWeights(state.model, state.tok.itos), step: state.step } });
      break;
    case 'import': {
      running = false;
      const d = msg.data;
      setup(d.cfg, msg.words, d.itos);
      importWeights(state.model, d.weights);
      state.step = d.step ?? d.cfg.steps;
      postMessage({ type: 'ready', nParams: state.model.params.length, itos: state.tok.itos, step: state.step });
      break;
    }
  }
};
