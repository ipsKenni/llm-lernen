// =====================================================================
// worker-client.js – Die Seite spricht mit dem Trainings-Worker
// =====================================================================
// Hält den gemeinsamen Zustand, den Schritt 5 (Training) und Schritt 6
// (Generieren) beide brauchen.

import { DEFAULT_CONFIG, countParams } from './transformer.js';

const STORAGE_KEY = 'kleinstes-llm:gpt';

export function createGptClient() {
  const worker = new Worker(new URL('./train-worker.js', import.meta.url), { type: 'module' });
  const listeners = new Set();
  const pending = new Map();
  let nextId = 1;

  const state = {
    cfg: { ...DEFAULT_CONFIG },
    dataset: { kind: 'names', text: '' },
    words: [],
    ready: false,
    running: false,
    nParams: 0,
    itos: [],
    step: 0,
    losses: [],
    samples: [],      // [{ step, names: [{ name, known }] }]
    stepsPerSec: 0,
  };

  // Ohne diese Meldung würde ein kaputter Worker einfach schweigen.
  worker.onerror = (e) => {
    e.preventDefault?.();
    window.showBootError?.(`Der Trainings-Worker ist abgestürzt: ${e.message || 'unbekannter Fehler (evtl. unterstützt der Browser keine Module in Web Workern)'}`);
  };

  worker.onmessage = (e) => {
    const m = e.data;
    if (m.type === 'result') {
      pending.get(m.id)?.(m);
      pending.delete(m.id);
      return;
    }
    if (m.type === 'ready') {
      Object.assign(state, { ready: true, running: false, nParams: m.nParams, itos: m.itos, step: m.step });
    } else if (m.type === 'progress') {
      state.step = m.step;
      state.losses.push(...m.losses);
      if (m.stepsPerSec) state.stepsPerSec = m.stepsPerSec;
    } else if (m.type === 'samples') {
      state.samples.push({ step: m.step, names: m.names });
    } else if (m.type === 'running') {
      state.running = m.running;
    } else if (m.type === 'done') {
      state.running = false;
      save();
    }
    listeners.forEach((fn) => fn(m, state));
  };

  function request(msg) {
    const id = nextId++;
    return new Promise((resolve) => {
      pending.set(id, resolve);
      worker.postMessage({ ...msg, id });
    });
  }

  function init(cfg, dataset, words) {
    Object.assign(state, { cfg, dataset, words, ready: false, running: false, step: 0, losses: [], samples: [], stepsPerSec: 0 });
    worker.postMessage({ type: 'init', cfg, words });
  }

  // Trainiertes Modell im Browser merken (nur lokal, nur für dich).
  async function save() {
    try {
      const { data } = await request({ type: 'export' });
      const payload = { data, dataset: state.dataset, losses: state.losses, samples: state.samples };
      localStorage.setItem(STORAGE_KEY, JSON.stringify(payload));
    } catch { /* Speicher voll oder gesperrt – dann eben nicht */ }
  }

  async function restore(loadWords) {
    try {
      const payload = JSON.parse(localStorage.getItem(STORAGE_KEY) || 'null');
      if (!payload?.data?.weights) return false;
      const words = await loadWords(payload.dataset);
      Object.assign(state, {
        cfg: payload.data.cfg, dataset: payload.dataset, words,
        losses: payload.losses || [], samples: payload.samples || [], ready: false,
      });
      worker.postMessage({ type: 'import', data: payload.data, words });
      return true;
    } catch {
      return false;
    }
  }

  function importData(data, dataset, words) {
    Object.assign(state, { cfg: data.cfg, dataset, words, losses: [], samples: [], ready: false });
    worker.postMessage({ type: 'import', data, words });
  }

  return {
    state,
    init,
    restore,
    save,
    importData,
    train: () => worker.postMessage({ type: 'train' }),
    pause: () => worker.postMessage({ type: 'pause' }),
    step: () => worker.postMessage({ type: 'step' }),
    generate: (n, temperature, prefix = '') => request({ type: 'generate', n, temperature, prefix }),
    inspect: (prefix, temperature = 1) => request({ type: 'inspect', prefix, temperature }),
    exportWeights: () => request({ type: 'export' }),
    on: (fn) => { listeners.add(fn); return () => listeners.delete(fn); },
    countParams: (cfg, vocabSize) => countParams(cfg, vocabSize),
    isTrained: () => state.ready && state.step > 0,
  };
}
