// Gemeinsamer Zustand für Teil 2 (Schritte 7–10): ein Worker, ein Modell.
import { createTrainerClient, detectGpu } from '../llama/trainer-client.js';
import { makeTokenizer } from '../llama/bpe.js';
import { DATASETS } from '../llama/data.js';

let store = null;

export function part2() {
  if (store) return store;
  const trainer = createTrainerClient();
  const listeners = new Set();
  const state = {
    dataset: 'maerchen', vocab: DATASETS.maerchen.vocab, preset: DATASETS.maerchen.preset, total: 2000,
    tokInfo: null, tokenizer: null, gpu: undefined,
    setup: null, running: false, step: 0, losses: [], evals: [], samples: [], ft: null,
    stepsPerSec: 0, tokensPerSec: 0, lr: 0, savedStep: 0, error: '',
  };
  const emit = (what) => listeners.forEach((fn) => fn(what, state));

  trainer.on((m) => {
    if (m.type === 'progress') {
      Object.assign(state, { step: m.step, total: m.total, stepsPerSec: m.stepsPerSec, tokensPerSec: m.tokensPerSec, lr: m.lr });
      state.losses.push(...m.losses);
    } else if (m.type === 'eval') {
      state.evals.push({ step: m.step, valLoss: m.valLoss });
      state.samples.push(m.sample);
    } else if (m.type === 'state') {
      state.running = m.running;
    } else if (m.type === 'saved') {
      state.savedStep = m.step;
    } else if (m.type === 'error') {
      state.error = m.message;
    }
    emit(m.type);
  });

  function applyTokenizer(info) {
    state.tokInfo = info;
    state.tokenizer = makeTokenizer({ tokens: info.tokens, merges: info.merges });
  }

  function applySetup(s) {
    Object.assign(state, {
      setup: s, preset: s.preset, step: s.step, total: s.total, dataset: s.dataset, ft: s.ft ?? null,
      losses: s.history.slice(), evals: s.evals.slice(), samples: s.samples.slice(),
    });
  }

  // Ein gespeichertes Training wiederherstellen (einmal pro Seitenaufruf).
  const ready = (async () => {
    state.gpu = await detectGpu();
    emit('gpu');
    try {
      const { state: s } = await trainer.restore();
      if (s) {
        applyTokenizer(s);
        applySetup(s);
        state.vocab = s.tokenizer.vocab;
        state.savedStep = s.step;
        emit('restored');
      }
    } catch (e) {
      state.error = `Gespeichertes Training konnte nicht geladen werden: ${e.message}`;
      emit('error');
    }
  })();

  store = {
    state, trainer, ready,
    on: (fn) => { listeners.add(fn); return () => listeners.delete(fn); },
    emit,
    async tokenize(dataset, vocab, text) {
      state.running = false;
      Object.assign(state, { dataset, vocab, setup: null, step: 0, losses: [], evals: [], samples: [], error: '' });
      emit('tokenizing');
      const info = await trainer.tokenize(dataset, vocab, text);
      applyTokenizer(info);
      // Zum Datensatz passende Größe vorschlagen; ohne GPU nur die kleinste.
      state.preset = state.gpu === null ? 'winzig' : DATASETS[dataset].preset;
      emit('tokenized');
      return info;
    },
    async setupModel(preset, total, prefer) {
      state.error = '';
      const s = await trainer.setup(preset, total, prefer);
      applySetup(s);
      emit('setup');
      return s;
    },
    train(total) { state.error = ''; trainer.train(total); },
    async finetune(dataset, steps, text) {
      state.error = '';
      const s = await trainer.finetune(dataset, steps, text);
      applySetup(s);
      emit('finetune');
      return s;
    },
    async revert() {
      const { state: s } = await trainer.revert();
      applyTokenizer(s);
      applySetup(s);
      emit('restored');
    },
    pause() { trainer.pause(); },
  };
  return store;
}
