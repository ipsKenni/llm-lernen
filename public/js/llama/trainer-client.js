// =====================================================================
// trainer-client.js – Die Seite spricht mit dem Teil-2-Worker
// =====================================================================

export function createTrainerClient() {
  const worker = new Worker(new URL('./trainer-worker.js', import.meta.url), { type: 'module' });
  const listeners = new Set();
  const pending = new Map();
  const tokenListeners = new Map();
  let nextId = 1;

  worker.onerror = (e) => {
    e.preventDefault?.();
    window.showBootError?.(`Der Trainings-Worker (Teil 2) ist abgestürzt: ${e.message || 'unbekannter Fehler'}`);
  };
  worker.onmessage = (e) => {
    const m = e.data;
    if (m.type === 'result') {
      const p = pending.get(m.id);
      pending.delete(m.id);
      tokenListeners.delete(m.id);
      if (!p) return;
      if (m.error) p.reject(new Error(m.error)); else p.resolve(m);
      return;
    }
    if (m.type === 'token') { tokenListeners.get(m.id)?.(m.text); return; }
    listeners.forEach((fn) => fn(m));
  };

  function request(msg, onToken) {
    const id = nextId++;
    if (onToken) tokenListeners.set(id, onToken);
    return new Promise((resolve, reject) => {
      pending.set(id, { resolve, reject });
      worker.postMessage({ ...msg, id });
    });
  }

  return {
    tokenize: (dataset, vocabSize, text = '') => request({ type: 'tokenize', dataset, vocabSize, text }),
    setup: (preset, total, prefer = 'auto') => request({ type: 'setup', preset, total, prefer }),
    restore: () => request({ type: 'restore' }),
    describe: () => request({ type: 'describe' }),
    train: (total) => worker.postMessage({ type: 'train', total }),
    pause: () => worker.postMessage({ type: 'pause' }),
    generate: (opts, onToken) => request({ type: 'generate', ...opts }, onToken),
    exportModel: (name, format = 'f32') => request({ type: 'export', name, format }),
    reset: () => request({ type: 'reset' }),
    benchmark: (presets, prefer, vocab) => request({ type: 'benchmark', presets, prefer, vocab }),
    inspect: (prompt) => request({ type: 'inspect', prompt }),
    snapshot: () => request({ type: 'snapshot' }),
    finetune: (dataset, steps, text = '') => request({ type: 'finetune', dataset, steps, text }),
    revert: () => request({ type: 'revert' }),
    embeddings: (limit = 500) => request({ type: 'embeddings', limit }),
    on: (fn) => { listeners.add(fn); return () => listeners.delete(fn); },
  };
}

// WebGPU schon im Hauptthread erkennen, um die Auswahl anzupassen.
export async function detectGpu() {
  try {
    if (!navigator.gpu) return null;
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) return null;
    const info = adapter.info || {};
    return { vendor: info.vendor || '', architecture: info.architecture || '', description: info.description || '', fallback: !!adapter.isFallbackAdapter };
  } catch {
    return null;
  }
}
