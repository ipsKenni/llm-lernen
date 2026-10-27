// =====================================================================
// config.js – Größe und Bauplan des Llama-Modells
// =====================================================================
// Alle Tensor-Namen entsprechen exakt dem GGUF-Format von llama.cpp.
// So kann das trainierte Modell 1:1 nach Ollama exportiert werden.

export const PRESETS = {
  winzig: { label: 'Winzig', dim: 64, nLayers: 2, nHeads: 1, nKvHeads: 1, ffnDim: 192, ctx: 64, batch: 8, vocab: 512, lr: 3e-3 },
  mini: { label: 'Mini', dim: 128, nLayers: 4, nHeads: 2, nKvHeads: 1, ffnDim: 384, ctx: 128, batch: 8, vocab: 1024, lr: 2e-3 },
  klein: { label: 'Klein', dim: 256, nLayers: 6, nHeads: 4, nKvHeads: 2, ffnDim: 704, ctx: 128, batch: 12, vocab: 2048, lr: 1.5e-3 },
  mittel: { label: 'Mittel', dim: 384, nLayers: 8, nHeads: 6, nKvHeads: 2, ffnDim: 1024, ctx: 256, batch: 12, vocab: 4096, lr: 1e-3 },
};

export function makeConfig(preset, vocabSize) {
  const p = typeof preset === 'string' ? PRESETS[preset] : preset;
  return {
    vocabSize,
    dim: p.dim,
    nLayers: p.nLayers,
    nHeads: p.nHeads,
    nKvHeads: p.nKvHeads,
    headDim: p.dim / p.nHeads,
    ffnDim: p.ffnDim,
    ctx: p.ctx,
    ropeBase: 10000,
    normEps: 1e-5,
  };
}

// Jeder Tensor beginnt an einem Vielfachen von 64 Zahlen (256 Byte) –
// das verlangt WebGPU für Teilbereiche eines Buffers.
const ALIGN = 64;
const align = (n) => Math.ceil(n / ALIGN) * ALIGN;

//#region llama-layout
// Alle Parameter in einer festen Reihenfolge. shape = [Zeilen, Spalten],
// Matrizen sind [Ausgabe, Eingabe] – wie in PyTorch und GGUF.
export function tensorLayout(cfg) {
  const { vocabSize: V, dim: d, nHeads: H, nKvHeads: KV, headDim: hd, ffnDim: F } = cfg;
  const list = [];
  let offset = 0;
  const add = (name, rows, cols, kind) => {
    list.push({ name, shape: cols ? [rows, cols] : [rows], size: rows * (cols || 1), offset, kind });
    offset = align(offset + rows * (cols || 1));
  };
  add('token_embd.weight', V, d, 'embed');          // gleichzeitig Ausgabe-Schicht (Weight Tying)
  for (let l = 0; l < cfg.nLayers; l++) {
    add(`blk.${l}.attn_norm.weight`, d, 0, 'norm');
    add(`blk.${l}.attn_q.weight`, H * hd, d, 'matrix');
    add(`blk.${l}.attn_k.weight`, KV * hd, d, 'matrix');
    add(`blk.${l}.attn_v.weight`, KV * hd, d, 'matrix');
    add(`blk.${l}.attn_output.weight`, d, H * hd, 'residual');
    add(`blk.${l}.ffn_norm.weight`, d, 0, 'norm');
    add(`blk.${l}.ffn_gate.weight`, F, d, 'matrix');
    add(`blk.${l}.ffn_up.weight`, F, d, 'matrix');
    add(`blk.${l}.ffn_down.weight`, d, F, 'residual');
  }
  add('output_norm.weight', d, 0, 'norm');
  return { tensors: list, total: offset, byName: Object.fromEntries(list.map((t) => [t.name, t])) };
}
//#endregion

export function countParams(cfg) {
  return tensorLayout(cfg).tensors.reduce((s, t) => s + t.size, 0);
}

// Grobe Rechenarbeit pro Trainings-Token: 6 · Parameter (vorwärts 2, rückwärts 4)
// plus Attention (wächst mit der Kontextlänge).
export function trainFlopsPerToken(cfg) {
  const attn = cfg.nLayers * 2 * cfg.ctx * cfg.nHeads * cfg.headDim;
  return 6 * countParams(cfg) + 3 * attn;
}

// Speicher fürs Training (Parameter, Gradienten, Adam-Momente, Aktivierungen) in Byte.
export function trainMemoryBytes(cfg, batch) {
  const N = batch * cfg.ctx;
  const { dim: d, nHeads: H, nKvHeads: KV, headDim: hd, ffnDim: F, vocabSize: V } = cfg;
  const perLayer = N * (4 * d + 2 * H * hd + 2 * KV * hd + 3 * F + 2) + batch * H * cfg.ctx * cfg.ctx;
  const acts = cfg.nLayers * perLayer + N * (2 * d + V) + 2 * N * Math.max(d, F, V);
  return 4 * (4 * countParams(cfg) + acts + batch * H * cfg.ctx * cfg.ctx);
}
