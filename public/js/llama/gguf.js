// =====================================================================
// gguf.js – Das Modell als GGUF-Datei speichern (das Format von Ollama)
// =====================================================================
//
// GGUF ist ein einfaches Binärformat:
//   "GGUF" · Version · Anzahl Tensoren · Anzahl Metadaten
//   Metadaten:   Schlüssel → Wert   (Architektur, Größen, Tokenizer …)
//   Tensor-Infos: Name, Form, Datentyp, Position in der Datei
//   Tensor-Daten: die nackten Zahlen, jeweils auf 32 Byte ausgerichtet
//
// llama.cpp (und damit Ollama) liest diese Metadaten und baut daraus
// genau das Modell, das wir trainiert haben.

import { tensorLayout } from './config.js';
import { EOT } from './bpe.js';

const T = { UINT32: 4, INT32: 5, FLOAT32: 6, BOOL: 7, STRING: 8, ARRAY: 9, UINT64: 10 };
const ALIGNMENT = 32;
const TOKEN_NORMAL = 1, TOKEN_CONTROL = 3;

//#region gguf-quant
// Speicherformate. Q8_0: je 32 Zahlen ein Block aus einem gemeinsamen Faktor d
// (als f16) und 32 ganzen Zahlen von −127 bis 127: x ≈ d · q. Das braucht
// 34 statt 128 Byte pro Block – gut ein Viertel – bei kaum messbarem Verlust.
export const FORMATS = {
  f32: { label: 'F32', ggmlType: 0, fileType: 0, bytes: (n) => n * 4, note: 'volle Genauigkeit, 4 Byte pro Zahl' },
  f16: { label: 'F16', ggmlType: 1, fileType: 1, bytes: (n) => n * 2, note: 'halbe Genauigkeit, 2 Byte pro Zahl' },
  q8_0: { label: 'Q8_0', ggmlType: 8, fileType: 7, bytes: (n) => (n / 32) * 34, note: '8 Bit pro Zahl plus ein Faktor je 32 Zahlen' },
};

// float32 → float16 (IEEE 754, Runden zur nächsten geraden Zahl wie die Hardware)
const f32buf = new Float32Array(1), u32buf = new Uint32Array(f32buf.buffer);
export function toHalf(x) {
  f32buf[0] = x;
  const b = u32buf[0];
  const sign = (b >>> 16) & 0x8000;
  let exp = (b >>> 23) & 0xff;
  let mant = b & 0x7fffff;
  if (exp === 0xff) return sign | 0x7c00 | (mant ? 0x200 : 0);          // Inf/NaN
  exp = exp - 127 + 15;
  if (exp >= 0x1f) return sign | 0x7c00;                                 // zu groß → Inf
  if (exp <= 0) {                                                        // subnormal
    if (exp < -10) return sign;
    mant |= 0x800000;
    const shift = 14 - exp;
    let h = mant >>> shift;
    const rest = mant & ((1 << shift) - 1), half = 1 << (shift - 1);
    if (rest > half || (rest === half && (h & 1))) h += 1;
    return sign | h;
  }
  let h = sign | (exp << 10) | (mant >>> 13);
  const rest = mant & 0x1fff;
  if (rest > 0x1000 || (rest === 0x1000 && (h & 1))) h += 1;             // Übertrag in den Exponenten ist korrekt
  return h;
}
export function fromHalf(h) {
  const sign = h & 0x8000 ? -1 : 1, exp = (h >>> 10) & 0x1f, mant = h & 0x3ff;
  if (exp === 0) return sign * mant * 2 ** -24;
  if (exp === 0x1f) return mant ? NaN : sign * Infinity;
  return sign * (1 + mant / 1024) * 2 ** (exp - 15);
}

// Wie roundf() in C: x,5 wird von der Null weg gerundet (Math.round rundet −2,5 zu −2).
const roundAway = (x) => Math.sign(x) * Math.round(Math.abs(x));

// Zahlen → Bytes im gewählten Format (genau wie llama.cpps quantize_row_q8_0_ref)
export function encodeTensor(data, format) {
  if (format === 'f32') return new Uint8Array(data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength));
  if (format === 'f16') {
    const out = new Uint16Array(data.length);
    for (let i = 0; i < data.length; i++) out[i] = toHalf(data[i]);
    return new Uint8Array(out.buffer);
  }
  const blocks = data.length / 32;
  const out = new Uint8Array(blocks * 34);
  const view = new DataView(out.buffer);
  for (let b = 0; b < blocks; b++) {
    let amax = 0;
    for (let j = 0; j < 32; j++) amax = Math.max(amax, Math.abs(data[b * 32 + j]));
    const d = amax / 127, id = d ? 1 / d : 0;
    view.setUint16(b * 34, toHalf(d), true);
    for (let j = 0; j < 32; j++) view.setInt8(b * 34 + 2 + j, roundAway(Math.fround(data[b * 32 + j] * id)));
  }
  return out;
}

// Bytes → Zahlen (so liest llama.cpp die Gewichte)
export function decodeTensor(bytes, n, format) {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const out = new Float32Array(n);
  if (format === 'f32') for (let i = 0; i < n; i++) out[i] = view.getFloat32(i * 4, true);
  else if (format === 'f16') for (let i = 0; i < n; i++) out[i] = fromHalf(view.getUint16(i * 2, true));
  else {
    for (let b = 0; b < n / 32; b++) {
      const d = fromHalf(view.getUint16(b * 34, true));
      for (let j = 0; j < 32; j++) out[b * 32 + j] = d * view.getInt8(b * 34 + 2 + j);
    }
  }
  return out;
}
//#endregion

// Normgewichte (1-D) bleiben immer F32 – so hält es auch llama.cpp.
const tensorFormat = (t, format) => (t.shape.length === 1 ? 'f32' : format);

// Alle Gewichte so, wie sie nach dem Speichern im gewählten Format aussehen.
export function roundTripParams(cfg, params, format) {
  const out = params.slice();
  for (const t of tensorLayout(cfg).tensors) {
    const f = tensorFormat(t, format);
    if (f === 'f32') continue;
    const data = params.subarray(t.offset, t.offset + t.size);
    out.set(decodeTensor(encodeTensor(data, f), t.size, f), t.offset);
  }
  return out;
}

//#region gguf-metadata
export function ggufMetadata(cfg, tok, name, format = 'f32') {
  return [
    ['general.architecture', T.STRING, 'llama'],
    ['general.name', T.STRING, name],
    ['general.file_type', T.UINT32, FORMATS[format].fileType], // 0 = F32, 1 = F16, 7 = Q8_0
    ['llama.context_length', T.UINT32, cfg.ctx],
    ['llama.embedding_length', T.UINT32, cfg.dim],
    ['llama.block_count', T.UINT32, cfg.nLayers],
    ['llama.feed_forward_length', T.UINT32, cfg.ffnDim],
    ['llama.attention.head_count', T.UINT32, cfg.nHeads],
    ['llama.attention.head_count_kv', T.UINT32, cfg.nKvHeads],
    ['llama.attention.layer_norm_rms_epsilon', T.FLOAT32, cfg.normEps],
    ['llama.rope.dimension_count', T.UINT32, cfg.headDim],
    ['llama.rope.freq_base', T.FLOAT32, cfg.ropeBase],
    ['llama.vocab_size', T.UINT32, cfg.vocabSize],
    ['tokenizer.ggml.model', T.STRING, 'gpt2'],              // Byte-Level-BPE wie GPT-2
    ['tokenizer.ggml.pre', T.STRING, 'gpt-2'],               // dieselbe Wort-Vorzerlegung wie bpe.js
    ['tokenizer.ggml.tokens', T.ARRAY, [T.STRING, tok.tokens]],
    ['tokenizer.ggml.token_type', T.ARRAY, [T.INT32, tok.tokens.map((t) => (t === EOT ? TOKEN_CONTROL : TOKEN_NORMAL))]],
    ['tokenizer.ggml.merges', T.ARRAY, [T.STRING, tok.merges.map(([a, b]) => `${tok.tokens[a]} ${tok.tokens[b]}`)]],
    ['tokenizer.ggml.bos_token_id', T.UINT32, tok.eot],
    ['tokenizer.ggml.eos_token_id', T.UINT32, tok.eot],
    ['tokenizer.ggml.padding_token_id', T.UINT32, tok.eot],
    ['tokenizer.ggml.add_bos_token', T.BOOL, true],
    ['tokenizer.ggml.add_eos_token', T.BOOL, false],
  ];
}
//#endregion

// Kleiner Byte-Schreiber, der automatisch wächst.
class Writer {
  constructor() { this.buf = new Uint8Array(1 << 16); this.view = new DataView(this.buf.buffer); this.pos = 0; }
  ensure(n) {
    if (this.pos + n <= this.buf.length) return;
    let size = this.buf.length;
    while (size < this.pos + n) size *= 2;
    const next = new Uint8Array(size);
    next.set(this.buf);
    this.buf = next;
    this.view = new DataView(next.buffer);
  }
  u32(v) { this.ensure(4); this.view.setUint32(this.pos, v, true); this.pos += 4; }
  i32(v) { this.ensure(4); this.view.setInt32(this.pos, v, true); this.pos += 4; }
  u64(v) { this.ensure(8); this.view.setBigUint64(this.pos, BigInt(v), true); this.pos += 8; }
  f32(v) { this.ensure(4); this.view.setFloat32(this.pos, v, true); this.pos += 4; }
  u8(v) { this.ensure(1); this.buf[this.pos++] = v; }
  bytes(b) { this.ensure(b.length); this.buf.set(b, this.pos); this.pos += b.length; }
  str(s) { const b = new TextEncoder().encode(s); this.u64(b.length); this.bytes(b); }
  pad(align) { while (this.pos % align) this.u8(0); }
  value(type, v) {
    switch (type) {
      case T.UINT32: return this.u32(v);
      case T.INT32: return this.i32(v);
      case T.FLOAT32: return this.f32(v);
      case T.BOOL: return this.u8(v ? 1 : 0);
      case T.STRING: return this.str(v);
      case T.UINT64: return this.u64(v);
      case T.ARRAY: {
        const [elemType, items] = v;
        this.u32(elemType);
        this.u64(items.length);
        for (const item of items) this.value(elemType, item);
        return undefined;
      }
      default: throw new Error(`GGUF-Typ ${type} nicht unterstützt`);
    }
  }
}

//#region gguf-write
// params: Float32Array mit allen Gewichten (Layout aus config.js).
export function writeGGUF(cfg, params, tok, name = 'mein-modell', format = 'f32') {
  const { tensors } = tensorLayout(cfg);
  const meta = ggufMetadata(cfg, tok, name, format);
  const w = new Writer();
  w.bytes(new TextEncoder().encode('GGUF'));
  w.u32(3);                        // GGUF-Version 3
  w.u64(tensors.length);
  w.u64(meta.length);
  for (const [key, type, value] of meta) { w.str(key); w.u32(type); w.value(type, value); }
  // Tensor-Infos. GGUF zählt Dimensionen "von innen nach außen": eine
  // Matrix [Zeilen, Spalten] wird als [Spalten, Zeilen] eingetragen.
  let offset = 0;
  const placed = tensors.map((t) => {
    const f = tensorFormat(t, format);
    const bytes = encodeTensor(params.subarray(t.offset, t.offset + t.size), f);
    const at = offset;
    offset += Math.ceil(bytes.length / ALIGNMENT) * ALIGNMENT;
    return { ...t, at, bytes, ggmlType: FORMATS[f].ggmlType };
  });
  for (const t of placed) {
    w.str(t.name);
    w.u32(t.shape.length);
    for (const dim of [...t.shape].reverse()) w.u64(dim);
    w.u32(t.ggmlType);
    w.u64(t.at);
  }
  w.pad(ALIGNMENT);
  for (const t of placed) {
    w.bytes(t.bytes);
    w.pad(ALIGNMENT);
  }
  return w.buf.subarray(0, w.pos);
}
//#endregion

// Zum Prüfen: eine GGUF-Datei wieder einlesen (Metadaten + Tensor-Infos).
export function readGGUF(bytes) {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  let pos = 0;
  const u32 = () => { const v = view.getUint32(pos, true); pos += 4; return v; };
  const u64 = () => { const v = Number(view.getBigUint64(pos, true)); pos += 8; return v; };
  const str = () => { const n = u64(); const s = new TextDecoder().decode(bytes.subarray(pos, pos + n)); pos += n; return s; };
  const value = (type) => {
    switch (type) {
      case T.UINT32: return u32();
      case T.INT32: { const v = view.getInt32(pos, true); pos += 4; return v; }
      case T.FLOAT32: { const v = view.getFloat32(pos, true); pos += 4; return v; }
      case T.BOOL: return bytes[pos++] === 1;
      case T.STRING: return str();
      case T.UINT64: return u64();
      case T.ARRAY: { const et = u32(); const n = u64(); return Array.from({ length: n }, () => value(et)); }
      default: throw new Error(`GGUF-Typ ${type} unbekannt`);
    }
  };
  const magic = new TextDecoder().decode(bytes.subarray(0, 4));
  pos = 4;
  const version = u32(), nTensors = u64(), nMeta = u64();
  const meta = {};
  for (let i = 0; i < nMeta; i++) { const k = str(); meta[k] = value(u32()); }
  const tensors = [];
  for (let i = 0; i < nTensors; i++) {
    const name = str(); const nd = u32();
    const dims = Array.from({ length: nd }, () => u64());
    tensors.push({ name, dims, type: u32(), offset: u64() });
  }
  const dataStart = Math.ceil(pos / ALIGNMENT) * ALIGNMENT;
  return { magic, version, meta, tensors, dataStart };
}

//#region gguf-modelfile
// Das Modelfile sagt Ollama, welche Datei es laden soll und wie es
// generieren soll. TEMPLATE "{{ .Prompt }}" heißt: kein Chat-Format,
// der eingegebene Text wird einfach fortgesetzt.
export function makeModelfile(fileName, cfg) {
  return [
    `FROM ./${fileName}`,
    'TEMPLATE """{{ .Prompt }}"""',
    'PARAMETER temperature 0.8',
    'PARAMETER top_k 40',
    'PARAMETER top_p 0.95',
    `PARAMETER num_ctx ${cfg.ctx}`,
    'PARAMETER num_predict 300',
    '',
  ].join('\n');
}
//#endregion
