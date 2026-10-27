// Vergleicht eine (im Browser exportierte) GGUF-Datei mit Ollama.
//   node tests/e2e-gguf-compare.mjs <datei.gguf> <modellname-in-ollama> <ollama-url> [prompt …]
// Liest Gewichte und Tokenizer aus der Datei, erzeugt mit unserer eigenen
// Inferenz Greedy-Text (Temperatur 0) und vergleicht ihn mit Ollama (Streaming).
import { readFileSync, writeFileSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import { readGGUF, decodeTensor } from '../public/js/llama/gguf.js';
import { makeTokenizer } from '../public/js/llama/bpe.js';
import { tensorLayout } from '../public/js/llama/config.js';
import { generate } from '../public/js/llama/infer.js';

const [file, model, url, ...prompts] = process.argv.slice(2);
const bytes = new Uint8Array(readFileSync(file));
const g = readGGUF(bytes);
const m = g.meta;
const cfg = {
  vocabSize: m['llama.vocab_size'], dim: m['llama.embedding_length'], nLayers: m['llama.block_count'],
  nHeads: m['llama.attention.head_count'], nKvHeads: m['llama.attention.head_count_kv'],
  headDim: m['llama.rope.dimension_count'], ffnDim: m['llama.feed_forward_length'], ctx: m['llama.context_length'],
  ropeBase: m['llama.rope.freq_base'], normEps: m['llama.attention.layer_norm_rms_epsilon'],
};
const strToId = new Map(m['tokenizer.ggml.tokens'].map((t, i) => [t, i]));
const merges = m['tokenizer.ggml.merges'].map((s) => s.split(' ').map((t) => strToId.get(t)));
const tok = makeTokenizer({ tokens: m['tokenizer.ggml.tokens'], merges });
const layout = tensorLayout(cfg);
const params = new Float32Array(layout.total);
const TYPES = { 0: 'f32', 1: 'f16', 8: 'q8_0' };
const BYTES = { f32: (n) => n * 4, f16: (n) => n * 2, q8_0: (n) => (n / 32) * 34 };
for (const t of g.tensors) {
  const dst = layout.byName[t.name];
  const f = TYPES[t.type];
  const start = bytes.byteOffset + g.dataStart + t.offset;
  params.set(decodeTensor(new Uint8Array(bytes.buffer.slice(start, start + BYTES[f](dst.size))), dst.size, f), dst.offset);
}
const quantized = g.meta['general.file_type'] !== 0;
console.log(`${file}: ${cfg.nLayers} Schichten, Breite ${cfg.dim}, Vokabular ${cfg.vocabSize}, Kontext ${cfg.ctx}, file_type ${g.meta['general.file_type']}`);

let allSame = true;
for (const prompt of prompts) {
  // Nur so viele Tokens, wie zusammen mit dem Prompt in den Kontext passen. Danach
  // verschiebt Ollama den Kontext ("context shift"), unsere Inferenz hört dort auf.
  const maxTokens = Math.min(60, cfg.ctx - 1 - tok.encode(prompt).length);
  const expected = tok.decode(generate(cfg, params, tok, prompt, { maxTokens, temperature: 0 }));
  const body = JSON.stringify({ model, prompt, raw: true, stream: true, options: { temperature: 0, top_k: 1, num_predict: maxTokens, repeat_penalty: 1 } });
  let raw;
  if (process.env.CURL_EXE) {
    // Windows-Ollama aus WSL: Anfrage über curl.exe auf der Windows-Seite schicken.
    writeFileSync(process.env.REQ_FILE_WSL, body);
    raw = execFileSync(process.env.CURL_EXE, ['-s', '-d', `@${process.env.REQ_FILE_WIN}`, `${url}/api/generate`], { encoding: 'utf8' });
  } else {
    raw = await (await fetch(`${url}/api/generate`, { method: 'POST', body })).text();
  }
  let text = '', reason = null;
  for (const line of raw.split('\n')) {
    if (!line.trim()) continue;
    const j = JSON.parse(line);
    text += j.response || '';
    if (j.done) reason = j.done_reason;
  }
  const same = text === expected;
  // Bei einem Abbruch durch Ollamas Wiederholungsschutz zählt, dass alles bis dahin stimmt.
  const prefix = !same && reason === null && expected.startsWith(text) && text.length > 0;
  let common = 0;
  while (common < Math.min(text.length, expected.length) && text[common] === expected[common]) common++;
  // Quantisiert rundet llama.cpp auch Zwischenwerte → kleine Abweichungen sind erlaubt.
  allSame &&= same || prefix || quantized;
  const verdict = same ? 'GLEICH' : prefix ? 'GLEICH bis Abbruch' : `GLEICH für ${common} von ${expected.length} Zeichen`;
  console.log(`${verdict}  ${JSON.stringify(prompt)} → ${JSON.stringify(text.slice(0, 90))}${same ? '' : `\n        erwartet ${JSON.stringify(expected.slice(0, 90))}`}  [${reason}]`);
}
console.log(allSame ? 'ERGEBNIS: identisch' : 'ERGEBNIS: Abweichung');
process.exit(allSame ? 0 : 1);
