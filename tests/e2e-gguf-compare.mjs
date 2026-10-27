// Vergleicht eine (im Browser exportierte) GGUF-Datei mit Ollama.
//   node tests/e2e-gguf-compare.mjs <datei.gguf> <modellname-in-ollama> <ollama-url> [prompt …]
// Liest Gewichte und Tokenizer aus der Datei, erzeugt mit unserer eigenen
// Inferenz Greedy-Text (Temperatur 0) und vergleicht ihn mit Ollama (Streaming).
import { readFileSync } from 'node:fs';
import { readGGUF } from '../public/js/llama/gguf.js';
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
for (const t of g.tensors) {
  const dst = layout.byName[t.name];
  const start = bytes.byteOffset + g.dataStart + t.offset;
  params.set(new Float32Array(bytes.buffer.slice(start, start + dst.size * 4)), dst.offset);
}
console.log(`${file}: ${cfg.nLayers} Schichten, Breite ${cfg.dim}, Vokabular ${cfg.vocabSize}, Kontext ${cfg.ctx}`);

let allSame = true;
for (const prompt of prompts) {
  // Nur so viele Tokens, wie zusammen mit dem Prompt in den Kontext passen. Danach
  // verschiebt Ollama den Kontext ("context shift"), unsere Inferenz hört dort auf.
  const maxTokens = Math.min(60, cfg.ctx - 1 - tok.encode(prompt).length);
  const expected = tok.decode(generate(cfg, params, tok, prompt, { maxTokens, temperature: 0 }));
  const res = await fetch(`${url}/api/generate`, {
    method: 'POST',
    body: JSON.stringify({ model, prompt, raw: true, stream: true, options: { temperature: 0, top_k: 1, num_predict: maxTokens, repeat_penalty: 1 } }),
  });
  let text = '', reason = null;
  for (const line of (await res.text()).split('\n')) {
    if (!line.trim()) continue;
    const j = JSON.parse(line);
    text += j.response || '';
    if (j.done) reason = j.done_reason;
  }
  const same = text === expected;
  // Bei einem Abbruch durch Ollamas Wiederholungsschutz zählt, dass alles bis dahin stimmt.
  const prefix = !same && reason === null && expected.startsWith(text) && text.length > 0;
  allSame &&= same || prefix;
  console.log(`${same ? 'GLEICH' : prefix ? 'GLEICH bis Abbruch' : 'ANDERS'}  ${JSON.stringify(prompt)} → ${JSON.stringify(text.slice(0, 90))}${same ? '' : `\n        erwartet ${JSON.stringify(expected.slice(0, 90))}`}  [${reason}]`);
}
console.log(allSame ? 'ERGEBNIS: identisch' : 'ERGEBNIS: Abweichung');
process.exit(allSame ? 0 : 1);
