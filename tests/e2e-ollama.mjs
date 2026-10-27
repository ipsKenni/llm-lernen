// End-to-End-Test gegen echtes Ollama (nicht Teil von `npm test`).
//
//   node tests/e2e-ollama.mjs <ausgabe-ordner>
//
// Trainiert zwei kleine Modelle auf der CPU, exportiert GGUF + Modelfile und
// schreibt expected.json mit den Greedy-Ausgaben (Temperatur 0) unserer eigenen
// Inferenz. tests/e2e-ollama.sh vergleicht diese dann mit Ollama.
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { join } from 'node:path';
import { createCpuBackend } from '../public/js/llama/backend-cpu.js';
import { createLlama } from '../public/js/llama/model.js';
import { trainBPE } from '../public/js/llama/bpe.js';
import { writeGGUF, makeModelfile } from '../public/js/llama/gguf.js';
import { generate } from '../public/js/llama/infer.js';
import { makeRng } from '../public/js/rng.js';

const out = process.argv[2];
mkdirSync(out, { recursive: true });
const data = (f) => readFileSync(new URL(`../public/data/${f}`, import.meta.url), 'utf8');

async function trainAndExport({ name, docs, vocab, cfg, batch, steps, lr, prompts }) {
  const tok = trainBPE(docs, vocab);
  const stream = [];
  for (const d of docs) stream.push(tok.eot, ...tok.encode(d));
  const fullCfg = { vocabSize: tok.vocabSize, ropeBase: 10000, normEps: 1e-5, headDim: cfg.dim / cfg.nHeads, ...cfg };
  const model = createLlama(fullCfg, createCpuBackend(), { batch });
  model.setParams(model.initialParams(1));
  const rng = makeRng(7);
  const T = fullCfg.ctx;
  let loss = 0;
  const t0 = performance.now();
  for (let s = 1; s <= steps; s++) {
    const inp = new Int32Array(batch * T), tgt = new Int32Array(batch * T);
    for (let b = 0; b < batch; b++) {
      const start = rng.int(stream.length - T - 1);
      for (let t = 0; t < T; t++) { inp[b * T + t] = stream[start + t]; tgt[b * T + t] = stream[start + t + 1]; }
    }
    model.setBatch(inp, tgt);
    const warm = Math.min(1, s / 30);
    loss = await model.trainStep({ lr: lr * warm * (0.1 + 0.9 * 0.5 * (1 + Math.cos(Math.PI * s / steps))), wd: 0.1, t: s });
  }
  const params = await model.readParams();
  writeFileSync(join(out, `${name}.gguf`), writeGGUF(fullCfg, params, tok, name));
  writeFileSync(join(out, `${name}.Modelfile`), makeModelfile(`${name}.gguf`, fullCfg));
  const expected = prompts.map((prompt) => {
    const ids = generate(fullCfg, params, tok, prompt, { maxTokens: 40, temperature: 0 });
    return { prompt, promptIds: [tok.eot, ...tok.encode(prompt)], ids, text: tok.decode(ids) };
  });
  console.log(`${name}: ${steps} Schritte in ${((performance.now() - t0) / 1000).toFixed(0)} s, Loss ${loss.toFixed(3)}`);
  for (const e of expected) console.log(`  „${e.prompt}“ → „${e.text.replaceAll('\n', '⏎')}“`);
  return { name, numPredict: 40, expected };
}

const names = data('names.txt').split('\n').filter(Boolean);
const maerchen = data('maerchen.txt').split('\n<|endoftext|>\n');
const results = [
  await trainAndExport({
    name: 'namen', docs: names, vocab: 320, batch: 8, steps: 300, lr: 3e-3,
    cfg: { dim: 64, nLayers: 2, nHeads: 1, nKvHeads: 1, ffnDim: 128, ctx: 32 },
    prompts: ['', 'ma', 'jo', 'Kr'],
  }),
  await trainAndExport({
    name: 'maerchen', docs: maerchen, vocab: 512, batch: 4, steps: 200, lr: 2e-3,
    cfg: { dim: 128, nLayers: 2, nHeads: 2, nKvHeads: 1, ffnDim: 384, ctx: 64 },
    prompts: ['Es war einmal', 'Der König sprach: »', 'Hänsel und Gretel'],
  }),
];
writeFileSync(join(out, 'expected.json'), JSON.stringify(results, null, 1));
