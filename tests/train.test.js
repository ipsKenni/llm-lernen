// Smoke-Tests: Lernen die Modelle wirklich etwas?
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { parseWords, buildTokenizer } from '../public/js/tokenizer.js';
import { makeRng } from '../public/js/rng.js';
import { countBigrams, countsToProbs, averageNll, sampleWord } from '../public/js/bigram-count.js';
import { bigramPairs, createBigramNet, trainStep as bigramStep } from '../public/js/bigram-neural.js';
import { DEFAULT_CONFIG, createGPT, createAdam, trainStep, generate, inspect, countParams } from '../public/js/transformer.js';

const words = parseWords(readFileSync(new URL('../public/data/names.txt', import.meta.url), 'utf8'));
const tok = buildTokenizer(words);

test('Tokenizer: 27 Tokens, Roundtrip', () => {
  assert.equal(tok.vocabSize, 27);
  assert.equal(tok.decode(tok.encode('emma')), 'emma');
});

test('Zähl-Bigram: Loss ≈ 2,45', () => {
  const P = countsToProbs(countBigrams(words, tok));
  const nll = averageNll(words, tok, P);
  console.log('  Zähl-Bigram NLL:', nll.toFixed(4));
  assert.ok(nll > 2.4 && nll < 2.5);
  assert.ok(sampleWord(P, tok, makeRng(1)).word.length >= 0);
});

test('Neuronales Bigram lernt Richtung Zähl-Loss', () => {
  const rng = makeRng(42);
  const net = createBigramNet(tok.vocabSize, rng);
  const pairs = bigramPairs(words, tok);
  const t0 = performance.now();
  let first, last = [];
  for (let i = 0; i < 300; i++) {
    const l = bigramStep(net, pairs, rng);
    if (i === 0) first = l;
    if (i >= 280) last.push(l);
  }
  const avg = last.reduce((a, b) => a + b) / last.length;
  console.log(`  Neural-Bigram: ${first.toFixed(3)} → ${avg.toFixed(3)} (${((performance.now() - t0) / 300).toFixed(1)} ms/Schritt)`);
  assert.ok(first > 3.2);
  assert.ok(avg < 2.65);
});

test('Mini-GPT: Loss fällt deutlich', () => {
  const cfg = { ...DEFAULT_CONFIG };
  const rng = makeRng(cfg.seed);
  const model = createGPT(cfg, tok.vocabSize, rng);
  assert.equal(model.params.length, countParams(cfg, tok.vocabSize));
  console.log('  Parameter:', model.params.length);
  const opt = createAdam(model.params.length);
  const steps = Number(process.env.GPT_STEPS ?? 300);
  const losses = [];
  const t0 = performance.now();
  for (let s = 0; s < steps; s++) {
    const batch = Array.from({ length: cfg.batchSize }, () => tok.encode(words[rng.int(words.length)]));
    const lr = cfg.lr * (1 - s / steps);
    losses.push(trainStep(model, opt, batch, lr));
  }
  const ms = (performance.now() - t0) / steps;
  const avg = (a) => a.reduce((x, y) => x + y) / a.length;
  const start = avg(losses.slice(0, 10)), end = avg(losses.slice(-50));
  console.log(`  GPT: ${start.toFixed(3)} → ${end.toFixed(3)} in ${steps} Schritten (${ms.toFixed(1)} ms/Schritt)`);
  const samples = Array.from({ length: 8 }, () => tok.decode(generate(model, rng, { temperature: 0.6 })));
  console.log('  Beispiele:', samples.join(', '));
  const insp = inspect(model, tok.encode('emm'));
  assert.equal(insp.attn[0][0].length, 4);
  assert.ok(Math.abs(insp.probs.reduce((a, b) => a + b) - 1) < 1e-9);
  assert.ok(start > 2.9);
  assert.ok(end < 2.7);
});
