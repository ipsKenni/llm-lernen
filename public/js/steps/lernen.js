// Schritt 3 – Die Bigram-Tabelle per Gradientenabstieg lernen.
import { makeRng } from '../rng.js';
import { bigramPairs, createBigramNet, trainStep } from '../bigram-neural.js';
import { softmaxNumbers } from '../nn.js';
import { createHeatmap, createLineChart, fmt, int, pct } from '../viz.js';

const MAX_STEPS = 1500;
const FRAME_BUDGET_MS = 6;   // bewusst gebremst, damit man beim Lernen zusehen kann

export function movingAverage(values, window) {
  const out = [];
  let sum = 0;
  for (let i = 0; i < values.length; i++) {
    sum += values[i];
    if (i >= window) sum -= values[i - window];
    out.push(sum / Math.min(i + 1, window));
  }
  return out;
}

let api = null;

export function init(root, ctx) {
  const { tok, P } = ctx;
  const itos = tok.itos;
  const pairs = bigramPairs(ctx.words, tok);
  let rng, net, losses, running = false, raf = 0;

  const runBtn = root.querySelector('#nb-run');
  const stepBtn = root.querySelector('#nb-step');
  const lrSel = root.querySelector('#nb-lr');
  const stepNo = root.querySelector('#nb-stepno');
  const lossEl = root.querySelector('#nb-loss');
  root.querySelector('#nb-target').textContent = fmt(ctx.countLoss);

  const chart = createLineChart(root.querySelector('#nb-chart'), {
    references: [
      { value: ctx.countLoss, label: 'Zähl-Modell' },
      { value: Math.log(tok.vocabSize), label: 'Raten' },
    ],
  });
  const describe = (matrix) => (r, c) => [
    [`${itos[r] === '.' ? 'Anfang' : `„${itos[r]}“`} → ${itos[c] === '.' ? 'Ende' : `„${itos[c]}“`}`],
    [pct(matrix()[r][c]), 'Wahrscheinlichkeit'],
  ];
  let learned = [];
  const heatLearned = createHeatmap(root.querySelector('#nb-heat-learned'), { rowLabels: itos, describe: describe(() => learned) });
  const heatCounted = createHeatmap(root.querySelector('#nb-heat-counted'), { rowLabels: itos, describe: describe(() => P) });
  heatCounted.update(P);

  function reset() {
    pause();
    rng = makeRng(42);
    net = createBigramNet(tok.vocabSize, rng);
    losses = [];
    draw();
  }

  function draw() {
    learned = net.W.map((row) => softmaxNumbers(row.map((w) => w.data)));
    heatLearned.update(learned);
    chart.update([
      { name: 'Loss pro Schritt', values: losses, alpha: 0.3, width: 1, scale: false },
      { name: 'Mittel (20 Schritte)', values: movingAverage(losses, 20) },
    ], Math.max(100, losses.length));
    stepNo.textContent = int(losses.length);
    lossEl.textContent = losses.length ? fmt(losses[losses.length - 1]) : '–';
    runBtn.textContent = running ? 'Pause' : losses.length >= MAX_STEPS ? 'Nochmal von vorn' : losses.length ? 'Weiter trainieren' : 'Training starten';
    stepBtn.disabled = running;
  }

  const step = () => losses.push(trainStep(net, pairs, rng, { lr: Number(lrSel.value) }));

  function frame() {
    const t0 = performance.now();
    while (performance.now() - t0 < FRAME_BUDGET_MS && losses.length < MAX_STEPS) step();
    if (losses.length >= MAX_STEPS) running = false;
    draw();
    if (running) raf = requestAnimationFrame(frame);
  }
  function pause() {
    running = false;
    cancelAnimationFrame(raf);
    if (net) draw();
  }

  runBtn.addEventListener('click', () => {
    if (running) return pause();
    if (losses.length >= MAX_STEPS) reset();
    running = true;
    raf = requestAnimationFrame(frame);
  });
  stepBtn.addEventListener('click', () => { step(); draw(); });
  root.querySelector('#nb-reset').addEventListener('click', reset);

  reset();
  api = { pause };
}

export function hide() { api?.pause(); }
