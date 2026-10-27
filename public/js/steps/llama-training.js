// Schritt 9 – Training starten, pausieren, fortsetzen; Kurven und Beispiele.
import { PRESETS, countParams } from '../llama/config.js';
import { DATASETS, learningRate } from '../llama/data.js';
import { createLineChart, fmt, int } from '../viz.js';
import { movingAverage } from './lernen.js';
import { part2 } from './teil2.js';

const duration = (sec) => {
  if (!Number.isFinite(sec) || sec <= 0) return '–';
  if (sec < 90) return `${Math.round(sec)} s`;
  if (sec < 5400) return `${Math.round(sec / 60)} min`;
  return `${fmt(sec / 3600, 1)} h`;
};

export function init(root) {
  const p2 = part2();
  const S = p2.state;
  const $ = (sel) => root.querySelector(sel);
  const runBtn = $('#tr-run');
  const resetBtn = $('#tr-reset');
  const stepsSel = $('#tr-steps');
  const chart = createLineChart($('#tr-chart'), { references: [], height: 280 });
  const lrChart = createLineChart($('#tr-lr'), { references: [], height: 150 });
  let busy = false;
  let pending = false;

  // Passt das angelegte Modell noch zu Tokenizer und gewählter Größe?
  const modelMatches = () => S.setup && S.setup.preset === S.preset && S.setup.cfg.vocabSize === S.tokenizer?.vocabSize;

  async function start() {
    busy = true;
    render();
    try {
      if (!modelMatches()) {
        await p2.setupModel(S.preset, Number(stepsSel.value), S.gpu === null ? 'cpu' : 'auto');
      }
      const total = S.step >= S.total ? S.step + Number(stepsSel.value) : Math.max(Number(stepsSel.value), S.step + 1);
      S.total = total;
      p2.train(total);
    } catch (e) {
      S.error = e.message;
    } finally {
      busy = false;
      render();
    }
  }

  runBtn.addEventListener('click', () => {
    if (S.running) p2.pause();
    else start();
  });
  resetBtn.addEventListener('click', async () => {
    if (S.step > 0 && !confirm('Das bisherige Training wird gelöscht und das Modell beginnt wieder bei Zufall. Fortfahren?')) return;
    p2.pause();
    busy = true;
    render();
    await p2.trainer.reset();
    await p2.setupModel(S.preset, Number(stepsSel.value), S.gpu === null ? 'cpu' : 'auto');
    busy = false;
    render();
  });
  stepsSel.addEventListener('change', () => {
    if (!S.running && S.step < S.total && S.setup) S.total = Math.max(Number(stepsSel.value), S.step + 1);
    render();
  });

  function render() {
    pending = false;
    const ready = !!S.tokenizer;
    $('#tr-none').hidden = ready;
    $('#tr-panel').hidden = !ready;
    if (!ready) return;
    const ds = DATASETS[S.dataset];
    const preset = PRESETS[S.preset];
    const cfg = modelMatches() ? S.setup.cfg : null;
    const nParams = cfg ? countParams(cfg) : null;
    const where = S.setup?.backend === 'webgpu' ? 'auf der Grafikkarte (WebGPU)' : S.setup?.backend === 'cpu' ? 'auf dem Prozessor' : (S.gpu ? 'auf der Grafikkarte (WebGPU)' : 'auf dem Prozessor');
    $('#tr-summary').textContent = `${ds.label}, Modell „${preset.label}“${nParams ? ` mit ${fmt(nParams / 1e6, 2)} Mio. Parametern` : ''}, Training ${where}.${cfg ? '' : ' Das Modell wird beim Start neu angelegt.'}`;

    const finished = S.setup && S.step >= S.total;
    runBtn.disabled = busy;
    runBtn.textContent = busy ? 'Modell wird angelegt …' : S.running ? 'Pause' : !modelMatches() || S.step === 0 ? 'Training starten' : finished ? `${int(Number(stepsSel.value))} weitere Schritte` : 'Weiter trainieren';
    resetBtn.disabled = busy || S.running;
    stepsSel.disabled = S.running;

    const recent = S.losses.slice(-50);
    $('#tr-step').textContent = S.setup ? `${int(S.step)} / ${int(S.total)}` : '0';
    $('#tr-loss').textContent = recent.length ? fmt(recent.reduce((a, b) => a + b, 0) / recent.length) : '–';
    $('#tr-val').textContent = S.evals.length ? fmt(S.evals.at(-1).valLoss) : '–';
    $('#tr-speed').textContent = S.running && S.tokensPerSec ? int(S.tokensPerSec) : '–';
    $('#tr-eta').textContent = S.running && S.stepsPerSec ? duration((S.total - S.step) / S.stepsPerSec) : '–';

    const vocab = S.tokenizer.vocabSize;
    chart.setReferences([{ value: Math.log(vocab), label: 'Raten' }]);
    const total = Math.max(S.total || Number(stepsSel.value), S.losses.length, 1);
    chart.update([
      { name: 'Train-Loss', values: S.losses, alpha: 0.25, width: 1, scale: false },
      { name: 'Train-Loss geglättet', values: movingAverage(S.losses, 50) },
      { name: 'Val-Loss', points: S.evals.map((e) => [e.step, e.valLoss]), color: '--ink', dash: [5, 4] },
    ], total);
    const maxLr = preset.lr;
    const lrs = Array.from({ length: Math.min(total, 600) }, (_, i) => learningRate(Math.round((i / 600) * total), total, maxLr) * 1000);
    lrChart.update([{ name: 'Lernrate (×1000)', points: lrs.map((v, i) => [Math.round((i / 600) * total), v]), width: 2 }], total);

    const items = S.samples.slice(-12).reverse().map((smp) => {
      const li = document.createElement('li');
      const when = document.createElement('span');
      when.className = 'when';
      when.textContent = `Schritt ${int(smp.step)}`;
      const what = document.createElement('span');
      what.className = 'what';
      const b = document.createElement('b');
      b.textContent = smp.prompt;
      what.append(b, smp.text);
      li.append(when, what);
      return li;
    });
    $('#tr-samples').replaceChildren(...items);
    $('#tr-saved').textContent = S.error ? `Fehler: ${S.error}` : S.savedStep ? `Im Browser gespeichert bei Schritt ${int(S.savedStep)}.` : '';
  }
  const schedule = () => { if (!pending) { pending = true; requestAnimationFrame(render); } };
  p2.on(schedule);
  render();
}
