// Schritt 5 – Das GPT trainieren: Einstellungen, Loss-Kurve, Beispiele.
import { buildTokenizer } from '../tokenizer.js';
import { countBigrams, countsToProbs, averageNll } from '../bigram-count.js';
import { DEFAULT_CONFIG, countParams } from '../transformer.js';
import { createLineChart, fmt, int } from '../viz.js';
import { movingAverage } from './lernen.js';

const CONFIG_KEYS = ['nEmbd', 'nHead', 'nLayer', 'lr', 'batchSize', 'steps'];
const CUSTOM_EXAMPLE = 'apfel\nbirne\nkirsche\npflaume\nmango\nbanane\nerdbeere\nhimbeere\nheidelbeere\nstachelbeere\nquitte\naprikose\npfirsich\nnektarine\nmelone\nzitrone\nlimette\norange\nmandarine\ngrapefruit\nananas\nkiwi\nfeige\ndattel\nlitschi\npapaya\nmaracuja\nholunder\nbrombeere\njohannisbeere';

export function init(root, ctx) {
  const gpt = ctx.gpt;
  const $ = (sel) => root.querySelector(sel);
  const selects = Object.fromEntries(CONFIG_KEYS.map((k) => [k, $(`#cfg-${k}`)]));
  const datasetSel = $('#cfg-dataset');
  const customBox = $('#custom-data');
  const customText = $('#cfg-custom');
  const runBtn = $('#gpt-run');
  const stepBtn = $('#gpt-step');
  const resetBtn = $('#gpt-reset');
  const paramInfo = $('#gpt-param-count');
  const stepNo = $('#gpt-stepno');
  const lossEl = $('#gpt-loss');
  const speedEl = $('#gpt-speed');
  const sampleLog = $('#gpt-samples');
  customText.value = CUSTOM_EXAMPLE;

  const chart = createLineChart($('#gpt-chart'), { references: [] });
  let refsFor = null;   // für welchen Datensatz die Vergleichslinien berechnet sind

  // --- Einstellungen ↔ Formular ------------------------------------------
  function readConfig() {
    const cfg = { ...DEFAULT_CONFIG };
    for (const k of CONFIG_KEYS) cfg[k] = Number(selects[k].value);
    return cfg;
  }
  function writeConfig(cfg, dataset) {
    for (const k of CONFIG_KEYS) selects[k].value = String(cfg[k]);
    datasetSel.value = dataset.kind;
    if (dataset.kind === 'custom') customText.value = dataset.text;
    customBox.hidden = dataset.kind !== 'custom';
  }
  function readDataset() {
    return { kind: datasetSel.value, text: datasetSel.value === 'custom' ? customText.value : '' };
  }

  async function reinit() {
    const dataset = readDataset();
    const words = await ctx.loadWords(dataset);
    if (words.length < 2) {
      paramInfo.textContent = 'Die eigene Liste braucht mindestens zwei Wörter, eins pro Zeile.';
      return;
    }
    gpt.init(readConfig(), dataset, words);
    sampleLog.replaceChildren();
    render();
  }

  // Vergleichslinien: Bigram-Zähl-Loss und blindes Raten für den aktuellen Datensatz.
  function updateReferences() {
    const s = gpt.state;
    if (!s.words.length || refsFor === s.words) return;
    refsFor = s.words;
    const tok = buildTokenizer(s.words);
    const P = countsToProbs(countBigrams(s.words, tok));
    chart.setReferences([
      { value: averageNll(s.words, tok, P), label: 'Bigram' },
      { value: Math.log(tok.vocabSize), label: 'Raten' },
    ]);
  }

  function renderSamples() {
    const items = gpt.state.samples.slice().reverse().map(({ step, names }) => {
      const li = document.createElement('li');
      const when = document.createElement('span');
      when.className = 'when';
      when.textContent = `Schritt ${int(step)}`;
      const what = document.createElement('span');
      what.className = 'what';
      names.forEach(({ name, known }, i) => {
        if (i) what.append(', ');
        const el = document.createElement(known ? 'span' : 'b');
        el.textContent = name || '(leer)';
        what.append(el);
      });
      li.append(when, what);
      return li;
    });
    sampleLog.replaceChildren(...items);
  }

  let rafPending = false;
  function render() {
    rafPending = false;
    const s = gpt.state;
    const vocab = s.itos.length || ctx.tok.vocabSize;
    const n = countParams(s.cfg, vocab);
    paramInfo.textContent = `${int(n)} Parameter bei ${vocab} Tokens im Vokabular. Pro Schritt werden ${s.cfg.batchSize} ${s.cfg.batchSize === 1 ? 'Wort' : 'Wörter'} gelernt.`;
    $('#gpt-param-inline').textContent = int(countParams(DEFAULT_CONFIG, ctx.tok.vocabSize));
    stepNo.textContent = `${int(s.step)} / ${int(s.cfg.steps)}`;
    const recent = s.losses.slice(-50);
    lossEl.textContent = recent.length ? fmt(recent.reduce((a, b) => a + b, 0) / recent.length) : '–';
    speedEl.textContent = s.stepsPerSec ? fmt(s.stepsPerSec, 1) : '–';
    const finished = s.step >= s.cfg.steps;
    runBtn.textContent = s.running ? 'Pause' : finished ? 'Fertig trainiert' : s.step > 0 ? 'Weiter trainieren' : 'Training starten';
    runBtn.disabled = !s.ready || (finished && !s.running);
    stepBtn.disabled = !s.ready || s.running || finished;
    for (const el of [...Object.values(selects), datasetSel, customText]) el.disabled = s.running;
    updateReferences();
    chart.update([
      { name: 'Loss pro Schritt', values: s.losses, alpha: 0.3, width: 1, scale: false },
      { name: 'Mittel (50 Schritte)', values: movingAverage(s.losses, 50) },
    ], Math.max(s.cfg.steps, s.losses.length));
  }
  const scheduleRender = () => {
    if (!rafPending) { rafPending = true; requestAnimationFrame(render); }
  };

  gpt.on((msg) => {
    if (msg.type === 'samples') renderSamples();
    if (msg.type === 'ready') { writeConfig(gpt.state.cfg, gpt.state.dataset); renderSamples(); }
    scheduleRender();
  });

  for (const el of Object.values(selects)) el.addEventListener('change', reinit);
  datasetSel.addEventListener('change', () => { customBox.hidden = datasetSel.value !== 'custom'; reinit(); });
  let typing = 0;
  customText.addEventListener('input', () => { clearTimeout(typing); typing = setTimeout(reinit, 600); });

  runBtn.addEventListener('click', () => (gpt.state.running ? gpt.pause() : gpt.train()));
  stepBtn.addEventListener('click', () => gpt.step());
  resetBtn.addEventListener('click', reinit);

  // --- Gewichte als Datei ----------------------------------------------
  $('#gpt-export').addEventListener('click', async () => {
    const { data } = await gpt.exportWeights();
    const blob = new Blob([JSON.stringify({ ...data, dataset: gpt.state.dataset })], { type: 'application/json' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `kleinstes-llm-schritt-${gpt.state.step}.json`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  });
  $('#gpt-import').addEventListener('change', async (e) => {
    const file = e.target.files[0];
    e.target.value = '';
    if (!file) return;
    try {
      const data = JSON.parse(await file.text());
      if (!Array.isArray(data.weights) || !data.cfg || !Array.isArray(data.itos)) throw new Error('Format');
      const dataset = data.dataset?.kind ? data.dataset : { kind: 'names', text: '' };
      const words = await ctx.loadWords(dataset);
      if (countParams(data.cfg, data.itos.length) !== data.weights.length) throw new Error('Größe');
      gpt.importData(data, dataset, words);
      paramInfo.textContent = `Gewichte aus ${file.name} geladen.`;
    } catch {
      paramInfo.textContent = 'Diese Datei enthält keine passenden Gewichte. Lade eine Datei, die über „Gewichte herunterladen“ gespeichert wurde.';
    }
  });

  writeConfig(gpt.state.cfg, gpt.state.dataset);
  renderSamples();
  render();
}
