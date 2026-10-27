// Schritt 8 – Architektur verstehen, RoPE ausprobieren, Modellgröße wählen.
import { PRESETS, makeConfig, countParams, trainMemoryBytes, trainFlopsPerToken } from '../llama/config.js';
import { fmt, int } from '../viz.js';
import { part2 } from './teil2.js';

const CPU_PRESETS = new Set(['winzig', 'mini']);
const PRESET_NOTES = {
  winzig: 'Für Namen oder zum Ausprobieren, läuft auch auf dem Prozessor.',
  mini: 'Der schnelle Einstieg für Texte.',
  klein: 'Die beste Wahl für Märchen und TinyStories.',
  mittel: 'Braucht eine kräftige Grafikkarte und Geduld.',
};

const durationText = (min) => (min < 1 ? `${Math.max(1, Math.round(min * 60))} s` : min < 90 ? `${fmt(min, min < 10 ? 1 : 0)} min` : `${fmt(min / 60, 1)} h`);

const SVG = 'http://www.w3.org/2000/svg';
const el = (tag, attrs, parent) => {
  const e = document.createElementNS(SVG, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  parent?.append(e);
  return e;
};

export function init(root) {
  const p2 = part2();
  const S = p2.state;
  const $ = (sel) => root.querySelector(sel);

  // --- RoPE-Demo: ein 2-D-Zahlenpaar, gedreht um Position · θ ------------
  const THETA = 0.45;
  const Q = [85, 0], K = [20, 70];
  const svg = $('#rope-svg');
  el('circle', { cx: 0, cy: 0, r: 100, class: 'rope-circle' }, svg);
  el('line', { x1: -105, y1: 0, x2: 105, y2: 0, class: 'rope-axis' }, svg);
  el('line', { x1: 0, y1: -105, x2: 0, y2: 105, class: 'rope-axis' }, svg);
  const defs = el('defs', {}, svg);
  for (const id of ['q', 'k']) {
    const m = el('marker', { id: `rope-arrow-${id}`, viewBox: '0 0 10 10', refX: 8, refY: 5, markerWidth: 6, markerHeight: 6, orient: 'auto-start-reverse' }, defs);
    el('path', { d: 'M0,0 L10,5 L0,10 z', class: `rope-head-${id}` }, m);
  }
  const qLine = el('line', { x1: 0, y1: 0, class: 'rope-q', 'marker-end': 'url(#rope-arrow-q)' }, svg);
  const kLine = el('line', { x1: 0, y1: 0, class: 'rope-k', 'marker-end': 'url(#rope-arrow-k)' }, svg);
  const qLabel = el('text', { class: 'rope-label' }, svg);
  const kLabel = el('text', { class: 'rope-label' }, svg);
  qLabel.textContent = 'q';
  kLabel.textContent = 'k';
  const rot = ([x, y], a) => [x * Math.cos(a) - y * Math.sin(a), x * Math.sin(a) + y * Math.cos(a)];
  function renderRope() {
    const m = Number($('#rope-m').value), n = Number($('#rope-n').value);
    $('#rope-m-out').textContent = m;
    $('#rope-n-out').textContent = n;
    const q = rot(Q, m * THETA), k = rot(K, n * THETA);
    // SVG-y zeigt nach unten → für die Anzeige spiegeln
    qLine.setAttribute('x2', q[0]); qLine.setAttribute('y2', -q[1]);
    kLine.setAttribute('x2', k[0]); kLine.setAttribute('y2', -k[1]);
    qLabel.setAttribute('x', q[0] * 1.15); qLabel.setAttribute('y', -q[1] * 1.15);
    kLabel.setAttribute('x', k[0] * 1.15); kLabel.setAttribute('y', -k[1] * 1.15);
    const score = (q[0] * k[0] + q[1] * k[1]) / 100;
    $('#rope-readout').textContent = `Abstand m − n = ${m - n}. Score q·k = ${fmt(score, 2)}. Dieselbe Differenz ergibt immer denselben Score, egal wo im Text.`;
  }
  $('#rope-m').addEventListener('input', renderRope);
  $('#rope-n').addEventListener('input', renderRope);
  renderRope();

  // --- Modellgröße ------------------------------------------------------
  const grid = $('#preset-choices');
  const cards = {};
  for (const [key, p] of Object.entries(PRESETS)) {
    const b = document.createElement('button');
    b.type = 'button';
    b.className = 'choice';
    b.setAttribute('role', 'radio');
    b.addEventListener('click', () => { if (!b.disabled) { S.preset = key; p2.emit('preset'); } });
    grid.append(b);
    cards[key] = b;
  }

  function render() {
    const vocab = S.tokenizer?.vocabSize ?? S.vocab;
    const cpuOnly = S.gpu === null;
    if (cpuOnly && !CPU_PRESETS.has(S.preset)) S.preset = 'winzig';
    for (const [key, b] of Object.entries(cards)) {
      const p = PRESETS[key];
      const cfg = makeConfig(p, vocab);
      const n = countParams(cfg);
      const mem = trainMemoryBytes(cfg, p.batch);
      b.disabled = cpuOnly && !CPU_PRESETS.has(key);
      b.setAttribute('aria-checked', String(key === S.preset));
      b.replaceChildren();
      const title = document.createElement('strong');
      title.textContent = `${p.label}: ${fmt(n / 1e6, n < 1e6 ? 2 : 1)} Mio. Parameter`;
      const dims = document.createElement('span');
      dims.textContent = `${p.nLayers} Schichten, Breite ${p.dim}, ${p.nHeads} Köpfe (${p.nKvHeads} KV), Kontext ${p.ctx}`;
      const memory = document.createElement('span');
      // Ohne GPU schafft JavaScript grob 1 Milliarde Rechenschritte pro Sekunde.
      const cpuMinutes = (trainFlopsPerToken(cfg) * p.batch * p.ctx * 1000) / 1e9 / 60;
      const measured = S.bench?.results.find((e) => e.preset === key && !e.error);
      memory.textContent = measured
        ? `Gemessen: ${durationText(measured.minutesPer1000)} pro 1.000 Schritte`
        : cpuOnly
          ? `Auf dem Prozessor ca. ${cpuMinutes < 1.5 ? '1' : int(cpuMinutes)} min pro 1.000 Schritte`
          : `Speicher beim Training ca. ${int(mem / 2 ** 20)} MB`;
      const note = document.createElement('span');
      note.textContent = PRESET_NOTES[key];
      b.append(title, dims, memory, note);
    }
    const g = S.gpu;
    $('#arch-gpu').textContent = g === undefined
      ? 'Suche nach einer Grafikkarte …'
      : g === null
        ? 'Dein Browser bietet kein WebGPU an. Das Training läuft dann auf dem Prozessor, deshalb sind nur „Winzig“ und „Mini“ wählbar. Mit einem aktuellen Chrome oder Edge geht es auf der Grafikkarte deutlich schneller.'
        : `WebGPU gefunden${g.vendor ? `: ${g.vendor}${g.architecture ? ` (${g.architecture})` : ''}` : ''}. Das Training läuft auf deiner Grafikkarte.${g.fallback ? ' Achtung: Das ist ein Software-Adapter ohne echte GPU, er ist sehr langsam.' : ''}`;
    const p = PRESETS[S.preset];
    const perStep = p.batch * p.ctx;
    $('#preset-info').textContent = `Pro Trainingsschritt lernt das Modell aus ${p.batch} Textstücken mit je ${p.ctx} Tokens, also ${int(perStep)} Tokens. ${S.tokInfo ? `Nach 2.000 Schritten hat es rund ${fmt((2000 * perStep) / S.tokInfo.trainTokens, 1)}-mal den ganzen Datensatz gesehen.` : ''}`;
  }

  // --- Benchmark ---------------------------------------------------------
  const benchBtn = $('#bench-run');
  const benchBody = $('#bench-body');
  let benchText = '';
  function benchRow(e) {
    const tr = document.createElement('tr');
    const cells = e.error
      ? [PRESETS[e.preset].label, `${fmt(e.params / 1e6, 2)} Mio.`, `nicht möglich: ${e.error}`, '–', '–']
      : [PRESETS[e.preset].label, `${fmt(e.params / 1e6, 2)} Mio.`, int(e.tokensPerSec), `${int(e.msPerStep)} ms`, durationText(e.minutesPer1000)];
    for (const c of cells) { const td = document.createElement('td'); td.textContent = c; tr.append(td); }
    return tr;
  }
  p2.trainer.on((m) => { if (m.type === 'benchmark') benchBody.append(benchRow(m.entry)); });
  benchBtn.addEventListener('click', async () => {
    if (S.running) { $('#bench-status').textContent = 'Pausiere zuerst das Training in Schritt 9.'; return; }
    benchBtn.disabled = true;
    benchBody.replaceChildren();
    $('#bench-wrap').hidden = false;
    $('#bench-copy').hidden = true;
    const cpuOnly = S.gpu === null;
    const presets = Object.keys(PRESETS).filter((k) => !cpuOnly || CPU_PRESETS.has(k));
    $('#bench-status').textContent = 'Messe …';
    try {
      const r = await p2.trainer.benchmark(presets, cpuOnly ? 'cpu' : 'auto', S.tokenizer?.vocabSize ?? 2048);
      S.bench = r;
      const where = r.kind === 'webgpu' ? `WebGPU (${[r.adapter?.vendor, r.adapter?.architecture, r.adapter?.description].filter(Boolean).join(', ') || 'unbekannte GPU'})` : 'Prozessor (kein WebGPU)';
      const c = r.compare;
      const cmp = c && !c.error
        ? ` Die aktuelle Engine ${r.engine} ist bei „${PRESETS[c.preset].label}“ ${fmt(c.speedup, 2)}-mal so schnell wie Engine 1 (${int(c.oldMs)} → ${int(c.newMs)} ms pro Schritt).`
        : '';
      $('#bench-status').textContent = `Gemessen auf: ${where}. Reine Rechenleistung der Matrixmultiplikation: ${fmt(r.gflops, 1)} GFLOP/s.${cmp}`;
      benchText = [
        `Benchmark llm.ips.codes, Engine ${r.engine}, ${new Date().toLocaleString('de-DE')}`,
        `Gerät: ${where}`,
        `Matmul: ${fmt(r.gflops, 1)} GFLOP/s`,
        ...r.results.map((e) => (e.error ? `${PRESETS[e.preset].label}: ${e.error}` : `${PRESETS[e.preset].label}: ${int(e.tokensPerSec)} Tokens/s, ${int(e.msPerStep)} ms/Schritt${e.variant ? `, Matmul-Varianten ${JSON.stringify(e.variant)}` : ''}`)),
        ...(r.compare && !r.compare.error ? [`Vergleich ${PRESETS[r.compare.preset].label}: Engine 1 ${int(r.compare.oldMs)} ms, Engine ${r.engine} ${int(r.compare.newMs)} ms pro Schritt (${fmt(r.compare.speedup, 2)}×)`] : []),
        `Browser: ${navigator.userAgent}`,
      ].join('\n');
      $('#bench-copy').hidden = false;
      render();
    } catch (e) {
      $('#bench-status').textContent = `Messung fehlgeschlagen: ${e.message}`;
    } finally {
      benchBtn.disabled = false;
    }
  });
  $('#bench-copy').addEventListener('click', async () => {
    try { await navigator.clipboard.writeText(benchText); $('#bench-copy').textContent = 'Kopiert'; } catch { $('#bench-copy').textContent = 'Kopieren nicht erlaubt'; }
    setTimeout(() => { $('#bench-copy').textContent = 'Ergebnis kopieren'; }, 1500);
  });

  p2.on((what) => { if (['gpu', 'preset', 'tokenized', 'restored', 'setup'].includes(what)) render(); });
  render();
}
