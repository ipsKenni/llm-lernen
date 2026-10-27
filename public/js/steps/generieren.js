// Schritt 6 – Mit dem trainierten GPT generieren und Attention ansehen.
import { DEFAULT_CONFIG } from '../transformer.js';
import { makeRng } from '../rng.js';
import { createHeatmap, renderProbBars, renderTiles, fmt, pct, int } from '../viz.js';

export function init(root, ctx) {
  const gpt = ctx.gpt;
  const $ = (sel) => root.querySelector(sel);
  const rng = makeRng((Date.now() & 0xffffffff) >>> 0);
  const untrained = $('#gen-untrained');
  const main = $('#gen-main');
  const input = $('#gen-input');
  const temp = $('#gen-temp');
  const tempOut = $('#gen-temp-out');
  const tiles = $('#gen-tiles');
  const readout = $('#gen-readout');
  const bars = $('#gen-bars');
  const attnBox = $('#gen-attn');
  const layerWrap = $('#gen-layer-wrap');
  const layerSel = $('#gen-layer');
  const list = $('#gen-list');
  const quickStatus = $('#gen-quick-status');

  let last = null;          // letztes Ergebnis von inspect()
  let maps = [];            // Heatmaps pro Kopf
  let ended = false;
  let busy = false;

  const itos = () => gpt.state.itos;
  const label = (c) => (c === '.' ? '.' : c);

  function syncVisibility() {
    const trained = gpt.isTrained() && !(gpt.state.running && gpt.state.step < 50);
    untrained.hidden = trained;
    main.hidden = !trained;
    if (gpt.state.running && !trained) quickStatus.textContent = `Training läuft: Schritt ${int(gpt.state.step)} von ${int(gpt.state.cfg.steps)} …`;
    return trained;
  }

  async function refresh() {
    if (!syncVisibility() || busy) return;
    busy = true;
    const t = Number(temp.value);
    tempOut.textContent = fmt(t, 1);
    const res = await gpt.inspect(input.value.toLowerCase(), t);
    busy = false;
    if (res.prefix !== input.value.toLowerCase()) input.value = res.prefix;
    last = res;
    const chars = res.tokens.map((id) => itos()[id]);
    const shown = ended ? [...chars, '.'] : chars;
    renderTiles(tiles, shown, ended ? [...res.tokens, 0] : res.tokens, { activeIndex: shown.length - 1 });
    renderProbBars(bars, res.probs, itos(), { topK: 8 });
    drawAttention(res, chars);
  }

  function drawAttention(res, chars) {
    const nLayer = res.attn.length;
    const nHead = res.attn[0].length;
    layerWrap.hidden = nLayer < 2;
    if (layerSel.options.length !== nLayer) {
      layerSel.replaceChildren(...Array.from({ length: nLayer }, (_, l) => new Option(String(l + 1), String(l))));
    }
    const layer = Math.min(Number(layerSel.value) || 0, nLayer - 1);
    if (maps.length !== nHead) {
      attnBox.replaceChildren();
      maps = Array.from({ length: nHead }, (_, h) => {
        const fig = document.createElement('figure');
        const cap = document.createElement('figcaption');
        cap.textContent = `Kopf ${h + 1}`;
        const canvas = document.createElement('canvas');
        canvas.className = 'heatmap';
        canvas.setAttribute('role', 'img');
        canvas.setAttribute('aria-label', `Attention-Gewichte von Kopf ${h + 1}`);
        fig.append(cap, canvas);
        attnBox.append(fig);
        let current = { matrix: [], labels: [] };
        const map = createHeatmap(canvas, {
          rowLabels: [],
          sqrt: false,
          fixedMax: 1,
          describe: (r, c) => c > r
            ? [[`Position ${r} → ${c}`], ['maskiert', '(Zukunft)']]
            : [[`„${current.labels[r]}“ (Pos. ${r}) schaut auf „${current.labels[c]}“ (Pos. ${c})`], [pct(current.matrix[r][c]), 'Aufmerksamkeit']],
        });
        return { set: (matrix, labels) => { current = { matrix, labels }; map.update(matrix, labels); } };
      });
    }
    const labels = chars.map(label);
    maps.forEach((m, h) => {
      const rows = res.attn[layer][h];
      const matrix = rows.map((w) => labels.map((_, t) => w[t] ?? 0));
      m.set(matrix, labels);
    });
  }

  // Ein Zeichen ziehen – mit derselben Temperatur, mit der die Balken berechnet wurden.
  async function drawNext() {
    if (!last) await refresh();
    if (ended) { ended = false; input.value = ''; await refresh(); }
    const next = rng.sample(last.probs);
    const p = last.probs[next];
    renderProbBars(bars, last.probs, itos(), { topK: 8, highlight: next });
    if (next === 0 || last.tokens.length >= gpt.state.cfg.blockSize) {
      ended = true;
      readout.textContent = `Gezogen: Ende (${pct(p)}). Der Name ist „${input.value}“.`;
      const chars = last.tokens.map((id) => itos()[id]);
      renderTiles(tiles, [...chars, '.'], [...last.tokens, 0], { activeIndex: chars.length });
      return false;
    }
    input.value += itos()[next];
    readout.textContent = `Gezogen: „${itos()[next]}“ mit ${pct(p)}. Darunter siehst du schon die Verteilung für das nächste Zeichen.`;
    await refresh();
    return true;
  }

  $('#gen-step').addEventListener('click', drawNext);
  $('#gen-finish').addEventListener('click', async () => {
    if (ended) { ended = false; input.value = ''; }
    for (let i = 0; i < gpt.state.cfg.blockSize && (await drawNext()); i++) { /* weiter */ }
  });
  $('#gen-clear').addEventListener('click', () => { input.value = ''; ended = false; readout.textContent = ''; refresh(); });
  input.addEventListener('input', () => { ended = false; readout.textContent = ''; refresh(); });
  temp.addEventListener('input', refresh);
  layerSel.addEventListener('change', refresh);

  $('#gen-ten').addEventListener('click', async () => {
    const { names } = await gpt.generate(20, Number(temp.value));
    list.replaceChildren(...names.map(({ name, known }) => {
      const li = document.createElement('li');
      li.textContent = name || '(leer)';
      if (!known) li.classList.add('is-novel');
      li.title = known ? 'steht so in den Trainingsdaten' : 'neu erfunden';
      return li;
    }));
  });

  $('#gen-quick').addEventListener('click', async () => {
    const words = await ctx.loadWords({ kind: 'names', text: '' });
    gpt.init({ ...DEFAULT_CONFIG, steps: 300 }, { kind: 'names', text: '' }, words);
    quickStatus.textContent = 'Starte …';
    const off = gpt.on((msg) => {
      if (msg.type === 'ready') gpt.train();
      if (msg.type === 'done') { off(); quickStatus.textContent = ''; }
    });
  });

  gpt.on((msg) => {
    if (msg.type === 'ready') { maps = []; last = null; ended = false; }
    if (['ready', 'done', 'progress', 'running'].includes(msg.type)) {
      const wasHidden = main.hidden;
      syncVisibility();
      if (!main.hidden && (wasHidden || msg.type !== 'progress')) refresh();
    }
  });

  refresh();
}
