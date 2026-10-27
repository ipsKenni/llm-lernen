// Schritt 10 – Ins trainierte Llama-Modell hineinschauen.
import { createHeatmap, createScatter, renderProbBars, renderTiles, pct } from '../viz.js';
import { part2 } from './teil2.js';

const START = {
  maerchen: 'Es war einmal ein König, der hatte drei',
  tinystories: 'Once upon a time, there was a little girl named',
  namen: 'emma\nolivia\n',
  upload: '',
};

let api = null;

export function init(root) {
  const p2 = part2();
  const S = p2.state;
  const $ = (sel) => root.querySelector(sel);
  const input = $('#in-prompt');
  const layerSel = $('#in-layer');
  const attnBox = $('#in-attn');
  let maps = [];
  let last = null;
  let lastDataset = null;
  let busy = false, again = false;
  let labels = [];   // Anzeigetext für jedes Token-ID (für die Balken)

  async function refresh() {
    if (!S.setup || S.step === 0) return;
    if (busy) { again = true; return; }
    busy = true;
    try {
      const r = await p2.trainer.inspect(input.value);
      last = r;
      if (labels.length !== S.tokenizer.vocabSize) labels = Array.from({ length: S.tokenizer.vocabSize }, (_, i) => S.tokenizer.display(i));
      renderTiles($('#in-tokens'), r.labels, r.ids);
      renderProbBars($('#in-bars'), r.probs, labels, { topK: 10, endId: S.tokenizer.eot, endLabel: '⟨Ende⟩', endTitle: 'Ende des Textes' });
      drawAttention();
    } finally {
      busy = false;
      if (again) { again = false; refresh(); }
    }
  }

  function drawAttention() {
    if (!last) return;
    const nLayer = last.attn.length, nHead = last.attn[0].length;
    if (layerSel.options.length !== nLayer) {
      layerSel.replaceChildren(...Array.from({ length: nLayer }, (_, l) => new Option(`${l + 1} von ${nLayer}`, String(l))));
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
        let cur = { matrix: [], labels: [] };
        const map = createHeatmap(canvas, {
          rowLabels: [], sqrt: false, fixedMax: 1,
          describe: (r, c) => (c > r
            ? [[`Position ${r} → ${c}`], ['maskiert', '(Zukunft)']]
            : [[`„${cur.labels[r]}“ schaut auf „${cur.labels[c]}“`], [pct(cur.matrix[r][c]), 'Aufmerksamkeit']]),
        });
        return { set(matrix, lab) { cur = { matrix, labels: lab }; map.update(matrix, lab); } };
      });
    }
    maps.forEach((m, h) => {
      const rows = last.attn[layer][h];
      m.set(rows.map((w) => last.labels.map((_, t) => w[t] ?? 0)), last.labels);
    });
  }

  // --- Token-Karte -------------------------------------------------------
  const scatter = createScatter($('#in-map'), {
    height: 460,
    labelIf: (p, i) => (p.merged ? i < 94 + 120 : /[a-zA-Z0-9]/.test(p.label)),
  });
  async function drawMap() {
    $('#in-map-status').textContent = 'Berechne die Karte …';
    const { points } = await p2.trainer.embeddings(500);
    scatter.update(points);
    $('#in-map-status').textContent = `${points.length} Tokens. Fahr über einen Punkt, um zu sehen, welches Token es ist.`;
  }

  function render() {
    const ready = !!S.setup && S.step > 0;
    $('#in-none').hidden = ready;
    $('#in-main').hidden = !ready;
    if (!ready) return;
    if (lastDataset !== S.dataset) {
      lastDataset = S.dataset;
      input.value = START[S.dataset] || S.samples.at(-1)?.text.slice(0, 40) || '';
      maps = [];
    }
    refresh();
  }

  let typing = 0;
  input.addEventListener('input', () => { clearTimeout(typing); typing = setTimeout(refresh, 250); });
  layerSel.addEventListener('change', drawAttention);
  $('#in-map-run').addEventListener('click', drawMap);
  p2.on((what) => { if (['setup', 'restored', 'done', 'finetune'].includes(what)) render(); });
  render();
  api = { render };
}

// Bei jedem Öffnen mit dem aktuellen Trainingsstand neu rechnen.
export function show() { api?.render(); }
