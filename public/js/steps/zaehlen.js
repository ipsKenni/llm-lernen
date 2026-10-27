// Schritt 1 – Bigram durch Zählen: Heatmap, Sampling, Loss.
import { makeRng } from '../rng.js';
import { sampleWord } from '../bigram-count.js';
import { createHeatmap, renderProbBars, renderTiles, fmt, pct, int } from '../viz.js';

// '.' bedeutet am Zeilenanfang "Anfang", als Ziel "Ende".
const asFrom = (c) => (c === '.' ? 'Anfang' : `„${c}“`);
const asTo = (c) => (c === '.' ? 'Ende' : `„${c}“`);

export function init(root, ctx) {
  const { tok, N, P } = ctx;
  const itos = tok.itos;
  const rng = makeRng((Date.now() & 0xffffffff) >>> 0);

  // --- Heatmap + Zeilendetail ------------------------------------------
  const rowChar = root.querySelector('#count-row-char');
  const rowBars = root.querySelector('#count-row-bars');
  const showRow = (r) => {
    rowChar.textContent = itos[r] === '.' ? '. (Anfang)' : itos[r];
    renderProbBars(rowBars, P[r], itos, { topK: 10 });
  };
  const heat = createHeatmap(root.querySelector('#count-heatmap'), {
    rowLabels: itos,
    rowTitle: 'aktuelles Zeichen',
    colTitle: 'nächstes Zeichen',
    describe: (r, c) => [
      [`${asFrom(itos[r])} → ${asTo(itos[c])}`],
      [`${int(N[r][c])}-mal`, 'gezählt'],
      [pct(P[r][c]), 'Wahrscheinlichkeit'],
    ],
    onSelect: (r) => showRow(r),
  });
  heat.update(N);
  heat.select(1);
  showRow(1);

  // --- Namen Schritt für Schritt ziehen ----------------------------------
  const tiles = root.querySelector('#sample-tiles');
  const readout = root.querySelector('#sample-readout');
  const bars = root.querySelector('#sample-bars');
  const list = root.querySelector('#sample-list');
  let current = 0;
  let chars = [];
  let done = false;

  const addToList = (word) => {
    const li = document.createElement('li');
    if (word) li.textContent = word;
    else { li.textContent = 'leer'; li.className = 'is-empty'; }
    list.prepend(li);
    while (list.children.length > 30) list.lastChild.remove();
  };
  const drawTiles = (ended) => {
    const shown = ['.', ...chars, ...(ended ? ['.'] : [])];
    renderTiles(tiles, shown, shown.map((c) => tok.stoi.get(c)), { activeIndex: shown.length - 1 });
  };
  const reset = () => {
    current = 0; chars = []; done = false;
    drawTiles(false);
    readout.textContent = 'Noch nichts gezogen. Der Name beginnt beim Punkt.';
    renderProbBars(bars, P[0], itos, { topK: 8 });
  };

  root.querySelector('#sample-step').addEventListener('click', () => {
    if (done) reset();
    const probs = P[current];
    const next = rng.sample(probs);
    renderProbBars(bars, probs, itos, { topK: 8, highlight: next });
    if (next === 0) {
      done = true;
      drawTiles(true);
      const word = chars.join('');
      readout.textContent = `Aus der Zeile ${asFrom(itos[current])} wurde das Ende gezogen (${pct(probs[next])}). Fertig: „${word}“. Klick nochmal für einen neuen Namen.`;
      addToList(word);
    } else {
      readout.textContent = `Aus der Zeile ${asFrom(itos[current])} wurde „${itos[next]}“ gezogen, Wahrscheinlichkeit ${pct(probs[next])}.`;
      chars.push(itos[next]);
      current = next;
      drawTiles(false);
    }
  });
  root.querySelector('#sample-new').addEventListener('click', reset);
  root.querySelector('#sample-ten').addEventListener('click', () => {
    for (let i = 0; i < 10; i++) addToList(sampleWord(P, tok, rng).word);
  });
  reset();

  // --- Loss ------------------------------------------------------------
  root.querySelector('#count-loss').textContent = fmt(ctx.countLoss);
  const nllInput = root.querySelector('#nll-input');
  const tbody = root.querySelector('#nll-table tbody');
  const avgCell = root.querySelector('#nll-avg');
  function updateNll() {
    const word = nllInput.value.toLowerCase().trim();
    const chars = ['.', ...word, '.'];
    const rows = [];
    let sum = 0;
    let n = 0;
    for (let i = 0; i < chars.length - 1; i++) {
      const a = tok.stoi.get(chars[i]);
      const b = tok.stoi.get(chars[i + 1]);
      const tr = document.createElement('tr');
      const cells = [`${chars[i]}${chars[i + 1]}`];
      if (a === undefined || b === undefined) {
        cells.push('unbekanntes Zeichen', '–');
      } else {
        const p = P[a][b];
        sum -= Math.log(p);
        n += 1;
        cells.push(pct(p, 2), fmt(-Math.log(p)));
      }
      for (const c of cells) {
        const td = document.createElement('td');
        td.textContent = c;
        tr.append(td);
      }
      rows.push(tr);
    }
    tbody.replaceChildren(...rows);
    avgCell.textContent = n ? fmt(sum / n) : '–';
  }
  nllInput.addEventListener('input', updateNll);
  updateNll();
}
