// Schritt 4 – Attention zum Nachrechnen und der Bauplan des GPT.
import { renderCode } from '../code-view.js';
import { fmt, pct } from '../viz.js';

// Von Hand gewählte 2-D-Vektoren für ".emma". Dimension 1 ≈ "Vokal",
// Dimension 2 ≈ "Wortanfang". Im echten Modell sind das 16 gelernte Zahlen.
const EXAMPLE = [
  { char: '.', q: [0, 1], k: [0, 2], v: [0, 1] },
  { char: 'e', q: [0, 1.5], k: [2, 0], v: [1, 0] },
  { char: 'm', q: [1.5, 0], k: [-1, 0], v: [0, -1] },
  { char: 'm', q: [1.2, 0.6], k: [-1, 0.5], v: [0, -1] },
  { char: 'a', q: [-1, 0.5], k: [2, 0], v: [1, 0] },
];

const BLOCKS = [
  {
    id: 'embed', title: 'Embedding', sub: 'Token + Position', region: 'gpt-embedding',
    text: 'Die Token-ID wählt eine Zeile aus der Embedding-Matrix, die Position eine Zeile aus der Positions-Matrix. Beide Vektoren werden addiert. Ab hier ist jedes Zeichen eine Liste von 16 Zahlen.',
  },
  {
    id: 'attn', title: 'Attention', sub: 'Positionen tauschen sich aus', region: 'gpt-attention', repeat: true,
    text: 'Jede Position berechnet Query, Key und Value. Mehrere Köpfe arbeiten parallel auf je einem Ausschnitt des Vektors, so kann ein Kopf etwa auf den Wortanfang achten und ein anderer auf den letzten Vokal. Das Ergebnis wird auf den Eingang addiert (Residual-Verbindung), damit nichts verloren geht.',
  },
  {
    id: 'mlp', title: 'MLP', sub: 'Jede Position denkt nach', region: 'gpt-mlp', repeat: true,
    text: 'Zwei lineare Schichten mit einer ReLU dazwischen. Hier wird verarbeitet, was die Attention eingesammelt hat. Auch hier gibt es eine Residual-Verbindung. Attention und MLP zusammen bilden einen Transformer-Block, und große Modelle stapeln davon Dutzende.',
  },
  {
    id: 'head', title: 'lm_head + Softmax', sub: 'Wahrscheinlichkeiten', region: 'gpt-head',
    text: 'Eine letzte lineare Schicht macht aus dem Vektor 27 Logits, eins pro Zeichen im Vokabular. Softmax wandelt sie in Wahrscheinlichkeiten fürs nächste Zeichen um. Beim Training vergleicht der Cross-Entropy-Loss sie mit dem richtigen Zeichen.',
  },
];
const HELPERS = [
  { id: 'linear', title: 'Lineare Schicht', region: 'gpt-linear', text: 'Matrix mal Vektor. Jede Ausgabe ist eine gewichtete Summe aller Eingaben. Fast die gesamte Rechenzeit eines LLM steckt in dieser einen Operation.' },
  { id: 'rmsnorm', title: 'RMSNorm', region: 'gpt-rmsnorm', text: 'Bringt jeden Vektor auf eine einheitliche Größe, bevor er in Attention oder MLP geht. Ohne Normalisierung würden die Zahlen über viele Schichten explodieren oder verschwinden.' },
  { id: 'params', title: 'Alle Parameter', region: 'gpt-params', text: 'Alles, was das Modell je lernt, steckt in diesen Matrizen. Am Anfang sind sie Zufall. Training verschiebt sie Schritt für Schritt.' },
];

export function init(root) {
  // --- Attention von Hand ------------------------------------------------
  const seg = root.querySelector('#att-pos');
  const sharp = root.querySelector('#att-sharp');
  const sharpOut = root.querySelector('#att-sharp-out');
  const tbody = root.querySelector('#att-table tbody');
  const readout = root.querySelector('#att-readout');
  let query = 4;

  const vec = (a) => `[${a.map((x) => fmt(x, 1)).join('; ')}]`;
  const buttons = EXAMPLE.map((tok, i) => {
    const b = document.createElement('button');
    b.type = 'button';
    b.setAttribute('role', 'radio');
    b.textContent = `${i}: ${tok.char}`;
    b.addEventListener('click', () => { query = i; update(); });
    seg.append(b);
    return b;
  });

  function update() {
    buttons.forEach((b, i) => b.setAttribute('aria-checked', String(i === query)));
    const factor = Number(sharp.value);
    sharpOut.textContent = fmt(factor, 1);
    const q = EXAMPLE[query].q;
    // Scores nur für Positionen ≤ query, dann Softmax.
    const scores = EXAMPLE.map((t, i) => (i <= query ? (factor * (q[0] * t.k[0] + q[1] * t.k[1])) / Math.SQRT2 : null));
    const visible = scores.filter((s) => s !== null);
    const max = Math.max(...visible);
    const exps = visible.map((s) => Math.exp(s - max));
    const sum = exps.reduce((a, b) => a + b, 0);
    const weights = exps.map((e) => e / sum);
    const out = [0, 0];
    weights.forEach((w, i) => { out[0] += w * EXAMPLE[i].v[0]; out[1] += w * EXAMPLE[i].v[1]; });

    tbody.replaceChildren(...EXAMPLE.map((t, i) => {
      const tr = document.createElement('tr');
      const masked = i > query;
      if (masked) tr.className = 'is-masked';
      const cells = [`${i}: ${t.char}`, vec(t.k), masked ? 'maskiert' : fmt(scores[i], 2)];
      for (const c of cells) {
        const td = document.createElement('td');
        td.textContent = c;
        tr.append(td);
      }
      const wTd = document.createElement('td');
      if (masked) wTd.textContent = '0 %';
      else {
        const bar = document.createElement('span');
        bar.className = 'wbar';
        bar.style.width = `${Math.round(weights[i] * 80)}px`;
        wTd.append(bar, pct(weights[i]));
      }
      const vTd = document.createElement('td');
      vTd.textContent = vec(t.v);
      tr.append(wTd, vTd);
      return tr;
    }));
    const best = weights.indexOf(Math.max(...weights));
    readout.textContent = `Query von Position ${query} („${EXAMPLE[query].char}“): q = ${vec(q)}. Am meisten Aufmerksamkeit bekommt Position ${best} („${EXAMPLE[best].char}“) mit ${pct(weights[best])}. Ausgabe = Σ Gewicht · Value = ${vec(out.map((x) => Math.round(x * 100) / 100))}. Die Positionen nach ${query} sind maskiert: Das Modell darf nicht in die Zukunft schauen.`;
  }
  sharp.addEventListener('input', update);
  update();

  // --- Bauplan ---------------------------------------------------------
  const stack = root.querySelector('#arch-stack');
  const detail = root.querySelector('#arch-detail');
  const allButtons = [];
  const select = (block, btn) => {
    allButtons.forEach((b) => b.setAttribute('aria-pressed', String(b === btn)));
    const h = document.createElement('h4');
    h.textContent = block.title;
    const p = document.createElement('p');
    p.textContent = block.text;
    const code = document.createElement('div');
    code.className = 'code';
    code.dataset.code = `transformer.js#${block.region}`;
    detail.replaceChildren(h, p, code);
    renderCode(code);
  };
  const makeButton = (block) => {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.setAttribute('aria-pressed', 'false');
    btn.textContent = block.title;
    if (block.sub) {
      const small = document.createElement('small');
      small.textContent = block.sub;
      btn.append(small);
    }
    if (block.repeat) btn.classList.add('arch-block');
    btn.addEventListener('click', () => select(block, btn));
    allButtons.push(btn);
    return btn;
  };
  for (const block of BLOCKS) {
    const li = document.createElement('li');
    li.append(makeButton(block));
    stack.append(li);
  }
  root.querySelector('#arch-helpers').append(...HELPERS.map(makeButton));
  select(BLOCKS[1], allButtons[1]);
}
