// Schritt 2 – Autograd am einzelnen Neuron: Graph, Backward Schritt für Schritt.
import { Value, buildTopo } from '../value.js';
import { fmt } from '../viz.js';

const DEFAULTS = { x1: 2, w1: -3, x2: 0, w2: 1, b: 6.8814, y: -0.5 };
const SLIDERS = [
  { key: 'x1', label: 'x₁', min: -3, max: 3 },
  { key: 'w1', label: 'w₁', min: -3, max: 3 },
  { key: 'x2', label: 'x₂', min: -3, max: 3 },
  { key: 'w2', label: 'w₂', min: -3, max: 3 },
  { key: 'b', label: 'b', min: -8, max: 8 },
  { key: 'y', label: 'y', min: -1, max: 1 },
];
const LEARN_RATE = 0.1;

// Darstellung jedes Knotens: Name, Rechenvorschrift, Position im Raster.
const LAYOUT = {
  x1: { name: 'x₁', op: 'Eingabe', col: 0, row: 0 },
  w1: { name: 'w₁', op: 'Gewicht', col: 0, row: 1 },
  x2: { name: 'x₂', op: 'Eingabe', col: 0, row: 2 },
  w2: { name: 'w₂', op: 'Gewicht', col: 0, row: 3 },
  x1w1: { name: 'x₁w₁', op: 'x₁ · w₁', col: 1, row: 0.5 },
  x2w2: { name: 'x₂w₂', op: 'x₂ · w₂', col: 1, row: 2.5 },
  s: { name: 's', op: 'x₁w₁ + x₂w₂', col: 2, row: 1.5 },
  b: { name: 'b', op: 'Bias', col: 2, row: 3 },
  n: { name: 'n', op: 's + b', col: 3, row: 2.25 },
  o: { name: 'o', op: 'tanh(n)', col: 4, row: 2.25 },
  yNeg: { name: '−y', op: 'Ziel', col: 4, row: 3.4 },
  d: { name: 'd', op: 'o − y', col: 5, row: 2.8 },
  L: { name: 'L', op: 'd²', col: 6, row: 2.8 },
};
const COL_W = 150, ROW_H = 84, BOX_W = 124, BOX_H = 66, PAD = 6;

// Der Forward-Pass: genau die Formel aus der Überschrift, mit Value-Objekten.
function forward(p) {
  const v = {
    x1: new Value(p.x1), w1: new Value(p.w1), x2: new Value(p.x2), w2: new Value(p.w2),
    b: new Value(p.b), yNeg: new Value(-p.y),
  };
  v.x1w1 = v.x1.mul(v.w1);
  v.x2w2 = v.x2.mul(v.w2);
  v.s = v.x1w1.add(v.x2w2);
  v.n = v.s.add(v.b);
  v.o = v.n.tanh();
  v.d = v.o.add(v.yNeg);
  v.L = v.d.pow(2);
  for (const [k, node] of Object.entries(v)) node.label = k;
  return v;
}

const SVG = 'http://www.w3.org/2000/svg';
const el = (tag, attrs = {}, parent) => {
  const e = document.createElementNS(SVG, tag);
  for (const [k, val] of Object.entries(attrs)) e.setAttribute(k, val);
  parent?.append(e);
  return e;
};
const pos = (key) => ({ x: PAD + LAYOUT[key].col * COL_W, y: PAD + LAYOUT[key].row * ROW_H });

export function init(root) {
  const params = { ...DEFAULTS };
  const svg = root.querySelector('#ag-graph');
  const readout = root.querySelector('#ag-readout');
  const checkBody = root.querySelector('#ag-check tbody');
  let v;          // aktuelle Value-Knoten
  let order;      // Knoten in Rückwärts-Reihenfolge (nur Rechenknoten, keine Blätter)
  let cursor;     // wie viele davon schon abgearbeitet sind
  let active = null;

  // --- Schieberegler -----------------------------------------------------
  const sliderBox = root.querySelector('#ag-sliders');
  const inputs = {};
  for (const s of SLIDERS) {
    const wrap = document.createElement('label');
    wrap.className = 'slider';
    const name = document.createElement('span');
    name.textContent = s.label;
    const input = document.createElement('input');
    Object.assign(input, { type: 'range', min: s.min, max: s.max, step: 0.01, value: params[s.key] });
    const out = document.createElement('output');
    input.addEventListener('input', () => { params[s.key] = Number(input.value); rebuild(); });
    wrap.append(name, input, out);
    sliderBox.append(wrap);
    inputs[s.key] = { input, out };
  }
  const syncSliders = () => {
    for (const s of SLIDERS) {
      inputs[s.key].input.value = params[s.key];
      inputs[s.key].out.textContent = fmt(params[s.key], 2);
    }
  };

  // --- SVG-Gerüst einmal anlegen ---------------------------------------
  const cols = Math.max(...Object.values(LAYOUT).map((l) => l.col)) + 1;
  const rows = Math.max(...Object.values(LAYOUT).map((l) => l.row)) + 1;
  svg.setAttribute('viewBox', `0 0 ${PAD * 2 + (cols - 1) * COL_W + BOX_W} ${PAD * 2 + (rows - 1) * ROW_H + BOX_H}`);
  const edgeLayer = el('g', {}, svg);
  const nodeLayer = el('g', {}, svg);
  const edges = [];   // { parent, child, path }
  const nodeEls = {};
  const probe = forward(params);
  for (const [key, node] of Object.entries(probe)) {
    for (const child of node.children) {
      const a = pos(child.label), b = pos(key);
      const x1 = a.x + BOX_W, y1 = a.y + BOX_H / 2, x2 = b.x, y2 = b.y + BOX_H / 2;
      const mx = (x1 + x2) / 2;
      const path = el('path', { d: `M${x1},${y1} C${mx},${y1} ${mx},${y2} ${x2},${y2}`, class: 'g-edge' }, edgeLayer);
      edges.push({ parent: key, child: child.label, path });
    }
  }
  for (const key of Object.keys(LAYOUT)) {
    const { x, y } = pos(key);
    const isLeaf = probe[key].children.length === 0;
    const g = el('g', { class: `g-node${isLeaf ? ' is-leaf' : ''}`, transform: `translate(${x},${y})` }, nodeLayer);
    el('rect', { width: BOX_W, height: BOX_H, rx: 8 }, g);
    const name = el('text', { x: 10, y: 20, class: 'g-name' }, g);
    name.textContent = LAYOUT[key].name;
    const op = el('text', { x: BOX_W - 10, y: 20, class: 'g-op', 'text-anchor': 'end' }, g);
    op.textContent = LAYOUT[key].op;
    const data = el('text', { x: 10, y: 40, class: 'g-data' }, g);
    const grad = el('text', { x: 10, y: 58, class: 'g-grad' }, g);
    nodeEls[key] = { g, data, grad };
  }

  // --- Zeichnen --------------------------------------------------------
  function render() {
    const doneKeys = new Set(order.slice(0, cursor).map((n) => n.label));
    for (const key of Object.keys(LAYOUT)) {
      const node = v[key];
      const e = nodeEls[key];
      e.data.textContent = `Wert ${fmt(node.data)}`;
      const known = key === 'L' ? cursor > 0 : doneKeys.size > 0 && hasGrad(key, doneKeys);
      e.grad.textContent = known ? `Grad ${fmt(node.grad)}` : 'Grad ?';
      e.grad.classList.toggle('is-zero', !known);
      e.g.classList.toggle('is-active', key === active);
    }
    for (const ed of edges) ed.path.classList.toggle('is-active', ed.parent === active);
  }
  // Ein Knoten kennt seinen Gradienten, sobald alle seine Eltern ihn weitergegeben haben.
  function hasGrad(key, doneKeys) {
    const parents = edges.filter((e) => e.child === key).map((e) => e.parent);
    return parents.length > 0 && parents.every((p) => doneKeys.has(p));
  }

  function rebuild() {
    v = forward(params);
    order = buildTopo(v.L).reverse().filter((n) => n.children.length > 0);
    cursor = 0;
    active = null;
    syncSliders();
    readout.textContent = `Vorwärts gerechnet: o = ${fmt(v.o.data)}, Ziel y = ${fmt(params.y, 2)}, Loss L = ${fmt(v.L.data)}. Die Gradienten sind noch unbekannt.`;
    render();
    renderCheck();
  }

  // --- Ein Backward-Schritt: ein Knoten gibt seinen Gradienten weiter ----
  function explain(node) {
    const k = node.label;
    if (node.op === '+') return 'Bei „+“ ist die lokale Ableitung 1: Der Gradient wird unverändert an beide Summanden weitergereicht.';
    if (node.op === '*') return 'Bei „·“ ist die lokale Ableitung jeweils der andere Faktor.';
    if (node.op === 'tanh') return `tanh′(n) = 1 − tanh(n)² = 1 − ${fmt(v.o.data)}² = ${fmt(node.localGrads[0])}.`;
    if (k === 'L') return `(d²)′ = 2 · d = 2 · ${fmt(v.d.data)} = ${fmt(node.localGrads[0])}.`;
    return '';
  }

  function backwardStep() {
    if (cursor >= order.length) return false;
    const node = order[cursor];
    let intro = '';
    if (cursor === 0) {
      node.grad = 1;
      intro = 'Start: ∂L/∂L = 1. ';
    }
    const parts = node.children.map((c, j) => {
      const contrib = node.localGrads[j] * node.grad;
      c.grad += contrib;
      return `∂L/∂${LAYOUT[c.label].name} += ${fmt(node.localGrads[j])} · ${fmt(node.grad)} = ${fmt(contrib)}`;
    });
    cursor += 1;
    active = node.label;
    readout.textContent = `${intro}Knoten ${LAYOUT[node.label].name} = ${LAYOUT[node.label].op}. ${explain(node)} ${parts.join(';  ')}.`;
    if (cursor === order.length) {
      readout.textContent += ` Fertig: Jeder Knoten kennt seinen Gradienten. Ein negativer Gradient heißt: Diesen Wert zu erhöhen senkt L.`;
    }
    render();
    return true;
  }

  function backwardAll() {
    while (backwardStep()) { /* weiter */ }
    active = null;
    render();
  }

  root.querySelector('#ag-step').addEventListener('click', () => {
    if (!backwardStep()) { rebuild(); backwardStep(); }
  });
  root.querySelector('#ag-all').addEventListener('click', backwardAll);
  root.querySelector('#ag-reset').addEventListener('click', () => { Object.assign(params, DEFAULTS); rebuild(); });
  root.querySelector('#ag-learn').addEventListener('click', () => {
    if (cursor < order.length) backwardAll();
    const before = v.L.data;
    // Gradientenabstieg: nur die lernbaren Parameter bewegen, nicht die Eingaben.
    for (const key of ['w1', 'w2', 'b']) params[key] -= LEARN_RATE * v[key].grad;
    rebuild();
    backwardAll();
    readout.textContent = `Lernschritt: w ← w − ${fmt(LEARN_RATE, 1)} · Gradient. Der Loss ist von ${fmt(before)} auf ${fmt(v.L.data)} gefallen, o = ${fmt(v.o.data)} liegt näher am Ziel ${fmt(params.y, 2)}. Die neuen Gradienten sind schon eingetragen.`;
  });

  // --- Numerische Gegenprobe --------------------------------------------
  function renderCheck() {
    const auto = forward(params);
    auto.L.backward();
    const h = 1e-5;
    const rowsEls = ['x1', 'w1', 'x2', 'w2', 'b'].map((key) => {
      const plus = forward({ ...params, [key]: params[key] + h }).L.data;
      const minus = forward({ ...params, [key]: params[key] - h }).L.data;
      const tr = document.createElement('tr');
      for (const text of [LAYOUT[key].name, fmt(auto[key].grad, 6), fmt((plus - minus) / (2 * h), 6)]) {
        const td = document.createElement('td');
        td.textContent = text;
        tr.append(td);
      }
      return tr;
    });
    checkBody.replaceChildren(...rowsEls);
  }

  rebuild();
}
