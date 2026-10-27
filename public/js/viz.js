// =====================================================================
// viz.js – Zeichnen: Heatmaps, Loss-Kurven, Wahrscheinlichkeitsbalken
// =====================================================================
// Reines Canvas/DOM, keine Bibliotheken. Farben kommen aus CSS-Variablen,
// damit hell/dunkel automatisch passt.

export const fmt = (x, digits = 3) =>
  Number.isFinite(x) ? x.toLocaleString('de-DE', { minimumFractionDigits: digits, maximumFractionDigits: digits }) : '–';
export const pct = (p, digits = 1) => `${(p * 100).toLocaleString('de-DE', { minimumFractionDigits: digits, maximumFractionDigits: digits })} %`;
export const int = (n) => Math.round(n).toLocaleString('de-DE');
// Token für die Anzeige: '.' heißt bei uns Anfang/Ende.
export const showChar = (c) => (c === ' ' ? '␣' : c);

export function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

// ---------------------------------------------------------------------
// Farbrampe (sequenziell, ein Farbton): 0 = Fläche, 1 = kräftigstes Blau
// ---------------------------------------------------------------------
let rampStops = null;
function hexToRgb(hex) {
  const n = parseInt(hex.replace('#', ''), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}
function loadRamp() {
  rampStops = [0, 1, 2, 3, 4].map((i) => hexToRgb(cssVar(`--ramp-${i}`)));
}
export function rampColor(t) {
  if (!rampStops) loadRamp();
  t = Math.min(1, Math.max(0, t));
  const pos = t * (rampStops.length - 1);
  const i = Math.min(rampStops.length - 2, Math.floor(pos));
  const f = pos - i;
  const a = rampStops[i], b = rampStops[i + 1];
  const c = a.map((v, k) => Math.round(v + (b[k] - v) * f));
  return `rgb(${c[0]},${c[1]},${c[2]})`;
}

// Alles neu zeichnen, wenn das Farbschema wechselt.
const redrawers = new Set();
matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => {
  loadRamp();
  redrawers.forEach((fn) => fn());
});

// Canvas scharf auf HiDPI-Bildschirmen: interne Auflösung × devicePixelRatio.
// `heightFor(w)` berechnet die Höhe aus der verfügbaren Breite.
function setupCanvas(canvas, heightFor) {
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth || 300;
  const h = heightFor(w);
  canvas.style.height = `${h}px`;
  if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
  }
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  return { ctx, w, h };
}

function onResize(el, fn) {
  let last = el.clientWidth;
  new ResizeObserver(() => {
    if (el.clientWidth !== last && el.clientWidth > 0) { last = el.clientWidth; fn(); }
  }).observe(el);
  redrawers.add(fn);
}

// ---------------------------------------------------------------------
// Tooltip (eins für die ganze Seite). Inhalt immer per textContent.
// ---------------------------------------------------------------------
const tip = document.createElement('div');
tip.className = 'tooltip';
tip.setAttribute('role', 'status');
document.body.append(tip);

export function showTooltip(event, lines) {
  tip.replaceChildren(...lines.map(([strong, rest], i) => {
    const row = document.createElement('div');
    if (i === 0) row.className = 'tooltip-head';
    const s = document.createElement('strong');
    s.textContent = strong;
    row.append(s);
    if (rest) row.append(document.createTextNode(` ${rest}`));
    return row;
  }));
  tip.classList.add('visible');
  const pad = 14;
  const r = tip.getBoundingClientRect();
  let x = event.clientX + pad;
  let y = event.clientY + pad;
  if (x + r.width > innerWidth - 8) x = event.clientX - r.width - pad;
  if (y + r.height > innerHeight - 8) y = event.clientY - r.height - pad;
  tip.style.transform = `translate(${Math.max(8, x)}px, ${Math.max(8, y)}px)`;
}
export function hideTooltip() { tip.classList.remove('visible'); }

// ---------------------------------------------------------------------
// Heatmap: Matrix mit Zeilen-/Spaltenbeschriftung.
// rows = "aktuelles Zeichen", cols = "nächstes Zeichen"
// ---------------------------------------------------------------------
export function createHeatmap(canvas, { rowLabels, colLabels = rowLabels, describe, onSelect, sqrt = true, fixedMax = null, rowTitle = '', colTitle = '' }) {
  let matrix = null;
  let selectedRow = -1;
  let hover = null;
  let geom = null;

  function draw() {
    if (!matrix) return;
    const nR = rowLabels.length, nC = colLabels.length;
    const gutter = 20;
    const { ctx, w } = setupCanvas(canvas, (width) => gutter + ((width - gutter) / nC) * nR);
    const cell = (w - gutter) / nC;
    geom = { gutter, cell, nR, nC };
    let max = fixedMax ?? 0;
    if (fixedMax === null) for (const row of matrix) for (const v of row) if (v > max) max = v;
    const norm = (v) => (max > 0 ? (sqrt ? Math.sqrt(v / max) : v / max) : 0);
    const gap = cell > 10 ? 1 : 0;
    for (let r = 0; r < nR; r++) {
      for (let c = 0; c < nC; c++) {
        ctx.fillStyle = rampColor(norm(matrix[r][c]));
        ctx.fillRect(gutter + c * cell, gutter + r * cell, cell - gap, cell - gap);
      }
    }
    // Beschriftung
    const ink = cssVar('--muted');
    const strong = cssVar('--ink');
    ctx.font = `${Math.min(13, Math.max(8, cell * 0.62))}px ${cssVar('--font-mono')}`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    for (let c = 0; c < nC; c++) {
      ctx.fillStyle = hover && hover.c === c ? strong : ink;
      ctx.fillText(showChar(colLabels[c]), gutter + c * cell + cell / 2, gutter / 2);
    }
    for (let r = 0; r < nR; r++) {
      ctx.fillStyle = (hover && hover.r === r) || r === selectedRow ? strong : ink;
      ctx.fillText(showChar(rowLabels[r]), gutter / 2, gutter + r * cell + cell / 2);
    }
    // Auswahl / Hover hervorheben
    ctx.lineWidth = 2;
    if (selectedRow >= 0) {
      ctx.strokeStyle = strong;
      ctx.strokeRect(gutter - 1, gutter + selectedRow * cell - 1, nC * cell + 1, cell + 1);
    }
    if (hover) {
      ctx.strokeStyle = strong;
      ctx.strokeRect(gutter + hover.c * cell - 1, gutter + hover.r * cell - 1, cell + 1, cell + 1);
    }
  }

  function cellAt(e) {
    if (!geom) return null;
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left - geom.gutter;
    const y = e.clientY - rect.top - geom.gutter;
    const c = Math.floor(x / geom.cell), r = Math.floor(y / geom.cell);
    if (r < 0 || c < 0 || r >= geom.nR || c >= geom.nC) return null;
    return { r, c };
  }

  canvas.addEventListener('pointermove', (e) => {
    const h = cellAt(e);
    if (!h) { hover = null; hideTooltip(); draw(); return; }
    if (!hover || hover.r !== h.r || hover.c !== h.c) { hover = h; draw(); }
    showTooltip(e, describe(h.r, h.c, matrix));
  });
  canvas.addEventListener('pointerleave', () => { hover = null; hideTooltip(); draw(); });
  if (onSelect) {
    canvas.addEventListener('click', (e) => {
      const h = cellAt(e);
      if (h) { selectedRow = h.r; draw(); onSelect(h.r, h.c); }
    });
  }
  if (rowTitle || colTitle) canvas.setAttribute('aria-label', `${rowTitle} (Zeilen) gegen ${colTitle} (Spalten)`);
  onResize(canvas, draw);

  return {
    update(m, labels) { matrix = m; if (labels) { rowLabels = labels; colLabels = labels; } draw(); },
    select(r) { selectedRow = r; draw(); },
  };
}

// ---------------------------------------------------------------------
// Liniendiagramm für den Loss: eine Achse, Referenzlinien, Fadenkreuz.
// ---------------------------------------------------------------------
export function createLineChart(canvas, { references = [], xLabel = 'Schritt', height = 260 }) {
  let series = [];
  // Tabellen-Alternative: <details class="chart-table"> direkt nach dem Canvas
  const tableBox = canvas.parentElement?.querySelector(`details.chart-table[data-for="${canvas.id}"]`);
  const renderTable = () => {
    if (!tableBox?.open) return;
    const body = series.filter((s) => (s.values || s.points || []).length);
    const n = Math.max(0, ...body.map((s) => (s.values ? s.values.length : s.points.at(-1)[0])));
    const every = Math.max(1, Math.ceil(n / 30));
    const head = document.createElement('tr');
    for (const t of [xLabel, ...body.map((s) => s.name)]) { const th = document.createElement('th'); th.scope = 'col'; th.textContent = t; head.append(th); }
    const rows = [];
    for (let x = every; x <= n; x += every) {
      const tr = document.createElement('tr');
      const cells = [int(x), ...body.map((s) => {
        if (s.values) return s.values[x - 1] !== undefined ? fmt(s.values[x - 1]) : '–';
        const p = s.points.filter(([px]) => px <= x).at(-1);
        return p ? fmt(p[1]) : '–';
      })];
      for (const c of cells) { const td = document.createElement('td'); td.textContent = c; tr.append(td); }
      rows.push(tr);
    }
    const table = document.createElement('table');
    table.className = 'data-table';
    const thead = document.createElement('thead');
    thead.append(head);
    const tbody = document.createElement('tbody');
    tbody.append(...rows);
    table.append(thead, tbody);
    tableBox.querySelector('.table-wrap').replaceChildren(rows.length ? table : document.createTextNode('Noch keine Werte.'));
  };
  tableBox?.addEventListener('toggle', renderTable);
  let xMax = 1;
  let hoverX = null;
  let geom = null;

  function yRange() {
    const vals = [];
    for (const s of series) {
      if (s.scale === false) continue;
      for (const v of s.values || []) if (Number.isFinite(v)) vals.push(v);
      for (const [, v] of s.points || []) if (Number.isFinite(v)) vals.push(v);
    }
    for (const r of references) vals.push(r.value);
    if (vals.length === 0) return [0, 1];
    let lo = Math.min(...vals), hi = Math.max(...vals);
    if (hi - lo < 0.2) { hi += 0.1; lo -= 0.1; }
    return [Math.floor(lo * 10) / 10 - 0.05, Math.ceil(hi * 10) / 10 + 0.1];
  }

  function draw() {
    const { ctx, w, h } = setupCanvas(canvas, () => height);
    const m = { l: 44, r: 16, t: 12, b: 46 };
    const pw = w - m.l - m.r, ph = h - m.t - m.b;
    const [y0, y1] = yRange();
    const X = (i) => m.l + (i / Math.max(1, xMax)) * pw;
    const Y = (v) => m.t + (1 - (v - y0) / (y1 - y0)) * ph;
    geom = { m, pw, X };

    const muted = cssVar('--muted'), grid = cssVar('--grid'), ink = cssVar('--ink');
    ctx.font = `12px ${cssVar('--font-sans')}`;
    ctx.lineWidth = 1;
    // Gitter + y-Achse
    const step = niceStep((y1 - y0) / 4);
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    for (let v = Math.ceil(y0 / step) * step; v <= y1 + 1e-9; v += step) {
      ctx.strokeStyle = grid;
      ctx.beginPath(); ctx.moveTo(m.l, Y(v)); ctx.lineTo(w - m.r, Y(v)); ctx.stroke();
      ctx.fillStyle = muted;
      ctx.fillText(fmt(v, 1), m.l - 8, Y(v));
    }
    // x-Achse
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    const xs = niceStep(xMax / 5);
    for (let i = 0; i <= xMax + 1e-9; i += xs) ctx.fillText(int(i), X(i), h - m.b + 8);
    ctx.textAlign = 'center';
    ctx.fillText(xLabel, m.l + pw / 2, h - 16);

    ctx.save();
    ctx.beginPath(); ctx.rect(m.l, m.t, pw, ph); ctx.clip();
    // Referenzlinien (gestrichelt, mit Beschriftung)
    for (const r of references) {
      ctx.strokeStyle = muted;
      ctx.setLineDash([4, 4]);
      ctx.beginPath(); ctx.moveTo(m.l, Y(r.value)); ctx.lineTo(w - m.r, Y(r.value)); ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = muted;
      ctx.textAlign = 'right';
      ctx.textBaseline = 'bottom';
      ctx.fillText(`${r.label} ${fmt(r.value, 2)}`, w - m.r - 4, Y(r.value) - 3);
    }
    // Serien
    for (const s of series) {
      const pts = s.points || (s.values || []).map((v, i) => [i + 1, v]);
      if (pts.length < 1) continue;
      ctx.strokeStyle = cssVar(s.color || '--fwd');
      ctx.fillStyle = ctx.strokeStyle;
      ctx.globalAlpha = s.alpha ?? 1;
      ctx.lineWidth = s.width ?? 2;
      ctx.lineJoin = 'round';
      ctx.setLineDash(s.dash || []);
      ctx.beginPath();
      pts.forEach(([x, v], i) => (i === 0 ? ctx.moveTo(X(x), Y(v)) : ctx.lineTo(X(x), Y(v))));
      ctx.stroke();
      ctx.setLineDash([]);
      if (s.points) for (const [x, v] of pts) { ctx.beginPath(); ctx.arc(X(x), Y(v), 4, 0, 2 * Math.PI); ctx.fill(); }
      ctx.globalAlpha = 1;
    }
    ctx.restore();
    // Fadenkreuz
    if (hoverX !== null && series[0]?.values?.length) {
      ctx.strokeStyle = ink;
      ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(X(hoverX), m.t); ctx.lineTo(X(hoverX), m.t + ph); ctx.stroke();
    }
  }

  canvas.addEventListener('pointermove', (e) => {
    if (!geom || !series[0]?.values?.length) return;
    const rect = canvas.getBoundingClientRect();
    const n = series[0].values.length;
    const i = Math.round(((e.clientX - rect.left - geom.m.l) / geom.pw) * Math.max(1, xMax));
    hoverX = Math.min(n, Math.max(1, i));
    draw();
    showTooltip(e, [
      [`Schritt ${int(hoverX)}`],
      ...series.map((s) => {
        if (s.points) {
          const near = s.points.filter(([x]) => x <= hoverX).at(-1);
          return near ? [fmt(near[1]), `${s.name} (bei Schritt ${int(near[0])})`] : null;
        }
        return s.values[hoverX - 1] !== undefined ? [fmt(s.values[hoverX - 1]), s.name] : null;
      }).filter(Boolean),
    ]);
  });
  canvas.addEventListener('pointerleave', () => { hoverX = null; hideTooltip(); draw(); });
  onResize(canvas, draw);

  return {
    update(newSeries, newXMax) { series = newSeries; xMax = newXMax; draw(); renderTable(); },
    setReferences(refs) { references = refs; draw(); },
  };
}

function niceStep(raw) {
  const p = 10 ** Math.floor(Math.log10(raw || 1));
  const f = raw / p;
  return (f <= 1 ? 1 : f <= 2 ? 2 : f <= 5 ? 5 : 10) * p;
}

// ---------------------------------------------------------------------
// Wahrscheinlichkeitsbalken: die Top-k Kandidaten fürs nächste Token.
// ---------------------------------------------------------------------
export function renderProbBars(container, probs, itos, { topK = 8, highlight = -1, endId = 0, endLabel = '.', endTitle = 'Ende des Namens' } = {}) {
  const order = probs.map((p, i) => [p, i]).sort((a, b) => b[0] - a[0]).slice(0, topK);
  if (highlight >= 0 && !order.some(([, i]) => i === highlight)) order[order.length - 1] = [probs[highlight], highlight];
  const max = order[0][0];
  container.replaceChildren(...order.map(([p, i]) => {
    const row = document.createElement('div');
    row.className = 'pbar' + (i === highlight ? ' is-picked' : '');
    const label = document.createElement('span');
    label.className = 'tok';
    label.textContent = i === endId ? endLabel : showChar(itos[i]);
    if (i === endId) label.title = endTitle;
    const track = document.createElement('span');
    track.className = 'pbar-track';
    const fill = document.createElement('span');
    fill.className = 'pbar-fill';
    fill.style.width = `${Math.max(1, (p / max) * 100)}%`;
    track.append(fill);
    const val = document.createElement('span');
    val.className = 'pbar-val';
    val.textContent = pct(p);
    row.append(label, track, val);
    return row;
  }));
}

// ---------------------------------------------------------------------
// Token-Kacheln: Zeichen oben, ID darunter.
// ---------------------------------------------------------------------
export function renderTiles(container, chars, ids, { activeIndex = -1 } = {}) {
  container.replaceChildren(...chars.map((c, i) => {
    const t = document.createElement('span');
    t.className = 'tile' + (i === activeIndex ? ' is-active' : '') + (ids[i] === undefined ? ' is-unknown' : '');
    const ch = document.createElement('span');
    ch.className = 'tile-char';
    ch.textContent = showChar(c);
    const id = document.createElement('span');
    id.className = 'tile-id';
    id.textContent = ids[i] === undefined ? '?' : String(ids[i]);
    t.append(ch, id);
    return t;
  }));
}

// ---------------------------------------------------------------------
// Streudiagramm (für die Token-Karte). points: [{ x, y, label, merged }]
// Die Achsen einer PCA haben keine Einheit – deshalb ohne Achsenbeschriftung.
// ---------------------------------------------------------------------
export function createScatter(canvas, { height = 420, labelIf = () => false } = {}) {
  let points = [];
  let placed = [];
  let hover = -1;

  function draw() {
    const { ctx, w, h } = setupCanvas(canvas, () => height);
    if (!points.length) return;
    const pad = 28;
    const xs = points.map((p) => p.x), ys = points.map((p) => p.y);
    const [x0, x1] = [Math.min(...xs), Math.max(...xs)], [y0, y1] = [Math.min(...ys), Math.max(...ys)];
    const X = (v) => pad + ((v - x0) / (x1 - x0 || 1)) * (w - 2 * pad);
    const Y = (v) => h - pad - ((v - y0) / (y1 - y0 || 1)) * (h - 2 * pad);
    placed = points.map((p) => [X(p.x), Y(p.y)]);
    const fwd = cssVar('--fwd'), muted = cssVar('--muted'), ink = cssVar('--ink'), surface = cssVar('--surface');
    ctx.font = `12px ${cssVar('--font-mono')}`;
    ctx.textBaseline = 'middle';
    points.forEach((p, i) => {
      const [px, py] = placed[i];
      ctx.fillStyle = p.merged ? fwd : muted;
      ctx.beginPath(); ctx.arc(px, py, i === hover ? 6 : 3.5, 0, 2 * Math.PI); ctx.fill();
    });
    // Beschriftungen mit kleinem Hintergrund, damit sie lesbar bleiben
    points.forEach((p, i) => {
      if (!labelIf(p, i) && i !== hover) return;
      const [px, py] = placed[i];
      const tw = ctx.measureText(p.label).width;
      ctx.fillStyle = surface;
      ctx.globalAlpha = 0.8;
      ctx.fillRect(px + 5, py - 7, tw + 4, 14);
      ctx.globalAlpha = 1;
      ctx.fillStyle = i === hover ? ink : p.merged ? ink : muted;
      ctx.fillText(p.label, px + 7, py);
    });
  }

  canvas.addEventListener('pointermove', (e) => {
    const r = canvas.getBoundingClientRect();
    const mx = e.clientX - r.left, my = e.clientY - r.top;
    let best = -1, bestD = 14 * 14;
    placed.forEach(([px, py], i) => { const dd = (px - mx) ** 2 + (py - my) ** 2; if (dd < bestD) { bestD = dd; best = i; } });
    if (best !== hover) { hover = best; draw(); }
    if (best >= 0) showTooltip(e, [[points[best].label], [points[best].merged ? 'gelerntes Token' : 'einzelnes Zeichen', `(ID ${points[best].id})`]]);
    else hideTooltip();
  });
  canvas.addEventListener('pointerleave', () => { hover = -1; hideTooltip(); draw(); });
  onResize(canvas, draw);
  return { update(p) { points = p; hover = -1; draw(); } };
}
