// Schritt 10 – Live-Übersicht: was das Modell beim Training lernt.
// Jede Momentaufnahme (snapshot) kommt aus dem Worker; diese Datei zeichnet sie.
import { createHeatmap, createLineChart, createScatter, drawSparkline, renderProbBars, fmt, int, pct } from '../viz.js';
import { movingAverage } from './lernen.js';
import { part2 } from './teil2.js';

const duration = (sec) => (!Number.isFinite(sec) || sec <= 0 ? '–' : sec < 90 ? `${Math.round(sec)} s` : sec < 5400 ? `${Math.round(sec / 60)} min` : `${fmt(sec / 3600, 1)} h`);

// Höchstens n Einträge, gleichmäßig verteilt, erster und letzter immer dabei.
function spread(list, n) {
  if (list.length <= n) return list;
  return Array.from({ length: n }, (_, i) => list[Math.round((i * (list.length - 1)) / (n - 1))]);
}

export function initLive(root) {
  const p2 = part2();
  const S = p2.state;
  const $ = (sel) => root.querySelector(sel);
  const chart = createLineChart($('#live-chart'), { references: [], height: 200 });
  const map = createScatter($('#live-map'), {
    height: 420,
    labelIf: (p, i) => (p.merged ? i < 94 + 60 : /[A-Za-z]/.test(p.label)),
  });
  let heatmaps = [];
  let shownSnapshots = -1;
  let labels = [];

  // --- Start/Pause direkt von hier ---------------------------------------
  $('#live-run').addEventListener('click', () => {
    if (S.running) { p2.pause(); return; }
    const total = S.step >= S.total ? S.step + 1000 : S.total;
    p2.train(total);
  });
  $('#live-refresh').addEventListener('click', async () => {
    $('#live-note').textContent = 'Nehme eine Momentaufnahme auf …';
    try {
      const { snapshot } = await p2.trainer.snapshot();
      S.snapshots.push(snapshot);
      $('#live-note').textContent = '';
      render(true);
    } catch (e) {
      $('#live-note').textContent = e.message;
    }
  });

  function renderStatus() {
    const pctDone = S.total ? Math.min(100, (S.step / S.total) * 100) : 0;
    $('#live-bar').style.width = `${pctDone}%`;
    $('#live-progress').setAttribute('aria-valuenow', String(Math.round(pctDone)));
    $('#live-step').textContent = `${int(S.step)} / ${int(S.total)}`;
    const recent = S.losses.slice(-50);
    $('#live-loss').textContent = recent.length ? fmt(recent.reduce((a, b) => a + b, 0) / recent.length) : '–';
    $('#live-val').textContent = S.evals.length ? fmt(S.evals.at(-1).valLoss) : '–';
    $('#live-speed').textContent = S.running && S.tokensPerSec ? int(S.tokensPerSec) : '–';
    $('#live-eta').textContent = S.running && S.stepsPerSec ? duration((S.total - S.step) / S.stepsPerSec) : '–';
    const finished = S.step >= S.total;
    $('#live-run').textContent = S.running ? 'Pause' : S.step === 0 ? 'Training starten' : finished ? '1.000 weitere Schritte' : 'Weiter trainieren';
    $('#live-refresh').disabled = S.running;
    chart.setReferences([{ value: Math.log(S.tokenizer.vocabSize), label: 'Raten' }]);
    chart.update([
      { name: 'Train-Loss geglättet', values: movingAverage(S.losses, 50) },
      { name: 'Val-Loss', points: S.evals.map((e) => [e.step, e.valLoss]), color: '--ink', dash: [5, 4] },
    ], Math.max(S.total, S.losses.length, 1));
  }

  function renderTimeline(snaps) {
    const items = spread(snaps, 10).map((snap) => {
      const li = document.createElement('li');
      const when = document.createElement('span');
      when.className = 'when';
      when.textContent = `Schritt ${int(snap.step)}`;
      const what = document.createElement('span');
      what.className = 'what';
      const b = document.createElement('b');
      b.textContent = snap.prompt;
      what.append(b, snap.greedy || ' (sofort Ende)');
      li.append(when, what);
      return li;
    });
    $('#live-timeline').replaceChildren(...items);
  }

  function renderProbes(snaps) {
    const box = $('#live-probes');
    const current = snaps.filter((s) => s.probeSet === S.probeSet);
    if (!S.probes.length) {
      box.replaceChildren();
      $('#live-probes-hint').textContent = 'In diesem Text gibt es keine wiederkehrenden Wendungen aus mehreren Wörtern (zum Beispiel bei Namenslisten). Achte stattdessen auf den Zeitraffer.';
      return;
    }
    box.replaceChildren(...S.probes.map((probe, i) => {
      const li = document.createElement('li');
      const phrase = document.createElement('span');
      phrase.className = 'probe-phrase';
      const pre = document.createElement('span');
      pre.textContent = `${probe.prefix} `;
      const target = document.createElement('b');
      target.textContent = `→${probe.target}`;
      phrase.append(pre, target);
      const history = current.map((s) => s.probes[i]);
      const now = history.at(-1) ?? 0;
      const bar = document.createElement('span');
      bar.className = 'pbar-track';
      const fill = document.createElement('span');
      fill.className = 'pbar-fill';
      fill.style.width = `${Math.max(1, now * 100)}%`;
      bar.append(fill);
      const spark = document.createElement('canvas');
      spark.className = 'sparkline';
      spark.setAttribute('aria-hidden', 'true');
      const val = document.createElement('span');
      val.className = 'pbar-val';
      val.textContent = pct(now, 0);
      li.append(phrase, bar, val, spark);
      // Eigener Maßstab je Zeile, damit auch kleine Anstiege sichtbar werden
      requestAnimationFrame(() => drawSparkline(spark, history, { max: Math.max(0.02, ...history) }));
      return li;
    }));
  }

  function renderAttention(snap) {
    const box = $('#live-attn');
    const heads = snap.attn.length;
    if (heatmaps.length !== heads) {
      box.replaceChildren();
      heatmaps = Array.from({ length: heads }, (_, h) => {
        const fig = document.createElement('figure');
        const cap = document.createElement('figcaption');
        cap.textContent = `Kopf ${h + 1}`;
        const canvas = document.createElement('canvas');
        canvas.className = 'heatmap';
        canvas.setAttribute('role', 'img');
        canvas.setAttribute('aria-label', `Attention von Kopf ${h + 1}`);
        fig.append(cap, canvas);
        box.append(fig);
        let cur = { m: [], l: [] };
        const hm = createHeatmap(canvas, {
          rowLabels: [], sqrt: false, fixedMax: 1,
          describe: (r, c) => (c > r ? [[`Position ${r} → ${c}`], ['maskiert']] : [[`„${cur.l[r]}“ schaut auf „${cur.l[c]}“`], [pct(cur.m[r][c]), 'Aufmerksamkeit']]),
        });
        return { set(m, l) { cur = { m, l }; hm.update(m, l); } };
      });
    }
    heatmaps.forEach((hm, h) => hm.set(snap.attn[h].map((row) => snap.labels.map((_, t) => row[t] ?? 0)), snap.labels));
  }

  function renderMap(snap, animate) {
    const tok = S.tokenizer;
    const points = snap.map.ids.map((id, i) => ({ id, label: tok.display(id), x: snap.map.xs[i], y: snap.map.ys[i], merged: id >= 256 }));
    map.update(points, { animate });
  }

  function render(newSnapshot = false) {
    const ready = !!S.setup && !!S.tokenizer;
    $('#in-none').hidden = ready;
    $('#in-main').hidden = !ready;
    if (!ready) return;
    renderStatus();
    const snaps = S.snapshots;
    $('#live-probe-text').textContent = S.probeText.replaceAll('\n', '⏎');
    if (!snaps.length) {
      $('#live-timeline').replaceChildren();
      $('#live-note').textContent = 'Noch keine Momentaufnahme. Starte das Training oder klick „Jetzt aktualisieren“.';
      return;
    }
    if ($('#live-note').textContent.startsWith('Noch keine')) $('#live-note').textContent = '';
    if (snaps.length === shownSnapshots && !newSnapshot) return;   // nichts Neues zu zeichnen
    const animate = shownSnapshots > 0 && snaps.length > shownSnapshots;
    shownSnapshots = snaps.length;
    const last = snaps.at(-1);
    if (labels.length !== S.tokenizer.vocabSize) labels = Array.from({ length: S.tokenizer.vocabSize }, (_, i) => S.tokenizer.display(i));
    renderTimeline(snaps);
    renderProbes(snaps);
    const probs = new Array(S.tokenizer.vocabSize).fill(0);
    for (const { id, p } of last.top) probs[id] = p;
    renderProbBars($('#live-next'), probs, labels, { topK: 8, endId: S.tokenizer.eot, endLabel: '⟨Ende⟩', endTitle: 'Ende des Textes' });
    renderAttention(last);
    renderMap(last, animate);
  }

  let pending = false;
  p2.on((what) => {
    if (what === 'snapshot' || what === 'restored' || what === 'setup' || what === 'finetune' || what === 'tokenized') { shownSnapshots = what === 'snapshot' ? shownSnapshots : -1; render(what === 'snapshot'); return; }
    if (!pending) { pending = true; requestAnimationFrame(() => { pending = false; if (!$('#in-main').hidden) renderStatus(); }); }
  });
  render();
  return { render: () => render(true) };
}
