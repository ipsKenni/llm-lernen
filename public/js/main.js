// =====================================================================
// main.js – Startet die Seite: Daten laden, Schritte umschalten
// =====================================================================

import { parseWords, buildTokenizer } from './tokenizer.js';
import { countBigrams, countsToProbs, averageNll } from './bigram-count.js';
import { DEFAULT_CONFIG } from './transformer.js';
import { renderTiles, fmt, int } from './viz.js';
import { renderAllCode } from './code-view.js';
import { createGptClient } from './worker-client.js';
import * as idee from './steps/idee.js';
import * as zaehlen from './steps/zaehlen.js';
import * as ableiten from './steps/ableiten.js';
import * as lernen from './steps/lernen.js';
import * as attention from './steps/attention.js';
import * as training from './steps/training.js';
import * as generieren from './steps/generieren.js';
import * as bpeStep from './steps/bpe-step.js';
import * as bauplan from './steps/bauplan.js';
import * as llamaTraining from './steps/llama-training.js';
import * as ollamaExport from './steps/ollama-export.js';
import * as hineinschauen from './steps/hineinschauen.js';
import * as finetuning from './steps/finetuning.js';
import { part2 } from './steps/teil2.js';

const STEPS = [
  { title: 'Die Idee', mod: idee },
  { title: 'Zählen', mod: zaehlen },
  { title: 'Ableiten', mod: ableiten },
  { title: 'Lernen', mod: lernen },
  { title: 'Aufmerksamkeit', mod: attention },
  { title: 'Trainieren', mod: training },
  { title: 'Generieren', mod: generieren },
  { title: 'Tokenizer', mod: bpeStep },
  { title: 'Bauplan', mod: bauplan },
  { title: 'Training', mod: llamaTraining },
  { title: 'Hineinschauen', mod: hineinschauen },
  { title: 'Fine-Tuning', mod: finetuning },
  { title: 'Ab nach Ollama', mod: ollamaExport },
  { title: 'Wie geht es weiter?', mod: null },
];
const FIRST_PART2 = 7;

const textCache = new Map();
function fetchText(url) {
  if (!textCache.has(url)) textCache.set(url, fetch(url).then((r) => r.text()));
  return textCache.get(url);
}

// Wortliste für einen Datensatz (Namen, Städte oder eigene Liste).
async function loadWords(dataset) {
  if (dataset.kind === 'cities') return parseWords(await fetchText('data/staedte.txt'));
  if (dataset.kind === 'custom') return parseWords(dataset.text.toLowerCase());
  return parseWords(await fetchText('data/names.txt'));
}

function renderTitle(tok) {
  const el = document.getElementById('title-tiles');
  el.replaceChildren(...'das kleinste llm'.split(' ').map((word) => {
    const w = document.createElement('span');
    w.className = 'title-word';
    renderTiles(w, [...word], [...word].map((c) => tok.stoi.get(c)));
    return w;
  }));
}

async function boot() {
  const words = await loadWords({ kind: 'names' });
  const tok = buildTokenizer(words);
  const N = countBigrams(words, tok);
  const P = countsToProbs(N);
  const ctx = {
    words, tok, N, P,
    countLoss: averageNll(words, tok, P),
    gpt: createGptClient(),
    loadWords,
  };
  renderTitle(tok);

  // Ein früher trainiertes Modell wiederherstellen, sonst frisch anlegen.
  if (!(await ctx.gpt.restore(loadWords))) ctx.gpt.init({ ...DEFAULT_CONFIG }, { kind: 'names', text: '' }, words);

  // Status des GPT-Trainings in der Navigation – sichtbar in jedem Schritt.
  const navStatus = document.getElementById('nav-status');
  ctx.gpt.on((msg, s) => {
    if (msg.type === 'progress' || msg.type === 'running' || msg.type === 'done' || msg.type === 'ready') {
      const recent = s.losses.slice(-50);
      const loss = recent.length ? recent.reduce((a, b) => a + b, 0) / recent.length : NaN;
      if (s.running) navStatus.textContent = `GPT trainiert: Schritt ${int(s.step)} von ${int(s.cfg.steps)}`;
      else if (s.step >= s.cfg.steps && s.step > 0) navStatus.textContent = `GPT trainiert, Loss ${fmt(loss, 2)}`;
      else navStatus.textContent = '';
    }
  });

  // Teil 2 startet seinen Worker erst, wenn man dort hinkommt.
  let part2Hooked = false;
  const hookPart2 = () => {
    if (part2Hooked) return;
    part2Hooked = true;
    const p2 = part2();
    p2.on(() => {
      const s = p2.state;
      if (s.running) navStatus.textContent = `Dein Llama trainiert: Schritt ${int(s.step)} von ${int(s.total)}`;
      else if (!ctx.gpt.state.running && s.setup && s.step > 0) navStatus.textContent = `Dein Llama: ${int(s.step)} Schritte trainiert`;
    });
  };

  const sections = [...document.querySelectorAll('.step')];
  const navLinks = [...document.querySelectorAll('[data-nav]')];
  const initialized = new Set();
  let current = -1;

  function show(i) {
    if (i === current) return;
    if (current >= 0) STEPS[current].mod?.hide?.();
    const first = current === -1;
    current = i;
    sections.forEach((s, k) => { s.hidden = k !== i; });
    navLinks.forEach((a, k) => {
      if (k === i) a.setAttribute('aria-current', 'step'); else a.removeAttribute('aria-current');
      if (k < i) a.classList.add('is-done');
    });
    const section = sections[i];
    const mod = STEPS[i].mod;
    if (i >= FIRST_PART2) hookPart2();
    if (mod && !initialized.has(i)) { initialized.add(i); mod.init(section, ctx); }
    mod?.show?.();
    renderAllCode(section);
    updatePager(i);
    document.title = i === 0 ? 'Das kleinste LLM' : `${STEPS[i].title} – Das kleinste LLM`;
    if (!first) {
      document.querySelector('.layout').scrollIntoView({ block: 'start' });
      document.getElementById('main').focus({ preventScroll: true });
    }
    navLinks[i].scrollIntoView({ block: 'nearest', inline: 'center' });
  }

  function updatePager(i) {
    const prev = document.getElementById('pager-prev');
    const next = document.getElementById('pager-next');
    const fill = (a, k, label) => {
      a.hidden = k < 0 || k >= STEPS.length;
      if (a.hidden) return;
      a.href = `#${k}`;
      const small = document.createElement('small');
      small.textContent = label;
      const span = document.createElement('span');
      span.textContent = `${k} ${STEPS[k].title}`;
      a.replaceChildren(small, span);
    };
    fill(prev, i - 1, 'Zurück');
    fill(next, i + 1, 'Weiter');
  }

  const fromHash = () => {
    const n = Number(location.hash.slice(1));
    return Number.isInteger(n) && n >= 0 && n < STEPS.length ? n : 0;
  };
  addEventListener('hashchange', () => show(fromHash()));
  show(fromHash());
  window.llmBooted = true;
}

boot().catch((e) => window.showBootError?.(`Start fehlgeschlagen: ${e.message}`));
