// Schritt 7 – Datensatz wählen und BPE-Tokenizer trainieren.
import { DATASETS } from '../llama/data.js';
import { bytesForDisplay } from '../llama/bpe.js';
import { renderTiles, fmt, int } from '../viz.js';
import { part2 } from './teil2.js';

const DESCRIPTIONS = {
  maerchen: '62 Märchen, 0,5 MB, deutsch',
  tinystories: '6.014 Kindergeschichten, 5 MB, englisch',
  namen: '32.033 Vornamen, 0,2 MB, je 10 als Liste',
  upload: 'deine eigene .txt-Datei',
};
const EXAMPLES = {
  maerchen: 'Es war einmal ein König, der hatte drei Töchter.',
  tinystories: 'Once upon a time, there was a little girl named Lily.',
  namen: 'emma\nolivia\nsven',
  upload: '',
};

export function init(root) {
  const p2 = part2();
  const S = p2.state;
  const $ = (sel) => root.querySelector(sel);
  const choices = $('#ds-choices');
  const vocabSel = $('#bpe-vocab');
  const runBtn = $('#bpe-run');
  const status = $('#bpe-status');
  const mergesList = $('#bpe-merges');
  const fileInput = $('#ds-file');
  const tryInput = $('#bpe-try');
  let selected = S.dataset;
  let uploadText = '';
  let shownMerges = 0;

  // --- Datensatz-Auswahl ----------------------------------------------
  const buttons = Object.entries(DATASETS).map(([key, ds]) => {
    const b = document.createElement('button');
    b.type = 'button';
    b.className = 'choice';
    b.setAttribute('role', 'radio');
    const title = document.createElement('strong');
    title.textContent = ds.label;
    const desc = document.createElement('span');
    desc.textContent = DESCRIPTIONS[key];
    b.append(title, desc);
    b.addEventListener('click', () => select(key));
    choices.append(b);
    return [key, b];
  });
  function select(key) {
    selected = key;
    for (const [k, b] of buttons) b.setAttribute('aria-checked', String(k === key));
    $('#ds-upload').hidden = key !== 'upload';
    vocabSel.value = String(DATASETS[key].vocab);
  }

  fileInput.addEventListener('change', async () => {
    const file = fileInput.files[0];
    if (!file) return;
    uploadText = await file.text();
    $('#ds-upload-info').textContent = `${file.name}: ${int(uploadText.length)} Zeichen geladen.`;
  });

  // --- Tokenizer trainieren ------------------------------------------
  // Die Merges kommen live aus dem Worker. Wir führen die Bytes jedes neuen
  // Tokens mit, um es lesbar anzuzeigen: neu = Bytes(a) + Bytes(b).
  let pieces = null;
  const resetPieces = () => { pieces = new Map(Array.from({ length: 256 }, (_, b) => [b, [b]])); };
  const show = (bytes) => bytesForDisplay(bytes);
  p2.trainer.on((m) => {
    if (m.type !== 'merges' || !pieces) return;
    for (const { a, b, id, count } of m.items) {
      pieces.set(id, [...pieces.get(a), ...pieces.get(b)]);
      if (shownMerges < 60) addMerge(show(pieces.get(a)), show(pieces.get(b)), count);
    }
  });
  function addMerge(left, right, count) {
    const li = document.createElement('li');
    const l = document.createElement('span');
    l.textContent = left;
    const r = document.createElement('span');
    r.textContent = right;
    li.append(l, ' + ', r);
    if (count) li.title = `${int(count)}-mal im Text`;
    mergesList.append(li);
    shownMerges += 1;
  }

  runBtn.addEventListener('click', async () => {
    if (selected === 'upload' && uploadText.length < 1000) {
      status.textContent = 'Wähle zuerst eine Textdatei mit mindestens 1.000 Zeichen.';
      return;
    }
    if (S.setup && S.step > 0 && !confirm('Ein neuer Tokenizer bedeutet ein neues Modell. Das bisherige Training wird beim nächsten Start überschrieben. Fortfahren?')) return;
    runBtn.disabled = true;
    mergesList.replaceChildren();
    shownMerges = 0;
    resetPieces();
    $('#bpe-merges-wrap').hidden = false;
    status.textContent = 'Lade Text und zähle Wortpaare …';
    const t0 = performance.now();
    try {
      await p2.tokenize(selected, Number(vocabSel.value), selected === 'upload' ? uploadText : '');
      status.textContent = `Fertig in ${fmt((performance.now() - t0) / 1000, 1)} s. Weiter geht es mit dem Bauplan in Schritt 8.`;
      tryInput.value = EXAMPLES[selected] || uploadText.slice(0, 80).split('\n')[0];
      render();
    } catch (e) {
      status.textContent = `Das hat nicht geklappt: ${e.message}`;
    } finally {
      runBtn.disabled = false;
    }
  });

  // --- Ergebnis anzeigen -----------------------------------------------
  function render() {
    const info = S.tokInfo, tok = S.tokenizer;
    if (!info || !tok) return;
    $('#bpe-stats').hidden = false;
    $('#bpe-try-panel').hidden = false;
    $('#bpe-merges-wrap').hidden = false;
    $('#bpe-s-vocab').textContent = int(tok.vocabSize);
    $('#bpe-s-docs').textContent = int(info.docs);
    $('#bpe-s-tokens').textContent = int(info.trainTokens);
    $('#bpe-s-ratio').textContent = fmt(info.chars / (info.trainTokens + info.valTokens), 2);
    // Nach einem Neuladen die Merges aus dem gespeicherten Tokenizer zeigen
    if (shownMerges === 0) tok.merges.slice(0, 60).forEach(([a, b]) => addMerge(tok.display(a), tok.display(b), 0));
    if (!tryInput.value) tryInput.value = EXAMPLES[S.dataset] || '';
    renderTry();
  }

  function renderTry() {
    const tok = S.tokenizer;
    if (!tok) return;
    const ids = tok.encode(tryInput.value);
    renderTiles($('#bpe-try-out'), ids.map((id) => tok.display(id)), ids);
    const chars = [...tryInput.value].length;
    $('#bpe-try-info').textContent = ids.length
      ? `${int(chars)} Zeichen werden zu ${int(ids.length)} Tokens. In Teil 1 wären es ${int(chars)} gewesen.`
      : '';
  }
  tryInput.addEventListener('input', renderTry);

  p2.on((what) => {
    if (what === 'restored') {
      select(S.dataset);
      vocabSel.value = String(S.tokenizer.vocabSize);
      status.textContent = `Gespeichertes Training gefunden: ${DATASETS[S.dataset].label}, Tokenizer mit ${int(S.tokenizer.vocabSize)} Tokens, ${int(S.step)} Trainingsschritte. Du kannst direkt in Schritt 9 weitermachen.`;
      render();
    }
  });

  select(S.dataset);
  if (S.tokenizer) render();
}
