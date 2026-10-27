// Schritt 11 – Fine-Tuning: ein fertiges Modell mit neuem Text weitertrainieren.
import { DATASETS } from '../llama/data.js';
import { createLineChart, fmt, int } from '../viz.js';
import { movingAverage } from './lernen.js';
import { part2 } from './teil2.js';

export function init(root) {
  const p2 = part2();
  const S = p2.state;
  const $ = (sel) => root.querySelector(sel);
  const choices = $('#ft-choices');
  const runBtn = $('#ft-run');
  const status = $('#ft-status');
  const chart = createLineChart($('#ft-chart'), { references: [], height: 220 });
  let selected = 'upload';
  let uploadText = '';
  let busy = false;

  // Datensatz-Auswahl (alle außer dem, auf dem das Modell gerade steht)
  const buttons = Object.entries(DATASETS).map(([key, ds]) => {
    const b = document.createElement('button');
    b.type = 'button';
    b.className = 'choice';
    b.setAttribute('role', 'radio');
    const t = document.createElement('strong');
    t.textContent = ds.label;
    b.append(t);
    b.addEventListener('click', () => { selected = key; render(); });
    choices.append(b);
    return [key, b];
  });
  $('#ft-file').addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    uploadText = await file.text();
    $('#ft-file-info').textContent = `${file.name}: ${int(uploadText.length)} Zeichen geladen.`;
  });

  runBtn.addEventListener('click', async () => {
    if (S.running) { p2.pause(); return; }
    if (selected === 'upload' && uploadText.length < 1000) { status.textContent = 'Wähle zuerst eine Textdatei mit mindestens 1.000 Zeichen.'; return; }
    busy = true;
    render();
    try {
      // Erst beim ersten Klick vorbereiten; danach nur weitertrainieren.
      if (!S.ft || S.ft.to !== selected) {
        status.textContent = 'Sichere das Basismodell und tokenisiere den neuen Text …';
        await p2.finetune(selected, Number($('#ft-steps').value), selected === 'upload' ? uploadText : '');
      }
      status.textContent = '';
      p2.train(S.step >= S.total ? S.step + Number($('#ft-steps').value) : S.total);
    } catch (e) {
      status.textContent = `Das hat nicht geklappt: ${e.message}`;
    } finally {
      busy = false;
      render();
    }
  });
  $('#ft-revert').addEventListener('click', async () => {
    if (!confirm('Das Fine-Tuning wird verworfen und das gesicherte Basismodell wiederhergestellt. Fortfahren?')) return;
    p2.pause();
    await p2.revert();
    status.textContent = 'Basismodell wiederhergestellt.';
  });

  function sampleCell(sample) {
    const td = document.createElement('td');
    if (!sample) { td.textContent = '–'; return td; }
    const b = document.createElement('b');
    b.textContent = sample.prompt;
    td.append(b, sample.text);
    td.className = 'ft-sample';
    return td;
  }

  function render() {
    const ready = !!S.setup && S.step > 0;
    $('#ft-none').hidden = ready;
    $('#ft-main').hidden = !ready;
    if (!ready) return;
    for (const [k, b] of buttons) {
      b.setAttribute('aria-checked', String(k === selected));
      b.disabled = S.running || (k === (S.ft?.from ?? S.dataset) && k !== 'upload');
    }
    $('#ft-upload').hidden = selected !== 'upload';
    runBtn.disabled = busy;
    const same = S.ft && S.ft.to === selected;
    runBtn.textContent = S.running ? 'Pause' : same ? (S.step < S.total ? 'Weiter fine-tunen' : 'Weitere Schritte fine-tunen') : 'Fine-Tuning starten';
    $('#ft-revert').hidden = !S.ft;
    $('#ft-from').textContent = DATASETS[S.ft?.from ?? S.dataset].label;

    const ft = S.ft;
    $('#ft-result').hidden = !ft;
    if (!ft) return;
    const after = S.evals.filter((e) => e.step > ft.baseStep).at(-1);
    const afterSample = S.samples.filter((smp) => smp.step > ft.baseStep).at(-1);
    $('#ft-before-loss').textContent = fmt(ft.before.valLoss);
    $('#ft-after-loss').textContent = after ? fmt(after.valLoss) : '–';
    $('#ft-before-sample').replaceWith(Object.assign(sampleCell(ft.before.sample), { id: 'ft-before-sample' }));
    $('#ft-after-sample').replaceWith(Object.assign(sampleCell(afterSample), { id: 'ft-after-sample' }));
    $('#ft-info').textContent = `Neuer Text: ${DATASETS[ft.to].label}. Mit dem alten Tokenizer ergibt er ${fmt(ft.charsPerToken, 2)} Zeichen pro Token. Schritt ${int(S.step)} von ${int(S.total)}.`;
    const losses = S.losses.slice(ft.baseStep);
    chart.update([
      { name: 'Train-Loss geglättet', values: movingAverage(losses, 25) },
      { name: 'Val-Loss (neuer Text)', points: [[0, ft.before.valLoss], ...S.evals.filter((e) => e.step > ft.baseStep).map((e) => [e.step - ft.baseStep, e.valLoss])], color: '--ink', dash: [5, 4] },
    ], Math.max(1, S.total - ft.baseStep));
  }

  let pending = false;
  p2.on(() => { if (!pending) { pending = true; requestAnimationFrame(() => { pending = false; render(); }); } });
  render();
}
