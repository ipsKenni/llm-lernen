// Schritt 10 – Modell ausprobieren, als GGUF exportieren, in Ollama starten.
import { DATASETS } from '../llama/data.js';
import { makeModelfile } from '../llama/gguf.js';
import { fmt } from '../viz.js';
import { part2 } from './teil2.js';

const OS = {
  windows: { label: 'Windows', cd: 'cd $HOME\\Downloads', shell: 'PowerShell' },
  mac: { label: 'macOS', cd: 'cd ~/Downloads', shell: 'Terminal' },
  linux: { label: 'Linux', cd: 'cd ~/Downloads', shell: 'Terminal' },
};
const DEFAULT_NAMES = { maerchen: 'maerchen', tinystories: 'tinystories', namen: 'namen', upload: 'mein-modell' };

const sanitize = (s) => s.toLowerCase().replace(/[^a-z0-9-]+/g, '-').replace(/^-+|-+$/g, '').slice(0, 40) || 'mein-modell';
const quote = (s) => `"${s.replaceAll('"', '\\"')}"`;

function download(data, fileName, type) {
  const url = URL.createObjectURL(new Blob([data], { type }));
  const a = document.createElement('a');
  a.href = url;
  a.download = fileName;
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 5000);
}

// Befehlsblock mit Kopier-Knopf
function commandBlock(container, lines) {
  const pre = document.createElement('pre');
  const code = document.createElement('code');
  code.textContent = lines.join('\n');
  pre.append(code);
  const btn = document.createElement('button');
  btn.type = 'button';
  btn.className = 'btn btn-small cmd-copy';
  btn.textContent = 'Kopieren';
  btn.addEventListener('click', async () => {
    try { await navigator.clipboard.writeText(lines.join('\n')); btn.textContent = 'Kopiert'; } catch { btn.textContent = 'Bitte von Hand kopieren'; }
    setTimeout(() => { btn.textContent = 'Kopieren'; }, 1500);
  });
  container.replaceChildren(pre, btn);
}

export function init(root) {
  const p2 = part2();
  const S = p2.state;
  const $ = (sel) => root.querySelector(sel);
  const prompt = $('#ex-prompt');
  const nameInput = $('#ex-name');
  const out = $('#ex-out');
  const status = $('#ex-status');
  let os = /Win/.test(navigator.platform || navigator.userAgent) ? 'windows' : /Mac/.test(navigator.platform || navigator.userAgent) ? 'mac' : 'linux';
  let exported = null;   // { name, step, greedy, prompt }
  let lastDataset = null;

  const sliders = [['temp', (v) => fmt(v, 2)], ['topk', (v) => String(v)], ['topp', (v) => fmt(v, 2)], ['len', (v) => `${v} Tokens`]];
  for (const [id, f] of sliders) {
    const input = $(`#ex-${id}`), output = $(`#ex-${id}-out`);
    const upd = () => { output.textContent = f(Number(input.value)); };
    input.addEventListener('input', upd);
    upd();
  }

  // --- Ausprobieren ----------------------------------------------------
  $('#ex-gen').addEventListener('click', async () => {
    const btn = $('#ex-gen');
    btn.disabled = true;
    const head = document.createElement('span');
    head.className = 'gen-prompt';
    head.textContent = prompt.value;
    const body = document.createElement('span');
    out.replaceChildren(head, body);
    try {
      const r = await p2.trainer.generate({
        prompt: prompt.value, temperature: Number($('#ex-temp').value), topK: Number($('#ex-topk').value),
        topP: Number($('#ex-topp').value), maxTokens: Number($('#ex-len').value),
      }, (text) => { body.textContent = text; });
      body.textContent = r.text || ' (Das Modell hat sofort das Ende-Token gewählt.)';
    } catch (e) {
      body.textContent = ` Fehler: ${e.message}`;
    } finally {
      btn.disabled = false;
    }
  });

  // --- Export ------------------------------------------------------------
  const name = () => sanitize(nameInput.value);
  async function ensureExport() {
    if (exported && exported.name === name() && exported.step === S.step) return exported;
    status.textContent = 'Erzeuge die GGUF-Datei …';
    const r = await p2.trainer.exportModel(name());
    exported = { name: name(), step: S.step, ...r };
    const rows = r.meta.map(([k, v]) => {
      const tr = document.createElement('tr');
      const th = document.createElement('th');
      th.scope = 'row';
      th.textContent = k;
      const td = document.createElement('td');
      td.textContent = v;
      tr.append(th, td);
      return tr;
    });
    $('#ex-meta').replaceChildren(...rows);
    $('#ex-meta-wrap').hidden = false;
    $('#ex-ref').textContent = r.greedy ? `${r.prompt}${r.greedy}` : `${r.prompt} (sofort Ende)`;
    renderCommands();
    return exported;
  }

  $('#ex-gguf').addEventListener('click', async () => {
    try {
      const r = await ensureExport();
      download(r.gguf, `${r.name}.gguf`, 'application/octet-stream');
      status.textContent = `${r.name}.gguf heruntergeladen: ${fmt(r.gguf.byteLength / 2 ** 20, 1)} MB, ${r.tensors} Tensoren, Stand nach ${S.step} Trainingsschritten. Lade auch das Modelfile herunter.`;
    } catch (e) {
      status.textContent = `Export fehlgeschlagen: ${e.message}`;
    }
  });
  $('#ex-modelfile').addEventListener('click', () => {
    if (!S.setup) return;
    download(makeModelfile(`${name()}.gguf`, S.setup.cfg), `${name()}.Modelfile`, 'text/plain');
    status.textContent = `${name()}.Modelfile heruntergeladen. Lege es in denselben Ordner wie ${name()}.gguf.`;
  });
  nameInput.addEventListener('input', () => renderCommands());
  nameInput.addEventListener('change', () => { nameInput.value = name(); renderCommands(); });

  // --- Anleitung ---------------------------------------------------------
  const osBox = $('#ex-os');
  const osButtons = Object.entries(OS).map(([key, o]) => {
    const b = document.createElement('button');
    b.type = 'button';
    b.setAttribute('role', 'radio');
    b.textContent = o.label;
    b.addEventListener('click', () => { os = key; renderCommands(); });
    osBox.append(b);
    return [key, b];
  });

  function renderCommands() {
    for (const [k, b] of osButtons) b.setAttribute('aria-checked', String(k === os));
    const n = name();
    const p = prompt.value.trim() || DATASETS[S.dataset]?.prompt || 'Es war einmal';
    commandBlock($('#ex-cmds'), [
      OS[os].cd,
      `ollama create ${n} -f ${n}.Modelfile`,
      `ollama run ${n} ${quote(p)}`,
    ]);
    const refPrompt = exported?.prompt ?? DATASETS[S.dataset]?.prompt ?? p;
    commandBlock($('#ex-check-cmds'), [
      `ollama run ${n}`,
      '/set parameter temperature 0',
      '/set parameter repeat_penalty 1',
      refPrompt,
    ]);
  }

  function render() {
    const ready = !!S.setup && S.step > 0;
    $('#ex-none').hidden = ready;
    $('#ex-main').hidden = !ready;
    if (!ready) return;
    if (lastDataset !== S.dataset) {
      lastDataset = S.dataset;
      prompt.value = DATASETS[S.dataset].prompt;
      nameInput.value = DEFAULT_NAMES[S.dataset];
      exported = null;
      $('#ex-ref').textContent = 'erscheint nach dem Herunterladen der GGUF-Datei';
    }
    renderCommands();
  }
  p2.on((what) => { if (['setup', 'restored', 'done', 'state', 'tokenized'].includes(what)) render(); });
  render();
}
