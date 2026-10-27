// =====================================================================
// code-view.js – Zeigt den echten Quelltext, der gerade im Browser läuft
// =====================================================================
// Die Ausschnitte auf der Seite sind keine Kopien: Sie werden zur Laufzeit
// aus genau den Dateien geladen, die auch ausgeführt werden. Markiert sind
// sie im Code mit  //#region name  …  //#endregion

const cache = new Map();

async function loadSource(file) {
  if (!cache.has(file)) cache.set(file, fetch(`js/${file}`).then((r) => r.text()));
  return cache.get(file);
}

export function extractRegion(source, name) {
  const lines = source.split('\n');
  const start = lines.findIndex((l) => l.trim() === `//#region ${name}`);
  if (start < 0) return `// Abschnitt "${name}" nicht gefunden`;
  const out = [];
  let depth = 0;
  for (let i = start + 1; i < lines.length; i++) {
    const t = lines[i].trim();
    if (t.startsWith('//#region')) { depth++; continue; }
    if (t === '//#endregion') { if (depth === 0) break; depth--; continue; }
    out.push(lines[i]);
  }
  // gemeinsame Einrückung entfernen
  const indent = Math.min(...out.filter((l) => l.trim()).map((l) => l.match(/^ */)[0].length));
  return out.map((l) => l.slice(indent)).join('\n').replace(/\s+$/, '');
}

// Minimales Syntax-Highlighting: Kommentare, Strings, Zahlen, Schlüsselwörter.
const KEYWORDS = new Set('const let var function return for while if else new class export import from of in static this break continue'.split(' '));

function highlight(code) {
  const frag = document.createDocumentFragment();
  const re = /(\/\/[^\n]*)|('(?:[^'\\]|\\.)*'|`(?:[^`\\]|\\.)*`)|(\b\d+(?:\.\d+)?(?:e-?\d+)?\b)|([A-Za-z_$][\w$]*)|([\s\S])/g;
  let m;
  let plain = '';
  const flush = () => { if (plain) { frag.append(plain); plain = ''; } };
  const span = (cls, text) => { flush(); const s = document.createElement('span'); s.className = cls; s.textContent = text; frag.append(s); };
  while ((m = re.exec(code))) {
    if (m[1]) span('c-com', m[1]);
    else if (m[2]) span('c-str', m[2]);
    else if (m[3]) span('c-num', m[3]);
    else if (m[4]) KEYWORDS.has(m[4]) ? span('c-kw', m[4]) : (plain += m[4]);
    else plain += m[5];
  }
  flush();
  return frag;
}

// <div data-code="value.js#value-ops"></div>  →  Code-Block
export async function renderCode(el) {
  const [file, region] = el.dataset.code.split('#');
  const source = await loadSource(file);
  const pre = document.createElement('pre');
  const code = document.createElement('code');
  code.append(highlight(extractRegion(source, region)));
  pre.append(code);
  const cap = document.createElement('div');
  cap.className = 'code-cap';
  const a = document.createElement('a');
  a.href = `js/${file}`;
  a.target = '_blank';
  a.rel = 'noopener';
  a.textContent = `js/${file}`;
  cap.append('Aus ', a, ' – genau dieser Code läuft gerade in deinem Browser.');
  el.replaceChildren(pre, cap);
}

export function renderAllCode(root) {
  return Promise.all([...root.querySelectorAll('[data-code]')].map((el) => el.dataset.rendered ? null : (el.dataset.rendered = '1', renderCode(el))));
}
