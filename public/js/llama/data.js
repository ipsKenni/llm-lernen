// =====================================================================
// data.js – Trainingsdaten: Dokumente → ein langer Token-Strom
// =====================================================================
//
// Jedes Dokument (ein Name, ein Märchen, eine Geschichte) wird tokenisiert
// und hinter ein <|endoftext|> gehängt. Das Modell lernt so auch, wann ein
// Text zu Ende ist – und Ollama hört später an genau dieser Stelle auf.
//
// Trainiert wird auf zufälligen Ausschnitten dieses Stroms. Die letzten 5 %
// der Dokumente sieht das Modell beim Training nie: Daran messen wir den
// Validierungs-Loss, also wie gut es auf UNBEKANNTEM Text ist.

export const DATASETS = {
  maerchen: { label: 'Grimms Märchen', file: 'maerchen.txt', prompt: 'Es war einmal', vocab: 2048, preset: 'klein', source: 'Projekt Gutenberg #77905, gemeinfrei' },
  tinystories: { label: 'TinyStories', file: 'tinystories.txt', prompt: 'Once upon a time', vocab: 2048, preset: 'klein', source: 'roneneldan/TinyStories, CDLA-Sharing-1.0' },
  namen: { label: 'Vornamen', file: 'names.txt', prompt: 'emma', vocab: 512, preset: 'mini', source: 'karpathy/makemore, MIT' },
  upload: { label: 'Eigener Text', file: null, prompt: '', vocab: 1024, preset: 'mini', source: 'deine Datei' },
};

// Text → Dokumente
export function splitDocs(kind, text) {
  text = text.replace(/\r\n?/g, '\n');
  if (text.includes('<|endoftext|>')) return text.split(/\n?<\|endoftext\|>\n?/).map((d) => d.trim()).filter(Boolean);
  const lines = text.split('\n').map((l) => l.trim()).filter(Boolean);
  // Namen: je 10 Stück untereinander als ein Dokument. So lernt das Modell
  // Namenslisten – in Ollama tippt man einen Namen und bekommt weitere.
  if (kind === 'namen') {
    const docs = [];
    for (let i = 0; i < lines.length; i += 10) docs.push(lines.slice(i, i + 10).join('\n'));
    return docs;
  }
  // Eigene Dateien: Bei kurzen Zeilen (Listen) ist jede Zeile ein Dokument,
  // sonst jeder durch eine Leerzeile getrennte Absatz.
  const avg = lines.reduce((s, l) => s + l.length, 0) / Math.max(1, lines.length);
  if (avg < 40) return lines;
  return text.split(/\n\s*\n/).map((d) => d.trim()).filter(Boolean);
}

export function buildStreams(docs, tok) {
  const nVal = Math.max(1, Math.round(docs.length * 0.05));
  const encode = (list) => {
    const parts = list.map((d) => tok.encode(d));
    const out = new Int32Array(parts.reduce((s, p) => s + p.length + 1, 0) + 1);
    let i = 0;
    for (const p of parts) { out[i++] = tok.eot; out.set(p, i); i += p.length; }
    out[i] = tok.eot;
    return out;
  };
  // Bei sehr wenigen Dokumenten (z. B. ein einziger langer Text) den Strom selbst teilen.
  if (docs.length < 20) {
    const all = encode(docs);
    const cut = Math.floor(all.length * 0.95);
    return { train: all.subarray(0, cut), val: all.subarray(cut) };
  }
  return { train: encode(docs.slice(0, docs.length - nVal)), val: encode(docs.slice(docs.length - nVal)) };
}

// B zufällige Ausschnitte der Länge T (+1 fürs Ziel).
export function sampleBatch(stream, B, T, rng) {
  const inp = new Int32Array(B * T), tgt = new Int32Array(B * T);
  for (let b = 0; b < B; b++) {
    const start = rng.int(Math.max(1, stream.length - T - 1));
    for (let t = 0; t < T; t++) {
      inp[b * T + t] = stream[(start + t) % stream.length];
      tgt[b * T + t] = stream[(start + t + 1) % stream.length];
    }
  }
  return { inp, tgt };
}

//#region llama-schedule
// Lernrate: kurz hochfahren (Warmup), dann als Kosinus auf 10 % absenken.
export function learningRate(step, total, maxLr) {
  const warmup = Math.min(200, Math.max(20, Math.round(total * 0.05)));
  if (step < warmup) return (maxLr * (step + 1)) / warmup;
  const progress = Math.min(1, (step - warmup) / Math.max(1, total - warmup));
  return maxLr * (0.1 + 0.9 * 0.5 * (1 + Math.cos(Math.PI * progress)));
}
//#endregion
