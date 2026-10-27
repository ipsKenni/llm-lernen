// =====================================================================
// bpe.js – Byte-Pair-Encoding: der Tokenizer echter LLMs
// =====================================================================
//
// In Teil 1 war jedes Zeichen ein Token. Echte LLMs fassen häufige
// Zeichenfolgen zu einem Token zusammen: " und", "ch", "Königs" …
// Das macht Texte kürzer (mehr Inhalt passt in den Kontext) und gibt
// dem Modell größere Bausteine.
//
// Der Algorithmus (Sennrich et al. 2016, so auch in GPT-2/GPT-4/Llama 3):
//   1. Text in Bytes zerlegen. Jedes der 256 möglichen Bytes ist ein Token.
//      → Jeder beliebige Text ist darstellbar, auch Emojis und Tippfehler.
//   2. Das häufigste benachbarte Token-Paar finden, z. B. ("e", "n").
//   3. Daraus ein neues Token "en" machen und überall ersetzen.
//   4. Wiederholen, bis das Vokabular groß genug ist.
// Die Liste der Verschmelzungen ("Merges") IST der Tokenizer.
//
// Wir bauen das kompatibel zu llama.cpp (tokenizer.ggml.model = "gpt2"),
// damit Ollama später exakt dieselben Tokens erzeugt wie unser Training.

export const EOT = '<|endoftext|>';

//#region bpe-pretokenize
// Vor-Zerlegung in "Wörter" (GPT-2-Regel): Merges passieren nur innerhalb
// eines Wortes, nie über Wortgrenzen hinweg. Ein Leerzeichen gehört immer
// zum folgenden Wort: "Es war" → ["Es", " war"].
export const PRETOKENIZE = /'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu;
//#endregion

//#region bpe-bytes
// GPT-2 speichert Byte-Tokens als druckbare Unicode-Zeichen, damit im
// Vokabular keine Steuerzeichen stehen: Das Leerzeichen (Byte 32) wird
// zu "Ġ", der Zeilenumbruch (Byte 10) zu "Ċ". Druckbare Bytes bleiben,
// wie sie sind. Genau diese Tabelle erwartet auch llama.cpp.
export const BYTE_TO_CHAR = (() => {
  const printable = [];
  for (let b = 33; b <= 126; b++) printable.push(b);
  for (let b = 161; b <= 172; b++) printable.push(b);
  for (let b = 174; b <= 255; b++) printable.push(b);
  const table = new Array(256);
  let extra = 0;
  for (let b = 0; b < 256; b++) {
    table[b] = String.fromCodePoint(printable.includes(b) ? b : 256 + extra++);
  }
  return table;
})();
//#endregion
const CHAR_TO_BYTE = new Map(BYTE_TO_CHAR.map((c, b) => [c, b]));
const utf8 = new TextEncoder();

// Bytes lesbar machen: vollständige UTF-8-Zeichen als Text, einzelne Bytes
// eines angefangenen Zeichens (z. B. das erste Byte von „ä“) als ‹C3›.
// Leerzeichen und Umbrüche werden sichtbar: ␣ ⏎ ⇥
export function bytesForDisplay(bytes) {
  let out = '';
  for (let i = 0; i < bytes.length;) {
    const b = bytes[i];
    const len = b < 0x80 ? 1 : b >= 0xf0 ? 4 : b >= 0xe0 ? 3 : b >= 0xc0 ? 2 : 0;
    let ok = len > 0 && i + len <= bytes.length;
    for (let j = 1; ok && j < len; j++) ok = (bytes[i + j] & 0xc0) === 0x80;
    if (ok) {
      out += new TextDecoder().decode(Uint8Array.from(bytes.slice(i, i + len)));
      i += len;
    } else {
      out += `‹${b.toString(16).toUpperCase().padStart(2, '0')}›`;
      i += 1;
    }
  }
  return out.replaceAll(' ', '␣').replaceAll('\n', '⏎').replaceAll('\t', '⇥');
}

// Ein Wort als Folge von Byte-IDs.
const wordBytes = (word) => Array.from(utf8.encode(word));

// Kleiner Max-Heap für [Häufigkeit, Paar-Schlüssel]; bei Gleichstand gewinnt
// der kleinere Schlüssel, damit das Ergebnis immer gleich ist.
class PairHeap {
  constructor() { this.a = []; }
  better(i, j) { const x = this.a[i], y = this.a[j]; return x[0] > y[0] || (x[0] === y[0] && x[1] < y[1]); }
  push(item) {
    const a = this.a;
    a.push(item);
    let i = a.length - 1;
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (!this.better(i, p)) break;
      [a[i], a[p]] = [a[p], a[i]];
      i = p;
    }
  }
  pop() {
    const a = this.a;
    const top = a[0];
    const last = a.pop();
    if (a.length > 0) {
      a[0] = last;
      let i = 0;
      for (;;) {
        const l = 2 * i + 1, r = l + 1;
        let m = i;
        if (l < a.length && this.better(l, m)) m = l;
        if (r < a.length && this.better(r, m)) m = r;
        if (m === i) break;
        [a[i], a[m]] = [a[m], a[i]];
        i = m;
      }
    }
    return top;
  }
  get size() { return this.a.length; }
}

const KEY = 1 << 20;  // Paar (a, b) → eine Zahl a·2^20 + b
const pairKey = (a, b) => a * KEY + b;

//#region bpe-train
// Trainiert einen Tokenizer mit `vocabSize` Tokens (inkl. 256 Bytes und EOT).
// `onMerge` wird nach jedem neuen Token aufgerufen (für die Anzeige).
export function trainBPE(docs, vocabSize, { onMerge } = {}) {
  // 1) Wörter zählen. Jedes Wort ist eine Liste von Token-IDs (am Anfang Bytes).
  const counts = new Map();
  for (const doc of docs) {
    for (const [w] of doc.matchAll(PRETOKENIZE)) counts.set(w, (counts.get(w) || 0) + 1);
  }
  const words = [...counts.entries()].map(([w, freq]) => ({ ids: wordBytes(w), freq }));

  // 2) Alle benachbarten Paare zählen und merken, in welchen Wörtern sie vorkommen.
  const pairCount = new Map();
  const pairWords = new Map();
  const addPairs = (wi, sign) => {
    const { ids, freq } = words[wi];
    for (let i = 0; i < ids.length - 1; i++) {
      const k = pairKey(ids[i], ids[i + 1]);
      pairCount.set(k, (pairCount.get(k) || 0) + sign * freq);
      if (sign > 0) {
        let s = pairWords.get(k);
        if (!s) pairWords.set(k, (s = new Set()));
        s.add(wi);
      }
    }
  };
  words.forEach((_, wi) => addPairs(wi, +1));
  const heap = new PairHeap();
  for (const [k, c] of pairCount) heap.push([c, k]);

  const tokens = BYTE_TO_CHAR.slice();          // ID → Text (Byte-Darstellung)
  const tokenId = new Map(tokens.map((t, i) => [t, i]));
  const merges = [];                            // [[a, b, neu], …] in Reihenfolge = Rang

  // 3) Immer das häufigste Paar verschmelzen.
  while (tokens.length < vocabSize - 1 && heap.size > 0) {
    const [c, k] = heap.pop();
    if (pairCount.get(k) !== c || c < 2) continue;   // veralteter Heap-Eintrag
    const a = Math.floor(k / KEY), b = k % KEY;
    const text = tokens[a] + tokens[b];
    let id = tokenId.get(text);
    if (id === undefined) { id = tokens.length; tokens.push(text); tokenId.set(text, id); }
    merges.push([a, b, id]);

    const touched = new Set();
    for (const wi of pairWords.get(k)) {
      const ids = words[wi].ids;
      let has = false;
      for (let i = 0; i < ids.length - 1; i++) if (ids[i] === a && ids[i + 1] === b) { has = true; break; }
      if (!has) continue;
      addPairs(wi, -1);
      const out = [];
      for (let i = 0; i < ids.length; i++) {
        if (i < ids.length - 1 && ids[i] === a && ids[i + 1] === b) { out.push(id); i++; } else out.push(ids[i]);
      }
      words[wi].ids = out;
      addPairs(wi, +1);
      for (let i = 0; i < out.length - 1; i++) touched.add(pairKey(out[i], out[i + 1]));
    }
    pairCount.delete(k);
    for (const t of touched) { const n = pairCount.get(t); if (n > 0) heap.push([n, t]); }
    onMerge?.({ rank: merges.length - 1, a, b, id, count: c, text });
  }
  tokens.push(EOT);
  return makeTokenizer({ tokens, merges: merges.map(([a, b]) => [a, b]) });
}
//#endregion

// Aus gespeicherten Daten ({ tokens, merges }) einen benutzbaren Tokenizer bauen.
export function makeTokenizer({ tokens, merges }) {
  const tokenId = new Map(tokens.map((t, i) => [t, i]));
  const rank = new Map(merges.map(([a, b], r) => [pairKey(a, b), r]));
  const mergedId = new Map(merges.map(([a, b]) => [pairKey(a, b), tokenId.get(tokens[a] + tokens[b])]));
  const eot = tokenId.get(EOT);
  const cache = new Map();

  //#region bpe-encode
  // Ein Wort kodieren: mit Bytes anfangen und immer das Paar mit dem
  // niedrigsten Rang (= am frühesten gelernt) verschmelzen – genau wie llama.cpp.
  function encodeWord(word) {
    let ids = cache.get(word);
    if (ids) return ids;
    ids = wordBytes(word);
    for (;;) {
      let best = -1, bestRank = Infinity;
      for (let i = 0; i < ids.length - 1; i++) {
        const r = rank.get(pairKey(ids[i], ids[i + 1]));
        if (r !== undefined && r < bestRank) { bestRank = r; best = i; }
      }
      if (best < 0) break;
      const [a, b] = [ids[best], ids[best + 1]];
      const id = mergedId.get(pairKey(a, b));
      const out = [];
      for (let i = 0; i < ids.length; i++) {
        if (i < ids.length - 1 && ids[i] === a && ids[i + 1] === b) { out.push(id); i++; } else out.push(ids[i]);
      }
      ids = out;
    }
    if (cache.size < 200000) cache.set(word, ids);
    return ids;
  }
  //#endregion

  function encode(text) {
    const out = [];
    for (const [w] of text.matchAll(PRETOKENIZE)) for (const id of encodeWord(w)) out.push(id);
    return out;
  }

  // Tokens → Bytes → Text. Unvollständige UTF-8-Folgen werden als � gezeigt.
  function decodeBytes(ids) {
    const bytes = [];
    for (const id of ids) {
      if (id === eot) continue;
      for (const ch of tokens[id]) bytes.push(CHAR_TO_BYTE.get(ch));
    }
    return new Uint8Array(bytes);
  }
  const decode = (ids) => new TextDecoder().decode(decodeBytes(ids));
  // Ein einzelnes Token lesbar machen (für Kacheln und Listen).
  const display = (id) => (id === eot ? '⟨Ende⟩' : bytesForDisplay(decodeBytes([id])));

  return { tokens, merges, eot, vocabSize: tokens.length, encode, encodeWord, decode, decodeBytes, display };
}
