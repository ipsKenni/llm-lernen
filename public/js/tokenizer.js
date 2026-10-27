// =====================================================================
// tokenizer.js – Text ↔ Zahlen
// =====================================================================
//
// Ein neuronales Netz kann nur mit Zahlen rechnen. Der Tokenizer legt
// fest, welches Textstück welche Zahl bekommt.
//
// Wir machen es so einfach wie möglich: jedes Zeichen ist ein Token.
// Dazu ein Sonder-Token '.' mit der ID 0. Es bedeutet "Anfang" bzw.
// "Ende" eines Wortes. So weiß das Modell, wann ein Name beginnt und
// wann es aufhören soll.
//
// Echte LLMs nutzen Byte-Pair-Encoding (BPE): häufige Zeichenfolgen wie
// "ung" oder " the" werden zu einem einzigen Token zusammengefasst.
// Das Prinzip bleibt aber dasselbe: Text rein, Liste von IDs raus.

//#region tokenizer
export const BOS = 0;   // "Beginning/End Of Sequence" – unser '.'

export function buildTokenizer(words) {
  // Alle vorkommenden Zeichen, sortiert. '.' kommt immer an Position 0.
  const chars = [...new Set(words.join(''))].sort();
  return tokenizerFromItos(['.', ...chars]);
}

export function tokenizerFromItos(itos) {             // itos: ID → Zeichen
  const stoi = new Map(itos.map((c, i) => [c, i]));   // Zeichen → ID
  return {
    itos,
    stoi,
    vocabSize: itos.length,
    encode: (text) => [...text].map((c) => stoi.get(c)),
    decode: (ids) => ids.map((i) => itos[i]).join(''),
  };
}
//#endregion

// Rohtext → saubere Wortliste: eine Zeile pro Wort, '.' ist reserviert.
export function parseWords(text) {
  return text
    .split(/\r?\n/)
    .map((w) => w.trim().replaceAll('.', ''))
    .filter((w) => w.length > 0);
}
