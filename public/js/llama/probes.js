// =====================================================================
// probes.js – Typische Wendungen finden, an denen man Lernfortschritt sieht
// =====================================================================
// Wir zählen im Trainingstext, welche drei Wörter oft aufeinander folgen und
// welches vierte Wort dann fast immer kommt („Es war einmal → ein“). Gute Proben
// sind häufig UND eindeutig: Dann kann das Modell sie lernen, und man sieht es.

import { PRETOKENIZE } from './bpe.js';

const WORD = /^ ?\p{L}+$/u;

//#region probes
export function findProbes(docs, { count = 5, maxChars = 1_500_000 } = {}) {
  for (const n of [3, 2]) {                     // erst 3 Wörter Kontext, notfalls 2
    const follow = new Map();                   // Kontext → Map(nächstes Wort → Anzahl)
    let chars = 0;
    for (const doc of docs) {
      if ((chars += doc.length) > maxChars) break;
      for (const line of doc.split('\n')) {
        const words = [];
        for (const [piece] of line.matchAll(PRETOKENIZE)) {
          if (!WORD.test(piece)) { words.length = 0; continue; }   // Satzzeichen trennen Wendungen
          words.push(piece);
          if (words.length > n) {
            const key = words.slice(-n - 1, -1).join('');
            let m = follow.get(key);
            if (!m) follow.set(key, (m = new Map()));
            const next = words[words.length - 1];
            m.set(next, (m.get(next) || 0) + 1);
          }
        }
      }
    }
    const candidates = [];
    for (const [key, m] of follow) {
      let total = 0, best = null, bestCount = 0;
      for (const [w, c] of m) { total += c; if (c > bestCount) { bestCount = c; best = w; } }
      if (bestCount < 5) continue;
      const share = bestCount / total;
      candidates.push({ prefix: key.trimStart(), target: best, count: bestCount, share, score: bestCount * share * share });
    }
    candidates.sort((a, b) => b.score - a.score);
    // Abwechslungsreich: zwei Proben teilen sich höchstens ein Wort
    const out = [], chosen = [];
    const wordsOf = (c) => new Set(`${c.prefix}${c.target}`.toLowerCase().split(' ').filter(Boolean));
    for (const c of candidates) {
      const w = wordsOf(c);
      if (chosen.some((o) => [...w].filter((x) => o.has(x)).length >= 2)) continue;
      chosen.push(w);
      out.push({ prefix: c.prefix, target: c.target, share: c.share });
      if (out.length === count) break;
    }
    if (out.length >= 3) return out;
  }
  return [];
}
//#endregion
