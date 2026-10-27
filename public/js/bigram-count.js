// =====================================================================
// bigram-count.js – Das einfachste Sprachmodell der Welt: Zählen
// =====================================================================
//
// Ein Sprachmodell beantwortet immer dieselbe Frage:
//
//     "Welches Token kommt als Nächstes – und mit welcher
//      Wahrscheinlichkeit?"
//
// Ein Bigram-Modell schaut dafür nur auf das EINE vorherige Zeichen.
// Wir zählen einfach in allen Namen, wie oft auf 'a' ein 'n' folgt,
// wie oft auf 'n' ein '.' (Ende) folgt usw. Fertig ist das Modell.
// Kein Training, keine Gradienten – nur eine Tabelle.

import { BOS } from './tokenizer.js';

//#region bigram-zaehlen
// N[a][b] = wie oft folgt Token b auf Token a?
export function countBigrams(words, tok) {
  const V = tok.vocabSize;
  const N = Array.from({ length: V }, () => new Array(V).fill(0));
  for (const w of words) {
    // ".emma." → Paare: (.,e) (e,m) (m,m) (m,a) (a,.)
    const ids = [BOS, ...tok.encode(w), BOS];
    for (let i = 0; i < ids.length - 1; i++) N[ids[i]][ids[i + 1]] += 1;
  }
  return N;
}

// Zählungen → Wahrscheinlichkeiten: jede Zeile durch ihre Summe teilen.
// `smoothing` addiert zu jeder Zählung etwas dazu, damit nie eine
// Wahrscheinlichkeit exakt 0 ist (log(0) wäre −∞).
export function countsToProbs(N, smoothing = 1) {
  return N.map((row) => {
    const total = row.reduce((s, c) => s + c + smoothing, 0);
    return row.map((c) => (c + smoothing) / total);
  });
}
//#endregion

//#region bigram-loss
// Wie gut ist das Modell? Wir nehmen alle echten Paare aus den Daten und
// schauen, welche Wahrscheinlichkeit das Modell dem tatsächlich folgenden
// Zeichen gegeben hat. Davon der negative Logarithmus, gemittelt:
//
//     Loss = −(1/n) · Σ log P(nächstes | vorheriges)
//
// Perfektes Modell (immer P = 1) → Loss 0.
// Raten mit 27 gleich wahrscheinlichen Zeichen → Loss = ln(27) ≈ 3,30.
export function averageNll(words, tok, P) {
  let sum = 0;
  let n = 0;
  for (const w of words) {
    const ids = [BOS, ...tok.encode(w), BOS];
    for (let i = 0; i < ids.length - 1; i++) {
      sum -= Math.log(P[ids[i]][ids[i + 1]]);
      n += 1;
    }
  }
  return sum / n;
}
//#endregion

//#region bigram-sample
// Einen neuen Namen erzeugen: bei '.' starten, Zeile nachschlagen,
// zufällig gemäß den Wahrscheinlichkeiten ziehen, wiederholen, bis
// wieder '.' gezogen wird. Gibt auch jeden Zwischenschritt zurück,
// damit die Seite ihn anzeigen kann.
export function sampleWord(P, tok, rng, maxLen = 30) {
  const steps = [];
  let current = BOS;
  for (let i = 0; i < maxLen; i++) {
    const probs = P[current];
    const next = rng.sample(probs);
    steps.push({ from: current, probs, next });
    if (next === BOS) break;
    current = next;
  }
  const word = tok.decode(steps.map((s) => s.next).filter((id) => id !== BOS));
  return { word, steps };
}
//#endregion
