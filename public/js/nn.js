// =====================================================================
// nn.js – Bausteine, die jedes Sprachmodell braucht
// =====================================================================

import { Value } from './value.js';

//#region softmax
// Softmax: beliebige Zahlen ("Logits") → Wahrscheinlichkeiten.
//
//     p_i = e^(x_i) / Σ_j e^(x_j)
//
// e^x macht alles positiv, das Teilen durch die Summe sorgt dafür, dass
// alles zusammen 1 ergibt. Große Logits → große Wahrscheinlichkeit.
//
// Trick für numerische Stabilität: Wir ziehen vorher das Maximum ab.
// Das ändert das Ergebnis nicht (kürzt sich im Bruch weg), verhindert
// aber, dass e^x bei großen x zu Infinity wird.
export function softmax(logits) {
  const max = Math.max(...logits.map((l) => l.data));
  const exps = logits.map((l) => l.add(-max).exp());
  const invSum = Value.sum(exps).pow(-1);
  return exps.map((e) => e.mul(invSum));
}
//#endregion

//#region cross-entropy
// Cross-Entropy-Loss für EIN Beispiel: −log(p_ziel).
// Das Modell wird bestraft, wenn es dem richtigen Token eine kleine
// Wahrscheinlichkeit gibt. p = 1 → Loss 0.  p = 0,01 → Loss 4,6.
//
// Statt erst softmax und dann log zu rechnen, nutzen wir die
// gleichwertige, stabilere Form:
//
//     −log(p_ziel) = log(Σ_j e^(x_j)) − x_ziel
export function crossEntropy(logits, target) {
  const max = Math.max(...logits.map((l) => l.data));
  const sumExp = Value.sum(logits.map((l) => l.add(-max).exp()));
  return sumExp.log().sub(logits[target].add(-max));
}
//#endregion

// Softmax auf normalen Zahlen (für Anzeige und Generierung – dort brauchen
// wir keine Gradienten). `temperature` < 1 macht die Verteilung spitzer,
// > 1 flacher.
export function softmaxNumbers(xs, temperature = 1) {
  const max = Math.max(...xs);
  const exps = xs.map((x) => Math.exp((x - max) / temperature));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => e / sum);
}

// Matrix (rows × cols) aus Values, normalverteilt initialisiert.
export function randomMatrix(rows, cols, rng, std) {
  return Array.from({ length: rows }, () =>
    Array.from({ length: cols }, () => new Value(rng.gauss(std))),
  );
}
