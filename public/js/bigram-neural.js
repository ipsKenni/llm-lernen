// =====================================================================
// bigram-neural.js – Dieselbe Tabelle, aber gelernt statt gezählt
// =====================================================================
//
// Das Zähl-Modell ist schon optimal für "ein Zeichen Kontext". Warum also
// ein neuronales Netz? Weil Zählen nicht skaliert: Mit 10 Zeichen Kontext
// bräuchte man 27^10 ≈ 200 Billionen Tabellenzeilen. Ein Netz dagegen
// kann verallgemeinern.
//
// Hier üben wir das Handwerk an einem Problem, dessen Lösung wir kennen:
//   - Parameter: eine 27×27-Matrix W, anfangs Zufall
//   - Forward:   Zeile W[x] sind die Logits für "was folgt auf x?"
//                (mathematisch: One-Hot-Vektor von x mal W)
//   - Loss:      Cross-Entropy gegen das tatsächliche nächste Zeichen
//   - Backward:  unser Autograd berechnet dLoss/dW
//   - Update:    W ← W − Lernrate · Gradient
//
// Nach genug Schritten landet der Loss beim Zähl-Modell (≈ 2,45).
// Gradientenabstieg hat die Tabelle selbst "erzählt".

import { Value } from './value.js';
import { BOS } from './tokenizer.js';
import { crossEntropy, randomMatrix } from './nn.js';

// Alle (vorher, nachher)-Paare aus den Wörtern.
export function bigramPairs(words, tok) {
  const pairs = [];
  for (const w of words) {
    const ids = [BOS, ...tok.encode(w), BOS];
    for (let i = 0; i < ids.length - 1; i++) pairs.push([ids[i], ids[i + 1]]);
  }
  return pairs;
}

export function createBigramNet(vocabSize, rng) {
  // Kleine Startwerte → alle Logits ≈ 0 → Softmax ≈ gleichverteilt
  // → Start-Loss ≈ ln(27) ≈ 3,30. Ein ehrlicher Anfang: "Ich weiß nichts."
  const W = randomMatrix(vocabSize, vocabSize, rng, 0.01);
  return { W, params: W.flat() };
}

//#region bigram-train-step
// Ein Trainingsschritt auf einem zufälligen Mini-Batch von Paaren.
export function trainStep(net, pairs, rng, { batchSize = 256, lr = 30 } = {}) {
  // 1. Forward: Loss für jedes Paar, dann Mittelwert
  const losses = [];
  for (let b = 0; b < batchSize; b++) {
    const [x, y] = pairs[rng.int(pairs.length)];
    const logits = net.W[x];                  // One-Hot · W = Zeile x
    losses.push(crossEntropy(logits, y));
  }
  const loss = Value.sum(losses).mul(1 / batchSize);

  // 2. Backward: alte Gradienten löschen, dann Kettenregel
  for (const p of net.params) p.grad = 0;
  loss.backward();

  // 3. Update: jeden Parameter ein Stück gegen seinen Gradienten schieben
  for (const p of net.params) p.data -= lr * p.grad;

  return loss.data;
}
//#endregion
