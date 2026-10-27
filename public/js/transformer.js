// =====================================================================
// transformer.js – Ein vollständiges GPT, gebaut nur aus unserer Value-Klasse
// =====================================================================
//
// Das hier ist dieselbe Architektur wie GPT-2, GPT-4, Claude & Co. –
// nur winzig: ~4.000 statt Milliarden Parameter, Zeichen statt BPE-Tokens,
// einzelne Zahlen statt Tensoren auf der GPU.
//
// Ablauf für jedes Token (= jede Position im Text):
//
//   Token-ID ──► Token-Embedding ─┐
//   Position ──► Pos-Embedding ───┴─► + ─► RMSNorm
//        │
//        ▼  (n_layer mal)
//   ┌── RMSNorm ─► Attention (schaut auf alle VORHERIGEN Positionen) ─► + ◄─┐ Residual
//   └── RMSNorm ─► MLP (denkt über jede Position einzeln nach)       ─► + ◄─┘ Residual
//        │
//        ▼
//   lm_head ─► Logits (eine Zahl pro möglichem nächsten Token) ─► Softmax
//
// Inspiriert von Andrej Karpathys microgpt / nanoGPT.

import { Value } from './value.js';
import { BOS } from './tokenizer.js';
import { softmax, crossEntropy, randomMatrix, softmaxNumbers } from './nn.js';

export const DEFAULT_CONFIG = {
  nEmbd: 16,      // Breite jedes Vektors ("wie viele Zahlen beschreiben ein Token")
  nHead: 4,       // Anzahl Attention-Köpfe (nEmbd muss durch nHead teilbar sein)
  nLayer: 1,      // Anzahl Transformer-Blöcke übereinander
  blockSize: 16,  // maximale Kontextlänge in Tokens
  batchSize: 4,   // Wörter pro Trainingsschritt
  lr: 0.01,       // Lernrate (Adam)
  steps: 1000,    // Trainingsschritte insgesamt
  seed: 42,
};

// Anzahl Parameter, ohne das Modell zu bauen (für die Anzeige).
export function countParams(cfg, vocabSize) {
  const C = cfg.nEmbd;
  return 2 * vocabSize * C + cfg.blockSize * C + cfg.nLayer * 12 * C * C;
}

//#region gpt-params
// Alle lernbaren Parameter. Am Anfang reiner Zufall (klein, std 0,08).
// Alles Wissen, das das Modell je haben wird, steckt am Ende in genau
// diesen Zahlen.
export function createGPT(cfg, vocabSize, rng) {
  const C = cfg.nEmbd;
  const std = 0.08;
  const sd = {
    wte: randomMatrix(vocabSize, C, rng, std),     // Token-Embedding: Zeile i = Vektor für Token i
    wpe: randomMatrix(cfg.blockSize, C, rng, std), // Positions-Embedding: Zeile p = Vektor für Position p
    layers: [],
    lmHead: randomMatrix(vocabSize, C, rng, std),  // Vektor → Logits für jedes Token
  };
  for (let l = 0; l < cfg.nLayer; l++) {
    sd.layers.push({
      wq: randomMatrix(C, C, rng, std),            // Query: "Wonach suche ich?"
      wk: randomMatrix(C, C, rng, std),            // Key:   "Was biete ich an?"
      wv: randomMatrix(C, C, rng, std),            // Value: "Was gebe ich weiter, wenn ich gewählt werde?"
      wo: randomMatrix(C, C, rng, std),            // mischt die Köpfe wieder zusammen
      fc1: randomMatrix(4 * C, C, rng, std),       // MLP: auf 4× Breite aufblähen …
      fc2: randomMatrix(C, 4 * C, rng, std),       // … und wieder zurück
    });
  }
  const params = [
    ...sd.wte.flat(), ...sd.wpe.flat(),
    ...sd.layers.flatMap((L) => [L.wq, L.wk, L.wv, L.wo, L.fc1, L.fc2].flatMap((m) => m.flat())),
    ...sd.lmHead.flat(),
  ];
  return { cfg, vocabSize, sd, params };
}
//#endregion

//#region gpt-linear
// Lineare Schicht = Matrix mal Vektor. Jede Ausgabe ist eine gewichtete
// Summe aller Eingaben:  out_i = Σ_j W[i][j] · x[j]
// Das ist die Operation, mit der echte LLMs >90 % ihrer Rechenzeit verbringen.
function linear(x, W) {
  return W.map((row) => Value.sum(row.map((w, j) => w.mul(x[j]))));
}
//#endregion

//#region gpt-rmsnorm
// RMSNorm: skaliert den Vektor so, dass seine Zahlen im Mittel die Größe
// 1 haben. Hält die Werte über viele Schichten hinweg stabil, damit sie
// weder explodieren noch verschwinden.
//     x_i / sqrt( mean(x²) + ε )
function rmsnorm(x) {
  const meanSquare = Value.sum(x.map((xi) => xi.mul(xi))).mul(1 / x.length);
  const scale = meanSquare.add(1e-5).pow(-0.5);
  return x.map((xi) => xi.mul(scale));
}
//#endregion

export function newCache(cfg) {
  return {
    keys: Array.from({ length: cfg.nLayer }, () => []),
    values: Array.from({ length: cfg.nLayer }, () => []),
  };
}

// -------------------------------------------------------------------
// Ein Forward-Schritt für EIN Token an Position `pos`.
//
// `cache` sammelt die Keys und Values aller bisherigen Positionen. Da wir
// Token für Token von links nach rechts gehen, kann die Attention nur auf
// die Vergangenheit schauen – das ist die "kausale Maske" von GPT, hier
// ganz natürlich gelöst. (Echte LLMs nennen das den KV-Cache.)
//
// `trace` (optional) sammelt die Attention-Gewichte für die Anzeige.
// -------------------------------------------------------------------
export function gptStep(model, tokenId, pos, cache, trace = null) {
  const { sd, cfg } = model;
  const nHead = cfg.nHead;
  const headDim = cfg.nEmbd / nHead;

  //#region gpt-embedding
  // 1) Embedding: Token-Vektor + Positions-Vektor.
  //    Das Modell weiß so, WELCHES Zeichen es ist und WO es steht.
  let x = sd.wte[tokenId].map((t, i) => t.add(sd.wpe[pos][i]));
  x = rmsnorm(x);
  //#endregion

  for (let l = 0; l < cfg.nLayer; l++) {
    const L = sd.layers[l];

    //#region gpt-attention
    // 2) Attention: Tokens tauschen Informationen aus.
    let residual = x;
    x = rmsnorm(x);
    const q = linear(x, L.wq);   // Was sucht diese Position?
    const k = linear(x, L.wk);   // Was bietet diese Position an?
    const v = linear(x, L.wv);   // Was gibt sie weiter?
    cache.keys[l].push(k);
    cache.values[l].push(v);

    const heads = [];
    for (let h = 0; h < nHead; h++) {
      // Jeder Kopf arbeitet auf seinem eigenen Ausschnitt des Vektors.
      const s = h * headDim;
      const qh = q.slice(s, s + headDim);
      // Score für jede bisherige Position t: Skalarprodukt q·k_t.
      // Passen Frage und Angebot zusammen → großer Score.
      // Geteilt durch sqrt(headDim), damit die Scores nicht zu groß werden.
      const scores = cache.keys[l].map((kt) =>
        Value.sum(qh.map((qi, j) => qi.mul(kt[s + j]))).mul(1 / Math.sqrt(headDim)),
      );
      // Softmax → Gewichte, die sich zu 1 addieren: "wie viel Aufmerksamkeit
      // bekommt jede vorherige Position?"
      const weights = softmax(scores);
      if (trace) trace.attn[l][h].push(weights.map((w) => w.data));
      // Ausgabe = gewichteter Mittelwert der Values.
      for (let j = 0; j < headDim; j++) {
        heads.push(Value.sum(weights.map((w, t) => w.mul(cache.values[l][t][s + j]))));
      }
    }
    // Köpfe zusammenmischen und auf den Residual-Strom addieren.
    x = linear(heads, L.wo).map((xi, i) => xi.add(residual[i]));
    //#endregion

    //#region gpt-mlp
    // 3) MLP: jede Position "denkt" für sich über das Gesammelte nach.
    //    Aufblähen auf 4×nEmbd, ReLU (Nichtlinearität!), wieder verkleinern.
    residual = x;
    x = rmsnorm(x);
    x = linear(x, L.fc1).map((xi) => xi.relu());
    x = linear(x, L.fc2).map((xi, i) => xi.add(residual[i]));
    //#endregion
  }

  //#region gpt-head
  // 4) lm_head: Vektor → eine Zahl (Logit) pro Token im Vokabular.
  return linear(x, sd.lmHead);
  //#endregion
}

//#region gpt-loss
// Loss für ein Wort: An jeder Position das nächste Zeichen vorhersagen.
//   ".emma." → Eingabe '.' Ziel 'e', Eingabe 'e' Ziel 'm', … Eingabe 'a' Ziel '.'
// Ein Wort liefert also gleich mehrere Trainingsbeispiele.
export function wordLosses(model, ids) {
  const tokens = [BOS, ...ids, BOS];
  const n = Math.min(model.cfg.blockSize, tokens.length - 1);
  const cache = newCache(model.cfg);
  const losses = [];
  for (let pos = 0; pos < n; pos++) {
    const logits = gptStep(model, tokens[pos], pos, cache);
    losses.push(crossEntropy(logits, tokens[pos + 1]));
  }
  return losses;
}
//#endregion

//#region gpt-adam
// Adam: Gradientenabstieg mit Gedächtnis.
//   m = gleitender Mittelwert der Gradienten     ("in welche Richtung geht's?")
//   v = gleitender Mittelwert der Gradienten²    ("wie wackelig ist das?")
// Update: p ← p − lr · m / (sqrt(v) + ε)
// Parameter mit ruhigen Gradienten machen große Schritte, wackelige kleine.
export function createAdam(nParams, { beta1 = 0.85, beta2 = 0.99, eps = 1e-8 } = {}) {
  return { m: new Float64Array(nParams), v: new Float64Array(nParams), t: 0, beta1, beta2, eps };
}

function adamUpdate(params, opt, lr) {
  opt.t += 1;
  const { m, v, beta1, beta2, eps, t } = opt;
  for (let i = 0; i < params.length; i++) {
    const g = params[i].grad;
    m[i] = beta1 * m[i] + (1 - beta1) * g;
    v[i] = beta2 * v[i] + (1 - beta2) * g * g;
    const mHat = m[i] / (1 - beta1 ** t);  // Korrektur: m und v starten bei 0
    const vHat = v[i] / (1 - beta2 ** t);
    params[i].data -= lr * mHat / (Math.sqrt(vHat) + eps);
  }
}
//#endregion

//#region gpt-train-step
// Ein kompletter Trainingsschritt: Forward → Loss → Backward → Update.
export function trainStep(model, opt, batch, lr) {
  const losses = batch.flatMap((ids) => wordLosses(model, ids));
  const loss = Value.sum(losses).mul(1 / losses.length);
  for (const p of model.params) p.grad = 0;
  loss.backward();
  adamUpdate(model.params, opt, lr);
  return loss.data;
}
//#endregion

//#region gpt-generate
// Generieren: Mit '.' starten, Logits berechnen, mit Temperatur zu
// Wahrscheinlichkeiten machen, eins ziehen, als nächste Eingabe benutzen.
// Genau so "schreibt" jedes LLM – Token für Token.
export function generate(model, rng, { temperature = 1, prefix = [] } = {}) {
  const cache = newCache(model.cfg);
  const tokens = [BOS, ...prefix];
  let logits;
  for (let pos = 0; pos < tokens.length; pos++) {
    logits = gptStep(model, tokens[pos], pos, cache);
  }
  while (tokens.length < model.cfg.blockSize) {
    const probs = softmaxNumbers(logits.map((l) => l.data), temperature);
    const next = rng.sample(probs);
    if (next === BOS) break;
    tokens.push(next);
    logits = gptStep(model, next, tokens.length - 1, cache);
  }
  return tokens.slice(1);
}
//#endregion

// Für die Anzeige: Wahrscheinlichkeiten fürs nächste Token und alle
// Attention-Gewichte für eine Eingabe.
export function inspect(model, prefix, temperature = 1) {
  const cfg = model.cfg;
  const tokens = [BOS, ...prefix].slice(0, cfg.blockSize);
  const cache = newCache(cfg);
  const trace = {
    attn: Array.from({ length: cfg.nLayer }, () => Array.from({ length: cfg.nHead }, () => [])),
  };
  let logits;
  for (let pos = 0; pos < tokens.length; pos++) {
    logits = gptStep(model, tokens[pos], pos, cache, trace);
  }
  return { tokens, probs: softmaxNumbers(logits.map((l) => l.data), temperature), attn: trace.attn };
}

export function exportWeights(model, itos) {
  return { cfg: model.cfg, itos, weights: model.params.map((p) => p.data) };
}

export function importWeights(model, weights) {
  model.params.forEach((p, i) => { p.data = weights[i]; });
}
