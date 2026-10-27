// =====================================================================
// value.js – Autograd von Hand: eine Zahl, die sich merkt, woher sie kommt
// =====================================================================
//
// Ein neuronales Netz ist nichts anderes als eine riesige Formel aus
// Plus, Mal, exp, log … Um es zu trainieren, brauchen wir für JEDEN
// Parameter die Ableitung des Fehlers (Loss) nach diesem Parameter:
//
//     "Wenn ich diesen Parameter ein winziges bisschen erhöhe,
//      wie stark steigt oder fällt dann der Loss?"
//
// Genau das berechnet diese Datei. Jede Zahl wird in ein Value-Objekt
// verpackt. Jede Rechenoperation erzeugt ein neues Value, das sich merkt:
//   1. aus welchen Values es entstanden ist (children)
//   2. die lokale Ableitung nach jedem dieser Kinder (localGrads)
//
// So entsteht beim Rechnen ganz nebenbei ein Graph. backward() läuft
// diesen Graphen rückwärts ab und wendet die Kettenregel an.
// Das ist Backpropagation – nicht mehr und nicht weniger.
//
// Dies ist dieselbe Idee wie in PyTorch, nur dass PyTorch mit ganzen
// Tensoren (Matrizen) rechnet statt mit einzelnen Zahlen. Wir bleiben
// bei einzelnen Zahlen: langsam, aber jede Rechnung ist sichtbar.

//#region value-klasse
export class Value {
  constructor(data, children = [], localGrads = [], op = '') {
    this.data = data;              // der eigentliche Zahlenwert (Forward)
    this.grad = 0;                 // dLoss/dDiesesValue (wird von backward() gefüllt)
    this.children = children;      // aus welchen Values bin ich entstanden?
    this.localGrads = localGrads;  // ∂ich/∂kind – für jedes Kind eine Zahl
    this.op = op;                  // nur zur Anzeige: '+', '*', 'tanh', …
    this.label = '';               // nur zur Anzeige: 'w1', 'x1', …
    this.visited = 0;              // Hilfsmarke für buildTopo()
  }
//#endregion

  // -------------------------------------------------------------------
  // Grundoperationen. Jede liefert ein neues Value und trägt die lokale
  // Ableitung ein. Zahlen (number) werden automatisch zu Values.
  // -------------------------------------------------------------------

  //#region value-ops
  // a + b   →  ∂/∂a = 1,  ∂/∂b = 1
  add(other) {
    other = toValue(other);
    return new Value(this.data + other.data, [this, other], [1, 1], '+');
  }

  // a · b   →  ∂/∂a = b,  ∂/∂b = a
  mul(other) {
    other = toValue(other);
    return new Value(this.data * other.data, [this, other], [other.data, this.data], '*');
  }

  // a^k (k ist eine feste Zahl, kein Value)  →  ∂/∂a = k · a^(k-1)
  pow(k) {
    return new Value(this.data ** k, [this], [k * this.data ** (k - 1)], `^${k}`);
  }

  // e^a  →  ∂/∂a = e^a  (die e-Funktion ist ihre eigene Ableitung)
  exp() {
    const e = Math.exp(this.data);
    return new Value(e, [this], [e], 'exp');
  }

  // ln(a)  →  ∂/∂a = 1/a
  log() {
    return new Value(Math.log(this.data), [this], [1 / this.data], 'log');
  }

  // ReLU: max(0, a)  →  ∂/∂a = 1 wenn a > 0, sonst 0
  relu() {
    return new Value(Math.max(0, this.data), [this], [this.data > 0 ? 1 : 0], 'relu');
  }

  // tanh(a)  →  ∂/∂a = 1 − tanh(a)²
  tanh() {
    const t = Math.tanh(this.data);
    return new Value(t, [this], [1 - t * t], 'tanh');
  }
  //#endregion

  // Abgeleitete Operationen – zusammengesetzt aus den Grundoperationen,
  // brauchen deshalb keine eigene Ableitung.
  neg() { return this.mul(-1); }
  sub(other) { return this.add(toValue(other).neg()); }
  div(other) { return this.mul(toValue(other).pow(-1)); }

  //#region value-sum
  // Summe vieler Values als EIN Knoten (statt einer langen Kette von '+').
  // Mathematisch identisch, aber viel weniger Objekte → schneller.
  // ∂(a1 + a2 + … + an)/∂ai = 1 für jedes i.
  static sum(values) {
    let s = 0;
    for (const v of values) s += v.data;
    return new Value(s, values, values.map(() => 1), 'Σ');
  }
  //#endregion

  //#region value-backward
  // -------------------------------------------------------------------
  // backward(): die Kettenregel, rückwärts durch den ganzen Graphen.
  //
  // Schritt 1: Alle Knoten in "topologischer Reihenfolge" sortieren –
  //            jedes Kind steht vor seinen Eltern.
  // Schritt 2: Beim Ergebnis anfangen (dL/dL = 1) und rückwärts gehen.
  //            Jeder Knoten verteilt seinen Gradienten an seine Kinder:
  //
  //                kind.grad += lokaleAbleitung · knoten.grad
  //
  //            Das "+=" ist wichtig: Wird ein Value an mehreren Stellen
  //            benutzt, addieren sich die Beiträge aller Pfade.
  // -------------------------------------------------------------------
  backward() {
    const topo = buildTopo(this);
    this.grad = 1;
    for (let i = topo.length - 1; i >= 0; i--) {
      const node = topo[i];
      for (let j = 0; j < node.children.length; j++) {
        node.children[j].grad += node.localGrads[j] * node.grad;
      }
    }
  }
  //#endregion
}

export function toValue(x) {
  return x instanceof Value ? x : new Value(x);
}

//#region value-topo
// Topologische Sortierung per Tiefensuche: Ein Knoten landet erst in
// `topo`, wenn alle seine Kinder drin sind.
// Ohne Rekursion (eigener Stack), damit auch Graphen mit hunderttausend
// Knoten keinen Stack-Überlauf erzeugen. Statt eines Set merken wir uns
// "schon besucht" mit einer Stempel-Nummer direkt am Knoten – das ist
// bei 100.000+ Knoten deutlich schneller.
let currentStamp = 0;

export function buildTopo(root) {
  const stamp = ++currentStamp;
  const topo = [];
  const nodes = [root];   // Pfad von der Wurzel zum aktuellen Knoten
  const next = [0];       // welches Kind kommt beim jeweiligen Knoten als Nächstes dran?
  root.visited = stamp;
  while (nodes.length > 0) {
    const top = nodes.length - 1;
    const node = nodes[top];
    const i = next[top];
    if (i < node.children.length) {
      next[top] = i + 1;
      const child = node.children[i];
      if (child.visited !== stamp) {   // Kind noch nicht gesehen → tiefer gehen
        child.visited = stamp;
        nodes.push(child);
        next.push(0);
      }
    } else {                           // alle Kinder erledigt → Knoten fertig
      topo.push(node);
      nodes.pop();
      next.pop();
    }
  }
  return topo;
}
//#endregion
