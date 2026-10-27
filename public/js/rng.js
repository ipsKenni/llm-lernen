// =====================================================================
// rng.js – Zufall, aber reproduzierbar
// =====================================================================
//
// Math.random() lässt sich nicht "seeden". Für Lernzwecke wollen wir aber
// bei gleichem Seed exakt dieselben Startgewichte und Stichproben – sonst
// kann man Experimente nicht vergleichen.

// mulberry32: ein winziger, guter Pseudozufallsgenerator (32 Bit Zustand).
export function makeRng(seed = 42) {
  let a = seed >>> 0;
  const random = () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };

  // Normalverteilung (Mittelwert 0, Standardabweichung std) per Box-Muller.
  const gauss = (std = 1) => {
    const u = 1 - random();
    const v = random();
    return std * Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  };

  //#region sample
  // Ziehe einen Index gemäß einer Wahrscheinlichkeitsverteilung.
  // Bild: Alle Wahrscheinlichkeiten liegen nebeneinander auf einem
  // Zahlenstrahl von 0 bis 1. Wir werfen einen Pfeil (random()) und
  // schauen, in welchem Abschnitt er landet.
  const sample = (probs) => {
    let r = random();
    for (let i = 0; i < probs.length; i++) {
      r -= probs[i];
      if (r <= 0) return i;
    }
    return probs.length - 1;  // Rundungsfehler abfangen
  };
  //#endregion

  const int = (n) => Math.floor(random() * n);

  return { random, gauss, sample, int };
}
