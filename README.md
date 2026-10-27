# Das kleinste LLM

Interaktive Lernseite: Ein Sprachmodell von Grund auf verstehen und ein eigenes Llama-Modell trainieren,
das man herunterlädt und in [Ollama](https://ollama.com) benutzt. Alles läuft im Browser, ohne
Bibliotheken.

**Live:** https://llm.ips.codes

## Aufbau

**Teil 1 – Verstehen** (Schritte 0–6): Bigram durch Zählen, Autograd mit einzelnen Zahlen
(`Value`-Objekte), neuronales Bigram, Attention, ein Mini-GPT mit ~4.000 Parametern.

**Teil 2 – Dein eigenes Modell** (ab Schritt 7): Byte-Level-BPE-Tokenizer, Llama-Architektur (RoPE,
Grouped-Query-Attention, SwiGLU, RMSNorm, Weight Tying), Training auf der Grafikkarte per WebGPU mit
CPU-Fallback, AdamW mit Warmup, Kosinus-Abfall und Gradient Clipping, Export als GGUF + Modelfile.

```
public/                     wird 1:1 als statische Seite ausgeliefert
  index.html, styles.css
  data/                     Trainingsdaten (Quellen und Lizenzen: data/LIZENZEN.md)
  js/
    value.js                Teil 1: Skalar-Autograd
    transformer.js          Teil 1: Mini-GPT aus Value-Objekten
    bigram-*.js, nn.js, tokenizer.js, rng.js, viz.js, code-view.js
    train-worker.js, worker-client.js      Teil 1: Training im Web Worker
    llama/
      bpe.js                Byte-Level-BPE (kompatibel zu llama.cpp "gpt2")
      config.js             Größen-Presets und Tensor-Layout (GGUF-Namen)
      model.js              Vorwärts, Rückwärts, Lernschritt
      backend-cpu.js        alle Operationen in JavaScript (Referenz)
      backend-webgpu.js     dieselben Operationen als WGSL-Shader
      infer.js              Generierung mit KV-Cache, Sampling
      gguf.js               GGUF-Writer/-Reader, Modelfile
      data.js, checkpoint.js, trainer-worker.js, trainer-client.js
    steps/                  Logik der einzelnen Seiten-Schritte
tests/                      Node-Tests und Browser-/Ollama-Prüfskripte
tools/                      Datenaufbereitung, Werbeblocker-Check
```

Die Code-Ausschnitte auf der Seite werden zur Laufzeit aus den echten Dateien geladen. Markiert sind sie
mit `//#region name` … `//#endregion`. Diese Marker beim Umbauen bitte erhalten.

## Lokal starten

Es gibt keinen Build-Schritt. Jeder statische Server für `public/` reicht, zum Beispiel:

```sh
npx wrangler dev          # wie in Produktion (Cloudflare Workers Assets)
python3 -m http.server -d public 8000
```

WebGPU braucht einen sicheren Kontext (`https://` oder `localhost`).

## Tests

```sh
npm test                                      # Autograd, Gradient-Check, BPE, Modell, GGUF (Node ≥ 22)
python3 tools/adblock_check.py                # würden Werbeblocker eine unserer Dateien sperren?
node tests/webgpu-check.mjs <playwright-core> # WebGPU-Shader gegen CPU-Referenz (Headless-Chromium)
node tests/e2e-ollama.mjs <ordner>            # Testmodelle trainieren + exportieren
node tests/e2e-gguf-compare.mjs <datei.gguf> <modell> <ollama-url> "Es war einmal"
```

Der Ollama-Vergleich erwartet bei Temperatur 0 tokengenau dieselbe Ausgabe wie die eigene Inferenz.
Ollama bricht ab, wenn rund 30-mal hintereinander dasselbe Token kommt. Das Skript wertet das als
„gleich bis Abbruch“.

Warum es den Werbeblocker-Check gibt: EasyList sperrt jede Datei `…/gpt.js` (Googles Werbe-Skript heißt so).
Das hat die Seite einmal komplett lahmgelegt.

## Daten neu aufbereiten

```sh
python3 tools/prepare_data.py <ordner-mit-rohdaten>
```

Erwartet `grimm-77905.txt` (Gutenberg #77905) und `tinystories-valid.txt` (Hugging Face).

## Deploy

```sh
npm run deploy            # wrangler deploy → llm.ips.codes
```

## Credits

Aufbau inspiriert von Andrej Karpathys micrograd, makemore, microgpt und llama2.c. Datensätze und
Schrift: siehe `public/data/LIZENZEN.md` und `public/fonts/OFL.txt`.
