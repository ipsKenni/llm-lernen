# Datensätze und ihre Lizenzen

| Datei | Inhalt | Quelle | Lizenz |
|---|---|---|---|
| `names.txt` | 32.033 englische Vornamen | [karpathy/makemore](https://github.com/karpathy/makemore) | MIT, siehe `LICENSE-names.txt` |
| `maerchen.txt` | 62 Märchen aus „Deutsche Märchen gesammelt durch die Brüder Grimm“ (hg. M. Thilo-Luyken, 1921) | [Projekt Gutenberg #77905](https://www.gutenberg.org/ebooks/77905) | gemeinfrei; Gutenberg-Kopf und -Fuß entfernt |
| `tinystories.txt` | 6.014 Geschichten aus `TinyStories-valid.txt`, Anführungszeichen auf ASCII vereinheitlicht | [roneneldan/TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories), Ronen Eldan und Yuanzhi Li | [Community Data License Agreement – Sharing 1.0](https://cdla.dev/sharing-1-0/) |
| `staedte.txt` | 250 deutsche Städtenamen | eigene Zusammenstellung | frei verwendbar |

`maerchen.txt` und `tinystories.txt` werden mit `tools/prepare_data.py` aus den Originaldateien erzeugt.
Die geänderte TinyStories-Teilmenge wird unter denselben Bedingungen (CDLA-Sharing-1.0) weitergegeben.

Die Schrift Atkinson Hyperlegible Next/Mono steht unter der SIL Open Font License 1.1, siehe `../fonts/OFL.txt`.
