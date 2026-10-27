#!/usr/bin/env python3
"""Bereitet die Trainingsdaten für Teil 2 auf.

Aufruf:  python3 tools/prepare_data.py <rohdaten-ordner>

Erwartet im Rohdaten-Ordner:
  grimm-77905.txt        https://www.gutenberg.org/ebooks/77905.txt.utf-8
  tinystories-valid.txt  https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-valid.txt

Schreibt nach public/data/:
  maerchen.txt     Grimms Märchen, ein Märchen pro Dokument
  tinystories.txt  ~5 MB TinyStories (CDLA-Sharing-1.0), eine Geschichte pro Dokument
Dokumente sind durch eine Zeile <|endoftext|> getrennt.
"""
import re
import sys
from pathlib import Path

SEP = '\n<|endoftext|>\n'
# Der Titel ist im Original dreispaltig gesetzt.
TITLE_FIXES = {'Tischchen Goldesel, Knüppel aus deck dich, und dem Sack': 'Tischchen deck dich, Goldesel und Knüppel aus dem Sack'}
OUT = Path(__file__).resolve().parent.parent / 'public' / 'data'
TINYSTORIES_BYTES = 5_000_000


def prepare_grimm(raw: str) -> list[str]:
    lines = raw.replace('\r', '').split('\n')
    start = next(i for i, l in enumerate(lines) if l.startswith('*** START OF'))
    end = next(i for i, l in enumerate(lines) if l.strip() == 'Inhalt' and i > start)
    lines = lines[start + 1:end]
    # Vorspann (Titelei) überspringen: das erste Märchen ist "Marienkind".
    first = next(i for i, l in enumerate(lines) if l.strip() == 'Marienkind')
    lines = lines[first:]

    docs, title, paras, current = [], None, [], []
    expect_title = True   # Titel stehen immer direkt nach einer [Illustration]

    def flush_para():
        if current:
            paras.append(' '.join(current))
            current.clear()

    def flush_doc():
        flush_para()
        if title and paras:
            docs.append(TITLE_FIXES.get(title, title) + '\n' + '\n'.join(paras))

    for line in lines:
        text = line.strip().replace('~', '').replace('·', ' ')
        if text.startswith('[Illustration'):
            expect_title = True
            continue
        indent = len(line) - len(line.lstrip(' '))
        if expect_title and text and indent >= 8 and not re.match(r'^\w+es Märchen\.$', text):
            if title is not None and not paras and not current:  # mehrzeiliger Titel
                title = f'{title} {text}'
            else:
                flush_doc()
                title, paras = text, []
            title = re.sub(r'\s+', ' ', title)
            continue
        if text:
            expect_title = False
            current.append(text)
        else:
            flush_para()
    flush_doc()
    return docs


ASCII_MAP = {'“': '"', '”': '"', '„': '"', '‘': "'", '’': "'", '…': '...', '–': '-', '—': '-', ' ': ' '}


def prepare_tinystories(raw: str) -> list[str]:
    stories = raw.split('<|endoftext|>')[1:]  # erste Geschichte ist abgeschnitten
    docs, size = [], 0
    for s in stories:
        for a, b in ASCII_MAP.items():
            s = s.replace(a, b)
        s = '\n'.join(l.strip() for l in s.strip().split('\n') if l.strip())
        if not s or not s.isascii():
            continue
        docs.append(s)
        size += len(s) + len(SEP)
        if size >= TINYSTORIES_BYTES:
            break
    return docs


def main():
    raw_dir = Path(sys.argv[1])
    grimm = prepare_grimm((raw_dir / 'grimm-77905.txt').read_text(encoding='utf-8'))
    tiny = prepare_tinystories((raw_dir / 'tinystories-valid.txt').read_text(encoding='utf-8'))
    for name, docs in (('maerchen.txt', grimm), ('tinystories.txt', tiny)):
        text = SEP.join(docs) + '\n'
        (OUT / name).write_text(text, encoding='utf-8')
        print(f'{name}: {len(docs)} Dokumente, {len(text.encode()) / 1e6:.2f} MB')


if __name__ == '__main__':
    main()
