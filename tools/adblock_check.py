#!/usr/bin/env python3
"""Prüft, ob Werbeblocker Dateien der Seite sperren würden.

Aufruf:  python3 tools/adblock_check.py

Lädt EasyList, EasyPrivacy, uBlock-Filter, AdGuard Base und EasyList Germany
und prüft jede Datei unter public/ als https://llm.ips.codes/<pfad> gegen alle
Netzwerkregeln. Hintergrund: Am 2026-10-27 hat die EasyList-Regel
"/gpt.js$script" die ganze Seite lahmgelegt, weil unser Modell gpt.js hieß.
"""
import glob
import os
import re
import sys
import urllib.request

LISTS = [
    'https://easylist.to/easylist/easylist.txt',
    'https://easylist.to/easylist/easyprivacy.txt',
    'https://ublockorigin.github.io/uAssets/filters/filters.txt',
    'https://filters.adtidy.org/extension/ublock/filters/2.txt',
    'https://easylist.to/easylistgermany/easylistgermany.txt',
]
SITE = 'https://llm.ips.codes/'
OUR_DOMAINS = ('ips.codes', 'llm.ips.codes')
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'public')


def to_regex(pattern):
    """Netzwerkregel im Adblock-Format → regulärer Ausdruck."""
    if pattern.startswith('/') and pattern.endswith('/') and len(pattern) > 2:
        return pattern[1:-1]
    out = ''
    if pattern.startswith('||'):
        out, pattern = r'^[a-z-]+://([^/]*\.)?', pattern[2:]
    elif pattern.startswith('|'):
        out, pattern = '^', pattern[1:]
    end = ''
    if pattern.endswith('|'):
        end, pattern = '$', pattern[:-1]
    for ch in pattern:
        out += '.*' if ch == '*' else r'(?:[^\w.%-]|$)' if ch == '^' else re.escape(ch)
    return out + end


def rules(text):
    for line in text.splitlines():
        line = line.strip()
        if not line or line[0] in '![' or line.startswith('@@') or re.search(r'#[@?$%]?#|#\+js', line):
            continue
        pattern, options = line, []
        m = re.match(r'^(.*?)\$([^/$]*)$', line)
        if m and not (line.startswith('/') and line.endswith('/')):
            pattern, options = m.group(1), m.group(2).split(',')
        domains = [o.split('=', 1)[1] for o in options if o.startswith(('domain=', 'from='))]
        if domains and not any(d in OUR_DOMAINS for d in domains[0].split('|')):
            continue                     # Regel gilt nur auf fremden Seiten
        if any(o in ('third-party', '3p') for o in options):
            continue                     # unsere Dateien sind first-party
        if pattern in ('', '*', '|', '||'):
            continue
        try:
            yield line, re.compile(to_regex(pattern), re.I)
        except re.error:
            continue


def main():
    files = [p for p in glob.glob(os.path.join(ROOT, '**', '*'), recursive=True) if os.path.isfile(p)]
    urls = [SITE] + [SITE + os.path.relpath(p, ROOT).replace(os.sep, '/') for p in files]
    hits = 0
    for url in LISTS:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (adblock_check.py)'})
        text = urllib.request.urlopen(req, timeout=60).read().decode('utf-8', 'replace')
        for line, rx in rules(text):
            for u in urls:
                if rx.search(u):
                    hits += 1
                    print(f'{url.rsplit("/", 1)[-1]}: {line}  →  {u}')
    print(f'{len(urls)} Adressen gegen {len(LISTS)} Filterlisten geprüft, {hits} Treffer')
    sys.exit(1 if hits else 0)


if __name__ == '__main__':
    main()
