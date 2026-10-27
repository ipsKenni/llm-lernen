// Schritt 0 – Tokenisierung und "was kommt als Nächstes?" zum Anfassen.
import { renderTiles, renderProbBars } from '../viz.js';

export function init(root, ctx) {
  const input = root.querySelector('#idee-input');
  const tiles = root.querySelector('#idee-tiles');
  const bars = root.querySelector('#idee-bars');
  const last = root.querySelector('#idee-last');

  function update() {
    const chars = [...input.value.toLowerCase()];
    const ids = chars.map((c) => ctx.tok.stoi.get(c));
    renderTiles(tiles, ['.', ...chars], [0, ...ids], { activeIndex: chars.length });
    const lastId = chars.length ? ids[ids.length - 1] : 0;
    last.textContent = chars.length ? chars[chars.length - 1] : '.';
    if (lastId === undefined) {
      const p = document.createElement('p');
      p.className = 'hint';
      p.textContent = `„${chars[chars.length - 1]}“ kennt das Modell nicht. Es hat nur die Kleinbuchstaben a bis z gesehen.`;
      bars.replaceChildren(p);
      return;
    }
    renderProbBars(bars, ctx.P[lastId], ctx.tok.itos, { topK: 8 });
  }

  input.addEventListener('input', update);
  update();
}
