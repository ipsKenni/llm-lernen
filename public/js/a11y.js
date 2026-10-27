// =====================================================================
// a11y.js – Tastaturbedienung für alle Auswahlgruppen (role="radiogroup")
// =====================================================================
// Muster aus den ARIA Authoring Practices: Nur die gewählte Option ist per
// Tab erreichbar; Pfeiltasten wechseln die Auswahl, Pos1/Ende springen an
// den Anfang bzw. das Ende. Die Optionen werden von den Schritten dynamisch
// erzeugt – ein MutationObserver hält die Tab-Reihenfolge aktuell.

const options = (group) => [...group.querySelectorAll('[role="radio"]')].filter((o) => !o.disabled);

function syncTabindex(group) {
  const all = [...group.querySelectorAll('[role="radio"]')];
  const usable = all.filter((o) => !o.disabled);
  const current = usable.find((o) => o.getAttribute('aria-checked') === 'true') || usable[0];
  for (const o of all) o.tabIndex = o === current ? 0 : -1;
}

function onKey(e) {
  const group = e.currentTarget;
  const list = options(group);
  const i = list.indexOf(document.activeElement);
  if (i < 0) return;
  const keys = { ArrowRight: i + 1, ArrowDown: i + 1, ArrowLeft: i - 1, ArrowUp: i - 1, Home: 0, End: list.length - 1 };
  if (!(e.key in keys)) return;
  e.preventDefault();
  const next = list[(keys[e.key] + list.length) % list.length];
  next.focus();
  next.click();   // Auswahl folgt dem Fokus
}

export function enhanceRadioGroups(root = document) {
  const wire = (group) => {
    if (group.dataset.a11y) return;
    group.dataset.a11y = '1';
    group.addEventListener('keydown', onKey);
    new MutationObserver(() => syncTabindex(group)).observe(group, { subtree: true, childList: true, attributes: true, attributeFilter: ['aria-checked', 'disabled'] });
    syncTabindex(group);
  };
  root.querySelectorAll('[role="radiogroup"]').forEach(wire);
}
