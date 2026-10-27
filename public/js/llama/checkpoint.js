// =====================================================================
// checkpoint.js – Trainingsstand im Browser sichern (IndexedDB)
// =====================================================================
// Gewichte, Adam-Momente, Tokenizer und Verlauf landen lokal in deinem
// Browser. So überlebt das Training ein Neuladen der Seite und kann
// weiterlaufen. Nichts davon verlässt deinen Rechner.

const DB = 'kleinstes-llm';
const STORE = 'checkpoints';

function open() {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB, 1);
    req.onupgradeneeded = () => req.result.createObjectStore(STORE);
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

async function tx(mode, fn) {
  const db = await open();
  return new Promise((resolve, reject) => {
    const t = db.transaction(STORE, mode);
    const req = fn(t.objectStore(STORE));
    t.oncomplete = () => { db.close(); resolve(req?.result); };
    t.onerror = () => { db.close(); reject(t.error); };
    t.onabort = () => { db.close(); reject(t.error); };
  });
}

export const saveCheckpoint = (key, value) => tx('readwrite', (s) => s.put(value, key)).catch(() => false);
export const loadCheckpoint = (key) => tx('readonly', (s) => s.get(key)).catch(() => null);
export const deleteCheckpoint = (key) => tx('readwrite', (s) => s.delete(key)).catch(() => false);
