import { createStore, entries, set, del, clear } from "idb-keyval";

const store = createStore("krishibee", "photos");

export async function listPhotos() {
  const all = await entries(store);
  return all
    .map(([id, record]) => ({ id, ...record }))
    .sort((a, b) => b.takenAt.localeCompare(a.takenAt));
}

export function savePhoto(record) {
  if (navigator.storage && navigator.storage.persist) {
    navigator.storage.persist();
  }
  const id = `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
  return set(id, record, store);
}

export function deletePhoto(id) {
  return del(id, store);
}

export function clearPhotos() {
  return clear(store);
}