export const STORAGE_KEY = 'openlearn.learning.v1';
export const emptyState = () => ({ format: 'openlearn-progress', version: 1, entries: {}, enrolled: [], activeCourse: null, lastOpened: null });
const safeKey = (key) => typeof key === 'string' && key.length < 700 && !['__proto__', 'constructor', 'prototype'].includes(key);

export function validateState(data) {
  if (!data || data.format !== 'openlearn-progress' || data.version !== 1 || !data.entries || Array.isArray(data.entries) || typeof data.entries !== 'object') throw new Error('Choose an OpenLearn progress backup.');
  const pairs = Object.entries(data.entries);
  if (pairs.length > 6000) throw new Error('This backup contains too many lessons.');
  const result = emptyState();
  for (const [key, value] of pairs) {
    if (!safeKey(key) || !value || typeof value.read !== 'boolean' || typeof value.bookmarked !== 'boolean' || typeof value.note !== 'string' || value.note.length > 20000 || !Number.isFinite(value.updatedAt) || value.updatedAt < 0) throw new Error('This backup contains an invalid lesson.');
    result.entries[key] = { read: value.read, bookmarked: value.bookmarked, note: value.note, updatedAt: value.updatedAt };
  }
  if (data.enrolled !== undefined && (!Array.isArray(data.enrolled) || data.enrolled.length > 1000 || data.enrolled.some((id) => typeof id !== 'string' || !/^[a-z0-9-]+$/.test(id)))) throw new Error('This backup contains invalid course selections.');
  result.enrolled = [...new Set(data.enrolled ?? [])];
  if (typeof data.activeCourse === 'string' && /^[a-z0-9-]+$/.test(data.activeCourse)) result.activeCourse = data.activeCourse;
  const last = data.lastOpened;
  if (last && safeKey(last.key) && Number.isFinite(last.at) && last.at >= 0) result.lastOpened = { key: last.key, at: last.at };
  return result;
}

export function mergeState(current, incoming) {
  const next = validateState(current);
  const other = validateState(incoming);
  for (const [key, entry] of Object.entries(other.entries)) {
    if (!next.entries[key] || entry.updatedAt > next.entries[key].updatedAt) next.entries[key] = entry;
  }
  next.enrolled = [...new Set([...next.enrolled, ...other.enrolled])];
  next.activeCourse = other.activeCourse ?? next.activeCourse;
  if (other.lastOpened && (!next.lastOpened || other.lastOpened.at > next.lastOpened.at)) next.lastOpened = other.lastOpened;
  return next;
}

export function courseProgress(course, state) {
  const lessons = course.lessons;
  const read = lessons.filter((l) => state.entries[l.key]?.read).length;
  const last = lessons.find((l) => l.key === state.lastOpened?.key);
  return { read, total: lessons.length, next: last && !state.entries[last.key]?.read ? last : lessons.find((l) => !state.entries[l.key]?.read) ?? lessons[0] };
}
export function isCurrentDownload(saved, build) {
  return Boolean(saved && typeof build === 'string' && saved.build === build);
}
