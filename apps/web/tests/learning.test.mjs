import test from 'node:test';
import assert from 'node:assert/strict';
import { emptyState, validateState, mergeState, courseProgress, isCurrentDownload } from '../src/lib/learning.mjs';
const note = (updatedAt, changes = {}) => ({ read: false, bookmarked: false, note: '', updatedAt, ...changes });

test('initial offline state cannot advertise or remove a download that does not exist', () => {
  assert.equal(isCurrentDownload(undefined, undefined), false);
  assert.equal(isCurrentDownload(undefined, 'new'), false);
  assert.equal(isCurrentDownload({ build: 'old' }, 'new'), false);
  assert.equal(isCurrentDownload({ build: 'new' }, 'new'), true);
});

test('a backup preserves newer notes while adding progress from another device', () => {
  const a = emptyState(); a.entries.lesson = note(20, { note: 'Keep this reasoning', read: true });
  const b = emptyState(); b.entries.lesson = note(10, { note: 'Old note' }); b.entries.other = note(30, { bookmarked: true });
  const result = mergeState(a, b);
  assert.equal(result.entries.lesson.note, 'Keep this reasoning');
  assert.equal(result.entries.lesson.read, true);
  assert.equal(result.entries.other.bookmarked, true);
  assert.equal(a.entries.other, undefined);
});
test('malformed or oversized note backups do not overwrite valid progress', () => {
  for (const data of [null, {}, { ...emptyState(), version: 5 }, { ...emptyState(), entries: [] }, { ...emptyState(), entries: { x: note(NaN) } }, { ...emptyState(), entries: { x: note(1, { note: 'x'.repeat(20001) }) } }, JSON.parse('{"format":"openlearn-progress","version":1,"entries":{"__proto__":{"read":false,"bookmarked":false,"note":"","updatedAt":1}}}')]) assert.throws(() => validateState(data));
});
test('browser completion is reading progress and does not import grading claims', () => {
  const data = emptyState(); data.entries.lesson = note(1, { read: true, verified: true, verdict: 'pass' });
  assert.deepEqual(validateState(data).entries.lesson, note(1, { read: true }));
});
test('continue reading follows lesson order and does not skip an unread lesson', () => {
  const course = { lessons: [{ key: 'a', url: '/a/' }, { key: 'b', url: '/b/' }] };
  const state = emptyState(); state.entries.a = note(1, { read: true });
  assert.deepEqual(courseProgress(course, state), { read: 1, total: 2, next: course.lessons[1] });
  state.entries.a.read = false; state.lastOpened = { key: 'b', at: 10 };
  assert.equal(courseProgress(course, state).next.key, 'b');
});

test('shared readings do not implicitly start every course that references them', () => {
  const a = emptyState(); a.enrolled = ['mathematics'];
  const b = emptyState(); b.enrolled = ['systems'];
  assert.deepEqual(mergeState(a, b).enrolled.sort(), ['mathematics', 'systems']);
  const legacy = { format: 'openlearn-progress', version: 1, entries: { shared: note(1, { read: true }) } };
  assert.deepEqual(validateState(legacy).enrolled, []);
});
