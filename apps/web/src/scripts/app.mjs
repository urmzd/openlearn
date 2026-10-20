import { STORAGE_KEY, emptyState, validateState, mergeState, courseProgress } from '../lib/learning.mjs';
import { enhance, startReader } from './reader.mjs';

let state = emptyState();
try {
  const saved = localStorage.getItem(STORAGE_KEY);
  if (saved) state = validateState(JSON.parse(saved));
} catch { /* Reading remains available when device storage is unavailable. */ }
function persist() {
  try {
    const saved = localStorage.getItem(STORAGE_KEY);
    if (saved) state = mergeState(validateState(JSON.parse(saved)), state);
    localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
  } catch { /* Reading does not depend on saving progress. */ }
}

// A small, local title/subject filter complements full-text search in the header.
const filter = document.querySelector('#book-filter');
filter?.addEventListener('input', () => {
  const words = filter.value.toLocaleLowerCase().match(/[\p{L}\p{N}]+/gu) ?? [];
  let count = 0;
  for (const row of document.querySelectorAll('#book-list [data-course]')) {
    const terms = row.querySelector('.book-link').textContent.toLocaleLowerCase().match(/[\p{L}\p{N}]+/gu) ?? [];
    row.hidden = !words.every((word) => terms.some((term) => term.startsWith(word)));
    if (!row.hidden) count++;
  }
  document.querySelector('#filter-count').textContent = `${count} ${count === 1 ? 'title' : 'titles'}`;
  for (const group of document.querySelectorAll('[data-shelf-group]')) group.hidden = !group.querySelector('[data-course]:not([hidden])');
  document.querySelector('#no-books').hidden = count !== 0;
});

async function init() {
  try {
    const response = await fetch('/app/catalog.json');
    if (!response.ok) throw new Error('Catalog unavailable');
    const library = await response.json();
    const lessons = new Map(library.courses.flatMap((course) => course.lessons.map((lesson) => [lesson.key, lesson])));
    const lesson = [...lessons.values()].find((item) => item.url === location.pathname);
    const course = library.courses.find((item) => item.href === location.pathname)
      ?? library.courses.find((item) => item.id === state.activeCourse && item.lessons.some((item) => item.key === lesson?.key))
      ?? library.courses.find((item) => item.lessons.some((item) => item.key === lesson?.key));
    const opened = (item) => {
      state.lastOpened = { key: item.key, at: Date.now() };
      persist();
    };
    if (course) { state.activeCourse = course.id; persist(); }
    if (lesson && location.pathname !== course?.href) opened(lesson);
    const resume = lessons.get(state.lastOpened?.key);
    const card = document.querySelector('#continue-card');
    if (resume && card) {
      card.href = resume.url;
      document.querySelector('#continue-title').textContent = resume.title;
      document.querySelector('#continue-book').textContent = library.courses.find((item) => item.lessons.some((item) => item.key === resume.key))?.title ?? '';
      card.hidden = false;
    }
    for (const book of library.courses) {
      const reading = courseProgress({ ...book, lessons: book.lessons.filter((item) => item.url !== book.href) }, state);
      if (!reading.next) continue;
      for (const begin of document.querySelectorAll(`[data-begin="${book.id}"]`)) {
        begin.href = reading.next.url;
        if (reading.read) begin.textContent = 'Continue reading';
      }
    }
    startReader({ course, lesson: location.pathname === course?.href ? undefined : lesson,
      markRead: (key) => {
        if (state.entries[key]?.read) return;
        state.entries[key] = { bookmarked: false, note: '', ...state.entries[key], read: true, updatedAt: Date.now() };
        persist();
      }, onSection: opened,
    });
  } catch { enhance(document.querySelector('.sl-markdown-content') ?? document.createElement('div')); }
}
init();
