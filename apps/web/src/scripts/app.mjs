import { STORAGE_KEY, emptyState, validateState, mergeState, courseProgress, isCurrentDownload } from '../lib/learning.mjs';

let state = emptyState();
let library;
let lessonIndex = new Map();
let workerStatus = { saved: [], downloads: [], available: {} };
let workerRegistration;
let installPrompt;
let currentLesson;
let currentCourse;
let filterOffline = false;
let toastTimer;
const $ = (selector) => document.querySelector(selector);
const all = (selector) => [...document.querySelectorAll(selector)];
const byID = (id) => library?.courses.find((c) => c.id === id);
const lessons = () => lessonIndex;
const entry = (key) => state.entries[key] ?? { read: false, bookmarked: false, note: '', updatedAt: 0 };
const size = (bytes) => `${(bytes / 1024 / 1024).toFixed(1)} MB`;

function notify(message, persistent = false) {
  const box = $('#app-message');
  if (!box) return;
  box.textContent = message; box.hidden = false;
  clearTimeout(toastTimer);
  if (!persistent) toastTimer = setTimeout(() => { box.hidden = true; }, 6500);
}
function exportBackup() {
  const blob = new Blob([JSON.stringify(state, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob); const a = link(url, 'Download');
  a.download = `openlearn-progress-${new Date().toISOString().slice(0, 10)}.json`; a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
function load() {
  try { const raw = localStorage.getItem(STORAGE_KEY); if (raw) state = validateState(JSON.parse(raw)); }
  catch { notify('Your saved progress could not be opened. Import a backup from My learning to restore it.', true); }
}
function persist(next, restore = false) {
  try {
    const latest = localStorage.getItem(STORAGE_KEY);
    if (latest) {
      try { next = mergeState(validateState(JSON.parse(latest)), next); }
      catch (error) { if (!restore) throw error; }
    }
    localStorage.setItem(STORAGE_KEY, JSON.stringify(next)); state = next;
    return true;
  } catch {
    state = next; notify('Storage is full or unavailable. Keep this page open and download a progress backup.', true);
    const backup = text('button', 'Export backup'); backup.type = 'button'; backup.addEventListener('click', exportBackup);
    $('#app-message').append(document.createElement('br'), backup);
    return false;
  }
}
function updateEntry(key, changes) {
  const next = structuredClone(state);
  next.entries[key] = { ...entry(key), ...changes, updatedAt: Math.max(Date.now(), entry(key).updatedAt + 1) };
  const saved = persist(next); render(); return saved;
}
function link(url, label, className) {
  const a = document.createElement('a'); a.href = url; a.textContent = label;
  if (className) a.className = className;
  return a;
}
function text(tag, value, className) {
  const el = document.createElement(tag); el.textContent = value;
  if (className) el.className = className;
  return el;
}

async function messageWorker(type, data = {}, onProgress) {
  const worker = navigator.serviceWorker?.controller;
  if (!worker) throw new Error('Offline reading is getting ready. Reconnect and refresh if this message persists.');
  return new Promise((resolve, reject) => {
    const channel = new MessageChannel();
    let timer;
    const reset = () => { clearTimeout(timer); timer = setTimeout(() => { channel.port1.close(); reject(new Error('The offline operation timed out. Check your connection and try again.')); }, 60000); };
    channel.port1.onmessage = (event) => {
      reset();
      if (event.data.type === 'PROGRESS') { onProgress?.(event.data); return; }
      clearTimeout(timer); channel.port1.close();
      if (event.data.type === 'ERROR') reject(new Error(event.data.message));
      else resolve(event.data);
    };
    reset(); worker.postMessage({ type, ...data }, [channel.port2]);
  });
}
async function refreshOffline() {
  if (!navigator.serviceWorker?.controller) return;
  try { workerStatus = await messageWorker('STATUS'); render(); }
  catch (error) { notify(error.message); }
}
function downloadControls() {
  for (const button of all('[data-download]')) {
    const id = button.dataset.download;
    const saved = workerStatus.saved.find((c) => c.id === id);
    const downloading = workerStatus.downloads.find((c) => c.id === id);
    const caption = all('[data-download-state]').find((el) => el.dataset.downloadState === id);
    button.textContent = downloading ? 'Cancel download' : isCurrentDownload(saved, workerStatus.build) ? 'Remove download' : saved ? 'Update offline copy' : 'Save offline';
    button.setAttribute('aria-label', `${button.textContent}: ${byID(id)?.title ?? id}`);
    if (caption) caption.textContent = downloading ? `${downloading.done} of ${downloading.total} pages` : saved ? `${saved.build === workerStatus.build ? 'Saved on this device' : 'An older edition is saved'} · ${size(saved.bytes)}` : workerStatus.available?.[id] ? `${size(workerStatus.available[id].bytes)} download` : 'Offline reading is getting ready';
    button.closest('[data-course]')?.setAttribute('data-offline', saved ? 'true' : 'false');
  }
}
async function download(id) {
  const existing = workerStatus.saved.find((c) => c.id === id);
  try {
    if (workerStatus.downloads.some((c) => c.id === id)) { await messageWorker('CANCEL_DOWNLOAD', { id }); return; }
    if (isCurrentDownload(existing, workerStatus.build)) {
      await messageWorker('REMOVE_COURSE', { id });
      notify('Offline copy removed. Your notes and reading progress are kept.');
    } else {
      workerStatus.downloads.push({ id, done: 0, total: 0 }); render();
      await messageWorker('DOWNLOAD_COURSE', { id }, (event) => {
        const operation = workerStatus.downloads.find((c) => c.id === id);
        if (operation) Object.assign(operation, event);
        downloadControls();
      });
      await navigator.storage?.persist?.();
      notify(`${byID(id).title} is ready for offline reading.`);
    }
  } catch (error) { notify(error.message, true); }
  finally { await refreshOffline(); }
}

function filterCourses() {
  const term = $('#course-filter')?.value.toLowerCase().trim() ?? '';
  let count = 0;
  for (const row of all('#course-library [data-course]')) {
    row.hidden = !row.dataset.search.includes(term) || (filterOffline && !workerStatus.saved.some((c) => c.id === row.dataset.course));
    if (!row.hidden) count++;
  }
  if ($('#course-count')) $('#course-count').textContent = `${count} ${count === 1 ? 'course' : 'courses'}`;
  if ($('#no-courses')) $('#no-courses').hidden = count !== 0;
}
function renderMyLearning() {
  const holder = $('#my-courses'); if (!holder) return;
  holder.replaceChildren();
  let count = 0;
  for (const course of library.courses) {
    const saved = workerStatus.saved.some((c) => c.id === course.id);
    const started = state.enrolled.includes(course.id);
    if (!started && !saved) continue;
    count++;
    const progress = courseProgress(course, state);
    const row = text('article', '', 'learning-course'); row.dataset.course = course.id;
    const heading = document.createElement('h3'); heading.append(link(course.href, course.title)); row.append(heading);
    row.append(text('p', `${progress.read} of ${progress.total} readings complete`, 'course-meta'));
    if (progress.next) row.append(link(progress.next.url, progress.read ? 'Continue learning' : 'Start reading', 'course-open'));
    const button = text('button', 'Save offline'); button.type = 'button'; button.dataset.download = course.id;
    const caption = text('span', '', 'download-state'); caption.dataset.downloadState = course.id;
    row.append(button, caption); holder.append(row);
  }
  $('#learning-empty').hidden = count !== 0;
  const savedLessons = $('#saved-lessons'); savedLessons.replaceChildren();
  for (const [key, value] of Object.entries(state.entries).sort((a, b) => b[1].updatedAt - a[1].updatedAt)) {
    const lesson = lessons().get(key);
    if (!lesson || (!value.bookmarked && !value.note)) continue;
    const li = document.createElement('li'); li.append(link(lesson.url, lesson.title));
    if (value.note) li.append(text('p', value.note, 'saved-note'));
    savedLessons.append(li);
  }
  $('#bookmarks-empty').hidden = savedLessons.children.length > 0;
  const resume = lessons().get(state.lastOpened?.key);
  const box = $('#resume-learning'); box.replaceChildren();
  if (resume) box.append(link(resume.url, `Continue: ${resume.title}`, 'continue-link'));
}
function render() {
  if (!library) return;
  if (currentLesson) {
    const value = entry(currentLesson.key);
    $('#mark-read').textContent = value.read ? 'Read' : 'Mark as read';
    $('#mark-read').setAttribute('aria-pressed', String(value.read));
    $('#bookmark-lesson').textContent = value.bookmarked ? 'Bookmarked' : 'Bookmark';
    $('#bookmark-lesson').setAttribute('aria-pressed', String(value.bookmarked));
    $('#open-note').textContent = value.note ? 'Edit note' : 'Add a note';
  }
  for (const course of library.courses) {
    const progress = courseProgress(course, state);
    for (const el of all('[data-course-progress]').filter((e) => e.dataset.courseProgress === course.id)) el.textContent = progress.read ? `${progress.read} of ${progress.total} read` : 'Choose where to begin';
    for (const box of all('[data-meter]').filter((e) => e.dataset.meter === course.id)) { box.hidden = !progress.read; box.querySelector('progress').value = progress.read; }
  }
  renderMyLearning(); downloadControls(); filterCourses();
  const resume = lessons().get(state.lastOpened?.key);
  if ($('#library-resume') && resume) { $('#library-resume').hidden = false; $('#library-resume').href = resume.url; $('#library-resume').textContent = `Continue: ${resume.title}`; }
}

async function setupWorker() {
  if (!('serviceWorker' in navigator) || !window.isSecureContext || !import.meta.env.PROD) return;
  try {
    workerRegistration = await navigator.serviceWorker.register('/sw.js', { updateViaCache: 'none' });
    const offerUpdate = () => { if (workerRegistration.waiting && navigator.serviceWorker.controller) $('#update-banner').hidden = false; };
    offerUpdate();
    workerRegistration.addEventListener('updatefound', () => workerRegistration.installing?.addEventListener('statechange', offerUpdate));
    navigator.serviceWorker.addEventListener('controllerchange', () => { if (window.__openlearnUpdating) location.reload(); else refreshOffline(); });
    await navigator.serviceWorker.ready; await refreshOffline();
  } catch { notify('Offline reading could not be enabled. Check your connection and refresh to try again.'); }
}
function install() {
  if (installPrompt) { installPrompt.prompt(); installPrompt = null; }
  else $('#install-dialog').showModal();
}

async function init() {
  load();
  try {
    const response = await fetch('/app/catalog.json');
    if (!response.ok) throw new Error();
    library = await response.json();
    lessonIndex = new Map(library.courses.flatMap((c) => c.lessons.map((l) => [l.key, l])));
  } catch { notify('The course library could not be loaded. Reconnect and refresh this page.', true); return; }
  currentLesson = [...lessons().values()].find((l) => l.url === location.pathname);
  currentCourse = library.courses.find((c) => c.href === location.pathname) ?? library.courses.find((c) => c.id === state.activeCourse && c.lessons.some((l) => l.key === currentLesson?.key)) ?? library.courses.find((c) => c.lessons.some((l) => l.key === currentLesson?.key));
  if (currentCourse) {
    const next = structuredClone(state); next.activeCourse = currentCourse.id;
    if (!next.enrolled.includes(currentCourse.id)) next.enrolled.push(currentCourse.id);
    persist(next);
  }
  if (currentLesson) {
    $('#lesson-tools').hidden = false;
    const next = structuredClone(state); next.lastOpened = { key: currentLesson.key, at: Date.now() }; persist(next);
    const index = currentCourse.lessons.findIndex((l) => l.key === currentLesson.key);
    const following = currentCourse.lessons[index + 1];
    if (following) { $('#next-lesson').hidden = false; $('#next-lesson').href = following.url; $('#next-lesson').textContent = `Next: ${following.title}`; }
  }
  $('#course-filter')?.addEventListener('input', filterCourses);
  $('#offline-filter')?.addEventListener('click', () => { filterOffline = !filterOffline; $('#offline-filter').setAttribute('aria-pressed', String(filterOffline)); filterCourses(); });
  $('#mark-read')?.addEventListener('click', () => updateEntry(currentLesson.key, { read: !entry(currentLesson.key).read }));
  $('#bookmark-lesson')?.addEventListener('click', () => updateEntry(currentLesson.key, { bookmarked: !entry(currentLesson.key).bookmarked }));
  $('#open-note')?.addEventListener('click', () => { $('#lesson-note').value = entry(currentLesson.key).note; $('#note-title').textContent = `Note: ${currentLesson.title}`; $('#note-dialog').showModal(); });
  $('#note-form')?.addEventListener('submit', (event) => { event.preventDefault(); if (updateEntry(currentLesson.key, { note: $('#lesson-note').value })) $('#note-dialog').close(); });
  all('[data-close-dialog]').forEach((button) => button.addEventListener('click', () => button.closest('dialog').close()));
  document.addEventListener('click', (event) => { const button = event.target.closest?.('[data-download]'); if (button) download(button.dataset.download); });
  $('#install-app')?.addEventListener('click', install);
  $('#dismiss-update')?.addEventListener('click', () => { $('#update-banner').hidden = true; });
  $('#update-app')?.addEventListener('click', async () => {
    try {
      const status = await messageWorker('STATUS');
      if (status.downloads.length) { notify('Finish or cancel course downloads before updating the app.'); return; }
      if (workerRegistration?.waiting) { window.__openlearnUpdating = true; workerRegistration.waiting.postMessage({ type: 'ACTIVATE_UPDATE' }); }
    } catch (error) { notify(error.message); }
  });
  window.addEventListener('beforeinstallprompt', (event) => { event.preventDefault(); installPrompt = event; });
  window.addEventListener('appinstalled', () => { $('#install-app').hidden = true; notify('OpenLearn is installed.'); });
  if (matchMedia('(display-mode: standalone)').matches || navigator.standalone) $('#install-app').hidden = true;
  const connection = () => { $('#connection-state').textContent = navigator.onLine ? 'Ready to learn' : 'Offline reading'; };
  window.addEventListener('online', () => { connection(); refreshOffline(); }); window.addEventListener('offline', connection); connection();
  window.addEventListener('storage', (event) => { if (event.key === STORAGE_KEY) { load(); render(); } });
  $('#export-progress')?.addEventListener('click', exportBackup);
  $('#import-progress')?.addEventListener('change', async (event) => {
    try {
      const file = event.target.files[0]; if (!file) return;
      if (file.size > 2 * 1024 * 1024) throw new Error('Choose a progress backup smaller than 2 MB.');
      const incoming = validateState(JSON.parse(await file.text()));
      if (persist(mergeState(state, incoming), true)) { render(); notify('Progress imported. Newer notes and progress were kept.'); }
    } catch (error) { notify(error instanceof SyntaxError ? 'This file is not a valid progress backup.' : error.message, true); }
    finally { event.target.value = ''; }
  });
  if (navigator.storage?.estimate && $('#storage-usage')) {
    const estimate = await navigator.storage.estimate(); $('#storage-usage').textContent = `${size(estimate.usage ?? 0)} used by this app on this device`;
  }
  all('[data-app-nav]').forEach((a) => { if ((location.pathname === '/my-learning/' && a.dataset.appNav === 'learning') || (location.pathname === '/' && a.dataset.appNav === 'library')) a.setAttribute('aria-current', 'page'); });
  render(); setupWorker();
  if (document.querySelector('pre.mermaid')) {
    try {
      const { default: mermaid } = await import('mermaid');
      mermaid.initialize({ startOnLoad: false, securityLevel: 'strict', theme: document.documentElement.dataset.theme === 'dark' ? 'dark' : 'neutral' });
      await mermaid.run({ querySelector: 'pre.mermaid' });
    } catch { /* The diagram's source remains readable if rendering fails. */ }
  }
}
init();
