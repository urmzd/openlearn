import { STORAGE_KEY, emptyState, validateState, mergeState, courseProgress, isCurrentDownload } from '../lib/learning.mjs';
import { enhance, startReader } from './reader.mjs';

let state = emptyState();
let library;
let lessonIndex = new Map();
let workerStatus = { saved: [], downloads: [], available: {} };
let currentLesson;
let currentCourse;
let noteKey;
let toastTimer;
const $ = (selector) => document.querySelector(selector);
const all = (selector) => [...document.querySelectorAll(selector)];
const byID = (id) => library?.courses.find((c) => c.id === id);
const entry = (key) => state.entries[key] ?? { read: false, bookmarked: false, note: '', updatedAt: 0 };
const size = (bytes) => `${(bytes / 1024 / 1024).toFixed(1)} MB`;

function text(tag, value, className) {
	const node = document.createElement(tag);
	node.textContent = value;
	if (className) node.className = className;
	return node;
}
function link(url, label, className) {
	const a = text('a', label, className);
	a.href = url;
	return a;
}
// Messages live on <body> so no page panel can cover them.
function notify(message, persistent = false) {
	let box = $('#app-message');
	if (!box) {
		box = text('div', '', 'app-message');
		box.id = 'app-message';
		box.setAttribute('role', 'status');
		box.setAttribute('aria-live', 'polite');
		document.body.append(box);
	}
	box.textContent = message;
	box.hidden = false;
	clearTimeout(toastTimer);
	if (!persistent) toastTimer = setTimeout(() => { box.hidden = true; }, 5000);
}

// ---------- progress on this device ----------

function exportBackup() {
	const blob = new Blob([JSON.stringify(state, null, 2)], { type: 'application/json' });
	const url = URL.createObjectURL(blob);
	const a = link(url, 'Download');
	a.download = `openlearn-progress-${new Date().toISOString().slice(0, 10)}.json`;
	a.click();
	setTimeout(() => URL.revokeObjectURL(url), 1000);
}
function load() {
	try {
		const raw = localStorage.getItem(STORAGE_KEY);
		if (raw) state = validateState(JSON.parse(raw));
	} catch { notify('Your saved progress could not be opened. Import a backup from your shelf to restore it.', true); }
}
function persist(next, restore = false) {
	try {
		const latest = localStorage.getItem(STORAGE_KEY);
		if (latest) {
			try { next = mergeState(validateState(JSON.parse(latest)), next); }
			catch (error) { if (!restore) throw error; }
		}
		localStorage.setItem(STORAGE_KEY, JSON.stringify(next));
		state = next;
		return true;
	} catch {
		state = next;
		notify('Storage is full or unavailable, so progress is not being saved. Export a backup from your shelf.', true);
		return false;
	}
}
function updateEntry(key, changes) {
	const next = structuredClone(state);
	next.entries[key] = { ...entry(key), ...changes, updatedAt: Math.max(Date.now(), entry(key).updatedAt + 1) };
	const saved = persist(next);
	render();
	return saved;
}
function markRead(key) {
	if (key && lessonIndex.has(key) && !entry(key).read) updateEntry(key, { read: true });
}
function opened(lesson) {
	const next = structuredClone(state);
	next.lastOpened = { key: lesson.key, at: Date.now() };
	persist(next);
}

// ---------- offline copies ----------

async function messageWorker(type, data = {}, onProgress) {
	const worker = navigator.serviceWorker?.controller;
	if (!worker) throw new Error('Offline reading is getting ready. Reconnect and refresh if this persists.');
	return new Promise((resolve, reject) => {
		const channel = new MessageChannel();
		let timer;
		const reset = () => {
			clearTimeout(timer);
			timer = setTimeout(() => { channel.port1.close(); reject(new Error('The download stalled. Check your connection and try again.')); }, 60000);
		};
		channel.port1.onmessage = (event) => {
			reset();
			if (event.data.type === 'PROGRESS') { onProgress?.(event.data); return; }
			clearTimeout(timer);
			channel.port1.close();
			if (event.data.type === 'ERROR') reject(new Error(event.data.message));
			else resolve(event.data);
		};
		reset();
		worker.postMessage({ type, ...data }, [channel.port2]);
	});
}
async function refreshOffline() {
	if (!navigator.serviceWorker?.controller) return;
	try { workerStatus = await messageWorker('STATUS'); render(); } catch { /* Shown again on the next action. */ }
}
function downloadControls() {
	for (const button of all('[data-download]')) {
		const id = button.dataset.download;
		const saved = workerStatus.saved.find((c) => c.id === id);
		const downloading = workerStatus.downloads.find((c) => c.id === id);
		const current = isCurrentDownload(saved, workerStatus.build);
		button.textContent = downloading ? 'Cancel download' : current ? 'Remove offline copy' : saved ? 'Refresh offline copy' : 'Save offline';
		button.setAttribute('aria-label', `${button.textContent}: ${byID(id)?.title ?? id}`);
		const caption = all('[data-download-state]').find((node) => node.dataset.downloadState === id);
		if (caption) caption.textContent = downloading ? `${downloading.done} of ${downloading.total} pages` : saved ? `Saved on this device · ${size(saved.bytes)}` : workerStatus.available?.[id] ? `${size(workerStatus.available[id].bytes)} to save` : '';
	}
}
async function download(id) {
	const existing = workerStatus.saved.find((c) => c.id === id);
	try {
		if (workerStatus.downloads.some((c) => c.id === id)) { await messageWorker('CANCEL_DOWNLOAD', { id }); return; }
		if (isCurrentDownload(existing, workerStatus.build)) {
			await messageWorker('REMOVE_COURSE', { id });
			notify('Offline copy removed. Your notes and progress are kept.');
			return;
		}
		workerStatus.downloads.push({ id, done: 0, total: 0 });
		render();
		await messageWorker('DOWNLOAD_COURSE', { id }, (event) => {
			const operation = workerStatus.downloads.find((c) => c.id === id);
			if (operation) Object.assign(operation, event);
			downloadControls();
		});
		await navigator.storage?.persist?.();
		notify(`${byID(id).title} is saved for offline reading.`);
	} catch (error) { notify(error.message, true); }
	finally { await refreshOffline(); }
}

// ---------- the shelf ----------

function renderShelf() {
	const list = $('#book-list');
	if (!list) return;
	const rows = [...list.children];
	const started = (id) => state.enrolled.includes(id) || courseProgress(byID(id), state).read > 0;
	// Books you have started come first; the rest keep their order.
	rows.sort((a, b) => Number(started(b.dataset.course)) - Number(started(a.dataset.course)));
	list.append(...rows);
	for (const row of rows) row.classList.toggle('is-started', started(row.dataset.course));

	const resume = lessonIndex.get(state.lastOpened?.key);
	const card = $('#continue-card');
	if (card && resume) {
		const book = library.courses.find((c) => c.lessons.some((l) => l.key === resume.key));
		card.href = resume.url;
		$('#continue-title').textContent = resume.title;
		$('#continue-book').textContent = book?.title ?? '';
		card.hidden = false;
	}

	const notes = $('#saved-notes');
	if (notes) {
		notes.replaceChildren();
		for (const [key, value] of Object.entries(state.entries).sort((a, b) => b[1].updatedAt - a[1].updatedAt)) {
			const lesson = lessonIndex.get(key);
			if (!lesson || !value.note) continue;
			const li = document.createElement('li');
			li.append(link(lesson.url, lesson.title), text('p', value.note, 'saved-note'));
			notes.append(li);
		}
		$('#shelf-notes').hidden = notes.children.length === 0;
	}
}
function render() {
	if (!library) return;
	for (const course of library.courses) {
		const progress = courseProgress(course, state);
		for (const node of all(`[data-course-progress="${course.id}"]`)) node.textContent = progress.read ? `${progress.read} of ${progress.total} read` : `${progress.total} sections`;
		for (const meter of all(`[data-meter="${course.id}"]`)) { meter.hidden = !progress.read; meter.value = progress.read; }
		// The button on a book's front page skips the front page itself.
		const inside = { ...course, lessons: course.lessons.filter((l) => l.url !== course.href) };
		const reading = courseProgress(inside, state);
		const began = reading.read > 0 || inside.lessons.some((l) => l.key === state.lastOpened?.key);
		for (const begin of all(`[data-begin="${course.id}"]`)) {
			if (!reading.next) continue;
			begin.href = reading.next.url;
			begin.textContent = began ? 'Continue reading' : 'Start reading';
		}
	}
	for (const button of all('[data-note-for]')) button.textContent = entry(button.dataset.noteFor).note ? 'Edit your note' : 'Add a note';
	renderShelf();
	downloadControls();
}

// ---------- notes ----------

function openNote(key) {
	const lesson = lessonIndex.get(key);
	if (!lesson) return;
	noteKey = key;
	$('#lesson-note').value = entry(key).note;
	$('#note-title').textContent = `Note: ${lesson.title}`;
	$('#note-dialog').showModal();
}

// ---------- updates ----------

// A new release installs and takes over quietly (the worker skips waiting).
// Pages always load fresh when online, so there is nothing for the reader to
// approve; saved books show "Refresh offline copy" when a newer one exists.
async function setupWorker() {
	if (!('serviceWorker' in navigator) || !window.isSecureContext || !import.meta.env.PROD) return;
	try {
		await navigator.serviceWorker.register('/sw.js', { updateViaCache: 'none' });
		navigator.serviceWorker.addEventListener('controllerchange', refreshOffline);
		await navigator.serviceWorker.ready;
		await refreshOffline();
	} catch { /* Reading works without offline support. */ }
}

async function init() {
	load();
	try {
		const response = await fetch('/app/catalog.json');
		if (!response.ok) throw new Error();
		library = await response.json();
		lessonIndex = new Map(library.courses.flatMap((c) => c.lessons.map((l) => [l.key, l])));
	} catch {
		enhance(document.querySelector('.sl-markdown-content') ?? document.createElement('div'));
		return;
	}
	currentLesson = [...lessonIndex.values()].find((l) => l.url === location.pathname);
	currentCourse = library.courses.find((c) => c.href === location.pathname)
		?? library.courses.find((c) => c.id === state.activeCourse && c.lessons.some((l) => l.key === currentLesson?.key))
		?? library.courses.find((c) => c.lessons.some((l) => l.key === currentLesson?.key));
	if (currentCourse) {
		const next = structuredClone(state);
		next.activeCourse = currentCourse.id;
		if (!next.enrolled.includes(currentCourse.id)) next.enrolled.push(currentCourse.id);
		persist(next);
	}
	if (currentLesson) {
		$('#lesson-tools').hidden = false;
		$('#lesson-tools [data-note-for]').dataset.noteFor = currentLesson.key;
		opened(currentLesson);
	}

	document.addEventListener('click', (event) => {
		const download_ = event.target.closest?.('[data-download]');
		if (download_) download(download_.dataset.download);
		const note = event.target.closest?.('[data-note-for]');
		if (note) openNote(note.dataset.noteFor);
	});
	$('#note-form')?.addEventListener('submit', (event) => {
		event.preventDefault();
		if (updateEntry(noteKey, { note: $('#lesson-note').value })) $('#note-dialog').close();
	});
	all('[data-close-dialog]').forEach((button) => button.addEventListener('click', () => button.closest('dialog').close()));
	$('#export-progress')?.addEventListener('click', exportBackup);
	$('#import-progress')?.addEventListener('change', async (event) => {
		try {
			const file = event.target.files[0];
			if (!file) return;
			if (file.size > 2 * 1024 * 1024) throw new Error('Choose a progress backup smaller than 2 MB.');
			const incoming = validateState(JSON.parse(await file.text()));
			if (persist(mergeState(state, incoming), true)) { render(); notify('Progress imported. Newer notes and progress were kept.'); }
		} catch (error) { notify(error instanceof SyntaxError ? 'This file is not a progress backup.' : error.message, true); }
		finally { event.target.value = ''; }
	});
	if (navigator.storage?.estimate && $('#storage-usage')) {
		const estimate = await navigator.storage.estimate();
		$('#storage-usage').textContent = `${size(estimate.usage ?? 0)} used on this device`;
	}
	addEventListener('offline', () => notify('You are offline. Books you saved still open.'));
	addEventListener('online', refreshOffline);
	addEventListener('storage', (event) => { if (event.key === STORAGE_KEY) { load(); render(); } });

	startReader({
		course: currentCourse,
		lesson: currentLesson,
		markRead,
		onSection: (lesson) => { currentLesson = lesson; opened(lesson); },
		onAppend: render,
	});
	render();
	setupWorker();
}
init();
