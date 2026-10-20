/* Generated with a content-derived revision by scripts/build-pwa.mjs. */
const BUILD = '__OPENLEARN_BUILD__';
const SHELL_URLS = __OPENLEARN_SHELL__;
const SHELL = `openlearn-shell:${BUILD}`;
const VISITED = `openlearn-visited:${BUILD}`;
const META = 'openlearn-offline-metadata';
const COURSE_PREFIX = 'openlearn-course:';
const inflight = new Map();
const origin = self.location.origin;
const metaKey = (id) => `${origin}/__openlearn_offline/${id}`;
const courseCache = (id, build) => `${COURSE_PREFIX}${id}:${build}`;
const sameOrigin = (path) => new URL(path, origin).origin === origin;

async function metadata() {
  const cache = await caches.open(META);
  const names = await caches.keys();
  const rows = [];
  for (const request of await cache.keys()) {
    try {
      const row = await (await cache.match(request)).json();
      const name = courseCache(row.id, row.build);
      const present = /^[a-z0-9-]+$/.test(row.id) && typeof row.build === 'string' && names.includes(name);
      if (present && (!row.fileCount || (await (await caches.open(name)).keys()).length === row.fileCount)) rows.push(row);
      else await cache.delete(request);
    } catch { await cache.delete(request); }
  }
  return rows;
}

async function cleanup() {
  const saved = await metadata();
  const keepBuilds = new Set([BUILD, ...saved.map((r) => r.build)]);
  const keepCourses = new Set([...saved.map((r) => courseCache(r.id, r.build)), ...[...inflight.keys()].map((id) => courseCache(id, BUILD))]);
  for (const name of await caches.keys()) {
    if ((name.startsWith('openlearn-shell:') && !keepBuilds.has(name.slice('openlearn-shell:'.length))) || (name.startsWith(COURSE_PREFIX) && !keepCourses.has(name)) || (name.startsWith('openlearn-visited:') && name !== VISITED)) await caches.delete(name);
  }
}

self.addEventListener('install', (event) => {
  // A failed shell download leaves the previous service worker in control.
  event.waitUntil((async () => {
    const cache = await caches.open(SHELL);
    try { for (let i = 0; i < SHELL_URLS.length; i += 8) await cache.addAll(SHELL_URLS.slice(i, i + 8)); }
    catch (error) { await caches.delete(SHELL); throw error; }
    // Take over at once: pages load fresh online, so a new release needs no
    // approval. A course download running in the old worker fails with "The
    // app updated during this download" and can be retried.
    await self.skipWaiting();
  })());
});
self.addEventListener('activate', (event) => event.waitUntil((async () => { await cleanup(); await self.clients.claim(); })()));

async function offlineIndex() {
  const shell = await caches.open(SHELL);
  const response = await shell.match('/app/offline.json');
  if (!response) throw new Error('Reconnect and refresh the app before saving a course.');
  const index = await response.json();
  if (index.build !== BUILD) throw new Error('A new course edition is available. Refresh and try again.');
  return index;
}

async function downloadCourse(id, send) {
  if (!/^[a-z0-9-]+$/.test(id)) throw new Error('Unknown course.');
  if (inflight.has(id)) throw new Error('This course is already downloading.');
  const index = await offlineIndex();
  const course = index.courses[id];
  if (!course) throw new Error('This course is not in this edition.');
  const old = (await metadata()).find((r) => r.id === id);
  if (old?.build === BUILD) { send({ type: 'DONE', id }); return; }
  const name = courseCache(id, BUILD);
  await caches.delete(name);
  const cache = await caches.open(name);
  const controller = new AbortController();
  const operation = { controller, done: 0, total: course.files.length };
  inflight.set(id, operation);
  let cursor = 0;
  let failure;
  const task = async () => {
    while (cursor < course.files.length && !controller.signal.aborted) {
      const file = course.files[cursor++];
      try {
        if (!sameOrigin(file.url) || !file.url.startsWith('/')) throw new Error('Invalid offline resource.');
        const response = await fetch(new Request(new URL(file.url, origin), { cache: 'reload', redirect: 'error', signal: controller.signal }));
        if (!response.ok || response.type === 'opaque') throw new Error('A course page could not be downloaded. Reconnect and try again.');
        const bytes = await response.clone().arrayBuffer();
        const digest = await crypto.subtle.digest('SHA-256', bytes);
        const hash = Array.from(new Uint8Array(digest), (b) => b.toString(16).padStart(2, '0')).join('');
        if (hash !== file.hash) throw new Error('The course changed during the download. Refresh the app and try again.');
        await cache.put(file.url, response);
        operation.done++;
        send({ type: 'PROGRESS', id, done: operation.done, total: operation.total });
      } catch (error) { failure ??= error; controller.abort(); }
    }
  };
  try {
    await Promise.all(Array.from({ length: Math.min(4, course.files.length) }, task));
    if (controller.signal.aborted && (!failure || failure.name === 'AbortError')) throw new Error('Download cancelled.');
    if (failure) throw failure;
    if (controller.signal.aborted) throw new Error('Download cancelled.');
    const stored = await caches.open(name);
    if ((await stored.keys()).length !== course.files.length || !(await caches.keys()).includes(SHELL)) throw new Error('The app updated during this download. Refresh and try again.');
    const meta = await caches.open(META);
    await meta.put(metaKey(id), new Response(JSON.stringify({ id, title: course.title, build: BUILD, savedAt: Date.now(), bytes: course.bytes, fileCount: course.files.length }), { headers: { 'Content-Type': 'application/json' } }));
    if (old && old.build !== BUILD) await caches.delete(courseCache(id, old.build));
    await cleanup();
    send({ type: 'DONE', id });
  } catch (error) {
    await caches.delete(name);
    if (error.name === 'QuotaExceededError') throw new Error('Storage is full. Remove a saved course, then try again.');
    throw error;
  } finally { inflight.delete(id); }
}

self.addEventListener('message', (event) => {
  if (!event.source?.url || new URL(event.source.url).origin !== origin) return;
  const port = event.ports[0];
  const send = (data) => { try { port?.postMessage(data); } catch {} };
  const { type, id } = event.data ?? {};
  event.waitUntil((async () => {
    try {
      if (type === 'ACTIVATE_UPDATE') { await self.skipWaiting(); send({ type: 'DONE' }); }
      else if (type === 'STATUS') {
        const index = await offlineIndex();
        send({ type: 'STATUS', build: BUILD, saved: await metadata(), available: Object.fromEntries(Object.entries(index.courses).map(([id, course]) => [id, { bytes: course.bytes, files: course.files.length }])), downloads: [...inflight].map(([id, o]) => ({ id, done: o.done, total: o.total })) });
      }
      else if (type === 'DOWNLOAD_COURSE') await downloadCourse(id, send);
      else if (type === 'CANCEL_DOWNLOAD') { inflight.get(id)?.controller.abort(); send({ type: 'DONE', id }); }
      else if (type === 'REMOVE_COURSE' && /^[a-z0-9-]+$/.test(id)) {
        inflight.get(id)?.controller.abort();
        const old = (await metadata()).find((r) => r.id === id);
        if (old) { await (await caches.open(META)).delete(metaKey(id)); await caches.delete(courseCache(id, old.build)); }
        await cleanup(); send({ type: 'DONE', id });
      }
    } catch (error) { send({ type: 'ERROR', message: error.message }); }
  })());
});

async function cachedResource(request) {
  const names = await caches.keys();
  const ordered = [SHELL, VISITED, ...names.filter((n) => n.startsWith(COURSE_PREFIX) && n.endsWith(`:${BUILD}`)), ...names];
  for (const name of [...new Set(ordered)]) {
    if (!names.includes(name) || !name.startsWith('openlearn-') || name === META) continue;
    const response = await (await caches.open(name)).match(request, { ignoreSearch: true });
    if (response) return response;
  }
}

async function navigation(request, event) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 5000);
  try {
    const response = await fetch(request, { signal: controller.signal });
    if (response.ok && response.headers.get('content-type')?.includes('text/html')) {
      const copy = response.clone();
      event.waitUntil((async () => {
        const visited = await caches.open(VISITED);
        await visited.put(request, copy);
        const keys = await visited.keys();
        for (const key of keys.slice(0, Math.max(0, keys.length - 60))) await visited.delete(key);
      })().catch(() => {}));
    }
    return response;
  } catch {
    return await cachedResource(request) ?? await (await caches.open(SHELL)).match('/offline/') ?? new Response('Connect to the internet to open this page.', { status: 503 });
  } finally { clearTimeout(timer); }
}

self.addEventListener('fetch', (event) => {
  const request = event.request;
  const url = new URL(request.url);
  if (request.method !== 'GET' || url.origin !== origin || url.pathname === '/sw.js') return;
  if (request.mode === 'navigate') event.respondWith(navigation(request, event));
  else event.respondWith((async () => await cachedResource(request) ?? fetch(request))());
});
