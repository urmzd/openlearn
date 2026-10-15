import test from 'node:test';
import assert from 'node:assert/strict';
import vm from 'node:vm';
import fs from 'node:fs/promises';
import { webcrypto, createHash } from 'node:crypto';
const origin = 'https://learning.test';
const digest = (body) => createHash('sha256').update(body).digest('hex');
const template = await fs.readFile(new URL('../src/service-worker.js', import.meta.url), 'utf8');

async function harness(fetcher = async () => new Response('chapter')) {
  const stores = new Map(); const events = new Map();
  const key = (input) => new URL(typeof input === 'string' ? input : input.url, origin).href;
  const caches = {
    async open(name) {
      if (!stores.has(name)) stores.set(name, new Map());
      const store = stores.get(name);
      return {
        async put(input, response) { store.set(key(input), response.clone()); },
        async match(input) { return store.get(key(input))?.clone(); },
        async keys() { return [...store.keys()].map((url) => new Request(url)); },
        async delete(input) { return store.delete(key(input)); },
        async addAll(urls) { for (const url of urls) { const r = await fetcher(new Request(new URL(url, origin))); if (!r.ok) throw new Error('download failed'); store.set(key(url), r); } },
      };
    },
    async keys() { return [...stores.keys()]; },
    async delete(name) { return stores.delete(name); },
  };
  const context = vm.createContext({ caches, fetch: fetcher, Request, Response, URL, AbortController, crypto: webcrypto, Uint8Array, setTimeout, clearTimeout, self: { location: { origin }, addEventListener: (type, handler) => events.set(type, handler), clients: { claim: async () => {} }, skipWaiting: async () => {} } });
  vm.runInContext(template.replace('__OPENLEARN_BUILD__', 'new').replace('__OPENLEARN_SHELL__', '[]'), context);
  const index = { build: 'new', courses: { maths: { title: 'Maths', bytes: 7, files: [{ url: '/chapter/', hash: digest('chapter') }] } } };
  await (await caches.open('openlearn-shell:new')).put('/app/offline.json', new Response(JSON.stringify(index)));
  const message = async (data, source = `${origin}/`) => {
    const messages = []; let work;
    events.get('message')({ data, source: { url: source }, ports: [{ postMessage: (m) => messages.push(m) }], waitUntil: (p) => { work = p; } });
    await work; return messages;
  };
  return { stores, caches, message, events, context };
}

test('a complete, verified course is saved and can be read without a network', async () => {
  let online = true;
  const h = await harness(async () => { if (!online) throw new Error('offline'); return new Response('chapter'); });
  assert.equal((await h.message({ type: 'DOWNLOAD_COURSE', id: 'maths' })).at(-1).type, 'DONE');
  const status = (await h.message({ type: 'STATUS' }))[0]; assert.equal(status.saved[0].id, 'maths');
  online = false;
  let response; const work = [];
  h.events.get('fetch')({ request: { url: `${origin}/chapter/`, method: 'GET', mode: 'navigate' }, respondWith: (p) => { response = p; }, waitUntil: (p) => work.push(p) });
  assert.equal(await (await response).text(), 'chapter'); await Promise.all(work);
});
test('a failed update keeps the previous edition and deletes the incomplete download', async () => {
  const h = await harness(async () => new Response('wrong edition'));
  await (await h.caches.open('openlearn-course:maths:old')).put('/chapter/', new Response('old chapter'));
  await (await h.caches.open('openlearn-offline-metadata')).put('/__openlearn_offline/maths', new Response(JSON.stringify({ id: 'maths', title: 'Maths', build: 'old' })));
  const result = await h.message({ type: 'DOWNLOAD_COURSE', id: 'maths' });
  assert.equal(result.at(-1).type, 'ERROR');
  assert.match(result.at(-1).message, /changed/);
  assert.equal(h.stores.has('openlearn-course:maths:new'), false);
  assert.equal(await (await (await h.caches.open('openlearn-course:maths:old')).match('/chapter/')).text(), 'old chapter');
  assert.equal((await h.message({ type: 'STATUS' }))[0].saved[0].build, 'old');
});
test('404 responses never mark a course as downloaded', async () => {
  const h = await harness(async () => new Response('not found', { status: 404 }));
  assert.equal((await h.message({ type: 'DOWNLOAD_COURSE', id: 'maths' })).at(-1).type, 'ERROR');
  assert.equal((await h.message({ type: 'STATUS' }))[0].saved.length, 0);
  assert.equal(h.stores.has('openlearn-course:maths:new'), false);
});
test('updates retain the shell needed by saved courses, then release it on removal', async () => {
  const h = await harness(); await h.caches.open('openlearn-shell:old'); await h.caches.open('openlearn-shell:unused');
  await h.caches.open('openlearn-course:maths:old');
  await (await h.caches.open('openlearn-offline-metadata')).put('/__openlearn_offline/maths', new Response(JSON.stringify({ id: 'maths', build: 'old' })));
  let work; h.events.get('activate')({ waitUntil: (p) => { work = p; } }); await work;
  assert.equal(h.stores.has('openlearn-shell:old'), true); assert.equal(h.stores.has('openlearn-shell:unused'), false);
  await h.message({ type: 'REMOVE_COURSE', id: 'maths' });
  assert.equal(h.stores.has('openlearn-shell:old'), false); assert.equal(h.stores.has('openlearn-course:maths:old'), false);
});
test('foreign clients cannot trigger offline mutations', async () => {
  const h = await harness(); assert.deepEqual(await h.message({ type: 'DOWNLOAD_COURSE', id: 'maths' }, 'https://other.test/'), []);
  assert.equal(h.stores.has('openlearn-course:maths:new'), false);
});

test('cancelling an in-progress download never publishes a partial course', async () => {
  let started;
  const entered = new Promise((resolve) => { started = resolve; });
  const h = await harness(async (request) => new Promise((resolve, reject) => {
    started(); request.signal.addEventListener('abort', () => reject(new DOMException('aborted', 'AbortError')));
  }));
  const downloading = h.message({ type: 'DOWNLOAD_COURSE', id: 'maths' });
  await entered;
  await h.message({ type: 'CANCEL_DOWNLOAD', id: 'maths' });
  const result = await downloading;
  assert.equal(result.at(-1).type, 'ERROR'); assert.match(result.at(-1).message, /cancelled/);
  assert.equal((await h.message({ type: 'STATUS' }))[0].saved.length, 0);
  assert.equal(h.stores.has('openlearn-course:maths:new'), false);
});
test('completing another course does not delete a simultaneous download', async () => {
  let release; let started;
  const entered = new Promise((resolve) => { started = resolve; });
  const pending = new Promise((resolve) => { release = resolve; });
  const h = await harness(async (request) => {
    if (request.url.endsWith('/chapter/')) { started(); await pending; }
    return new Response('chapter');
  });
  const shell = await h.caches.open('openlearn-shell:new');
  const index = await (await shell.match('/app/offline.json')).json();
  index.courses.physics = { title: 'Physics', bytes: 7, files: [{ url: '/physics/', hash: digest('chapter') }] };
  await shell.put('/app/offline.json', new Response(JSON.stringify(index)));
  const downloading = h.message({ type: 'DOWNLOAD_COURSE', id: 'maths' }); await entered;
  assert.equal((await h.message({ type: 'DOWNLOAD_COURSE', id: 'physics' })).at(-1).type, 'DONE');
  assert.equal(h.stores.has('openlearn-course:maths:new'), true);
  release(); assert.equal((await downloading).at(-1).type, 'DONE');
  assert.equal((await h.message({ type: 'STATUS' }))[0].saved.length, 2);
});

test('an evicted course is no longer advertised as saved', async () => {
  const h = await harness();
  await h.message({ type: 'DOWNLOAD_COURSE', id: 'maths' });
  await h.caches.delete('openlearn-course:maths:new');
  assert.equal((await h.message({ type: 'STATUS' }))[0].saved.length, 0);
});
