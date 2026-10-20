import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import { createHash } from 'node:crypto';
import sharp from 'sharp';
import { pathToFileURL } from 'node:url';
import path from 'node:path';
const dist = process.env.OPENLEARN_DIST ? pathToFileURL(path.resolve(process.env.OPENLEARN_DIST) + path.sep) : new URL('../dist/', import.meta.url);
const read = (file) => fs.readFile(new URL(file, dist));
const json = async (file) => JSON.parse(await read(file));

test('the built app has a scoped install manifest and correctly sized icons', async () => {
  const manifest = await json('manifest.webmanifest');
  assert.equal(manifest.display, 'standalone'); assert.equal(manifest.scope, '/'); assert.equal(manifest.start_url, '/');
  for (const icon of manifest.icons) {
    const image = await sharp(await read(icon.src.slice(1))).metadata();
    assert.equal(`${image.width}x${image.height}`, icon.sizes);
    assert.match(icon.purpose, /maskable/);
  }
  const home = (await read('index.html')).toString();
  assert.match(home, /manifest.webmanifest/); assert.match(home, /Continue reading/); assert.match(home, /Save offline/);
  assert.doesNotMatch(home, /update-banner|app-dock/);
});
test('every offline page and diagram matches the content manifest', async () => {
  const index = await json('app/offline.json'); const library = await json('app/catalog.json');
  assert.equal(Object.keys(index.courses).length, library.courses.length);
  for (const course of Object.values(index.courses)) for (const file of course.files) {
    assert.ok(file.url.startsWith('/') && !file.url.startsWith('//'));
    const content = await read(file.url.slice(1) + (file.url.endsWith('/') ? 'index.html' : ''));
    assert.equal(createHash('sha256').update(content).digest('hex'), file.hash, file.url);
  }
  const worker = (await read('sw.js')).toString();
  assert.ok(worker.includes(index.build)); assert.ok(!worker.includes('__OPENLEARN_'));
  assert.ok(worker.includes('/offline/')); assert.ok(worker.includes('skipWaiting'));
});
