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
  const library = await json('app/catalog.json');
  // Independent publications use the book's title; the full library uses the brand.
  const expectedTitle = library.courses.length === 1 ? library.courses[0].title : 'Openlearn';
  assert.equal(manifest.name, expectedTitle);
  assert.equal(library.title, expectedTitle);
  assert.ok(home.includes(expectedTitle.replaceAll('&', '&amp;')), `Home page must show ${expectedTitle}`);
  assert.doesNotMatch(home, /Supersource/);
  assert.match(home, /manifest.webmanifest/); assert.match(home, /Continue reading/); assert.match(home, /Find a book/);
  if (library.courses.some((course) => course.pdf)) assert.match(home, /Download [^"]+ PDF/);
  else assert.doesNotMatch(home, /aria-label="Download [^"]+ PDF/);
  assert.doesNotMatch(home, /Save offline|Export progress|Import progress/);
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

// Verify the public book surface, including chapters that are otherwise below the fold.
test('books provide ordered contents and only verified PDF downloads', async () => {
  const library = await json('app/catalog.json');
  const assets = JSON.parse(await fs.readFile(new URL('../release-assets.json', import.meta.url)));
  const published = new Set(assets.map((asset) => asset.url));
  for (const course of library.courses) {
    const html = (await read(course.href.slice(1) + 'index.html')).toString();
    for (const lesson of course.lessons) {
      const page = (await read(lesson.url.slice(1) + 'index.html')).toString();
      assert.doesNotMatch(page, /http-equiv="refresh"/i, lesson.url + ' must contain its lesson, not a redirect');
    }
    assert.doesNotMatch(html, /Save offline|data-download=/, course.title);
    if (course.pdf) {
      assert.ok(published.has(course.pdf), course.title);
      assert.ok(html.includes(course.pdf), course.title);
    }
    if (course.lessons.some((lesson) => /^Chapter [0-9]+:/.test(lesson.title))) {
      assert.match(html, /contents-chapter/, course.title);
    }
  }
});

test('Systems Engineering keeps legacy links within one book', async (t) => {
  const library = await json('app/catalog.json');
  const book = library.courses.find((course) => course.id === 'systems');
  if (!book) { t.skip('Systems Engineering is outside this publication'); return; }
  assert.equal(book.title, 'Systems Engineering');
  assert.ok(!library.courses.some((course) => course.id === 'infrastructure'));
  assert.ok(book.lessons.some((lesson) => lesson.url === '/infrastructure/01-containers-kubernetes/' && lesson.title.startsWith('Chapter 10:')));
  assert.ok(book.lessons.some((lesson) => lesson.key === 'content/courses/systems/04-observability/README.md' && lesson.title.startsWith('Chapter 11:')));
  const previousHome = (await read('infrastructure/index.html')).toString();
  assert.match(previousHome, /url=\/systems\//i);
  const html = (await read('systems/index.html')).toString();
  assert.match(html, /Chapter 12|12\. Incident Response/);
  assert.match(html, /Appendix A/);
});

test('the library separates six books from guides and retains old home routes', async (t) => {
  const library = await json('app/catalog.json');
  if (library.courses.length === 1) { t.skip('independent publication'); return; }
  assert.equal(library.courses.filter((c) => c.kind === 'book').length, 6);
  assert.equal(library.courses.filter((c) => c.kind === 'field-guide').length, 2);
  assert.equal(library.courses.filter((c) => c.kind === 'reading-path').length, 2);
  for (const [old, current] of [['ml', 'ai-engineering'], ['courses/build-an-llm', 'ai-engineering'], ['software-craftsmanship', 'software-engineering'], ['algorithms', 'programming-and-algorithms'], ['math', 'mathematics']]) {
    assert.match((await read(`${old}/index.html`)).toString(), new RegExp(`url=/${current}/`, 'i'));
  }
  for (const id of ['ai-engineering', 'systems', 'software-engineering', 'programming-and-algorithms', 'mathematics']) {
    const book = library.courses.find((c) => c.id === id);
    assert.equal(book.pdf, undefined, 'pre-merge PDFs must not represent a consolidated book');
  }
  const ai = library.courses.find((c) => c.id === 'ai-engineering');
  assert.ok(ai.lessons.some((l) => l.url === '/ml/08-tinyllm/02-tokenizers/'));
  assert.match((await read('ai-engineering/index.html')).toString(), /Learning foundations/);
});
