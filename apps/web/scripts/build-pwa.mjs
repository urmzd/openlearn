import fs from 'node:fs/promises';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { fileURLToPath } from 'node:url';
import sharp from 'sharp';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const dist = process.env.OPENLEARN_DIST ? path.resolve(process.env.OPENLEARN_DIST) : path.join(root, 'dist');
const hash = (bytes) => createHash('sha256').update(bytes).digest('hex');
async function walk(dir) {
  const entries = await fs.readdir(dir, { withFileTypes: true });
  return (await Promise.all(entries.map((e) => e.isDirectory() ? walk(path.join(dir, e.name)) : path.join(dir, e.name)))).flat();
}
const generated = new Set(['sw.js', 'manifest.webmanifest', 'icon-192.png', 'icon-512.png', 'apple-touch-icon.png', 'app/offline.json']);
const files = (await walk(dist)).filter((file) => !generated.has(path.relative(dist, file).split(path.sep).join('/')));
const workerSource = await fs.readFile(path.join(root, 'src/service-worker.js'), 'utf8');
const builderSource = await fs.readFile(fileURLToPath(import.meta.url));
const data = JSON.parse(await fs.readFile(path.join(dist, 'app/catalog.json'), 'utf8'));
const shared = files.filter((p) => /\/(?:_astro|pagefind)\//.test(p) && !p.endsWith('.map'));
const toURL = (p) => '/' + path.relative(dist, p).split(path.sep).join('/');
const descriptions = new Map();
for (const p of files) {
  const bytes = await fs.readFile(p);
  descriptions.set(toURL(p), { hash: hash(bytes), bytes: bytes.length });
}
const build = hash([...descriptions].map(([url, meta]) => `${url}:${meta.hash}`).sort().join('\n') + hash(workerSource) + hash(builderSource)).slice(0, 16);
const courses = {};
for (const course of data.courses) {
  const urls = new Set([course.href, ...course.lessons.map((l) => l.url)]);
  const resources = new Set();
  for (const url of urls) {
    const htmlURL = url.endsWith('/') ? `${url}index.html` : url;
    if (!descriptions.has(htmlURL)) throw new Error(`Missing course page: ${url}`);
    resources.add(htmlURL);
    const html = await fs.readFile(path.join(dist, htmlURL), 'utf8');
    for (const match of html.matchAll(/(?:src|href)=["'](\/repo\/[^"'#?]+)["']/g)) {
      const asset = decodeURI(match[1]);
      if (descriptions.has(asset)) resources.add(asset);
    }
  }
  const entries = [...resources].map((url) => ({ url: url.endsWith('/index.html') ? url.slice(0, -10) : url, hash: descriptions.get(url).hash }));
  courses[course.id] = { title: course.title, files: entries, bytes: [...resources].reduce((n, url) => n + descriptions.get(url).bytes, 0) };
}
await fs.writeFile(path.join(dist, 'app/offline.json'), JSON.stringify({ build, courses }));
// Use the same icon source for all install surfaces; maskable icons include padding.
const icon = await fs.readFile(path.join(root, 'public/icon.svg'));
for (const size of [192, 512]) await sharp(icon).resize(size, size).png().toFile(path.join(dist, `icon-${size}.png`));
await sharp(icon).resize(180, 180).png().toFile(path.join(dist, 'apple-touch-icon.png'));
const name = data.title;
const manifest = { id: '/', name, short_name: 'OpenLearn', description: 'Free books on technology, mathematics, and engineering. Read online or save a book for offline reading.', start_url: '/', scope: '/', display: 'standalone', background_color: '#f3f7fb', theme_color: '#163d54', lang: 'en', icons: [{ src: '/icon-192.png', sizes: '192x192', type: 'image/png', purpose: 'any maskable' }, { src: '/icon-512.png', sizes: '512x512', type: 'image/png', purpose: 'any maskable' }] };
await fs.writeFile(path.join(dist, 'manifest.webmanifest'), JSON.stringify(manifest));
const shell = ['/', '/offline/', '/app/catalog.json', '/app/offline.json', '/manifest.webmanifest', '/icon-192.png', '/icon-512.png', '/apple-touch-icon.png', '/icon.svg', ...shared.map(toURL)];
const worker = workerSource.replace('__OPENLEARN_BUILD__', build).replace('__OPENLEARN_SHELL__', JSON.stringify(shell));
await fs.writeFile(path.join(dist, 'sw.js'), worker);
console.log(`PWA ${build}: ${Object.keys(courses).length} offline courses, ${shell.length} shell resources`);
