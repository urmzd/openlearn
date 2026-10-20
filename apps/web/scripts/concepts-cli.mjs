// Print the concepts each section borrows, for the PDF build.
//
// The website and the PDF must agree on which terms a section uses, so the
// PDF asks this script instead of matching terms itself. Terms are indexed
// across the whole library (a mathematics term is borrowed by AI Engineering);
// the output covers the selected course.
//
// Usage: node apps/web/scripts/concepts-cli.mjs [--course ID]
// Prints: { "<file>": [{ "label", "definition", "home", "homeTitle", "book", "bookTitle" }] }

import { execFileSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { buildIndex, harvest, linkConcepts } from './concepts.mjs';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../../..');
const course = process.argv.includes('--course') ? process.argv[process.argv.indexOf('--course') + 1] : undefined;
const publications = JSON.parse(execFileSync('python3', ['scripts/course_catalog.py', 'index'], { cwd: ROOT, encoding: 'utf8', maxBuffer: 64 * 1024 * 1024 }));

const heading = (text) => (text.match(/^#\s+(.+)$/m)?.[1] ?? '').replace(/[`*]/g, '').replace(/!?\[([^\]]*)\]\([^)]*\)/g, '$1').trim();
const sources = new Map();
const titles = new Map();
const order = new Map();
const sections = new Set();
const bookTitles = new Map(publications.map((p) => [p.manifest.id, p.manifest.title]));
for (const { reading } of publications) {
	for (const section of reading) {
		for (const file of section.files) {
			if (!file.endsWith('.md') || sources.has(file)) continue;
			sources.set(file, fs.readFileSync(path.join(ROOT, file), 'utf8'));
			order.set(file, order.size);
			const label = section.labels?.[file];
			const title = heading(sources.get(file)) || section.title;
			if (!label) { titles.set(file, title); continue; }
			sections.add(file);
			const word = section.kind === 'appendix' ? 'Appendix' : 'Chapter';
			titles.set(file, label === section.number ? `${word} ${label}: ${title}` : `${label} ${title}`);
		}
	}
}

const index = buildIndex([...sources].flatMap(([file, text]) => harvest(file, text).map((e) => ({ ...e, title: titles.get(file) }))), order);
const wanted = course ? new Set(publications.find((p) => p.manifest.id === course)?.documents ?? []) : sections;
const out = {};
for (const file of sections) {
	if (!wanted.has(file)) continue;
	const { used } = linkConcepts(sources.get(file), file, index, () => '#');
	if (!used.length) continue;
	out[file] = used.map(({ entry }) => {
		const book = entry.file.split('/')[2];
		return { label: entry.label, definition: entry.definition, home: entry.file, homeTitle: titles.get(entry.file), book, bookTitle: bookTitles.get(book) ?? book };
	});
}
process.stdout.write(JSON.stringify(out));
