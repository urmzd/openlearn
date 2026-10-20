// Turn the repo's markdown into Starlight pages.
//
// The READMEs stay the source of truth: this script copies them into
// src/content/docs at build time, never the other way. It does four things a
// plain copy cannot:
//
// 1. Gives every page frontmatter (title from the first H1, which is then
//    dropped because Starlight renders the title itself).
// 2. Rewrites relative links: markdown -> site routes, images -> /repo/ assets,
//    everything else (code, configs) -> the file on GitHub.
// 3. Turns ```mermaid fences into <pre class="mermaid"> for the client renderer.
// 4. Builds the sidebar (src/generated/sidebar.json) in learning order, and
//    appends a generated stage table to each role path from its path.tsv.
//
// Usage: node scripts/sync-content.mjs   (run from apps/web/, wired into dev/build)

import { execFileSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { courseOutlines, chapterNeighbors, chapterParts } from './course-navigation.mjs';
import { stableSourcePath } from './legacy-source-paths.mjs';
import { buildIndex, conceptsBox, harvest, linkConcepts, withConcepts } from './concepts.mjs';

const SITE = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const ROOT = path.resolve(SITE, '../..');
const DOCS = path.join(SITE, 'src/content/docs');
const ASSETS = path.join(SITE, 'public/repo');
const GENERATED = path.join(SITE, 'src/generated');
const BASE = '';
const GITHUB = 'https://github.com/urmzd/openlearn';
// Verified release assets: omit downloads for books without a published PDF.
const releaseAssets = JSON.parse(fs.readFileSync(path.join(SITE, 'release-assets.json'), 'utf8'));
const pdfOf = (id) => releaseAssets.find((asset) => asset.name === `openlearn-${id}.pdf`)?.url;

const selected = process.env.OPENLEARN_COURSE;
const courseArgs = ['scripts/course_catalog.py', 'index', ...(selected ? ['--course', selected] : [])];
const publications = JSON.parse(execFileSync('python3', courseArgs, { cwd: ROOT, encoding: 'utf8', maxBuffer: 16 * 1024 * 1024 }));
const manifest = selected ? publications[0].manifest : JSON.parse(fs.readFileSync(path.join(ROOT, 'openlearn.json'), 'utf8'));
// Of course/, only the contracts are published as pages. Chapters quote the
// solution code they explain, and links into course/solutions/ resolve to GitHub.
const SKIP = /^(apps\/|packs\/|\.github\/|skills\/|archive\/|course\/(?!contracts\/)|CHANGELOG\.md|CODE_OF_CONDUCT\.md|SECURITY\.md)|\/tests\/fixtures\//;
const IMAGE = /\.(svg|png|jpe?g|gif|webp)$/i;

const tracked = [...new Set(execFileSync('git', ['ls-files', '--cached', '--others', '--exclude-standard'], { cwd: ROOT, encoding: 'utf8' })
	.split('\n')
	.filter((f) => f && fs.existsSync(path.join(ROOT, f))))];
const trackedSet = new Set(tracked);
const dirs = new Set();
for (const f of tracked) {
	for (let d = path.posix.dirname(f); d !== '.'; d = path.posix.dirname(d)) dirs.add(d);
}
const sources = [...new Set(publications.flatMap((p) => p.documents).concat(selected ? [] : ['README.md', 'content/courses/README.md']))].filter((f) => !SKIP.test(f));
for (const f of sources) {
	if (!trackedSet.has(f)) throw new Error(`Unpublishable course document: ${f}`);
}

// ---------- routes ----------

function slugOf(repoPath) {
	if (repoPath === 'README.md') return 'overview';
	const canonical = repoPath.match(/^content\/courses\/(ai-engineering|software-engineering|programming-and-algorithms|mathematics)\/README\.md$/);
	if (canonical) return canonical[1];
	// Keep published URLs stable when source directories move.
	const publicPath = stableSourcePath(repoPath)
		.replace(/^content\/courses\/build-an-llm\/paths\//, 'paths/')
		.replace(/^content\/courses\/build-an-llm\/concepts\.md$/, 'ml/08-tinyllm/README.md')
		.replace(/^content\/courses\/build-an-llm\/(?=\d\d-|capstones)/, 'ml/08-tinyllm/')
		.replace(/^content\/courses\/computer-science\/README\.md$/, 'CS-CURRICULUM.md')
		.replace(/^content\/courses\/README\.md$/, 'STUDY-PLAN.md')
		.replace(/^content\/references\/sources\.md$/, 'SOURCES.md')
		.replace(/^content\/courses\/diagramming\/01-diagramming-c4\//, 'software-craftsmanship/05-diagramming-c4/')
		.replace(/^content\/courses\/diagramming\/02-documentation-writing\//, 'software-craftsmanship/06-documentation-writing/')
		.replace(/^content\/courses\/programming-languages\/01-type-systems\//, 'software-craftsmanship/07-type-systems/')
		.replace(/^content\/courses\/programming-languages\/02-language-and-tool-primers\//, 'software-craftsmanship/12-language-and-tool-primers/')
		.replace(/^content\/courses\/software-craftsmanship\/05-code-review-and-ci\//, 'software-craftsmanship/08-code-review-and-ci/')
		.replace(/^content\/courses\/software-craftsmanship\/06-releases-and-deprecation\//, 'software-craftsmanship/09-releases-and-deprecation/')
		.replace(/^content\/courses\/software-craftsmanship\/07-maintenance\//, 'software-craftsmanship/10-maintenance/')
		.replace(/^content\/courses\/software-craftsmanship\/08-security\//, 'software-craftsmanship/11-security/')
		.replace(/^content\/courses\/mathematics-for-computing\//, 'math/')
		.replace(/^content\/courses\/machine-learning\/(?!openlearn|study-plan)/, 'ml/')
		.replace(/^content\/courses\/interview-preparation\/(?!openlearn)/, 'interviews/')
		.replace(/^content\/courses\/case-studies\/(?!study-plan|openlearn)/, 'case-studies/')
		.replace(/^content\/courses\/(ai-platform-engineering|algorithms|competitive-programming|data-engineering|field-engineering|infrastructure|responsible-ai|software-craftsmanship|systems)\/(?!study-plan)/, '$1/')
		.replace(/^content\/tracks\//, '')
		.replace(/^content\//, '');
	const noExt = publicPath.replace(/\.md$/, '');
	const slug = noExt.endsWith('README') ? path.posix.dirname(noExt) : noExt;
	return slug.toLowerCase();
}

const routeOf = (repoPath) => `${BASE}/${slugOf(repoPath)}/`;
// Starlight prefixes sidebar links with the base itself.
const navOf = (repoPath) => `/${slugOf(repoPath)}/`;

// Resolve a relative link target from a source file to a URL.
function rewriteTarget(from, target) {
	if (/^([a-z]+:|#|\/|mailto:)/i.test(target) || target.startsWith('{')) return target;
	const [rawPath, hash = ''] = target.split('#');
	const frag = hash ? `#${hash}` : '';
	if (!rawPath) return target;
	const resolved = path.posix.normalize(path.posix.join(path.posix.dirname(from), decodeURI(rawPath)));
	const clean = resolved.replace(/\/$/, '');
	if (clean === '.' || clean === '') return routeOf('README.md') + frag;
	if (IMAGE.test(clean) && trackedSet.has(clean)) {
		copyAsset(clean);
		return `${BASE}/repo/${stableSourcePath(clean)}`;
	}
	if (clean.endsWith('.md') && sources.includes(clean)) return routeOf(clean) + frag;
	if (dirs.has(clean)) {
		const readme = `${clean}/README.md`;
		if (sources.includes(readme)) return routeOf(readme) + frag;
		return `${GITHUB}/tree/main/${clean}`;
	}
	return `${GITHUB}/blob/main/${clean}${frag}`;
}

function copyAsset(repoPath) {
	const dest = path.join(ASSETS, stableSourcePath(repoPath));
	fs.mkdirSync(path.dirname(dest), { recursive: true });
	fs.copyFileSync(path.join(ROOT, repoPath), dest);
}

// ---------- markdown transforms ----------

const escapeHtml = (s) => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
const stripInline = (s) =>
	s
		.replace(/!?\[([^\]]*)\]\([^)]*\)/g, '$1')
		.replace(/[`*]/g, '')
		// Underscores mark emphasis only at word edges; keep snake_case names.
		.replace(/(^|[\s(])_+|_+(?=[\s).,:;!?]|$)/g, '$1')
		.replace(/<[^>]+>/g, '')
		.trim();

// remark-math pairs any two `$` on a line, so "$7 per hour, against $15" becomes
// math. Apply pandoc's rule instead: an inline opener is followed by a non-space,
// a closer follows a non-space and is not followed by a digit. An opener with no
// valid closer is a literal dollar and is escaped. `$$`, `\$`, and code spans
// are left alone.
function escapeLoneDollars(line) {
	return line
		.split(/(`+[^`]*`+)/)
		.map((part, i) => {
			if (i % 2 === 1 || !part.includes('$')) return part;
			let out = '';
			let k = 0;
			while (k < part.length) {
				const c = part[k];
				if (c === '\\') {
					out += part.slice(k, k + 2);
					k += 2;
					continue;
				}
				if (c !== '$') {
					out += c;
					k++;
					continue;
				}
				if (part[k + 1] === '$') {
					const end = part.indexOf('$$', k + 2);
					const stop = end < 0 ? part.length : end + 2;
					out += part.slice(k, stop);
					k = stop;
					continue;
				}
				let close = -1;
				if (part[k + 1] && !/\s/.test(part[k + 1])) {
					for (let j = k + 1; j < part.length; j++) {
						if (part[j] === '\\') {
							j++;
							continue;
						}
						if (part[j] === '$') {
							if (!/\s/.test(part[j - 1]) && !/\d/.test(part[j + 1] ?? '')) close = j;
							break;
						}
					}
				}
				if (close < 0) {
					out += '\\$';
					k++;
					continue;
				}
				out += part.slice(k, close + 1);
				k = close + 1;
			}
			return out;
		})
		.join('');
}

function transform(repoPath, text) {
	if (repoPath === 'README.md') {
		// The GitHub header block is replaced by the site's own home page.
		text = text.replace(/^<p align="center">[\s\S]*?\n<\/p>\n/, '');
	}
	const lines = text.split('\n');
	const out = [];
	let title = null;
	let fence = null; // the opening fence marker while inside a code block
	let mermaid = null;

	for (const line of lines) {
		const open = line.match(/^(\s*)(`{3,}|~{3,})\s*([\w-]*)/);
		if (fence) {
			if (open && open[2].startsWith(fence) && !line.trim().slice(fence.length).trim()) {
				fence = null;
				if (mermaid) {
					out.push(`<pre class="mermaid">${escapeHtml(mermaid.join('\n'))}</pre>`);
					mermaid = null;
					continue;
				}
			} else if (mermaid) {
				// A blank line would end the HTML block in markdown.
				if (line.trim()) mermaid.push(line);
				continue;
			}
			out.push(line);
			continue;
		}
		if (open) {
			fence = open[2];
			if (open[3] === 'mermaid') {
				mermaid = [];
				continue;
			}
			out.push(line);
			continue;
		}
		if (title === null) {
			const h1 = line.match(/^#\s+(.+)/);
			if (h1) {
				title = stripInline(h1[1]);
				continue;
			}
		}
		out.push(
			escapeLoneDollars(line)
				.replace(/(!?\[[^\]]*\]\()([^)\s]+)((?:\s+"[^"]*")?\))/g, (_, a, t, b) => a + rewriteTarget(repoPath, t) + b)
				.replace(/^(\[[^\]]+\]:\s*)(\S+)/, (_, a, t) => a + rewriteTarget(repoPath, t))
				.replace(/\b(href|src)="([^"]+)"/g, (_, attr, t) => `${attr}="${rewriteTarget(repoPath, t)}"`),
		);
	}
	if (!title) title = path.posix.basename(path.posix.dirname(repoPath)) || 'Openlearn';
	if (repoPath === 'README.md') title = 'Overview';
	return { title, body: out.join('\n') };
}

// ---------- role paths ----------

function stageTable(pathDir) {
	const tsv = path.join(ROOT, pathDir, 'path.tsv');
	if (!fs.existsSync(tsv)) return '';
	const rows = ['| Stage | Read | Done when |', '|---|---|---|'];
	for (const line of fs.readFileSync(tsv, 'utf8').split('\n')) {
		if (!line.trim() || line.startsWith('#')) continue;
		if (line.startsWith('@')) {
			const name = line.slice(1).trim();
			const readme = `${path.posix.dirname(pathDir)}/${name}/README.md`;
			rows.push(`| Part | [${titles.get(readme) ?? name}](${routeOf(readme)}) | Every stage of that path |`);
			continue;
		}
		const [stage, title, files = '', done = ''] = line.split('\t');
		const links = files
			.split(',')
			.map((f) => f.trim())
			.filter(Boolean)
			.map((f) => (sources.includes(f) ? `[${titles.get(f) ?? f}](${routeOf(f)})` : `\`${f}\``))
			.join('<br />');
		rows.push(`| ${stage} | **${title}**<br />${links} | ${done.replace(/\|/g, '\\|')} |`);
	}
	return `\n\n## Stages\n\nGenerated from \`path.tsv\`.\n\n${rows.join('\n')}\n`;
}

// ---------- main ----------

fs.rmSync(ASSETS, { recursive: true, force: true });
fs.mkdirSync(DOCS, { recursive: true });
for (const entry of fs.readdirSync(DOCS)) {
	fs.rmSync(path.join(DOCS, entry), { recursive: true, force: true });
}

const titles = new Map();
const pages = new Map();
for (const f of sources) {
	const page = transform(f, fs.readFileSync(path.join(ROOT, f), 'utf8'));
	titles.set(f, page.title);
	pages.set(f, page);
}

const outlines = courseOutlines(publications, (file) => pages.has(file));
const chapterNavigation = chapterNeighbors(outlines, navOf);

// Number pages by their place in the book: "Chapter 5: ...", "5.3 ...",
// "Appendix A: ...". The catalog derives the numbers from manifest order.
const chapterLabel = ({ title, number, kind }) => {
	if (kind === 'appendix') return `Appendix ${number}: ${title}`;
	if (kind === 'chapter') return `${number}. ${title}`;
	return title;
};
const isOpening = (chapter, file) => chapter.number && chapter.labels[file] === chapter.number;
for (const { chapters } of outlines) {
	for (const chapter of chapters) {
		for (const file of chapter.files) {
			const label = chapter.labels?.[file];
			const page = pages.get(file);
			if (!label || !page) continue;
			const word = chapter.kind === 'appendix' ? 'Appendix' : 'Chapter';
			page.title = isOpening(chapter, file) ? `${word} ${label}: ${page.title}` : `${label} ${page.title}`;
			titles.set(file, page.title);
		}
	}
}

// Concept previews. Every section's Key terms list is the glossary: a section
// that uses a term taught elsewhere links its first use to a preview and
// lists it at the top, with a link to the section that teaches it in full.
const readingOrder = new Map();
const sectionPages = new Set();
for (const { chapters } of outlines) {
	for (const chapter of chapters) {
		for (const file of chapter.files) {
			if (!readingOrder.has(file)) readingOrder.set(file, readingOrder.size);
			if (chapter.labels?.[file]) sectionPages.add(file);
		}
	}
}
const conceptEntries = [...pages].flatMap(([file, page]) => harvest(file, page.body).map((entry) => ({ ...entry, title: page.title })));
const conceptIndex = buildIndex(conceptEntries, readingOrder);
const bookTitles = new Map(publications.map((p) => [p.manifest.id, p.manifest.title]));
let linkedConcepts = 0;
for (const file of sectionPages) {
	const page = pages.get(file);
	if (!page) continue;
	const { body, used } = linkConcepts(page.body, file, conceptIndex, routeOf);
	linkedConcepts += used.length;
	page.body = withConcepts(body, conceptsBox(used, {
		page: file,
		titleOf: (f) => titles.get(f),
		bookTitleOf: (id) => bookTitles.get(id) ?? id,
		hrefOf: routeOf,
	}));
}

// A book's front page (its README) is the book's landing page: a Start or
// Continue button, PDF download, and contents, generated here so no
// overview or study plan has to repeat the chapter list.
const bookHome = new Map();
const landing = new Map();
for (const { course, chapters } of outlines) {
	const home = course.start?.[0]?.file;
	if (!home || !pages.has(home)) continue;
	bookHome.set(course.id, home);
	const numbered = chapters.filter((c) => c.number);
	const first = numbered[0]?.files[0];
	const total = chapters.reduce((n, c) => n + c.files.length, 0);
	const actions = [
		`<section class="book-actions" data-course="${course.id}" aria-label="This book">`,
		`<p><span data-course-progress="${course.id}">${total} sections</span></p>`,
		...(first ? [`<a class="primary" data-begin="${course.id}" href="${routeOf(first)}">Start reading</a>`] : []),
		...(pdfOf(course.id) ? [`<a class="pdf-link" href="${pdfOf(course.id)}">Download PDF</a>`] : []),
		'</section>',
	].join('');
	const contents = chapterParts(numbered).map((part) => `${part.title ? `\n\n### ${part.title}\n\n` : ''}${part.chapters.map((c) => {
		const sections = c.files.slice(1).map((file) => `- [${titles.get(file)}](${routeOf(file)})`).join('\n');
		return `<div class="contents-chapter">\n\n${part.title ? '####' : '###'} [${chapterLabel(c)}](${routeOf(c.files[0])})\n\n${sections}\n\n</div>`;
	}).join('\n\n')}`);
	landing.set(home, {
		actions,
		contents: contents.length ? `\n\n## Contents\n\n${contents.join('\n\n')}\n` : '',
	});
}
const homeOf = (id) => routeOf(bookHome.get(id) ?? 'README.md');
// Actions go after the opening paragraph, so the pitch comes first.
function withActions(body, actions) {
	const blocks = body.split(/\n\s*\n/);
	const i = blocks.findIndex((b) => b.trim());
	blocks.splice(i + 1, 0, actions);
	return blocks.join('\n\n');
}

// Each book's glossary: every term it defines, A to Z, with the section that
// teaches it. Generated from the Key terms lists, so it never drifts.
const glossaries = new Map();
const sortKey = (label) => label.replace(/[`$\\*_]/g, '').toLocaleLowerCase('en');
for (const { manifest: course, documents } of publications) {
	if (!bookHome.has(course.id)) continue;
	const own = new Set(documents);
	const entries = conceptEntries.filter((e) => own.has(e.file))
		.sort((a, b) => sortKey(a.label).localeCompare(sortKey(b.label), 'en') || (readingOrder.get(a.file) ?? 0) - (readingOrder.get(b.file) ?? 0));
	if (!entries.length) continue;
	const groups = new Map();
	for (const e of entries) {
		const first = sortKey(e.label)[0] ?? '#';
		const letter = /[a-z]/.test(first) ? first.toUpperCase() : '#';
		if (!groups.has(letter)) groups.set(letter, []);
		groups.get(letter).push(e);
	}
	const route = `${homeOf(course.id)}glossary/`;
	const body = [
		`Every term *${course.title}* defines, from A to Z, with the section that teaches it. A section that uses a term from elsewhere, in this book or another, also lists it under "Concepts from other sections" at the top of the page.`,
		'',
		'<div class="glossary-filter">',
		`<label for="glossary-filter">Find a term</label>`,
		`<input id="glossary-filter" type="search" autocomplete="off" spellcheck="false" placeholder="e.g. eigenvalue, KV cache" />`,
		`<p id="glossary-count" aria-live="polite">${entries.length} terms</p>`,
		'</div>',
		'',
		`<nav class="glossary-letters" aria-label="Letters">${[...groups.keys()].map((l) => `<a href="#${l === '#' ? 'symbols' : l.toLowerCase()}">${l}</a>`).join('')}</nav>`,
		'',
		...[...groups].flatMap(([letter, list]) => [
			`<section class="glossary-letter">`,
			'',
			`## ${letter === '#' ? 'Symbols' : letter}`,
			'',
			...list.map((e) => `- **${e.label}**: ${e.definition} <span class="concept-source">Taught in [${titles.get(e.file) ?? e.title}](${routeOf(e.file)})</span>`),
			'',
			'</section>',
			'',
		]),
	].join('\n');
	glossaries.set(course.id, { route, count: entries.length });
	const dest = path.join(DOCS, route.replace(/^\//, ''), 'index.md');
	fs.mkdirSync(path.dirname(dest), { recursive: true });
	fs.writeFileSync(dest, ['---', `title: ${JSON.stringify(`Glossary: ${course.title}`)}`, 'editUrl: false', 'prev: false', 'next: false', '---', '', body, ''].join('\n'));
}
for (const [id, { route }] of glossaries) {
	const book = landing.get(bookHome.get(id));
	if (book) book.actions = book.actions.replace('</section>', `<a class="pdf-link" href="${route}">Glossary</a></section>`);
}

for (const [f, { title, body: original }] of pages) {
	const book = landing.get(f);
	const body = book ? withActions(original, book.actions + book.contents) : original;
	const extra = publications.some((p) => p.manifest.paths && f.startsWith(`${p.manifest.paths}/`)) && path.posix.basename(f) === 'README.md' ? stageTable(path.posix.dirname(f)) : '';
	const owners = chapterNavigation.get(f) ?? [];
	// A shared page has no single next chapter. Return readers to their course
	// rather than following the next occurrence in the catalog sidebar.
	const navigation = owners.length === 1 ? owners[0] : { prev: false, next: false };
	const returnLinks = owners.length > 1
		? `\n\n## Continue in your course\n\n${owners.map(({ course }) => `- [${course.title}](${homeOf(course.id)})`).join('\n')}\n`
		: '';
	const front = [
		'---',
		`title: ${JSON.stringify(title)}`,
		`editUrl: ${JSON.stringify(`${GITHUB}/edit/main/${f}`)}`,
		`prev: ${JSON.stringify(navigation.prev)}`,
		`next: ${JSON.stringify(navigation.next)}`,
		'---',
		'',
	].join('\n');
	const isIndex = f.endsWith('/README.md');
	const dest = path.join(DOCS, `${slugOf(f)}${isIndex ? '/index' : ''}.md`);
	fs.mkdirSync(path.dirname(dest), { recursive: true });
	fs.writeFileSync(dest, front + body + extra + returnLinks);
}

// The sidebar uses the same ordered chapter outline as the front pages.
// Inside a chapter's group the opening page is its introduction; the group
// label already carries the chapter number and title.
const pageLabel = (section, file) => (isOpening(section, file) ? 'Introduction' : titles.get(file));
const sectionItem = (section) => section.files.length === 1
	? { label: chapterLabel(section), link: navOf(section.files[0]) }
	: { label: chapterLabel(section), collapsed: true, items: section.files.map((file) => ({ label: pageLabel(section, file), link: navOf(file) })) };
const courseGroups = [];
const redirects = { '/my-learning/': '/' };
const retiredHomes = {
 'ai-engineering': ['ml', 'courses/build-an-llm', 'ai-platform-engineering', 'responsible-ai'],
 'software-engineering': ['software-craftsmanship', 'courses/diagramming'],
 'programming-and-algorithms': ['algorithms', 'courses/programming-languages', 'practice'],
 mathematics: ['math', 'courses/mathematics'],
};
const aliases = JSON.parse(fs.readFileSync(path.join(ROOT, 'openlearn.json'), 'utf8')).course_aliases;
for (const [id, homes] of Object.entries(retiredHomes)) {
 if (bookHome.has(id)) for (const home of homes) redirects[`/${home}/`] = homeOf(id);
}
for (const [oldId, id] of Object.entries(aliases)) {
 if (bookHome.has(id)) redirects[`/learn/${oldId}/`] = homeOf(id);
}
if (pages.has('content/references/worked-projects.md')) {
 redirects['/case-studies/'] = routeOf('content/references/worked-projects.md');
 redirects['/learn/case-studies/'] = routeOf('content/references/worked-projects.md');
}
if (bookHome.has('systems')) {
	redirects['/infrastructure/'] = homeOf('systems');
	redirects['/learn/infrastructure/'] = homeOf('systems');
}
for (const { course, chapters, references } of outlines) {
	courseGroups.push({
		label: course.title,
		collapsed: false,
		items: [
			...chapterParts(chapters).flatMap((part) => part.title ? [{ label: part.title, collapsed: true, items: part.chapters.map(sectionItem) }] : part.chapters.map(sectionItem)),
			...(references.length ? [{ label: 'References', collapsed: true, items: references.map(sectionItem) }] : []),
			...(glossaries.has(course.id) ? [{ label: 'Glossary', link: glossaries.get(course.id).route }] : []),
		],
	});
	// Old course pages point at the book's front page.
	redirects[`/learn/${course.id}/`] = homeOf(course.id);
}
// A book's one-line description: the first sentences of its front matter.
function blurb(file, enough = 90) {
	if (!file || !fs.existsSync(path.join(ROOT, file))) return undefined;
	let fence = false;
	for (const block of fs.readFileSync(path.join(ROOT, file), 'utf8').split(/\n\s*\n/)) {
		const text = block.trim();
		if (text.startsWith('```')) fence = !fence;
		if (fence || !text || /^(#|>|\||<|-|\*|\d+\.|!\[)/.test(text)) continue;
		const plain = stripInline(text.replace(/\s+/g, ' ')).replace(/\$/g, '');
		// A sentence ends at . ! or ? followed by a capital, so "3 a.m., and" stays whole.
		let out = '';
		for (const sentence of plain.split(/(?<=[.!?])\s+(?=[A-Z])/)) {
			if (out.length >= enough) break;
			out = out ? `${out} ${sentence}` : sentence;
		}
		const cap = enough < 90 ? 170 : 240;
		return out.length > cap ? `${out.slice(0, out.lastIndexOf(' ', cap - 3))}…` : out;
	}
	return undefined;
}
const library = {
	title: manifest.title,
	courses: [...publications].sort((a, b) => a.manifest.title.localeCompare(b.manifest.title, 'en', { sensitivity: 'base' })).map(({ manifest: course, reading }) => ({
		id: course.id,
		kind: course.kind ?? 'book',
		title: course.title,
		description: course.tagline ?? blurb(course.start?.[0]?.file) ?? course.tracks.map((t) => t.title).slice(0, 3).join(', '),
		href: homeOf(course.id),
		pdf: pdfOf(course.id),
		// `chapter` groups lessons so the reader can scroll through one chapter
		// and pause at the next chapter's opening.
		lessons: reading.flatMap((section, i) => section.files.filter((f) => sources.includes(f)).map((f) => ({
			key: stableSourcePath(f), title: titles.get(f), url: routeOf(f), chapter: i, chapterTitle: chapterLabel(section),
		}))),
	})),
};
fs.mkdirSync(GENERATED, { recursive: true });
fs.writeFileSync(path.join(GENERATED, 'library.json'), JSON.stringify(library));
fs.mkdirSync(path.join(SITE, 'public/app'), { recursive: true });
fs.writeFileSync(path.join(SITE, 'public/app/catalog.json'), JSON.stringify(library));
// The home page is the shelf: your books, where you left off, and your notes.
const home = [
	'---', `title: ${JSON.stringify(selected ? manifest.title : 'The library')}`, 'description: Free books on technology, mathematics, and engineering. Read online, explore the code, or download a published PDF.', 'template: splash', '---', '',
	"import Shelf from '../../components/Shelf.astro';", '', '<Shelf />', '',
];
fs.writeFileSync(path.join(DOCS, 'index.mdx'), home.join('\n'));

// One group per book; src/routeData.ts shows only the group of the book
// being read.
const sidebar = courseGroups;
fs.mkdirSync(GENERATED, { recursive: true });
fs.writeFileSync(path.join(GENERATED, 'redirects.json'), JSON.stringify(redirects));
fs.writeFileSync(path.join(GENERATED, 'sidebar.json'), `${JSON.stringify(sidebar, null, '\t')}\n`);
fs.writeFileSync(path.join(GENERATED, 'course.json'), JSON.stringify({ title: manifest.title, description: manifest.tagline ?? manifest.title }));
console.log(`synced ${pages.size} unique source pages for ${publications.length} courses, ${conceptIndex.byAlias.size} concept names, ${linkedConcepts} concept links`);
