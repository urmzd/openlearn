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
import { courseOutlines, chapterNeighbors } from './course-navigation.mjs';

const SITE = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const ROOT = path.resolve(SITE, '../..');
const DOCS = path.join(SITE, 'src/content/docs');
const ASSETS = path.join(SITE, 'public/repo');
const GENERATED = path.join(SITE, 'src/generated');
const BASE = '';
const GITHUB = 'https://github.com/urmzd/openlearn';

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
	// Keep published URLs stable when source directories move.
	const publicPath = repoPath
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
		return `${BASE}/repo/${clean}`;
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
	const dest = path.join(ASSETS, repoPath);
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
	if (!title) title = path.posix.basename(path.posix.dirname(repoPath)) || 'Supersource';
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

for (const [f, { title, body }] of pages) {
	const extra = publications.some((p) => p.manifest.paths && f.startsWith(`${p.manifest.paths}/`)) && path.posix.basename(f) === 'README.md' ? stageTable(path.posix.dirname(f)) : '';
	const owners = chapterNavigation.get(f) ?? [];
	// A shared page has no single next chapter. Return readers to their course
	// rather than following the next occurrence in the catalog sidebar.
	const navigation = owners.length === 1 ? owners[0] : { prev: false, next: false };
	const returnLinks = owners.length > 1
		? `\n\n## Continue in your course\n\n${owners.map(({ course }) => `- [${course.title}](/learn/${course.id}/)`).join('\n')}\n`
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

// Course landing pages and the sidebar use the same ordered chapter outline.
const releases = `${GITHUB}/releases/latest/download`;
// Inside a chapter's group the opening page is its introduction; the group
// label already carries the chapter number and title.
const pageLabel = (section, file) => (isOpening(section, file) ? 'Introduction' : titles.get(file));
const sectionItem = (section) => section.files.length === 1
	? { label: chapterLabel(section), link: navOf(section.files[0]) }
	: { label: chapterLabel(section), collapsed: true, items: section.files.map((file) => ({ label: pageLabel(section, file), link: navOf(file) })) };
const sectionMarkdown = (sections) => sections.flatMap((section) => section.files.length === 1
	? [`- [${chapterLabel(section)}](${routeOf(section.files[0])})`, '']
	: [`### ${chapterLabel(section)}`, '', ...section.files.map((file) => `- [${pageLabel(section, file)}](${routeOf(file)})`), '']);
const courseItems = [];
const courseGroups = [];
for (const { course, chapters, references } of outlines) {
	const url = `/learn/${course.id}/`;
	courseItems.push({ label: course.title, link: url });
	courseGroups.push({
		label: course.title,
		collapsed: false,
		items: [
			{ label: 'Contents', link: url },
			...chapters.map(sectionItem),
			...(references.length ? [{ label: 'References', collapsed: true, items: references.map(sectionItem) }] : []),
		],
	});
	const body = [
		'---', `title: ${JSON.stringify(course.title)}`, 'prev: false', 'next: false', '---', '',
		"import CourseActions from '../../../../components/CourseActions.astro';", '',
		...(course.tagline ? [course.tagline, ''] : []),
		`<CourseActions id="${course.id}" />`, '',
		'## Contents', '', ...sectionMarkdown(chapters),
		...(references.length ? [
			'<details>', '<summary>References</summary>', '',
			...sectionMarkdown(references), '</details>', '',
		] : []),
		`[PDF](${releases}/openlearn-${course.id}.pdf) · [Source pack](${releases}/openlearn-${course.id}.zip)`, '',
	];
	const dest = path.join(DOCS, 'learn', course.id, 'index.mdx');
	fs.mkdirSync(path.dirname(dest), { recursive: true });
	fs.writeFileSync(dest, body.join('\n'));
}
// A book's one-line description: the first sentences of its front matter.
function blurb(file) {
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
			if (out.length >= 90) break;
			out = out ? `${out} ${sentence}` : sentence;
		}
		return out.length > 240 ? `${out.slice(0, out.lastIndexOf(' ', 237))}…` : out;
	}
	return undefined;
}
const library = {
	title: manifest.title,
	courses: [...publications].sort((a, b) => a.manifest.title.localeCompare(b.manifest.title, 'en', { sensitivity: 'base' })).map(({ manifest: course, reading }) => ({
		id: course.id,
		title: course.title,
		description: course.tagline ?? blurb(course.start?.[0]?.file) ?? course.tracks.map((t) => t.title).slice(0, 3).join(', '),
		href: `/learn/${course.id}/`,
		// `chapter` groups lessons so the reader can scroll through one chapter
		// and pause at the next chapter's opening.
		lessons: reading.flatMap((section, i) => section.files.filter((f) => sources.includes(f)).map((f) => ({
			key: f, title: titles.get(f), url: routeOf(f), chapter: i, chapterTitle: chapterLabel(section),
		}))),
	})),
};
fs.mkdirSync(GENERATED, { recursive: true });
fs.writeFileSync(path.join(GENERATED, 'library.json'), JSON.stringify(library));
fs.mkdirSync(path.join(SITE, 'public/app'), { recursive: true });
fs.writeFileSync(path.join(SITE, 'public/app/catalog.json'), JSON.stringify(library));
// The home page is the shelf: your books, where you left off, and your notes.
const home = [
	'---', `title: ${JSON.stringify(selected ? manifest.title : 'Your books')}`, 'description: Free books on technology, mathematics, and engineering. Read online or save a book for offline reading.', 'template: splash', '---', '',
	"import Shelf from '../../components/Shelf.astro';", '', '<Shelf />', '',
];
fs.writeFileSync(path.join(DOCS, 'index.mdx'), home.join('\n'));

// One group per book; src/routeData.ts shows only the group of the book
// being read.
const sidebar = courseGroups;
fs.mkdirSync(GENERATED, { recursive: true });
fs.writeFileSync(path.join(GENERATED, 'sidebar.json'), `${JSON.stringify(sidebar, null, '\t')}\n`);
fs.writeFileSync(path.join(GENERATED, 'course.json'), JSON.stringify({ title: manifest.title, description: manifest.tagline ?? manifest.title }));
console.log(`synced ${pages.size} unique source pages for ${publications.length} courses`);
