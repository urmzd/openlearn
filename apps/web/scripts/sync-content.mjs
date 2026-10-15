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
const TRACKS = manifest.tracks.map(({ dir, title }) => [dir, title]);
const EXTRA_TREES = (manifest.reference ?? []).map(({ dir, title }) => [dir, title]);
const START = manifest.start.map(({ file, title }) => [file, title]);
const PATHS = manifest.paths ?? '';
// Of course/, only the contracts are published (course/DESIGN.md 5.16): the
// reference system, course tests, and design notes never reach the site.
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
		.replace(/^content\/courses\/build-an-llm\/(?=p\d|capstones)/, 'ml/08-tinyllm/')
		.replace(/^content\/courses\/computer-science\/README\.md$/, 'CS-CURRICULUM.md')
		.replace(/^content\/courses\/README\.md$/, 'STUDY-PLAN.md')
		.replace(/^content\/references\/sources\.md$/, 'SOURCES.md')
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
		.replace(/[`*_]/g, '')
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
	return `\n\n## Stages\n\nGenerated from \`path.tsv\`. Track progress locally with \`practice/bin/ol learn\`.\n\n${rows.join('\n')}\n`;
}

// ---------- sidebar ----------

const shortLabel = (t) => t.split(/\s+[—–-]\s+|:\s+/)[0].trim();

function tree(dir, label) {
	const readme = `${dir}/README.md`;
	const files = sources.filter((f) => path.posix.dirname(f) === dir && f !== readme).sort();
	const subdirs = [...dirs]
		.filter((d) => path.posix.dirname(d) === dir && sources.some((f) => f.startsWith(`${d}/`)))
		// Capstones close a spine, so they sort after every numbered part.
		.sort((a, b) => (a.endsWith('/capstones') - b.endsWith('/capstones')) || a.localeCompare(b));
	const items = [];
	if (sources.includes(readme)) items.push({ label: 'Overview', link: navOf(readme) });
	for (const f of files) items.push({ label: shortLabel(titles.get(f)), link: navOf(f) });
	for (const d of subdirs) items.push(tree(d, null));
	const name = label ?? shortLabel(titles.get(readme) ?? path.posix.basename(dir));
	if (items.length === 1 && sources.includes(readme)) return { label: name, link: navOf(readme) };
	return { label: name, collapsed: true, items };
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

for (const [f, { title, body }] of pages) {
	const extra = publications.some((p) => p.manifest.paths && f.startsWith(`${p.manifest.paths}/`)) && path.posix.basename(f) === 'README.md' ? stageTable(path.posix.dirname(f)) : '';
	const front = [
		'---',
		`title: ${JSON.stringify(title)}`,
		`editUrl: ${JSON.stringify(`${GITHUB}/edit/main/${f}`)}`,
		'---',
		'',
	].join('\n');
	const isIndex = f.endsWith('/README.md');
	const dest = path.join(DOCS, `${slugOf(f)}${isIndex ? '/index' : ''}.md`);
	fs.mkdirSync(path.dirname(dest), { recursive: true });
	fs.writeFileSync(dest, front + body + extra);
}

// Course landing pages contain navigation and downloads, never copied lessons.
const releases = `${GITHUB}/releases/latest/download`;
const courseItems = [];
for (const { manifest: course, reading } of publications) {
	const url = `/learn/${course.id}/`;
	courseItems.push({ label: course.title, link: url });
	const rows = [];
	for (const section of reading) {
		for (const f of section.files) rows.push(`- [${titles.get(f) ?? section.title}](${sources.includes(f) ? routeOf(f) : `${GITHUB}/blob/main/${f}`})`);
	}
	const body = [
		'---', `title: ${JSON.stringify(course.title)}`, '---', '',
		`[Download PDF](${releases}/openlearn-${course.id}.pdf) · [Download course pack](${releases}/openlearn-${course.id}.zip)`, '',
		'The reading list below, the PDF, and the portable pack use the same source lessons.', '',
		'## Reading order', '', ...rows, '',
	];
	const dest = path.join(DOCS, 'learn', course.id, 'index.md');
	fs.mkdirSync(path.dirname(dest), { recursive: true });
	fs.writeFileSync(dest, body.join('\n'));
}
const home = [
	'---', `title: ${JSON.stringify(selected ? manifest.title : `${manifest.title} courses`)}`, '---', '',
	'Study on the web, download a PDF, or open a portable course pack in the desktop platform.', '',
	...courseItems.map(({ label, link }) => `- [${label}](${link})`), '',
];
fs.writeFileSync(path.join(DOCS, 'index.md'), home.join('\n'));
const sidebar = [
	{ label: 'Courses', items: courseItems },
	{ label: 'Start here', items: START.filter(([f]) => sources.includes(f)).map(([f, label]) => ({ label, link: navOf(f) })) },
	{ label: 'Lessons', items: TRACKS.filter(([d]) => sources.some((f) => f.startsWith(`${d}/`))).map(([d, label]) => tree(d, label)) },
	...EXTRA_TREES.filter(([d]) => sources.some((f) => f.startsWith(`${d}/`))).map(([d, label]) => tree(d, label)),
];
fs.mkdirSync(GENERATED, { recursive: true });
fs.writeFileSync(path.join(GENERATED, 'sidebar.json'), `${JSON.stringify(sidebar, null, '\t')}\n`);
fs.writeFileSync(path.join(GENERATED, 'course.json'), JSON.stringify({ title: manifest.title, description: manifest.tagline ?? manifest.title }));
console.log(`synced ${pages.size} unique source pages for ${publications.length} courses`);
