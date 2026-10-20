// Concept previews: every section's "Key terms" list is the glossary.
//
// A term is defined once, in the Key terms of the section that teaches it.
// When another section uses that term, the website links its first use to a
// preview of the definition and lists it under "Concepts from other sections"
// at the top of the page, with a link to the section that teaches it in full.
// So each section explains what it borrows, and says where to read more,
// without anyone copying definitions between books.
//
// Nothing here changes the Markdown sources; GitHub shows them as written.
// The PDF prints the same "Concepts from other sections" lists by asking
// concepts-cli.mjs, so both use this one matcher.

import path from 'node:path';

// Single words are linked only when they cannot be mistaken for plain
// English: acronyms, hyphenated or numbered names, and the words below. A
// word like "batch", "identity", or "kernel" means different things in
// different books, so it is left as text.
const WORDS = new Set([
	'backpressure', 'backpropagation', 'backtracking', 'bijection', 'bijective', 'coroutine', 'cotangent', 'covariance', 'currying',
	'deadlock', 'determinant', 'discriminant', 'dropout', 'eigenvalue', 'eigenvector', 'entropy', 'fsync',
	'goroutine', 'gradcheck', 'heapify', 'hessian', 'idempotence', 'idempotency', 'injective', 'involution',
	'jacobian', 'lakehouse', 'likelihood', 'linearizability', 'logits', 'memoization', 'microscaling', 'monomorphization',
	'orthonormal', 'patchify', 'perplexity', 'quantization', 'quickselect', 'quorum', 'regularization', 'roofline',
	'safetensors', 'semaphore', 'sigmoid', 'singleflight', 'softplus', 'subgradient', 'subnormal', 'surprisal',
	'tombstone', 'trie', 'upsert', 'variance', 'varint', 'viterbi', 'vtable',
]);

// Phrases too common to be worth a preview wherever they appear.
const PHRASES = new Set([
	'software engineering', 'file format', 'false positive', 'false alarm', 'status code', 'exit status', 'personal data',
	'work directory', 'error path', 'unknown id', 'unknown field', 'peak memory', 'hidden state', 'step function',
]);

// Field guides apply the books; a book's page should not send its reader to one.
const LEAF_BOOKS = new Set(['interview-preparation', 'field-engineering']);
// When a term is taught in several books, prefer the more foundational one.
const FOUNDATIONS = ['mathematics', 'programming-and-algorithms', 'software-engineering', 'data-engineering', 'systems', 'ai-engineering'];

const bookOf = (file) => file.split('/')[2] ?? '';
const chapterOf = (file) => file.split('/').slice(0, 4).join('/');
export const slug = (s) => s.toLowerCase().replace(/<[^>]+>|[`*$\\]/g, '').replace(/[^\p{L}\p{N}]+/gu, '-').replace(/^-|-$/g, '');

// "Eigenvector / eigenvalue" and "Server-sent events (SSE)" each name the
// concept more than one way.
export function aliasesOf(label) {
	const names = [];
	for (const part of label.split(/\s+\/\s+/)) {
		const short = part.match(/\(([^)]+)\)/)?.[1];
		const base = part.replace(/\s*\([^)]*\)/g, '').trim();
		names.push(base);
		if (short && /^[A-Z][\w-]*$/.test(short)) names.push(short);
	}
	return names.filter((name) => name.length >= 3 && !/[`$\\]/.test(name));
}

export function linkable(alias) {
	if (PHRASES.has(alias.toLowerCase())) return false;
	if (/\s/.test(alias)) return true;
	if (/\d|-|^[A-Z]{2,}|[a-z][A-Z]/.test(alias)) return true;
	return WORDS.has(alias.toLowerCase());
}

/** The entries of a page's `## Key terms` list: `- **Term**: definition`. */
export function harvest(file, text) {
	const section = text.match(/^## Key terms[ \t]*\n([\s\S]*?)(?=^## |(?![\s\S]))/m);
	if (!section) return [];
	const entries = [];
	let fence = false;
	for (const line of section[1].split('\n')) {
		if (/^\s*(```|~~~)/.test(line)) fence = !fence;
		const item = !fence && line.match(/^[-*]\s+\*\*(.+?)\*\*\s*[:.—–-]?\s*(.*)$/);
		if (!item) continue;
		const label = item[1].replace(/[:.]$/, '').trim();
		const definition = item[2].replace(/^[:—–-]\s*/, '').trim();
		if (definition) entries.push({ file, label, definition, aliases: aliasesOf(label).filter(linkable) });
	}
	return entries;
}

const WORD = /[\p{L}\p{N}][\p{L}\p{N}_'’-]*/gu;
const escapeRegex = (s) => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');

/** Index every linkable alias, and the one pattern that finds them in prose. */
export function buildIndex(entries, order = new Map()) {
	const byAlias = new Map();
	entries.forEach((entry, i) => {
		entry.order = order.get(entry.file) ?? Number.MAX_SAFE_INTEGER;
		entry.seq = i;
		for (const alias of entry.aliases) {
			const key = alias.toLowerCase();
			if (!byAlias.has(key)) byAlias.set(key, { forms: new Set(), entries: [] });
			byAlias.get(key).forms.add(alias);
			byAlias.get(key).entries.push(entry);
		}
	});
	// Look up candidates by first word, then confirm with a sticky pattern.
	const byFirstWord = new Map();
	for (const [key, { forms }] of byAlias) {
		const first = key.match(WORD)?.[0];
		if (!first) continue;
		const sticky = new RegExp(`${escapeRegex(key)}(?:e?s)?(?![\\p{L}\\p{N}_\\-])`, 'iuy');
		if (!byFirstWord.has(first)) byFirstWord.set(first, []);
		byFirstWord.get(first).push({ key, sticky });
	}
	for (const list of byFirstWord.values()) list.sort((a, b) => b.key.length - a.key.length);
	const pattern = byFirstWord.size ? byFirstWord : null;
	return { byAlias, pattern };
}

/** The definition a page should show for an alias, or none. */
export function senseFor(candidates, page) {
	// A page that defines the term explains it itself.
	if (candidates.some((e) => e.file === page)) return null;
	const book = bookOf(page);
	// The section titled after a term is the one that teaches it.
	const titled = (e) => e.aliases.some((alias) => new RegExp(`(?<![\\p{L}\\p{N}])${escapeRegex(alias)}`, 'iu').test(e.title ?? ''));
	const foundation = (e) => {
		const i = FOUNDATIONS.indexOf(bookOf(e.file));
		return i < 0 ? FOUNDATIONS.length : i;
	};
	// Lower is better: [tier, foundation, reading order, position in the list].
	const rank = (e) => {
		const same = bookOf(e.file) === book;
		if (chapterOf(e.file) === chapterOf(page)) return [0, 0, e.order, e.seq];
		if (!same && LEAF_BOOKS.has(bookOf(e.file))) return null;
		if (titled(e)) return [same ? 1 : 2, same ? 0 : foundation(e), e.order, e.seq];
		return [same ? 3 : 4, same ? 0 : foundation(e), e.order, e.seq];
	};
	const better = (a, b) => {
		for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return a[i] < b[i];
		return false;
	};
	let best = null;
	for (const e of candidates) {
		const r = rank(e);
		if (r && (!best || better(r, best.r))) best = { e, r };
	}
	return best?.e ?? null;
}

// Spans that must never gain a link: code, math, links, HTML, bold (a term
// being introduced), italics (book titles and emphasis), and bare URLs.
const PROTECTED = /(`+)[\s\S]*?\1|\$\$[\s\S]*?\$\$|(?<!\\)\$(?:\\.|[^$\\])+\$|!?\[[^\]]*\]\([^)]*\)|\[[^\]]*\]\[[^\]]*\]|<a\b[\s\S]*?<\/a>|<[^>]+>|\*\*[\s\S]*?\*\*|(?<![\w*])\*(?!\s)[^*\n]+?\*(?![\w*])|(?<!\w)_(?!\s)[^_\n]+?_(?!\w)|https?:\/\/\S+/g;

function linkLine(line, link) {
	let out = '';
	let last = 0;
	for (const m of line.matchAll(PROTECTED)) {
		out += link(line.slice(last, m.index)) + m[0];
		last = m.index + m[0].length;
	}
	return out + link(line.slice(last));
}

/**
 * Link the first use of each borrowed term in a page body (Markdown after
 * sync-content's transform) and return the entries it used, in order.
 */
export function linkConcepts(body, page, index, hrefOf) {
	if (!index.pattern) return { body, used: [] };
	const used = new Map();
	const lines = body.split('\n');
	let fence = null;
	let pre = false;
	let display = false;
	let keyTerms = false;
	const wrap = (match, alias) => {
		const found = index.byAlias.get(alias);
		// Acronyms and CamelCase names match only as written.
		const loose = [...found.forms].some((form) => !/[A-Z].*[A-Z]|[a-z][A-Z]/.test(form));
		if (!loose && ![...found.forms].some((form) => match.startsWith(form))) return null;
		const entry = senseFor(found.entries, page);
		if (!entry || used.has(entry)) return null;
		const taken = new Set(used.values());
		let id = slug(entry.label);
		for (let n = 2; taken.has(id); n++) id = `${slug(entry.label)}-${n}`;
		used.set(entry, id);
		return `<a class="term" data-term="${id}" href="${hrefOf(entry.file)}">${match}</a>`;
	};
	const link = (text) => {
		let out = '';
		let last = 0;
		for (const word of text.matchAll(WORD)) {
			if (word.index < last) continue;
			const w = word[0].toLowerCase();
			// "eigenvalues" and "caches" start the names "eigenvalue" and "cache".
			const candidates = index.pattern.get(w)
				?? (w.endsWith('s') ? index.pattern.get(w.slice(0, -1)) : undefined)
				?? (w.endsWith('es') ? index.pattern.get(w.slice(0, -2)) : undefined)
				?? [];
			for (const { key, sticky } of candidates) {
				sticky.lastIndex = word.index;
				const hit = sticky.exec(text);
				if (!hit) continue;
				const html = wrap(hit[0], key);
				if (!html) break;
				out += text.slice(last, word.index) + html;
				last = word.index + hit[0].length;
				break;
			}
		}
		return out + text.slice(last);
	};
	for (let i = 0; i < lines.length; i++) {
		const line = lines[i];
		const fenceMark = line.match(/^\s*(`{3,}|~{3,})/);
		if (fence) {
			if (fenceMark && fenceMark[1].startsWith(fence)) fence = null;
			continue;
		}
		if (fenceMark) { fence = fenceMark[1]; continue; }
		if (pre) { if (line.includes('</pre>')) pre = false; continue; }
		if (/^\s*<pre\b/.test(line)) { pre = !line.includes('</pre>'); continue; }
		const dollars = (line.match(/\$\$/g) ?? []).length;
		if (display) { if (dollars % 2) display = false; continue; }
		if (/^\s*\$\$/.test(line) && dollars % 2) { display = true; continue; }
		const heading = line.match(/^(#{1,6})\s+(.*)/);
		if (heading) {
			if (heading[1].length <= 2) keyTerms = /^Key terms\s*$/i.test(heading[2]);
			continue;
		}
		if (keyTerms || /^\s*</.test(line) || /^\s*\[[^\]]+\]:/.test(line)) continue;
		lines[i] = linkLine(line, link);
	}
	return { body: lines.join('\n'), used: [...used].map(([entry, id]) => ({ entry, id })) };
}

/**
 * The "Concepts from other sections" box: each borrowed term with its
 * definition and the section that teaches it. The page's term links open
 * these entries as previews; without JavaScript the box is the preview.
 */
export function conceptsBox(used, { page, titleOf, bookTitleOf, hrefOf }) {
	if (!used.length) return '';
	const book = bookOf(page);
	const items = used.map(({ entry, id }) => {
		const where = titleOf(entry.file) ?? path.posix.basename(entry.file, '.md');
		const other = bookOf(entry.file) !== book ? ` in *${bookTitleOf(bookOf(entry.file))}*` : '';
		return `- <span data-concept="${id}"></span>**${entry.label}**: ${entry.definition} <span class="concept-source">Read more: [${where}](${hrefOf(entry.file)})${other}</span>`;
	});
	const count = used.length;
	return [
		'<details class="concepts">',
		`<summary>Concepts from other sections (${count})</summary>`,
		'',
		'This section uses these ideas. Each has a short definition and a link to the section that explains it in full. Tap a dotted term in the text to preview it.',
		'',
		...items,
		'',
		'</details>',
	].join('\n');
}

/** Put the box after the page's opening paragraph, where a reader decides whether they are ready. */
export function withConcepts(body, box) {
	if (!box) return body;
	const opening = body.match(/^\s*(\S[\s\S]*?)(?:\n[ \t]*\n|(?![\s\S]))/);
	// Only after a plain paragraph: a code fence or HTML block may hold blank lines.
	if (!opening || /^(```|~~~|<|#|\||\$\$)/.test(opening[1])) return `${box}\n\n${body}`;
	const end = opening[0].length;
	return `${body.slice(0, end).trimEnd()}\n\n${box}\n\n${body.slice(end)}`;
}
