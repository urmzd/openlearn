import test from 'node:test';
import assert from 'node:assert/strict';
import { aliasesOf, buildIndex, conceptsBox, harvest, linkable, linkConcepts, withConcepts } from './concepts.mjs';

const math = 'content/courses/mathematics/09-linear-algebra/README.md';
const eig = 'content/courses/mathematics/09-linear-algebra/04-eigenvalues.md';
const py = 'content/courses/programming-and-algorithms/17-predict-output/python/README.md';
const lstm = 'content/courses/ai-engineering/09-recurrent-networks/02-lstm.md';
const guide = 'content/courses/interview-preparation/01-shared/README.md';

const keyTerms = (lines) => `# Page\n\nText.\n\n## Key terms\n\n${lines.join('\n')}\n\n## Summary\n\nDone.\n`;
const index = () => buildIndex([
	...harvest(math, keyTerms(['- **Determinant**: the factor by which $A$ scales volume.', '- **Identity matrix**: ones on the diagonal.'])),
	...harvest(eig, keyTerms(['- **Eigenvector / eigenvalue**: $Av = \\lambda v$.', '- **Spectral radius**: the largest $|\\lambda|$.'])),
	...harvest(py, keyTerms(['- **Identity**: whether two names refer to the same object.'])),
	...harvest(guide, keyTerms(['- **Spectral radius**: an interview answer.', '- **Server-sent events (SSE)**: a one-way stream.'])),
].map((e) => ({ ...e, title: e.file === eig ? '9.4 Eigenvalues and the spectral radius' : 'Other' })));
const href = (file) => `/${file.split('/').at(-1)}/`;

test('Key terms entries become named concepts', () => {
	const [entry] = harvest(eig, keyTerms(['- **Eigenvector / eigenvalue**: $Av = \\lambda v$.']));
	assert.equal(entry.label, 'Eigenvector / eigenvalue');
	assert.equal(entry.definition, '$Av = \\lambda v$.');
	assert.deepEqual(aliasesOf('Server-sent events (SSE)'), ['Server-sent events', 'SSE']);
	assert.deepEqual(harvest(math, '# Page\n\nNo glossary.\n'), []);
});

test('only unambiguous names are linked', () => {
	assert.ok(linkable('spectral radius'));
	assert.ok(linkable('SSE') && linkable('B-tree') && linkable('PCG32') && linkable('determinant'));
	assert.ok(!linkable('identity') && !linkable('batch') && !linkable('status code'));
});

test('the first use of a borrowed term links to the section that teaches it', () => {
	const body = 'The determinant and eigenvalues decide it. Every eigenvalue matters, and so does identity.';
	const { body: out, used } = linkConcepts(body, lstm, index(), href);
	assert.match(out, /<a class="term" data-term="determinant" href="\/README.md\/">determinant<\/a>/);
	assert.match(out, /<a class="term" data-term="eigenvector-eigenvalue" href="\/04-eigenvalues.md\/">eigenvalues<\/a>/);
	assert.equal((out.match(/class="term"/g) ?? []).length, 2, 'one link per concept per page; "identity" stays text');
	assert.deepEqual(used.map((u) => u.entry.label), ['Determinant', 'Eigenvector / eigenvalue']);
});

test('a page does not link terms it defines, and books do not link into field guides', () => {
	const { used } = linkConcepts('The spectral radius and the determinant.', eig, index(), href);
	assert.deepEqual(used.map((u) => u.entry.label), ['Determinant']);
	const { body } = linkConcepts('Stream it with SSE.', math, index(), href);
	assert.equal(body, 'Stream it with SSE.');
	assert.match(linkConcepts('Stream it with SSE.', guide.replace('01-shared', '02-other'), index(), href).body, /class="term"/);
});

test('code, math, links, headings, bold, italics, and Key terms stay untouched', () => {
	const body = [
		'## The determinant',
		'Use `determinant()` or $\\det$ for the [determinant](/x/) in *the determinant book*; the **determinant** is defined here.',
		'```',
		'determinant = 1',
		'```',
		'$$',
		'\\text{determinant}',
		'$$',
		'## Key terms',
		'- the determinant again',
	].join('\n');
	assert.equal(linkConcepts(body, lstm, index(), href).body, body);
});

test('acronyms match only as written', () => {
	assert.equal(linkConcepts('a sse value', guide.replace('01-shared', '02-other'), index(), href).used.length, 0);
});

test('the concepts box lists each borrowed term with where to read more', () => {
	const { body, used } = linkConcepts('The determinant matters.\n\nMore text.', lstm, index(), href);
	const box = conceptsBox(used, { page: lstm, titleOf: () => '9. Linear Algebra', bookTitleOf: () => 'Mathematics', hrefOf: href });
	assert.match(box, /<details class="concepts">/);
	assert.match(box, /<span data-concept="determinant"><\/span>\*\*Determinant\*\*: the factor/);
	assert.match(box, /Read more: \[9\. Linear Algebra\]\(\/README\.md\/\) in \*Mathematics\*/);
	const page = withConcepts(body, box);
	assert.ok(page.indexOf('The determinant') < page.indexOf('<details') && page.indexOf('<details') < page.indexOf('More text.'));
	assert.equal(conceptsBox([], {}), '');
	assert.ok(withConcepts('```\ncode\n\nmore\n```', 'BOX').startsWith('BOX\n\n```'));
});

test('when sections in two books are titled after a term, the more foundational book wins', () => {
	const algo = 'content/courses/programming-and-algorithms/04-two-pointers-sliding-window/README.md';
	const masks = 'content/courses/ai-engineering/11-transformer/02-masks.md';
	const idx = buildIndex([
		...harvest(masks, keyTerms(['- **Sliding window**: an attention mask over the last $w$ tokens.'])).map((e) => ({ ...e, title: '11.2 Masks: causal, padding, sliding window' })),
		...harvest(algo, keyTerms(['- **Sliding window**: a range whose ends only move forward.'])).map((e) => ({ ...e, title: 'Chapter 4: Two pointers and sliding window' })),
	], new Map([[masks, 0], [algo, 1]]));
	const systems = 'content/courses/systems/01-system-design/README.md';
	assert.match(linkConcepts('A sliding window limiter.', systems, idx, href).body, /href="\/README.md\/"/);
	const lstm2 = 'content/courses/ai-engineering/13-decoder-block/05-gqa.md';
	assert.match(linkConcepts('A sliding window mask.', lstm2, idx, href).body, /href="\/02-masks.md\/"/);
});

test('a titled section in a less foundational book does not outrank the reader\'s own book', () => {
	const graphs = 'content/courses/programming-and-algorithms/08-graphs/README.md';
	const dedup = 'content/courses/data-engineering/05-corpus-pipeline/04-near-dedup.md';
	const idx = buildIndex([
		...harvest(graphs, keyTerms(['- **Union-find**: a disjoint-set forest.'])).map((e) => ({ ...e, title: 'Chapter 8: Graphs' })),
		...harvest(dedup, keyTerms(['- **Union-find**: merges confirmed pairs into clusters.'])).map((e) => ({ ...e, title: '5.4 Near-duplicate dedup: MinHash, LSH, union-find' })),
	], new Map([[graphs, 0], [dedup, 1]]));
	const cp = 'content/courses/programming-and-algorithms/16-competitive-programming.md';
	assert.match(linkConcepts('Use union-find here.', cp, idx, href).body, /href="\/README.md\/"/);
	const math = 'content/courses/mathematics/11-discrete-math-1/README.md';
	assert.match(linkConcepts('Use union-find here.', math, idx, href).body, /href="\/README.md\/"/);
});
