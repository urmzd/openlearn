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
