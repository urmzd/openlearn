import assert from 'node:assert/strict';
import test from 'node:test';
import { courseOutlines, chapterNeighbors, chapterParts } from './course-navigation.mjs';

const publication = (id, title, extra = {}) => ({
	manifest: {
		id, title,
		start: [{ file: `${id}/intro.md`, title: 'Introduction' }],
		reference: [{ dir: 'sources' }],
		...extra,
	},
	reading: [
		{ title: 'Introduction', files: [`${id}/intro.md`] },
		{ title: 'Core', files: [`${id}/02.md`, `${id}/01.md`, 'shared.md'] },
		{ title: 'Sources', files: ['sources/book.md'] },
	],
});

test('courses sort alphabetically while chapter order follows the publication', () => {
	const outlines = courseOutlines([publication('z', 'Zebra'), publication('a', 'Algebra')]);
	assert.deepEqual(outlines.map(({ course }) => course.id), ['a', 'z']);
	assert.deepEqual(outlines[0].chapters.flatMap(({ files }) => files), ['a/intro.md', 'a/02.md', 'a/01.md', 'shared.md']);
	assert.deepEqual(outlines[0].references, [{ title: 'Sources', files: ['sources/book.md'] }]);
});

test('guided courses keep deeper material in references and exclude unpublished files', () => {
	const [outline] = courseOutlines([publication('a', 'Algebra', { guided: true })], (file) => file !== 'a/02.md');
	assert.deepEqual(outline.chapters.flatMap(({ files }) => files), ['a/intro.md']);
	assert.deepEqual(outline.references.flatMap(({ files }) => files), ['a/01.md', 'shared.md', 'sources/book.md']);
});

test('navigation distinguishes shared chapters and ends at the course boundary', () => {
	const outlines = courseOutlines([publication('a', 'Algebra'), publication('z', 'Zebra')]);
	const neighbors = chapterNeighbors(outlines, (file) => `/${file}`);
	assert.equal(neighbors.get('shared.md').length, 2);
	assert.equal(neighbors.get('shared.md')[0].next, false);
	assert.deepEqual(neighbors.get('a/intro.md')[0].prev, { label: 'All books', link: '/' });
	assert.deepEqual(neighbors.get('a/02.md')[0].next, { link: '/a/01.md' });
});

// Reading-state identities and public URLs survive the systems book consolidation.
test('merged systems chapters retain their published source identities', async () => {
  const { stableSourcePath } = await import('./legacy-source-paths.mjs');
  const cases = [
    ['05-messaging-and-queueing', 'infrastructure/02-messaging-and-queueing'],
    ['08-search-and-indexing', 'infrastructure/03-search-and-indexing'],
    ['09-cloud-native', 'systems/03-cloud-native'],
    ['10-containers-kubernetes', 'infrastructure/01-containers-kubernetes'],
    ['11-observability', 'systems/04-observability'],
    ['12-incident-response-and-chaos', 'systems/05-incident-response-and-chaos'],
    ['14-apache-stack', 'infrastructure/04-apache-stack'],
  ];
  for (const [current, previous] of cases) {
    assert.equal(stableSourcePath(`content/courses/systems/${current}/README.md`), `content/courses/${previous}/README.md`);
    assert.equal(stableSourcePath(`content/courses/systems/${current}/code/example.py`), `content/courses/${previous}/code/example.py`);
  }
  assert.equal(stableSourcePath('content/courses/systems/01-system-design/README.md'), 'content/courses/systems/01-system-design/README.md');
  assert.equal(stableSourcePath('content/courses/programming-and-algorithms/README.md'), 'content/courses/programming-and-algorithms/README.md');
});

test('part grouping preserves order and keeps introductions separate', () => {
  const chapters = [{ title: 'Start' }, { title: 'One', part: 'Foundations' }, { title: 'Two', part: 'Foundations' }, { title: 'Three', part: 'Application' }, { title: 'Sources', kind: 'appendix' }];
  const groups = chapterParts(chapters);
  assert.deepEqual(groups.map((g) => g.title), ['', 'Foundations', 'Application', 'Appendices']);
  assert.deepEqual(groups.flatMap((g) => g.chapters), chapters);
  const [outline] = courseOutlines([{ manifest: { id: 'a', title: 'A', start: [] }, reading: [{ title: 'One', part: 'Foundations', files: ['one.md'] }] }]);
  assert.equal(outline.chapters[0].part, 'Foundations');
});

test('AI, programming and mathematics keep old lesson identities', async () => {
  const { stableSourcePath } = await import('./legacy-source-paths.mjs');
  assert.equal(stableSourcePath('content/courses/ai-engineering/07-tokenizers/README.md'), 'content/courses/build-an-llm/02-tokenizers/README.md');
  assert.equal(stableSourcePath('content/courses/mathematics/09-linear-algebra/README.md'), 'content/courses/mathematics-for-computing/04-linear-algebra/README.md');
  assert.equal(stableSourcePath('content/courses/programming-and-algorithms/03-arrays-hashing/README.md'), 'content/courses/algorithms/01-arrays-hashing/README.md');
});
