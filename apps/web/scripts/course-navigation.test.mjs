import assert from 'node:assert/strict';
import test from 'node:test';
import { courseOutlines, chapterNeighbors } from './course-navigation.mjs';

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
	assert.deepEqual(neighbors.get('a/intro.md')[0].prev, { label: 'Algebra', link: '/learn/a/' });
	assert.deepEqual(neighbors.get('a/02.md')[0].next, { link: '/a/01.md' });
});
