import test from 'node:test';
import assert from 'node:assert/strict';
import { checkPrompt } from '../src/lib/check-prompt.mjs';

test('a check prompt names the source, the question and the reference answer', () => {
  const prompt = checkPrompt({
    book: 'Mathematics',
    section: 'Chapter 1: Quantities and relationships',
    url: 'https://openlearn.urmzd.com/courses/mathematics/01-quantities-and-relationships/',
    question: 'Does $C(n)$ mean $C$ times $n$?',
    answer: 'No. $C(n)$ is the output of the function $C$ when the input is $n$.',
  });
  assert.match(prompt, /"Chapter 1: Quantities and relationships" in the Openlearn book "Mathematics"/);
  assert.match(prompt, /Question:\nDoes \$C\(n\)\$ mean \$C\$ times \$n\$\?/);
  assert.match(prompt, /My answer:\n\[Type your answer here/);
  assert.ok(prompt.indexOf('My answer:') < prompt.indexOf('Reference answer'), 'the reader answers before the reference');
  assert.match(prompt, /Reference answer from the book, for your check only:\nNo\./);
});

test('a check prompt works without a reference answer or a book', () => {
  const prompt = checkPrompt({ question: 'What breaks when the queue is full?\n\n\n\nName one signal.', answer: '  ' });
  assert.match(prompt, /^I am studying an Openlearn book\./);
  assert.match(prompt, /What breaks when the queue is full\?\n\nName one signal\./);
  assert.doesNotMatch(prompt, /Reference answer/);
});
