// Review questions: work each one on paper, then copy a prompt that asks an AI
// assistant to check the answer. The answer stays closed until the reader
// opens it. Pages work the same without this script, less the button.
import { checkPrompt } from '../lib/check-prompt.mjs';

const ICONS = {
	copy: '<rect width="14" height="14" x="8" y="8" rx="2" ry="2"></rect><path d="M4 16c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2h10c1.1 0 2 .9 2 2"></path>',
	check: '<path d="M20 6 9 17l-5-5"></path>',
};
const icon = (name) => `<svg viewBox="0 0 24 24" aria-hidden="true" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">${ICONS[name]}</svg>`;

// Plain text for a prompt: TeX for mathematics, fences for code.
function textOf(nodes) {
	const box = document.createElement('div');
	box.append(...nodes.map((node) => node.cloneNode(true)));
	for (const extra of box.querySelectorAll('.check-prompt, .exercise-method, details, .sl-anchor-link, .copy, .sr-only')) extra.remove();
	for (const math of box.querySelectorAll('.katex-display, .katex')) {
		if (!math.isConnected) continue;
		const tex = math.querySelector('annotation[encoding="application/x-tex"]')?.textContent;
		if (tex) math.replaceWith(math.classList.contains('katex-display') ? `\n$$${tex}$$\n` : `$${tex}$`);
	}
	for (const block of box.querySelectorAll('.expressive-code, pre')) {
		if (!block.isConnected) continue;
		const lines = [...block.querySelectorAll('.ec-line')].map((line) => line.textContent);
		block.replaceWith(`\n\`\`\`\n${lines.length ? lines.join('\n') : block.textContent.trim()}\n\`\`\`\n`);
	}
	for (const item of box.querySelectorAll('p, li, tr, br, div')) item.append('\n');
	return box.textContent.replace(/[ \t]+/g, ' ').replace(/ ?\n ?/g, '\n').replace(/\n{3,}/g, '\n\n').trim();
}

// The question is the list item around the answer. For a check inside the
// text, it is the blocks between the answer and the heading above it.
function questionOf(answer) {
	const item = answer.parentElement;
	if (item?.tagName === 'LI') return textOf([...item.childNodes].filter((node) => node !== answer));
	const blocks = [];
	for (let node = answer.previousElementSibling; node && blocks.length < 4; node = node.previousElementSibling) {
		if (node.matches('details, hr, .sl-heading-wrapper, h1, h2, h3, h4')) break;
		blocks.unshift(node);
	}
	return textOf(blocks);
}

async function copy(text) {
	try { await navigator.clipboard.writeText(text); return true; } catch { /* Fall back to a selection. */ }
	const field = document.createElement('textarea');
	field.value = text;
	field.setAttribute('readonly', '');
	field.style.cssText = 'position:fixed;inset:0 auto auto 0;opacity:0';
	document.body.append(field);
	field.select();
	let copied = false;
	try { copied = document.execCommand('copy'); } catch { /* The button reports the failure. */ }
	field.remove();
	return copied;
}

export function startExercises(root) {
	let cue = false;
	for (const answer of root.querySelectorAll('details:not(.concepts):not([data-exercise])')) {
		const summary = answer.querySelector(':scope > summary');
		if (!summary || !/answer|explanation|reasoning/i.test(summary.textContent)) continue;
		const question = questionOf(answer);
		if (!question) continue;
		answer.dataset.exercise = '';
		const button = document.createElement('button');
		button.type = 'button';
		button.className = 'check-prompt';
		button.innerHTML = `${icon('copy')}<span>Copy check prompt</span>`;
		button.addEventListener('click', async () => {
			const section = answer.closest('.reader-section')?.querySelector('.reader-title') ?? document.querySelector('h1');
			const copied = await copy(checkPrompt({
				book: document.querySelector('[data-book-title]')?.dataset.bookTitle,
				section: section?.textContent,
				url: location.origin + location.pathname,
				question,
				answer: textOf([...answer.childNodes].filter((node) => node !== summary)),
			}));
			button.innerHTML = copied ? `${icon('check')}<span>Copied</span>` : `${icon('copy')}<span>Copy failed</span>`;
			setTimeout(() => { button.innerHTML = `${icon('copy')}<span>Copy check prompt</span>`; }, 2000);
		});
		answer.before(button);
		// Say the method once, above the first list of questions in a section.
		const list = answer.parentElement?.tagName === 'LI' ? answer.parentElement.parentElement : null;
		if (list && !cue && !list.previousElementSibling?.matches('.exercise-method')) {
			const method = document.createElement('p');
			method.className = 'exercise-method';
			method.textContent = 'Work each question on paper. Then copy its check prompt, add your answer and send it to an AI assistant.';
			list.before(method);
		}
		if (list) cue = true;
	}
}
