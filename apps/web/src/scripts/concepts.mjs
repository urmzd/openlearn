// Concept previews: tapping a dotted term shows its definition and the
// section that teaches it, without leaving the page.
//
// The definition comes from the page's own "Concepts from other sections"
// box, so the preview works offline and in sections appended by the
// continuous reader. Without JavaScript, or in a browser without the popover
// API, a term is an ordinary link to the section that teaches it.

const supported = typeof HTMLElement !== 'undefined' && Object.hasOwn(HTMLElement.prototype, 'popover');

let card;
let trigger;
let hovering = false;
let pressed;
let showTimer = 0;
let hideTimer = 0;

function create() {
	card = document.createElement('div');
	card.className = 'concept-card';
	card.popover = 'auto';
	card.setAttribute('role', 'dialog');
	card.tabIndex = -1;
	card.addEventListener('toggle', (event) => {
		if (event.newState !== 'closed') return;
		hovering = false;
		// Return focus to the term only when focus was inside the card.
		if (card.contains(document.activeElement)) trigger?.focus({ preventScroll: true });
		trigger = undefined;
	});
	// A card opened by hovering stays while the pointer is on it.
	card.addEventListener('pointerenter', () => clearTimeout(hideTimer));
	card.addEventListener('pointerleave', (event) => { if (hovering && event.pointerType === 'mouse') hideSoon(); });
	document.body.append(card);
}

function hideSoon() {
	clearTimeout(hideTimer);
	hideTimer = setTimeout(() => { if (hovering && card?.matches(':popover-open')) card.hidePopover(); }, 250);
}

function place(anchor) {
	if (matchMedia('(max-width: 50rem)').matches) {
		card.classList.add('sheet');
		card.style.removeProperty('top');
		card.style.removeProperty('left');
		return;
	}
	card.classList.remove('sheet');
	const gap = 8;
	const box = anchor.getClientRects()[0] ?? anchor.getBoundingClientRect();
	const { width, height } = card.getBoundingClientRect();
	const below = box.bottom + gap;
	const top = below + height > innerHeight - gap && box.top - gap - height > gap ? box.top - gap - height : below;
	const left = Math.min(Math.max(gap, box.left), innerWidth - width - gap);
	card.style.top = `${Math.max(gap, top)}px`;
	card.style.left = `${Math.max(gap, left)}px`;
}

function itemFor(anchor) {
	return anchor.closest('.sl-markdown-content')
		?.querySelector(`.concepts [data-concept="${CSS.escape(anchor.dataset.term)}"]`)
		?.closest('li');
}

function open(anchor, item, { hover = false } = {}) {
	card ?? create();
	const body = item.cloneNode(true);
	body.querySelector('[data-concept]')?.remove();
	const title = body.querySelector('strong');
	const heading = document.createElement('p');
	heading.className = 'concept-title';
	heading.id = 'concept-card-title';
	heading.textContent = title?.textContent ?? anchor.textContent;
	// The label is the heading; drop it and the ": " that followed it.
	if (title) {
		const after = title.nextSibling;
		if (after?.nodeType === Node.TEXT_NODE) after.textContent = after.textContent.replace(/^\s*:\s*/, '');
		title.remove();
	}
	const close = document.createElement('button');
	close.type = 'button';
	close.className = 'concept-close';
	close.setAttribute('aria-label', 'Close');
	close.textContent = '×';
	close.addEventListener('click', () => card.hidePopover());
	const definition = document.createElement('div');
	definition.className = 'concept-definition';
	definition.append(...body.childNodes);
	card.setAttribute('aria-labelledby', heading.id);
	card.replaceChildren(close, heading, definition);
	if (card.matches(':popover-open')) card.hidePopover();
	trigger = anchor;
	card.showPopover();
	hovering = hover;
	place(anchor);
	if (!hover) card.focus({ preventScroll: true });
}

export function startConcepts() {
	if (!supported) return;
	document.addEventListener('click', (event) => {
		const anchor = event.target.closest?.('a.term[data-term]');
		if (!anchor || event.defaultPrevented || event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
		const item = itemFor(anchor);
		if (!item) return;
		event.preventDefault();
		clearTimeout(showTimer);
		clearTimeout(hideTimer);
		// What was open when the pointer went down: the browser closes the
		// card on pointerdown, before this click arrives.
		const before = pressed ?? { trigger, hover: hovering, open: card?.matches(':popover-open') };
		pressed = undefined;
		// A second click on a term whose card the reader opened closes it; a
		// click on a hover preview keeps it open.
		if (before.open && before.trigger === anchor && !before.hover) {
			if (card.matches(':popover-open')) card.hidePopover();
			return;
		}
		open(anchor, item);
	});
	addEventListener('pointerdown', (event) => {
		pressed = event.target.closest?.('a.term[data-term]')
			? { trigger, hover: hovering, open: !!card?.matches(':popover-open') }
			: undefined;
	}, { capture: true });
	// With a mouse, resting on a term previews it; a click keeps the card open.
	if (matchMedia('(hover: hover) and (pointer: fine)').matches) {
		document.addEventListener('pointerover', (event) => {
			const anchor = event.target.closest?.('a.term[data-term]');
			if (!anchor || event.pointerType !== 'mouse') return;
			clearTimeout(hideTimer);
			if (trigger === anchor) return;
			// Another card the reader opened on purpose stays until they close it.
			if (card?.matches(':popover-open') && !hovering) return;
			clearTimeout(showTimer);
			showTimer = setTimeout(() => {
				const item = itemFor(anchor);
				if (item && anchor.matches(':hover')) open(anchor, item, { hover: true });
			}, 350);
		});
		document.addEventListener('pointerout', (event) => {
			const anchor = event.target.closest?.('a.term[data-term]');
			if (!anchor || event.pointerType !== 'mouse' || anchor.contains(event.relatedTarget)) return;
			clearTimeout(showTimer);
			if (trigger === anchor && hovering) hideSoon();
		});
	}
	// A card floats next to its term on wide screens; scrolling detaches it.
	addEventListener('scroll', () => {
		if (card?.matches(':popover-open') && !card.classList.contains('sheet')) card.hidePopover();
	}, { passive: true });
}

// A book's glossary page: filter the terms as the reader types.
export function startGlossary() {
	const input = document.querySelector('#glossary-filter');
	if (!input) return;
	const items = [...document.querySelectorAll('.glossary-letter li')];
	const words = (s) => s.toLocaleLowerCase().match(/[\p{L}\p{N}]+/gu) ?? [];
	const terms = new Map(items.map((li) => [li, words(li.querySelector('strong')?.textContent ?? '')]));
	const count = document.querySelector('#glossary-count');
	const letters = document.querySelector('.glossary-letters');
	input.addEventListener('input', () => {
		const query = words(input.value);
		let shown = 0;
		for (const [li, label] of terms) {
			li.hidden = !query.every((q) => label.some((w) => w.startsWith(q)));
			if (!li.hidden) shown++;
		}
		for (const group of document.querySelectorAll('.glossary-letter')) group.hidden = !group.querySelector('li:not([hidden])');
		if (letters) letters.hidden = query.length > 0;
		count.textContent = `${shown} ${shown === 1 ? 'term' : 'terms'}`;
	});
}
