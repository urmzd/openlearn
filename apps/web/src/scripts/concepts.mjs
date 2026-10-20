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
let closedAt = 0;
let closedFrom;

function create() {
	card = document.createElement('div');
	card.className = 'concept-card';
	card.popover = 'auto';
	card.setAttribute('role', 'dialog');
	card.tabIndex = -1;
	card.addEventListener('toggle', (event) => {
		if (event.newState !== 'closed') return;
		closedAt = performance.now();
		closedFrom = trigger;
		// Return focus to the term only when focus was inside the card.
		if (card.contains(document.activeElement)) trigger?.focus({ preventScroll: true });
		trigger = undefined;
	});
	document.body.append(card);
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

function open(anchor, item) {
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
	trigger = anchor;
	card.showPopover();
	place(anchor);
	card.focus({ preventScroll: true });
}

export function startConcepts() {
	if (!supported) return;
	document.addEventListener('click', (event) => {
		const anchor = event.target.closest?.('a.term[data-term]');
		if (!anchor || event.defaultPrevented || event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
		const item = anchor.closest('.sl-markdown-content')
			?.querySelector(`.concepts [data-concept="${CSS.escape(anchor.dataset.term)}"]`)
			?.closest('li');
		if (!item) return;
		event.preventDefault();
		// A tap on the open term's own link closes the card (light dismiss
		// already did); it should not reopen it.
		if (closedFrom === anchor && performance.now() - closedAt < 400) return;
		open(anchor, item);
	});
	// A card floats next to its term on wide screens; scrolling detaches it.
	addEventListener('scroll', () => {
		if (card?.matches(':popover-open') && !card.classList.contains('sheet')) card.hidePopover();
	}, { passive: true });
}
