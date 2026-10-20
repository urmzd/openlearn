// The reading surface: one chapter scrolls as one page.
//
// When the reader nears the end of a section, the next section of the same
// chapter is fetched and appended below it, and the address bar follows
// whichever section is on screen. A chapter ends with a card that links to
// the next chapter, so a break between chapters is a choice, not a click per
// section. Reaching the end of a section marks it read.
//
// Everything here enhances server-rendered pages: without JavaScript each
// section is still its own page with previous and next links.

const clamp = (n) => Math.min(1, Math.max(0, n));
function el(tag, className, text) {
	const node = document.createElement(tag);
	if (className) node.className = className;
	if (text !== undefined) node.textContent = text;
	return node;
}

async function diagrams(root) {
	const blocks = [...root.querySelectorAll('pre.mermaid:not([data-processed])')];
	if (!blocks.length) return;
	try {
		const { default: mermaid } = await import('mermaid');
		mermaid.initialize({ startOnLoad: false, securityLevel: 'strict', theme: document.documentElement.dataset.theme === 'dark' ? 'dark' : 'neutral' });
		await mermaid.run({ nodes: blocks });
	} catch { /* The diagram's source stays readable if rendering fails. */ }
}

export function enhance(root) {
	diagrams(root);
}

// ---------- continuous reading ----------

export function startReader({ course, lesson, markRead, onSection, onAppend }) {
	const first = document.querySelector('.sl-markdown-content');
	if (!first) return;
	enhance(first);
	if (!course || !lesson) return;

	const lessons = course.lessons;
	const chapter = lessons.filter((l) => l.chapter === lesson.chapter);
	document.documentElement.classList.add('reader');

	const bar = el('div', 'reading-progress');
	bar.setAttribute('aria-hidden', 'true');
	const fill = el('span');
	bar.append(fill);
	document.body.append(bar);

	const sections = [{ lesson, node: first }];
	const flow = el('div', 'reader-flow');
	first.after(flow);

	const readObserver = new IntersectionObserver((entries) => {
		for (const entry of entries) {
			if (!entry.isIntersecting) continue;
			markRead(entry.target.dataset.key);
			readObserver.unobserve(entry.target);
		}
	});
	const marker = (key) => {
		const node = el('div', 'read-marker');
		node.dataset.key = key;
		readObserver.observe(node);
		return node;
	};
	first.append(marker(lesson.key));

	let loading = false;
	let done = false;
	const nearEnd = new IntersectionObserver((entries) => {
		if (entries.some((e) => e.isIntersecting)) loadNext();
	}, { rootMargin: '0px 0px 1400px 0px' });
	const watchEnd = () => {
		nearEnd.disconnect();
		// Watch the end of the last section, not its start, or every section
		// would load at once.
		const markers = sections.at(-1).node.querySelectorAll('.read-marker');
		if (!done && markers.length) nearEnd.observe(markers[markers.length - 1]);
	};

	function endCard(next) {
		done = true;
		nearEnd.disconnect();
		const card = el('a', 'chapter-break');
		if (next) {
			card.href = next.url;
			card.append(el('span', 'eyebrow', 'Next chapter'), el('strong', '', next.chapterTitle));
		} else {
			card.href = '/';
			card.append(el('span', 'eyebrow', `You finished ${course.title}`), el('strong', '', 'Back to your books'));
		}
		flow.append(card);
	}

	async function loadNext() {
		if (loading || done) return;
		const current = sections.at(-1).lesson;
		const next = lessons[lessons.indexOf(current) + 1];
		if (!next || next.chapter !== current.chapter) { endCard(next); return; }
		loading = true;
		try {
			const response = await fetch(next.url);
			if (!response.ok) throw new Error(String(response.status));
			const doc = new DOMParser().parseFromString(await response.text(), 'text/html');
			const content = doc.querySelector('.sl-markdown-content');
			if (!content) throw new Error('no content');
			const article = el('article', 'reader-section');
			article.dataset.key = next.key;
			const header = el('header', 'reader-section-header');
			header.append(el('h2', 'reader-title', next.title));
			const body = el('div', 'sl-markdown-content');
			body.append(...[...content.childNodes].map((n) => document.importNode(n, true)));
			body.append(marker(next.key));
			article.append(header, body);
			flow.append(article);
			sections.push({ lesson: next, node: article });
			enhance(body);
			onAppend?.();
		} catch {
			// Offline without a saved copy, or the page moved: link to it instead.
			const card = el('a', 'chapter-break');
			card.href = next.url;
			card.append(el('span', 'eyebrow', 'Continue'), el('strong', '', next.title));
			flow.append(card);
			done = true;
		} finally {
			loading = false;
			watchEnd();
		}
	}

	// The section whose top has passed the upper third of the screen is the
	// one being read: it owns the address bar, the title, and the note button.
	let shown = lesson;
	let frame = 0;
	function track() {
		frame = 0;
		let active = sections[0];
		for (const section of sections) if (section.node.getBoundingClientRect().top < innerHeight * 0.33) active = section;
		if (active.lesson !== shown) {
			shown = active.lesson;
			history.replaceState(history.state, '', shown.url);
			document.title = document.title.replace(/^.*?( \| [^|]+)$/, `${shown.title}$1`);
			// The menu highlights the section on screen.
			for (const a of document.querySelectorAll('#starlight__sidebar a')) {
				if (new URL(a.href).pathname === shown.url) a.setAttribute('aria-current', 'page');
				else a.removeAttribute('aria-current');
			}
			onSection?.(shown);
		}
		const rect = active.node.getBoundingClientRect();
		const within = clamp(-rect.top / Math.max(1, rect.height - innerHeight));
		const position = Math.max(0, chapter.indexOf(active.lesson));
		fill.style.transform = `scaleX(${clamp((position + within) / Math.max(1, chapter.length))})`;
	}
	addEventListener('scroll', () => { frame ||= requestAnimationFrame(track); }, { passive: true });
	addEventListener('resize', () => { frame ||= requestAnimationFrame(track); });
	track();
	watchEnd();
}
