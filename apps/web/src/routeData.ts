import { defineRouteMiddleware } from '@astrojs/starlight/route-data';

type Entry = { type: 'link'; isCurrent: boolean } | { type: 'group'; entries: Entry[] };
const holdsCurrent = (entry: Entry): boolean => (entry.type === 'link' ? entry.isCurrent : entry.entries.some(holdsCurrent));

// The menu shows only the book being read, under a link back to the shelf.
export const onRequest = defineRouteMiddleware((context) => {
	const route = context.locals.starlightRoute;
	const book = route.sidebar.find((entry) => entry.type === 'group' && holdsCurrent(entry as Entry));
	const shelf = { type: 'link' as const, label: 'All books', href: '/', isCurrent: false, badge: undefined, attrs: { class: 'shelf-link' } };
	route.sidebar = book?.type === 'group' ? [shelf, ...book.entries] : [shelf];
});
