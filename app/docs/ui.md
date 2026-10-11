# UI standards

How screens in the app are built, so a new one looks and behaves like the ones beside it.

## Stack

| Layer | Choice |
|-------|--------|
| Styling | [Tailwind CSS v4](https://tailwindcss.com), utilities only |
| Components | [shadcn/ui](https://ui.shadcn.com) (`base-nova` style, on Base UI), copied into `frontend/src/components/ui` |
| Icons | [lucide-react](https://lucide.dev), 16px (`size-4`) unless a component sizes them |
| Tokens | CSS variables in `frontend/src/index.css` |

Add a component with `npx shadcn@latest add <name>` from `frontend/`. Files under `components/ui` are generated: change behavior by wrapping them, not by editing them, so they can be regenerated.

## Rules

1. **Tokens, never literals.** Colors come from tokens (`bg-background`, `text-muted-foreground`, `border`, `bg-muted`, `text-destructive`). No hex values and no arbitrary color utilities in a view.
2. **Status has three colors.** Success is `emerald-500`, caution is `amber-500`, failure is `destructive`. Nothing else carries meaning by color, and color is never the only signal: a verdict also says "Pass" or "Fail".
3. **One primary action per view.** It is the only `Button` with the default variant. Everything else is `outline` or `ghost`.
4. **Sentence case** for every label, heading, and button. No trailing punctuation on labels.
5. **No new CSS files.** A view that needs a stylesheet needs a component instead. `markdown.css` is the exception, because it styles HTML the app did not write.

## Layout

The window is three columns: the sidebar, one main view, and the professor panel.

Every main view is built from `components/page.tsx`:

| Part | Use |
|------|-----|
| `PageHeader` | 48px bar. Breadcrumbs on the left say where you are; actions on the right say what you can do. It is also the window's drag handle. It holds no layout controls. |
| `PageBody` | The single scroll container. `width="form"` (672px) for lists and forms, `"prose"` (768px) for reading, `"wide"` for views that need room. |
| `PageTitle` | One `h1`, one line of description, one optional action. |
| `SectionLabel` | The small label over a block. |
| `ErrorNote` | An error, shown where it happened. |

A view does not draw its own header or manage its own scrolling.

## The sidebar

Flat lists, one line per item, like a list of recent conversations. The sidebar never nests: a track opens in the main pane and its topics are listed there. From the top: the open pack's title with the collapse control beside it, search, navigation, paths, tracks, reference. At the bottom: settings, then the pack switcher, where a workspace switcher goes.

Paths show only the top level. A path included by another path is reachable from the path that includes it.

Search replaces the lists with results while there is a query.

Collapsing leaves a rail of icons. It never hides the sidebar, so the control that collapses it is also the one that expands it and no view needs a second one. Anything without an icon is hidden in the rail.

## The professor

The professor is a docked panel on the right, open by default. It has no place in the navigation: Cmd/Ctrl+J toggles it from anywhere, its header has a hide control, and while it is hidden a single floating button in the bottom right brings it back.

## Lists

Every list in a view is `ListRow` inside a `ListGroup` (`components/list-row.tsx`): a title, an optional second line, an optional leading icon, and something small on the right. Paths, topics, exercises, packs, and past sessions all use it.

## Forms

`components/choice.tsx` has the three controls forms are made of:

| Control | When |
|---------|------|
| `Field` | Wraps any input with its label |
| `ChoiceGroup` | Two to five options, all worth seeing at once |
| `Picker` | More options than fit on a line |

A setting that takes effect immediately saves on change. There is no Save button for it.

## States

- **Loading**: a `Skeleton` in the shape of what is coming. No spinners in views.
- **Empty**: one sentence saying what is missing and what would fill it.
- **Error**: an `ErrorNote` with the actual message, next to the control that caused it.

## What is still on the old stylesheet

The live session and the review screen predate this and still use `theme.css` plus their own stylesheets. Each is wrapped in `.legacy`, which is what scopes those rules and their tokens. Do not add to them; move a screen over whole when it is next changed.
