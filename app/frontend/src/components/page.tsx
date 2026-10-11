// The frame every view in the main pane uses: a header with where you are on
// the left and what you can do on the right, then one scrolling body. Views do
// not build their own headers, which is what keeps them lined up.

import type { ReactNode } from "react";
import { ChevronRight } from "lucide-react";
import { useSidebar } from "@/components/ui/sidebar";
import { cn } from "@/lib/utils";

export interface Crumb {
  label: string;
  onClick?: () => void;
}

export function PageHeader({
  crumbs,
  actions,
}: {
  crumbs: Crumb[];
  actions?: ReactNode;
}) {
  // The sidebar holds its own collapse control, so the header has none. With
  // the sidebar down to its rail, the macOS window controls reach past it, and
  // the extra left padding keeps the breadcrumbs clear of them.
  const { state } = useSidebar();
  return (
    <header
      className={cn(
        "drag flex h-12 shrink-0 items-center gap-2 border-b px-4",
        state === "collapsed" && "pl-10",
      )}
    >
      <nav className="flex min-w-0 flex-1 items-center gap-1 text-sm">
        {crumbs.map((c, i) => {
          const last = i === crumbs.length - 1;
          return (
            <span
              key={i}
              className={cn(
                "flex items-center gap-1",
                last ? "min-w-0" : "shrink-0",
              )}
            >
              {i > 0 && (
                <ChevronRight className="size-3.5 shrink-0 text-muted-foreground/60" />
              )}
              {c.onClick && !last ? (
                <button
                  onClick={c.onClick}
                  className="rounded px-1 text-muted-foreground hover:text-foreground"
                >
                  {c.label}
                </button>
              ) : (
                <span
                  className={cn(
                    "truncate px-1",
                    last ? "font-medium" : "text-muted-foreground",
                  )}
                >
                  {c.label}
                </span>
              )}
            </span>
          );
        })}
      </nav>
      {actions && (
        <div className="flex shrink-0 items-center gap-2">{actions}</div>
      )}
    </header>
  );
}

const WIDTHS = {
  /** Long-form reading. */
  prose: "max-w-3xl",
  /** Forms and lists. */
  form: "max-w-2xl",
  /** Views that want the room. */
  wide: "max-w-5xl",
};

export function PageBody({
  width = "form",
  className,
  children,
}: {
  width?: keyof typeof WIDTHS;
  className?: string;
  children: ReactNode;
}) {
  return (
    <div className="min-h-0 flex-1 overflow-y-auto">
      <div
        className={cn(
          "mx-auto w-full px-8 pt-8 pb-24",
          WIDTHS[width],
          className,
        )}
      >
        {children}
      </div>
    </div>
  );
}

/** A view's title block: one heading, one line under it, optional action. */
export function PageTitle({
  title,
  description,
  action,
}: {
  title: string;
  description?: ReactNode;
  action?: ReactNode;
}) {
  return (
    <div className="mb-6 flex items-start justify-between gap-6">
      <div className="min-w-0">
        <h1 className="text-2xl font-semibold tracking-tight">{title}</h1>
        {description && (
          <p className="mt-1 text-sm text-muted-foreground">{description}</p>
        )}
      </div>
      {action && <div className="shrink-0">{action}</div>}
    </div>
  );
}

/** A small uppercase label over a block of content. */
export function SectionLabel({
  children,
  className,
}: {
  children: ReactNode;
  className?: string;
}) {
  return (
    <div
      className={cn(
        "mb-2 text-xs font-medium text-muted-foreground",
        className,
      )}
    >
      {children}
    </div>
  );
}

/** Inline message for something that went wrong, where it went wrong. */
export function ErrorNote({ children }: { children: ReactNode }) {
  return (
    <div className="rounded-lg border border-destructive/30 bg-destructive/10 px-3 py-2 text-sm text-destructive">
      {children}
    </div>
  );
}
