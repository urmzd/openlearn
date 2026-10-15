// The one row every list in a view is made of: a title, an optional second
// line, and something small on the right. Paths, topics, exercises, packs, and
// search results are all this row, so they scan the same way.

import type { ReactNode } from "react";
import { cn } from "@/lib/utils";

interface Props {
  title: ReactNode;
  description?: ReactNode;
  leading?: ReactNode;
  trailing?: ReactNode;
  active?: boolean;
  onClick?: () => void;
  className?: string;
}

export function ListRow({ title, description, leading, trailing, active, onClick, className }: Props) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={cn(
        "flex w-full items-center gap-3 rounded-lg px-3 py-2.5 text-left transition-colors outline-none",
        "hover:bg-muted/60 focus-visible:ring-2 focus-visible:ring-ring/50",
        active && "bg-muted",
        className,
      )}
    >
      {leading && <span className="shrink-0 text-muted-foreground">{leading}</span>}
      <span className="min-w-0 flex-1">
        <span className="block truncate text-sm font-medium">{title}</span>
        {description && (
          <span className="mt-0.5 block truncate text-xs text-muted-foreground">{description}</span>
        )}
      </span>
      {trailing && <span className="shrink-0 text-xs text-muted-foreground">{trailing}</span>}
    </button>
  );
}

/** A bordered group of rows. */
export function ListGroup({ children, className }: { children: ReactNode; className?: string }) {
  return <div className={cn("flex flex-col gap-0.5 rounded-xl border p-1", className)}>{children}</div>;
}
