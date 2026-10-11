// The OpenLearn mark and wordmark. The mark is an open book reduced to two
// pages, drawn in the current color so it follows the theme. Everything that
// shows the brand uses this file, so the brand changes in one place.

import { cn } from "@/lib/utils";

export function LogoMark({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 24 24" fill="none" aria-hidden="true" className={cn("size-5", className)}>
      <path
        d="M12 6.5c-1.8-1.4-4.2-2-7-2v13c2.8 0 5.2.6 7 2 1.8-1.4 4.2-2 7-2v-13c-2.8 0-5.2.6-7 2Z"
        stroke="currentColor"
        strokeWidth="1.8"
        strokeLinejoin="round"
      />
      <path d="M12 6.5v13" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
    </svg>
  );
}

export function Logo({ className }: { className?: string }) {
  return (
    <span className={cn("flex items-center gap-2 text-[15px] font-semibold tracking-tight", className)}>
      <LogoMark />
      OpenLearn
    </span>
  );
}
