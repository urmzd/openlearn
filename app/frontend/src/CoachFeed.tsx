// The coach's channel. Messages appear the moment the agent decides to speak
// and fill in as it writes, which is the whole reason the backend streams:
// a message that pops in complete reads as a notification, one that types
// itself reads as someone watching you work.

import { useEffect, useRef } from "react";
import type { Message } from "./agui";
import "./coachfeed.css";

const KIND_META: Record<string, { label: string; icon: string }> = {
  hint: { label: "Hint", icon: "◐" },
  probe: { label: "Question", icon: "?" },
  praise: { label: "Good call", icon: "✓" },
  curveball: { label: "Requirements changed", icon: "!" },
  pacing: { label: "Pacing", icon: "◷" },
  system: { label: "", icon: "·" },
};

function timestamp(sec: number): string {
  const m = Math.floor(sec / 60);
  const s = sec % 60;
  return `${m}:${String(s).padStart(2, "0")}`;
}

interface Props {
  messages: Message[];
  /** Step names in flight, so the feed can show the coach thinking. */
  activeSteps: string[];
  coachEnabled: boolean;
}

export function CoachFeed({ messages, activeSteps, coachEnabled }: Props) {
  const endRef = useRef<HTMLDivElement>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const pinnedRef = useRef(true);

  // Follow new output, but stop following the moment the user scrolls up to
  // re-read something. Yanking them back down mid-read is worse than a
  // slightly stale view.
  const onScroll = () => {
    const el = scrollRef.current;
    if (!el) return;
    pinnedRef.current = el.scrollHeight - el.scrollTop - el.clientHeight < 60;
  };

  useEffect(() => {
    if (pinnedRef.current) endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const thinking = activeSteps.includes("coach_thinking");

  return (
    <div className="feed">
      <div className="feed-head spread">
        <span className="label" style={{ margin: 0 }}>
          Coach
        </span>
        <span className="faint feed-status">
          {!coachEnabled ? "off" : thinking ? "watching…" : "listening"}
        </span>
      </div>

      <div className="feed-scroll" ref={scrollRef} onScroll={onScroll}>
        {messages.length === 0 && (
          <p className="faint feed-empty">
            {coachEnabled
              ? "The coach speaks only when it is worth interrupting for. Silence means you are fine."
              : "Coach is off for this session. You will still be scored at the end."}
          </p>
        )}

        {messages.map((m) => {
          const meta = KIND_META[m.kind] ?? KIND_META.system;
          return (
            <article key={m.id} className={`msg sev-${m.severity} kind-${m.kind}`}>
              <header className="msg-head">
                <span className="msg-icon">{meta.icon}</span>
                <span className="msg-title">{m.title}</span>
                <span className="msg-time faint">{timestamp(m.elapsedSec)}</span>
              </header>
              <div className="msg-body">
                {m.body}
                {!m.done && <span className="caret" />}
              </div>
            </article>
          );
        })}

        {thinking && messages.every((m) => m.done) && (
          <div className="msg-thinking faint">
            <span className="dot" />
            <span className="dot" />
            <span className="dot" />
          </div>
        )}

        <div ref={endRef} />
      </div>
    </div>
  );
}
