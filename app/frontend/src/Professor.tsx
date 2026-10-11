// The professor: a conversation about the page you have open.
//
// The coach in a session interrupts on its own schedule and mostly stays quiet.
// This is the opposite arrangement: it never speaks first, and it always knows
// which document the question is about.

import { useEffect, useRef, useState } from "react";
import { AskProfessor } from "../wailsjs/go/main/App";
import { EventsOn } from "../wailsjs/runtime/runtime";
import type { harness } from "../wailsjs/go/models";
import { Markdown } from "./Markdown";

interface Turn {
  role: "user" | "assistant";
  text: string;
}

interface Props {
  docPath: string;
  docTitle: string;
  repo?: string;
  onOpenDoc: (path: string) => void;
  onOpenSettings: () => void;
  onClose: () => void;
}

export function Professor({ docPath, docTitle, repo, onOpenDoc, onOpenSettings, onClose }: Props) {
  const [turns, setTurns] = useState<Turn[]>([]);
  const [draft, setDraft] = useState("");
  const [streaming, setStreaming] = useState<string | null>(null);
  const [error, setError] = useState("");
  const end = useRef<HTMLDivElement>(null);

  useEffect(
    () =>
      EventsOn("professor", (e: { id: string; delta: string }) => {
        if (e.delta) setStreaming((s) => (s ?? "") + e.delta);
      }),
    [],
  );

  useEffect(() => {
    end.current?.scrollIntoView({ block: "end" });
  }, [turns, streaming]);

  const ask = async () => {
    const question = draft.trim();
    if (!question || streaming !== null) return;
    const history = turns;
    setTurns([...history, { role: "user", text: question }]);
    setDraft("");
    setError("");
    setStreaming("");
    try {
      const answer = await AskProfessor(question, docPath, history as harness.Turn[]);
      setTurns((t) => [...t, { role: "assistant", text: answer }]);
    } catch (e) {
      setError(String(e));
    } finally {
      setStreaming(null);
    }
  };

  return (
    <aside className="professor">
      <header className="professor-header">
        <div>
          <span className="label" style={{ margin: 0 }}>
            Professor
          </span>
          <span className="faint professor-context" title={docPath}>
            {docTitle ? `Reading: ${docTitle}` : "No page open"}
          </span>
        </div>
        <button className="icon-btn" onClick={onClose} title="Close">
          ✕
        </button>
      </header>

      <div className="professor-feed">
        {turns.length === 0 && streaming === null && (
          <p className="faint professor-empty">
            Ask about what you are reading: what a step means, why it is done this way, or to be
            quizzed on it. Answers are grounded in the open page.
          </p>
        )}
        {turns.map((t, i) => (
          <div key={i} className={`turn ${t.role}`}>
            {t.role === "user" ? (
              t.text
            ) : (
              <Markdown path={docPath} markdown={t.text} repo={repo} onOpenDoc={onOpenDoc} />
            )}
          </div>
        ))}
        {streaming !== null && (
          <div className="turn assistant">
            {streaming ? (
              <Markdown path={docPath} markdown={streaming} repo={repo} onOpenDoc={onOpenDoc} />
            ) : (
              <span className="faint">Thinking…</span>
            )}
          </div>
        )}
        {error && (
          <div className="error-box">
            {error}{" "}
            <button className="link-btn" onClick={onOpenSettings}>
              Model settings
            </button>
          </div>
        )}
        <div ref={end} />
      </div>

      <div className="professor-input">
        <textarea
          rows={3}
          value={draft}
          placeholder="Ask the professor…"
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              ask();
            }
          }}
        />
        <button className="primary" onClick={ask} disabled={streaming !== null || !draft.trim()}>
          Ask
        </button>
      </div>
    </aside>
  );
}
