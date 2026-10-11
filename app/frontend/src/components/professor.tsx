// The professor: a conversation about the page you have open.
//
// The coach in a session interrupts on its own schedule and mostly stays quiet.
// This is the opposite arrangement: it never speaks first, and it always knows
// which document the question is about.

import { useEffect, useRef, useState } from "react";
import { ArrowUp, PanelRightClose, Sparkles } from "lucide-react";
import { AskProfessor } from "../../wailsjs/go/main/App";
import { EventsOn } from "../../wailsjs/runtime/runtime";
import type { harness } from "../../wailsjs/go/models";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { ErrorNote } from "@/components/page";
import { Markdown } from "@/Markdown";

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

const STARTERS = ["Explain this page in plain words", "Quiz me on this", "What should I already know?"];

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

  const ask = async (text: string) => {
    const question = text.trim();
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
    <aside className="flex w-96 shrink-0 flex-col border-l bg-sidebar">
      <header className="drag flex h-12 shrink-0 items-center gap-2 border-b px-3">
        <Sparkles className="size-4 text-muted-foreground" />
        <div className="min-w-0 flex-1">
          <div className="text-sm font-medium">Professor</div>
        </div>
        <Button variant="ghost" size="icon-sm" onClick={onClose} title="Hide the professor (⌘J)">
          <PanelRightClose />
        </Button>
      </header>

      <div className="min-h-0 flex-1 overflow-y-auto p-4">
        {turns.length === 0 && streaming === null ? (
          <div className="flex flex-col gap-3">
            <p className="text-sm text-muted-foreground">
              {docTitle ? (
                <>
                  Ask about <span className="text-foreground">{docTitle}</span>. Answers are grounded in
                  that page.
                </>
              ) : (
                "Open a page and ask about it, or ask anything about the course."
              )}
            </p>
            {docTitle && (
              <div className="flex flex-col items-start gap-1.5">
                {STARTERS.map((s) => (
                  <Button key={s} variant="outline" size="sm" onClick={() => ask(s)}>
                    {s}
                  </Button>
                ))}
              </div>
            )}
          </div>
        ) : (
          <div className="flex flex-col gap-4">
            {turns.map((t, i) =>
              t.role === "user" ? (
                <div key={i} className="max-w-[88%] self-end rounded-2xl bg-muted px-3 py-2 text-sm whitespace-pre-wrap">
                  {t.text}
                </div>
              ) : (
                <div key={i} className="professor-answer">
                  <Markdown path={docPath} markdown={t.text} repo={repo} onOpenDoc={onOpenDoc} />
                </div>
              ),
            )}
            {streaming !== null && (
              <div className="professor-answer">
                {streaming ? (
                  <Markdown path={docPath} markdown={streaming} repo={repo} onOpenDoc={onOpenDoc} />
                ) : (
                  <span className="text-sm text-muted-foreground">Thinking…</span>
                )}
              </div>
            )}
          </div>
        )}
        {error && (
          <div className="mt-4">
            <ErrorNote>
              {error}{" "}
              <button className="underline underline-offset-4" onClick={onOpenSettings}>
                Model settings
              </button>
            </ErrorNote>
          </div>
        )}
        <div ref={end} />
      </div>

      <div className="shrink-0 p-3">
        <div className="relative">
          <Textarea
            rows={2}
            value={draft}
            placeholder="Ask the professor"
            className="min-h-0 resize-none pr-11"
            onChange={(e) => setDraft(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                ask(draft);
              }
            }}
          />
          <Button
            size="icon-sm"
            className="absolute right-2 bottom-2"
            onClick={() => ask(draft)}
            disabled={streaming !== null || !draft.trim()}
            title="Ask"
          >
            <ArrowUp />
          </Button>
        </div>
      </div>
    </aside>
  );
}
