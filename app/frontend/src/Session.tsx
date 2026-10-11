// The live session: the brief on the left, the work in the middle, the coach
// on the right, and a clock that never leaves the screen.

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import CodeMirror from "@uiw/react-codemirror";
import { oneDark } from "@codemirror/theme-one-dark";
import { javascript } from "@codemirror/lang-javascript";
import { python } from "@codemirror/lang-python";
import { go } from "@codemirror/lang-go";
import { java } from "@codemirror/lang-java";
import { rust } from "@codemirror/lang-rust";
import { Whiteboard, type WhiteboardHandle } from "./Whiteboard";
import { CoachFeed } from "./CoachFeed";
import type { SessionState } from "./agui";
import {
  FinishSession,
  PauseSession,
  ResumeSession,
  UpdateSnapshot,
  AbandonSession,
} from "../wailsjs/go/main/App";
import type { domain } from "../wailsjs/go/models";
import "./session.css";

/** How long the user must stop changing things before a snapshot is sent.
 *  Long enough not to spam the backend on every keystroke, short enough that
 *  the coach is never looking at work more than a few seconds stale. */
const SNAPSHOT_DEBOUNCE_MS = 1200;

function langExtension(lang: string) {
  switch (lang) {
    case "python":
      return [python()];
    case "typescript":
      return [javascript({ typescript: true })];
    case "javascript":
      return [javascript()];
    case "java":
      return [java()];
    case "rust":
      return [rust()];
    default:
      return [go()];
  }
}

function clock(sec: number): string {
  const abs = Math.max(0, sec);
  const m = Math.floor(abs / 60);
  const s = abs % 60;
  return `${m}:${String(s).padStart(2, "0")}`;
}

interface Props {
  state: SessionState;
  onFinished: () => void;
  onAbandoned: () => void;
}

export function Session({ state, onFinished, onAbandoned }: Props) {
  const status = state.status;
  const problem = status?.problem ?? null;
  const isCoding = problem?.mode === "coding";

  const [code, setCode] = useState("");
  const [notes, setNotes] = useState("");
  const [board, setBoard] = useState<WhiteboardHandle>({ nodes: [], edges: [] });
  const [finishing, setFinishing] = useState(false);
  const [briefOpen, setBriefOpen] = useState(true);
  const [confirmEnd, setConfirmEnd] = useState(false);

  // Seed the editor with the generated starter exactly once per problem.
  const seeded = useRef<string>("");
  useEffect(() => {
    if (problem && seeded.current !== problem.id) {
      seeded.current = problem.id;
      setCode(problem.starter ?? "");
      setNotes("");
      setBoard({ nodes: [], edges: [] });
    }
  }, [problem]);

  const snapshot: domain.Snapshot = useMemo(
    () =>
      ({
        nodes: board.nodes,
        edges: board.edges,
        notes,
        code: isCoding ? code : "",
        language: problem?.language ?? "",
      }) as domain.Snapshot,
    [board, notes, code, isCoding, problem],
  );

  // Push the work to the backend on a debounce. The coach reads whatever
  // landed most recently; it never asks the UI for state.
  const snapshotRef = useRef(snapshot);
  snapshotRef.current = snapshot;
  useEffect(() => {
    const t = setTimeout(() => {
      UpdateSnapshot(snapshotRef.current).catch(() => {
        /* a dropped snapshot costs one stale coach tick, not correctness */
      });
    }, SNAPSHOT_DEBOUNCE_MS);
    return () => clearTimeout(t);
  }, [snapshot]);

  const paused = status?.phase === "paused";
  const reviewing = status?.phase === "reviewing";
  const remaining = status?.remainingSec ?? 0;
  const duration = status?.durationSec ?? 1;
  const elapsed = status?.elapsedSec ?? 0;
  const progress = Math.min(100, (elapsed / Math.max(1, duration)) * 100);
  const overtime = remaining === 0 && elapsed > 0;

  const finish = useCallback(async () => {
    setFinishing(true);
    try {
      await FinishSession(snapshotRef.current);
      onFinished();
    } catch {
      setFinishing(false);
    }
  }, [onFinished]);

  const abandon = useCallback(async () => {
    await AbandonSession();
    onAbandoned();
  }, [onAbandoned]);

  const covered = new Set(status?.coveredAreas ?? []);
  const rubricSize = problem?.rubric?.length ?? 0;
  const coverage = rubricSize
    ? Math.round(
        ((problem?.rubric ?? []).filter((c) => covered.has(c.area)).length / rubricSize) * 100,
      )
    : 0;

  if (reviewing || finishing) {
    return (
      <div className="scoring">
        <div className="scoring-inner">
          <div className="spinner" />
          <h2>Scoring your work</h2>
          <p className="muted">
            Grading against the rubric that was written before you started, criterion by criterion.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="session">
      <header className="topbar">
        <div className="topbar-left">
          <button className="icon-btn" onClick={() => setBriefOpen(!briefOpen)} title="Toggle brief">
            {briefOpen ? "◧" : "▢"}
          </button>
          <span className="topbar-title">{problem?.title ?? "Session"}</span>
          <span className="tag">{problem?.level}</span>
        </div>

        <div className={`timer ${overtime ? "overtime" : remaining < 300 ? "urgent" : ""}`}>
          <span className="timer-value mono">{overtime ? `+${clock(elapsed - duration)}` : clock(remaining)}</span>
          <span className="timer-label faint">{overtime ? "overtime" : "left"}</span>
        </div>

        <div className="topbar-right">
          {rubricSize > 0 && (
            <div className="coverage" title="How much of the hidden rubric the coach has seen you touch">
              <div className="coverage-bar">
                <div className="coverage-fill" style={{ width: `${coverage}%` }} />
              </div>
              <span className="faint">{coverage}% covered</span>
            </div>
          )}
          <button onClick={paused ? ResumeSession : PauseSession}>{paused ? "Resume" : "Pause"}</button>
          <button className="primary" onClick={finish}>
            Finish &amp; score
          </button>
          <button className="icon-btn danger" onClick={() => setConfirmEnd(true)} title="Discard session">
            ✕
          </button>
        </div>
      </header>

      <div className="progress-rail">
        <div className={`progress-fill ${overtime ? "overtime" : ""}`} style={{ width: `${progress}%` }} />
        {(problem?.curveballs ?? []).map((c, i) => (
          <span key={i} className={`curve-mark ${c.fired ? "fired" : ""}`} style={{ left: `${c.atPct}%` }} />
        ))}
      </div>

      <div className="workspace">
        {briefOpen && (
          <aside className="brief">
            <div className="brief-scroll">
              <span className="label">The brief</span>
              <p className="brief-statement">{problem?.statement}</p>

              {!!problem?.requirements?.length && (
                <>
                  <span className="label">Requirements</span>
                  <ul className="brief-list">
                    {problem.requirements.map((r, i) => (
                      <li key={i}>{r}</li>
                    ))}
                  </ul>
                </>
              )}

              {!!problem?.constraints?.length && (
                <>
                  <span className="label">Constraints</span>
                  <ul className="brief-list constraints">
                    {problem.constraints.map((c, i) => (
                      <li key={i}>{c}</li>
                    ))}
                  </ul>
                </>
              )}

              <span className="label">Notes</span>
              <textarea
                className="notes"
                rows={9}
                value={notes}
                placeholder="Your talking track. Assumptions, numbers, things you would do with more time. This is graded too."
                onChange={(e) => setNotes(e.target.value)}
              />
            </div>
          </aside>
        )}

        <main className="canvas">
          {isCoding ? (
            <CodeMirror
              value={code}
              height="100%"
              theme={oneDark}
              extensions={langExtension(problem?.language ?? "go")}
              onChange={setCode}
              className="editor"
              basicSetup={{ lineNumbers: true, foldGutter: false, highlightActiveLine: true }}
            />
          ) : (
            <Whiteboard onChange={setBoard} />
          )}
        </main>

        <aside className="coach">
          <CoachFeed
            messages={state.messages}
            activeSteps={state.activeSteps}
            coachEnabled={status?.coachEnabled ?? false}
          />
        </aside>
      </div>

      {paused && (
        <div className="overlay">
          <div className="overlay-card">
            <h2>Paused</h2>
            <p className="muted">The clock and the coach are stopped.</p>
            <button className="primary" onClick={ResumeSession}>
              Resume
            </button>
          </div>
        </div>
      )}

      {confirmEnd && (
        <div className="overlay">
          <div className="overlay-card">
            <h2>Discard this session?</h2>
            <p className="muted">
              It will not be scored and will not count toward your profile. Finish and score instead
              if you want the feedback.
            </p>
            <div className="row" style={{ justifyContent: "center", marginTop: 6 }}>
              <button onClick={() => setConfirmEnd(false)}>Keep working</button>
              <button className="danger" onClick={abandon}>
                Discard
              </button>
            </div>
          </div>
        </div>
      )}

      {state.runError && <div className="toast-error">{state.runError}</div>}
    </div>
  );
}
