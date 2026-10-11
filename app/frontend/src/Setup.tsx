// Pre-session screen: pick what to practice, the bar, and the clock.
//
// A session can be started cold, or from a document, in which case the exercise
// and its hidden rubric are written from that document. Which modes are offered
// is the pack's call: a language guide has no use for a system design board.

import { useEffect, useState } from "react";
import { GetSettings, SaveSettings, StartSession } from "../wailsjs/go/main/App";
import type { domain, store } from "../wailsjs/go/models";
import type { Level, Mode } from "./types";
import "./setup.css";

const DURATIONS = [
  { label: "15 min", sec: 15 * 60, blurb: "warm-up" },
  { label: "30 min", sec: 30 * 60, blurb: "focused" },
  { label: "45 min", sec: 45 * 60, blurb: "interview length" },
  { label: "60 min", sec: 60 * 60, blurb: "deep" },
];

const LEVELS: { id: Level; label: string; blurb: string }[] = [
  { id: "junior", label: "Junior", blurb: "concrete and bounded" },
  { id: "mid", label: "Mid", blurb: "one clear right shape" },
  { id: "senior", label: "Senior", blurb: "tradeoffs matter" },
  { id: "staff", label: "Staff", blurb: "ambiguous on purpose" },
];

const LANGUAGES = ["go", "python", "typescript", "java", "rust", "javascript"];

const MODES: Record<Mode, { icon: string; name: string; blurb: string; topic: string }> = {
  system_design: {
    icon: "◫",
    name: "System design",
    blurb: "Boxes, arrows, and the numbers behind them on a whiteboard.",
    topic: "e.g. rate limiter, feed fanout",
  },
  coding: {
    icon: "{ }",
    name: "Coding",
    blurb: "An editor, a signature to fill in, and a clock.",
    topic: "e.g. sliding window, graphs, parsing",
  },
  written: {
    icon: "¶",
    name: "Written answer",
    blurb: "Explain it, derive it, or produce it from memory, in prose.",
    topic: "e.g. backprop through softmax, ordering at a restaurant",
  },
};

export interface Source {
  path: string;
  title: string;
}

interface Props {
  /** The modes the open pack offers, in its order. */
  modes: string[];
  source: Source | null;
  onClearSource: () => void;
  onOpenSettings: () => void;
}

export function Setup({ modes, source, onClearSource, onOpenSettings }: Props) {
  const offered = modes.filter((m): m is Mode => m in MODES);
  const [model, setModel] = useState<string | null>(null);
  const [spec, setSpec] = useState<domain.SessionSpec>({
    mode: "system_design",
    level: "senior",
    topic: "",
    language: "go",
    durationSec: 45 * 60,
    customStatement: "",
    coachEnabled: true,
    coachIntervalSec: 45,
  } as domain.SessionSpec);

  const [starting, setStarting] = useState(false);
  const [error, setError] = useState("");
  const [showAdvanced, setShowAdvanced] = useState(false);

  useEffect(() => {
    GetSettings().then((s) => {
      setModel(s.llm.model ?? "");
      if (s.defaults) setSpec((prev) => ({ ...prev, ...s.defaults, customStatement: "", topic: "" }));
    });
  }, []);

  // The remembered mode may not be one this pack offers.
  const mode: Mode = offered.includes(spec.mode as Mode) ? (spec.mode as Mode) : (offered[0] ?? "written");
  const patch = (p: Partial<domain.SessionSpec>) => setSpec({ ...spec, ...p } as domain.SessionSpec);

  const start = async () => {
    setStarting(true);
    setError("");
    try {
      const settings = await GetSettings();
      if (!settings.llm.model) throw new Error("Pick a model first, in Settings.");
      const next = { ...spec, mode, source: source?.path ?? "", sourceTitle: "" } as domain.SessionSpec;
      // Remember these choices for next launch, but not the chapter: the next
      // session is as likely to be about something else.
      await SaveSettings({ ...settings, defaults: { ...next, source: "" } } as store.Settings);
      await StartSession(next);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setStarting(false);
    }
  };

  return (
    <div className="setup">
      <div className="setup-inner">
        <header className="setup-header">
          <div>
            <h1>Practice under a clock</h1>
            <p className="muted">
              A timed exercise, a coach that watches, and a score against a rubric written before
              you start.
            </p>
          </div>
        </header>

        {source && (
          <section className="card source-card">
            <div>
              <span className="label">Drawn from</span>
              <span className="source-title">{source.title}</span>
              <span className="faint mono source-path">{source.path}</span>
            </div>
            <button className="link-btn" onClick={onClearSource}>
              Practice without a chapter
            </button>
          </section>
        )}

        <section className="card">
          <span className="label">What are you practicing</span>
          <div className="mode-grid" style={{ gridTemplateColumns: `repeat(${Math.max(1, offered.length)}, 1fr)` }}>
            {offered.map((m) => (
              <button
                key={m}
                className={`mode-card ${mode === m ? "active" : ""}`}
                onClick={() => patch({ mode: m })}
              >
                <span className="mode-icon">{MODES[m].icon}</span>
                <span className="mode-name">{MODES[m].name}</span>
                <span className="mode-blurb faint">{MODES[m].blurb}</span>
              </button>
            ))}
          </div>
        </section>

        <section className="card">
          <div className="field-grid">
            <div>
              <span className="label">Bar to hit</span>
              <div className="pill-row">
                {LEVELS.map((l) => (
                  <button
                    key={l.id}
                    className={`pill ${spec.level === l.id ? "active" : ""}`}
                    onClick={() => patch({ level: l.id })}
                    title={l.blurb}
                  >
                    {l.label}
                  </button>
                ))}
              </div>
            </div>
            <div>
              <span className="label">Timebox</span>
              <div className="pill-row">
                {DURATIONS.map((d) => (
                  <button
                    key={d.sec}
                    className={`pill ${spec.durationSec === d.sec ? "active" : ""}`}
                    onClick={() => patch({ durationSec: d.sec })}
                    title={d.blurb}
                  >
                    {d.label}
                  </button>
                ))}
              </div>
            </div>
          </div>

          <div className="field-grid" style={{ marginTop: 14 }}>
            <div>
              <span className="label">{source ? "Focus within the chapter (optional)" : "Topic (optional)"}</span>
              <input
                value={spec.topic}
                placeholder={MODES[mode].topic}
                onChange={(e) => patch({ topic: e.target.value })}
              />
            </div>
            {mode === "coding" && (
              <div>
                <span className="label">Language</span>
                <select value={spec.language} onChange={(e) => patch({ language: e.target.value })}>
                  {LANGUAGES.map((l) => (
                    <option key={l} value={l}>
                      {l}
                    </option>
                  ))}
                </select>
              </div>
            )}
          </div>
        </section>

        <section className="card">
          <div className="spread">
            <span>
              <span className="label" style={{ display: "inline", marginRight: 8 }}>
                Model
              </span>
              {model === null ? "" : model ? <span className="mono">{model}</span> : <span className="warn-text">none chosen</span>}
            </span>
            <span className="row">
              <button className="link-btn" onClick={onOpenSettings}>
                Change
              </button>
              <button className="link-btn" onClick={() => setShowAdvanced(!showAdvanced)}>
                {showAdvanced ? "Hide options" : "More options"}
              </button>
            </span>
          </div>

          {showAdvanced && (
            <div className="advanced">
              <label className="checkline">
                <input
                  type="checkbox"
                  checked={spec.coachEnabled}
                  onChange={(e) => patch({ coachEnabled: e.target.checked })}
                />
                <span>
                  Live coach
                  <span className="faint"> - interrupts only when it is worth it</span>
                </span>
              </label>
              {spec.coachEnabled && (
                <div>
                  <span className="label">How often it may consider speaking</span>
                  <div className="pill-row">
                    {[30, 45, 60, 90].map((s) => (
                      <button
                        key={s}
                        className={`pill ${spec.coachIntervalSec === s ? "active" : ""}`}
                        onClick={() => patch({ coachIntervalSec: s })}
                      >
                        {s}s
                      </button>
                    ))}
                  </div>
                </div>
              )}
              <div>
                <span className="label">Bring your own problem (optional)</span>
                <textarea
                  rows={3}
                  value={spec.customStatement}
                  placeholder="Paste a problem statement to practice against. The rubric is still generated from it."
                  onChange={(e) => patch({ customStatement: e.target.value })}
                />
              </div>
            </div>
          )}
        </section>

        {error && <div className="error-box">{error}</div>}

        <div className="setup-actions">
          <button className="primary big" onClick={start} disabled={starting}>
            {starting ? "Writing your problem…" : "Start session"}
          </button>
          {starting && (
            <span className="faint">
              The model is writing an exercise and a hidden rubric. This takes a moment on local
              models.
            </span>
          )}
        </div>
      </div>
    </div>
  );
}
