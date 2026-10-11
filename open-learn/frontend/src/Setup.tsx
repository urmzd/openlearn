// Pre-session screen: pick the mode, the bar, the clock, and the model.
//
// The model picker lives here rather than behind a settings menu because a
// misconfigured model is the one failure that makes the whole app do nothing,
// and finding that out 40 seconds into generation is the worst place to learn it.

import { useEffect, useState } from "react";
import {
  CheckModel,
  GetSettings,
  ListModels,
  SaveSettings,
  StartSession,
} from "../wailsjs/go/main/App";
import type { domain, llm, store } from "../wailsjs/go/models";
import type { Level, Mode, ProviderKind } from "./types";
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

interface Props {
  onStarted: () => void;
  onOpenHistory: () => void;
}

export function Setup({ onStarted, onOpenHistory }: Props) {
  const [settings, setSettings] = useState<store.Settings | null>(null);
  const [models, setModels] = useState<llm.ModelInfo[]>([]);
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
  const [probe, setProbe] = useState("");
  const [probing, setProbing] = useState(false);
  const [showAdvanced, setShowAdvanced] = useState(false);

  useEffect(() => {
    GetSettings().then((s) => {
      setSettings(s);
      if (s.defaults) setSpec((prev) => ({ ...prev, ...s.defaults, customStatement: "" }));
    });
    refreshModels();
  }, []);

  const refreshModels = () => {
    ListModels()
      .then(setModels)
      .catch((e) => setError(String(e)));
  };

  const updateLLM = async (patch: Partial<llm.Config>) => {
    if (!settings) return;
    const next = { ...settings, llm: { ...settings.llm, ...patch } } as store.Settings;
    setSettings(next);
    setProbe("");
    await SaveSettings(next).catch((e) => setError(String(e)));
    if (patch.kind) refreshModels();
  };

  const check = async () => {
    setProbing(true);
    setProbe("");
    setError("");
    try {
      setProbe(await CheckModel());
    } catch (e) {
      setError(String(e));
    } finally {
      setProbing(false);
    }
  };

  const start = async () => {
    if (!settings?.llm.model) {
      setError("Pick a model first.");
      return;
    }
    setStarting(true);
    setError("");
    try {
      // Remember these choices for next launch.
      await SaveSettings({ ...settings, defaults: spec } as store.Settings);
      await StartSession(spec);
      onStarted();
    } catch (e) {
      setError(String(e));
      setStarting(false);
    }
  };

  const isCoding = spec.mode === "coding";
  const kind = settings?.llm.kind ?? "ollama";
  const isOllama = kind === "ollama";

  return (
    <div className="setup">
      <div className="setup-inner">
        <header className="setup-header">
          <div>
            <h1>whiteboardy</h1>
            <p className="muted">
              Timeboxed practice with a coach that watches, and a score that tells you where you are.
            </p>
          </div>
          <button onClick={onOpenHistory}>History &amp; progress</button>
        </header>

        <section className="card">
          <span className="label">What are you practicing</span>
          <div className="mode-grid">
            <button
              className={`mode-card ${!isCoding ? "active" : ""}`}
              onClick={() => setSpec({ ...spec, mode: "system_design" } as domain.SessionSpec)}
            >
              <span className="mode-icon">◫</span>
              <span className="mode-name">System design</span>
              <span className="mode-blurb faint">
                Boxes, arrows, and the numbers behind them on a whiteboard.
              </span>
            </button>
            <button
              className={`mode-card ${isCoding ? "active" : ""}`}
              onClick={() => setSpec({ ...spec, mode: "coding" } as domain.SessionSpec)}
            >
              <span className="mode-icon">{"{ }"}</span>
              <span className="mode-name">Coding</span>
              <span className="mode-blurb faint">
                An editor, a signature to fill in, and a clock.
              </span>
            </button>
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
                    onClick={() => setSpec({ ...spec, level: l.id } as domain.SessionSpec)}
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
                    onClick={() => setSpec({ ...spec, durationSec: d.sec } as domain.SessionSpec)}
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
              <span className="label">Topic (optional)</span>
              <input
                value={spec.topic}
                placeholder={
                  isCoding ? "e.g. sliding window, graphs, parsing" : "e.g. rate limiter, feed fanout"
                }
                onChange={(e) => setSpec({ ...spec, topic: e.target.value } as domain.SessionSpec)}
              />
            </div>
            {isCoding && (
              <div>
                <span className="label">Language</span>
                <select
                  value={spec.language}
                  onChange={(e) =>
                    setSpec({ ...spec, language: e.target.value } as domain.SessionSpec)
                  }
                >
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
            <span className="label" style={{ margin: 0 }}>
              Model
            </span>
            <button className="link-btn" onClick={() => setShowAdvanced(!showAdvanced)}>
              {showAdvanced ? "Hide options" : "More options"}
            </button>
          </div>

          <div className="field-grid" style={{ marginTop: 10 }}>
            <div>
              <span className="label">Provider</span>
              <select value={kind} onChange={(e) => updateLLM({ kind: e.target.value as ProviderKind })}>
                <option value="ollama">Ollama (local)</option>
                <option value="anthropic">Anthropic</option>
                <option value="openai">OpenAI</option>
                <option value="google">Google</option>
              </select>
            </div>
            <div>
              <span className="label">Model</span>
              {isOllama && models.length > 0 ? (
                <select
                  value={settings?.llm.model ?? ""}
                  onChange={(e) => updateLLM({ model: e.target.value })}
                >
                  <option value="">Choose a model…</option>
                  {models.map((m) => (
                    <option key={m.name} value={m.name}>
                      {m.name}
                      {m.sizeBytes ? ` (${(m.sizeBytes / 1e9).toFixed(1)} GB)` : ""}
                    </option>
                  ))}
                </select>
              ) : (
                <input
                  value={settings?.llm.model ?? ""}
                  placeholder={isOllama ? "no models found - is ollama running?" : "model id"}
                  onChange={(e) => updateLLM({ model: e.target.value })}
                />
              )}
            </div>
          </div>

          {!isOllama && (
            <div style={{ marginTop: 12 }}>
              <span className="label">API key</span>
              <input
                type="password"
                value={settings?.llm.apiKey ?? ""}
                placeholder="stored locally"
                onChange={(e) => updateLLM({ apiKey: e.target.value })}
              />
            </div>
          )}

          {showAdvanced && (
            <div className="advanced">
              {isOllama && (
                <div>
                  <span className="label">Ollama host</span>
                  <input
                    value={settings?.llm.host ?? ""}
                    placeholder="http://localhost:11434"
                    onChange={(e) => updateLLM({ host: e.target.value })}
                  />
                </div>
              )}
              <label className="checkline">
                <input
                  type="checkbox"
                  checked={spec.coachEnabled}
                  onChange={(e) =>
                    setSpec({ ...spec, coachEnabled: e.target.checked } as domain.SessionSpec)
                  }
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
                        onClick={() =>
                          setSpec({ ...spec, coachIntervalSec: s } as domain.SessionSpec)
                        }
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
                  onChange={(e) =>
                    setSpec({ ...spec, customStatement: e.target.value } as domain.SessionSpec)
                  }
                />
              </div>
            </div>
          )}

          <div className="row" style={{ marginTop: 12 }}>
            <button onClick={check} disabled={probing || !settings?.llm.model}>
              {probing ? "Checking…" : "Test connection"}
            </button>
            {probe && <span className="probe-ok">{probe}</span>}
          </div>
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
