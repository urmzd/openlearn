// One role path: its stages, what "done" means for each, and which you have
// finished. Progress is the pack's, not the app's: a stage marked here shows as
// done in `ss learn`, and one marked there shows here.

import { useCallback, useEffect, useState } from "react";
import { GetPath, MarkStage } from "../wailsjs/go/main/App";
import type { course } from "../wailsjs/go/models";

interface Props {
  name: string;
  onOpenDoc: (path: string) => void;
  /** Called after a mark changes, so the sidebar's counts follow. */
  onProgress: () => void;
}

export function PathView({ name, onOpenDoc, onProgress }: Props) {
  const [path, setPath] = useState<course.Path | null>(null);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState<{ key: string; text: string } | null>(null);

  const load = useCallback(() => {
    GetPath(name)
      .then((p) => {
        setPath(p);
        setError("");
      })
      .catch((e) => setError(String(e)));
  }, [name]);

  useEffect(() => {
    setPath(null);
    setNotice(null);
    load();
  }, [load]);

  const mark = async (s: course.Stage) => {
    setNotice(null);
    try {
      const res = await MarkStage(name, s.key, !s.done);
      if (res.verdict === "fail") setNotice({ key: s.key, text: res.output });
      load();
      onProgress();
    } catch (e) {
      setNotice({ key: s.key, text: String(e) });
    }
  };

  if (error) return <div className="pane-pad error-box">{error}</div>;
  if (!path) return <div className="pane-pad faint">Loading…</div>;

  const done = path.stages.filter((s) => s.done).length;
  const next = path.stages.find((s) => !s.done);
  let owner = "";

  return (
    <div className="pane-scroll">
      <div className="pathview">
        <header className="pathview-header">
          <div>
            <h1>{path.title}</h1>
            <p className="muted">
              {done} of {path.stages.length} stages finished.{" "}
              <button className="link-btn" onClick={() => onOpenDoc(path.readme)}>
                About this path
              </button>
            </p>
          </div>
          {next && (
            <button className="primary" onClick={() => onOpenDoc(next.files[0])}>
              Continue: {next.title}
            </button>
          )}
        </header>

        {path.stages.map((s) => {
          const heading = s.owner !== owner && s.owner !== name ? s.owner : "";
          owner = s.owner;
          return (
            <div key={s.key}>
              {heading && <span className="label part-label">Part: {heading}</span>}
              <div className={`stage ${s.done ? "done" : ""} ${next?.key === s.key ? "next" : ""}`}>
                <button
                  className={`stage-check ${s.done ? "done" : ""}`}
                  onClick={() => mark(s)}
                  title={s.done ? "Mark as not finished" : "Mark as finished"}
                >
                  {s.done ? "✓" : ""}
                </button>
                <div className="stage-body">
                  <div className="stage-title">
                    <span className="mono faint">{s.key}</span> {s.title}
                  </div>
                  <div className="stage-done-when">
                    <span className="faint">Done when:</span> {s.doneWhen}
                  </div>
                  {s.check && (
                    <div className="stage-check-line faint">
                      Verified by <span className="mono">{s.check}</span>
                    </div>
                  )}
                  <div className="stage-files">
                    {s.files.map((f) => (
                      <button key={f} className="chip" onClick={() => onOpenDoc(f)} title={f}>
                        {f.replace(/\/README\.md$/, "").replace(/\.md$/, "")}
                      </button>
                    ))}
                  </div>
                  {notice?.key === s.key && <pre className="stage-notice">{notice.text}</pre>}
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
