// The practice runner: the pack's exercises, an editor on your attempt, and the
// verdict from the pack's own CLI.
//
// Nothing here decides whether an attempt is right. The CLI clones the exercise
// into the scratchpad, this edits the files it put there, and "Check" runs the
// same command you would type in a terminal. The exit code is the verdict.

import { useCallback, useEffect, useMemo, useState } from "react";
import CodeMirror from "@uiw/react-codemirror";
import { oneDark } from "@codemirror/theme-one-dark";
import {
  CheckExercise,
  DiffExercise,
  ListExercises,
  ResetExercise,
  SaveExerciseFile,
  StartExercise,
} from "../wailsjs/go/main/App";
import type { practice } from "../wailsjs/go/models";
import { langForFile } from "./lang";

export interface ExerciseRef {
  kind: string;
  lang: string;
  name: string;
}

const same = (a: ExerciseRef | null, b: ExerciseRef) =>
  !!a && a.kind === b.kind && a.lang === b.lang && a.name === b.name;

const KIND_BLURB: Record<string, string> = {
  predict: "Write the exact output before you run it",
  build: "Implement it from scratch against its own tests",
  reattempt: "Something you solved before, with the solution removed",
};

export function PracticeList({
  active,
  refresh,
  onOpen,
}: {
  active: ExerciseRef | null;
  refresh: number;
  onOpen: (ref: ExerciseRef) => void;
}) {
  const [groups, setGroups] = useState<practice.Group[] | null>(null);
  const [error, setError] = useState("");
  const [open, setOpen] = useState<Record<string, boolean>>({});

  useEffect(() => {
    ListExercises()
      .then((g) => {
        setGroups(g ?? []);
        setError("");
      })
      .catch((e) => setError(String(e)));
  }, [refresh]);

  if (error) return <div className="side-pad error-box">{error}</div>;
  if (!groups) return <div className="side-pad faint">Listing exercises…</div>;
  if (groups.length === 0) {
    return (
      <div className="side-pad faint">
        This pack has no exercises that can run here. Exercises need the pack opened from a folder
        on disk, with its practice CLI.
      </div>
    );
  }

  const kinds = [...new Set(groups.map((g) => g.kind))];
  return (
    <div className="side-scroll">
      {kinds.map((kind) => (
        <section key={kind} className="side-section">
          <span className="label" title={KIND_BLURB[kind]}>
            {kind}
          </span>
          {groups
            .filter((g) => g.kind === kind)
            .map((g) => {
              const key = `${g.kind}/${g.lang}`;
              const isOpen = open[key] ?? (active?.kind === g.kind && active.lang === g.lang);
              const started = g.exercises.filter((e) => e.started).length;
              return (
                <div key={key}>
                  <button
                    className="tree-row dir"
                    onClick={() => setOpen((o) => ({ ...o, [key]: !isOpen }))}
                  >
                    <span className="twist">{isOpen ? "▾" : "▸"}</span>
                    {g.lang}
                    <span className="faint mono row-count">
                      {started}/{g.exercises.length}
                    </span>
                  </button>
                  {isOpen &&
                    g.exercises.map((e) => {
                      const ref = { kind: g.kind, lang: g.lang, name: e.name };
                      return (
                        <button
                          key={e.name}
                          className={`tree-row doc ${same(active, ref) ? "active" : ""}`}
                          style={{ paddingLeft: 24 }}
                          onClick={() => onOpen(ref)}
                        >
                          <span className={`dot ${e.started ? "started" : ""}`} />
                          {e.name}
                        </button>
                      );
                    })}
                </div>
              );
            })}
        </section>
      ))}
    </div>
  );
}

export function PracticeRunner({
  exercise,
  onChanged,
}: {
  exercise: ExerciseRef;
  /** Called when an attempt is started or reset, so the list's marks follow. */
  onChanged: () => void;
}) {
  const { kind, lang, name } = exercise;
  const [ws, setWs] = useState<practice.Workspace | null>(null);
  const [file, setFile] = useState("");
  const [edits, setEdits] = useState<Record<string, string>>({});
  const [result, setResult] = useState<practice.Result | null>(null);
  const [busy, setBusy] = useState("");
  const [error, setError] = useState("");

  const start = useCallback(async () => {
    setBusy("Starting…");
    setError("");
    try {
      const w = await StartExercise(kind, lang, name);
      setWs(w);
      setEdits({});
      // Open on the file you write, not the one you read.
      const first = w.files.find((f) => !f.readOnly && /(^|\/)main\./i.test(f.path)) ?? w.files.find((f) => !f.readOnly);
      setFile(first?.path ?? w.files[0]?.path ?? "");
      onChanged();
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy("");
    }
  }, [kind, lang, name, onChanged]);

  useEffect(() => {
    setWs(null);
    setResult(null);
    start();
  }, [start]);

  const current = ws?.files.find((f) => f.path === file);
  const value = current ? (edits[current.path] ?? current.content) : "";
  const dirty = Object.keys(edits).length > 0;

  const save = useCallback(async () => {
    for (const [path, content] of Object.entries(edits)) {
      await SaveExerciseFile(kind, lang, name, path, content);
    }
    if (ws && dirty) {
      setWs({ ...ws, files: ws.files.map((f) => (f.path in edits ? { ...f, content: edits[f.path] } : f)) } as practice.Workspace);
      setEdits({});
    }
  }, [edits, kind, lang, name, ws, dirty]);

  const run = async (label: string, fn: () => Promise<practice.Result>) => {
    setBusy(label);
    setError("");
    try {
      await save();
      setResult(await fn());
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy("");
    }
  };

  const reset = async () => {
    setBusy("Resetting…");
    try {
      await ResetExercise(kind, lang, name);
      setResult(null);
      await start();
    } catch (e) {
      setError(String(e));
      setBusy("");
    }
  };

  const extensions = useMemo(() => langForFile(file), [file]);
  const predict = kind === "predict";

  return (
    <div className="runner">
      <header className="runner-bar">
        <div className="runner-title">
          <span className="tag">{kind}</span>
          <span className="mono">
            {lang}/{name}
          </span>
          {dirty && <span className="faint">unsaved</span>}
        </div>
        <div className="row">
          {busy && <span className="faint">{busy}</span>}
          {!predict && (
            <button
              onClick={() => run("Diffing…", () => DiffExercise(kind, lang, name))}
              disabled={!!busy}
              title="Compare your attempt with the reference. This spoils the exercise."
            >
              Diff with reference
            </button>
          )}
          <button className="danger" onClick={reset} disabled={!!busy} title="Throw this attempt away and start cold">
            Reset
          </button>
          <button
            className="primary"
            onClick={() => run("Checking…", () => CheckExercise(kind, lang, name))}
            disabled={!!busy}
          >
            Check
          </button>
        </div>
      </header>

      {error && <div className="error-box runner-error">{error}</div>}

      <div className="runner-tabs">
        {ws?.files.map((f) => (
          <button
            key={f.path}
            className={`tab ${f.path === file ? "active" : ""}`}
            onClick={() => setFile(f.path)}
            title={f.readOnly ? "Read only" : f.path}
          >
            {f.path.split("/").pop()}
            {f.readOnly && <span className="faint"> (read)</span>}
            {f.path in edits && " •"}
          </button>
        ))}
      </div>

      <div className="runner-editor">
        {predict && current && !current.readOnly && (
          <div className="runner-hint faint">
            Write the exact output you expect, then check. A prediction edited after you have seen
            the answer is worth nothing.
          </div>
        )}
        {current && (
          <CodeMirror
            key={current.path}
            value={value}
            height="100%"
            theme={oneDark}
            extensions={extensions}
            editable={!current.readOnly}
            onChange={(v) => !current.readOnly && setEdits((e) => ({ ...e, [current.path]: v }))}
            className="editor"
            basicSetup={{ lineNumbers: true, foldGutter: false, highlightActiveLine: !current.readOnly }}
          />
        )}
      </div>

      <div className={`runner-output ${result?.verdict ?? ""}`}>
        <div className="runner-output-head">
          <span className="label" style={{ margin: 0 }}>
            Output
          </span>
          {result && result.verdict !== "none" && (
            <span className={`verdict ${result.verdict}`}>{result.verdict === "pass" ? "PASS" : "FAIL"}</span>
          )}
        </div>
        <pre className="mono">
          {result ? result.output || "(no output)" : "Check runs your attempt. The exit code is the verdict."}
        </pre>
      </div>
    </div>
  );
}
