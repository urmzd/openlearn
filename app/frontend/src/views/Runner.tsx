// One exercise: an editor on your attempt and the verdict from the pack's CLI.
//
// Nothing here decides whether an attempt is right. The backend clones the
// exercise into the scratchpad, this edits the files it put there, and "Check"
// runs the attempt against the exercise's own tests. The exit code is the
// verdict.

import { useCallback, useEffect, useMemo, useState } from "react";
import CodeMirror from "@uiw/react-codemirror";
import { oneDark } from "@codemirror/theme-one-dark";
import { GitCompare, Play, RotateCcw } from "lucide-react";
import {
  CheckExercise,
  DiffExercise,
  ResetExercise,
  SaveExerciseFile,
  StartExercise,
} from "../../wailsjs/go/main/App";
import type { practice } from "../../wailsjs/go/models";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { ErrorNote, PageHeader } from "@/components/page";
import { langForFile } from "@/lang";
import { cn } from "@/lib/utils";
import type { ExerciseRef } from "@/routes";

interface Props {
  exercise: ExerciseRef;
  onBack: () => void;
  /** Called when an attempt is started or reset, so the list's marks follow. */
  onChanged: () => void;
}

export function Runner({ exercise, onBack, onChanged }: Props) {
  const { kind, lang, name } = exercise;
  const [ws, setWs] = useState<practice.Workspace | null>(null);
  const [file, setFile] = useState("");
  const [edits, setEdits] = useState<Record<string, string>>({});
  const [result, setResult] = useState<practice.Result | null>(null);
  const [busy, setBusy] = useState("");
  const [error, setError] = useState("");

  const start = useCallback(async () => {
    setBusy("Starting");
    setError("");
    try {
      const w = await StartExercise(kind, lang, name);
      setWs(w);
      setEdits({});
      // Open on the file you write, not the one you read.
      const writable = w.files.filter((f) => !f.readOnly);
      const first = writable.find((f) => /(^|\/)main\./i.test(f.path)) ?? writable[0] ?? w.files[0];
      setFile(first?.path ?? "");
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
      const files = ws.files.map((f) => (f.path in edits ? { ...f, content: edits[f.path] } : f));
      setWs({ ...ws, files } as practice.Workspace);
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
    setBusy("Resetting");
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
  const verdict = result && result.verdict !== "none" ? result.verdict : "";

  return (
    <>
      <PageHeader
        crumbs={[{ label: "Exercises", onClick: onBack }, { label: `${kind} / ${lang}` }, { label: name }]}
        actions={
          <>
            {busy && <span className="text-xs text-muted-foreground">{busy}…</span>}
            {!predict && (
              <Button
                variant="ghost"
                size="sm"
                disabled={!!busy}
                onClick={() => run("Diffing", () => DiffExercise(kind, lang, name))}
                title="Compare your attempt with the reference. This spoils the exercise."
              >
                <GitCompare data-icon="inline-start" />
                Diff
              </Button>
            )}
            <Button
              variant="ghost"
              size="sm"
              disabled={!!busy}
              onClick={reset}
              title="Throw this attempt away and start cold"
            >
              <RotateCcw data-icon="inline-start" />
              Reset
            </Button>
            <Button
              size="sm"
              disabled={!!busy}
              onClick={() => run("Checking", () => CheckExercise(kind, lang, name))}
            >
              <Play data-icon="inline-start" />
              Check
            </Button>
          </>
        }
      />

      {error && (
        <div className="p-3">
          <ErrorNote>{error}</ErrorNote>
        </div>
      )}

      <div className="flex h-9 shrink-0 items-stretch overflow-x-auto border-b bg-muted/30">
        {ws?.files.map((f) => (
          <button
            key={f.path}
            onClick={() => setFile(f.path)}
            title={f.path}
            className={cn(
              "flex items-center gap-1.5 border-r px-3 font-mono text-xs whitespace-nowrap text-muted-foreground",
              f.path === file && "bg-background text-foreground shadow-[inset_0_-2px_0_var(--foreground)]",
            )}
          >
            {f.path.split("/").pop()}
            {f.readOnly && <span className="text-muted-foreground/70">read only</span>}
            {f.path in edits && <span className="size-1.5 rounded-full bg-foreground" />}
          </button>
        ))}
      </div>

      {predict && current && !current.readOnly && (
        <p className="border-b px-3 py-2 text-xs text-muted-foreground">
          Write the exact output you expect, then check. A prediction edited after you have seen the
          answer is worth nothing.
        </p>
      )}

      <div className="min-h-0 flex-1">
        {current && (
          <CodeMirror
            key={current.path}
            value={value}
            height="100%"
            theme={oneDark}
            extensions={extensions}
            editable={!current.readOnly}
            onChange={(v) => !current.readOnly && setEdits((e) => ({ ...e, [current.path]: v }))}
            className="h-full text-[13px] [&_.cm-editor]:h-full"
            basicSetup={{ lineNumbers: true, foldGutter: false, highlightActiveLine: !current.readOnly }}
          />
        )}
      </div>

      <div
        className={cn(
          "flex h-[30%] min-h-32 shrink-0 flex-col border-t-2",
          verdict === "pass" && "border-t-emerald-500",
          verdict === "fail" && "border-t-destructive",
        )}
      >
        <div className="flex h-9 shrink-0 items-center justify-between px-3">
          <span className="text-xs font-medium text-muted-foreground">Output</span>
          {verdict === "pass" && <Badge className="bg-emerald-500 text-background">Pass</Badge>}
          {verdict === "fail" && <Badge variant="destructive">Fail</Badge>}
        </div>
        <pre className="min-h-0 flex-1 overflow-auto px-3 pb-3 font-mono text-xs leading-relaxed whitespace-pre-wrap text-muted-foreground">
          {result ? result.output || "(no output)" : "Check runs your attempt. The exit code is the verdict."}
        </pre>
      </div>
    </>
  );
}
