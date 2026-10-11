// Pre-session screen: pick what to practice, the bar, and the clock.
//
// A session can be started cold, or from a document, in which case the exercise
// and its hidden rubric are written from that document. Which modes are offered
// is the pack's call: a language guide has no use for a system design board.

import { useEffect, useState } from "react";
import { Braces, LayoutPanelLeft, Pilcrow, Play, X } from "lucide-react";
import {
  GetSettings,
  SaveSettings,
  StartSession,
} from "../../wailsjs/go/main/App";
import type { domain, store } from "../../wailsjs/go/models";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Switch } from "@/components/ui/switch";
import { Textarea } from "@/components/ui/textarea";
import { ChoiceGroup, Field, Picker } from "@/components/choice";
import { ModelSettings } from "@/components/model-settings";
import { ErrorNote, PageBody, PageHeader, PageTitle } from "@/components/page";
import { cn } from "@/lib/utils";
import type { Source } from "@/routes";
import type { Level, Mode } from "@/types";

const DURATIONS = [
  { value: 15 * 60, label: "15 min", hint: "warm-up" },
  { value: 30 * 60, label: "30 min", hint: "focused" },
  { value: 45 * 60, label: "45 min", hint: "interview length" },
  { value: 60 * 60, label: "60 min", hint: "deep" },
];

const LEVELS: { value: Level; label: string; hint: string }[] = [
  { value: "junior", label: "Junior", hint: "concrete and bounded" },
  { value: "mid", label: "Mid", hint: "one clear right shape" },
  { value: "senior", label: "Senior", hint: "tradeoffs matter" },
  { value: "staff", label: "Staff", hint: "ambiguous on purpose" },
];

const INTERVALS = [30, 45, 60, 90].map((s) => ({ value: s, label: `${s}s` }));

const LANGUAGES = [
  "go",
  "python",
  "typescript",
  "java",
  "rust",
  "javascript",
].map((l) => ({ value: l, label: l }));

const MODES: Record<
  Mode,
  { icon: typeof Braces; name: string; blurb: string; topic: string }
> = {
  system_design: {
    icon: LayoutPanelLeft,
    name: "System design",
    blurb: "Boxes, arrows, and the numbers behind them on a whiteboard.",
    topic: "e.g. rate limiter, feed fanout",
  },
  coding: {
    icon: Braces,
    name: "Coding",
    blurb: "An editor, a signature to fill in, and a clock.",
    topic: "e.g. sliding window, graphs, parsing",
  },
  written: {
    icon: Pilcrow,
    name: "Written answer",
    blurb: "Explain it, derive it, or produce it from memory, in prose.",
    topic: "e.g. backprop through softmax, ordering at a restaurant",
  },
};

interface Props {
  /** The modes the open pack offers, in its order. */
  modes: string[];
  source: Source | null;
  onClearSource: () => void;
}

export function SessionSetup({ modes, source, onClearSource }: Props) {
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
  // The model is configured here, where it is used. It stays folded away once
  // one is chosen, and shows itself when none is.
  const [showModel, setShowModel] = useState(false);
  const [starting, setStarting] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    GetSettings().then((s) => {
      setModel(s.llm.model ?? "");
      if (s.defaults)
        setSpec((prev) => ({
          ...prev,
          ...s.defaults,
          customStatement: "",
          topic: "",
        }));
    });
  }, []);

  // The remembered mode may not be one this pack offers.
  const mode: Mode = offered.includes(spec.mode as Mode)
    ? (spec.mode as Mode)
    : (offered[0] ?? "written");
  const patch = (p: Partial<domain.SessionSpec>) =>
    setSpec({ ...spec, ...p } as domain.SessionSpec);

  const start = async () => {
    setStarting(true);
    setError("");
    try {
      const settings = await GetSettings();
      if (!settings.llm.model)
        throw new Error("Choose a model first, under Model below.");
      const next = {
        ...spec,
        mode,
        source: source?.path ?? "",
        sourceTitle: "",
      } as domain.SessionSpec;
      // Remember these choices for next launch, but not the chapter: the next
      // session is as likely to be about something else.
      await SaveSettings({
        ...settings,
        defaults: { ...next, source: "" },
      } as store.Settings);
      await StartSession(next);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setStarting(false);
    }
  };

  return (
    <>
      <PageHeader crumbs={[{ label: "Timed session" }]} />
      <PageBody>
        <PageTitle
          title="Practice under a clock"
          description="A timed exercise, a coach that watches, and a score against a rubric written before you start."
        />

        <div className="flex flex-col gap-6">
          {source && (
            <div className="flex items-center justify-between gap-4 rounded-xl border bg-muted/40 p-4">
              <div className="min-w-0">
                <div className="text-xs font-medium text-muted-foreground">
                  Drawn from
                </div>
                <div className="truncate text-sm font-medium">
                  {source.title}
                </div>
                <div className="truncate font-mono text-xs text-muted-foreground">
                  {source.path}
                </div>
              </div>
              <Button
                variant="ghost"
                size="icon-sm"
                onClick={onClearSource}
                title="Practice without a chapter"
              >
                <X />
              </Button>
            </div>
          )}

          <Field label="What are you practicing">
            <div
              className="grid gap-2"
              style={{
                gridTemplateColumns: `repeat(${Math.max(1, offered.length)}, minmax(0, 1fr))`,
              }}
            >
              {offered.map((m) => {
                const Icon = MODES[m].icon;
                return (
                  <button
                    key={m}
                    type="button"
                    onClick={() => patch({ mode: m })}
                    className={cn(
                      "flex flex-col items-start gap-1 rounded-xl border p-4 text-left transition-colors hover:bg-muted/60",
                      mode === m && "border-foreground/40 bg-muted",
                    )}
                  >
                    <Icon className="mb-1 size-4" />
                    <span className="text-sm font-medium">{MODES[m].name}</span>
                    <span className="text-xs text-muted-foreground">
                      {MODES[m].blurb}
                    </span>
                  </button>
                );
              })}
            </div>
          </Field>

          <div className="grid grid-cols-2 gap-6">
            <Field label="Bar to hit">
              <ChoiceGroup
                value={spec.level as Level}
                options={LEVELS}
                onChange={(level) => patch({ level })}
              />
            </Field>
            <Field label="Timebox">
              <ChoiceGroup
                value={spec.durationSec}
                options={DURATIONS}
                onChange={(durationSec) => patch({ durationSec })}
              />
            </Field>
          </div>

          <div className="grid grid-cols-2 gap-6">
            <Field
              label={
                source
                  ? "Focus within the chapter (optional)"
                  : "Topic (optional)"
              }
            >
              <Input
                value={spec.topic}
                placeholder={MODES[mode].topic}
                onChange={(e) => patch({ topic: e.target.value })}
              />
            </Field>
            {mode === "coding" && (
              <Field label="Language">
                <Picker
                  value={spec.language}
                  options={LANGUAGES}
                  onChange={(language) => patch({ language })}
                />
              </Field>
            )}
          </div>

          <div className="flex flex-col gap-4 rounded-xl border p-4">
            <label className="flex items-center justify-between gap-4">
              <span>
                <span className="block text-sm font-medium">Live coach</span>
                <span className="block text-xs text-muted-foreground">
                  Watches your work and interrupts only when it is worth it.
                </span>
              </span>
              <Switch
                checked={spec.coachEnabled}
                onCheckedChange={(coachEnabled) => patch({ coachEnabled })}
              />
            </label>
            {spec.coachEnabled && (
              <Field label="How often it may consider speaking">
                <ChoiceGroup
                  value={spec.coachIntervalSec}
                  options={INTERVALS}
                  onChange={(coachIntervalSec) => patch({ coachIntervalSec })}
                />
              </Field>
            )}
            <Field label="Bring your own problem (optional)">
              <Textarea
                rows={3}
                value={spec.customStatement}
                placeholder="Paste a problem statement to practice against. The rubric is still generated from it."
                onChange={(e) => patch({ customStatement: e.target.value })}
              />
            </Field>
          </div>

          {(showModel || model === "") && (
            <Field label="Model">
              <ModelSettings onChange={setModel} />
            </Field>
          )}

          {error && <ErrorNote>{error}</ErrorNote>}

          <div className="flex items-center gap-4">
            <Button size="lg" onClick={start} disabled={starting}>
              <Play data-icon="inline-start" />
              {starting ? "Writing your exercise…" : "Start session"}
            </Button>
            <span className="text-xs text-muted-foreground">
              {starting ? (
                "The model is writing an exercise and a hidden rubric. This takes a moment on local models."
              ) : model === null ? (
                ""
              ) : (
                <>
                  Model:{" "}
                  {model ? (
                    <span className="font-mono">{model}</span>
                  ) : (
                    "none chosen"
                  )}{" "}
                  <button
                    className="underline underline-offset-4"
                    onClick={() => setShowModel(!showModel)}
                  >
                    {showModel ? "hide" : "change"}
                  </button>
                </>
              )}
            </span>
          </div>
        </div>
      </PageBody>
    </>
  );
}
