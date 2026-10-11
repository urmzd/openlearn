// Where you're at: what you have finished, and the cross-session view.
//
// A single session tells you how one exercise went. This tells you which skills
// hold up under a clock and which keep slipping, which is the thing you cannot
// see from inside any one session.

import { useEffect, useState } from "react";
import { ArrowDown, ArrowUp } from "lucide-react";
import { GetProfile, ListSessions } from "../../wailsjs/go/main/App";
import type { course, domain } from "../../wailsjs/go/models";
import { Progress as Bar } from "@/components/ui/progress";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { PageBody, PageHeader, PageTitle, SectionLabel } from "@/components/page";
import { ListGroup, ListRow } from "@/components/list-row";
import { cn } from "@/lib/utils";
import type { Mode } from "@/types";

const MODE_LABEL: Record<Mode, string> = {
  system_design: "System design",
  coding: "Coding",
  written: "Written answer",
};

function when(iso: string): string {
  const d = new Date(iso);
  const days = Math.floor((Date.now() - d.getTime()) / 86400000);
  if (days === 0) return "today";
  if (days === 1) return "yesterday";
  if (days < 30) return `${days} days ago`;
  return d.toLocaleDateString();
}

/** Scores are 0-4: 3 and up is solid, under 2 is a gap. */
function tone(avg: number): string {
  if (avg >= 3) return "bg-emerald-500";
  if (avg >= 2) return "bg-amber-500";
  return "bg-destructive";
}

function toneText(avg: number): string {
  if (avg >= 3) return "text-emerald-600 dark:text-emerald-400";
  if (avg >= 2) return "text-amber-600 dark:text-amber-400";
  return "text-destructive";
}

interface Props {
  /** The modes the open pack offers. */
  modes: string[];
  paths: course.PathSummary[];
  onOpenPath: (name: string) => void;
}

export function Progress({ modes, paths, onOpenPath }: Props) {
  const offered = modes.filter((m): m is Mode => m in MODE_LABEL);
  const [mode, setMode] = useState<Mode>(offered[0] ?? "system_design");
  const [profile, setProfile] = useState<domain.Profile | null>(null);
  const [sessions, setSessions] = useState<domain.Session[]>([]);

  useEffect(() => {
    GetProfile(mode).then(setProfile).catch(() => setProfile(null));
    ListSessions()
      .then((s) => setSessions(s ?? []))
      .catch(() => setSessions([]));
  }, [mode]);

  const forMode = sessions.filter((s) => s.spec.mode === mode && s.review);

  return (
    <>
      <PageHeader crumbs={[{ label: "Progress" }]} />
      <PageBody>
        <PageTitle
          title="Where you're at"
          description="What you have finished, and how your work holds up under a clock."
        />

        {paths.length > 0 && (
          <div className="mb-8">
            <SectionLabel>Paths</SectionLabel>
            <ListGroup>
              {paths.map((p) => (
                <ListRow
                  key={p.name}
                  title={p.title}
                  description={
                    <Bar value={p.total ? (p.done / p.total) * 100 : 0} className="mt-1.5" />
                  }
                  trailing={
                    <span className="font-mono">
                      {p.done}/{p.total}
                    </span>
                  }
                  onClick={() => onOpenPath(p.name)}
                />
              ))}
            </ListGroup>
          </div>
        )}

        <div className="mb-3 flex items-center justify-between">
          <SectionLabel className="mb-0">Timed sessions</SectionLabel>
          {offered.length > 1 && (
            <Tabs value={mode} onValueChange={(v) => setMode(v as Mode)}>
              <TabsList>
                {offered.map((m) => (
                  <TabsTrigger key={m} value={m} className="px-3">
                    {MODE_LABEL[m]}
                  </TabsTrigger>
                ))}
              </TabsList>
            </Tabs>
          )}
        </div>

        {!profile || profile.sessions === 0 ? (
          <div className="rounded-xl border border-dashed p-8 text-center">
            <p className="text-sm">No scored sessions in this mode yet.</p>
            <p className="mx-auto mt-1 max-w-sm text-xs text-muted-foreground">
              Finish a session and score it. Two or three runs in, the pattern in your skill areas
              starts to mean something.
            </p>
          </div>
        ) : (
          <div className="flex flex-col gap-4">
            <div className="grid grid-cols-3 gap-3">
              {[
                { value: profile.sessions, label: "Sessions" },
                { value: Math.round(profile.averageScore), label: "Average score" },
                { value: profile.totalMinutes, label: "Minutes practiced" },
              ].map((s) => (
                <div key={s.label} className="rounded-xl border p-4">
                  <div className="text-2xl font-semibold tabular-nums">{s.value}</div>
                  <div className="text-xs text-muted-foreground">{s.label}</div>
                </div>
              ))}
            </div>

            {(!!profile.strongest?.length || !!profile.weakest?.length) && (
              <div className="grid grid-cols-2 gap-3">
                {[
                  { label: "Holding up", items: profile.strongest, cls: "text-emerald-600 dark:text-emerald-400" },
                  { label: "Keeps slipping", items: profile.weakest, cls: "text-amber-600 dark:text-amber-400" },
                ]
                  .filter((g) => g.items?.length)
                  .map((g) => (
                    <div key={g.label} className="rounded-xl border p-4">
                      <div className={cn("mb-2 text-xs font-medium", g.cls)}>{g.label}</div>
                      <ul className="flex flex-col gap-1 text-sm">
                        {g.items.map((s) => (
                          <li key={s}>{s}</li>
                        ))}
                      </ul>
                    </div>
                  ))}
              </div>
            )}

            <div className="rounded-xl border p-4">
              <SectionLabel>By skill area</SectionLabel>
              <div className="flex flex-col gap-2.5">
                {profile.areas.map((a) => (
                  <div key={a.area} className="flex items-center gap-3 text-sm" title={`${a.samples} scored criteria`}>
                    <span className="w-52 shrink-0 truncate">{a.label}</span>
                    <span className="h-1.5 flex-1 overflow-hidden rounded-full bg-muted">
                      <span
                        className={cn("block h-full rounded-full", tone(a.average))}
                        style={{ width: `${(a.average / 4) * 100}%` }}
                      />
                    </span>
                    <span className="w-8 text-right font-mono text-xs text-muted-foreground">
                      {a.average.toFixed(1)}
                    </span>
                    <span className="w-4 text-muted-foreground">
                      {a.trend > 0.2 ? (
                        <ArrowUp className="size-3.5 text-emerald-500" />
                      ) : a.trend < -0.2 ? (
                        <ArrowDown className="size-3.5 text-destructive" />
                      ) : null}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {forMode.length > 0 && (
          <div className="mt-8">
            <SectionLabel>Past sessions</SectionLabel>
            <ListGroup>
              {forMode.map((s) => (
                <ListRow
                  key={s.id}
                  title={s.problem.title}
                  description={[
                    s.spec.sourceTitle,
                    s.spec.level,
                    `${Math.round(s.elapsedSec / 60)} min`,
                    when(s.startedAt),
                  ]
                    .filter(Boolean)
                    .join(" · ")}
                  trailing={
                    <span className={cn("font-mono text-sm font-semibold", toneText((s.review?.overall ?? 0) / 25))}>
                      {s.review?.overall}
                    </span>
                  }
                />
              ))}
            </ListGroup>
          </div>
        )}
      </PageBody>
    </>
  );
}
