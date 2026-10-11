// One role path: its stages, what "done" means for each, and which you have
// finished. Progress is the pack's, not the app's: a stage marked here shows as
// done in `ss learn`, and one marked there shows here.

import { useCallback, useEffect, useState } from "react";
import { ArrowRight, Check, ShieldCheck } from "lucide-react";
import { GetPath, MarkStage } from "../../wailsjs/go/main/App";
import type { course } from "../../wailsjs/go/models";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Skeleton } from "@/components/ui/skeleton";
import { ErrorNote, PageBody, PageHeader, PageTitle, SectionLabel } from "@/components/page";
import { cn } from "@/lib/utils";

interface Props {
  name: string;
  /** Titles of every path, for labelling the parts a path includes. */
  titles: Record<string, string>;
  onOpenDoc: (path: string) => void;
  /** Called after a mark changes, so counts elsewhere follow. */
  onProgress: () => void;
}

const short = (file: string) => file.replace(/\/README\.md$/, "").replace(/\.md$/, "");

export function PathView({ name, titles, onOpenDoc, onProgress }: Props) {
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

  const title = path?.title ?? titles[name] ?? name;
  const done = path?.stages.filter((s) => s.done).length ?? 0;
  const total = path?.stages.length ?? 0;
  const next = path?.stages.find((s) => !s.done);
  let owner = "";

  return (
    <>
      <PageHeader crumbs={[{ label: "Paths" }, { label: title }]} />
      <PageBody>
        {error && <ErrorNote>{error}</ErrorNote>}
        {!path && !error && (
          <div className="flex flex-col gap-3">
            <Skeleton className="h-8 w-2/3" />
            <Skeleton className="h-20 w-full" />
            <Skeleton className="h-20 w-full" />
          </div>
        )}
        {path && (
          <>
            <PageTitle
              title={path.title}
              description={
                <button className="underline-offset-4 hover:underline" onClick={() => onOpenDoc(path.readme)}>
                  About this path
                </button>
              }
              action={
                next && (
                  <Button onClick={() => onOpenDoc(next.files[0])} className="max-w-72">
                    <span className="truncate">Continue: {next.title}</span>
                    <ArrowRight data-icon="inline-end" />
                  </Button>
                )
              }
            />
            <div className="mb-8 flex items-center gap-3">
              <Progress value={total ? (done / total) * 100 : 0} className="flex-1" />
              <span className="font-mono text-xs text-muted-foreground">
                {done}/{total} stages
              </span>
            </div>

            <div className="flex flex-col gap-2">
              {path.stages.map((s) => {
                const heading = s.owner !== owner && s.owner !== name ? s.owner : "";
                owner = s.owner;
                return (
                  <div key={s.key}>
                    {heading && <SectionLabel className="mt-6">{titles[heading] ?? heading}</SectionLabel>}
                    <div
                      className={cn(
                        "flex gap-3 rounded-xl border p-4",
                        next?.key === s.key && "border-foreground/30 bg-muted/40",
                      )}
                    >
                      <button
                        onClick={() => mark(s)}
                        title={s.done ? "Mark as not finished" : "Mark as finished"}
                        className={cn(
                          "mt-0.5 flex size-5 shrink-0 items-center justify-center rounded-full border transition-colors",
                          s.done
                            ? "border-emerald-500 bg-emerald-500 text-background"
                            : "border-muted-foreground/40 hover:border-foreground",
                        )}
                      >
                        {s.done && <Check className="size-3" strokeWidth={3} />}
                      </button>
                      <div className="min-w-0 flex-1">
                        <div className={cn("text-sm font-medium", s.done && "text-muted-foreground")}>
                          {s.title}
                        </div>
                        <p className="mt-1 text-sm text-muted-foreground">
                          <span className="text-foreground/70">Done when </span>
                          {s.doneWhen.charAt(0).toLowerCase() + s.doneWhen.slice(1)}
                        </p>
                        {s.check && (
                          <p className="mt-1.5 flex items-center gap-1.5 text-xs text-muted-foreground">
                            <ShieldCheck className="size-3.5" />
                            Verified by <span className="font-mono">{s.check}</span>
                          </p>
                        )}
                        <div className="mt-3 flex flex-wrap gap-1.5">
                          {s.files.map((f) => (
                            <Button
                              key={f}
                              variant="outline"
                              size="xs"
                              className="font-mono font-normal"
                              onClick={() => onOpenDoc(f)}
                              title={f}
                            >
                              {short(f)}
                            </Button>
                          ))}
                        </div>
                        {notice?.key === s.key && (
                          <pre className="mt-3 rounded-lg border border-amber-500/30 bg-amber-500/10 p-2.5 font-mono text-xs whitespace-pre-wrap text-amber-600 dark:text-amber-400">
                            {notice.text}
                          </pre>
                        )}
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </>
        )}
      </PageBody>
    </>
  );
}
