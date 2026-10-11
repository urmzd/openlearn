// The first screen: what this pack is and the one thing to do next.

import {
  ArrowRight,
  BookOpen,
  Plug,
  SquareTerminal,
  Timer,
} from "lucide-react";
import type { course } from "../../wailsjs/go/models";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { PageBody, PageHeader, SectionLabel } from "@/components/page";
import { ListGroup, ListRow } from "@/components/list-row";
import type { Route } from "@/routes";

interface Props {
  info: course.Info | null;
  outline: course.Outline | null;
  paths: course.PathSummary[];
  onNavigate: (route: Route) => void;
  onOpenDoc: (path: string) => void;
}

export function Home({
  info,
  outline,
  paths,
  onNavigate,
  onOpenDoc,
}: Props) {
  const featured =
    paths.find((p) => p.featured && !p.broken) ??
    paths.find((p) => !p.part && !p.broken);

  return (
    <>
      <PageHeader crumbs={[{ label: "Home" }]} />
      <PageBody>
        <h1 className="text-3xl font-semibold tracking-tight">{info?.title}</h1>
        <p className="mt-2 max-w-xl text-base text-muted-foreground">
          {info?.tagline}
        </p>

        {featured && (
          <div className="mt-8 rounded-xl border p-5">
            <SectionLabel>
              {featured.done > 0 ? "Pick up where you left off" : "Start here"}
            </SectionLabel>
            <div className="flex items-center justify-between gap-6">
              <div className="min-w-0 flex-1">
                <div className="truncate text-lg font-medium">
                  {featured.title}
                </div>
                <div className="mt-3 flex items-center gap-3">
                  <Progress
                    value={
                      featured.total
                        ? (featured.done / featured.total) * 100
                        : 0
                    }
                    className="flex-1"
                  />
                  <span className="font-mono text-xs text-muted-foreground">
                    {featured.done}/{featured.total}
                  </span>
                </div>
              </div>
              <Button
                size="lg"
                onClick={() =>
                  onNavigate({ kind: "path", name: featured.name })
                }
              >
                {featured.done > 0 ? "Continue" : "Start"}
                <ArrowRight data-icon="inline-end" />
              </Button>
            </div>
          </div>
        )}

        {!!outline?.start?.length && (
          <div className="mt-8">
            <SectionLabel>Orientation</SectionLabel>
            <ListGroup>
              {outline.start.map((d) => (
                <ListRow
                  key={d.path}
                  leading={<BookOpen className="size-4" />}
                  title={d.title}
                  onClick={() => onOpenDoc(d.path)}
                />
              ))}
            </ListGroup>
          </div>
        )}

        <div className="mt-8">
          <SectionLabel>Ways to work</SectionLabel>
          <ListGroup>
            {info?.practice && (
              <ListRow
                leading={<SquareTerminal className="size-4" />}
                title="Exercises"
                description="Edit an attempt and run it against its own tests. The exit code is the verdict."
                onClick={() => onNavigate({ kind: "exercises" })}
              />
            )}
            <ListRow
              leading={<Timer className="size-4" />}
              title="Timed session"
              description="An exercise written from a chapter, a coach that watches, and a score against a hidden rubric."
              onClick={() => onNavigate({ kind: "session", source: null })}
            />
            <ListRow
              leading={<Plug className="size-4" />}
              title="Bring your own agent"
              description="Connect any MCP client and it can read the course, see your progress, and check your exercises."
              onClick={() => onNavigate({ kind: "connect" })}
            />
          </ListGroup>
        </div>
      </PageBody>
    </>
  );
}
