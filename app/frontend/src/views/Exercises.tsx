// The pack's exercises, grouped the way its CLI groups them.

import { useEffect, useState } from "react";
import { CircleDashed, CircleDot } from "lucide-react";
import { ListExercises } from "../../wailsjs/go/main/App";
import type { practice } from "../../wailsjs/go/models";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Skeleton } from "@/components/ui/skeleton";
import { ErrorNote, PageBody, PageHeader, PageTitle, SectionLabel } from "@/components/page";
import { ListGroup, ListRow } from "@/components/list-row";
import type { ExerciseRef } from "@/routes";

const KIND_BLURB: Record<string, string> = {
  predict: "Write the exact output of a snippet before you run it.",
  build: "Implement it from scratch against its own assertions.",
  reattempt: "Something you solved before, handed back with the solution removed.",
};

interface Props {
  /** Bumped when an attempt starts or resets, so the marks follow. */
  refresh: number;
  onOpen: (ref: ExerciseRef) => void;
}

export function Exercises({ refresh, onOpen }: Props) {
  const [groups, setGroups] = useState<practice.Group[] | null>(null);
  const [error, setError] = useState("");
  const [kind, setKind] = useState("");

  useEffect(() => {
    ListExercises()
      .then((g) => {
        setGroups(g ?? []);
        setError("");
      })
      .catch((e) => setError(String(e)));
  }, [refresh]);

  const kinds = [...new Set((groups ?? []).map((g) => g.kind))];
  const active = kinds.includes(kind) ? kind : (kinds[0] ?? "");
  const shown = (groups ?? []).filter((g) => g.kind === active);

  return (
    <>
      <PageHeader crumbs={[{ label: "Exercises" }]} />
      <PageBody>
        <PageTitle
          title="Exercises"
          description="Each one is cloned into your scratchpad, edited here, and checked by the pack's own CLI."
        />
        {error && <ErrorNote>{error}</ErrorNote>}
        {!groups && !error && <Skeleton className="h-40 w-full" />}
        {groups && groups.length === 0 && (
          <p className="text-sm text-muted-foreground">
            This pack has no exercises that can run here. Exercises need the pack opened from a
            folder on disk, with its practice CLI.
          </p>
        )}
        {kinds.length > 0 && (
          <>
            <Tabs value={active} onValueChange={(v) => setKind(String(v))}>
              <TabsList>
                {kinds.map((k) => (
                  <TabsTrigger key={k} value={k} className="px-3 capitalize">
                    {k}
                  </TabsTrigger>
                ))}
              </TabsList>
            </Tabs>
            <p className="mt-3 mb-6 text-sm text-muted-foreground">{KIND_BLURB[active]}</p>
            {shown.map((g) => (
              <div key={g.lang} className="mb-6">
                <SectionLabel>
                  {g.lang}
                  <span className="ml-2 font-mono font-normal">
                    {g.exercises.filter((e) => e.started).length}/{g.exercises.length} started
                  </span>
                </SectionLabel>
                <ListGroup>
                  {g.exercises.map((e) => (
                    <ListRow
                      key={e.name}
                      leading={
                        e.started ? (
                          <CircleDot className="size-4 text-foreground" />
                        ) : (
                          <CircleDashed className="size-4" />
                        )
                      }
                      title={<span className="font-mono font-normal">{e.name}</span>}
                      trailing={e.started ? "started" : undefined}
                      onClick={() => onOpen({ kind: g.kind, lang: g.lang, name: e.name })}
                    />
                  ))}
                </ListGroup>
              </div>
            ))}
          </>
        )}
      </PageBody>
    </>
  );
}
