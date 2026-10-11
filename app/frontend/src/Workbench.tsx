// The workbench: everything outside a live session.
//
// Two columns, each with one job. The sidebar is navigation. The main pane
// shows exactly one view, picked by a route. A session is not a view here: it
// takes the whole window (see App.tsx), because a running clock should not be
// one click away from something else.
//
// There is no assistant in the window. An agent connects from outside, over
// MCP, and sees what is open here through the focus the workbench records.

import { useCallback, useEffect, useState } from "react";
import {
  GetCourse,
  GetDoc,
  GetOutline,
  ListPacks,
  ListPaths,
  OpenPack,
  SetFocus,
} from "../wailsjs/go/main/App";
import type { course, packs } from "../wailsjs/go/models";
import { SidebarInset, SidebarProvider } from "@/components/ui/sidebar";
import { TooltipProvider } from "@/components/ui/tooltip";
import { AppSidebar } from "@/components/app-sidebar";
import type { Route } from "@/routes";
import { DocView } from "@/views/DocView";
import { Exercises } from "@/views/Exercises";
import { Home } from "@/views/Home";
import { PathView } from "@/views/PathView";
import { Progress } from "@/views/Progress";
import { Runner } from "@/views/Runner";
import { SessionSetup } from "@/views/SessionSetup";
import { Settings } from "@/views/Settings";
import { TrackView } from "@/views/TrackView";

interface Props {
  /** Bumped when a session ends, so progress and history reload. */
  epoch: number;
}

export function Workbench({ epoch }: Props) {
  const [info, setInfo] = useState<course.Info | null>(null);
  const [outline, setOutline] = useState<course.Outline | null>(null);
  const [paths, setPaths] = useState<course.PathSummary[]>([]);
  const [packList, setPackList] = useState<packs.Pack[]>([]);
  const [route, setRoute] = useState<Route>({ kind: "home" });
  const [doc, setDoc] = useState<course.Doc | null>(null);
  const [notice, setNotice] = useState("");
  const [practiceEpoch, setPracticeEpoch] = useState(0);

  const loadPaths = useCallback(() => {
    ListPaths()
      .then((p) => setPaths(p ?? []))
      .catch(() => setPaths([]));
  }, []);

  const loadPack = useCallback(async () => {
    const [i, o, all] = await Promise.all([
      GetCourse(),
      GetOutline(),
      ListPacks(),
    ]);
    setInfo(i);
    setOutline(o);
    setPackList(all ?? []);
    loadPaths();
    setDoc(null);
    setRoute({ kind: "home" });
  }, [loadPaths]);

  useEffect(() => {
    loadPack().catch(() => {});
  }, [loadPack]);

  useEffect(loadPaths, [epoch, loadPaths]);

  const openDoc = useCallback((path: string) => {
    setNotice("");
    GetDoc(path)
      .then((d) => {
        setDoc(d);
        setRoute({ kind: "doc", path: d.path });
      })
      .catch(() => setNotice(`${path} is not part of this course pack.`));
  }, []);

  const switchPack = useCallback(
    (ref: string) => {
      OpenPack(ref)
        .then(loadPack)
        .catch((e) => setNotice(String(e)));
    },
    [loadPack],
  );

  // Tell the backend which document is open, so an agent connected over MCP
  // can resolve "this page".
  useEffect(() => {
    SetFocus(route.kind === "doc" ? route.path : "").catch(() => {});
  }, [route]);

  const bumpPractice = useCallback(() => setPracticeEpoch((n) => n + 1), []);

  const track =
    route.kind === "track"
      ? [...(outline?.tracks ?? []), ...(outline?.reference ?? [])].find(
          (t) => t.dir === route.dir,
        )
      : undefined;
  const titles = Object.fromEntries(paths.map((p) => [p.name, p.title]));

  return (
    <TooltipProvider>
      <SidebarProvider className="relative h-full min-h-0">
        <AppSidebar
          info={info}
          outline={outline}
          paths={paths}
          route={route}
          packList={packList}
          onNavigate={setRoute}
          onOpenDoc={openDoc}
          onSwitchPack={switchPack}
        />
        <SidebarInset className="h-full min-h-0 min-w-0 overflow-hidden">
          {route.kind === "home" && (
            <Home
              info={info}
              outline={outline}
              paths={paths}
              onNavigate={setRoute}
              onOpenDoc={openDoc}
            />
          )}
          {route.kind === "doc" && doc && (
            <DocView
              doc={doc}
              outline={outline}
              repo={info?.repo}
              onNavigate={setRoute}
              onOpenDoc={openDoc}
            />
          )}
          {route.kind === "track" && track && (
            <TrackView track={track} onOpenDoc={openDoc} />
          )}
          {route.kind === "path" && (
            <PathView
              name={route.name}
              titles={titles}
              onOpenDoc={openDoc}
              onProgress={loadPaths}
            />
          )}
          {route.kind === "exercises" && (
            <Exercises
              refresh={practiceEpoch}
              onOpen={(ref) => setRoute({ kind: "exercise", ref })}
            />
          )}
          {route.kind === "exercise" && (
            <Runner
              key={`${route.ref.kind}/${route.ref.lang}/${route.ref.name}`}
              exercise={route.ref}
              onBack={() => setRoute({ kind: "exercises" })}
              onChanged={bumpPractice}
            />
          )}
          {route.kind === "session" && (
            <SessionSetup
              modes={info?.modes ?? []}
              source={route.source}
              onClearSource={() => setRoute({ kind: "session", source: null })}
              onOpenSettings={() => setRoute({ kind: "settings" })}
            />
          )}
          {route.kind === "progress" && (
            <Progress
              key={epoch}
              modes={info?.modes ?? []}
              paths={paths.filter((p) => !p.part)}
              onOpenPath={(name) => setRoute({ kind: "path", name })}
            />
          )}
          {route.kind === "settings" && (
            <Settings
              course={info}
              onPackChanged={() => loadPack().catch(() => {})}
            />
          )}

          {notice && (
            <button
              onClick={() => setNotice("")}
              className="absolute bottom-4 left-1/2 -translate-x-1/2 rounded-lg border border-destructive/30 bg-background px-3 py-2 text-sm text-destructive shadow-lg"
            >
              {notice}
            </button>
          )}
        </SidebarInset>
      </SidebarProvider>
    </TooltipProvider>
  );
}
