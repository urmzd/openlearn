// The workbench: everything outside a live session.
//
// It is laid out like an editor because learning from a course is the same
// motion as working in a codebase: a tree to find things in, a pane to read and
// write in, a panel for what running it said, and someone to ask alongside.
// An activity bar picks what the sidebar shows; the main pane shows one thing
// at a time; the professor docks on the right and follows the open document.

import { useCallback, useEffect, useState } from "react";
import { GetCourse, GetDoc, GetOutline, ListPaths } from "../wailsjs/go/main/App";
import type { course } from "../wailsjs/go/models";
import { Explorer } from "./Explorer";
import { SearchPanel } from "./SearchPanel";
import { PathView } from "./PathView";
import { PracticeList, PracticeRunner, type ExerciseRef } from "./Practice";
import { Professor } from "./Professor";
import { Progress } from "./Progress";
import { Settings } from "./Settings";
import { Setup, type Source } from "./Setup";
import { Markdown } from "./Markdown";
import "./workbench.css";

type Activity = "explorer" | "search" | "practice";

type Main =
  | { kind: "home" }
  | { kind: "doc"; path: string }
  | { kind: "path"; name: string }
  | { kind: "exercise"; ref: ExerciseRef }
  | { kind: "session"; source: Source | null }
  | { kind: "progress" }
  | { kind: "settings" };

const ACTIVITIES: { id: Activity; icon: string; title: string }[] = [
  { id: "explorer", icon: "☰", title: "Explorer: paths and tracks" },
  { id: "search", icon: "⌕", title: "Search the course" },
  { id: "practice", icon: "▷", title: "Exercises" },
];

interface Props {
  /** Bumped when a session ends, so progress and history reload. */
  epoch: number;
}

export function Workbench({ epoch }: Props) {
  const [info, setInfo] = useState<course.Info | null>(null);
  const [outline, setOutline] = useState<course.Outline | null>(null);
  const [paths, setPaths] = useState<course.PathSummary[]>([]);
  const [activity, setActivity] = useState<Activity>("explorer");
  const [sidebar, setSidebar] = useState(true);
  const [main, setMain] = useState<Main>({ kind: "home" });
  const [doc, setDoc] = useState<course.Doc | null>(null);
  const [docError, setDocError] = useState("");
  const [professor, setProfessor] = useState(false);
  const [practiceEpoch, setPracticeEpoch] = useState(0);

  const loadPaths = useCallback(() => {
    ListPaths()
      .then((p) => setPaths(p ?? []))
      .catch(() => setPaths([]));
  }, []);

  const loadPack = useCallback(async () => {
    const [i, o] = await Promise.all([GetCourse(), GetOutline()]);
    setInfo(i);
    setOutline(o);
    loadPaths();
    setDoc(null);
    setMain({ kind: "home" });
  }, [loadPaths]);

  useEffect(() => {
    loadPack().catch(() => {});
  }, [loadPack]);

  useEffect(loadPaths, [epoch, loadPaths]);

  const openDoc = useCallback((path: string) => {
    setDocError("");
    GetDoc(path)
      .then((d) => {
        setDoc(d);
        setMain({ kind: "doc", path: d.path });
      })
      .catch(() => setDocError(`${path} is not part of this course pack.`));
  }, []);

  const pick = (a: Activity) => {
    if (a === activity && sidebar) return setSidebar(false);
    setActivity(a);
    setSidebar(true);
  };

  const activeDoc = main.kind === "doc" ? main.path : "";
  const featured = paths.find((p) => p.featured && !p.broken);

  return (
    <div className="workbench">
      <nav className="activity">
        <div className="activity-group">
          {ACTIVITIES.filter((a) => a.id !== "practice" || info?.practice).map((a) => (
            <button
              key={a.id}
              className={`activity-btn ${activity === a.id && sidebar ? "active" : ""}`}
              onClick={() => pick(a.id)}
              title={a.title}
            >
              {a.icon}
            </button>
          ))}
          <span className="activity-rule" />
          <button
            className={`activity-btn ${main.kind === "session" ? "active" : ""}`}
            onClick={() => setMain({ kind: "session", source: null })}
            title="Timed session with a coach"
          >
            ◷
          </button>
          <button
            className={`activity-btn ${main.kind === "progress" ? "active" : ""}`}
            onClick={() => setMain({ kind: "progress" })}
            title="Progress"
          >
            ◔
          </button>
        </div>
        <div className="activity-group">
          <button
            className={`activity-btn ${professor ? "active" : ""}`}
            onClick={() => setProfessor(!professor)}
            title="Ask the professor"
          >
            ✦
          </button>
          <button
            className={`activity-btn ${main.kind === "settings" ? "active" : ""}`}
            onClick={() => setMain({ kind: "settings" })}
            title="Settings: course packs and model"
          >
            ⚙
          </button>
        </div>
      </nav>

      {sidebar && (
        <aside className="sidebar">
          <header className="sidebar-header">
            <span className="sidebar-title">{info?.title ?? ""}</span>
            <button className="link-btn" onClick={() => setMain({ kind: "settings" })} title="Switch course pack">
              switch
            </button>
          </header>
          {activity === "explorer" && (
            <Explorer
              outline={outline}
              paths={paths}
              activeDoc={activeDoc}
              activePath={main.kind === "path" ? main.name : ""}
              onOpenDoc={openDoc}
              onOpenPath={(name) => setMain({ kind: "path", name })}
            />
          )}
          {activity === "search" && <SearchPanel activeDoc={activeDoc} onOpenDoc={openDoc} />}
          {activity === "practice" && (
            <PracticeList
              active={main.kind === "exercise" ? main.ref : null}
              refresh={practiceEpoch}
              onOpen={(ref) => setMain({ kind: "exercise", ref })}
            />
          )}
        </aside>
      )}

      <main className="mainpane">
        {docError && (
          <div className="toast-error" onClick={() => setDocError("")}>
            {docError}
          </div>
        )}

        {main.kind === "home" && (
          <div className="pane-scroll">
            <div className="home">
              <h1>{info?.title}</h1>
              <p className="muted home-tagline">{info?.tagline}</p>
              <div className="home-actions">
                {featured && (
                  <button className="primary big" onClick={() => setMain({ kind: "path", name: featured.name })}>
                    {featured.done > 0 ? "Continue" : "Start"}: {featured.title}
                    <span className="home-count">
                      {featured.done}/{featured.total}
                    </span>
                  </button>
                )}
                {outline?.start?.[0] && (
                  <button className="big" onClick={() => openDoc(outline.start[0].path)}>
                    {outline.start[0].title}
                  </button>
                )}
              </div>
              <div className="home-grid">
                <div>
                  <span className="label">Read</span>
                  <p className="faint">
                    {outline?.tracks?.length ?? 0} tracks and {paths.length} paths in the explorer.
                    Search covers every document.
                  </p>
                </div>
                <div>
                  <span className="label">Practice</span>
                  <p className="faint">
                    {info?.practice
                      ? "Exercises run through the pack's own CLI: the exit code is the verdict."
                      : "Start a timed session from any chapter; the exercise is written from it."}
                  </p>
                </div>
                <div>
                  <span className="label">Ask</span>
                  <p className="faint">The professor answers questions about the page you have open.</p>
                </div>
              </div>
            </div>
          </div>
        )}

        {main.kind === "doc" && doc && (
          <>
            <header className="doc-bar">
              <span className="doc-path mono faint" title={doc.path}>
                {doc.path}
              </span>
              <span className="row">
                <button onClick={() => setProfessor(true)} title="Ask about this page">
                  ✦ Ask
                </button>
                <button
                  className="primary"
                  onClick={() => setMain({ kind: "session", source: { path: doc.path, title: doc.title } })}
                  title="A timed exercise written from this chapter, graded against a hidden rubric"
                >
                  Practice this
                </button>
              </span>
            </header>
            <div className="pane-scroll" key={doc.path}>
              <div className="doc-body">
                <Markdown path={doc.path} markdown={doc.markdown} repo={info?.repo} onOpenDoc={openDoc} />
              </div>
            </div>
          </>
        )}

        {main.kind === "path" && <PathView name={main.name} onOpenDoc={openDoc} onProgress={loadPaths} />}

        {main.kind === "exercise" && (
          <PracticeRunner
            key={`${main.ref.kind}/${main.ref.lang}/${main.ref.name}`}
            exercise={main.ref}
            onChanged={() => setPracticeEpoch((n) => n + 1)}
          />
        )}

        {main.kind === "session" && (
          <Setup
            modes={info?.modes ?? []}
            source={main.source}
            onClearSource={() => setMain({ kind: "session", source: null })}
            onOpenSettings={() => setMain({ kind: "settings" })}
          />
        )}

        {main.kind === "progress" && (
          <Progress
            key={epoch}
            modes={info?.modes ?? []}
            paths={paths}
            onOpenPath={(name) => setMain({ kind: "path", name })}
          />
        )}

        {main.kind === "settings" && <Settings course={info} onPackChanged={() => loadPack().catch(() => {})} />}
      </main>

      {professor && (
        <Professor
          docPath={activeDoc}
          docTitle={main.kind === "doc" ? (doc?.title ?? "") : ""}
          repo={info?.repo}
          onOpenDoc={openDoc}
          onOpenSettings={() => setMain({ kind: "settings" })}
          onClose={() => setProfessor(false)}
        />
      )}
    </div>
  );
}
