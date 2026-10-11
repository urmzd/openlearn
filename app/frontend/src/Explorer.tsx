// The explorer: role paths on top, because they answer "what do I read next",
// and the full track tree below for when you know what you are looking for.

import { useState } from "react";
import type { course } from "../wailsjs/go/models";

interface Props {
  outline: course.Outline | null;
  paths: course.PathSummary[];
  activeDoc: string;
  activePath: string;
  onOpenDoc: (path: string) => void;
  onOpenPath: (name: string) => void;
}

function Tree({
  tracks,
  activeDoc,
  onOpenDoc,
}: {
  tracks: course.Track[];
  activeDoc: string;
  onOpenDoc: (path: string) => void;
}) {
  // Expanded state is keyed by directory. A track holding the open document
  // starts expanded, so the tree always shows where you are.
  const [open, setOpen] = useState<Record<string, boolean>>({});
  const isOpen = (dir: string) => open[dir] ?? activeDoc.startsWith(`${dir}/`);
  const toggle = (dir: string) => setOpen((o) => ({ ...o, [dir]: !isOpen(dir) }));

  const doc = (d: course.DocRef, depth: number, label?: string) => (
    <button
      key={d.path}
      className={`tree-row doc ${activeDoc === d.path ? "active" : ""}`}
      style={{ paddingLeft: 10 + depth * 14 }}
      onClick={() => onOpenDoc(d.path)}
      title={d.path}
    >
      {label ?? d.title}
    </button>
  );

  return (
    <>
      {tracks.map((t) => (
        <div key={t.dir}>
          <button className="tree-row dir" onClick={() => toggle(t.dir)}>
            <span className="twist">{isOpen(t.dir) ? "▾" : "▸"}</span>
            {t.title}
          </button>
          {isOpen(t.dir) && (
            <>
              {t.docs.map((d) => doc(d, 1))}
              {t.topics.map((topic) =>
                topic.docs.length === 1 ? (
                  doc(topic.docs[0], 1, topic.title)
                ) : (
                  <div key={topic.dir}>
                    <button
                      className="tree-row dir"
                      style={{ paddingLeft: 24 }}
                      onClick={() => toggle(topic.dir)}
                    >
                      <span className="twist">{isOpen(topic.dir) ? "▾" : "▸"}</span>
                      {topic.title}
                    </button>
                    {isOpen(topic.dir) && topic.docs.map((d) => doc(d, 3))}
                  </div>
                ),
              )}
            </>
          )}
        </div>
      ))}
    </>
  );
}

export function Explorer({ outline, paths, activeDoc, activePath, onOpenDoc, onOpenPath }: Props) {
  return (
    <div className="side-scroll">
      {paths.length > 0 && (
        <section className="side-section">
          <span className="label">Paths</span>
          {paths.map((p) => (
            <button
              key={p.name}
              className={`path-row ${activePath === p.name ? "active" : ""}`}
              onClick={() => onOpenPath(p.name)}
              title={p.broken || p.name}
            >
              <span className="path-title">
                {p.title}
                {p.featured && <span className="tag">start here</span>}
              </span>
              {p.broken ? (
                <span className="warn-text path-count">broken</span>
              ) : (
                <>
                  <span className="path-bar">
                    <span
                      className="path-fill"
                      style={{ width: `${p.total ? (p.done / p.total) * 100 : 0}%` }}
                    />
                  </span>
                  <span className="path-count mono faint">
                    {p.done}/{p.total}
                  </span>
                </>
              )}
            </button>
          ))}
        </section>
      )}

      {!!outline?.start?.length && (
        <section className="side-section">
          <span className="label">Start</span>
          {outline.start.map((d) => (
            <button
              key={d.path}
              className={`tree-row doc ${activeDoc === d.path ? "active" : ""}`}
              onClick={() => onOpenDoc(d.path)}
            >
              {d.title}
            </button>
          ))}
        </section>
      )}

      {!!outline?.tracks?.length && (
        <section className="side-section">
          <span className="label">Tracks</span>
          <Tree tracks={outline.tracks} activeDoc={activeDoc} onOpenDoc={onOpenDoc} />
        </section>
      )}

      {!!outline?.reference?.length && (
        <section className="side-section">
          <span className="label">Reference</span>
          <Tree tracks={outline.reference} activeDoc={activeDoc} onOpenDoc={onOpenDoc} />
        </section>
      )}
    </div>
  );
}
