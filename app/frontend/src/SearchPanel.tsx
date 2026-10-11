// Full-text search across the open pack. The backend scans on every query, so
// the only thing to manage here is not asking on every keystroke.

import { useEffect, useState } from "react";
import { Search } from "../wailsjs/go/main/App";
import type { course } from "../wailsjs/go/models";

interface Props {
  activeDoc: string;
  onOpenDoc: (path: string) => void;
}

export function SearchPanel({ activeDoc, onOpenDoc }: Props) {
  const [query, setQuery] = useState("");
  const [hits, setHits] = useState<course.Hit[] | null>(null);

  useEffect(() => {
    if (!query.trim()) {
      setHits(null);
      return;
    }
    let live = true;
    const t = setTimeout(() => {
      Search(query)
        .then((h) => live && setHits(h ?? []))
        .catch(() => live && setHits([]));
    }, 180);
    return () => {
      live = false;
      clearTimeout(t);
    };
  }, [query]);

  return (
    <div className="side-scroll">
      <section className="side-section">
        <span className="label">Search the course</span>
        <input
          autoFocus
          value={query}
          placeholder="attention, raft, 注文…"
          onChange={(e) => setQuery(e.target.value)}
        />
      </section>
      {hits && (
        <section className="side-section">
          <span className="label">
            {hits.length === 0 ? "No documents match" : `${hits.length} documents`}
          </span>
          {hits.map((h) => (
            <button
              key={h.path}
              className={`hit ${activeDoc === h.path ? "active" : ""}`}
              onClick={() => onOpenDoc(h.path)}
              title={h.path}
            >
              <span className="hit-title">
                {h.title}
                <span className="faint mono hit-count">{h.matches}</span>
              </span>
              <span className="hit-snippet faint">{h.snippet}</span>
            </button>
          ))}
        </section>
      )}
    </div>
  );
}
