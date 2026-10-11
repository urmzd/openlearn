// Settings: which course pack is open, and which model everything talks to.
//
// A pack is a folder with an openlearn.json manifest. Linking a folder leaves it
// where it is, so a checkout you are editing stays live; importing a zip copies
// it into the app's own directory.

import { useEffect, useState } from "react";
import { GetStoreRoot, ImportPack, ListPacks, OpenPack } from "../wailsjs/go/main/App";
import type { course, packs } from "../wailsjs/go/models";
import { ModelCard } from "./ModelCard";

const SOURCE: Record<string, string> = {
  builtin: "built in",
  checkout: "this checkout",
  linked: "linked folder",
  installed: "imported",
};

interface Props {
  course: course.Info | null;
  /** Called after the open pack changes. */
  onPackChanged: () => void;
}

export function Settings({ course, onPackChanged }: Props) {
  const [list, setList] = useState<packs.Pack[]>([]);
  const [root, setRoot] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  const refresh = () => ListPacks().then((p) => setList(p ?? []));
  useEffect(() => {
    refresh();
    GetStoreRoot().then(setRoot);
  }, []);

  const open = async (ref: string) => {
    setError("");
    try {
      await OpenPack(ref);
      onPackChanged();
    } catch (e) {
      setError(String(e));
    }
  };

  const add = async (archive: boolean) => {
    setBusy(true);
    setError("");
    try {
      const ref = await ImportPack(archive);
      await refresh();
      if (ref) await open(ref);
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  };

  // The open pack is matched by where it was read from: a directory for packs
  // on disk, the id for built-in ones.
  const isOpen = (p: packs.Pack) =>
    !!course && p.id === course.id && (course.builtIn ? p.source === "builtin" : p.ref === course.root);

  return (
    <div className="setup">
      <div className="setup-inner">
        <header className="setup-header">
          <div>
            <h1>Settings</h1>
            <p className="muted">Your course packs and the model that coaches, grades, and answers.</p>
          </div>
        </header>

        <section className="card">
          <div className="spread">
            <span className="label" style={{ margin: 0 }}>
              Course packs
            </span>
            <span className="row">
              <button onClick={() => add(false)} disabled={busy}>
                Open folder…
              </button>
              <button onClick={() => add(true)} disabled={busy}>
                Import zip…
              </button>
            </span>
          </div>
          <div className="pack-list">
            {list.map((p) => (
              <button
                key={p.ref}
                className={`pack ${isOpen(p) ? "active" : ""}`}
                onClick={() => open(p.ref)}
              >
                <span className="pack-main">
                  <span className="pack-title">
                    {p.title}
                    {isOpen(p) && <span className="tag">open</span>}
                  </span>
                  <span className="faint pack-tagline">{p.tagline}</span>
                </span>
                <span className="pack-meta faint">
                  {SOURCE[p.source] ?? p.source}
                  {p.practice ? " · exercises" : ""}
                </span>
              </button>
            ))}
            {list.length === 0 && <p className="faint">No course packs found.</p>}
          </div>
          <p className="faint pack-note">
            A pack is any folder with an <span className="mono">openlearn.json</span> manifest. A
            pack with exercises runs its own practice CLI on your machine when you check an
            attempt, so only open packs you trust.
          </p>
          {error && <div className="error-box">{error}</div>}
        </section>

        <ModelCard />

        {root && (
          <p className="faint" style={{ fontSize: 12 }}>
            Sessions, settings, and imported packs live in <span className="mono">{root}</span>.
          </p>
        )}
      </div>
    </div>
  );
}
