// Courses: the packs this app can open, and how to add one.
//
// A course is a folder with an openlearn.json manifest. Opening a folder leaves
// it where it is, so a checkout you are editing stays live; importing a zip
// copies it into the app's own directory.

import { useEffect, useState } from "react";
import { FolderOpen, PackagePlus } from "lucide-react";
import { GetStoreRoot, ImportPack, ListPacks } from "../../wailsjs/go/main/App";
import type { course, packs } from "../../wailsjs/go/models";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { isOpenPack } from "@/components/app-sidebar";
import { ErrorNote, PageBody, PageHeader, PageTitle } from "@/components/page";
import { ListGroup, ListRow } from "@/components/list-row";

const SOURCE: Record<string, string> = {
  builtin: "Built in",
  checkout: "This checkout",
  linked: "Linked folder",
  installed: "Imported",
};

interface Props {
  course: course.Info | null;
  onSwitch: (ref: string) => void;
}

export function Courses({ course, onSwitch }: Props) {
  const [list, setList] = useState<packs.Pack[]>([]);
  const [root, setRoot] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  const refresh = () => ListPacks().then((p) => setList(p ?? []));
  useEffect(() => {
    refresh();
    GetStoreRoot().then(setRoot);
  }, [course]);

  const add = async (archive: boolean) => {
    setBusy(true);
    setError("");
    try {
      const ref = await ImportPack(archive);
      await refresh();
      if (ref) onSwitch(ref);
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <>
      <PageHeader crumbs={[{ label: "Courses" }]} />
      <PageBody>
        <PageTitle
          title="Courses"
          description="Pick the course to work in, or add another."
          action={
            <div className="flex gap-2">
              <Button variant="outline" onClick={() => add(false)} disabled={busy}>
                <FolderOpen data-icon="inline-start" />
                Open folder
              </Button>
              <Button variant="outline" onClick={() => add(true)} disabled={busy}>
                <PackagePlus data-icon="inline-start" />
                Import zip
              </Button>
            </div>
          }
        />
        <ListGroup>
          {list.map((p) => (
            <ListRow
              key={p.ref}
              active={isOpenPack(p, course)}
              title={
                <span className="flex items-center gap-2">
                  {p.title}
                  {isOpenPack(p, course) && <Badge variant="secondary">Open</Badge>}
                </span>
              }
              description={p.tagline}
              trailing={`${SOURCE[p.source] ?? p.source}${p.practice ? " · exercises" : ""}`}
              onClick={() => onSwitch(p.ref)}
            />
          ))}
          {list.length === 0 && <p className="p-3 text-sm text-muted-foreground">No courses found.</p>}
        </ListGroup>
        {error && (
          <div className="mt-3">
            <ErrorNote>{error}</ErrorNote>
          </div>
        )}
        <p className="mt-4 text-xs text-muted-foreground">
          A course is any folder with an <span className="font-mono">openlearn.json</span> manifest.
          Checking an exercise runs compilers and the course's own run scripts on your machine, so
          only open courses you trust.
        </p>
        {root && (
          <p className="mt-2 text-xs text-muted-foreground">
            Imported courses, sessions, and progress for built-in courses live in{" "}
            <span className="font-mono">{root}</span>.
          </p>
        )}
      </PageBody>
    </>
  );
}
