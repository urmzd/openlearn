// The reader: one document, with the two things you do to a chapter beside it.

import { Sparkles, Timer } from "lucide-react";
import type { course } from "../../wailsjs/go/models";
import { Button } from "@/components/ui/button";
import { PageBody, PageHeader, type Crumb } from "@/components/page";
import { Markdown } from "@/Markdown";
import type { Route } from "@/routes";

interface Props {
  doc: course.Doc;
  outline: course.Outline | null;
  repo?: string;
  onNavigate: (route: Route) => void;
  onOpenDoc: (path: string) => void;
  onAsk: () => void;
}

/** Where a document sits in the outline, as breadcrumbs. */
function crumbsFor(doc: course.Doc, outline: course.Outline | null, onNavigate: (r: Route) => void, onOpenDoc: (p: string) => void): Crumb[] {
  for (const t of [...(outline?.tracks ?? []), ...(outline?.reference ?? [])]) {
    if (!doc.path.startsWith(`${t.dir}/`)) continue;
    const crumbs: Crumb[] = [{ label: t.title, onClick: () => onNavigate({ kind: "track", dir: t.dir }) }];
    const topic = t.topics.find((x) => doc.path.startsWith(`${x.dir}/`));
    if (topic && topic.docs[0]?.path !== doc.path) {
      crumbs.push({ label: topic.title, onClick: () => onOpenDoc(topic.docs[0].path) });
    }
    crumbs.push({ label: doc.title });
    return crumbs;
  }
  return [{ label: doc.title }];
}

export function DocView({ doc, outline, repo, onNavigate, onOpenDoc, onAsk }: Props) {
  return (
    <>
      <PageHeader
        crumbs={crumbsFor(doc, outline, onNavigate, onOpenDoc)}
        actions={
          <>
            <Button variant="ghost" size="sm" onClick={onAsk}>
              <Sparkles data-icon="inline-start" />
              Ask
            </Button>
            <Button
              size="sm"
              onClick={() => onNavigate({ kind: "session", source: { path: doc.path, title: doc.title } })}
              title="A timed exercise written from this chapter, graded against a hidden rubric"
            >
              <Timer data-icon="inline-start" />
              Practice this
            </Button>
          </>
        }
      />
      <PageBody width="prose" key={doc.path}>
        <Markdown path={doc.path} markdown={doc.markdown} repo={repo} onOpenDoc={onOpenDoc} />
      </PageBody>
    </>
  );
}
