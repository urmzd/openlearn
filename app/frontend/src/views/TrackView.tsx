// One track: its topics in reading order. A topic with a single document opens
// it; a topic with several lists them under itself.

import { useState } from "react";
import { ChevronDown, ChevronRight, FileText } from "lucide-react";
import type { course } from "../../wailsjs/go/models";
import { PageBody, PageHeader, PageTitle, SectionLabel } from "@/components/page";
import { ListGroup, ListRow } from "@/components/list-row";

interface Props {
  track: course.Track;
  onOpenDoc: (path: string) => void;
}

export function TrackView({ track, onOpenDoc }: Props) {
  const [open, setOpen] = useState<Record<string, boolean>>({});
  const readme = track.docs.find((d) => d.path === `${track.dir}/README.md`);
  const loose = track.docs.filter((d) => d !== readme);

  return (
    <>
      <PageHeader crumbs={[{ label: track.title }]} />
      <PageBody>
        <PageTitle
          title={track.title}
          description={`${track.topics.length} topics${readme ? "" : ""}`}
        />

        {readme && (
          <ListGroup className="mb-6">
            <ListRow
              leading={<FileText className="size-4" />}
              title="About this track"
              description={readme.title}
              onClick={() => onOpenDoc(readme.path)}
            />
          </ListGroup>
        )}

        {track.topics.length > 0 && (
          <>
            <SectionLabel>Topics</SectionLabel>
            <ListGroup>
              {track.topics.map((t, i) => {
                const single = t.docs.length === 1;
                const isOpen = !!open[t.dir];
                return (
                  <div key={t.dir}>
                    <ListRow
                      leading={
                        <span className="inline-block w-5 text-right font-mono text-xs">{i + 1}</span>
                      }
                      title={t.title}
                      trailing={
                        single ? undefined : (
                          <span className="flex items-center gap-1">
                            {t.docs.length}
                            {isOpen ? <ChevronDown className="size-3.5" /> : <ChevronRight className="size-3.5" />}
                          </span>
                        )
                      }
                      onClick={() =>
                        single ? onOpenDoc(t.docs[0].path) : setOpen((o) => ({ ...o, [t.dir]: !isOpen }))
                      }
                    />
                    {!single && isOpen && (
                      <div className="mb-1 ml-11 flex flex-col gap-0.5 border-l pl-2">
                        {t.docs.map((d) => (
                          <ListRow
                            key={d.path}
                            title={<span className="font-normal">{d.title}</span>}
                            className="py-1.5"
                            onClick={() => onOpenDoc(d.path)}
                          />
                        ))}
                      </div>
                    )}
                  </div>
                );
              })}
            </ListGroup>
          </>
        )}

        {loose.length > 0 && (
          <>
            <SectionLabel className="mt-6">Also in this track</SectionLabel>
            <ListGroup>
              {loose.map((d) => (
                <ListRow
                  key={d.path}
                  leading={<FileText className="size-4" />}
                  title={d.title}
                  onClick={() => onOpenDoc(d.path)}
                />
              ))}
            </ListGroup>
          </>
        )}
      </PageBody>
    </>
  );
}
