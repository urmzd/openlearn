// Settings: which course pack is open, and which model everything talks to.
//
// A pack is a folder with an openlearn.json manifest. Linking a folder leaves it
// where it is, so a checkout you are editing stays live; importing a zip copies
// it into the app's own directory.

import { useEffect, useState } from "react";
import { AgentConnection } from "@/components/agent-connection";
import { Check, FolderOpen, PackagePlus } from "lucide-react";
import {
  CheckModel,
  GetSettings,
  GetStoreRoot,
  ImportPack,
  ListModels,
  ListPacks,
  OpenPack,
  SaveSettings,
} from "../../wailsjs/go/main/App";
import type { course, llm, packs, store } from "../../wailsjs/go/models";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Field, Picker } from "@/components/choice";
import {
  ErrorNote,
  PageBody,
  PageHeader,
  PageTitle,
  SectionLabel,
} from "@/components/page";
import { isOpenPack } from "@/components/app-sidebar";
import { ListGroup, ListRow } from "@/components/list-row";

const SOURCE: Record<string, string> = {
  builtin: "Built in",
  checkout: "This checkout",
  linked: "Linked folder",
  installed: "Imported",
};

const PROVIDERS = [
  { value: "ollama", label: "Ollama (local)" },
  { value: "anthropic", label: "Anthropic" },
  { value: "openai", label: "OpenAI" },
  { value: "google", label: "Google" },
];

/** Which provider and model the coach and the reviewer use. A
 *  misconfigured model is the one failure that makes the whole app do nothing,
 *  so the check lives next to the choice. */
function ModelSettings() {
  const [settings, setSettings] = useState<store.Settings | null>(null);
  const [models, setModels] = useState<llm.ModelInfo[]>([]);
  const [probe, setProbe] = useState("");
  const [probing, setProbing] = useState(false);
  const [error, setError] = useState("");

  const refreshModels = () => {
    ListModels()
      .then((m) => setModels(m ?? []))
      .catch((e) => setError(String(e)));
  };

  useEffect(() => {
    GetSettings().then(setSettings);
    refreshModels();
  }, []);

  const update = async (patch: Partial<llm.Config>) => {
    if (!settings) return;
    const next = {
      ...settings,
      llm: { ...settings.llm, ...patch },
    } as store.Settings;
    setSettings(next);
    setProbe("");
    await SaveSettings(next).catch((e) => setError(String(e)));
    if (patch.kind) refreshModels();
  };

  const check = async () => {
    setProbing(true);
    setProbe("");
    setError("");
    try {
      setProbe(await CheckModel());
    } catch (e) {
      setError(String(e));
    } finally {
      setProbing(false);
    }
  };

  const kind = settings?.llm.kind ?? "ollama";
  const isOllama = kind === "ollama";

  return (
    <div className="flex flex-col gap-4 rounded-xl border p-4">
      <div className="grid grid-cols-2 gap-4">
        <Field label="Provider">
          <Picker
            value={kind}
            options={PROVIDERS}
            onChange={(v) => update({ kind: v })}
          />
        </Field>
        <Field label="Model">
          {isOllama && models.length > 0 ? (
            <Picker
              value={settings?.llm.model ?? ""}
              placeholder="Choose a model"
              options={models.map((m) => ({
                value: m.name,
                label: m.sizeBytes
                  ? `${m.name} (${(m.sizeBytes / 1e9).toFixed(1)} GB)`
                  : m.name,
              }))}
              onChange={(model) => update({ model })}
            />
          ) : (
            <Input
              value={settings?.llm.model ?? ""}
              placeholder={
                isOllama ? "No models found. Is Ollama running?" : "Model id"
              }
              onChange={(e) => update({ model: e.target.value })}
            />
          )}
        </Field>
      </div>
      {isOllama ? (
        <Field label="Ollama host">
          <Input
            value={settings?.llm.host ?? ""}
            placeholder="http://localhost:11434"
            onChange={(e) => update({ host: e.target.value })}
          />
        </Field>
      ) : (
        <Field label="API key">
          <Input
            type="password"
            value={settings?.llm.apiKey ?? ""}
            placeholder="Stored on this machine"
            onChange={(e) => update({ apiKey: e.target.value })}
          />
        </Field>
      )}
      <div className="flex items-center gap-3">
        <Button
          variant="outline"
          size="sm"
          onClick={check}
          disabled={probing || !settings?.llm.model}
        >
          {probing ? "Checking…" : "Test connection"}
        </Button>
        {probe && (
          <span className="flex items-center gap-1.5 text-xs text-emerald-600 dark:text-emerald-400">
            <Check className="size-3.5" />
            {probe}
          </span>
        )}
      </div>
      {error && <ErrorNote>{error}</ErrorNote>}
    </div>
  );
}

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

  const isOpen = (p: packs.Pack) => isOpenPack(p, course);

  return (
    <>
      <PageHeader crumbs={[{ label: "Settings" }]} />
      <PageBody>
        <PageTitle
          title="Settings"
          description="Your course packs, the model behind timed sessions, and how to connect an agent."
        />

        <div className="mb-2 flex items-end justify-between">
          <SectionLabel className="mb-0">Course packs</SectionLabel>
          <div className="flex gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={() => add(false)}
              disabled={busy}
            >
              <FolderOpen data-icon="inline-start" />
              Open folder
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={() => add(true)}
              disabled={busy}
            >
              <PackagePlus data-icon="inline-start" />
              Import zip
            </Button>
          </div>
        </div>
        <ListGroup>
          {list.map((p) => (
            <ListRow
              key={p.ref}
              active={isOpen(p)}
              title={
                <span className="flex items-center gap-2">
                  {p.title}
                  {isOpen(p) && <Badge variant="secondary">Open</Badge>}
                </span>
              }
              description={p.tagline}
              trailing={`${SOURCE[p.source] ?? p.source}${p.practice ? " · exercises" : ""}`}
              onClick={() => open(p.ref)}
            />
          ))}
          {list.length === 0 && (
            <p className="p-3 text-sm text-muted-foreground">
              No course packs found.
            </p>
          )}
        </ListGroup>
        <p className="mt-2 text-xs text-muted-foreground">
          A pack is any folder with an{" "}
          <span className="font-mono">openlearn.json</span> manifest. A pack
          with exercises runs its own practice CLI on your machine when you
          check an attempt, so only open packs you trust.
        </p>
        {error && (
          <div className="mt-3">
            <ErrorNote>{error}</ErrorNote>
          </div>
        )}

        <SectionLabel className="mt-8">Timed sessions</SectionLabel>
        <p className="mb-2 text-xs text-muted-foreground">
          The model that writes timed exercises, coaches during them, and scores them.
        </p>
        <ModelSettings />

        <SectionLabel className="mt-8">Connect an agent</SectionLabel>
        <AgentConnection />

        {root && (
          <p className="mt-6 text-xs text-muted-foreground">
            Sessions, settings, and imported packs live in{" "}
            <span className="font-mono">{root}</span>.
          </p>
        )}
      </PageBody>
    </>
  );
}
