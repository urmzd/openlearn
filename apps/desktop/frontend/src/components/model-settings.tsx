// Which provider and model timed sessions use: it writes the exercise, coaches
// during it, and scores it. A misconfigured model is the one failure that makes
// a session do nothing, so the check lives next to the choice.

import { useEffect, useState } from "react";
import { Check } from "lucide-react";
import { CheckModel, GetSettings, ListModels, SaveSettings } from "../../wailsjs/go/main/App";
import type { llm, store } from "../../wailsjs/go/models";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Field, Picker } from "@/components/choice";
import { ErrorNote } from "@/components/page";

const PROVIDERS = [
  { value: "ollama", label: "Ollama (local)" },
  { value: "anthropic", label: "Anthropic" },
  { value: "openai", label: "OpenAI" },
  { value: "google", label: "Google" },
];

/** Which provider and model the coach and the reviewer use. A
 *  misconfigured model is the one failure that makes the whole app do nothing,
 *  so the check lives next to the choice. */
export function ModelSettings({ onChange }: { onChange?: (model: string) => void }) {
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
    onChange?.(next.llm.model ?? "");
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
