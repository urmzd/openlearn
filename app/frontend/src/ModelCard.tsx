// Model configuration: which provider and model the coach, the reviewer, and
// the professor all use. A misconfigured model is the one failure that makes
// the whole app do nothing, so the check lives next to the choice.

import { useEffect, useState } from "react";
import { CheckModel, GetSettings, ListModels, SaveSettings } from "../wailsjs/go/main/App";
import type { llm, store } from "../wailsjs/go/models";
import type { ProviderKind } from "./types";

export function ModelCard() {
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
    const next = { ...settings, llm: { ...settings.llm, ...patch } } as store.Settings;
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
    <section className="card">
      <span className="label">Model</span>
      <div className="field-grid">
        <div>
          <span className="label">Provider</span>
          <select value={kind} onChange={(e) => update({ kind: e.target.value as ProviderKind })}>
            <option value="ollama">Ollama (local)</option>
            <option value="anthropic">Anthropic</option>
            <option value="openai">OpenAI</option>
            <option value="google">Google</option>
          </select>
        </div>
        <div>
          <span className="label">Model</span>
          {isOllama && models.length > 0 ? (
            <select value={settings?.llm.model ?? ""} onChange={(e) => update({ model: e.target.value })}>
              <option value="">Choose a model…</option>
              {models.map((m) => (
                <option key={m.name} value={m.name}>
                  {m.name}
                  {m.sizeBytes ? ` (${(m.sizeBytes / 1e9).toFixed(1)} GB)` : ""}
                </option>
              ))}
            </select>
          ) : (
            <input
              value={settings?.llm.model ?? ""}
              placeholder={isOllama ? "no models found - is ollama running?" : "model id"}
              onChange={(e) => update({ model: e.target.value })}
            />
          )}
        </div>
      </div>

      {isOllama ? (
        <div style={{ marginTop: 12 }}>
          <span className="label">Ollama host</span>
          <input
            value={settings?.llm.host ?? ""}
            placeholder="http://localhost:11434"
            onChange={(e) => update({ host: e.target.value })}
          />
        </div>
      ) : (
        <div style={{ marginTop: 12 }}>
          <span className="label">API key</span>
          <input
            type="password"
            value={settings?.llm.apiKey ?? ""}
            placeholder="stored locally"
            onChange={(e) => update({ apiKey: e.target.value })}
          />
        </div>
      )}

      <div className="row" style={{ marginTop: 12 }}>
        <button onClick={check} disabled={probing || !settings?.llm.model}>
          {probing ? "Checking…" : "Test connection"}
        </button>
        {probe && <span className="probe-ok">{probe}</span>}
      </div>
      {error && (
        <div className="error-box" style={{ marginTop: 10 }}>
          {error}
        </div>
      )}
    </section>
  );
}
