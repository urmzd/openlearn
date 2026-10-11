// Wails' TypeScript generator emits structs but not named string types, so the
// Go enums arrive as bare `string`. These unions restore the constraint on the
// frontend side. They must stay in step with internal/domain and internal/llm.

export type Mode = "system_design" | "coding";

export type Level = "junior" | "mid" | "senior" | "staff";

export type ProviderKind = "ollama" | "openai" | "anthropic" | "google";

export type Phase =
  | "idle"
  | "generating"
  | "active"
  | "paused"
  | "reviewing"
  | "done"
  | "failed";

export type EventKind = "hint" | "probe" | "curveball" | "praise" | "pacing" | "system";

export type Severity = "info" | "warn" | "critical";
