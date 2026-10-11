// The AG-UI event stream, as the frontend sees it.
//
// The backend publishes every event on one Wails channel in emission order.
// This module turns that stream into two things the UI renders: a state object
// patched by STATE_SNAPSHOT / STATE_DELTA, and a message list built from
// TEXT_MESSAGE_START / CONTENT / END. Nothing here polls; the UI only ever
// reacts to what the agent decided to send.

import { EventsOn } from "../wailsjs/runtime/runtime";
import type { domain } from "../wailsjs/go/models";

export const CHANNEL = "agui";

export type EventType =
  | "RUN_STARTED"
  | "RUN_FINISHED"
  | "RUN_ERROR"
  | "STEP_STARTED"
  | "STEP_FINISHED"
  | "TEXT_MESSAGE_START"
  | "TEXT_MESSAGE_CONTENT"
  | "TEXT_MESSAGE_END"
  | "STATE_SNAPSHOT"
  | "STATE_DELTA"
  | "CUSTOM";

export interface AguiEvent {
  type: EventType;
  at: string;
  runId?: string;
  name?: string;
  data?: unknown;
}

export interface TextMessageStart {
  messageId: string;
  role: "assistant" | "system";
  kind?: string;
  severity?: string;
  title?: string;
  elapsedSec: number;
}

export interface TextMessageContent {
  messageId: string;
  delta: string;
}

export interface TextMessageEnd {
  messageId: string;
  areas?: string[];
}

export interface StateDelta {
  patch: Record<string, unknown>;
}

export interface RunError {
  runId: string;
  message: string;
  code?: string;
}

/** A message as the UI holds it, complete or still streaming. */
export interface Message {
  id: string;
  role: "assistant" | "system";
  kind: string;
  severity: string;
  title: string;
  body: string;
  elapsedSec: number;
  /** False until TEXT_MESSAGE_END arrives; drives the typing caret. */
  done: boolean;
  areas: string[];
}

/** Everything the session view renders. */
export interface SessionState {
  status: domain.Status | null;
  messages: Message[];
  /** Steps currently in flight, e.g. "generate_problem". */
  activeSteps: string[];
  runError: string;
}

export const emptyState: SessionState = {
  status: null,
  messages: [],
  activeSteps: [],
  runError: "",
};

/**
 * reduce applies one event to the state, returning a new object. Keeping this
 * a pure function means the same stream always produces the same UI, and it
 * can be tested without a backend.
 */
export function reduce(state: SessionState, event: AguiEvent): SessionState {
  switch (event.type) {
    case "RUN_STARTED":
      // A new run clears the previous run's transcript.
      return { ...emptyState, status: state.status };

    case "RUN_ERROR": {
      const d = event.data as RunError;
      return { ...state, runError: d?.message ?? "Something went wrong." };
    }

    case "STEP_STARTED":
      if (!event.name || state.activeSteps.includes(event.name)) return state;
      return { ...state, activeSteps: [...state.activeSteps, event.name] };

    case "STEP_FINISHED":
      if (!event.name) return state;
      return { ...state, activeSteps: state.activeSteps.filter((s) => s !== event.name) };

    case "STATE_SNAPSHOT": {
      const status = event.data as domain.Status | null;
      // A snapshot carries the authoritative transcript, so use it to rebuild
      // the feed. Without this, any message emitted while the window was
      // reloading is gone for good: the feed is otherwise assembled purely
      // from live events, and nothing replays them.
      return { ...state, status, messages: rehydrate(state.messages, status) };
    }

    case "STATE_DELTA": {
      const d = event.data as StateDelta;
      if (!state.status || !d?.patch) return state;
      return { ...state, status: { ...state.status, ...d.patch } as domain.Status };
    }

    case "TEXT_MESSAGE_START": {
      const d = event.data as TextMessageStart;
      const message: Message = {
        id: d.messageId,
        role: d.role,
        kind: d.kind ?? "system",
        severity: d.severity ?? "info",
        title: d.title ?? "",
        body: "",
        elapsedSec: d.elapsedSec,
        done: false,
        areas: [],
      };
      return { ...state, messages: [...state.messages, message] };
    }

    case "TEXT_MESSAGE_CONTENT": {
      const d = event.data as TextMessageContent;
      return {
        ...state,
        messages: state.messages.map((m) =>
          m.id === d.messageId ? { ...m, body: m.body + d.delta } : m,
        ),
      };
    }

    case "TEXT_MESSAGE_END": {
      const d = event.data as TextMessageEnd;
      return {
        ...state,
        messages: state.messages.map((m) =>
          m.id === d.messageId ? { ...m, done: true, areas: d.areas ?? [] } : m,
        ),
      };
    }

    default:
      return state;
  }
}

/**
 * rehydrate merges the transcript from a state snapshot into the live message
 * list. A message still streaming keeps whatever has arrived so far, because
 * the transcript only gains an entry once the backend has finished writing it;
 * overwriting a half-streamed bubble with nothing would make it flicker.
 */
function rehydrate(current: Message[], status: domain.Status | null): Message[] {
  const events = status?.events ?? [];
  if (events.length === 0) return current;

  const live = new Map(current.map((m) => [m.id, m]));
  const merged: Message[] = events.map((e) => {
    const existing = live.get(e.id);
    if (existing && !existing.done) return existing;
    return {
      id: e.id,
      role:
        e.kind === "hint" || e.kind === "probe" || e.kind === "praise"
          ? "assistant"
          : "system",
      kind: e.kind,
      severity: e.severity,
      title: e.title,
      body: e.body,
      elapsedSec: e.elapsedSec,
      done: true,
      areas: e.areas ?? [],
    } as Message;
  });

  // A message that opened but has not reached the transcript yet is still in
  // flight; keep it on the end so its caret survives the snapshot.
  const seen = new Set(merged.map((m) => m.id));
  for (const m of current) {
    if (!seen.has(m.id)) merged.push(m);
  }
  return merged;
}

/** subscribe wires the Wails channel to a handler and returns an unsubscriber. */
export function subscribe(onEvent: (e: AguiEvent) => void): () => void {
  return EventsOn(CHANNEL, (e: AguiEvent) => onEvent(e));
}
