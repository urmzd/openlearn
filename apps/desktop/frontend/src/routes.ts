// What the main pane can show. One thing at a time, named here so the sidebar
// and the views agree on the vocabulary.

export interface ExerciseRef {
  kind: string;
  lang: string;
  name: string;
}

/** The chapter a timed session is drawn from. */
export interface Source {
  path: string;
  title: string;
}

export type Route =
  | { kind: "home" }
  | { kind: "doc"; path: string }
  | { kind: "track"; dir: string }
  | { kind: "path"; name: string }
  | { kind: "exercises" }
  | { kind: "exercise"; ref: ExerciseRef }
  | { kind: "session"; source: Source | null }
  | { kind: "progress" }
  | { kind: "courses" }
  | { kind: "connect" };
