// App shell. It holds one piece of state: the reduction of the AG-UI event
// stream. Which screen shows is derived from the phase inside that state rather
// than from navigation tracked separately, so the backend and the UI can never
// disagree about what is happening.

import { useEffect, useReducer, useState } from "react";
import { GetStatus } from "../wailsjs/go/main/App";
import { Setup } from "./Setup";
import { Session } from "./Session";
import { Review } from "./Review";
import { Progress } from "./Progress";
import { emptyState, reduce, subscribe, type AguiEvent, type SessionState } from "./agui";
import "./theme.css";

export default function App() {
  const [state, dispatch] = useReducer((s: SessionState, e: AguiEvent) => reduce(s, e), emptyState);
  const [showProgress, setShowProgress] = useState(false);

  useEffect(() => {
    const off = subscribe(dispatch);
    // One read at startup, so a session already running in the backend (after a
    // frontend reload in dev) is picked up without waiting for the next event.
    GetStatus()
      .then((status) =>
        dispatch({ type: "STATE_SNAPSHOT", at: new Date().toISOString(), data: status }),
      )
      .catch(() => {});
    return off;
  }, []);

  const reset = () =>
    dispatch({ type: "STATE_SNAPSHOT", at: new Date().toISOString(), data: null });

  if (showProgress) {
    return <Progress onBack={() => setShowProgress(false)} />;
  }

  const phase = state.status?.phase ?? "idle";

  if (phase === "done" && state.status?.review) {
    return (
      <Review
        problem={state.status.problem ?? null}
        review={state.status.review}
        elapsedSec={state.status.elapsedSec}
        onDone={reset}
        onViewProgress={() => setShowProgress(true)}
      />
    );
  }

  if (phase === "active" || phase === "paused" || phase === "reviewing") {
    return <Session state={state} onFinished={() => {}} onAbandoned={reset} />;
  }

  return <Setup onStarted={() => {}} onOpenHistory={() => setShowProgress(true)} />;
}
