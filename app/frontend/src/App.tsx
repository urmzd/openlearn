// App shell. It holds one piece of state: the reduction of the AG-UI event
// stream. Whether a session owns the window is derived from the phase inside
// that state rather than from navigation tracked separately, so the backend and
// the UI can never disagree about what is happening.

import { useEffect, useReducer, useState } from "react";
import { GetStatus } from "../wailsjs/go/main/App";
import { Session } from "./Session";
import { Review } from "./Review";
import { Workbench } from "./Workbench";
import {
  emptyState,
  reduce,
  subscribe,
  type AguiEvent,
  type SessionState,
} from "./agui";
import "./index.css";
import "./theme.css";

export default function App() {
  const [state, dispatch] = useReducer(
    (s: SessionState, e: AguiEvent) => reduce(s, e),
    emptyState,
  );
  // Counts finished sessions, so the workbench reloads progress when one ends.
  const [epoch, setEpoch] = useState(0);

  useEffect(() => {
    const off = subscribe(dispatch);
    // One read at startup, so a session already running in the backend (after a
    // frontend reload in dev) is picked up without waiting for the next event.
    GetStatus()
      .then((status) =>
        dispatch({
          type: "STATE_SNAPSHOT",
          at: new Date().toISOString(),
          data: status,
        }),
      )
      .catch(() => {});
    return off;
  }, []);

  const reset = () => {
    dispatch({
      type: "STATE_SNAPSHOT",
      at: new Date().toISOString(),
      data: null,
    });
    setEpoch((n) => n + 1);
  };

  const phase = state.status?.phase ?? "idle";
  const inSession =
    phase === "active" || phase === "paused" || phase === "reviewing";
  const inReview = phase === "done" && !!state.status?.review;

  // A session takes the whole window: a clock is running and nothing else
  // should be one click away. The workbench stays mounted underneath so the
  // document and the scroll position are still there afterwards.
  return (
    <>
      <div style={{ display: inSession || inReview ? "none" : "contents" }}>
        <Workbench epoch={epoch} />
      </div>
      {inReview && state.status?.review && (
        <div className="legacy">
          <Review
            problem={state.status.problem ?? null}
            review={state.status.review}
            elapsedSec={state.status.elapsedSec}
            onDone={reset}
            onViewProgress={reset}
          />
        </div>
      )}
      {inSession && (
        <div className="legacy">
          <Session state={state} onFinished={() => {}} onAbandoned={reset} />
        </div>
      )}
    </>
  );
}
