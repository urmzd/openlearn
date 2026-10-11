// Where you're at: the cross-session view.
//
// A single session tells you how one exercise went. This tells you which skills
// hold up under a clock and which keep slipping, which is the thing you cannot
// see from inside any one session.

import { useEffect, useState } from "react";
import { GetProfile, ListSessions } from "../wailsjs/go/main/App";
import type { domain } from "../wailsjs/go/models";
import type { Mode } from "./types";
import "./progress.css";

function when(iso: string): string {
  const d = new Date(iso);
  const days = Math.floor((Date.now() - d.getTime()) / 86400000);
  if (days === 0) return "today";
  if (days === 1) return "yesterday";
  if (days < 30) return `${days} days ago`;
  return d.toLocaleDateString();
}

function barClass(avg: number): string {
  if (avg >= 3) return "good";
  if (avg >= 2) return "mid";
  return "low";
}

interface Props {
  onBack: () => void;
}

export function Progress({ onBack }: Props) {
  const [mode, setMode] = useState<Mode>("system_design");
  const [profile, setProfile] = useState<domain.Profile | null>(null);
  const [sessions, setSessions] = useState<domain.Session[]>([]);

  useEffect(() => {
    GetProfile(mode).then(setProfile).catch(() => setProfile(null));
    ListSessions().then(setSessions).catch(() => setSessions([]));
  }, [mode]);

  const forMode = sessions.filter((s) => s.spec.mode === mode && s.review);

  return (
    <div className="progress">
      <div className="progress-inner">
        <header className="progress-header">
          <div>
            <h1>Where you're at</h1>
            <p className="muted">Averaged across every scored session, by skill area.</p>
          </div>
          <button onClick={onBack}>Back</button>
        </header>

        <div className="pill-row" style={{ marginBottom: 4 }}>
          <button
            className={`pill ${mode === "system_design" ? "active" : ""}`}
            onClick={() => setMode("system_design")}
          >
            System design
          </button>
          <button
            className={`pill ${mode === "coding" ? "active" : ""}`}
            onClick={() => setMode("coding")}
          >
            Coding
          </button>
        </div>

        {!profile || profile.sessions === 0 ? (
          <div className="empty-state">
            <p>No scored sessions in this mode yet.</p>
            <p className="faint">
              Finish a session and score it. Two or three runs in, the pattern in your skill areas
              starts to mean something.
            </p>
          </div>
        ) : (
          <>
            <section className="stat-row">
              <div className="stat">
                <span className="stat-value">{profile.sessions}</span>
                <span className="stat-label faint">sessions</span>
              </div>
              <div className="stat">
                <span className="stat-value">{Math.round(profile.averageScore)}</span>
                <span className="stat-label faint">avg score</span>
              </div>
              <div className="stat">
                <span className="stat-value">{profile.totalMinutes}</span>
                <span className="stat-label faint">minutes practiced</span>
              </div>
            </section>

            {(!!profile.strongest?.length || !!profile.weakest?.length) && (
              <div className="review-cols">
                {!!profile.strongest?.length && (
                  <section className="review-card">
                    <span className="label good-text">Holding up</span>
                    <ul className="review-list">
                      {profile.strongest.map((s) => (
                        <li key={s}>{s}</li>
                      ))}
                    </ul>
                  </section>
                )}
                {!!profile.weakest?.length && (
                  <section className="review-card">
                    <span className="label warn-text">Keeps slipping</span>
                    <ul className="review-list">
                      {profile.weakest.map((s) => (
                        <li key={s}>{s}</li>
                      ))}
                    </ul>
                  </section>
                )}
              </div>
            )}

            <section className="review-card">
              <span className="label">By skill area</span>
              <div className="areas">
                {profile.areas.map((a) => (
                  <div key={a.area} className="area-row">
                    <span className="area-label">{a.label}</span>
                    <div className="area-bar">
                      <div
                        className={`area-fill ${barClass(a.average)}`}
                        style={{ width: `${(a.average / 4) * 100}%` }}
                      />
                    </div>
                    <span className="area-value mono">{a.average.toFixed(1)}</span>
                    <span
                      className={`area-trend ${a.trend > 0.2 ? "up" : a.trend < -0.2 ? "down" : ""}`}
                      title={`${a.samples} scored criteria`}
                    >
                      {a.trend > 0.2 ? "↑" : a.trend < -0.2 ? "↓" : "·"}
                    </span>
                  </div>
                ))}
              </div>
            </section>
          </>
        )}

        {forMode.length > 0 && (
          <section className="review-card">
            <span className="label">Past sessions</span>
            <div className="history">
              {forMode.map((s) => (
                <div key={s.id} className="history-row">
                  <span className="history-title">{s.problem.title}</span>
                  <span className="faint history-meta">
                    {s.spec.level} · {Math.round(s.elapsedSec / 60)}m · {when(s.startedAt)}
                  </span>
                  <span className={`history-score ${barClass((s.review?.overall ?? 0) / 25)}`}>
                    {s.review?.overall}
                  </span>
                </div>
              ))}
            </div>
          </section>
        )}
      </div>
    </div>
  );
}
