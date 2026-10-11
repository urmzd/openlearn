// The debrief. This is the payload of the whole app: the score, what earned it,
// and what to do next. Scores are shown with their evidence attached, because a
// number with no reason behind it is not something you can act on.

import type { domain } from "../wailsjs/go/models";
import "./review.css";

const SCORE_WORDS = ["absent", "named only", "partial", "solid", "excellent"];

function scoreClass(score: number): string {
  if (score >= 3) return "good";
  if (score === 2) return "mid";
  return "low";
}

interface Props {
  problem: domain.Problem | null;
  review: domain.Review;
  elapsedSec: number;
  onDone: () => void;
  onViewProgress: () => void;
}

export function Review({ problem, review, elapsedSec, onDone, onViewProgress }: Props) {
  const minutes = Math.round(elapsedSec / 60);

  return (
    <div className="review">
      <div className="review-inner">
        <header className="review-header">
          <div>
            <span className="label">Debrief</span>
            <h1>{problem?.title}</h1>
            <p className="faint">
              {minutes} minute{minutes === 1 ? "" : "s"} · targeted at {problem?.level}
            </p>
          </div>
          <div className="score-block">
            <div className={`score-ring ${scoreClass(Math.round(review.overall / 25))}`}>
              <span className="score-num">{review.overall}</span>
            </div>
            <span className="faint score-caption">reads as {review.verdict}</span>
          </div>
        </header>

        <section className="review-card summary">
          <p>{review.summary}</p>
        </section>

        <section className="review-card">
          <span className="label">Scored against the hidden rubric</span>
          <div className="criteria">
            {review.scores.map((s) => (
              <div key={s.criterionId} className="criterion">
                <div className="criterion-head">
                  <span className="criterion-title">{s.title}</span>
                  <span className={`criterion-score ${scoreClass(s.score)}`}>
                    {s.score}/4 <span className="faint">{SCORE_WORDS[s.score]}</span>
                  </span>
                </div>
                <div className="criterion-bar">
                  <div
                    className={`criterion-fill ${scoreClass(s.score)}`}
                    style={{ width: `${(s.score / 4) * 100}%` }}
                  />
                </div>
                <p className="criterion-evidence">{s.evidence}</p>
              </div>
            ))}
          </div>
        </section>

        <div className="review-cols">
          {!!review.strengths?.length && (
            <section className="review-card">
              <span className="label good-text">What worked</span>
              <ul className="review-list">
                {review.strengths.map((s, i) => (
                  <li key={i}>{s}</li>
                ))}
              </ul>
            </section>
          )}
          {!!review.gaps?.length && (
            <section className="review-card">
              <span className="label warn-text">What was missing</span>
              <ul className="review-list">
                {review.gaps.map((g, i) => (
                  <li key={i}>{g}</li>
                ))}
              </ul>
            </section>
          )}
        </div>

        {!!review.nextSteps?.length && (
          <section className="review-card">
            <span className="label">Practice next</span>
            <ol className="review-list numbered">
              {review.nextSteps.map((n, i) => (
                <li key={i}>{n}</li>
              ))}
            </ol>
          </section>
        )}

        {!!review.missedOutline?.length && (
          <section className="review-card">
            <span className="label">A strong answer would also have reached</span>
            <ul className="review-list faint-list">
              {review.missedOutline.map((m, i) => (
                <li key={i}>{m}</li>
              ))}
            </ul>
          </section>
        )}

        <div className="review-actions">
          <button className="primary big" onClick={onDone}>
            New session
          </button>
          <button onClick={onViewProgress}>See progress over time</button>
        </div>
      </div>
    </div>
  );
}
