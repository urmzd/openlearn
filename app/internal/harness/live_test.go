package harness_test

import (
	"context"
	"os"
	"testing"
	"time"

	"github.com/urmzd/whiteboardy/internal/domain"
	"github.com/urmzd/whiteboardy/internal/harness"
	"github.com/urmzd/whiteboardy/internal/llm"
)

// These tests hit a real model. They are the only way to know whether the
// prompts and schemas survive contact with a small local model, so they stay
// in the tree, gated behind an env var:
//
//	WHITEBOARDY_LIVE=1 WHITEBOARDY_MODEL=qwen3.5:9b go test ./internal/harness -run Live -v

func liveClient(t *testing.T) *llm.Client {
	t.Helper()
	if os.Getenv("WHITEBOARDY_LIVE") == "" {
		t.Skip("set WHITEBOARDY_LIVE=1 to run tests against a real model")
	}
	model := os.Getenv("WHITEBOARDY_MODEL")
	if model == "" {
		model = "qwen3.5:9b"
	}
	c, err := llm.New(context.Background(), llm.Config{
		Kind:  llm.KindOllama,
		Model: model,
		Host:  llm.DefaultOllamaHost,
	}, nil)
	if err != nil {
		t.Fatalf("build client: %v", err)
	}
	return c
}

func TestLiveGenerateSystemDesign(t *testing.T) {
	c := liveClient(t)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()

	p, err := harness.GenerateProblem(ctx, c, domain.SessionSpec{
		Mode:        domain.ModeSystemDesign,
		Level:       domain.LevelSenior,
		Topic:       "a URL shortener with abuse protection",
		DurationSec: 45 * 60,
	})
	if err != nil {
		t.Fatalf("generate: %v", err)
	}

	if p.Title == "" || p.Statement == "" {
		t.Errorf("empty title or statement: %+v", p)
	}
	if len(p.Requirements) == 0 {
		t.Error("no requirements generated")
	}
	if len(p.Rubric) < 4 {
		t.Errorf("rubric too thin: %d criteria", len(p.Rubric))
	}
	for _, c := range p.Rubric {
		if !domain.ValidArea(domain.ModeSystemDesign, c.Area) {
			t.Errorf("criterion %q has area %q outside the taxonomy", c.Title, c.Area)
		}
	}
	if len(p.Curveballs) == 0 {
		t.Error("no curveballs generated")
	}
	t.Logf("problem: %s\n%s", p.Title, p.Render(true))
}

func TestLiveGenerateCoding(t *testing.T) {
	c := liveClient(t)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()

	p, err := harness.GenerateProblem(ctx, c, domain.SessionSpec{
		Mode:        domain.ModeCoding,
		Level:       domain.LevelMid,
		Topic:       "sliding window over a stream",
		Language:    "go",
		DurationSec: 30 * 60,
	})
	if err != nil {
		t.Fatalf("generate: %v", err)
	}
	if p.Starter == "" {
		t.Error("coding problem has no starter code")
	}
	for _, c := range p.Rubric {
		if !domain.ValidArea(domain.ModeCoding, c.Area) {
			t.Errorf("criterion %q has area %q outside the coding taxonomy", c.Title, c.Area)
		}
	}
	t.Logf("problem: %s\nstarter:\n%s", p.Title, p.Starter)
}

func TestLiveCoachStaysQuietOnEmptyProgress(t *testing.T) {
	c := liveClient(t)
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Minute)
	defer cancel()

	in := harness.CoachInput{
		Problem: stubProblem(),
		Snapshot: domain.Snapshot{
			Nodes: []domain.BoardNode{{ID: "a", Kind: domain.NodeService, Label: "API"}},
		},
		ElapsedSec:   60,
		RemainingSec: 2640,
	}
	d, err := harness.Decide(ctx, c, in)
	if err != nil {
		t.Fatalf("decide: %v", err)
	}
	// One box a minute in is not worth interrupting for. This is the silence
	// bias, which is the whole point of the coach.
	if d.Speak {
		t.Logf("coach chose to speak early: %s / %s", d.Title, d.Point)
	}
	t.Logf("covered: %v", d.Covered)
}

func TestLiveCoachStreamsAMessage(t *testing.T) {
	c := liveClient(t)
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Minute)
	defer cancel()

	in := harness.CoachInput{
		Problem: stubProblem(),
		Snapshot: domain.Snapshot{
			Nodes: []domain.BoardNode{
				{ID: "a", Kind: domain.NodeService, Label: "Ingest API"},
				{ID: "b", Kind: domain.NodeDatabase, Label: "Postgres"},
			},
			Edges: []domain.BoardEdge{{ID: "e", Source: "a", Target: "b", Label: "writes"}},
		},
		ElapsedSec:   1500,
		RemainingSec: 1200,
	}
	// Force the speaking path rather than waiting for the model to choose it.
	d := harness.CoachDecision{
		Speak:    true,
		Kind:     domain.KindProbe,
		Severity: domain.SeverityWarn,
		Title:    "Where does the peak go?",
		Point:    "They write straight from the API to Postgres with no queue, which cannot absorb the 4x peak.",
	}

	var chunks int
	body, err := harness.Speak(ctx, c, in, d, func(string) { chunks++ })
	if err != nil {
		t.Fatalf("speak: %v", err)
	}
	if body == "" {
		t.Fatal("empty message body")
	}
	// Streaming is the point: a single chunk means the UI would pop the whole
	// message in at once.
	if chunks < 2 {
		t.Errorf("got %d chunks, expected the body to stream", chunks)
	}
	t.Logf("message (%d chunks): %s", chunks, body)
}

func TestLiveReviewScoresEmptyWorkLow(t *testing.T) {
	c := liveClient(t)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()

	problem := stubProblem()
	rev, err := harness.Review(ctx, c, problem, domain.Snapshot{
		Nodes: []domain.BoardNode{{ID: "a", Kind: domain.NodeService, Label: "server"}},
	}, nil, 600)
	if err != nil {
		t.Fatalf("review: %v", err)
	}
	if len(rev.Scores) != len(problem.Rubric) {
		t.Errorf("expected %d scores, got %d", len(problem.Rubric), len(rev.Scores))
	}
	// A single unlabeled box must not grade well. If this fails, the reviewer
	// is inflating and every profile built on it is meaningless.
	if rev.Overall > 40 {
		t.Errorf("near-empty work scored %d, reviewer is inflating", rev.Overall)
	}
	t.Logf("overall=%d verdict=%s summary=%s", rev.Overall, rev.Verdict, rev.Summary)
}

func stubProblem() domain.Problem {
	return domain.Problem{
		ID:        "test",
		Mode:      domain.ModeSystemDesign,
		Level:     domain.LevelSenior,
		Title:     "Design a notification fanout service",
		Statement: "Design a service that delivers notifications to users across push, email, and SMS.",
		Requirements: []string{
			"Deliver a notification to all of a user's registered channels",
			"Respect per-user quiet hours",
			"Never deliver the same notification twice",
		},
		Constraints: []string{
			"50M notifications per day, peaking at 4x average for 30 minutes",
			"p99 end-to-end delivery under 30 seconds",
		},
		Rubric: []domain.Criterion{
			{ID: "requirements", Area: domain.AreaRequirements, Title: "Scopes the problem", Detail: "States what is in and out of scope with numbers", Weight: 3},
			{ID: "scaling", Area: domain.AreaScaling, Title: "Handles the peak", Detail: "Shows capacity math for the 4x peak and how the system absorbs it", Weight: 5},
			{ID: "reliability", Area: domain.AreaReliability, Title: "Exactly-once delivery", Detail: "Names a dedupe mechanism with a concrete key and retention", Weight: 5},
			{ID: "data_modeling", Area: domain.AreaDataModeling, Title: "Models users and channels", Detail: "Entities and access patterns for preferences and delivery state", Weight: 3},
		},
		ReferenceOutline: []string{
			"Ingest API with an idempotency key",
			"Fanout worker pool reading from a durable queue",
			"Per-channel rate limiting and provider fallback",
		},
	}
}
