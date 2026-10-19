package loadgen

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"net/http"
	"sort"
	"strings"
	"sync"
	"time"
)

// Clock is the time seam: the real clock in the main, a fake
// one in tests. Its method set is a subset of the course testkit's
// clock.Clock, so a *clock.Fake satisfies it.
type Clock interface {
	Now() time.Time
	After(d time.Duration) <-chan time.Time
}

type realClock struct{}

func (realClock) Now() time.Time                         { return time.Now() }
func (realClock) After(d time.Duration) <-chan time.Time { return time.After(d) }

// RealClock is the wall clock.
var RealClock Clock = realClock{}

// Config is one run.
type Config struct {
	// Target is the base URL without /v1, for example http://127.0.0.1:30080.
	Target string
	// Model, Prompts (request i uses Prompts[i % len]), and MaxTokens shape
	// each POST /v1/chat/completions request (stream: true, temperature 0).
	Model     string
	Prompts   []string
	MaxTokens int
	// APIKey, when set, is sent as "Authorization: Bearer <APIKey>".
	APIKey string
	// Schedule gives the arrival gaps; the first arrival is at its first gap.
	Schedule Schedule
	// Duration ends the arrivals: no request is scheduled at or after
	// start + Duration. MaxRequests > 0 also caps the count. In-flight
	// requests always finish (or fail) before Run returns.
	Duration    time.Duration
	MaxRequests int
	// SLO bounds per request, keyed like the report's slo object:
	// "ttft_ms_p95": 500 bounds each request's TTFT at 500 ms (the suffix
	// after the metric names the percentile the SLO is stated at; goodput
	// counts requests within every bound).
	SLO   map[string]float64
	RunID string
	Seed  uint64
	// Clock defaults to RealClock; Client to a client with no timeout.
	Clock  Clock
	Client *http.Client
}

// EventKind is what one SSE event carried.
type EventKind int

const (
	// Content is a chunk with non-empty delta content (one or more tokens).
	Content EventKind = iota
	// Usage is the final usage chunk (choices: []).
	Usage
	// Done is "data: [DONE]".
	Done
)

// Event is one SSE event of a streamed response, stamped on arrival.
type Event struct {
	At     time.Time
	Kind   EventKind
	Tokens int // Content: 1 per chunk; Usage: completion_tokens
}

// Sample is one request as the client saw it. Start is its INTENDED send
// time from the schedule, not the moment it was sent: a request delayed by
// a stalled client or server is charged for the delay (no coordinated
// omission).
type Sample struct {
	Start  time.Time
	TTFT   time.Duration   // Start to the first content event
	E2E    time.Duration   // Start to [DONE]
	ITL    []time.Duration // between consecutive content events
	Tokens int             // usage.completion_tokens if sent, else content events
	Err    error
}

// TPOT is (E2E - TTFT) / (Tokens - 1); ok is false below 2 tokens.
func (s Sample) TPOT() (time.Duration, bool) {
	if s.Err != nil || s.Tokens < 2 {
		return 0, false
	}
	return (s.E2E - s.TTFT) / time.Duration(s.Tokens-1), true
}

// ReadStream reads an SSE body (openapi SSE framing) until [DONE], calling
// now() once per data event as soon as it is read. A body that ends before
// [DONE], an `event: error`, or a data payload with an "error" member is an
// error; ": ping" comments and blank lines are skipped.
func ReadStream(r io.Reader, now func() time.Time) ([]Event, error) {
	br := bufio.NewReader(r)
	var out []Event
	errEvent := false
	for {
		line, err := br.ReadString('\n')
		if len(line) > 0 {
			line = strings.TrimRight(line, "\r\n")
			switch {
			case line == "" || strings.HasPrefix(line, ":"):
			case strings.HasPrefix(line, "event:"):
				errEvent = strings.TrimSpace(line[len("event:"):]) == "error"
			case strings.HasPrefix(line, "data:"):
				at := now()
				data := strings.TrimSpace(line[len("data:"):])
				if data == "[DONE]" {
					return append(out, Event{At: at, Kind: Done}), nil
				}
				if errEvent {
					return out, fmt.Errorf("stream error event: %s", data)
				}
				ev, ok, perr := parseChunk(data)
				if perr != nil {
					return out, perr
				}
				if ok {
					ev.At = at
					out = append(out, ev)
				}
			}
		}
		if err != nil {
			if errors.Is(err, io.EOF) {
				return out, errors.New("stream ended before [DONE]")
			}
			return out, err
		}
	}
}

// parseChunk classifies one data payload: content (delta.content or the
// completions text non-empty), usage (choices empty, usage present), or
// nothing to record.
func parseChunk(data string) (Event, bool, error) {
	var c struct {
		Error   json.RawMessage `json:"error"`
		Choices []struct {
			Text  string `json:"text"`
			Delta struct {
				Content   string            `json:"content"`
				ToolCalls []json.RawMessage `json:"tool_calls"`
			} `json:"delta"`
		} `json:"choices"`
		Usage *struct {
			CompletionTokens int `json:"completion_tokens"`
		} `json:"usage"`
	}
	if err := json.Unmarshal([]byte(data), &c); err != nil {
		return Event{}, false, fmt.Errorf("chunk is not JSON: %w", err)
	}
	if len(c.Error) > 0 && string(c.Error) != "null" {
		return Event{}, false, fmt.Errorf("stream error: %s", c.Error)
	}
	if len(c.Choices) == 0 {
		if c.Usage != nil {
			return Event{Kind: Usage, Tokens: c.Usage.CompletionTokens}, true, nil
		}
		return Event{}, false, nil
	}
	ch := c.Choices[0]
	if ch.Delta.Content != "" || ch.Text != "" || len(ch.Delta.ToolCalls) > 0 {
		return Event{Kind: Content, Tokens: 1}, true, nil
	}
	return Event{}, false, nil
}

// Measure turns one request's events into a Sample. start is the intended
// send time. Without a content event the request is an error.
func Measure(start time.Time, events []Event) Sample {
	s := Sample{Start: start}
	var prev time.Time
	usage := -1
	for _, ev := range events {
		switch ev.Kind {
		case Content:
			if s.Tokens == 0 {
				s.TTFT = ev.At.Sub(start)
			} else {
				s.ITL = append(s.ITL, ev.At.Sub(prev))
			}
			prev = ev.At
			s.Tokens++
		case Usage:
			usage = ev.Tokens
		case Done:
			s.E2E = ev.At.Sub(start)
		}
	}
	if s.Tokens == 0 {
		s.Err = errors.New("no content before [DONE]")
		return s
	}
	if usage >= 0 {
		s.Tokens = usage
	}
	return s
}

// one sends request i and measures it against its intended start.
func one(ctx context.Context, cfg *Config, client *http.Client, clk Clock, i int, start time.Time) Sample {
	prompt := ""
	if len(cfg.Prompts) > 0 {
		prompt = cfg.Prompts[i%len(cfg.Prompts)]
	}
	body, _ := json.Marshal(map[string]any{
		"model":          cfg.Model,
		"messages":       []map[string]string{{"role": "user", "content": prompt}},
		"max_tokens":     cfg.MaxTokens,
		"temperature":    0,
		"stream":         true,
		"stream_options": map[string]bool{"include_usage": true},
	})
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, strings.TrimRight(cfg.Target, "/")+"/v1/chat/completions", bytes.NewReader(body))
	if err != nil {
		return Sample{Start: start, Err: err}
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "text/event-stream")
	if cfg.APIKey != "" {
		req.Header.Set("Authorization", "Bearer "+cfg.APIKey)
	}
	resp, err := client.Do(req)
	if err != nil {
		return Sample{Start: start, Err: err}
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
		return Sample{Start: start, Err: fmt.Errorf("HTTP %d: %s", resp.StatusCode, strings.TrimSpace(string(b)))}
	}
	events, err := ReadStream(resp.Body, clk.Now)
	s := Measure(start, events)
	if err != nil {
		s.Err = err
	}
	return s
}

// Run drives one open-loop run and returns its report. Arrivals follow
// cfg.Schedule from the moment Run starts, each request in its own
// goroutine, so a slow or stalled server never slows the send rate. ctx
// cancels in-flight requests and stops new arrivals.
func Run(ctx context.Context, cfg Config) (*Report, error) {
	if cfg.Schedule == nil {
		return nil, errors.New("loadgen: Config.Schedule is required")
	}
	if cfg.Duration <= 0 && cfg.MaxRequests <= 0 {
		return nil, errors.New("loadgen: set Duration or MaxRequests")
	}
	clk := cfg.Clock
	if clk == nil {
		clk = RealClock
	}
	client := cfg.Client
	if client == nil {
		client = &http.Client{}
	}
	start := clk.Now()
	var (
		mu      sync.Mutex
		samples []Sample
		wg      sync.WaitGroup
	)
	next := start
	for i := 0; cfg.MaxRequests <= 0 || i < cfg.MaxRequests; i++ {
		// The intended time comes from the schedule alone, never from when
		// the previous request finished.
		next = next.Add(cfg.Schedule.Next())
		if cfg.Duration > 0 && !next.Before(start.Add(cfg.Duration)) {
			break
		}
		if wait := next.Sub(clk.Now()); wait > 0 {
			select {
			case <-clk.After(wait):
			case <-ctx.Done():
			}
		}
		if ctx.Err() != nil {
			break
		}
		wg.Add(1)
		go func(i int, at time.Time) {
			defer wg.Done()
			s := one(ctx, &cfg, client, clk, i, at)
			mu.Lock()
			samples = append(samples, s)
			mu.Unlock()
		}(i, next)
	}
	wg.Wait()
	end := clk.Now()
	sort.SliceStable(samples, func(a, b int) bool { return samples[a].Start.Before(samples[b].Start) })
	elapsed := end.Sub(start)
	if cfg.Duration > 0 && elapsed < cfg.Duration {
		elapsed = cfg.Duration
	}
	return BuildReport(cfg, samples, elapsed), nil
}

// Quantiles is one latency metric of the report, in milliseconds.
type Quantiles struct {
	P50  float64 `json:"p50"`
	P90  float64 `json:"p90"`
	P95  float64 `json:"p95"`
	P99  float64 `json:"p99"`
	Mean float64 `json:"mean"`
}

// Report is formats/loadgen-report.schema.json.
type Report struct {
	RunID      string                  `json:"run_id"`
	Target     string                  `json:"target"`
	Model      string                  `json:"model,omitempty"`
	Mode       string                  `json:"mode"`
	RateRPS    float64                 `json:"rate_rps"`
	DurationS  float64                 `json:"duration_s"`
	Seed       uint64                  `json:"seed"`
	Requests   int                     `json:"requests"`
	Errors     int                     `json:"errors"`
	TTFTms     Quantiles               `json:"ttft_ms"`
	TPOTms     Quantiles               `json:"tpot_ms"`
	ITLms      Quantiles               `json:"itl_ms"`
	E2Ems      Quantiles               `json:"e2e_ms"`
	ErrorRate  float64                 `json:"error_rate"`
	Goodput    float64                 `json:"goodput"`
	TokensPerS float64                 `json:"tokens_per_s"`
	SLO        map[string]float64      `json:"slo,omitempty"`
	Histogram  map[string][][2]float64 `json:"histogram"`
}

// summarize is a histogram's quantiles in milliseconds.
func summarize(h *Histogram) Quantiles {
	ms := func(ns int64) float64 { return float64(ns) / 1e6 }
	return Quantiles{
		P50: ms(h.Quantile(0.50)), P90: ms(h.Quantile(0.90)),
		P95: ms(h.Quantile(0.95)), P99: ms(h.Quantile(0.99)),
		Mean: h.Mean() / 1e6,
	}
}

// sloMetric maps an SLO key ("ttft_ms_p95") to the metric it bounds
// ("ttft_ms"); ok is false for a key that names no metric.
func sloMetric(key string) (string, bool) {
	for _, m := range []string{"ttft_ms", "tpot_ms", "itl_ms", "e2e_ms"} {
		if key == m || strings.HasPrefix(key, m+"_") {
			return m, true
		}
	}
	return "", false
}

// good reports whether one successful sample is within every SLO bound.
func good(s Sample, slo map[string]float64) bool {
	ms := func(d time.Duration) float64 { return float64(d) / 1e6 }
	for k, bound := range slo {
		m, ok := sloMetric(k)
		if !ok {
			continue
		}
		var v float64
		switch m {
		case "ttft_ms":
			v = ms(s.TTFT)
		case "e2e_ms":
			v = ms(s.E2E)
		case "tpot_ms":
			t, ok := s.TPOT()
			if !ok {
				continue
			}
			v = ms(t)
		case "itl_ms":
			for _, d := range s.ITL {
				v = math.Max(v, ms(d))
			}
		}
		if v > bound {
			return false
		}
	}
	return true
}

// BuildReport aggregates samples. Failed requests count in requests,
// errors, and error_rate, never in a latency histogram; goodput is the
// share of ALL requests that succeeded within every SLO bound; tokens_per_s
// is successful output tokens over elapsed.
func BuildReport(cfg Config, samples []Sample, elapsed time.Duration) *Report {
	hs := map[string]*Histogram{
		"ttft_ms": NewHistogram(), "tpot_ms": NewHistogram(),
		"itl_ms": NewHistogram(), "e2e_ms": NewHistogram(),
	}
	r := &Report{
		RunID: cfg.RunID, Target: cfg.Target, Model: cfg.Model,
		DurationS: elapsed.Seconds(), Seed: cfg.Seed, SLO: cfg.SLO,
		Requests: len(samples),
	}
	if cfg.Schedule != nil {
		r.Mode, r.RateRPS = cfg.Schedule.Name(), cfg.Schedule.Rate()
	}
	if r.RunID == "" {
		r.RunID = "run"
	}
	tokens, nGood := 0, 0
	for _, s := range samples {
		if s.Err != nil {
			r.Errors++
			continue
		}
		hs["ttft_ms"].Record(int64(s.TTFT))
		hs["e2e_ms"].Record(int64(s.E2E))
		if t, ok := s.TPOT(); ok {
			hs["tpot_ms"].Record(int64(t))
		}
		for _, d := range s.ITL {
			hs["itl_ms"].Record(int64(d))
		}
		tokens += s.Tokens
		if good(s, cfg.SLO) {
			nGood++
		}
	}
	if r.Requests > 0 {
		r.ErrorRate = float64(r.Errors) / float64(r.Requests)
		r.Goodput = float64(nGood) / float64(r.Requests)
	}
	if elapsed > 0 {
		r.TokensPerS = float64(tokens) / elapsed.Seconds()
	}
	r.TTFTms, r.TPOTms = summarize(hs["ttft_ms"]), summarize(hs["tpot_ms"])
	r.ITLms, r.E2Ems = summarize(hs["itl_ms"]), summarize(hs["e2e_ms"])
	r.Histogram = map[string][][2]float64{}
	for name, h := range hs {
		rows := [][2]float64{}
		for _, b := range h.Buckets() {
			rows = append(rows, [2]float64{float64(b.Upper) / 1e6, float64(b.Count)})
		}
		r.Histogram[name] = rows
	}
	return r
}
