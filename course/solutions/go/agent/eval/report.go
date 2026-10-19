package eval

// report.go: suites in (formats/eval-case.schema.json, one case per line)
// and reports out (formats/eval-result.schema.json: results.jsonl rows and
// summary.json).

import (
	"bufio"
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"time"
)

// LoadSuite reads eval cases, one JSON object per line (blank lines
// skipped): case_id becomes ID, input Input, ground_truth GroundTruth, and
// tags and scorer_args become the annotations "tags" and "scorer_args".
// case_id, input, tags, and scorer_args are required, case ids unique, and
// any other key is an error; errors name the line.
func LoadSuite(r io.Reader) ([]Observation, error) {
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 1<<20), 16<<20)
	var out []Observation
	ids := map[string]bool{}
	line := 0
	for sc.Scan() {
		line++
		b := bytes.TrimSpace(sc.Bytes())
		if len(b) == 0 {
			continue
		}
		var raw map[string]json.RawMessage
		if err := json.Unmarshal(b, &raw); err != nil {
			return nil, fmt.Errorf("%w: line %d: %v", ErrSuite, line, err)
		}
		for k := range raw {
			switch k {
			case "case_id", "input", "ground_truth", "tags", "scorer_args":
			default:
				return nil, fmt.Errorf("%w: line %d: unknown key %q", ErrSuite, line, k)
			}
		}
		for _, k := range []string{"case_id", "input", "tags", "scorer_args"} {
			if _, ok := raw[k]; !ok {
				return nil, fmt.Errorf("%w: line %d: missing %s", ErrSuite, line, k)
			}
		}
		var id string
		if err := json.Unmarshal(raw["case_id"], &id); err != nil || id == "" {
			return nil, fmt.Errorf("%w: line %d: case_id must be a non-empty string", ErrSuite, line)
		}
		if ids[id] {
			return nil, fmt.Errorf("%w: line %d: case_id %q repeated", ErrSuite, line, id)
		}
		ids[id] = true
		out = append(out, Observation{
			ID:          id,
			Input:       raw["input"],
			GroundTruth: raw["ground_truth"],
			Annotations: map[string]json.RawMessage{"tags": raw["tags"], "scorer_args": raw["scorer_args"]},
		})
	}
	if err := sc.Err(); err != nil {
		return nil, err
	}
	return out, nil
}

// InputSHA is the hex SHA-256 of a case's input bytes.
func InputSHA(input json.RawMessage) string {
	h := sha256.Sum256(input)
	return hex.EncodeToString(h[:])
}

// ResultRow is one line of results.jsonl.
type ResultRow struct {
	Suite     string              `json:"suite"`
	CaseID    string              `json:"case_id"`
	Subject   string              `json:"subject"`
	Sample    int                 `json:"sample"`
	InputSHA  string              `json:"input_sha"`
	Output    json.RawMessage     `json:"output"`
	Scores    map[string]*float64 `json:"scores"`
	Errors    map[string]string   `json:"errors,omitempty"`
	LatencyMS float64             `json:"latency_ms"`
	TTFTMS    *float64            `json:"ttft_ms"`
	Tokens    int                 `json:"tokens"`
	TraceID   *string             `json:"trace_id"`
}

// ABStat is one score's paired comparison (filled by ag.12).
type ABStat struct {
	Delta  float64 `json:"delta"`
	CILow  float64 `json:"ci_low"`
	CIHigh float64 `json:"ci_high"`
	PValue float64 `json:"p_value"`
	NPairs int     `json:"n_pairs"`
}

// AB is the summary's A/B block.
type AB struct {
	Base    string            `json:"base"`
	Exp     string            `json:"exp"`
	Metrics map[string]ABStat `json:"metrics"`
}

// Summary is summary.json.
type Summary struct {
	Suite    string                     `json:"suite"`
	RunID    string                     `json:"run_id"`
	Subjects []string                   `json:"subjects"`
	Metrics  map[string]map[string]Stat `json:"metrics"` // subject -> score -> stat
	AB       *AB                        `json:"ab,omitempty"`
	NBoot    int                        `json:"n_boot"`
	Seed     uint64                     `json:"seed"`
}

func ms(d time.Duration) float64 { return float64(d) / float64(time.Millisecond) }

// ResultRows converts r's rows to results.jsonl rows. A failed score is null in
// scores and its message is in errors; an output that is not JSON is
// written as a JSON string; ttft_ms is null when no token arrived and
// trace_id null when the subject did not trace.
func (r *SuiteResult) ResultRows() []ResultRow {
	out := make([]ResultRow, 0, len(r.Rows))
	for _, row := range r.Rows {
		rr := ResultRow{
			Suite: r.Suite, CaseID: row.ID, Subject: row.Subject, Sample: row.Sample,
			InputSHA: InputSHA(row.Input), Output: row.Output, Scores: map[string]*float64{},
			LatencyMS: ms(row.Timing.Total), Tokens: row.Tokens,
		}
		if len(rr.Output) == 0 || !json.Valid(rr.Output) {
			s, _ := json.Marshal(string(row.Output))
			rr.Output = s
		}
		for _, s := range row.Scores {
			if s.Error != "" {
				rr.Scores[s.Name] = nil
				if rr.Errors == nil {
					rr.Errors = map[string]string{}
				}
				rr.Errors[s.Name] = s.Error
				continue
			}
			v := s.Value
			rr.Scores[s.Name] = &v
		}
		if row.Timing.TTFT > 0 {
			t := ms(row.Timing.TTFT)
			rr.TTFTMS = &t
		}
		if row.TraceID != "" {
			id := row.TraceID
			rr.TraceID = &id
		}
		out = append(out, rr)
	}
	return out
}

// Summary is r's summary.json.
func (r *SuiteResult) Summary(runID string) Summary {
	return Summary{
		Suite: r.Suite, RunID: runID, Subjects: []string{r.Subject},
		Metrics: map[string]map[string]Stat{r.Subject: r.Metrics},
		NBoot:   r.NBoot, Seed: r.Seed,
	}
}

// WriteResults writes dir/results.jsonl (one row per line, in row order)
// and dir/summary.json, creating dir. Each file is written to a temporary
// name and renamed, so a crash never leaves half a report.
func WriteResults(dir string, rows []ResultRow, sum Summary) error {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	var b bytes.Buffer
	enc := json.NewEncoder(&b)
	for _, r := range rows {
		if err := enc.Encode(r); err != nil {
			return err
		}
	}
	if err := writeAtomic(filepath.Join(dir, "results.jsonl"), b.Bytes()); err != nil {
		return err
	}
	sb, err := json.MarshalIndent(sum, "", "  ")
	if err != nil {
		return err
	}
	return writeAtomic(filepath.Join(dir, "summary.json"), append(sb, '\n'))
}

func writeAtomic(path string, data []byte) error {
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, data, 0o644); err != nil {
		return err
	}
	return os.Rename(tmp, path)
}
