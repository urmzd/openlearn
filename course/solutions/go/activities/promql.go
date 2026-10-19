// promql.go (dur.12): the Prometheus query activity of ModelRelease.
//
// The release's SLO burn check is one PromQL instant query against the
// Prometheus HTTP API (GET /api/v1/query, docs: prometheus.io/docs/
// prometheus/latest/querying/api/). The expression must aggregate to one
// number, for example the fast burn rate of the canary's error SLO:
//
//	sum(rate(http_server_request_duration_seconds_count{model="m-v2",http_response_status_code=~"5.."}[5m]))
//	  / sum(rate(http_server_request_duration_seconds_count{model="m-v2"}[5m])) / (1 - 0.99)
//
// Prometheus answers a vector (one sample per series) or a scalar. No
// series, or a NaN sample (0/0 when the canary saw no traffic), is "no
// data": the activity reports it as such and never invents a 0, because the
// workflow treats no data as a failed check (fail closed).
package activities

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"net/http"
	"net/url"
	"strconv"
	"time"
)

// Prom is a client of one Prometheus server.
type Prom struct {
	Base   string       // for example http://127.0.0.1:30090 (system.toml [deploy].prometheus), no trailing /api
	Client *http.Client // nil: http.DefaultClient
}

// QueryInput is the input of the "promql.query" activity.
type QueryInput struct {
	Expr string    `json:"expr"`
	At   time.Time `json:"at"` // evaluation time; the workflow passes its Now() so a retry asks the same question
}

// Sample is one instant-query answer.
type Sample struct {
	Value  float64 `json:"value"`   // meaningful only when NoData is false; +Inf is a real answer
	NoData bool    `json:"no_data"` // no series, or a NaN sample
	Series int     `json:"series"`  // how many series the answer held (0 or 1)
}

// ErrQuery marks an answer Prometheus refused (status "error": a bad
// expression, a timeout) or one the activity cannot use (several series).
var ErrQuery = errors.New("promql")

// Permanent wraps an error that a retry cannot fix (a bad expression, a
// route that does not exist). The worker's composition root maps it to
// tl.durable.v1 Failure.non_retryable; the workflow sees the failure at once
// instead of after every attempt.
type Permanent struct{ Err error }

func (p Permanent) Error() string      { return p.Err.Error() }
func (p Permanent) Unwrap() error      { return p.Err }
func (p Permanent) NonRetryable() bool { return true }

type promResponse struct {
	Status    string          `json:"status"`
	ErrorType string          `json:"errorType"`
	Error     string          `json:"error"`
	Data      json.RawMessage `json:"data"`
}

type promData struct {
	ResultType string          `json:"resultType"`
	Result     json.RawMessage `json:"result"`
}

type promSeries struct {
	Metric map[string]string `json:"metric"`
	Value  []json.RawMessage `json:"value"`
}

// ParseSample reads one Prometheus value pair [<unix seconds>, "<number>"].
// The number is a string so that NaN and +Inf survive JSON; NaN means no
// data.
func ParseSample(pair []json.RawMessage) (float64, bool, error) {
	if len(pair) != 2 {
		return 0, false, fmt.Errorf("%w: a sample is [time, value], got %d elements", ErrQuery, len(pair))
	}
	var s string
	if err := json.Unmarshal(pair[1], &s); err != nil {
		return 0, false, fmt.Errorf("%w: sample value is not a string: %s", ErrQuery, pair[1])
	}
	v, err := strconv.ParseFloat(s, 64)
	if err != nil {
		return 0, false, fmt.Errorf("%w: sample value %q: %v", ErrQuery, s, err)
	}
	if math.IsNaN(v) {
		return 0, true, nil
	}
	return v, false, nil
}

// Decode turns a /api/v1/query response body into a Sample: a scalar, or a
// vector of at most one series. A status "error" answer, a matrix or string
// result, or a vector of several series is ErrQuery.
func Decode(body []byte) (Sample, error) {
	var r promResponse
	if err := json.Unmarshal(body, &r); err != nil {
		return Sample{}, fmt.Errorf("%w: response is not JSON: %v", ErrQuery, err)
	}
	if r.Status != "success" {
		return Sample{}, fmt.Errorf("%w: %s: %s", ErrQuery, r.ErrorType, r.Error)
	}
	var d promData
	if err := json.Unmarshal(r.Data, &d); err != nil {
		return Sample{}, fmt.Errorf("%w: data: %v", ErrQuery, err)
	}
	switch d.ResultType {
	case "scalar":
		var pair []json.RawMessage
		if err := json.Unmarshal(d.Result, &pair); err != nil {
			return Sample{}, fmt.Errorf("%w: scalar: %v", ErrQuery, err)
		}
		v, nan, err := ParseSample(pair)
		if err != nil {
			return Sample{}, err
		}
		return Sample{Value: v, NoData: nan, Series: 1}, nil
	case "vector":
		var series []promSeries
		if err := json.Unmarshal(d.Result, &series); err != nil {
			return Sample{}, fmt.Errorf("%w: vector: %v", ErrQuery, err)
		}
		switch len(series) {
		case 0:
			return Sample{NoData: true}, nil
		case 1:
			v, nan, err := ParseSample(series[0].Value)
			if err != nil {
				return Sample{}, err
			}
			return Sample{Value: v, NoData: nan, Series: 1}, nil
		default:
			return Sample{}, fmt.Errorf("%w: %d series; the expression must aggregate to one (wrap it in sum(...))", ErrQuery, len(series))
		}
	default:
		return Sample{}, fmt.Errorf("%w: result type %q; want an instant vector or a scalar", ErrQuery, d.ResultType)
	}
}

// Query evaluates expr at the instant `at` (GET /api/v1/query with query
// and time). A transport error or a 5xx is returned as is (the activity is
// retried); an answer Decode rejects (a 4xx carries Prometheus's error) is
// Permanent: asking again gives the same answer.
func (p Prom) Query(ctx context.Context, in QueryInput) (Sample, error) {
	if in.Expr == "" {
		return Sample{}, Permanent{fmt.Errorf("%w: empty expression", ErrQuery)}
	}
	q := url.Values{}
	q.Set("query", in.Expr)
	if !in.At.IsZero() {
		q.Set("time", strconv.FormatFloat(float64(in.At.UnixMilli())/1000, 'f', 3, 64))
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, p.Base+"/api/v1/query?"+q.Encode(), nil)
	if err != nil {
		return Sample{}, err
	}
	c := p.Client
	if c == nil {
		c = http.DefaultClient
	}
	resp, err := c.Do(req)
	if err != nil {
		return Sample{}, err
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(io.LimitReader(resp.Body, 4<<20))
	if err != nil {
		return Sample{}, err
	}
	if resp.StatusCode >= 500 {
		return Sample{}, fmt.Errorf("prometheus: HTTP %d: %s", resp.StatusCode, promTrunc(body, 200))
	}
	s, err := Decode(body)
	if err != nil {
		return Sample{}, Permanent{err}
	}
	return s, nil
}

// promTrunc shortens a response body for an error message.
func promTrunc(b []byte, n int) string {
	if len(b) > n {
		return string(b[:n]) + "..."
	}
	return string(b)
}
