// Package promscrape scrapes a Prometheus text-format endpoint (every
// service's /metrics on its health port) so tests can
// assert metrics without a Prometheus server: sample lookup by name and
// labels, sums over a label subset, and histogram quantiles computed the way
// PromQL's histogram_quantile does (linear interpolation inside a bucket).
package promscrape

import (
	"bufio"
	"fmt"
	"io"
	"math"
	"net/http"
	"sort"
	"strconv"
	"strings"
	"time"
)

type Sample struct {
	Name   string
	Labels map[string]string
	Value  float64
}

type Metrics struct {
	Samples []Sample
	Types   map[string]string // from # TYPE lines
}

// Scrape GETs url and parses the body.
func Scrape(url string) (*Metrics, error) {
	c := &http.Client{Timeout: 10 * time.Second}
	r, err := c.Get(url)
	if err != nil {
		return nil, err
	}
	defer r.Body.Close()
	if r.StatusCode != 200 {
		return nil, fmt.Errorf("GET %s: HTTP %d", url, r.StatusCode)
	}
	return Parse(r.Body)
}

// Parse reads the Prometheus text exposition format (version 0.0.4).
func Parse(rd io.Reader) (*Metrics, error) {
	m := &Metrics{Types: map[string]string{}}
	sc := bufio.NewScanner(rd)
	sc.Buffer(make([]byte, 1<<20), 1<<20)
	ln := 0
	for sc.Scan() {
		ln++
		line := strings.TrimSpace(sc.Text())
		if line == "" {
			continue
		}
		if strings.HasPrefix(line, "#") {
			f := strings.Fields(line)
			if len(f) >= 4 && f[1] == "TYPE" {
				m.Types[f[2]] = f[3]
			}
			continue
		}
		s, err := parseSample(line)
		if err != nil {
			return nil, fmt.Errorf("line %d: %v", ln, err)
		}
		m.Samples = append(m.Samples, s)
	}
	return m, sc.Err()
}

func parseSample(line string) (Sample, error) {
	s := Sample{Labels: map[string]string{}}
	i := strings.IndexAny(line, "{ ")
	if i < 0 {
		return s, fmt.Errorf("no value in %q", line)
	}
	s.Name = line[:i]
	rest := line[i:]
	if rest[0] == '{' {
		j := 1
		for j < len(rest) && rest[j] != '}' {
			eq := strings.IndexByte(rest[j:], '=')
			if eq < 0 {
				return s, fmt.Errorf("bad labels in %q", line)
			}
			key := strings.TrimSpace(strings.TrimLeft(rest[j:j+eq], ","))
			j += eq + 1
			if j >= len(rest) || rest[j] != '"' {
				return s, fmt.Errorf("bad label value in %q", line)
			}
			j++
			var b strings.Builder
			for j < len(rest) && rest[j] != '"' {
				if rest[j] == '\\' && j+1 < len(rest) {
					j++
					switch rest[j] {
					case 'n':
						b.WriteByte('\n')
					default:
						b.WriteByte(rest[j])
					}
				} else {
					b.WriteByte(rest[j])
				}
				j++
			}
			s.Labels[key] = b.String()
			j++ // closing quote
			for j < len(rest) && (rest[j] == ',' || rest[j] == ' ') {
				j++
			}
		}
		if j >= len(rest) {
			return s, fmt.Errorf("unterminated labels in %q", line)
		}
		rest = rest[j+1:]
	}
	f := strings.Fields(rest)
	if len(f) == 0 {
		return s, fmt.Errorf("no value in %q", line)
	}
	v, err := parseFloat(f[0])
	if err != nil {
		return s, fmt.Errorf("bad value %q", f[0])
	}
	s.Value = v
	return s, nil
}

func parseFloat(s string) (float64, error) {
	switch s {
	case "+Inf", "Inf":
		return math.Inf(1), nil
	case "-Inf":
		return math.Inf(-1), nil
	case "NaN":
		return math.NaN(), nil
	}
	return strconv.ParseFloat(s, 64)
}

func matches(s Sample, labels map[string]string) bool {
	for k, v := range labels {
		if s.Labels[k] != v {
			return false
		}
	}
	return true
}

// Get the value of the one sample with this name whose labels include labels.
func (m *Metrics) Get(name string, labels map[string]string) (float64, bool) {
	for _, s := range m.Samples {
		if s.Name == name && matches(s, labels) {
			return s.Value, true
		}
	}
	return 0, false
}

// Sum over every sample of name whose labels include labels.
func (m *Metrics) Sum(name string, labels map[string]string) float64 {
	t := 0.0
	for _, s := range m.Samples {
		if s.Name == name && matches(s, labels) {
			t += s.Value
		}
	}
	return t
}

// Bucket of a cumulative histogram.
type Bucket struct {
	LE    float64
	Count float64
}

// Histogram returns the cumulative buckets (summed over matching series),
// the _count, and the _sum of a histogram family.
func (m *Metrics) Histogram(name string, labels map[string]string) ([]Bucket, float64, float64) {
	by := map[float64]float64{}
	for _, s := range m.Samples {
		if s.Name == name+"_bucket" && matches(s, labels) {
			le, err := parseFloat(s.Labels["le"])
			if err == nil {
				by[le] += s.Value
			}
		}
	}
	bs := make([]Bucket, 0, len(by))
	for le, c := range by {
		bs = append(bs, Bucket{le, c})
	}
	sort.Slice(bs, func(i, j int) bool { return bs[i].LE < bs[j].LE })
	return bs, m.Sum(name+"_count", labels), m.Sum(name+"_sum", labels)
}

// Quantile is histogram_quantile(q, ...) over the matching buckets.
func (m *Metrics) Quantile(q float64, name string, labels map[string]string) float64 {
	bs, _, _ := m.Histogram(name, labels)
	return quantile(q, bs)
}

func quantile(q float64, bs []Bucket) float64 {
	if len(bs) == 0 || math.IsNaN(q) {
		return math.NaN()
	}
	total := bs[len(bs)-1].Count
	if total == 0 {
		return math.NaN()
	}
	rank := q * total
	prevLE, prevC := 0.0, 0.0
	for i, b := range bs {
		if b.Count >= rank {
			if math.IsInf(b.LE, 1) {
				if i == 0 {
					return math.NaN()
				}
				return bs[i-1].LE
			}
			if b.Count == prevC {
				return b.LE
			}
			return prevLE + (b.LE-prevLE)*(rank-prevC)/(b.Count-prevC)
		}
		prevLE, prevC = b.LE, b.Count
	}
	return bs[len(bs)-1].LE
}
