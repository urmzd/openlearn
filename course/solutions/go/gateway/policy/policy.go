// Package policy implements the ordered usage policy (gw.08).
// Contract: contracts/formats/policy.v1.schema.json and
// contracts/formats/linear-head.schema.json. Chapter:
// content/courses/ai-platform-engineering/12-gateway/06-usage-policy-enforcement.md.
package policy

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"time"

	"gopkg.in/yaml.v3"

	"tinyllm/gateway/auth"
	"tinyllm/gateway/route"
	"tinyllm/gateway/server"
)

var ErrEmbeddingUnavailable = errors.New("policy embeddings unavailable")

type Rule struct {
	ID         string          `json:"id" yaml:"id"`
	Match      Match           `json:"match" yaml:"match"`
	Classifier *ClassifierRule `json:"classifier,omitempty" yaml:"classifier,omitempty"`
	Action     string          `json:"action" yaml:"action"`
	Reason     string          `json:"reason" yaml:"reason"`
}
type Match struct {
	Model     []string `json:"model,omitempty" yaml:"model,omitempty"`
	Tenant    []string `json:"tenant,omitempty" yaml:"tenant,omitempty"`
	MaxTokens *int     `json:"max_tokens,omitempty" yaml:"max_tokens,omitempty"`
}
type ClassifierRule struct {
	Head      string   `json:"head" yaml:"head"`
	Class     string   `json:"class,omitempty" yaml:"class,omitempty"`
	Threshold *float64 `json:"threshold,omitempty" yaml:"threshold,omitempty"`
}
type Document struct {
	Version       int    `json:"version" yaml:"version"`
	DefaultAction string `json:"default_action" yaml:"default_action"`
	Rules         []Rule `json:"rules" yaml:"rules"`
}
type Head struct {
	EmbeddingModel string      `json:"embedding_model"`
	Dim            int         `json:"dim"`
	Classes        []string    `json:"classes"`
	W              [][]float64 `json:"W"`
	B              []float64   `json:"b"`
	Threshold      float64     `json:"threshold"`
}
type Decision struct {
	Allow          bool
	RuleID, Reason string
}
type AuditEvent struct {
	RequestID, Tenant, Model, RuleID, Outcome string
	Reason                                    string
	At                                        time.Time
}
type Audit func(context.Context, AuditEvent) error
type Embedder interface {
	Embed(context.Context, string, string) ([]float64, error)
}

type HTTPEmbedder struct {
	BaseURL string
	Client  *http.Client
}

func (e HTTPEmbedder) Embed(ctx context.Context, model, text string) ([]float64, error) {
	client := e.Client
	if client == nil {
		client = &http.Client{Timeout: 5 * time.Second}
	}
	body, _ := json.Marshal(map[string]any{"model": model, "input": text})
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, strings.TrimRight(e.BaseURL, "/")+"/v1/embeddings", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("embeddings status %d", resp.StatusCode)
	}
	var wire struct {
		Data []struct {
			Embedding []float64 `json:"embedding"`
		} `json:"data"`
	}
	if err := json.NewDecoder(io.LimitReader(resp.Body, 4<<20)).Decode(&wire); err != nil || len(wire.Data) != 1 {
		return nil, ErrEmbeddingUnavailable
	}
	return wire.Data[0].Embedding, nil
}

type Evaluator struct {
	Document Document
	Heads    map[string]Head
	Embedder Embedder
	Audit    Audit
	Now      func() time.Time
}

// UnmarshalYAML accepts the policy schema's scalar or list form for model and tenant matches.
func (m *Match) UnmarshalYAML(node *yaml.Node) error {
	if node.Kind != yaml.MappingNode {
		return errors.New("policy match must be a mapping")
	}
	for i := 0; i < len(node.Content); i += 2 {
		key, value := node.Content[i].Value, node.Content[i+1]
		switch key {
		case "model", "tenant":
			var values []string
			if value.Kind == yaml.ScalarNode {
				values = []string{value.Value}
			} else if err := value.Decode(&values); err != nil {
				return err
			}
			if key == "model" {
				m.Model = values
			} else {
				m.Tenant = values
			}
		case "max_tokens":
			var n int
			if err := value.Decode(&n); err != nil {
				return err
			}
			m.MaxTokens = &n
		default:
			return fmt.Errorf("unknown policy match field %q", key)
		}
	}
	return nil
}

// Load reads the ordered YAML policy and referenced JSON heads. Head paths
// are constrained to artifactRoot so policy input cannot read arbitrary files.
func Load(policyPath, artifactRoot string, embedder Embedder, audit Audit) (*Evaluator, error) {
	b, err := os.ReadFile(policyPath)
	if err != nil {
		return nil, err
	}
	var doc Document
	dec := yaml.NewDecoder(bytes.NewReader(b))
	dec.KnownFields(true)
	if err = dec.Decode(&doc); err != nil {
		return nil, err
	}
	if doc.Version != 1 || len(doc.Rules) == 0 {
		return nil, errors.New("invalid or empty policy document")
	}
	heads := map[string]Head{}
	for _, rule := range doc.Rules {
		if rule.Classifier == nil {
			continue
		}
		rel := filepath.Clean(rule.Classifier.Head)
		if filepath.IsAbs(rel) || rel == "." || rel == ".." || strings.HasPrefix(rel, ".."+string(filepath.Separator)) {
			return nil, fmt.Errorf("head path %q escapes artifacts", rel)
		}
		raw, e := os.ReadFile(filepath.Join(artifactRoot, rel))
		if e != nil {
			return nil, e
		}
		var h Head
		if e = json.Unmarshal(raw, &h); e != nil {
			return nil, e
		}
		if h.Dim <= 0 || len(h.Classes) < 2 || len(h.W) != len(h.Classes) || len(h.B) != len(h.Classes) {
			return nil, fmt.Errorf("invalid linear head %q", rel)
		}
		heads[rule.Classifier.Head] = h
	}
	return &Evaluator{Document: doc, Heads: heads, Embedder: embedder, Audit: audit}, nil
}

func (p *Evaluator) Check(ctx context.Context, principal auth.Principal, req *route.InferenceRequest, maxTokens int, text, requestID string) (Decision, error) {
	for _, rule := range p.Document.Rules {
		if !matches(rule.Match, principal, req, maxTokens) {
			continue
		}
		if rule.Classifier != nil {
			head, ok := p.Heads[rule.Classifier.Head]
			if !ok || p.Embedder == nil {
				return Decision{}, ErrEmbeddingUnavailable
			}
			emb, err := p.Embedder.Embed(ctx, head.EmbeddingModel, text)
			if err != nil {
				return Decision{}, ErrEmbeddingUnavailable
			}
			probs, err := Probabilities(head, emb)
			if err != nil {
				return Decision{}, err
			}
			class := rule.Classifier.Class
			if class == "" {
				class = head.Classes[len(head.Classes)-1]
			}
			idx := -1
			for i, c := range head.Classes {
				if c == class {
					idx = i
					break
				}
			}
			if idx < 0 {
				return Decision{}, fmt.Errorf("unknown policy class %q", class)
			}
			threshold := head.Threshold
			if rule.Classifier.Threshold != nil {
				threshold = *rule.Classifier.Threshold
			}
			if probs[idx] < threshold {
				continue
			}
		}
		decision := Decision{Allow: rule.Action == "allow", RuleID: rule.ID, Reason: rule.Reason}
		if !decision.Allow {
			p.audit(ctx, AuditEvent{RequestID: requestID, Tenant: principal.Tenant, Model: req.Model, RuleID: rule.ID, Reason: rule.Reason, Outcome: "deny", At: p.now()})
		}
		return decision, nil
	}
	if p.Document.DefaultAction == "deny" {
		decision := Decision{Allow: false, RuleID: "default-deny", Reason: "request denied by default policy"}
		p.audit(ctx, AuditEvent{RequestID: requestID, Tenant: principal.Tenant, Model: req.Model, RuleID: decision.RuleID, Reason: decision.Reason, Outcome: "deny", At: p.now()})
		return decision, nil
	}
	return Decision{Allow: true}, nil
}

func matches(m Match, principal auth.Principal, req *route.InferenceRequest, maxTokens int) bool {
	if len(m.Model) > 0 && !contains(m.Model, req.Model) {
		return false
	}
	if len(m.Tenant) > 0 && !contains(m.Tenant, principal.Tenant) {
		return false
	}
	if m.MaxTokens != nil && maxTokens <= *m.MaxTokens {
		return false
	}
	return true
}
func contains(items []string, value string) bool {
	for _, x := range items {
		if x == value {
			return true
		}
	}
	return false
}
func (p *Evaluator) audit(ctx context.Context, event AuditEvent) {
	if p.Audit != nil {
		_ = p.Audit(ctx, event)
	}
}
func (p *Evaluator) now() time.Time {
	if p.Now != nil {
		return p.Now()
	}
	return time.Now()
}

// Probabilities applies the exported linear head to a normalized embedding.
func Probabilities(h Head, embedding []float64) ([]float64, error) {
	if len(embedding) != h.Dim || len(h.W) != len(h.Classes) || len(h.B) != len(h.Classes) {
		return nil, errors.New("linear head shape mismatch")
	}
	norm := 0.0
	for _, v := range embedding {
		norm = math.Hypot(norm, v)
	}
	x := make([]float64, len(embedding))
	if norm > 0 {
		for i, v := range embedding {
			x[i] = v / norm
		}
	}
	logits := make([]float64, len(h.Classes))
	max := math.Inf(-1)
	for c, row := range h.W {
		if len(row) != h.Dim {
			return nil, errors.New("linear head weight shape mismatch")
		}
		z := h.B[c]
		for j, w := range row {
			z += w * x[j]
		}
		logits[c] = z
		if z > max {
			max = z
		}
	}
	sum := 0.0
	for i, z := range logits {
		logits[i] = math.Exp(z - max)
		sum += logits[i]
	}
	for i := range logits {
		logits[i] /= sum
	}
	return logits, nil
}

var (
	emailPattern = regexp.MustCompile(`(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b`)
	phonePattern = regexp.MustCompile(`(^|[^[:alnum:]])(?:\+?1[-. ]?)?\(?\d{3}\)?[-. ]?\d{3}[-. ]?\d{4}\b`)
	cardPattern  = regexp.MustCompile(`\b(?:\d[ -]?){12,18}\d\b`)
	ipPattern    = regexp.MustCompile(`(?i)\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b|\b[0-9a-f:]{3,39}\b`)
	keyPattern   = regexp.MustCompile(`\b(?:AKIA[0-9A-Z]{16}|tl_[A-Za-z0-9_-]{20,})\b`)
)

func Redact(text string) string {
	text = emailPattern.ReplaceAllString(text, "<EMAIL>")
	text = phonePattern.ReplaceAllString(text, "$1<PHONE>")
	text = cardPattern.ReplaceAllStringFunc(text, func(candidate string) string {
		if validLuhn(candidate) {
			return "<CARD>"
		}
		return candidate
	})
	text = ipPattern.ReplaceAllStringFunc(text, func(candidate string) string {
		if net.ParseIP(candidate) != nil {
			return "<IP>"
		}
		return candidate
	})
	text = keyPattern.ReplaceAllString(text, "<KEY>")
	return text
}

func validLuhn(value string) bool {
	digits := make([]int, 0, len(value))
	for _, r := range value {
		if r >= '0' && r <= '9' {
			digits = append(digits, int(r-'0'))
		}
	}
	if len(digits) < 13 || len(digits) > 19 {
		return false
	}
	sum, double := 0, false
	for i := len(digits) - 1; i >= 0; i-- {
		d := digits[i]
		if double {
			d *= 2
			if d > 9 {
				d -= 9
			}
		}
		sum += d
		double = !double
	}
	return sum%10 == 0
}

// Middleware adapts the evaluator to server.Deps.Policy. auditText is
// deliberately not included in AuditEvent; only non-content identifiers are kept.
func Middleware(e *Evaluator, embeddingsURL string) server.Middleware {
	if e.Embedder == nil {
		e.Embedder = HTTPEmbedder{BaseURL: embeddingsURL}
	}
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			x := server.ExchangeFrom(r.Context())
			principal, ok := auth.PrincipalFrom(r.Context())
			if x == nil || !ok {
				server.WriteError(w, http.StatusUnauthorized, "unauthorized", "unauthorized", "", "authentication required")
				return
			}
			parsed, err := x.Request(r)
			if err != nil {
				server.WriteError(w, http.StatusBadRequest, "invalid_request", "invalid_request", "", err.Error())
				return
			}
			var raw struct {
				Messages []struct {
					Content string `json:"content"`
				} `json:"messages"`
				Prompt string `json:"prompt"`
			}
			_ = json.Unmarshal(parsed.Body, &raw)
			text := raw.Prompt
			for _, m := range raw.Messages {
				if m.Content != "" {
					text += "\n" + m.Content
				}
			}
			decision, err := e.Check(r.Context(), principal, &route.InferenceRequest{Model: parsed.Model}, parsed.MaxTokens, text, x.RequestID)
			if err != nil {
				e.audit(r.Context(), AuditEvent{RequestID: x.RequestID, Tenant: principal.Tenant, Model: parsed.Model, RuleID: "classifier-unavailable", Outcome: "deny", Reason: "usage policy dependency unavailable", At: e.now()})
				server.WriteError(w, http.StatusServiceUnavailable, "policy_unavailable", "policy_unavailable", "", "usage policy dependency unavailable")
				return
			}
			if !decision.Allow {
				server.WriteError(w, http.StatusUnavailableForLegalReasons, "policy_error", "usage_policy", "", decision.RuleID+": "+decision.Reason)
				return
			}
			next.ServeHTTP(w, r)
		})
	}
}

var _ = Redact
