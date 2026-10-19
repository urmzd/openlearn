package main

// policy.go: the usage policy stage (gw.08, ethics.05) of the --config form.
//
// [gateway].policy_file names the ordered policy (policy.v1); classifier
// heads are read under [paths].artifacts. A head's embedding_model picks
// the embedder: "hash16" is computed in process (cmd/internal/hashembed, the
// PR CI stand-in when no embedding model runs), any other model is asked
// for at [ext].policy_embeddings_url (default [gateway].upstream) through
// /v1/embeddings. Every deny is logged with the request id, tenant, model,
// and rule id, never the prompt.
//
// The stage sees only what it can judge: POSTs to the inference routes whose
// body parses. Every other request passes straight on, so a GET /v1/models
// or a malformed body gets the answer the later stages give, as without a
// policy.

import (
	"context"
	"log/slog"
	"net/http"

	"tinyllm/cmd/internal/hashembed"
	"tinyllm/config"
	"tinyllm/gateway/ledger"
	"tinyllm/gateway/policy"
	"tinyllm/gateway/server"
)

// modelEmbedder dispatches on the head's embedding model.
type modelEmbedder struct{ remote policy.HTTPEmbedder }

func (m modelEmbedder) Embed(ctx context.Context, model, text string) ([]float64, error) {
	if model != hashembed.Model {
		return m.remote.Embed(ctx, model, text)
	}
	v := hashembed.Vector(text)
	out := make([]float64, len(v))
	for i, x := range v {
		out[i] = float64(x)
	}
	return out, nil
}

func auditLog(_ context.Context, e policy.AuditEvent) error {
	slog.Info("usage policy deny", "request_id", e.RequestID, "tenant", e.Tenant, "model", e.Model,
		"rule_id", e.RuleID, "outcome", e.Outcome, "reason", e.Reason)
	return nil
}

// policyStage builds the stage, or nil without a policy_file.
func policyStage(cfg config.Config) (server.Middleware, error) {
	g := cfg.Gateway
	if g.PolicyFile == "" {
		return nil, nil
	}
	url, _ := cfg.Ext["policy_embeddings_url"].(string)
	if url == "" {
		url = g.Upstream
	}
	ev, err := policy.Load(g.PolicyFile, cfg.Paths.Artifacts, modelEmbedder{policy.HTTPEmbedder{BaseURL: url}}, auditLog)
	if err != nil {
		return nil, err
	}
	inner := policy.Middleware(ev, url)
	return func(next http.Handler) http.Handler {
		judged := inner(next)
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.Method == http.MethodPost && ledger.MeteredRoutes[r.URL.Path] {
				if x := server.ExchangeFrom(r.Context()); x != nil {
					if _, err := x.Request(r); err == nil {
						judged.ServeHTTP(w, r)
						return
					}
				}
			}
			next.ServeHTTP(w, r)
		})
	}, nil
}
