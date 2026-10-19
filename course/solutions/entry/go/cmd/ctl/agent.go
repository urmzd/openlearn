package main

// The agent verb (MS-agent): `ctl agent ask`, one question to the docs agent.
//
//	ctl agent ask --query <text> --suite <index> [--id <run id>] [--runs <dir>]
//	              [--durable <host:port>] [provider flags]
//
// --suite names the RAG index (agentkit.ResolveIndex: course-docs is the
// index [build] ingests from docs/). The run is durable (ag.05): its journal
// lives under --runs (default: a temporary directory, removed afterwards),
// and the same --id with the same --runs replays it. With --durable (or
// $TL_DURABLE_ADDR) the run is the AgentRun workflow on the agent worker's
// queue instead of this process.
//
// Provider flags (also for eval): --provider faketool|openai (default
// $TL_AGENT_PROVIDER; else openai when a base URL is set, else faketool),
// --rules (default $TL_AGENT_FAKETOOL_RULES, else config/agent.faketool.json),
// --base-url (default $TL_AGENT_BASE_URL), --model (default $TL_AGENT_MODEL,
// else smol-135m-instruct), --api-key-env (default TL_API_KEY), --usage-db.
// PR CI runs the faketool rules: the same prompts, tools, gate, and loop,
// with a deterministic model. A real model is opt-in by setting the base URL.
//
// Final JSON line: {run_id, status, answer, citations, cited, unknown_citations,
// tool_calls, tool_errors, tool_success}; exit 1 unless the run completed.

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"time"

	"tinyllm/agent/durableagent"
	"tinyllm/cmd/internal/agentkit"
)

func init() {
	register("agent", func(args []string) error { return subverbs("agent", map[string]verb{"ask": agentAsk}, args) })
}

func envOr(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

// providerFlags adds the provider flags and returns a builder for the kit.
func providerFlags(fs *flag.FlagSet) func() (*agentkit.Kit, error) {
	prov := fs.String("provider", os.Getenv("TL_AGENT_PROVIDER"), "faketool or openai")
	rules := fs.String("rules", envOr("TL_AGENT_FAKETOOL_RULES", "config/agent.faketool.json"), "faketool rules file")
	base := fs.String("base-url", os.Getenv("TL_AGENT_BASE_URL"), "OpenAI-compatible base URL ending in /v1")
	model := fs.String("model", envOr("TL_AGENT_MODEL", "smol-135m-instruct"), "model")
	keyEnv := fs.String("api-key-env", "TL_API_KEY", "environment variable holding the API key")
	usageDB := fs.String("usage-db", "", "the gateway's usage ledger (registers query_usage)")
	maxIter := fs.Int("max-iterations", 8, "model calls per run")
	return func() (*agentkit.Kit, error) {
		k, err := agentkit.New(agentkit.Options{Provider: *prov, Rules: *rules, BaseURL: *base, APIKey: os.Getenv(*keyEnv),
			Model: *model, MaxIter: *maxIter, UsageDB: *usageDB})
		if err != nil {
			return nil, usagef("%v", err)
		}
		return k, nil
	}
}

func agentAsk(args []string) error {
	fs := newFlags("agent ask")
	query := fs.String("query", "", "the question")
	suite := fs.String("suite", "course-docs", "RAG index: a directory, course-docs, or agent-smoke")
	id := fs.String("id", "", "run id (default ask-<unix ns>)")
	runs := fs.String("runs", "", "journal directory (default: a temporary one)")
	addr := fs.String("durable", os.Getenv("TL_DURABLE_ADDR"), "run as the AgentRun workflow on this durable server")
	queue := fs.String("queue", "agent", "the agent worker's task queue (with --durable)")
	build := providerFlags(fs)
	jsonFlag(fs)
	pos, err := parse(fs, args)
	if err != nil {
		return err
	}
	if err := exactly("agent ask", pos, 0, "no positional arguments"); err != nil {
		return err
	}
	if *query == "" {
		return usagef("agent ask: --query is required")
	}
	if *id == "" {
		*id = fmt.Sprintf("ask-%d", time.Now().UnixNano())
	}
	if !durableagent.ValidRunID(*id) {
		return usagef("agent ask: --id %q is not a valid run id", *id)
	}
	dir, err := agentkit.ResolveIndex(*suite)
	if err != nil {
		return usagef("agent ask: %v", err)
	}
	var a agentkit.Answer
	if *addr != "" {
		if a, err = askDurable(*addr, *queue, *id, *suite, *query); err != nil {
			return err
		}
	} else {
		kit, err := build()
		if err != nil {
			return err
		}
		defer kit.Close()
		jdir := *runs
		if jdir == "" {
			if jdir, err = os.MkdirTemp("", "agent-runs-"); err != nil {
				return err
			}
			defer os.RemoveAll(jdir)
		}
		store, err := durableagent.OpenFileStore(jdir)
		if err != nil {
			return err
		}
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Minute)
		defer cancel()
		if a, err = kit.Ask(ctx, store, *id, dir, *query); err != nil {
			return err
		}
	}
	if err := emit(a); err != nil {
		return err
	}
	if a.Status != string(durableagent.Completed) {
		return fmt.Errorf("agent run %s is %s", a.RunID, a.Status)
	}
	return nil
}

// askDurable starts AgentRun on the agent worker and waits for its answer.
func askDurable(addr, queue, id, suite, query string) (agentkit.Answer, error) {
	var a agentkit.Answer
	input, err := json.Marshal(map[string]string{"run_id": id, "suite": suite, "query": query})
	if err != nil {
		return a, err
	}
	d, err := dial(addr)
	if err != nil {
		return a, err
	}
	defer d.Close()
	_, info, err := d.run("AgentRun", id, queue, input, true)
	if err != nil {
		return a, err
	}
	if err := notCompleted(info); err != nil {
		return a, err
	}
	if err := json.Unmarshal(info.GetResult(), &a); err != nil {
		return a, fmt.Errorf("AgentRun result: %w", err)
	}
	return a, nil
}
