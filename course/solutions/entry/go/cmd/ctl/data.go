package main

import (
	"encoding/json"
	"fmt"
	"net/url"
	"path/filepath"

	"github.com/BurntSushi/toml"
)

// The data verbs: the corpus pipeline as the CorpusBuild workflow (data.09).
func init() {
	data := map[string]verb{"build": dataBuild}
	register("data", func(args []string) error { return subverbs("data", data, args) })
}

// corpusBuildResult is the part of CorpusBuild's result ctl reads.
type corpusBuildResult struct {
	Dataset string `json:"dataset"`
	Version string `json:"version"`
}

// dataBuild starts CorpusBuild on a corpus config (TOML). With --wait the
// final line is the build's RUN-<dataset>-<version>.json object (the keys
// and values of `{corpus} run`) plus workflow_id and status; without it,
// {workflow_id, run_id, started, status}.
func dataBuild(args []string) error {
	fs := newFlags("data build")
	cfgPath := fs.String("config", "", "corpus config, TOML (formats/corpus-config.schema.json)")
	id := fs.String("id", "", "workflow id")
	queue := fs.String("queue", "data", "task queue")
	wait := fs.Bool("wait", false, "wait for the build and print its RUN object")
	addr := durableFlag(fs)
	jsonFlag(fs)
	pos, err := parse(fs, args)
	if err != nil {
		return err
	}
	if err := exactly("data build", pos, 0, "no positional arguments"); err != nil {
		return err
	}
	if *cfgPath == "" || *id == "" {
		return usagef("data build: --config and --id are required")
	}
	cfg, err := corpusConfig(*cfgPath)
	if err != nil {
		return err
	}
	input, err := json.Marshal(map[string]any{"config": cfg})
	if err != nil {
		return err
	}
	d, err := dial(*addr)
	if err != nil {
		return err
	}
	defer d.Close()
	resp, info, err := d.run("CorpusBuild", *id, *queue, input, *wait)
	if err != nil {
		return err
	}
	status := statusName(info.GetStatus())
	if !*wait {
		return emit(map[string]any{"workflow_id": *id, "run_id": resp.GetRunId(), "started": resp.GetStarted(), "status": status})
	}
	if err := notCompleted(info); err != nil {
		_ = emit(map[string]any{"workflow_id": *id, "status": status})
		return err
	}
	var res corpusBuildResult
	if err := json.Unmarshal(info.GetResult(), &res); err != nil {
		return fmt.Errorf("CorpusBuild result: %w", err)
	}
	b, err := d.readArtifact(info, fmt.Sprintf("corpus/RUN-%s-%s.json", res.Dataset, res.Version))
	if err != nil {
		return err
	}
	run := map[string]any{}
	if err := json.Unmarshal(b, &run); err != nil {
		return fmt.Errorf("RUN object: %w", err)
	}
	run["workflow_id"], run["status"] = *id, status
	return emit(run)
}

// corpusConfig reads a corpus TOML as the JSON object CorpusBuild takes.
// The worker resolves relative paths against its artifacts root, not
// against this file, so local source urls, protected sets, and the
// tokenizer path are made absolute here (the TOML says they are relative
// to the config's directory).
func corpusConfig(path string) (map[string]any, error) {
	cfg := map[string]any{}
	if _, err := toml.DecodeFile(path, &cfg); err != nil {
		return nil, usagef("data build: --config %s: %v", path, err)
	}
	dir, err := filepath.Abs(filepath.Dir(path))
	if err != nil {
		return nil, err
	}
	abs := func(p string) string {
		if filepath.IsAbs(p) {
			return p
		}
		return filepath.Join(dir, p)
	}
	for _, s := range tables(cfg["sources"]) {
		if u, ok := s["url"].(string); ok && isLocal(u) {
			s["url"] = abs(u)
		}
	}
	if dc, ok := cfg["decontam"].(map[string]any); ok {
		if ps, ok := dc["protected"].([]any); ok {
			for i, p := range ps {
				if s, ok := p.(string); ok {
					ps[i] = abs(s)
				}
			}
		}
	}
	if tk, ok := cfg["tokenizer"].(map[string]any); ok {
		if p, ok := tk["path"].(string); ok {
			tk["path"] = abs(p)
		}
	}
	return cfg, nil
}

// tables is a TOML array of tables as maps.
func tables(v any) []map[string]any {
	switch t := v.(type) {
	case []map[string]any:
		return t
	case []any:
		var out []map[string]any
		for _, x := range t {
			if m, ok := x.(map[string]any); ok {
				out = append(out, m)
			}
		}
		return out
	}
	return nil
}

// isLocal is a url without a scheme: a file path.
func isLocal(u string) bool {
	p, err := url.Parse(u)
	return err != nil || p.Scheme == ""
}
