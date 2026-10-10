package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"strings"
	"time"
)

// The usage verb: the gateway's usage ledger through admin.v1 (MS-gateway).
func init() {
	register("usage", usage)
}

var groupBys = map[string]bool{"none": true, "model": true, "tenant": true, "key_id": true, "api_version": true}

// usageRow holds the counters of one admin.v1 UsageRow; the group key is
// passed through as the gateway sent it.
type usageRow struct {
	Requests         int64 `json:"requests"`
	Errors           int64 `json:"errors"`
	PromptTokens     int64 `json:"prompt_tokens"`
	CompletionTokens int64 `json:"completion_tokens"`
	CachedTokens     int64 `json:"cached_tokens"`
}

// usage prints GET <gateway>/admin/v1/usage, one row per line, then the
// totals over the rows: {requests, errors, prompt_tokens,
// completion_tokens, cached_tokens}. The key is $TL_API_KEY.
func usage(args []string) error {
	fs := newFlags("usage")
	base := fs.String("gateway", "", "gateway base URL, e.g. http://127.0.0.1:8080")
	query := map[string]*string{
		"tenant":   fs.String("tenant", "", "only this tenant"),
		"key_id":   fs.String("key-id", "", "only this key id"),
		"since":    fs.String("since", "", "RFC 3339 start, inclusive"),
		"until":    fs.String("until", "", "RFC 3339 end, exclusive"),
		"group_by": fs.String("group-by", "", "none, model, tenant, key_id, or api_version"),
	}
	jsonFlag(fs)
	pos, err := parse(fs, args)
	if err != nil {
		return err
	}
	if err := exactly("usage", pos, 0, "no positional arguments"); err != nil {
		return err
	}
	if *base == "" {
		return usagef("usage: --gateway is required")
	}
	if g := *query["group_by"]; g != "" && !groupBys[g] {
		return usagef("usage: --group-by %q: one of none, model, tenant, key_id, api_version", g)
	}
	q := url.Values{}
	for k, v := range query {
		if *v != "" {
			q.Set(k, *v)
		}
	}
	rows, err := fetchUsage(strings.TrimRight(*base, "/")+"/admin/v1/usage?"+q.Encode(), os.Getenv("TL_API_KEY"))
	if err != nil {
		return err
	}
	var total usageRow
	for _, raw := range rows {
		var r usageRow
		if err := json.Unmarshal(raw, &r); err != nil {
			return fmt.Errorf("usage row %s: %w", raw, err)
		}
		total.add(r)
		if err := emit(raw); err != nil {
			return err
		}
	}
	return emit(total)
}

func (t *usageRow) add(r usageRow) {
	t.Requests += r.Requests
	t.Errors += r.Errors
	t.PromptTokens += r.PromptTokens
	t.CompletionTokens += r.CompletionTokens
	t.CachedTokens += r.CachedTokens
}

// fetchUsage GETs the usage rows. A non-200 answer is an error carrying the
// gateway's error message.
func fetchUsage(u, key string) ([]json.RawMessage, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, u, nil)
	if err != nil {
		return nil, usagef("usage: --gateway: %v", err)
	}
	req.Header.Set("Authorization", "Bearer "+key)
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("GET /admin/v1/usage: %w", err)
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("GET /admin/v1/usage: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("GET /admin/v1/usage: HTTP %d: %s", resp.StatusCode, errorMessage(body))
	}
	var page struct {
		Data []json.RawMessage `json:"data"`
	}
	if err := json.Unmarshal(body, &page); err != nil {
		return nil, fmt.Errorf("GET /admin/v1/usage: %w", err)
	}
	return page.Data, nil
}

// errorMessage is error.message of an OpenAI-style error body, else the body.
func errorMessage(body []byte) string {
	var e struct {
		Error struct {
			Message string `json:"message"`
		} `json:"error"`
	}
	if json.Unmarshal(body, &e) == nil && e.Error.Message != "" {
		return e.Error.Message
	}
	return strings.TrimSpace(string(body))
}
