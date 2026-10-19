// Package gate decides, before dispatch, whether a tool call may run (ag.04):
// a policy gate (types.Gate) that sorts tools into reads, sends, and writes,
// checks every URL in the arguments against an egress allowlist, and runs
// generated SQL through a deterministic safety classifier (ClassifySQL,
// ported from case study 02). It contains no model, so it cannot be
// prompt-injected: text that arrived in a retrieved chunk or a tool result
// can make the model propose a call, but only this code decides whether the
// call runs.
//
// Chapter: content/courses/ai-platform-engineering/13-agent-sdk/04-tool-gate-and-injection.md.
package gate

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"net/url"
	"regexp"
	"strings"

	"tinyllm/agent/types"
)

// Policy sorts the tools an agent may call. A tool in no list is denied.
type Policy struct {
	// Read tools only read (search_docs, get_weather): allowed, also after
	// untrusted content entered the context.
	Read []string
	// Send tools move data out of the system (send_email, http_post):
	// allowed while the context is untainted; after any tool output, they
	// need approval, because that output may be an attacker's instruction.
	Send []string
	// Write tools change state (create_ticket, delete_key): always need
	// approval.
	Write []string
	// SQL maps a tool to the argument holding its query ("query_usage":
	// "sql"): a read passes, anything else is denied. SQL tools are reads.
	SQL map[string]string
	// EgressHosts are the hosts a URL anywhere in the arguments may name:
	// "docs.example" matches exactly; ".docs.example" matches its
	// subdomains. Any other host is denied, for every tool.
	EgressHosts []string
	// MaxRows caps a SQL read (DefaultMaxRows when 0).
	MaxRows int
}

// PolicyGate implements types.Gate over a Policy.
type PolicyGate struct {
	p     Policy
	class map[string]string // tool -> "read" | "send" | "write" | "sql"
}

// New compiles p.
func New(p Policy) *PolicyGate {
	g := &PolicyGate{p: p, class: map[string]string{}}
	for _, n := range p.Read {
		g.class[n] = "read"
	}
	for _, n := range p.Send {
		g.class[n] = "send"
	}
	for _, n := range p.Write {
		g.class[n] = "write"
	}
	for n := range p.SQL {
		g.class[n] = "sql"
	}
	return g
}

// IsWrite reports whether name is a write tool (ag.05 asks before replaying
// a write whose outcome it never recorded).
func (g *PolicyGate) IsWrite(name string) bool {
	return g.class[name] == "write"
}

// urlRE finds URLs inside free text ("see https://x/?q=...", a markdown
// image), not only arguments that are a URL.
var urlRE = regexp.MustCompile(`(?i)\b[a-z][a-z0-9+.-]*://[^\s"'<>()\[\]{}]+`)

// Check decides one call: unknown tool, egress, SQL, then the tool's class.
func (g *PolicyGate) Check(ctx context.Context, c types.ToolCall) (types.Verdict, error) {
	class, ok := g.class[c.Name]
	if !ok {
		return types.Verdict{Kind: types.Deny, Reason: fmt.Sprintf("tool %q is not in the policy", c.Name)}, nil
	}
	args, err := decodeArgs(c.Args)
	if err != nil {
		return types.Verdict{Kind: types.Deny, Reason: "arguments are not a JSON object"}, nil
	}
	var strs []string
	collectStrings(args, &strs)
	for _, s := range strs {
		for _, raw := range urlRE.FindAllString(s, -1) {
			if host, ok := g.egressAllowed(raw); !ok {
				return types.Verdict{Kind: types.Deny, Reason: fmt.Sprintf("egress to %q is not allowed", host)}, nil
			}
		}
	}
	switch class {
	case "sql":
		q, _ := args[g.p.SQL[c.Name]].(string)
		if r := ClassifySQL(q, g.p.MaxRows); !r.Allowed {
			return types.Verdict{Kind: types.Deny, Reason: r.Reason}, nil
		}
		return types.Verdict{Kind: types.Allow}, nil
	case "write":
		return types.Verdict{Kind: types.NeedApproval, Marker: Marker(c), Reason: "write tools need approval"}, nil
	case "send":
		if types.CallContextFrom(ctx).Tainted {
			return types.Verdict{Kind: types.NeedApproval, Marker: Marker(c), Reason: "sending after untrusted content entered the context needs approval"}, nil
		}
	}
	return types.Verdict{Kind: types.Allow}, nil
}

// egressAllowed parses one URL and checks its host against EgressHosts.
// A URL that does not parse, or has no host, is refused.
func (g *PolicyGate) egressAllowed(raw string) (string, bool) {
	u, err := url.Parse(raw)
	if err != nil || u.Hostname() == "" {
		return raw, false
	}
	host := strings.ToLower(strings.TrimSuffix(u.Hostname(), "."))
	for _, a := range g.p.EgressHosts {
		a = strings.ToLower(a)
		if strings.HasPrefix(a, ".") {
			if strings.HasSuffix(host, a) {
				return host, true
			}
		} else if host == a {
			return host, true
		}
	}
	return host, false
}

func decodeArgs(raw json.RawMessage) (map[string]any, error) {
	if len(bytes.TrimSpace(raw)) == 0 {
		return map[string]any{}, nil
	}
	var m map[string]any
	d := json.NewDecoder(bytes.NewReader(raw))
	d.UseNumber()
	if err := d.Decode(&m); err != nil || m == nil {
		return nil, fmt.Errorf("not an object")
	}
	return m, nil
}

// collectStrings appends every string in v (keys of objects included).
func collectStrings(v any, out *[]string) {
	switch x := v.(type) {
	case string:
		*out = append(*out, x)
	case []any:
		for _, e := range x {
			collectStrings(e, out)
		}
	case map[string]any:
		for k, e := range x {
			*out = append(*out, k)
			collectStrings(e, out)
		}
	}
}

// Marker identifies a call for approval: the first 16 hex digits of
// SHA-256(name, 0x00, canonical JSON of the arguments). Canonical JSON has
// sorted keys and no spaces, so the same call proposed again (keys in
// another order, other whitespace) has the same marker, and any change to
// the arguments changes it.
func Marker(c types.ToolCall) string {
	canon := []byte("{}")
	if args, err := decodeArgs(c.Args); err == nil {
		canon, _ = json.Marshal(args) // encoding/json sorts map keys
	} else {
		canon = c.Args
	}
	h := sha256.New()
	h.Write([]byte(c.Name))
	h.Write([]byte{0})
	h.Write(canon)
	return hex.EncodeToString(h.Sum(nil))[:16]
}
