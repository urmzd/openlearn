package gate

import (
	"fmt"
	"regexp"
	"strings"
)

// Operation is what a SQL statement does.
type Operation string

const (
	OpRead    Operation = "read"
	OpWrite   Operation = "write"
	OpAdmin   Operation = "admin"
	OpInvalid Operation = "invalid"
)

// SQLResult is the gate's decision about generated SQL. SQL is the text
// cleared to run: comments blanked, and a LIMIT appended when the statement
// has none at its top level.
type SQLResult struct {
	Allowed bool
	Op      Operation
	Reason  string
	SQL     string
}

// DefaultMaxRows caps a read without its own top-level LIMIT.
const DefaultMaxRows = 1000

var (
	readLeading  = map[string]bool{"SELECT": true, "WITH": true}
	writeWords   = map[string]bool{"INSERT": true, "UPDATE": true, "DELETE": true, "REPLACE": true, "MERGE": true, "UPSERT": true}
	adminWords   = map[string]bool{"DROP": true, "ALTER": true, "CREATE": true, "TRUNCATE": true, "ATTACH": true, "DETACH": true, "PRAGMA": true, "VACUUM": true, "REINDEX": true, "GRANT": true, "REVOKE": true, "COMMIT": true, "ROLLBACK": true, "SAVEPOINT": true, "BEGIN": true}
	selectWrites = map[string]bool{"INTO": true, "OUTFILE": true, "DUMPFILE": true}
	wordRE       = regexp.MustCompile(`[A-Za-z_]+`)
	limitParenRE = regexp.MustCompile(`(?i)[()]|\blimit\b`)
)

// scanSQL makes one pass over sql and returns two strings of the same length:
// cleaned (comments blanked, literals intact: safe to run) and masked (the
// contents of quoted literals and identifiers blanked too: what every check
// reads, so nothing inside a string can look like a keyword, a semicolon,
// or a comment).
func scanSQL(sql string) (cleaned, masked string) {
	var c, m strings.Builder
	n := len(sql)
	for i := 0; i < n; {
		ch := sql[i]
		switch {
		case ch == '\'' || ch == '"' || ch == '`':
			j := i + 1
			for j < n {
				if sql[j] == ch {
					if j+1 < n && sql[j+1] == ch { // doubled quote: an escaped quote
						j += 2
						continue
					}
					break
				}
				j++
			}
			end := min(j+1, n) // the closing quote, or the end of an unterminated literal
			span := sql[i:end]
			c.WriteString(span)
			if len(span) >= 2 {
				m.WriteByte(ch)
				m.WriteString(strings.Repeat(" ", len(span)-2))
				m.WriteByte(ch)
			} else {
				m.WriteByte(ch)
			}
			i = end
		case strings.HasPrefix(sql[i:], "--"):
			j := strings.IndexByte(sql[i:], '\n')
			if j < 0 {
				j = n - i
			}
			pad := strings.Repeat(" ", j)
			c.WriteString(pad)
			m.WriteString(pad)
			i += j
		case strings.HasPrefix(sql[i:], "/*"):
			j := strings.Index(sql[i+2:], "*/")
			end := n
			if j >= 0 {
				end = i + 2 + j + 2
			}
			pad := strings.Repeat(" ", end-i)
			c.WriteString(pad)
			m.WriteString(pad)
			i = end
		default:
			c.WriteByte(ch)
			m.WriteByte(ch)
			i++
		}
	}
	return c.String(), m.String()
}

type stmt struct{ cleaned, masked string }

// statements splits on the semicolons of masked (a ';' inside a literal is
// blanked there) and drops empty statements.
func statements(cleaned, masked string) []stmt {
	var out []stmt
	start := 0
	add := func(end int) {
		if strings.TrimSpace(masked[start:end]) != "" {
			out = append(out, stmt{strings.TrimSpace(cleaned[start:end]), strings.TrimSpace(masked[start:end])})
		}
	}
	for i := 0; i < len(masked); i++ {
		if masked[i] == ';' {
			add(i)
			start = i + 1
		}
	}
	add(len(masked))
	return out
}

// leadingToken is the first keyword, looking through opening parentheses
// so "(SELECT 1) UNION (SELECT 2)" is a read.
func leadingToken(masked string) string {
	t := strings.TrimLeft(masked, "( \t\r\n")
	if f := strings.Fields(t); len(f) > 0 {
		return strings.ToUpper(f[0])
	}
	return masked
}

// hasTopLevelLimit reports a LIMIT outside every parenthesis: a LIMIT in a
// subquery bounds the inner rows, not what comes back.
func hasTopLevelLimit(masked string) bool {
	depth := 0
	for _, tok := range limitParenRE.FindAllString(masked, -1) {
		switch tok {
		case "(":
			depth++
		case ")":
			depth = max(0, depth-1)
		default:
			if depth == 0 {
				return true
			}
		}
	}
	return false
}

func operationOf(masked string) Operation {
	words := map[string]bool{}
	for _, w := range wordRE.FindAllString(masked, -1) {
		words[strings.ToUpper(w)] = true
	}
	for w := range words {
		if adminWords[w] {
			return OpAdmin
		}
	}
	for w := range words {
		if writeWords[w] {
			return OpWrite
		}
	}
	if readLeading[leadingToken(masked)] {
		for w := range words {
			if selectWrites[w] { // SELECT ... INTO writes a table or a file
				return OpWrite
			}
		}
		return OpRead
	}
	return OpInvalid
}

// ClassifySQL decides whether generated SQL may run: exactly one statement,
// recognized as a read, with a row cap (maxRows <= 0 means DefaultMaxRows).
// Anything it does not positively recognize as a read is refused: an
// allowlist fails closed, a denylist fails open on the first syntax nobody
// thought of. Port of content/courses/ai-engineering/30-case-study-sql-agent/safety_gate.py.
func ClassifySQL(sql string, maxRows int) SQLResult {
	if maxRows <= 0 {
		maxRows = DefaultMaxRows
	}
	cleaned, masked := scanSQL(sql)
	if strings.TrimSpace(masked) == "" {
		return SQLResult{false, OpInvalid, "Empty query.", sql}
	}
	sts := statements(cleaned, masked)
	if len(sts) != 1 {
		return SQLResult{false, OpInvalid, fmt.Sprintf("Expected exactly one statement, found %d.", len(sts)), sql}
	}
	st := sts[0]
	switch op := operationOf(st.masked); op {
	case OpAdmin:
		return SQLResult{false, op, "Schema/admin statements are never allowed.", sql}
	case OpInvalid:
		return SQLResult{false, op, fmt.Sprintf("Not a recognised read query (got '%s').", leadingToken(st.masked)), sql}
	case OpWrite:
		return SQLResult{false, op, "Write operations are disabled; this agent is read-only.", st.cleaned}
	}
	safe := st.cleaned
	if !hasTopLevelLimit(st.masked) {
		safe = fmt.Sprintf("%s\nLIMIT %d", safe, maxRows)
	}
	return SQLResult{true, OpRead, "Read-only single statement.", safe}
}
