// Command ctl is the reference entry point for the `ctl` role
// (course/contracts/spec/cli-roles.md): the umbrella CLI of the platform.
// Entry points are learner territory (DESIGN D16): the learner writes their
// own go/cmd/ctl; this one shows what the milestones expect.
//
//	ctl wf start <type> --id <id> [--input-json <json>] [--queue <q>] [--wait] [--json]
//	ctl wf describe <id> --json
//	ctl wf history <id> --json
//	ctl wf dlq list <queue> --json
//	ctl wf dlq redrive <queue> (--all | <task id>...) --json
//	ctl data build --config <corpus toml> --id <id> [--queue <q>] [--wait] [--json]
//	ctl train --spec <file> --id <id> [--queue <q>] [--wait] [--json]
//	ctl usage --gateway <base URL> [--tenant T] [--key-id K] [--since T] [--until T] [--group-by G] [--json]
//
// The flags and final JSON lines are fixed by the milestone headers that
// first call each verb: MS-durable (wf, data build, train) and MS-gateway
// (usage). Every durable verb takes --durable <host:port> (default
// $TL_DURABLE_ADDR, else 127.0.0.1:7233).
//
// Every verb follows the role rules: exit 2 on a usage error, exit 1 on an
// RPC or HTTP error with the reason on stderr, and the last stdout line is
// one JSON object. Output is JSON lines with or without --json.
//
// Each verb group lives in its own file and registers itself in verbs from
// its init(): wf.go, data.go, train.go, usage.go. A later group (agent.go,
// eval.go for MS-agent) adds a file with one more register call; main does
// not change.
package main

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"sort"
	"strings"
)

// verb runs one top-level verb on the arguments after its name.
type verb func(args []string) error

// verbs is the dispatch table, filled by each verb file's init().
var verbs = map[string]verb{}

// register adds a top-level verb; a duplicate name is a programming error.
func register(name string, v verb) {
	if _, dup := verbs[name]; dup {
		panic("ctl: verb registered twice: " + name)
	}
	verbs[name] = v
}

// usageError is a wrong invocation: exit 2.
type usageError struct{ msg string }

func (e usageError) Error() string { return e.msg }

func usagef(format string, a ...any) error { return usageError{fmt.Sprintf(format, a...)} }

func main() {
	os.Exit(run(os.Args[1:]))
}

// run dispatches args to a verb and maps its error to the exit code.
func run(args []string) int {
	if len(args) == 0 {
		return fail(usagef("no verb; verbs: %s", verbNames()))
	}
	v, ok := verbs[args[0]]
	if !ok {
		return fail(usagef("unknown verb %q; verbs: %s", args[0], verbNames()))
	}
	return fail(v(args[1:]))
}

// fail prints err on stderr and returns its exit code (0 for nil).
func fail(err error) int {
	if err == nil {
		return 0
	}
	fmt.Fprintf(os.Stderr, "ctl: %v\n", err)
	var u usageError
	if errors.As(err, &u) {
		return 2
	}
	return 1
}

func verbNames() string {
	names := make([]string, 0, len(verbs))
	for n := range verbs {
		names = append(names, n)
	}
	sort.Strings(names)
	return strings.Join(names, ", ")
}

// subverbs dispatches a verb group (wf, data, wf dlq) on its first argument.
func subverbs(group string, table map[string]verb, args []string) error {
	if len(args) == 0 {
		return usagef("%s: no subverb; one of %s", group, keys(table))
	}
	v, ok := table[args[0]]
	if !ok {
		return usagef("%s: unknown subverb %q; one of %s", group, args[0], keys(table))
	}
	return v(args[1:])
}

func keys(table map[string]verb) string {
	names := make([]string, 0, len(table))
	for n := range table {
		names = append(names, n)
	}
	sort.Strings(names)
	return strings.Join(names, ", ")
}

// emit prints one JSON object on its own stdout line.
func emit(v any) error {
	b, err := json.Marshal(v)
	if err != nil {
		return err
	}
	_, err = fmt.Println(string(b))
	return err
}
