package main

import (
	"flag"
	"io"
	"os"
)

// newFlags is a flag set that reports errors instead of exiting, so a bad
// flag becomes a usage error (exit 2) through run.
func newFlags(name string) *flag.FlagSet {
	fs := flag.NewFlagSet(name, flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	return fs
}

// parse parses flags and positional arguments in any order (the flag
// package alone stops at the first positional one) and returns the
// positional arguments.
func parse(fs *flag.FlagSet, args []string) ([]string, error) {
	var pos []string
	for {
		if err := fs.Parse(args); err != nil {
			return nil, usagef("%s: %v", fs.Name(), err)
		}
		rest := fs.Args()
		if len(rest) == 0 {
			return pos, nil
		}
		pos = append(pos, rest[0])
		args = rest[1:]
	}
}

// durableFlag adds --durable with its default: $TL_DURABLE_ADDR, else
// 127.0.0.1:7233.
func durableFlag(fs *flag.FlagSet) *string {
	def := os.Getenv("TL_DURABLE_ADDR")
	if def == "" {
		def = "127.0.0.1:7233"
	}
	return fs.String("durable", def, "durable server host:port")
}

// jsonFlag adds --json. The output is JSON lines either way; the flag is
// accepted because the milestones pass it.
func jsonFlag(fs *flag.FlagSet) *bool {
	return fs.Bool("json", false, "JSON lines output (the default)")
}

// exactly checks the number of positional arguments.
func exactly(verb string, pos []string, n int, what string) error {
	if len(pos) != n {
		return usagef("%s: want %s, got %d argument(s)", verb, what, len(pos))
	}
	return nil
}
