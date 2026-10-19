// Package failpoint gives contracts named fault points:
//
//	TL_FAILPOINTS="dur/log/after-write=crash;gw/route=error(no upstream);kv/push=sleep(200ms)"
//
// Actions: crash (os.Exit(137), as a SIGKILL would leave it), panic,
// error(msg) (Inject returns an error), sleep(d), off. A `N*` prefix fires
// only on the Nth evaluation (`3*crash`); `%P` fires with probability P from
// a seeded stream (OL_SEED). Code calls Inject at the named point.
package failpoint

import (
	"errors"
	"fmt"
	"math/rand/v2"
	"os"
	"strconv"
	"strings"
	"sync"
	"time"
)

type action struct {
	kind string // crash panic error sleep off
	arg  string
	nth  int     // 0: every time
	prob float64 // 0: always
}

var (
	mu     sync.Mutex
	parsed map[string]action
	counts = map[string]int{}
	rng    *rand.Rand
	exit   = os.Exit
)

// Load parses spec (normally $TL_FAILPOINTS); tests call it directly.
func Load(spec string) error {
	mu.Lock()
	defer mu.Unlock()
	m := map[string]action{}
	for _, part := range strings.Split(spec, ";") {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		name, act, ok := strings.Cut(part, "=")
		if !ok {
			return fmt.Errorf("failpoint %q: want name=action", part)
		}
		a := action{}
		if n, rest, ok := strings.Cut(act, "*"); ok {
			v, err := strconv.Atoi(n)
			if err != nil || v < 1 {
				return fmt.Errorf("failpoint %q: bad count %q", name, n)
			}
			a.nth, act = v, rest
		}
		if p, rest, ok := strings.Cut(act, "%"); ok && p != "" {
			v, err := strconv.ParseFloat(p, 64)
			if err != nil || v < 0 || v > 1 {
				return fmt.Errorf("failpoint %q: bad probability %q", name, p)
			}
			a.prob, act = v, rest
		}
		kind, arg := act, ""
		if i := strings.IndexByte(act, '('); i >= 0 && strings.HasSuffix(act, ")") {
			kind, arg = act[:i], act[i+1:len(act)-1]
		}
		switch kind {
		case "crash", "panic", "error", "sleep", "off":
		default:
			return fmt.Errorf("failpoint %q: unknown action %q", name, kind)
		}
		if kind == "sleep" {
			if _, err := time.ParseDuration(arg); err != nil {
				return fmt.Errorf("failpoint %q: sleep(%s): %v", name, arg, err)
			}
		}
		a.kind, a.arg = kind, arg
		m[strings.TrimSpace(name)] = a
	}
	seed, _ := strconv.ParseUint(os.Getenv("OL_SEED"), 10, 64)
	parsed, counts, rng = m, map[string]int{}, rand.New(rand.NewPCG(seed, 54))
	return nil
}

// ErrInjected wraps every error a failpoint returns.
var ErrInjected = errors.New("failpoint")

func ensure() {
	if parsed == nil {
		mu.Unlock()
		if err := Load(os.Getenv("TL_FAILPOINTS")); err != nil {
			panic(err)
		}
		mu.Lock()
	}
}

// Inject evaluates the named failpoint: nil when it is not enabled.
func Inject(name string) error {
	mu.Lock()
	ensure()
	a, ok := parsed[name]
	if !ok || a.kind == "off" {
		mu.Unlock()
		return nil
	}
	counts[name]++
	fire := (a.nth == 0 || counts[name] == a.nth) && (a.prob == 0 || rng.Float64() < a.prob)
	mu.Unlock()
	if !fire {
		return nil
	}
	switch a.kind {
	case "crash":
		exit(137)
	case "panic":
		panic("failpoint " + name)
	case "sleep":
		d, _ := time.ParseDuration(a.arg)
		time.Sleep(d)
	case "error":
		return fmt.Errorf("%w %s: %s", ErrInjected, name, a.arg)
	}
	return nil
}

// Enabled reports whether a failpoint is configured (any action but off).
func Enabled(name string) bool {
	mu.Lock()
	defer mu.Unlock()
	ensure()
	a, ok := parsed[name]
	return ok && a.kind != "off"
}

// Count is how many times a failpoint was evaluated.
func Count(name string) int {
	mu.Lock()
	defer mu.Unlock()
	return counts[name]
}

// SetExit replaces os.Exit for crash actions (tests only).
func SetExit(f func(int)) { exit = f }
