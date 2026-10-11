package dur_06

// The course test workflows (DESIGN 4.4, dur.06). Their recorded runs are
// course/fixtures/dur/histories/*.json; course/oracle/dur/record-histories.sh
// regenerates them from these definitions (TestMain's record mode) against
// the reference. Only these workflows are replayed from fixtures; your own
// workflows are recorded in a first run and replayed in a second.

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"sync/atomic"
	"time"

	"tinyllm/durable/worker"
	"tinyllm/durable/workflow"

	"openlearn.urmzd.com/tl/testkit/failpoint"
)

var opts = workflow.ActivityOptions{StartToClose: 10 * time.Second}

// Sequence runs Upper("s1") .. Upper("sN") one after another.
func Sequence(ctx workflow.Context, in []byte) ([]byte, error) {
	var n int
	if err := json.Unmarshal(in, &n); err != nil {
		return nil, err
	}
	var out []string
	for i := 1; i <= n; i++ {
		s, err := workflow.ExecuteActivity[string](ctx, "Upper", fmt.Sprintf("s%d", i), opts).Get(ctx)
		if err != nil {
			return nil, err
		}
		out = append(out, s)
	}
	return json.Marshal(out)
}

// Parallel schedules three Upper calls at once, then gathers them in order.
func Parallel(ctx workflow.Context, in []byte) ([]byte, error) {
	var fs []workflow.Future[string]
	for _, s := range []string{"a", "b", "c"} {
		fs = append(fs, workflow.ExecuteActivity[string](ctx, "Upper", s, opts))
	}
	var out []string
	for _, f := range fs {
		s, err := f.Get(ctx)
		if err != nil {
			return nil, err
		}
		out = append(out, s)
	}
	return json.Marshal(out)
}

// sideCalls counts calls of SideEffects' functions: a replay must not call them.
var sideCalls atomic.Int32

// SideEffects records two non-deterministic values around an activity.
func SideEffects(ctx workflow.Context, in []byte) ([]byte, error) {
	a := workflow.SideEffect(ctx, func() int { return int(sideCalls.Add(1)) * 7 })
	if _, err := workflow.ExecuteActivity[string](ctx, "Upper", "x", opts).Get(ctx); err != nil {
		return nil, err
	}
	b := workflow.SideEffect(ctx, func() int { return int(sideCalls.Add(1)) * 7 })
	return json.Marshal([]int{a, b})
}

// Versioned moved from activity A to B behind GetVersion("use-b", 0, 1).
func Versioned(ctx workflow.Context, in []byte) ([]byte, error) {
	v := workflow.GetVersion(ctx, "use-b", 0, 1)
	typ := "A"
	if v == 1 {
		typ = "B"
	}
	r, err := workflow.ExecuteActivity[string](ctx, typ, "v", opts).Get(ctx)
	if err != nil {
		return nil, err
	}
	return json.Marshal([]any{v, r})
}

// VersionedBefore is Versioned as it was before the change; the fixture
// versioned-before.json was recorded with it, under the type "Versioned".
func VersionedBefore(ctx workflow.Context, in []byte) ([]byte, error) {
	r, err := workflow.ExecuteActivity[string](ctx, "A", "v", opts).Get(ctx)
	if err != nil {
		return nil, err
	}
	return json.Marshal([]any{0, r})
}

// Reorder schedules A then B; with TL_FAILPOINTS="dur/workflow/reorder=..."
// it schedules B then A: a code change that breaks determinism.
func Reorder(ctx workflow.Context, in []byte) ([]byte, error) {
	first, second := "A", "B"
	if failpoint.Enabled("dur/workflow/reorder") {
		first, second = "B", "A"
	}
	f1 := workflow.ExecuteActivity[string](ctx, first, "1", opts)
	f2 := workflow.ExecuteActivity[string](ctx, second, "2", opts)
	r1, err1 := f1.Get(ctx)
	r2, err2 := f2.Get(ctx)
	if err1 != nil || err2 != nil {
		return nil, fmt.Errorf("%v %v", err1, err2)
	}
	return json.Marshal([]string{r1, r2})
}

// Clock returns workflow.Now before and after an activity.
func Clock(ctx workflow.Context, in []byte) ([]byte, error) {
	t1 := workflow.Now(ctx)
	if _, err := workflow.ExecuteActivity[string](ctx, "Upper", "t", opts).Get(ctx); err != nil {
		return nil, err
	}
	t2 := workflow.Now(ctx)
	return json.Marshal([]int64{t1.UnixMilli(), t2.UnixMilli()})
}

// Counter continues as new until its input reaches 3: three short runs.
func Counter(ctx workflow.Context, in []byte) ([]byte, error) {
	var n int
	json.Unmarshal(in, &n)
	if n >= 3 {
		return json.Marshal(n)
	}
	if _, err := workflow.ExecuteActivity[string](ctx, "Upper", fmt.Sprintf("n%d", n), opts).Get(ctx); err != nil {
		return nil, err
	}
	next, _ := json.Marshal(n + 1)
	return nil, workflow.ContinueAsNew(ctx, next)
}

// registry is the code the fixtures were recorded with.
func registry() *workflow.Registry {
	r := workflow.NewRegistry()
	r.Register("Sequence", Sequence)
	r.Register("Parallel", Parallel)
	r.Register("SideEffects", SideEffects)
	r.Register("Versioned", Versioned)
	r.Register("Reorder", Reorder)
	r.Register("Clock", Clock)
	r.Register("Counter", Counter)
	return r
}

// activities are the course test activities.
var activities = map[string]worker.ActivityFunc{
	"Upper": func(_ context.Context, in []byte) ([]byte, error) {
		var s string
		if err := json.Unmarshal(in, &s); err != nil {
			return nil, err
		}
		return json.Marshal(strings.ToUpper(s))
	},
	"A": func(_ context.Context, in []byte) ([]byte, error) {
		return json.Marshal("A:" + strings.Trim(string(in), `"`))
	},
	"B": func(_ context.Context, in []byte) ([]byte, error) {
		return json.Marshal("B:" + strings.Trim(string(in), `"`))
	},
	"Tick": func(_ context.Context, in []byte) ([]byte, error) { return in, nil },
}
