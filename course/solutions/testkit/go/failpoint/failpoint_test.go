package failpoint_test

import (
	"errors"
	"testing"

	"openlearn.urmzd.com/tl/testkit/failpoint"
)

func TestActions(t *testing.T) {
	if err := failpoint.Load("a=error(disk full);b=3*crash;c=off"); err != nil {
		t.Fatal(err)
	}
	err := failpoint.Inject("a")
	if !errors.Is(err, failpoint.ErrInjected) || err.Error() != "failpoint a: disk full" {
		t.Fatalf("a: %v", err)
	}
	code := -1
	failpoint.SetExit(func(c int) { code = c })
	for i := 1; i <= 3; i++ {
		_ = failpoint.Inject("b")
		if (i < 3 && code != -1) || (i == 3 && code != 137) {
			t.Fatalf("evaluation %d: exit code %d", i, code)
		}
	}
	if failpoint.Inject("c") != nil || failpoint.Enabled("c") || failpoint.Inject("unknown") != nil {
		t.Fatal("off and unknown failpoints never fire")
	}
	if failpoint.Count("b") != 3 {
		t.Fatalf("count %d", failpoint.Count("b"))
	}
}

func TestBadSpecs(t *testing.T) {
	for _, s := range []string{"x", "x=boom", "x=0*crash", "x=sleep(forever)", "x=2%crash"} {
		if failpoint.Load(s) == nil {
			t.Errorf("%q should not parse", s)
		}
	}
}

func TestPanic(t *testing.T) {
	_ = failpoint.Load("p=panic")
	defer func() {
		if recover() == nil {
			t.Fatal("no panic")
		}
	}()
	_ = failpoint.Inject("p")
}
