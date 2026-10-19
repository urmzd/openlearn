// Reference learner tests for ds.09 (rung R2: the chapter names these tests
// and says what each checks; the learner writes the bodies). They prove the
// mutation threshold is reachable with black-box tests of the package API.
package ring_test

import (
	"fmt"
	"math"
	"testing"

	"tinyllm/ds/ring"
)

func table(m map[string]uint64) func([]byte) uint64 {
	return func(b []byte) uint64 { return m[string(b)] }
}

// TestOwnerIsFirstPointClockwise: with positions you choose, each key goes to
// the first vnode at or after it, and a key past the last vnode wraps.
func TestOwnerIsFirstPointClockwise(t *testing.T) {
	h := table(map[string]uint64{"a#0": 100, "b#0": 200, "k-50": 50, "k-100": 100, "k-150": 150, "k-250": 250})
	r := ring.New(1, h)
	r.Add("a")
	r.Add("b")
	for k, want := range map[string]string{"k-50": "a", "k-100": "a", "k-150": "b", "k-250": "a"} {
		if got, _ := r.Get([]byte(k)); got != want {
			t.Fatalf("%s -> %s, want %s", k, got, want)
		}
	}
}

// TestEmptyRingSaysNobody: Get and GetBounded on an empty ring return ok = false.
func TestEmptyRingSaysNobody(t *testing.T) {
	r := ring.New(3, nil)
	if _, ok := r.Get([]byte("x")); ok {
		t.Fatal("Get on an empty ring")
	}
	if _, ok := r.GetBounded([]byte("x"), func(string) int { return 0 }, 1); ok {
		t.Fatal("GetBounded on an empty ring")
	}
}

// TestRemoveOnlyMovesItsKeys: after Remove, no key maps to the removed node
// and every other key keeps its owner.
func TestRemoveOnlyMovesItsKeys(t *testing.T) {
	r := ring.New(20, nil)
	for i := 0; i < 4; i++ {
		r.Add(fmt.Sprint("n", i))
	}
	before := map[string]string{}
	for i := 0; i < 500; i++ {
		k := fmt.Sprint("key", i)
		before[k], _ = r.Get([]byte(k))
	}
	r.Remove("n1")
	for k, was := range before {
		now, _ := r.Get([]byte(k))
		if now == "n1" || (was != "n1" && now != was) {
			t.Fatalf("%s: %s -> %s", k, was, now)
		}
	}
}

// TestCapacityFormula: Capacity(total, n, c) = ceil(c * (total + 1) / n).
func TestCapacityFormula(t *testing.T) {
	if ring.Capacity(0, 4, 1) != 1 || ring.Capacity(7, 4, 1) != 2 || ring.Capacity(5, 3, 1.25) != 3 {
		t.Fatal("Capacity")
	}
}

// TestBoundedRespectsCapacity: placing keys one by one, no node ends above
// ceil(c * K / n), even with one hot key.
func TestBoundedRespectsCapacity(t *testing.T) {
	r := ring.New(30, nil)
	for i := 0; i < 4; i++ {
		r.Add(fmt.Sprint("n", i))
	}
	for _, c := range []float64{1, 1.5} {
		load := map[string]int{}
		for i := 0; i < 400; i++ {
			n, _ := r.GetBounded([]byte("hot"), func(x string) int { return load[x] }, c)
			load[n]++
		}
		limit := int(math.Ceil(c * 400 / 4))
		for n, l := range load {
			if l > limit {
				t.Fatalf("c=%v: %s holds %d > %d", c, n, l, limit)
			}
		}
	}
}

// TestBoundedClampsSmallC: c below 1 behaves as c = 1.
func TestBoundedClampsSmallC(t *testing.T) {
	r := ring.New(30, nil)
	for i := 0; i < 3; i++ {
		r.Add(fmt.Sprint("n", i))
	}
	l1, l2 := map[string]int{}, map[string]int{}
	for i := 0; i < 60; i++ {
		a, _ := r.GetBounded([]byte("hot"), func(x string) int { return l1[x] }, 1)
		b, _ := r.GetBounded([]byte("hot"), func(x string) int { return l2[x] }, 0.25)
		l1[a]++
		l2[b]++
		if a != b {
			t.Fatalf("step %d: %s vs %s", i, a, b)
		}
	}
}

// TestTiesGoToSmallerName: two vnodes at one position: the smaller node
// name owns it, whatever the join order.
func TestTiesGoToSmallerName(t *testing.T) {
	same := func([]byte) uint64 { return 7 }
	for _, order := range [][]string{{"x", "y"}, {"y", "x"}} {
		r := ring.New(1, same)
		for _, n := range order {
			r.Add(n)
		}
		if got, _ := r.Get([]byte("k")); got != "x" {
			t.Fatalf("order %v: %s", order, got)
		}
	}
}

// TestHashIsFNV1aThenSplitMix: Hash64 = Mix64(FNV1a64(b)) and FNV1a64 of "a"
// is 0xaf63dc4c8601ec8c; labels are "<node>#<i>".
func TestHashIsFNV1aThenSplitMix(t *testing.T) {
	if ring.FNV1a64([]byte("a")) != 0xaf63dc4c8601ec8c {
		t.Fatal("FNV1a64")
	}
	if ring.Hash64([]byte("a")) != ring.Mix64(ring.FNV1a64([]byte("a"))) || ring.Mix64(0) != 0 {
		t.Fatal("Hash64 or Mix64")
	}
	if string(ring.VnodeLabel("n", 0)) != "n#0" {
		t.Fatal("VnodeLabel")
	}
}
