// Course tests for ds.09, the consistent hash ring with bounded loads
// (go/ds/ring).
//
// The hand example and the tie-break cases inject a hash given as a table, so
// every position is a number you can check on paper. The statistical cases
// use the default FNV-1a hash over generated keys ("key-0", "key-1", ...):
// no random generator is involved, so every run sees the same keys, and the
// bounds below are fixed numbers, not flaky ones.
package ds_09

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"testing"

	"tinyllm/ds/ring"
)

// table is the worked example's hash: every label and key has a position you
// can read off this map (chapter section 3).
var table = map[string]uint64{
	"a#0": 10, "a#1": 60,
	"b#0": 30, "b#1": 80,
	"c#0": 50, "c#1": 95,
	"k1": 5, "k2": 55, "k3": 85, "k4": 97, "k5": 30,
}

func tableHash(t *testing.T) func([]byte) uint64 {
	return func(b []byte) uint64 {
		v, ok := table[string(b)]
		if !ok {
			t.Fatalf("the worked example hashes only its own labels and keys; got %q", b)
		}
		return v
	}
}

func handRing(t *testing.T) *ring.Ring {
	r := ring.New(2, tableHash(t))
	for _, n := range []string{"a", "b", "c"} {
		r.Add(n)
	}
	return r
}

func mustGet(t *testing.T, r *ring.Ring, key string) string {
	t.Helper()
	n, ok := r.Get([]byte(key))
	if !ok {
		t.Fatalf("Get(%q) on a non-empty ring returned ok = false", key)
	}
	return n
}

// hash64 is an independent Mix64(FNV-1a 64) (SplitMix64 finalizer) so the statistical tests can
// compute arcs without trusting the code under test.
func hash64(b []byte) uint64 {
	h := uint64(14695981039346656037)
	for _, c := range b {
		h ^= uint64(c)
		h *= 1099511628211
	}
	h = (h ^ (h >> 30)) * 0xbf58476d1ce4e5b9
	h = (h ^ (h >> 27)) * 0x94d049bb133111eb
	return h ^ (h >> 31)
}

func hash64Mix(h uint64) uint64 {
	h = (h ^ (h >> 30)) * 0xbf58476d1ce4e5b9
	h = (h ^ (h >> 27)) * 0x94d049bb133111eb
	return h ^ (h >> 31)
}

func names(n int) []string {
	out := make([]string, n)
	for i := range out {
		out[i] = fmt.Sprintf("engine-%d", i)
	}
	return out
}

func keys(prefix string, n int) [][]byte {
	out := make([][]byte, n)
	for i := range out {
		out[i] = []byte(fmt.Sprintf("%s-%d", prefix, i))
	}
	return out
}

func build(vnodes int, nodes []string) *ring.Ring {
	r := ring.New(vnodes, nil)
	for _, n := range nodes {
		r.Add(n)
	}
	return r
}

// arcShare is each node's fraction of the ring: the arc ending at each of
// its points, from the previous point (the wrap arc included), over 2^64.
func arcShare(vnodes int, nodes []string) map[string]float64 {
	type pt struct {
		pos  uint64
		node string
		idx  int
	}
	var pts []pt
	for _, n := range nodes {
		for i := 0; i < vnodes; i++ {
			pts = append(pts, pt{hash64([]byte(fmt.Sprintf("%s#%d", n, i))), n, i})
		}
	}
	sort.Slice(pts, func(i, j int) bool {
		if pts[i].pos != pts[j].pos {
			return pts[i].pos < pts[j].pos
		}
		if pts[i].node != pts[j].node {
			return pts[i].node < pts[j].node
		}
		return pts[i].idx < pts[j].idx
	})
	share := map[string]float64{}
	for i, p := range pts {
		prev := pts[(i+len(pts)-1)%len(pts)].pos
		share[p.node] += float64(p.pos-prev) / math.Exp2(64) // uint64 subtraction wraps: the arc past 2^64
	}
	return share
}

func TestHandExample(t *testing.T) {
	// WHY: the chapter's worked example, checked on paper: points a 10, b 30,
	//      c 50, a 60, b 80, c 95; a key belongs to the first point at or
	//      after its position, and k4 (97) wraps past the top to a.
	// KIND: unit
	// CATCHES: s01, s02
	// CHAPTER: ds.09 section 3, worked example
	r := handRing(t)
	for key, want := range map[string]string{"k1": "a", "k2": "a", "k3": "c", "k4": "a", "k5": "b"} {
		if got := mustGet(t, r, key); got != want {
			t.Errorf("Get(%s) = %s, want %s (section 3)", key, got, want)
		}
	}
}

func TestBoundedSkipsFullNodeHand(t *testing.T) {
	// WHY: the second half of the worked example: with loads a=2, b=0, c=1
	//      (total 3, n 3, c 1.25) the capacity is ceil(1.25 * 4 / 3) = 2, so
	//      k1's owner a is full and the walk continues clockwise to b.
	// KIND: unit
	// CATCHES: s04, s10
	// CHAPTER: ds.09 section 3, worked example
	r := handRing(t)
	load := map[string]int{"a": 2, "b": 0, "c": 1}
	got, ok := r.GetBounded([]byte("k1"), func(n string) int { return load[n] }, 1.25)
	if !ok || got != "b" {
		t.Fatalf("GetBounded(k1) = %q, %v; want b (a is at capacity 2)", got, ok)
	}
	// k3 is owned by c (load 1 < 2): bounded keeps the plain owner.
	if got, _ := r.GetBounded([]byte("k3"), func(n string) int { return load[n] }, 1.25); got != "c" {
		t.Fatalf("GetBounded(k3) = %q, want c (below capacity, so the owner keeps it)", got)
	}
}

func TestCapacity(t *testing.T) {
	// WHY: the bound is ceil(c * (total + 1) / n); the +1 counts the key being
	//      placed, so with c >= 1 some node is always strictly below it.
	// KIND: unit
	// CATCHES: s03
	// CHAPTER: ds.09 section 2.3
	for _, tc := range []struct {
		total, n int
		c        float64
		want     int
	}{
		{0, 3, 1.25, 1}, {3, 3, 1.25, 2}, {5, 3, 1.25, 3}, {8, 4, 1, 3}, {7, 4, 1, 2}, {3, 0, 1, 0},
	} {
		if got := ring.Capacity(tc.total, tc.n, tc.c); got != tc.want {
			t.Errorf("Capacity(%d, %d, %v) = %d, want %d", tc.total, tc.n, tc.c, got, tc.want)
		}
	}
}

func TestFNV1a64Vectors(t *testing.T) {
	// WHY: the ring's default hash is Mix64(FNV-1a 64): FNV-1a (xor, then
	//      multiply, the KV block hash's function) finalized by SplitMix64. FNV-1
	//      (multiply first) or a different finalizer places every vnode
	//      somewhere else.
	// KIND: unit
	// CATCHES: s07, s08, s14
	// CHAPTER: ds.09 section 2.1
	for in, want := range map[string]uint64{
		"":       0xcbf29ce484222325,
		"a":      0xaf63dc4c8601ec8c,
		"foobar": 0x85944171f73967e8,
	} {
		if got := ring.FNV1a64([]byte(in)); got != want {
			t.Errorf("FNV1a64(%q) = %#x, want %#x", in, got, want)
		}
	}
	// The SplitMix64 finalizer of 1, and Hash64 = Mix64(FNV1a64(b)),
	// against this file's own transcription of the chapter's steps.
	if got := ring.Mix64(1); got != hash64Mix(1) {
		t.Errorf("Mix64(1) = %#x, want %#x", got, hash64Mix(1))
	}
	if got := ring.Hash64([]byte("a")); got != hash64([]byte("a")) {
		t.Errorf("Hash64(a) = %#x, want Mix64(FNV1a64(a)) = %#x", got, hash64([]byte("a")))
	}
	if got := string(ring.VnodeLabel("engine-a", 7)); got != "engine-a#7" {
		t.Errorf("VnodeLabel(engine-a, 7) = %q, want engine-a#7", got)
	}
}

func TestEmptyRing(t *testing.T) {
	// WHY: a gateway with no healthy workers still asks the ring; it must say
	//      "nobody" (ok = false), never panic.
	// KIND: boundary
	// CATCHES: s12
	// CHAPTER: ds.09 section 4
	r := ring.New(4, nil)
	if n, ok := r.Get([]byte("x")); ok || n != "" {
		t.Fatalf("Get on an empty ring = %q, %v; want \"\", false", n, ok)
	}
	if n, ok := r.GetBounded([]byte("x"), func(string) int { return 0 }, 1.25); ok || n != "" {
		t.Fatalf("GetBounded on an empty ring = %q, %v; want \"\", false", n, ok)
	}
	r.Add("a")
	r.Remove("a")
	if _, ok := r.Get([]byte("x")); ok {
		t.Fatal("a ring whose only node was removed is empty again")
	}
}

func TestNewRejectsZeroVnodes(t *testing.T) {
	// WHY: a node with no positions is a member that never receives a key,
	//      which hides a misconfiguration; New refuses it.
	// KIND: boundary
	// CATCHES: s13
	// CHAPTER: ds.09 section 4
	defer func() {
		if recover() == nil {
			t.Fatal("ring.New(0, nil) must panic")
		}
	}()
	ring.New(0, nil)
}

func TestAddRemoveIdempotent(t *testing.T) {
	// WHY: heartbeats re-announce live workers every 2 s (gw.05); adding a
	//      member twice must not double its vnodes, and removing a stranger
	//      must not disturb anyone.
	// KIND: unit
	// CHAPTER: ds.09 section 2.2
	nodes := names(4)
	r := build(20, nodes)
	ks := keys("key", 500)
	before := make([]string, len(ks))
	for i, k := range ks {
		before[i], _ = r.Get(k)
	}
	r.Add(nodes[1])
	r.Remove("not-a-member")
	if r.Len() != 4 {
		t.Fatalf("Len = %d after a repeated Add, want 4", r.Len())
	}
	if got := r.Nodes(); fmt.Sprint(got) != fmt.Sprint(nodes) {
		t.Fatalf("Nodes = %v, want %v (sorted)", got, nodes)
	}
	for i, k := range ks {
		if n, _ := r.Get(k); n != before[i] {
			t.Fatalf("key %s moved from %s to %s after re-adding a member", k, before[i], n)
		}
	}
}

func TestCollisionsBreakTiesByName(t *testing.T) {
	// WHY: two vnodes can land on the same position; the ring must still be
	//      one function of its members, so ties go to the smaller node name,
	//      whatever order the nodes joined in.
	// KIND: boundary
	// CATCHES: s05
	// CHAPTER: ds.09 section 5, Pitfalls, item 4
	byLen := func(b []byte) uint64 { return uint64(len(b)) * 1000 } // "a#0" and "b#0" collide at 3000
	for _, order := range [][]string{{"a", "b"}, {"b", "a"}} {
		r := ring.New(1, byLen)
		for _, n := range order {
			r.Add(n)
		}
		if got := mustGet(t, r, "x"); got != "a" {
			t.Fatalf("join order %v: Get(x) = %s, want a (tie at 3000 goes to the smaller name)", order, got)
		}
	}
}

func TestInsertionOrderIrrelevant(t *testing.T) {
	// WHY: two gateway replicas that learned about the workers in different
	//      orders must agree on every key, or affinity is lost.
	// KIND: property
	// CHAPTER: ds.09 section 2.1
	nodes := names(6)
	a := build(50, nodes)
	rev := append([]string(nil), nodes...)
	sort.Sort(sort.Reverse(sort.StringSlice(rev)))
	b := build(50, rev)
	for _, k := range keys("key", 2000) {
		x, _ := a.Get(k)
		y, _ := b.Get(k)
		if x != y {
			t.Fatalf("key %s: %s with one join order, %s with the other", k, x, y)
		}
	}
}

func TestAddMovesOnlyToNewNode(t *testing.T) {
	// WHY: the reason to use a ring at all: adding a ninth replica moves only
	//      the keys the new node now owns (about K/9), and each moved key goes
	//      to the new node. mod-N hashing would move about 8/9 of them.
	// KIND: property, statistical
	// CHAPTER: ds.09 section 2.2
	const K, vn = 20000, 160
	nodes := names(8)
	r := build(vn, nodes)
	ks := keys("key", K)
	before := make([]string, K)
	for i, k := range ks {
		before[i], _ = r.Get(k)
	}
	r.Add("engine-new")
	moved := 0
	for i, k := range ks {
		n, _ := r.Get(k)
		if n != before[i] {
			moved++
			if n != "engine-new" {
				t.Fatalf("key %s moved from %s to %s, but only the new node may gain keys", k, before[i], n)
			}
		}
	}
	p := arcShare(vn, append(nodes, "engine-new"))["engine-new"]
	mean, sd := K*p, math.Sqrt(K*p*(1-p))
	if math.Abs(float64(moved)-mean) > 4*sd {
		t.Fatalf("moved %d keys; the new node's arc share %.4f predicts %.0f +- %.0f (4 sd)", moved, p, mean, 4*sd)
	}
	if p < 0.5/9 || p > 1.5/9 {
		t.Fatalf("the new node's share is %.4f, far from 1/9 = %.4f with 160 vnodes", p, 1.0/9)
	}
}

func TestRemoveMovesOnlyItsKeys(t *testing.T) {
	// WHY: when a worker dies, only its own keys may move; everyone else's
	//      prefix cache stays warm.
	// KIND: property
	// CATCHES: s06
	// CHAPTER: ds.09 section 2.2
	nodes := names(5)
	r := build(40, nodes)
	ks := keys("key", 3000)
	before := make([]string, len(ks))
	for i, k := range ks {
		before[i], _ = r.Get(k)
	}
	r.Remove("engine-2")
	for i, k := range ks {
		n, _ := r.Get(k)
		if n == "engine-2" {
			t.Fatalf("key %s still maps to the removed engine-2", k)
		}
		if before[i] != "engine-2" && n != before[i] {
			t.Fatalf("key %s moved from %s to %s although %s is still a member", k, before[i], n, before[i])
		}
	}
}

func TestKeysSpreadByArcLength(t *testing.T) {
	// WHY: each node gets the keys that hash into its arcs; with 8 nodes and
	//      20000 keys the counts fit the arc shares (chi-square, 7 degrees of
	//      freedom, below 24.32, the p = 1e-3 critical value).
	// KIND: statistical
	// CATCHES: s07, s14
	// CHAPTER: ds.09 section 2.1
	const K, vn = 20000, 160
	nodes := names(8)
	r := build(vn, nodes)
	count := map[string]int{}
	for _, k := range keys("key", K) {
		n, _ := r.Get(k)
		count[n]++
	}
	share := arcShare(vn, nodes)
	chi := 0.0
	for _, n := range nodes {
		e := K * share[n]
		chi += (float64(count[n]) - e) * (float64(count[n]) - e) / e
	}
	if chi > 24.32 {
		t.Fatalf("chi-square %.2f > 24.32 (p < 1e-3): counts %v do not follow the arc shares %v", chi, count, share)
	}
}

func boundedRun(r *ring.Ring, ks [][]byte, c float64) (map[string]int, []string) {
	load := map[string]int{}
	owners := make([]string, len(ks))
	for i, k := range ks {
		n, _ := r.GetBounded(k, func(x string) int { return load[x] }, c)
		load[n]++
		owners[i] = n
	}
	return load, owners
}

func TestBoundedNeverExceedsCapacity(t *testing.T) {
	// WHY: the guarantee gw.05 relies on: placing keys one by one with
	//      GetBounded, no node ever holds more than ceil(c * K / n), even when
	//      every request carries the same hot prefix.
	// KIND: property
	// CATCHES: s04, s10
	// CHAPTER: ds.09 section 2.3
	nodes := names(5)
	r := build(40, nodes)
	for _, tc := range []struct {
		name string
		ks   [][]byte
	}{
		{"all-distinct", keys("key", 1000)},
		{"one-hot-key", func() [][]byte {
			out := make([][]byte, 1000)
			for i := range out {
				out[i] = []byte("hot")
			}
			return out
		}()},
		{"three-hot-keys", func() [][]byte {
			out := make([][]byte, 999)
			for i := range out {
				out[i] = []byte(fmt.Sprintf("hot-%d", i%3))
			}
			return out
		}()},
	} {
		for _, c := range []float64{1, 1.25, 2} {
			load, _ := boundedRun(r, tc.ks, c)
			limit := int(math.Ceil(c * float64(len(tc.ks)) / float64(len(nodes))))
			for n, l := range load {
				if l > limit {
					t.Fatalf("%s, c = %v: %s holds %d keys > ceil(c*K/n) = %d", tc.name, c, n, l, limit)
				}
			}
		}
	}
}

func TestBoundedEqualsGetWhenUnloaded(t *testing.T) {
	// WHY: bounded loads only redirects when the owner is full; with every
	//      load at zero it must agree with Get, or affinity is thrown away.
	// KIND: property
	// CHAPTER: ds.09 section 2.3
	r := build(40, names(5))
	for _, k := range keys("key", 2000) {
		want, _ := r.Get(k)
		got, ok := r.GetBounded(k, func(string) int { return 0 }, 1.25)
		if !ok || got != want {
			t.Fatalf("key %s: GetBounded = %s with no load, Get = %s", k, got, want)
		}
	}
}

func TestBoundedClampsCBelowOne(t *testing.T) {
	// WHY: c < 1 would make the capacity smaller than the average, so no
	//      placement could satisfy it; the contract treats it as c = 1.
	// KIND: boundary
	// CATCHES: s11
	// CHAPTER: ds.09 section 4
	r := build(40, names(4))
	ks := keys("hot", 200)
	_, one := boundedRun(r, ks, 1)
	_, half := boundedRun(r, ks, 0.5)
	for i := range ks {
		if one[i] != half[i] {
			t.Fatalf("key %d: c = 0.5 placed it on %s, c = 1 on %s; c < 1 must act as 1", i, half[i], one[i])
		}
	}
}

type goldenCase struct {
	Name  string `json:"name"`
	Input struct {
		Vnodes  int      `json:"vnodes"`
		Nodes   []string `json:"nodes"`
		Remove  []string `json:"remove"`
		Keys    []string `json:"keys"`
		Bounded *struct {
			C float64 `json:"c"`
		} `json:"bounded"`
	} `json:"input"`
	Output struct {
		Owners []string `json:"owners"`
	} `json:"output"`
}

func TestGoldenKeyToNodeMap(t *testing.T) {
	// WHY: parity/ring.hash: the Python reference map of the chapter's rules
	//      (labels "<node>#<i>", FNV-1a, ties by name, bounded walk) and the
	//      Go ring must give every key the same node.
	// KIND: golden, conformance
	// CATCHES: s07, s08
	// CHAPTER: ds.09 section 4
	dir := os.Getenv("TINYLLM_FIXTURES")
	if dir == "" {
		t.Fatal("TINYLLM_FIXTURES is not set (ol check sets it)")
	}
	b, err := os.ReadFile(filepath.Join(dir, "parity", "ring_hash.json"))
	if err != nil {
		t.Fatal(err)
	}
	var f struct {
		Cases []goldenCase `json:"cases"`
	}
	if err := json.Unmarshal(b, &f); err != nil || len(f.Cases) == 0 {
		t.Fatalf("parity/ring_hash.json: %v (%d cases)", err, len(f.Cases))
	}
	for _, c := range f.Cases {
		r := build(c.Input.Vnodes, c.Input.Nodes)
		for _, n := range c.Input.Remove {
			r.Remove(n)
		}
		load := map[string]int{}
		for i, k := range c.Input.Keys {
			var got string
			if c.Input.Bounded != nil {
				got, _ = r.GetBounded([]byte(k), func(x string) int { return load[x] }, c.Input.Bounded.C)
				load[got]++
			} else {
				got, _ = r.Get([]byte(k))
			}
			if got != c.Output.Owners[i] {
				t.Fatalf("case %s key %q: %s, golden %s", c.Name, k, got, c.Output.Owners[i])
			}
		}
	}
}
