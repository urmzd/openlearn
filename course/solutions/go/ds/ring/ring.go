// Package ring is a consistent hash ring with virtual nodes and the bounded
// loads variant (ds.09): the gateway's prefix-affinity router (gw.05) keys a
// request by its first prompt block, so requests that share a prefix land on
// the replica whose prefix cache already holds it, and no replica takes more
// than ceil(c * average) of the in-flight work.
//
// Chapter: content/courses/algorithms/13-systems-data-structures/09-consistent-hash-ring.md.
// Conformance: parity/ring.hash (golden key-to-node map computed with the
// Python reference; generator not included).
//
// The ring is not safe for concurrent use; gw.05 guards it with its own lock.
package ring

import (
	"math"
	"sort"
	"strconv"
)

// Hash maps bytes to a position on the ring, [0, 2^64).
type Hash func([]byte) uint64

// point is one virtual node: replica `node`, copy `idx`, at position `pos`.
type point struct {
	pos  uint64
	node string
	idx  int
}

// Ring is a set of nodes, each placed at `vnodes` positions.
type Ring struct {
	vnodes int
	h      Hash
	points []point         // sorted by (pos, node, idx)
	nodes  map[string]bool // the members
}

// FNV-1a 64-bit parameters (the same hash as the KV block hash, D13).
const (
	fnvOffset uint64 = 14695981039346656037
	fnvPrime  uint64 = 1099511628211
)

// FNV1a64 is FNV-1a over the bytes, 64-bit: easy to write in any language,
// but on short labels that differ only in their last bytes ("engine-0#7",
// "engine-0#8") its output clusters, so the ring finalizes it (Hash64).
func FNV1a64(b []byte) uint64 {
	h := fnvOffset
	for _, c := range b {
		h ^= uint64(c)
		h *= fnvPrime
	}
	return h
}

// Mix64 is the SplitMix64 finalizer (the mix of M06.3 and ds.08): every
// input bit affects every output bit, which spreads FNV's clustered outputs
// over the whole ring.
func Mix64(z uint64) uint64 {
	z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9
	z = (z ^ (z >> 27)) * 0x94d049bb133111eb
	return z ^ (z >> 31)
}

// Hash64 is the ring's default hash: Mix64(FNV1a64(b)).
func Hash64(b []byte) uint64 {
	return Mix64(FNV1a64(b))
}

// VnodeLabel is the byte string hashed to place copy idx of node: the node
// name, '#', and idx in decimal ("engine-a#0", "engine-a#1", ...).
func VnodeLabel(node string, idx int) []byte {
	return []byte(node + "#" + strconv.Itoa(idx))
}

// New returns an empty ring that places every node at vnodes positions,
// hashed with h (nil means Hash64). vnodes < 1 panics: a node with no
// position would be a member that never receives a key.
func New(vnodes int, h func([]byte) uint64) *Ring {
	if vnodes < 1 {
		panic("ring: vnodes must be at least 1")
	}
	if h == nil {
		h = Hash64
	}
	return &Ring{vnodes: vnodes, h: h, nodes: map[string]bool{}}
}

// less orders points by position, then node name, then copy index, so the
// ring is the same whatever order nodes were added in, even when two
// positions collide.
func less(a, b point) bool {
	if a.pos != b.pos {
		return a.pos < b.pos
	}
	if a.node != b.node {
		return a.node < b.node
	}
	return a.idx < b.idx
}

// Add places node on the ring. Adding a member again changes nothing.
func (r *Ring) Add(node string) {
	if r.nodes[node] {
		return
	}
	r.nodes[node] = true
	for i := 0; i < r.vnodes; i++ {
		r.points = append(r.points, point{pos: r.h(VnodeLabel(node, i)), node: node, idx: i})
	}
	sort.Slice(r.points, func(i, j int) bool { return less(r.points[i], r.points[j]) })
}

// Remove takes node off the ring. Removing a non-member changes nothing.
func (r *Ring) Remove(node string) {
	if !r.nodes[node] {
		return
	}
	delete(r.nodes, node)
	kept := r.points[:0]
	for _, p := range r.points {
		if p.node != node {
			kept = append(kept, p)
		}
	}
	r.points = kept
}

// Len is the number of member nodes.
func (r *Ring) Len() int {
	return len(r.nodes)
}

// Nodes lists the members in ascending order.
func (r *Ring) Nodes() []string {
	out := make([]string, 0, len(r.nodes))
	for n := range r.nodes {
		out = append(out, n)
	}
	sort.Strings(out)
	return out
}

// first is the index of the first point at or clockwise after position x,
// wrapping past the top of the ring to index 0.
func (r *Ring) first(x uint64) int {
	i := sort.Search(len(r.points), func(i int) bool { return r.points[i].pos >= x })
	if i == len(r.points) {
		return 0
	}
	return i
}

// Get returns the owner of key: the node of the first virtual node at or
// clockwise after h(key). ok is false on an empty ring.
func (r *Ring) Get(key []byte) (node string, ok bool) {
	if len(r.points) == 0 {
		return "", false
	}
	return r.points[r.first(r.h(key))].node, true
}

// Capacity is the bounded-loads limit for placing one more key when `total`
// keys are already placed on n nodes: ceil(c * (total + 1) / n). The +1
// counts the key being placed, so with c >= 1 some node is always below it.
func Capacity(total, n int, c float64) int {
	if n <= 0 {
		return 0
	}
	return int(math.Ceil(c * float64(total+1) / float64(n)))
}

// GetBounded is Get with bounded loads: walking clockwise from h(key), it
// returns the first node whose current load is below
// Capacity(sum of loads, Len(), c). load reports a node's current load (for
// gw.05, the requests in flight on that replica plus its queue depth). c < 1
// is treated as 1. ok is false on an empty ring.
func (r *Ring) GetBounded(key []byte, load func(node string) int, c float64) (node string, ok bool) {
	if len(r.points) == 0 {
		return "", false
	}
	if c < 1 {
		c = 1
	}
	total := 0
	for n := range r.nodes {
		total += load(n)
	}
	limit := Capacity(total, len(r.nodes), c)
	start := r.first(r.h(key))
	seen := make(map[string]bool, len(r.nodes))
	for k := 0; k < len(r.points) && len(seen) < len(r.nodes); k++ {
		p := r.points[(start+k)%len(r.points)]
		if seen[p.node] {
			continue
		}
		seen[p.node] = true
		if load(p.node) < limit {
			return p.node, true
		}
	}
	// Unreachable for c >= 1 (some node is below the average); keep the
	// plain owner rather than failing a request.
	return r.points[start].node, true
}
