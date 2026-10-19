// Package rng is the Go port of the course's one random generator, PCG32
// (load.01), so a seed means the same stream in Python training, the Rust
// sampler, and the Go load generator.
//
// Spec: course/contracts/spec/pcg32.md (PCG-XSH-RR 64/32 as in O'Neill's
// pcg32_random_r, uniform_f64 from two draws, Box-Muller normals, unbiased
// bounded ints, Fisher-Yates, SplitMix64 sub-streams). Conformance:
// parity/rng (the Go driver calls this package).
// Chapter: content/courses/build-an-llm/11-serving/11-load-generator.md.
package rng

import "math"

// mult is the LCG multiplier of PCG32.
const mult uint64 = 6364136223846793005

// DefaultSeq is the stream of pcg32(seed): pcg32_srandom_r(seed, 54).
const DefaultSeq uint64 = 54

// Purpose ids of spec/pcg32.md (sub-streams).
const (
	PurposeInit     uint64 = 1
	PurposeDropout  uint64 = 2
	PurposeShuffle  uint64 = 3
	PurposeSample   uint64 = 4
	PurposeMutation uint64 = 5
)

// PCG32 is one generator. It is not safe for concurrent use: give each
// goroutine its own (Stream derives independent ones from one seed).
type PCG32 struct {
	state, inc uint64
	spare      float64
	hasSpare   bool
}

// New is pcg32_srandom_r(seed, seq): inc = (seq << 1) | 1, state = 0, one
// step, state += seed, one step.
func New(seed, seq uint64) *PCG32 {
	r := &PCG32{inc: seq<<1 | 1}
	r.Uint32()
	r.state += seed
	r.Uint32()
	return r
}

// Seeded is pcg32(seed), the default stream 54.
func Seeded(seed uint64) *PCG32 {
	return New(seed, DefaultSeq)
}

// Uint32 is one step; the output is a permutation of the OLD state.
func (r *PCG32) Uint32() uint32 {
	old := r.state
	r.state = old*mult + r.inc
	xs := uint32(((old >> 18) ^ old) >> 27)
	rot := uint32(old >> 59)
	return xs>>rot | xs<<((-rot)&31)
}

// Float64 is uniform_f64: a = Uint32() >> 5, b = Uint32() >> 6, then
// (a * 2^26 + b) * 2^-53, uniform on [0, 1) with 53 random bits.
func (r *PCG32) Float64() float64 {
	a := uint64(r.Uint32() >> 5)
	b := uint64(r.Uint32() >> 6)
	return float64(a<<26+b) * (1.0 / 9007199254740992.0)
}

// Normal is one standard normal by Box-Muller: without a spare, u1 and u2
// from Float64, r = sqrt(-2 ln(1 - u1)); return r cos(2 pi u2) and keep
// r sin(2 pi u2) for the next call.
func (r *PCG32) Normal() float64 {
	if r.hasSpare {
		r.hasSpare = false
		return r.spare
	}
	u1, u2 := r.Float64(), r.Float64()
	rad := math.Sqrt(-2.0 * math.Log(1.0-u1))
	r.spare = rad * math.Sin(2.0*math.Pi*u2)
	r.hasSpare = true
	return rad * math.Cos(2.0*math.Pi*u2)
}

// Below is an unbiased draw from [0, n) for 1 <= n <= 2^32: t = (2^32 - n)
// mod n, draw until r >= t, return r mod n. It panics for n outside that
// range.
func (r *PCG32) Below(n uint64) uint32 {
	if n < 1 || n > 1<<32 {
		panic("rng: Below needs 1 <= n <= 2^32")
	}
	t := ((1 << 32) - n) % n
	for {
		x := uint64(r.Uint32())
		if x >= t {
			return uint32(x % n)
		}
	}
}

// Shuffle is Fisher-Yates from the end: for i = n-1 down to 1, j =
// Below(i+1), swap(i, j).
func (r *PCG32) Shuffle(n int, swap func(i, j int)) {
	for i := n - 1; i >= 1; i-- {
		j := int(r.Below(uint64(i + 1)))
		swap(i, j)
	}
}

// State returns (state, inc), for tests and hand-offs.
func (r *PCG32) State() (state, inc uint64) {
	return r.state, r.inc
}

// Mix64 is SplitMix64's finalizer.
func Mix64(z uint64) uint64 {
	z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
	z = (z ^ (z >> 27)) * 0x94D049BB133111EB
	return z ^ (z >> 31)
}

// ChildSeed is mix64(seed + purpose * 0x9E3779B97F4A7C15).
func ChildSeed(seed, purpose uint64) uint64 {
	return Mix64(seed + purpose*0x9E3779B97F4A7C15)
}

// Stream is pcg32_srandom_r(ChildSeed(seed, purpose), purpose).
func Stream(seed, purpose uint64) *PCG32 {
	return New(ChildSeed(seed, purpose), purpose)
}
