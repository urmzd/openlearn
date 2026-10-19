// Generics in Go (1.18+) -- type parameters, constraints, and type sets.
//
// Go was deliberately monomorphic until 1.18 (2022). The design that landed is
// distinctive: constraints are *interfaces*, but extended so an interface can
// list a SET OF TYPES (a "type set"), not just a set of methods. A type
// parameter `T C` ranges over exactly the types in C's type set.
//
// Under the hood the compiler uses "GC shape stenciling": it generates one
// copy of a generic function per distinct *memory layout* (pointer-shaped
// types share a single copy and dispatch through a runtime dictionary), a
// middle point between C++'s full monomorphization (a copy per type, fast but
// code-bloated) and Java's type erasure (one copy, boxed, slower).
//
//   Run:  go run generics.go
//   Vet:  go vet generics.go
package main

import (
	"fmt"
	"strings"
)

// --------------------------------------------------------------------------- //
// 1. Parametric polymorphism: type parameters                                 //
// --------------------------------------------------------------------------- //

// Identity ranges over every type: `any` is the constraint with the largest
// possible type set (it is an alias for interface{}).
func Identity[T any](x T) T { return x }

// Map is the canonical parametric function: two type params, one body, works
// for every (T, U). The compiler infers T and U from the arguments.
func Map[T, U any](xs []T, f func(T) U) []U {
	out := make([]U, len(xs))
	for i, x := range xs {
		out[i] = f(x)
	}
	return out
}

// --------------------------------------------------------------------------- //
// 2. Constraints as type sets: the part unique to Go                          //
// --------------------------------------------------------------------------- //

// Numeric is a constraint whose type set is a UNION of concrete types. The `~`
// means "and any type whose underlying type is this" -- so a `type Celsius
// float64` still satisfies it. This is how `Sum` can use `+` (the operator is
// defined for every type in the set) without any method dispatch.
type Numeric interface {
	~int | ~int64 | ~float64
}

func Sum[T Numeric](xs []T) T {
	var total T // the zero value of T
	for _, x := range xs {
		total += x // legal: every type in Numeric supports +
	}
	return total
}

// Ordered constrains to types that support the < operator -- a smaller union,
// reused by Max below. (The standard library ships this as cmp.Ordered.)
type Ordered interface {
	~int | ~int64 | ~float64 | ~string
}

// 3. Bounded quantification: Max works for any Ordered type.
func Max[T Ordered](xs []T) T {
	best := xs[0]
	for _, x := range xs[1:] {
		if x > best {
			best = x
		}
	}
	return best
}

// --------------------------------------------------------------------------- //
// 4. Generic data structures: a type-safe Stack[T]                            //
// --------------------------------------------------------------------------- //

type Stack[T any] struct {
	items []T
}

func (s *Stack[T]) Push(x T) { s.items = append(s.items, x) }

func (s *Stack[T]) Pop() (T, bool) {
	var zero T
	if len(s.items) == 0 {
		return zero, false
	}
	x := s.items[len(s.items)-1]
	s.items = s.items[:len(s.items)-1]
	return x, true
}

// --------------------------------------------------------------------------- //
// 5. Method constraints + subtype-style dispatch via interfaces               //
// --------------------------------------------------------------------------- //

// A constraint can also require METHODS (ad-hoc polymorphism): any T that has
// Area() float64 works here. This blends the generic and interface worlds.
type Shape interface {
	Area() float64
}

type Circle struct{ R float64 }

func (c Circle) Area() float64 { return 3.141592653589793 * c.R * c.R }

type Rectangle struct{ W, H float64 }

func (r Rectangle) Area() float64 { return r.W * r.H }

// TotalArea is generic over any Shape; the call sites monomorphize per type.
func TotalArea[T Shape](shapes []T) float64 {
	var total float64
	for _, s := range shapes {
		total += s.Area()
	}
	return total
}

func main() {
	fmt.Println("1. Parametric: Identity + Map reuse one body")
	fmt.Printf("   Identity(42) = %d\n", Identity(42))
	doubled := Map([]int{1, 2, 3}, func(x int) int { return x * 2 })
	lengths := Map([]string{"go", "rust"}, func(s string) int { return len(s) })
	fmt.Printf("   Map double  = %v\n   Map len     = %v\n", doubled, lengths)

	fmt.Println("\n2. Type sets: Sum uses + across a union of numeric types")
	fmt.Printf("   Sum([]int)     = %d\n", Sum([]int{1, 2, 3, 4}))
	fmt.Printf("   Sum([]float64) = %.1f\n", Sum([]float64{1.5, 2.5}))

	fmt.Println("\n3. Bounded: Max over any Ordered type")
	fmt.Printf("   Max([]int)    = %d\n", Max([]int{3, 9, 2, 7}))
	fmt.Printf("   Max([]string) = %q\n", Max([]string{"go", "zig", "rust"}))

	fmt.Println("\n4. Generic Stack[T]: type-safe, no boxing, no casts")
	var s Stack[string]
	s.Push("a")
	s.Push("b")
	top, _ := s.Pop()
	fmt.Printf("   pushed a,b -> pop = %q\n", top)

	fmt.Println("\n5. Method constraint: TotalArea over a Shape type set")
	circles := []Circle{{R: 1}, {R: 2}}
	fmt.Printf("   TotalArea(circles) = %.3f\n", TotalArea(circles))

	fmt.Printf("\n%s\nOK\n", strings.Repeat("-", 4))
}
