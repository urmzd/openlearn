// Traits in Rust -- ad-hoc polymorphism done as a NOMINAL type system, plus
// the static/dynamic dispatch split that defines Rust's performance model.
//
// A trait is Rust's typeclass (Haskell) / interface (Go/Java): a named set of
// methods a type can implement. Unlike Go's structural interfaces, you must
// `impl Trait for Type` explicitly -- matching is by name, not by shape.
//
// The crucial axis is dispatch:
//   * generics `<T: Trait>`     -> STATIC dispatch, monomorphized: the compiler
//                                  stamps out one specialized copy per concrete
//                                  type. Zero runtime cost, larger binary.
//   * trait objects `dyn Trait` -> DYNAMIC dispatch via a vtable: one copy,
//                                  a pointer + method table, runtime lookup.
//                                  This is how Rust expresses subtype-style
//                                  "a heterogeneous list of Shapes".
//
//   Run:    rustc traits.rs -o /tmp/traits && /tmp/traits
//   Check:  rustc --edition 2021 traits.rs --crate-type bin -o /tmp/traits

// --------------------------------------------------------------------------- //
// 1. Ad-hoc polymorphism: a trait with per-type implementations               //
// --------------------------------------------------------------------------- //

trait Shape {
    fn area(&self) -> f64;
    // A default method: provided unless a type overrides it.
    fn describe(&self) -> String {
        format!("a shape of area {:.3}", self.area())
    }
}

struct Circle {
    radius: f64,
}
struct Rectangle {
    width: f64,
    height: f64,
}

impl Shape for Circle {
    fn area(&self) -> f64 {
        std::f64::consts::PI * self.radius * self.radius
    }
}

impl Shape for Rectangle {
    fn area(&self) -> f64 {
        self.width * self.height
    }
    fn describe(&self) -> String {
        format!("a {}x{} rectangle", self.width, self.height)
    }
}

// --------------------------------------------------------------------------- //
// 2. Static dispatch: bounded generics, monomorphized at compile time         //
// --------------------------------------------------------------------------- //
// `T: Shape` is bounded quantification. The compiler generates a separate
// `total_area` for Circle and for Rectangle -- each call is a direct, often
// inlined, function call. No vtable, no indirection.

fn total_area<T: Shape>(shapes: &[T]) -> f64 {
    shapes.iter().map(|s| s.area()).sum()
}

// --------------------------------------------------------------------------- //
// 3. Dynamic dispatch: trait objects for heterogeneous collections            //
// --------------------------------------------------------------------------- //
// A `Vec<Box<dyn Shape>>` holds DIFFERENT concrete types behind one interface
// -- the closest Rust gets to classic subtype polymorphism. Each element is a
// fat pointer (data ptr + vtable ptr); `area()` is a runtime vtable lookup.

fn describe_all(shapes: &[Box<dyn Shape>]) -> Vec<String> {
    shapes.iter().map(|s| s.describe()).collect()
}

// --------------------------------------------------------------------------- //
// 4. Parametric polymorphism + associated types: a generic container          //
// --------------------------------------------------------------------------- //

struct Stack<T> {
    items: Vec<T>,
}

impl<T> Stack<T> {
    fn new() -> Self {
        Stack { items: Vec::new() }
    }
    fn push(&mut self, x: T) {
        self.items.push(x);
    }
    fn pop(&mut self) -> Option<T> {
        self.items.pop()
    }
}

// Associated types: the Iterator trait fixes ONE output type per impl (`Item`),
// unlike a type parameter the caller chooses. This Counter yields u32s.
struct Counter {
    n: u32,
    max: u32,
}

impl Iterator for Counter {
    type Item = u32;
    fn next(&mut self) -> Option<u32> {
        if self.n < self.max {
            self.n += 1;
            Some(self.n)
        } else {
            None
        }
    }
}

// --------------------------------------------------------------------------- //
// 5. Bounded generics with a standard trait: max over anything Ordered        //
// --------------------------------------------------------------------------- //
// `T: PartialOrd + Copy` is the trait-bound spelling of "T is comparable".

fn largest<T: PartialOrd + Copy>(xs: &[T]) -> T {
    let mut best = xs[0];
    for &x in &xs[1..] {
        if x > best {
            best = x;
        }
    }
    best
}

fn main() {
    println!("1. Ad-hoc trait impls (default method overridden by Rectangle)");
    let c = Circle { radius: 2.0 };
    let r = Rectangle {
        width: 3.0,
        height: 4.0,
    };
    println!("   circle:    {}", c.describe());
    println!("   rectangle: {}", r.describe());

    println!("\n2. Static dispatch: total_area monomorphized per type");
    let circles = [Circle { radius: 1.0 }, Circle { radius: 2.0 }];
    println!("   total_area(circles) = {:.3}", total_area(&circles));

    println!("\n3. Dynamic dispatch: Vec<Box<dyn Shape>> via vtables");
    let shapes: Vec<Box<dyn Shape>> = vec![
        Box::new(Circle { radius: 1.0 }),
        Box::new(Rectangle {
            width: 2.0,
            height: 5.0,
        }),
    ];
    for d in describe_all(&shapes) {
        println!("   {d}");
    }

    println!("\n4. Parametric Stack<T> + associated-type Iterator");
    let mut s: Stack<&str> = Stack::new();
    s.push("a");
    s.push("b");
    println!("   pop = {:?}", s.pop());
    let collected: Vec<u32> = Counter { n: 0, max: 5 }.collect();
    println!("   Counter -> {collected:?}");

    println!("\n5. Bounded generic: largest over PartialOrd");
    println!("   largest([3, 9, 2, 7]) = {}", largest(&[3, 9, 2, 7]));
    println!("   largest([1.5, 0.5])   = {}", largest(&[1.5, 0.5]));

    println!("\nOK");
}
