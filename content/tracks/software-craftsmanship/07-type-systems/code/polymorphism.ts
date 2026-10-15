// The polymorphism taxonomy in TypeScript -- a STRUCTURAL type system.
//
// TypeScript types describe the *shape* of values, not their names. A value
// fits a type if it has the right members, regardless of what class produced
// it ("duck typing, checked at compile time"). Contrast Rust/Haskell/Java,
// which are nominal: a type matches only if you named that type.
//
// We walk Christopher Strachey's two axes of polymorphism, then Cardinelli's
// refinements (subtype + bounded), then the things only a structural system
// gives you: structural subtyping and type-level programming.
//
//   Typecheck:  npx -y typescript tsc --strict --noEmit polymorphism.ts
//   Run:        node polymorphism.ts        (Node 22+ strips types and runs)

// --------------------------------------------------------------------------- //
// 1. Parametric polymorphism: one implementation, every type                  //
// --------------------------------------------------------------------------- //
// The function cannot inspect T, so it must behave identically for all T --
// Reynolds' "parametricity". `identity<T>` literally cannot do anything but
// return its argument; its type `<T>(x: T) => T` has exactly one inhabitant.

function identity<T>(x: T): T {
  return x;
}

// A generic container: Stack<T> reuses one body for numbers, strings, anything.
class Stack<T> {
  private items: T[] = [];
  push(x: T): void {
    this.items.push(x);
  }
  pop(): T | undefined {
    return this.items.pop();
  }
  get size(): number {
    return this.items.length;
  }
}

// --------------------------------------------------------------------------- //
// 2. Ad-hoc polymorphism: different code per type (overloading / dispatch)    //
// --------------------------------------------------------------------------- //
// `area` does something *different* for each shape. TypeScript models the
// alternatives as a discriminated union and dispatches on the `kind` tag --
// the structural-typing analogue of a Haskell typeclass instance or a Rust
// trait impl. The `never` in the default branch makes the match exhaustive:
// add a shape and forget a case, and the compiler rejects the file.

type Shape =
  | { kind: "circle"; radius: number }
  | { kind: "rectangle"; width: number; height: number }
  | { kind: "triangle"; base: number; height: number };

function area(shape: Shape): number {
  switch (shape.kind) {
    case "circle":
      return Math.PI * shape.radius ** 2;
    case "rectangle":
      return shape.width * shape.height;
    case "triangle":
      return 0.5 * shape.base * shape.height;
    default: {
      const _exhaustive: never = shape;
      return _exhaustive;
    }
  }
}

// --------------------------------------------------------------------------- //
// 3. Subtype polymorphism: a Dog is usable wherever an Animal is expected     //
// --------------------------------------------------------------------------- //
// Liskov substitution. Because TypeScript is structural, Dog is a subtype of
// Animal purely by HAVING everything Animal has -- no `extends` needed.

interface Animal {
  name: string;
  speak(): string;
}

interface Dog {
  name: string;
  speak(): string;
  fetch(): string; // extra members are fine: a wider shape is a subtype
}

function greet(a: Animal): string {
  return `${a.name} says ${a.speak()}`;
}

const rex: Dog = {
  name: "Rex",
  speak: () => "woof",
  fetch: () => "got the ball",
};

// --------------------------------------------------------------------------- //
// 4. Bounded quantification: parametric + a constraint (F-bounded)            //
// --------------------------------------------------------------------------- //
// `largest` is generic, but only over types that have a `.valueOf(): number`.
// This is parametric polymorphism intersected with a subtype bound -- the
// "bounded quantification" of System F-sub, and exactly what Go's constraints
// and Rust's trait bounds (`T: Ord`) express.

function largest<T extends { valueOf(): number }>(xs: readonly T[]): T {
  return xs.reduce((best, x) => (x.valueOf() > best.valueOf() ? x : best));
}

// --------------------------------------------------------------------------- //
// 5. Variance: when is Container<Sub> a subtype of Container<Super>?          //
// --------------------------------------------------------------------------- //
// Covariance (outputs): a producer of Dogs is a producer of Animals.
// Contravariance (inputs): a consumer of Animals is a consumer of Dogs.
// Function types are contravariant in their parameter and covariant in their
// return -- the single most counterintuitive rule in type systems. Under
// `strictFunctionTypes`, TypeScript enforces it for function *types*.

type Producer<T> = () => T; // covariant in T
type Consumer<T> = (x: T) => void; // contravariant in T

const dogProducer: Producer<Dog> = () => rex;
const animalProducer: Producer<Animal> = dogProducer; // OK: covariant

const animalConsumer: Consumer<Animal> = (a) => void a.speak();
const dogConsumer: Consumer<Dog> = animalConsumer; // OK: contravariant
// const bad: Consumer<Animal> = dogConsumer;       // ERROR: would call fetch()

// --------------------------------------------------------------------------- //
// 6. Type-level programming: types that compute (a structural-system perk)    //
// --------------------------------------------------------------------------- //
// Conditional + mapped types make the type checker a small pure functional
// language. `DeepReadonly<T>` recurses over a type to freeze it at every depth
// -- a proof the system is doing real computation, not just shape-matching.

type DeepReadonly<T> = T extends (...args: never[]) => unknown
  ? T
  : T extends object
    ? { readonly [K in keyof T]: DeepReadonly<T[K]> }
    : T;

interface Config {
  server: { host: string; ports: number[] };
}
const frozen: DeepReadonly<Config> = {
  server: { host: "localhost", ports: [8080] },
};
// frozen.server.host = "x";   // ERROR: readonly all the way down

// --------------------------------------------------------------------------- //
// Demo                                                                        //
// --------------------------------------------------------------------------- //

function main(): void {
  console.log("1. Parametric: identity reuses one body");
  console.log(`   identity(42) = ${identity(42)}, identity("hi") = ${identity("hi")}`);
  const s = new Stack<number>();
  s.push(1);
  s.push(2);
  console.log(`   Stack<number> size=${s.size}, pop=${s.pop()}`);

  console.log("\n2. Ad-hoc: area dispatches per shape");
  const shapes: Shape[] = [
    { kind: "circle", radius: 2 },
    { kind: "rectangle", width: 3, height: 4 },
    { kind: "triangle", base: 6, height: 2 },
  ];
  for (const sh of shapes) {
    console.log(`   ${sh.kind.padEnd(9)} area = ${area(sh).toFixed(3)}`);
  }

  console.log("\n3. Subtype: a Dog flows where an Animal is wanted");
  console.log(`   ${greet(rex)}`); // structural: no `implements` needed

  console.log("\n4. Bounded: largest over anything with valueOf()");
  console.log(`   largest([3, 9, 2, 7]) = ${largest([3, 9, 2, 7])}`);

  console.log("\n5. Variance: contravariant consumer accepted");
  dogConsumer(rex);
  console.log(`   animalProducer() -> ${animalProducer().name}`);

  console.log("\n6. Type-level: DeepReadonly froze a nested config");
  console.log(`   frozen.server.ports = [${frozen.server.ports.join(", ")}]`);

  console.log("\nOK");
}

main();
