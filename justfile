# justfile for openlearn
# Use `just --parallel <recipe> ...` to run multiple recipes concurrently.

set shell := ["bash", "-cu"]

help:
  @just --list

# Verb-noun style entrypoints
run target:
  @just "run-{{target}}"

run-parallel +targets:
  @recipes=""
  @for t in {{targets}}; do recipes="$recipes run-$t"; done
  @just --parallel $recipes

# ---------- Per-folder run recipes (verb-noun style) ----------

run-01-arrays-hashing:
  command -v node >/dev/null
  node algorithms/01-arrays-hashing/contains-duplicate.js
  node algorithms/01-arrays-hashing/missing-number.js
  node algorithms/01-arrays-hashing/product-of-array-expect-self.js
  node algorithms/01-arrays-hashing/two-sum.js

run-02-two-pointers-sliding-window:
  command -v node >/dev/null
  node algorithms/02-two-pointers-sliding-window/3sum.js
  node algorithms/02-two-pointers-sliding-window/container-with-most-water.js

run-03-binary-search:
  command -v node >/dev/null
  node algorithms/03-binary-search/find-minimum-in-rotated-sorted-array.js
  node algorithms/03-binary-search/liss.js
  node algorithms/03-binary-search/search-in-rotated-sorted-array.js

run-04-linked-lists:
  command -v node >/dev/null
  node algorithms/04-linked-lists/add-two-numbers.js

run-05-trees:
  command -v python3 >/dev/null
  python3 algorithms/05-trees/example-7.py

run-06-graphs:
  command -v node >/dev/null
  node algorithms/06-graphs/clone-graph.js
  node algorithms/06-graphs/course-schedule.js
  node algorithms/06-graphs/number-of-islands.js
  if command -v ts-node >/dev/null; then \
    ts-node algorithms/06-graphs/pacific-atlantic-water-flow.ts; \
  else \
    echo "ts-node not installed; run: ts-node algorithms/06-graphs/pacific-atlantic-water-flow.ts"; \
  fi

run-07-dynamic-programming:
  command -v node >/dev/null
  node algorithms/07-dynamic-programming/change-coin.js
  node algorithms/07-dynamic-programming/climbing-stairs.js
  node algorithms/07-dynamic-programming/decode-ways.js
  node algorithms/07-dynamic-programming/house-robber.js
  node algorithms/07-dynamic-programming/lcs.js
  node algorithms/07-dynamic-programming/maximum-product-subarray.js
  node algorithms/07-dynamic-programming/maximum-subarray.js
  node algorithms/07-dynamic-programming/rob-houses-pt-2.js
  node algorithms/07-dynamic-programming/unique-paths.js
  node algorithms/07-dynamic-programming/word-break.js
  command -v python3 >/dev/null
  python3 algorithms/07-dynamic-programming/example-1.py
  python3 algorithms/07-dynamic-programming/example-2.py
  python3 algorithms/07-dynamic-programming/example-3.py
  python3 algorithms/07-dynamic-programming/example-4.py
  python3 algorithms/07-dynamic-programming/example-6.py
  # C examples require input; compile/run manually if desired.
  echo "C examples: cc algorithms/07-dynamic-programming/example-5.c -o /tmp/example-5 && /tmp/example-5"
  echo "C examples (requires input): cc algorithms/07-dynamic-programming/example-8.c -o /tmp/example-8 && /tmp/example-8"

run-08-greedy:
  command -v node >/dev/null
  node algorithms/08-greedy/best-time-to-buy-and-sell-stock.js
  node algorithms/08-greedy/jump-game.js
  command -v python3 >/dev/null
  python3 algorithms/08-greedy/example-1.py
  python3 algorithms/08-greedy/example-2.py
  python3 algorithms/08-greedy/example-3.py

run-09-backtracking:
  command -v node >/dev/null
  node algorithms/09-backtracking/combination-sum.js
  command -v python3 >/dev/null
  python3 algorithms/09-backtracking/example-2.py

run-10-math-bit:
  command -v node >/dev/null
  node algorithms/10-math-bit/count-number-of-bits.js
  node algorithms/10-math-bit/number-of-1-bits.js
  node algorithms/10-math-bit/reverse-bits.js
  node algorithms/10-math-bit/sum-of-two-integers.js

run-11-recursion-divide-conquer:
  command -v python3 >/dev/null
  python3 algorithms/11-recursion-divide-conquer/example-1.py
  python3 algorithms/11-recursion-divide-conquer/example-3.py

run-15-probabilistic-structures:
  command -v python3 >/dev/null
  python3 algorithms/15-probabilistic-structures/bloom.py

run-predict:
  practice/bin/ol verify predict

run-predict-lang lang:
  practice/bin/ol verify predict {{lang}}

run-predict-bench:
  practice/bin/ol bench --assert

# Verify every core build reference (zig/scala/java are optional; see run-build-lang)
run-build:
  practice/bin/ol verify build

# Verify one language by name, including the optional ones
run-build-lang lang:
  practice/bin/ol verify build {{lang}}

# Where the harness thinks everything is, and how many exercises each kind has.
run-practice-status:
  practice/bin/ol where

# ---------- Course (course/DESIGN.md) ----------

# Maintainer checks for the course registry, references, tests, and mutants.
course-verify:
  practice/bin/ol verify course

# Registry invariants, chapter contract, and every markdown link in the repo.
course-lint:
  practice/bin/ol lint --links

# Run one milestone against your system, e.g. `just milestone MS-P1`.
milestone M:
  practice/bin/ol milestone {{M}}

# Start a drill against your kind cluster, e.g. `just drill engine-crashloop`.
drill D:
  practice/bin/ol drill start {{D}}

# ---------- Aggregate recipes ----------

run-all: \
  run-01-arrays-hashing \
  run-02-two-pointers-sliding-window \
  run-03-binary-search \
  run-04-linked-lists \
  run-05-trees \
  run-06-graphs \
  run-07-dynamic-programming \
  run-08-greedy \
  run-09-backtracking \
  run-10-math-bit \
  run-11-recursion-divide-conquer \
  run-15-probabilistic-structures

run-all-parallel:
  @just --parallel \
    run-01-arrays-hashing \
    run-02-two-pointers-sliding-window \
    run-03-binary-search \
    run-04-linked-lists \
    run-05-trees \
    run-06-graphs \
    run-07-dynamic-programming \
    run-08-greedy \
    run-09-backtracking \
    run-10-math-bit \
    run-11-recursion-divide-conquer \
              run-15-probabilistic-structures
