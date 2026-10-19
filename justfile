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
  node content/courses/algorithms/01-arrays-hashing/contains-duplicate.js
  node content/courses/algorithms/01-arrays-hashing/missing-number.js
  node content/courses/algorithms/01-arrays-hashing/product-of-array-expect-self.js
  node content/courses/algorithms/01-arrays-hashing/two-sum.js

run-02-two-pointers-sliding-window:
  command -v node >/dev/null
  node content/courses/algorithms/02-two-pointers-sliding-window/3sum.js
  node content/courses/algorithms/02-two-pointers-sliding-window/container-with-most-water.js

run-03-binary-search:
  command -v node >/dev/null
  node content/courses/algorithms/03-binary-search/find-minimum-in-rotated-sorted-array.js
  node content/courses/algorithms/03-binary-search/liss.js
  node content/courses/algorithms/03-binary-search/search-in-rotated-sorted-array.js

run-04-linked-lists:
  command -v node >/dev/null
  node content/courses/algorithms/04-linked-lists/add-two-numbers.js

run-05-trees:
  command -v python3 >/dev/null
  python3 content/courses/algorithms/05-trees/example-7.py

run-06-graphs:
  command -v node >/dev/null
  node content/courses/algorithms/06-graphs/clone-graph.js
  node content/courses/algorithms/06-graphs/course-schedule.js
  node content/courses/algorithms/06-graphs/number-of-islands.js
  if command -v ts-node >/dev/null; then \
    ts-node content/courses/algorithms/06-graphs/pacific-atlantic-water-flow.ts; \
  else \
    echo "ts-node not installed; run: ts-node content/courses/algorithms/06-graphs/pacific-atlantic-water-flow.ts"; \
  fi

run-07-dynamic-programming:
  command -v node >/dev/null
  node content/courses/algorithms/07-dynamic-programming/change-coin.js
  node content/courses/algorithms/07-dynamic-programming/climbing-stairs.js
  node content/courses/algorithms/07-dynamic-programming/decode-ways.js
  node content/courses/algorithms/07-dynamic-programming/house-robber.js
  node content/courses/algorithms/07-dynamic-programming/lcs.js
  node content/courses/algorithms/07-dynamic-programming/maximum-product-subarray.js
  node content/courses/algorithms/07-dynamic-programming/maximum-subarray.js
  node content/courses/algorithms/07-dynamic-programming/rob-houses-pt-2.js
  node content/courses/algorithms/07-dynamic-programming/unique-paths.js
  node content/courses/algorithms/07-dynamic-programming/word-break.js
  command -v python3 >/dev/null
  python3 content/courses/algorithms/07-dynamic-programming/example-1.py
  python3 content/courses/algorithms/07-dynamic-programming/example-2.py
  python3 content/courses/algorithms/07-dynamic-programming/example-3.py
  python3 content/courses/algorithms/07-dynamic-programming/example-4.py
  python3 content/courses/algorithms/07-dynamic-programming/example-6.py
  # C examples require input; compile/run manually if desired.
  echo "C examples: cc content/courses/algorithms/07-dynamic-programming/example-5.c -o /tmp/example-5 && /tmp/example-5"
  echo "C examples (requires input): cc content/courses/algorithms/07-dynamic-programming/example-8.c -o /tmp/example-8 && /tmp/example-8"

run-08-greedy:
  command -v node >/dev/null
  node content/courses/algorithms/08-greedy/best-time-to-buy-and-sell-stock.js
  node content/courses/algorithms/08-greedy/jump-game.js
  command -v python3 >/dev/null
  python3 content/courses/algorithms/08-greedy/example-1.py
  python3 content/courses/algorithms/08-greedy/example-2.py
  python3 content/courses/algorithms/08-greedy/example-3.py

run-09-backtracking:
  command -v node >/dev/null
  node content/courses/algorithms/09-backtracking/combination-sum.js
  command -v python3 >/dev/null
  python3 content/courses/algorithms/09-backtracking/example-2.py

run-10-math-bit:
  command -v node >/dev/null
  node content/courses/algorithms/10-math-bit/count-number-of-bits.js
  node content/courses/algorithms/10-math-bit/number-of-1-bits.js
  node content/courses/algorithms/10-math-bit/reverse-bits.js
  node content/courses/algorithms/10-math-bit/sum-of-two-integers.js

run-11-recursion-divide-conquer:
  command -v python3 >/dev/null
  python3 content/courses/algorithms/11-recursion-divide-conquer/example-1.py
  python3 content/courses/algorithms/11-recursion-divide-conquer/example-3.py

run-15-probabilistic-structures:
  command -v python3 >/dev/null
  python3 content/courses/algorithms/15-probabilistic-structures/bloom.py

# Run every C and C++ practice solution (each one asserts its own behaviour).
run-practice-solutions:
  for d in practice/build/systems/c/*/; do \
    echo "--- $d"; \
    if [ -x "$d/run.sh" ]; then (cd "$d" && ./run.sh); \
    else cc -std=c11 -Wall -O2 -pthread "$d"*.c -o /tmp/ol-solution && /tmp/ol-solution; fi; \
  done
  for d in practice/build/systems/cpp/*/; do \
    echo "--- $d"; \
    if [ -x "$d/run.sh" ]; then (cd "$d" && ./run.sh); \
    else c++ -std=c++20 -Wall -O2 -pthread "$d"main.cpp -o /tmp/ol-solution && /tmp/ol-solution; fi; \
  done

# Course catalogue checks: manifests resolve and relative links exist.
course-lint:
  python3 scripts/course_catalog.py validate
  python3 scripts/check_links.py

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
