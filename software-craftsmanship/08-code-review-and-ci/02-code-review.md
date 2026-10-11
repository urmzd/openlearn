<!-- ol:module craft.08 -->
# Code review: review a seeded PR against your system

## Overview

| | |
|---|---|
| **Module** | `craft.08` · practice · Go, docs · Pass 7 · 3 to 4 h |
| **You build** | `docs/reviews/craft-08-pr-review.toml`: your review of a pull request against your gateway's rate limiter (branch `drill/craft-08`, built for you by the check), with a verdict, a summary, and one finding per problem, each pointing at a line of the PR |
| **Contract** | none: the review file format is section 4. The code under review implements gw.03's contract (`go/gateway/limit`, x-ratelimit headers and Retry-After from `openai-subset.v1.yaml`) |
| **Tests** | `course/tests/craft.08/`, run by its `check` script (what they check: section 4) |
| **Needs** | no code of yours: the pull request starts from the course reference of `gw.03`. Reading: [`craft.01`](01-your-repo-and-ci-gate.md) the gate a PR passes before review, [`craft.07`](../03-testing-mentality/05-mutation-testing-in-depth.md) reading a planted fault, `gw.03` the rate limiter, [`lang.06`](../12-language-and-tool-primers/06-go.md) goroutines and the race detector, [*Software Engineering at Google*](../02-swe-at-google/README.md) ch. 9 |
| **Used by** | no call site (a practice): you review your own changes this way from now on, and `craft.19` reviews your authz code with the same file format |
| **Milestone** | `MS-prod` (the Pass 7 production milestone requires a passing `craft.08`) |
| **Optional depth** | Google, "The Code Reviewer's Guide" (google.github.io/eng-practices, free); Sadowski et al., "Modern Code Review: A Case Study at Google" (ICSE SEIP 2018); Bacchelli and Bird, "Expectations, Outcomes, and Challenges of Modern Code Review" (ICSE 2013); The Go Memory Model (go.dev/ref/mem, free) |

## Key Takeaways

- A review exists to stop **defects a test suite did not anticipate** from merging; style is a linter's job. The PR you review passes its own test, and still ships five bugs (`test_each_seeded_defect_is_a_real_bug`).
- A finding is evidence, not an opinion: **where** (a line of the PR head), **what** goes wrong, **an input or interleaving that shows it**, and **what to change** (`test_review_is_well_formed`).
- Read every changed line against **every state that can reach it**: negative levels, in-flight requests, the 257th call, a concurrent reader. Most review misses are a state the author did not picture (`test_every_required_defect_is_found`).
- **A read is not exempt from the lock.** In Go, two accesses, one a write, with no happens-before between them, are a data race whatever the size of the value, and `go test -race` proves it when a test makes them overlap.
- **False alarms cost too.** A blocking comment on a correct line wastes the author's time and your credibility (`test_no_serious_finding_blocks_a_correct_line`).

## How to work this chapter

```bash
ol start craft.08                 # records the start; nothing to stub
ol tests craft.08                 # read the test catalog first
ol check craft.08                 # first run: builds branch drill/craft-08 and the review template, and fails
git log --stat drill/craft-08~2..drill/craft-08    # the two commits under review, with their messages
git diff drill/craft-08~2 drill/craft-08           # the change
cd .ol/craft.08/pr/go && go test -race ./gateway/limit/...   # your own tests, against the PR head
# write docs/reviews/craft-08-pr-review.toml (section 4), then:
ol check craft.08                 # grades your findings against the seeded defect list
```

The branch is built in a scratch clone of your repo (`.ol/craft.08/pr`, gitignored) and fetched into your repo; your working tree and your own `limit.go` are never touched. Its first commit replaces `limit.go` with the course reference of gw.03, so the diff you review is only the teammate's change, not the differences between your code and ours.

---

## 1. Why now

Your gateway is about to run on a cluster with real budgets behind it (`MS-prod`): rate limits that tenants are billed against, SLOs that page someone. Until now the only reviewer of your code has been a test suite, graded by mutation testing for what it catches. Tests catch what someone thought to test. A teammate's pull request arrives with a green test of its own and a plausible description, and the bugs it carries are exactly the ones nobody wrote a test for: a race that needs two goroutines, a refill argument that is true for every level except a negative one, a "tidy" one-liner that changes a rounding rule. This module puts you on the other side of the pull request. The PR changes your rate limiter (gw.03); you find what is wrong before it merges, and say so in a form the author can act on.

## 2. Principles

### 2.1 What a review is for

A **pull request** (PR) is a proposed change: a branch, a description, and a diff against the branch it wants to merge into (the **base**). A **review** reads it and returns a **verdict**: approve (merge as is), comment (no blocking concern, but notes), or request changes (it must not merge as it is). Studies of review at Google and Microsoft find the same order of value: finding defects, then keeping the design coherent, then spreading knowledge of the code. What a review is not for is anything a machine checks: formatting (`gofmt`), lint (`go vet`), and the test suite already ran in CI before you looked (craft.01).

### 2.2 The reviewer's loop

1. **Read the description.** What does the PR claim to do, and why? Write down the claims: each is something to verify.
2. **Read the tests first.** They say what the author believed the code must do. What they leave out is where to look hardest.
3. **Run them, and yours, against the PR.** Check the branch out (the scratch clone is already a checkout), run the suite under the race detector. A failing test is a finding with its evidence attached.
4. **Read the diff from the outside in.** Types and fields first (what state is new?), then the functions that write that state, then the ones that read it. For each changed line, list the states that can reach it and ask what the line does in each one.
5. **Look for what is not in the diff.** A new field must be maintained on every path that should touch it; a new lock rule must hold in every method; a new exported name is new API.
6. **Write findings, then the verdict.** The verdict follows from the most severe finding.

### 2.3 Severity and the shape of a finding

| Severity | Use it when | Example |
|---|---|---|
| `blocking` | the change breaks a contract, corrupts data or money, races, or opens a hole; it must not merge | a rate limiter admits a request it must reject |
| `major` | wrong in a case production reaches, or a performance cliff; fix before merge unless the author shows the case cannot happen | an O(n) walk on every request under a global lock |
| `minor` | correct but fragile, unclear, or under-tested | a test that covers only the happy path |
| `nit` | taste; never blocks | a name |
| `question` | you do not know whether it is a problem; ask | "can a stream outlive this TTL?" |

A finding the author can act on has four parts: the **location** (a line of the PR head), the **observation** (what the line does), the **consequence** with a concrete input or interleaving ("TPM 1000, reserve 300, settle 4000, wait 150 s: the key is dropped while it still owes 500 tokens"), and a **suggestion**. A consequence you cannot show with an input is a `question`, not a `blocking`.

### 2.4 Go concurrency facts a reviewer needs

| Fact | Consequence for review |
|---|---|
| A **data race** is two accesses to the same memory from different goroutines, at least one a write, with no happens-before order between them (a lock held by both, a channel operation, `sync/atomic`). | The size of the value does not matter: reading a counter, a pointer, or a length without the lock its writers take is a race. |
| A program with a data race has no defined behavior in the Go memory model. | "It just reads a slightly stale number" is not a valid argument; the runtime may also abort with `concurrent map read and map write`. |
| `go test -race` reports races that **happen** during the run. | A race needs a test where the two accesses overlap; a single-goroutine test proves nothing. |
| Deleting the current key of a `for k := range m` loop is allowed by the language spec. | Not a finding. |
| `sync.Once.Do(f)` runs `f` at most once, even from many goroutines. | Code inside a `once.Do` runs once per value; a decrement there cannot happen twice. |

### 2.5 The code under review: a token bucket with debt

The PR changes gw.03's limiter. Each API key has two **token buckets** (requests per minute and tokens per minute). A bucket of capacity $C$ refills continuously at $C$ per minute, never above $C$. A request **reserves** its estimated cost, which is subtracted at once; when it finishes it **settles** its actual cost, so unused tokens go back and an overrun is charged, which can drive the level below zero (**debt**). A request is admitted only when the level covers its cost.

| Symbol | Meaning | Type / shape |
|---|---|---|
| $C$ | bucket capacity, and its refill per minute | tokens (float64) |
| $\ell(t)$ | bucket level at time $t$; may be negative after an overrun | tokens (float64) |
| $\Delta t$ | time since the last refill | minutes |
| $\ell \leftarrow \min(C, \ell + C\,\Delta t)$ | refill | |
| $w(n) = \max(0, (n - \ell) / C)$ | minutes until the bucket holds $n$ tokens | minutes |

So after an idle period of $\Delta t$ minutes the bucket is full only if $\ell + C\,\Delta t \ge C$, that is $\Delta t \ge 1 - \ell / C$. For $\ell \ge 0$ that is at most one minute. For $\ell = -3C$ it is four.

## 3. Worked example by hand

A different PR, against a response cache like the one gw.06 builds: "fix(cache): expire entries lazily on Get". The changed function at the PR head:

```go
 1  // Get returns the entry for k if it has not expired.
 2  func (c *Cache) Get(k Key) (Entry, bool) {
 3  	c.mu.RLock()
 4  	e, ok := c.m[k]
 5  	c.mu.RUnlock()
 6  	if !ok {
 7  		return Entry{}, false
 8  	}
 9  	if c.clock.Now().After(e.expires) {
10  		delete(c.m, k) // expired: drop it now instead of waiting for eviction
11  		return Entry{}, false
12  	}
13  	return e, true
14  }
```

Walk the loop of 2.2. The description claims expired entries are dropped on read. State that can reach line 10: two goroutines in `Get` for the same expired key, or one in `Get` while `Put` writes the map under `c.mu.Lock()`. Line 10 writes the map holding no lock at all: two writers, or a writer and `Put`, with no happens-before. That is a data race, and with two concurrent deletes the runtime may abort the process (`concurrent map writes`). Line 5 releases the read lock before `e` is used: is that a bug too? No. `Entry` is a struct copied by value at line 4, so lines 6 to 13 read a private copy. Line 9 uses `After`: an entry is still served at the instant `Now() == expires`. Whether the TTL is inclusive is a contract question, not a defect you can show. The review:

```toml
pr      = "drill/cache-lazy-expiry"
blob    = "..."
verdict = "request-changes"
summary = "Lazy expiry on Get is a good idea and saves an eviction pass, but the delete runs without the write lock, so concurrent Gets of an expired key race on the map. One question on the TTL boundary."

[[finding]]
line       = 10
severity   = "blocking"
category   = "concurrency"
comment    = "delete writes c.m with no lock held. Two Gets of the same expired key, or a Get concurrent with Put, write the map at once: a data race, and the runtime can abort with 'concurrent map writes'."
suggestion = "Take c.mu.Lock() for the delete and re-check expiry under it (a Put may have refreshed the key since the read lock was released)."

[[finding]]
line       = 9
severity   = "question"
category   = "api"
comment    = "After serves the entry at exactly Now() == expires. Is the TTL inclusive in the cache contract?"
```

There is no finding on line 5: it looks like "use after unlock", but it reads a copy. A blocking finding there would be a false alarm.

## 4. The artifact and its check

### The pull request

`ol check craft.08` builds branch `drill/craft-08` in your repo from your current `HEAD`:

| Commit | Subject | What it is |
|---|---|---|
| 1 | `chore(review): start from the course reference of go/gateway/limit/limit.go` | review setup: the base of the PR, not part of it |
| 2 | `fix(limit): evict idle keys so the bucket map stays bounded` | the change: read its message, it is the PR description |
| 3 | `test(limit): cover idle-key eviction` | the PR's own test, `go/gateway/limit/idle_evict_test.go` |

The change is built from the course reference of gw.03 by applying the PR and its seeded defects with `patch` (`course/mutants/craft.08/`). It contains **five seeded defects**, three of them required, and several changes that look odd but are correct. Each seeded defect is a real bug: the check proves it with a Go test before it grades you. When the course changes the PR, the check rebuilds the branch and your review's `blob` no longer matches; review the new head.

### The review file

`docs/reviews/craft-08-pr-review.toml` (the first run writes a template):

| Key | Type | Rule |
|---|---|---|
| `pr` | string | `"drill/craft-08"` |
| `blob` | string | the git object id of `go/gateway/limit/limit.go` at the PR head; the template fills it in |
| `verdict` | string | `approve`, `comment`, or `request-changes` |
| `summary` | string | at least 80 characters: what the PR does and what blocks it |
| `[[finding]]` | table, one or more | one per problem |
| `finding.file` | string | `go/gateway/limit/limit.go` (the default) or `go/gateway/limit/idle_evict_test.go` |
| `finding.line` | integer | a line of that file **at the PR head** (`git show drill/craft-08:go/gateway/limit/limit.go \| cat -n`) |
| `finding.severity` | string | `blocking`, `major`, `minor`, `nit`, or `question` (2.3) |
| `finding.category` | string | `correctness`, `concurrency`, `performance`, `security`, `api`, `tests`, `docs`, or `style` |
| `finding.comment` | string | at least 30 characters: observation and consequence |
| `finding.suggestion` | string | required for `blocking` and `major`, at least 20 characters |

A finding **hits** a seeded defect when its line is on one of the lines the defect changed, or one line above or below; a finding hits at most one defect (the nearest). Findings on the test file are never counted against you.

### What the tests check

| Test | KIND | Checks | Why it matters |
|---|---|---|---|
| `test_pr_rebuilds_from_the_reference` | regression | the PR and its five defects still apply to the current gw.03 reference; each defect changes a line; the decoy lines are unique and clear of every defect | the grade is computed on the head you were given |
| `test_intended_pr_passes_every_proof` | regression | the PR without defects passes the five proofs and its own test under `go test -race` | a proof that fails on correct code would make any defect look real |
| `test_each_seeded_defect_is_a_real_bug` | fault | the PR plus one defect fails the Go proof written for it | you are graded only on bugs a program can show |
| `test_branch_is_the_course_pr` | smoke | `drill/craft-08` in your repo holds the current head, with the three commits above | you reviewed what you are graded on |
| `test_review_is_well_formed` | unit | the keys above; every line exists at the head; serious findings carry a suggestion; at most 10 serious findings | a review the author can act on |
| `test_verdict_requests_changes` | unit | the verdict is `request-changes` | this PR must not merge as it is |
| `test_every_required_defect_is_found` | unit | each required defect has a `blocking` or `major` finding on its lines | the defects a client or the race detector can see |
| `test_most_defects_are_found` | unit | at least 4 of the 5 defects are hit, at any severity | a careful review misses at most one |
| `test_no_serious_finding_blocks_a_correct_line` | unit | at most one `blocking` or `major` finding on `limit.go` hits no defect, and none sits on a line the PR gets right on purpose | false alarms have a cost |

A failed grade never says which defect you missed: go back to 2.2 and walk the states again. Once you pass, `course/mutants/craft.08/manifest.tsv` lists them (honor system).

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| Trusting the argument in a comment instead of checking it against every state that reaches the line (a negative level, a zero capacity) | you approve a line whose comment is true for the states the author pictured | `test_every_required_defect_is_found` (seeded `s01`) |
| Accepting "it is only a read" as a reason to skip the lock the writers take | a racy read merges; `go test -race` with a concurrent test would have failed | `test_every_required_defect_is_found` (seeded `s02`) |
| Reviewing only steady state, never a request that is in flight across the change | a long stream loses its charge when its key is dropped under it | `test_most_defects_are_found` (seeded `s03`) |
| Skimming the "while here" part of a PR | a one-line clean-up changes behavior nobody asked to change | `test_every_required_defect_is_found` (seeded `s04`), `test_verdict_requests_changes` |
| Reading a condition once instead of evaluating it at the boundary and one past it | a once-per-cadence check runs on every call after the first | `test_most_defects_are_found` (seeded `s05`) |
| Blocking a line that only looks wrong (a decrement inside `sync.Once`, a delete inside `range`) | the author spends a round trip proving the code correct | `test_no_serious_finding_blocks_a_correct_line` |

## 6. Where it's used next

| Direction | Module | How it connects |
|---|---|---|
| Back | `craft.01` | the gate every PR passes before a person reads it |
| Back | `craft.07` | reading a planted fault: the same skill, without a test to point at it |
| Back | `gw.03` | the rate limiter the PR changes, and its contract |
| Back | `lang.06` | goroutines, the race detector, and `sync` |
| Forward | `craft.20` | consumer-driven contract tests for gateway to engine: what a review asks for when an interface changes |
| Forward | `review.01` | the same discipline one level up: reviewing a design document against a rubric |
| Forward | `craft.19` | secrets and authz review of your gateway, with this findings format |
| Forward | `craft.16` | perf regression bisect: a scripted commit series built by the same scratch-copy machinery (drill `ops.07`) |

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| a findings file per PR | Gerrit, GitHub reviews, Google's Critique | inline threads, required approvals, ownership (OWNERS files), automated analyzers posting findings | google.github.io/eng-practices; *Software Engineering at Google*, ch. 9 and 19 |
| reading for races | `go test -race` in CI, `go vet -copylocks`, staticcheck | finds the race whenever a test exercises it | go.dev/doc/articles/race_detector |
| severity by hand | review bots and pre-merge static analysis (Tricorder at Google) | findings with fix suggestions before a human looks | Sadowski et al., "Tricorder" (ICSE 2015) |
