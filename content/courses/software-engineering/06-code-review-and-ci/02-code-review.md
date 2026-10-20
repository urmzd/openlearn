# Code review

The gateway is about to run on a cluster with real budgets behind it: rate limits that tenants are billed against, SLOs that page someone. Until now the only reviewer of its code has been a test suite, judged by mutation testing for what it catches. But tests catch what someone thought to test. A teammate's pull request arrives with a green test of its own and a plausible description, and the bugs it carries are exactly the ones nobody wrote a test for: a race that needs two goroutines, a refill argument that is true for every level except a negative one, a "tidy" one-liner that changes a rounding rule.

This section puts you on the other side of the pull request. The PR changes the gateway's rate limiter, and your job is to find what is wrong before it merges and to say so in a form the author can act on. The PR under review passes its own test and still ships five bugs. Finding them takes a discipline more than a talent: read every changed line against every state that can reach it, write each finding as evidence rather than opinion, and resist the false alarms that cost the author time and you credibility.

This section builds on several earlier ones. [The repository and its CI gate](01-your-repo-and-ci-gate.md) is the gate a PR passes before anyone reads it. [Mutation testing in depth](../03-testing/05-mutation-testing-in-depth.md) practiced reading a planted fault, the same skill without a test pointing at it. The code under review is the limiter from [Rate limiting](../10-authorization/02-rate-limiting.md) in *Software Engineering*; its contract is [`limit.go`](../../../../course/solutions/go/gateway/limit/limit.go) plus the x-ratelimit headers and Retry-After from [`openai-subset.v1.yaml`](../../../../course/contracts/openapi/openai-subset.v1.yaml). [Go](../../programming-and-algorithms/01-language-primers/06-go.md) in *Programming and Algorithms* covers goroutines and the race detector, and chapter 9 of [Software Engineering at Google](../02-engineering-at-scale/README.md) is the background on review at scale. The model answer is [`docs/reviews/craft-08-pr-review.toml`](../../../../course/solutions/docs/reviews/craft-08-pr-review.toml): a review of the PR (branch `drill/craft-08`) with a verdict, a summary, and one finding per problem, each pointing at a line of the PR. Read the PR in [The pull request](#the-pull-request) and write your own findings before you read the model review.

> **Learning objectives**
> - State what a code review is for and what it is not for.
> - Follow the reviewer's loop: claims, tests, race-detector run, outside-in reading, what is missing from the diff.
> - Write a finding with a location, an observation, a consequence shown by a concrete input or interleaving, and a suggestion, and assign it a severity that follows from the consequence.
> - Apply the Go memory model to recognize data races, including unlocked reads, and to dismiss correct code that only looks suspicious.
> - Review a token-bucket limiter change and find the debt, race, in-flight, cadence, and rounding defects.

## What a review is for

A **pull request** (PR) is a proposed change: a branch, a description, and a diff against the branch it wants to merge into (the **base**). A **review** reads it and returns a **verdict**: approve (merge as is), comment (no blocking concern, but notes), or request changes (it must not merge as it is).

Studies of review at Google and Microsoft find the same order of value: first finding defects, then keeping the design coherent, then spreading knowledge of the code. A review exists to stop **defects a test suite did not anticipate** from merging. What a review is not for is anything a machine checks. Formatting (`gofmt`), lint (`go vet`), and the test suite already ran in CI, through [the gate](01-your-repo-and-ci-gate.md), before anyone looked. Style is a linter's job.

## The reviewer's loop

Knowing what to look for is half of it; the other half is an order of work that makes you look. The loop has six steps.

1. **Read the description.** What does the PR claim to do, and why? Write down the claims: each is something to verify.
2. **Read the tests first.** They say what the author believed the code must do. What they leave out is where to look hardest.
3. **Run them, and yours, against the PR.** Check the branch out and run the suite under the race detector. A failing test is a finding with its evidence attached.
4. **Read the diff from the outside in.** Types and fields first (what state is new?), then the functions that write that state, then the ones that read it. For each changed line, list the states that can reach it and ask what the line does in each one. Most review misses are a state the author did not picture: negative levels, in-flight requests, the 257th call, a concurrent reader.
5. **Look for what is not in the diff.** A new field must be maintained on every path that should touch it; a new lock rule must hold in every method; a new exported name is new API.
6. **Write findings, then the verdict.** The verdict follows from the most severe finding.

## Severity and the shape of a finding

A finding is evidence, not an opinion. Its severity says what must happen before merge, and it should follow from the consequence, not from how strongly you feel:

| Severity | Use it when | Example |
|---|---|---|
| `blocking` | the change breaks a contract, corrupts data or money, races, or opens a hole; it must not merge | a rate limiter admits a request it must reject |
| `major` | wrong in a case production reaches, or a performance cliff; fix before merge unless the author shows the case cannot happen | an O(n) walk on every request under a global lock |
| `minor` | correct but fragile, unclear, or under-tested | a test that covers only the happy path |
| `nit` | taste; never blocks | a name |
| `question` | you do not know whether it is a problem; ask | "can a stream outlive this TTL?" |

A finding the author can act on has four parts: the **location** (a line of the PR head), the **observation** (what the line does), the **consequence** with a concrete input or interleaving ("TPM 1000, reserve 300, settle 4000, wait 150 s: the key is dropped while it still owes 500 tokens"), and a **suggestion**. A consequence you cannot show with an input is a `question`, not a `blocking`.

## Go concurrency facts a reviewer needs

The PR is Go code shared between goroutines, so the reviewer has to reason about concurrency precisely. The facts that matter, and what each means for a review:

| Fact | Consequence for review |
|---|---|
| A **data race** is two accesses to the same memory from different goroutines, at least one a write, with no happens-before order between them (a lock held by both, a channel operation, `sync/atomic`). | The size of the value does not matter: reading a counter, a pointer, or a length without the lock its writers take is a race. |
| A program with a data race has no defined behavior in the Go memory model. | "It just reads a slightly stale number" is not a valid argument; the runtime may also abort with `concurrent map read and map write`. |
| `go test -race` reports races that **happen** during the run. | A race needs a test where the two accesses overlap; a single-goroutine test proves nothing. |
| Deleting the current key of a `for k := range m` loop is allowed by the language spec. | Not a finding. |
| `sync.Once.Do(f)` runs `f` at most once, even from many goroutines. | Code inside a `once.Do` runs once per value; a decrement there cannot happen twice. |

The short version: **a read is not exempt from the lock.**

## The code under review: a token bucket with debt

One more piece of background, and then the review itself. The PR changes the limiter from [Rate limiting](../10-authorization/02-rate-limiting.md). Each API key has two **token buckets**, one for requests per minute and one for tokens per minute. A bucket of capacity $C$ refills continuously at $C$ per minute, never above $C$. A request **reserves** its estimated cost, which is subtracted at once. When it finishes it **settles** its actual cost, so unused tokens go back and an overrun is charged, which can drive the level below zero (**debt**). A request is admitted only when the level covers its cost.

The notation used in the rest of the section:

| Symbol | Meaning | Type / shape |
|---|---|---|
| $C$ | bucket capacity, and its refill per minute | tokens (float64) |
| $\ell(t)$ | bucket level at time $t$; may be negative after an overrun | tokens (float64) |
| $\Delta t$ | time since the last refill | minutes |
| $\ell \leftarrow \min(C, \ell + C\,\Delta t)$ | refill | |
| $w(n) = \max(0, (n - \ell) / C)$ | minutes until the bucket holds $n$ tokens | minutes |

It follows that after an idle period of $\Delta t$ minutes the bucket is full only if $\ell + C\,\Delta t \ge C$, that is, $\Delta t \ge 1 - \ell / C$. For $\ell \ge 0$ that is at most one minute. For $\ell = -3C$ it is four. Keep that inequality in mind; it decides the first finding.

## Working a smaller review by hand

Before the real PR, practice the loop on a smaller one, against a response cache like the one built in [Response cache](../../systems/06-distributed-data-and-caching/01-response-cache.md): "fix(cache): expire entries lazily on Get". The changed function at the PR head is:

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

The description claims expired entries are dropped on read. Which states can reach line 10? Two goroutines in `Get` for the same expired key, or one in `Get` while `Put` writes the map under `c.mu.Lock()`. Line 10 writes the map holding no lock at all: two writers, or a writer and `Put`, with no happens-before between them. That is a data race, and with two concurrent deletes the runtime may abort the process (`concurrent map writes`).

Line 5 releases the read lock before `e` is used. Is that a bug too? No. `Entry` is a struct copied by value at line 4, so lines 6 to 13 read a private copy. Line 9 uses `After`, so an entry is still served at the instant `Now() == expires`. Whether the TTL is inclusive is a contract question, not a defect you can show, so it becomes a question. The review:

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

Notice what is absent: there is no finding on line 5. It looks like "use after unlock", but it reads a copy, and a blocking finding there would be a false alarm.

## The pull request and the review file

Now the real one. This part gives you the PR and the format your review must take; stop after it and write your own findings.

### The pull request

The PR starts from the reference rate limiter and has two commits:

| Commit | Subject | What it is |
|---|---|---|
| 1 | `fix(limit): evict idle keys so the bucket map stays bounded` | the change: its message is the PR description |
| 2 | `test(limit): cover idle-key eviction` | the PR's own test, `go/gateway/limit/idle_evict_test.go` |

The description says that keys that go quiet are evicted (`sweep`), so the bucket map holds the keys active in the last `IdleTTL` instead of every key ever seen. The PR adds to `keyState` a `seen time.Time` (the last Reserve) and an `open int` (reservations not yet settled or cancelled), the constants `IdleTTL = 2 * time.Minute` and `sweepEvery = 256`, and a `calls` counter on `Limiter`. `Reserve` sets `st.seen = now` and increments `st.open`; `adjust` decrements `st.open`. The parts of the head that matter, with the line numbers the review cites:

```go
// in Reserve, under l.mu:
     	l.calls++
188  	if l.calls >= sweepEvery {
     		l.sweep(now)
     	}

// sweep drops every key that has no open reservation, has been idle for at
// least IdleTTL, and whose buckets have refilled to capacity: dropping it
// then changes nothing a later request can observe. The caller holds l.mu.
     func (l *Limiter) sweep(now time.Time) {
     	for k, st := range l.keys {
221  		if now.Sub(st.seen) < IdleTTL {
     			continue
     		}
224  		// Idle for IdleTTL means two minutes of refill, and a bucket refills
     		// its whole capacity every minute, so its buckets are full by now.
     		delete(l.keys, k)
     	}
     }

// Len is the number of keys the limiter holds (the size sweep bounds). A
// map's length is one word, so reading it needs no lock.
     func (l *Limiter) Len() int {
230  	return len(l.keys)
     }

// RetryAfterSeconds, "tidied" while here:
395  	return max(1, int(d.Round(time.Second)/time.Second))
```

The PR's test reserves and settles one key, advances a fake clock three minutes, makes 255 more Reserve calls on another key so the 256th sweeps, and expects `Len() == 1`. It passes.

The PR contains five seeded defects, three of them required, along with several changes that look odd but are correct. The `st.open--` in `adjust` sits inside the reservation's `sync.Once`, so it cannot run twice, and `delete` inside `for k := range l.keys` is allowed by the spec. Each seeded defect is a real bug that a Go test can show.

### The review file

Your review is a TOML file with these keys:

| Key | Type | Rule |
|---|---|---|
| `pr` | string | `"drill/craft-08"` |
| `blob` | string | the git object id of `go/gateway/limit/limit.go` at the PR head |
| `verdict` | string | `approve`, `comment`, or `request-changes` |
| `summary` | string | at least 80 characters: what the PR does and what blocks it |
| `[[finding]]` | table, one or more | one per problem |
| `finding.file` | string | `go/gateway/limit/limit.go` (the default) or `go/gateway/limit/idle_evict_test.go` |
| `finding.line` | integer | a line of that file **at the PR head** |
| `finding.severity` | string | `blocking`, `major`, `minor`, `nit`, or `question` |
| `finding.category` | string | `correctness`, `concurrency`, `performance`, `security`, `api`, `tests`, `docs`, or `style` |
| `finding.comment` | string | at least 30 characters: observation and consequence |
| `finding.suggestion` | string | required for `blocking` and `major`, at least 20 characters |

A finding **hits** a seeded defect when its line is on one of the lines the defect changed, or one line above or below.

### What a correct review must do

A correct review satisfies each of these checks:

| Case | Kind of check | Why it matters |
|---|---|---|
| the review names the PR head (`pr`, `blob`) | smoke | the review is about the code that would merge |
| every key is well formed; every line exists at the head; serious findings carry a suggestion; at most 10 serious findings | unit | a review the author can act on |
| the verdict is `request-changes` | unit | this PR must not merge as it is |
| each required defect (forgiven debt, the racy `Len`, the Retry-After rounding) has a `blocking` or `major` finding on its lines | unit | the defects a client or the race detector can see |
| at least 4 of the 5 defects are hit, at any severity | unit | a careful review misses at most one |
| at most one serious finding on `limit.go` hits no defect, and none sits on a line the PR gets right on purpose | unit | false alarms have a cost |

## Building the review step by step

If you have written your own findings, compare your reasoning with this chain. Seven steps lead to the review.

1. **Write down the PR's claim.** "Drop a key only when dropping it changes nothing a later request can observe." Every finding is a counterexample to that claim, or to an unchanged contract.
2. **A dropped key comes back full.** So dropping is safe only if the key would be full anyway. From the token-bucket inequality, a bucket is full after $\Delta t$ idle minutes only if $\Delta t \ge 1 - \ell / C$. For $\ell < 0$ (debt) two minutes is not enough. That is the first state to test: a key in debt.
3. **A key can be idle and busy at once.** `seen` is the last *Reserve*; a long stream reserved once and still running has not called Reserve since. So "idle since seen" does not mean "no request in flight". That is the second state: an open reservation.
4. **Every access to `l.keys` from another goroutine must hold `l.mu`.** `Reserve` and `sweep` write the map under the lock, so any method that reads it without the lock races.
5. **"Every 256th call" must be true exactly once per 256.** Evaluate the condition at call 256 and at call 257.
6. **Unrelated edits change behavior too.** Compare the rewritten `RetryAfterSeconds` with the contract (round up, at least 1) at a non-integer input such as 1.2 s.
7. **Severity follows from the consequence.** Breaking a billing promise, a race, or a client-visible contract blocks; an in-flight edge or an O(n) cost per request is major; a thin test is minor.

### The review, annotated

```toml
verdict = "request-changes"
summary = "The PR bounds the limiter's memory by sweeping idle keys on every 256th Reserve and adds Len for a keys gauge. The approach is right, but three changes break observable behavior: the sweep forgives overrun debt, Len races with Reserve, and the RetryAfterSeconds rewrite tells clients to retry too early. Two more need fixing before merge: keys with in-flight reservations are swept, and the cadence check sweeps on every call after the 256th. Requesting changes."
```

The summary says what the PR does and that the approach is right, so the author should keep going, then lists the blocking issues first. The verdict follows from the most severe finding.

**Finding 1, line 224, blocking: the sweep forgives debt.**

```toml
comment    = "Two idle minutes refill two capacities, not everything: a key that settled an overrun sits below zero (TPM 1000, reserve 300, settle 4000 leaves -3000; 150 s later it is still -500). Dropping it here hands it a full bucket on its next request, so the overrun is never paid. The comment's reasoning holds only for a level >= 0."
suggestion = "Refill both buckets at `now` and drop the key only when both levels are back at capacity (continue otherwise). Add a test that settles an overrun, idles past IdleTTL, sweeps, and expects a rejection."
```

This is step 2 with numbers: $1000 - 300 - (4000 - 300) = -3000$, and $-3000 + 1000 \times 2.5 = -500$ after 150 s. The comment on line 224 is true for every $\ell \ge 0$ and false for the one state that matters for billing. The suggestion is the check the reference sweep had: refill, then `continue` unless both levels are at capacity.

**Finding 2, line 230, blocking: `Len` races.**

```toml
comment    = "Len reads l.keys without l.mu while Reserve inserts and sweep deletes under it. That is a data race in the Go memory model whatever the size of the value read; go test -race with one goroutine calling Len while others Reserve new keys reports it, and a metrics scrape calls Len exactly that way."
suggestion = "Take l.mu in Len (Lock, defer Unlock) like every other method, or keep a counter updated under the lock and read it atomically if the gauge must not contend."
```

The code comment's argument ("one word, so no lock") is exactly the argument the memory model rules out. The finding names the interleaving that shows it (a gauge scrape concurrent with Reserve) and the test that proves it (`-race` with overlapping goroutines).

**Finding 3, line 221, major: open reservations are swept.**

```toml
comment    = "The guard no longer checks open reservations. A stream that outlives IdleTTL (a long completion) holds a reservation; its key can be swept mid-stream, and then adjust finds no key and drops the overrun charge of its Settle without a trace."
suggestion = "Keep `st.open > 0` in the guard so keys with in-flight reservations are never dropped, and test it: sweep during an open reservation, then settle an overrun and expect the next request rejected."
```

The PR adds `open` and maintains it, then never reads it. This is step 5 of the reviewer's loop: look for what is not in the diff. It is major rather than blocking because it needs a stream longer than two minutes, which production reaches but rarely.

**Finding 4, line 395, blocking: Retry-After rounds to nearest.**

```toml
comment    = "This rounds to the nearest second; the contract (and the code it replaces) rounds up with a floor of 1. A 1.2 s wait now says Retry-After: 1, so a client that obeys it comes back early and is rejected again. It is also a behavior change hidden in a tidy-up line of an unrelated PR."
suggestion = "Restore int(math.Ceil(d.Seconds())) with the floor of 1, and move clean-ups to their own PR."
```

The reference is `s := int(math.Ceil(d.Seconds())); if s < 1 { s = 1 }`. The one-liner agrees with it on whole seconds and on everything below 0.5 s, so a test with round inputs passes; 1.2 s separates them. It is blocking because it breaks the client-visible contract.

**Finding 5, line 188, major: the cadence check.**

```toml
comment    = "`l.calls >= sweepEvery` stays true after the 256th call, so from then on every Reserve walks every key under the global lock: O(keys) per request, the cost the cadence exists to amortize."
suggestion = "Use `l.calls%sweepEvery == 0` as the description says (or reset l.calls when a sweep runs)."
```

At call 256 both `>=` and `% == 0` are true; at call 257 only `>=` is. The PR's test makes exactly 256 calls, so it cannot see the difference. This is major, not blocking: results stay correct, but latency grows with the number of keys under a global lock.

**Finding 6, the test file, minor.**

```toml
file       = "go/gateway/limit/idle_evict_test.go"
line       = 17
severity   = "minor"
category   = "tests"
comment    = "The new test covers only the happy path, a settled and full key. It misses the cases that make eviction risky: a key in debt, a key with an open reservation, and Len called while other goroutines Reserve."
suggestion = "Add those three tests (the third under -race); they would have caught three of the issues above."
```

This finding turns the review into the tests that would have prevented it.

**What the review does not say.** There is no finding on `st.open--` (inside `sync.Once`, it runs once per reservation) and none on `delete` inside the `range` loop (allowed by the spec). Both look suspicious, both are correct, and a blocking comment on either would be a false alarm. The full review is [`craft-08-pr-review.toml`](../../../../course/solutions/docs/reviews/craft-08-pr-review.toml).

### Why this review works

Each serious finding carries **inputs, not adjectives**: the input or interleaving that shows the bug, so the author can turn it into a test instead of arguing. **Severity is tied to consequence**: three blocking findings (billing, a race, a client contract), two major (a rare in-flight case, a performance cliff), one minor. A review that marks everything blocking loses the signal. The review **requests changes but endorses the approach**: bounding the map is the right goal, the summary says so, and that keeps the next revision small.

There is a tradeoff. A line-by-line state walk takes longer than skimming. For a change to code that bills tenants, the cost is justified; for a doc change it would not be.

## What can go wrong

Each of the following traps is one the model review avoids, and each corresponds to one of its findings.

> **Warning:** Trusting the argument in a code comment instead of checking it against every state that reaches the line (a negative level, a zero capacity) means approving a line whose comment is true only for the states the author pictured. Finding 1 evaluates the comment's claim at $\ell = -3000$.

Accepting "it is only a read" as a reason to skip the lock the writers take lets a racy read merge, when `go test -race` with a concurrent test would have failed. Finding 2 cites the memory model and the overlapping test.

Reviewing only steady state, and never a request that is in flight across the change, misses the long stream that loses its charge when its key is dropped under it. Finding 3 asks what happens to an open reservation.

Skimming the "while here" part of a PR lets a one-line clean-up change behavior nobody asked to change. Finding 4 compares the rewrite with the contract at 1.2 s.

Reading a condition once instead of evaluating it at the boundary and one past it misses a once-per-cadence check that runs on every call after the first. Finding 5 evaluates call 256 and call 257.

> **Warning:** False alarms cost too. Blocking a line that only looks wrong, such as a decrement inside `sync.Once` or a delete inside `range`, makes the author spend a round trip proving the code correct. The model review has no finding on either line.

## Where this leads

This section rests on [The repository and its CI gate](01-your-repo-and-ci-gate.md), the gate every PR passes before a person reads it; on [Mutation testing in depth](../03-testing/05-mutation-testing-in-depth.md), whose reading of planted faults is the same skill without a test pointing at it; on [Rate limiting](../10-authorization/02-rate-limiting.md), the limiter the PR changes and its contract; and on [Go](../../programming-and-algorithms/01-language-primers/06-go.md), for goroutines, the race detector, and `sync`.

Every later change is reviewed this way, and the production setup expects reviewed changes. [Contract tests](../03-testing/06-contract-tests.md) covers consumer-driven tests from gateway to engine, which is what a review asks for when an interface changes. [Design review: the tracer and the serving platform](../../systems/01-system-design/01-design-review-tracer-and-serving-platform.md) in *Systems Engineering* applies the same discipline one level up, reviewing a design document against a rubric. [Secrets and authorization review](../09-security/03-secrets-and-authorization-review.md) reviews the gateway's authorization code with this same findings format. And [Performance regression bisect](../08-maintenance/04-performance-regression-bisect.md) hunts a regression over a scripted commit series.

The findings file is a teaching tool; production review has richer equivalents:

| This piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| a findings file per PR | Gerrit, GitHub reviews, Google's Critique | inline threads, required approvals, ownership (OWNERS files), automated analyzers posting findings | google.github.io/eng-practices; *Software Engineering at Google*, ch. 9 and 19 |
| reading for races | `go test -race` in CI, `go vet -copylocks`, staticcheck | finds the race whenever a test exercises it | go.dev/doc/articles/race_detector |
| severity by hand | review bots and pre-merge static analysis (Tricorder at Google) | findings with fix suggestions before a human looks | Sadowski et al., "Tricorder" (ICSE 2015) |

### Further reading

- Google, "The Code Reviewer's Guide", google.github.io/eng-practices (free).
- Sadowski et al., "Modern Code Review: A Case Study at Google", ICSE SEIP 2018.
- Bacchelli and Bird, "Expectations, Outcomes, and Challenges of Modern Code Review", ICSE 2013.
- The Go Memory Model, go.dev/ref/mem (free).

## Key terms

- **Pull request (PR)**: a proposed change, made of a branch, a description, and a diff against a base branch.
- **Verdict**: a review's outcome: approve, comment, or request changes.
- **Finding**: one problem in a review, with a location, an observation, a consequence shown by an input or interleaving, and a suggestion.
- **Severity**: `blocking`, `major`, `minor`, `nit`, or `question`, chosen by the consequence of the problem.
- **Data race**: two accesses to the same memory from different goroutines, at least one a write, with no happens-before order between them.
- **Token bucket with debt**: a rate limiter whose level refills at capacity per minute, is reserved up front, settled after, and may go negative on an overrun.
- **False alarm**: a serious finding on code that is correct, which costs the author a round trip and the reviewer credibility.

## Summary

- A review exists to stop defects a test suite did not anticipate; formatting, lint, and tests are the machine's job.
- A finding is evidence: where, what goes wrong, an input or interleaving that shows it, and what to change. Without an input, it is a question.
- Read every changed line against every state that can reach it, and look for what the diff leaves out.
- A read is not exempt from the lock: in Go, unsynchronized access with a concurrent write is a data race whatever the size of the value.
- Severity follows from consequence, and false alarms cost as much as misses.

## Review questions

1. Why is two minutes of idling enough to refill a key at level 0 but not one at $-3000$ with $C = 1000$?
   <details><summary>Answer</summary>Full requires $\Delta t \ge 1 - \ell / C$ minutes. At $\ell = 0$ that is 1 minute; at $\ell = -3000$ it is $1 + 3 = 4$ minutes. After two minutes the indebted key is still at $-1000$.</details>
2. The code comment on `Len` says a map length is one word. Why is that not a defense?
   <details><summary>Answer</summary>The Go memory model defines a data race by accesses and happens-before, not by size. An unsynchronized read concurrent with writes is a race with undefined behavior, and `go test -race` reports it when the accesses overlap.</details>
3. Why does the PR's own test not catch the cadence bug?
   <details><summary>Answer</summary>It makes exactly 256 calls; `calls >= 256` and `calls % 256 == 0` agree at 256. The difference shows only from call 257 on.</details>
4. Why is there no finding on `st.open--`, which decrements a counter from a goroutine that may call Settle and Cancel?
   <details><summary>Answer</summary>The decrement is inside the reservation's `sync.Once`, under `l.mu`, so it runs at most once per reservation whatever the caller does. Flagging it would be a false alarm.</details>
5. In the cache example, why is line 5 (releasing the read lock before using `e`) not a finding?
   <details><summary>Answer</summary>`Entry` is a struct copied by value at line 4, so the rest of the function reads a private copy. Nothing shared is accessed after the unlock, so there is no race; a blocking finding there would be a false alarm.</details>
