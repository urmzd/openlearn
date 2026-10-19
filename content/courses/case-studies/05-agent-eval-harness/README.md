# Agent Evaluation Harness

## Overview

- **Problem**: run an LLM agent against a simulated system it can change, and
  produce a grade that someone else can verify without trusting the agent, the
  harness author, or a second model.
- **Runnable**: [`eval_harness.py`](eval_harness.py) -- an isolated world,
  deterministic tools, replay verification, a set-based checker, and an oracle
  and mutant suite. Standard library only, `python eval_harness.py`. No API key
  needed.
- **Prerequisites**: [LLM Evaluation](../../ai-platform-engineering/09-llm-evaluation/),
  and the [Grounded SQL Agent](../02-grounded-sql-agent/) for the single-query
  version of the same ideas.
- **Estimated time**: 1-2 days

## Key Takeaways

- **Completed is not passed.** Execution outcome, evidence completeness, and
  grade are three separate facts. A run can finish cleanly after editing the
  wrong row.
- **The grade is a set relation.** Required changes must be a subset of what
  changed, and what changed must be a subset of what was allowed. One half
  catches missing work, the other catches unrequested work.
- **Test the checker before you trust a score.** A correct scripted oracle
  must pass and every deliberate mutant must fail. A criterion that no mutant
  can fail has never been tested.
- **Evidence that does not replay is not graded.** Uncertainty is never a
  pass.
- **You control the harness's determinism, not the provider's.** Temperature 0
  and a fixed seed still produced different call sequences. Sample the agent,
  never the checker.
- **A model judge is an opinion.** In this build, one judge sample in five
  approved an attempt that had made zero writes.
- **Most failures were the agent's, and one was the task's.** Attribute every
  failure to the agent, the world, or the task text before you change
  anything.

## How to Study

- Run the file and read the table it prints. Seven attempts end with
  `completed` and fail. That table is the argument for the whole design.
- Add a mutant of your own, predict which criteria it fails, then run it. When
  your prediction is wrong, find out whether the mutant or the checker is
  mistaken. In the original build one prediction was wrong and the checker was
  right.
- Delete the replay check from `grade` and watch three tests fail. Then edit a
  recorded read result by hand and see a wrong attempt pass.
- Write the evaluation protocol for a model swap before reading section 7, and
  compare.

---

# Concepts & Techniques

## The Problem

An agent is given an instruction and a set of tools over a business system: a
helpdesk, an inventory. It reads, it writes, it stops. Four things make grading
it harder than grading text:

1. The answer is a changed state, and there are many correct sequences that
   reach it.
2. Attempts must not see each other, or one agent's write becomes another's
   starting condition.
3. The model is stochastic and the provider is outside your control.
4. The harness is software. It has bugs, and a bug in a grader is silent.

## 1. Three facts, kept apart

| Fact | Question | Values |
|------|----------|--------|
| **Outcome** | Did execution end, and how? | completed, error, turn limit, budget limit, interrupted |
| **Evidence** | Is the record complete and consistent? | verified, ungraded, lost |
| **Grade** | Did every criterion pass? | passed, failed |

A report that collapses these cannot distinguish a model mistake from a dead
worker. Decide up front what counts in the denominator. In this build, a
budget cut-off counts as an attempt, because the budget is part of the task's
conditions. An external interruption, a setup failure, and an ungraded record
do not count, and are reported beside the quality figure instead of inside it.

## 2. Isolation: one shared world, private sessions

The world is loaded, validated, and indexed once, then never mutated. Each
attempt gets a thin private layer over it. A write stays in the layer. A read
falls through to the parent on a miss.

| Property | Why it matters |
|----------|----------------|
| An empty session copies nothing | Opening one measured about 40 ns against a 30,000-ticket seed and a 641-ticket seed alike |
| A read returns an independent copy | A caller that edits a returned row cannot reach shared memory |
| The final diff inspects only touched keys | Grading cost follows the work done, not the world size |

**Session isolation is not process isolation.** Private rows remove state
conflicts. Attempts still share CPU, memory, and one provider quota. The first
live batch exhausted a 30,000 token-per-minute limit, which is a harness
failure and says nothing about the agent.

**Preserve the shape of the data.** Real data distinguishes a field that is
absent from a field that is null. In the seed this build was given, one owner
field was absent 19 times and null 22 times. A harness that normalises the two
hides exactly the case agents get wrong. Two live attempts failed by handling
null owners and ignoring rows where the field did not exist.

## 3. Determinism you own

| Source of variation | Control |
|---------------------|---------|
| Timestamps | A private logical clock that advances per write. Never wall time |
| Generated ids | A pure function of the entity name and a per-session counter |
| Tool order | Tools execute one at a time, in request order |
| Setup errors | Collect every issue and sort them, so a broken seed reports the same list each run |

With those in place, the same call sequence produces the same state, which is
what makes replay possible.

What you do not own is the provider. At temperature 0 with a fixed seed, the
same task produced different call sequences across runs. Identical inputs
support a repeatability check. They do not promise identical output.

## 4. The checker: required, changed, allowed

Let `delta = diff(start, final)`, `R` the required changes, and `A` the allowed
changes. The state criterion is `R ⊆ delta ⊆ A`.

| Attempt | `R ⊆ delta` | `delta ⊆ A` | Reading |
|---------|-------------|-------------|---------|
| Correct row, correct value | yes | yes | Pass |
| Right item, wrong warehouse | no | no | Missing work and unauthorized work, reported separately |
| Did nothing | no | yes | Missing work only |
| Did the task, then added an unrequested comment | yes | no | Unauthorized work only |

Three rules that kept the checker honest:

- **Derive target sets from the starting state.** A task that names two
  example tickets may require all seven that match. Hardcoded ids would pass
  on a small seed and miss 145 targets on a large one.
- **State is not enough.** Final rows cannot show a read that never happened,
  a refused call, or a write that was reverted. Trace criteria sit beside state
  criteria.
- **A read is what was returned.** A read-before-write criterion credits only
  rows the tool actually handed back. A page that was never fetched does not
  count.

The checker is pure and contains no model. Evaluating the same verified record
twice gives the same grade.

## 5. Replay before grading

A record stores the world's hash, every call with its arguments and result,
the before and after image of every write, and the final touched rows. Before
any criterion is evaluated, the calls are re-executed against a fresh session
and compared.

This exists because of a real defect. An early verifier did not compare
recorded read results. Substituting one of them changed a failing attempt into
a passing one. The fix was to compare the whole observation and the whole effect:
results, typed errors, sequence numbers, write images, and final rows.

Two limits to state plainly:

- **Replay proves consistency, not authenticity.** A self-consistent record
  can still be fabricated.
- **A hash identifies an input. It does not archive it.** If the file is gone,
  knowing its hash does not bring it back.

## 6. Testing the harness

The supplied fixtures tested the world's behaviour. None of them showed that a
wrong answer receives a failing grade. That needs its own evidence.

| Technique | What it proves |
|-----------|----------------|
| **Oracle** | The task is solvable and the checker accepts a correct solution |
| **Mutant** | The checker rejects one specific mistake, on the expected criterion |
| **Coverage rule** | Every automated criterion has at least one mutant that fails it |
| **Break it on purpose** | A test that has never failed has not been shown to work |
| **Fault injection** | Behaviour at a crash, a short write, a lost response |

The original build ended with 13 oracles and 60 mutants across 69 automated
criteria. That is evidence for the tested fault set. It is not a proof against
every defect, and the write-up says so.

Two smaller lessons from the same work:

- **A passing fixture proves nothing until you see it fail.** A temporary
  defect was introduced to confirm that the relevant fixture could catch it.
- **Benchmarks lie when the compiler deletes the work.** The first allocation
  benchmark discarded its result and reported zero allocations. Keeping the
  result alive corrected it.

## 7. Sampling and replication

**Sample the agent, never the checker.** Each sample gets a fresh session and
its own seed. The record is graded once.

**Pin a revision before anything runs.** Hash the world, the selected tasks,
the system prompt, the driver settings, the sample count, the budgets, and the
build. Executing the same run directory with a changed turn limit produces a
revision mismatch that names the component that changed.

| In the revision | Recorded as context only |
|-----------------|--------------------------|
| World and contract hashes | File paths |
| Task text and criteria | Timestamps and process ids |
| System prompt hash | Parallelism |
| Model, temperature, limits | |
| Build identity | |

Parallelism stays out of the revision but must be written down, because it
changes error rates and latency.

**Change one factor at a time.** Between two live runs in this build, the code
and the provider both varied. The write-up says the difference cannot be
attributed to either.

**Small samples are small.** Two or three samples per task show that a task
can pass and can fail. They do not give a pass rate.

**A planned protocol for swapping a model**, written before any comparison:

| Gate | Measure |
|------|---------|
| Quality | Automated success plus required human acceptance |
| Critical cases | Results per required task or failure group, because an average hides a severe defect |
| Coverage | Accepted tasks over all requested attempts, so refusing hard cases does not look like accuracy |
| Reliability | Errors, incomplete records, recovery attempts |
| Latency | Queue time and p50 and p95 completion |
| Total cost | All cost, including retries and failed samples, over accepted tasks |
| Uncertainty | Sample count and an interval for each comparison |

The arithmetic that motivates the cost gate: a model that costs $10 for 100
accepted tasks is $0.10 each. A model that costs $6 and yields 40 accepted
tasks is $0.15 each. The cheaper request was the more expensive result.

## 8. What the live runs showed

| Run | Driver | Passed |
|-----|--------|--------|
| Reference | Scripted oracle | 20 of 20 |
| Live, temperature 0, first build | Small model | 13 of 20 |
| Live, temperature 0, second build | Small model | 14 of 20, at $0.11 over 128 requests |
| Live, temperature 1, two worlds | Small model | 23 of 39 |

In the temperature 0 runs, every failed attempt ended with `completed`. None
hit a turn, call, or budget limit. The agent stopped because it believed it
was done.

**Attribute before you fix.**

| Source | Meaning | Share of failures in the analysed run |
|--------|---------|----------------------------------------|
| Agent | Tools worked, the agent chose badly | All but one |
| World | A wrong tool result or a missing capability | None |
| Task text | The instruction asked for something the tools could not express | One |

The task-text failure is the instructive one. The instruction said to create a
ticket "describing" an issue. The create tool had no body field, so the model
put the description in a comment, which the criteria forbade. Rewording the
instruction took that task to 4 of 4. It also changed the task's hash, so the
new result is not comparable with the old runs and is reported separately.

**Three recurring patterns**, two failures each:

1. **Narrow search, then stop.** The agent takes the first result as the
   target. One attempt called a list with a limit of 1, received a total of 7
   and an offset for the next page, used the first item, and stopped.
2. **Incomplete filter.** The task names two statuses, the filter accepts one
   value, so two calls are needed. The agent made one.
3. **Skipped or added step.** A requested reply was omitted, or an unrequested
   comment was added.

## 9. Judges

A sampled, schema-constrained model judge was built for the prose criteria,
then removed.

- One judgment split four to one. A later three-sample judgment was unanimous.
  Neither fact established accuracy.
- One of five samples approved an attempt that had made zero writes.
- The judge read text written by the agent being judged, which is untrusted
  input.

The resulting rule: deterministic checks decide automated criteria, and prose
criteria stay `not_reviewed` until a person reviews them. If a judge returns
later, it triages immutable evidence for human review, its opinion is stored
apart from the grade, and it is calibrated on held-out human labels.
Unanimous models can share one error, so unanimous cases are audited too.

## 10. Budgets and feasibility

| Limit | Bounds |
|-------|--------|
| Turns | Model round trips |
| Tool calls | Work done, which a large target set can dominate |
| Context | Request size, which a turn cap does not bound |
| Requests and money | Spend, reserved before dispatch |

- **Check feasibility before spending.** Run the oracle for each selected task
  and count its calls and writes. One task's reference solution needed 42
  calls, 38 of them writes, under a 40-turn limit, and a live run had already
  spent all 40 turns on it. The precheck refuses a run whose limit is below
  the minimum number of writes, and warns when only the reference solution
  does not fit.
- **Retries multiply.** Eight outer attempts over a client that retries twice
  is up to 24 HTTP requests. Give retries one owner.
- **Reservations are estimates.** A lost response leaves a charge of unknown
  size. Keep it as uncertain exposure instead of treating it as free.
- **Bound context explicitly.** Replacing old tool results with short notices
  cut one simulated attempt from 454k to 315k estimated prompt tokens. The
  full trace is kept separately for audit.

## 11. Explaining a failure

A grade names the criterion that failed. It does not name the cause.
Advisory observations read the trace and report what it shows:
`unfollowed_page`, `no_writes`, `write_without_read`, `refused_calls`,
`ended_with_question`.

Two rules:

- **An observation never changes a grade**, and a passing attempt can carry
  one.
- **Report its reach.** These generic observations explained 2 of 5 failed
  live attempts. The rest needed the meaning of the instruction, which a
  generic rule does not have.

## 12. Counterfactuals

| Fork | What runs | What it measures |
|------|-----------|------------------|
| Replay fork | Recorded calls, under a different session policy | Sensitivity to the policy. The model is not called |
| Live fork, edited call | The edited call runs for real, then the model continues | What the agent does after a different action |
| Live fork, edited result | The original call runs, the model is shown an invented result | Robustness to an observation. Marked synthetic |

Forks are bounded on depth, count, and budget, and they never count as
attempts. Unlimited counterfactual search would select the branch that
succeeded.

## 13. Building it with coding agents

This harness was built with AI coding tools across many sessions, which
creates its own evidence problem: decisions live in transcripts.

- **Keep a decision log with stable ids and a status.** Each entry states the
  choice, the reason, an example, the limit, and the evidence. Status is one
  of built, partial, planned, deferred, or superseded. When a choice changes,
  keep the id and name its replacement.
- **Treat transcripts as searchable evidence.**
  [agentspec](https://github.com/urmzd/agentspec) reads the session stores of
  several coding tools through one interface:

  ```sh
  agentspec session search "judge" --project <name> --role user
  agentspec session list --since 7d --format json
  agentspec session export <source> <session-id>
  ```

  `--role user` finds what was asked, as opposed to an assistant restating it.
- **Export what you will need.** The tool reads each coding tool's own store
  and keeps no history itself. Retention belongs to each tool, and some prune
  after about a month.
- **Do not share a checkout between sessions.** An early commit swept up a
  concurrent session's change under an unrelated message. Use one worktree per
  session, stage files by name, and read the final diff before committing.
- **Disclose the assistance, and own the claims.** Commands, source, fixtures,
  and saved records are the evidence. The author remains responsible for
  verifying them.

## Build Log

1. **Inspect the data first.** Count field shapes, absent against null,
   near-duplicate names, and target-set sizes on the large seed. Most design
   decisions came from this step.
2. **Validate the contract, then the seed.** Reject a broken world before any
   model runs. Zero tokens spent on an invalid setup.
3. **Build the world and the session layer**, with isolation tests.
4. **Make time and ids deterministic**, then order the tools.
5. **Write the checker as pure functions** over the diff and the trace.
6. **Write oracles and mutants.** This is where the checker's own bugs
   surfaced.
7. **Run live, and keep the failures.** Early provider failures stay in their
   original records.
8. **Audit the first grades.** The substituted-read defect was found here.
9. **Add replay verification**, then tighten it to cover observations.
10. **Try a judge, measure it, remove it.**
11. **Add revisions, budgets, and the feasibility check.**
12. **Add diagnostics and a read-only monitor** over the saved runs.

Measurement came before optimisation at every step, and the harness was tested
before its scores were believed.

## Technique Catalog

| Technique | When to apply |
|-----------|---------------|
| Separate outcome, evidence, and grade | Any evaluation of something that executes |
| Required and allowed change sets | Any agent that mutates state |
| Derived target sets | Any criterion that must hold across data sizes |
| Trace criteria beside state criteria | Any task where order or coverage matters |
| Replay before grading | Any saved record that feeds a score |
| Oracles and mutants | Any grader, before its first score is reported |
| Logical clock and sequence-derived ids | Any simulation that must replay |
| Revision pinning | Any result that will be compared later |
| Feasibility precheck | Any run that costs money |
| Attribute to agent, world, or task text | Every failure, before changing anything |
| Advisory diagnostics | Explaining failures without touching grades |
| Decision log with status | Any build that spans many sessions |

## Connections to Other Tracks

| Concept | Connected Track | How |
|---------|-----------------|-----|
| Harness pattern, judge bias, contamination | [LLM Evaluation](../../ai-platform-engineering/09-llm-evaluation/) | The theory this study applies |
| Result-set scoring, bounded retry, deterministic gates | [Grounded SQL Agent](../02-grounded-sql-agent/) | The same ideas for a single query |
| Cost per successful task, escalation, RFT rewards | [Model Routing & Cascades](../../ai-platform-engineering/11-model-routing-and-cascades/) | A verified checker is a reward function |
| Journals, ownership, recovery | [Durable Orchestration & Workers](../../ai-platform-engineering/05-durable-orchestration-and-workers/) | What crash survival would require |
| Idempotency, atomic claims | [Exactly-Once Event API](../03-exactly-once-event-api/) | Commit evidence before acknowledging |
| Mutation testing, test doubles | [The Testing Mentality](../../software-craftsmanship/03-testing-mentality/) | Testing the tests |

## Company Relevance

| Company | How This Appears | Focus |
|---------|-----------------|-------|
| Agent evaluation and simulation vendors | Isolated environments with verifiable grading | The core product |
| AI infrastructure vendors | Reward functions for reinforcement fine-tuning | A checker that cannot be gamed |
| Any team shipping agents | Regression gates on tool-using behaviour | Outcome, evidence, grade |
| Any Staff+ AI role | Stating limits and separating claims from evidence | Judgment under ambiguity |
