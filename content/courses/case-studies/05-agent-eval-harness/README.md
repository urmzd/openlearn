# Agent Evaluation Harness

An agent is given an instruction and a set of tools over a business system, such as a helpdesk or an inventory. It reads, it writes, and it stops. Grading it is much harder than grading text, because the answer is a changed state that many different call sequences can reach, because attempts must not see each other, because the model is stochastic and its provider is outside your control, and because the harness is itself software that can have silent bugs. This case study builds a harness that runs an LLM agent against a simulated system it can change and produces a grade that someone else can verify without trusting the agent, the harness author, or a second model.

The runnable implementation is [`eval_harness.py`](eval_harness.py): an isolated world, deterministic tools, replay verification, a set-based checker, and an oracle and mutant suite. It uses the standard library only and needs no API key; run it with `python eval_harness.py` and read the table it prints. Seven attempts end with `completed` and fail. That table is the argument for the whole design.

> **Learning objectives**
> - Keep execution outcome, evidence completeness, and grade as three separate facts, and decide what counts in the denominator.
> - Isolate attempts with private sessions over a shared world, and make time, ids, and tool order deterministic.
> - Grade state changes with the set relation $R \subseteq \Delta \subseteq A$ alongside trace criteria, and replay every record before grading it.
> - Test the checker with oracles and mutants before trusting any score.
> - Sample, pin, budget, and attribute live runs so that results are comparable and failures are explained.

## Before you begin

This study applies the theory in [LLM evaluation](../../ai-platform-engineering/09-llm-evaluation/README.md). The [Grounded SQL agent](../02-grounded-sql-agent/README.md) develops the same ideas for a single query, and is a gentler place to start.

## The problem

Four things make grading an agent harder than grading text:

1. The answer is a changed state, and there are many correct sequences that reach it.
2. Attempts must not see each other, or one agent's write becomes another's starting condition.
3. The model is stochastic and the provider is outside your control.
4. The harness is software. It has bugs, and a bug in a grader is silent.

The sections that follow address them in turn: first what a result even is, then isolation and determinism, then the checker and the evidence it grades, then how to test the checker, and finally what live runs revealed and how to run them responsibly.

## Three facts, kept apart

Completed is not passed. A run can finish cleanly after editing the wrong row, so a result has three parts:

| Fact | Question | Values |
|------|----------|--------|
| **Outcome** | Did execution end, and how? | completed, error, turn limit, budget limit, interrupted |
| **Evidence** | Is the record complete and consistent? | verified, ungraded, lost |
| **Grade** | Did every criterion pass? | passed, failed |

A report that collapses these cannot distinguish a model mistake from a dead worker. Decide up front what counts in the denominator. In this build, a budget cut-off counts as an attempt, because the budget is part of the task's conditions. An external interruption, a setup failure, and an ungraded record do not count, and are reported beside the quality figure instead of inside it.

## Isolation: one shared world, private sessions

The world is loaded, validated, and indexed once, and then never mutated. Each attempt gets a thin private layer over it: a write stays in the layer, and a read falls through to the parent on a miss. That design has three properties worth having:

| Property | Why it matters |
|----------|----------------|
| An empty session copies nothing | Opening one measured about 40 ns against a 30,000-ticket seed and a 641-ticket seed alike |
| A read returns an independent copy | A caller that edits a returned row cannot reach shared memory |
| The final diff inspects only touched keys | Grading cost follows the work done, not the world size |

> **Warning:** Session isolation is not process isolation. Private rows remove state conflicts, but attempts still share CPU, memory, and one provider quota. The first live batch exhausted a 30,000 token-per-minute limit, which is a harness failure and says nothing about the agent.

Isolation must also preserve the shape of the data. Real data distinguishes a field that is absent from a field that is null. In the seed this build was given, one owner field was absent 19 times and null 22 times. A harness that normalizes the two hides exactly the case agents get wrong: two live attempts failed by handling null owners and ignoring rows where the field did not exist.

## Determinism you own

Replay, which the grade will depend on, needs the same call sequence to produce the same state. Four sources of variation are under the harness's control:

| Source of variation | Control |
|---------------------|---------|
| Timestamps | A private logical clock that advances per write. Never wall time |
| Generated ids | A pure function of the entity name and a per-session counter |
| Tool order | Tools execute one at a time, in request order |
| Setup errors | Collect every issue and sort them, so a broken seed reports the same list each run |

What you do not own is the provider. At temperature 0 with a fixed seed, the same task produced different call sequences across runs. Identical inputs support a repeatability check; they do not promise identical output. So the rule is to sample the agent, never the checker.

## The checker: required, changed, allowed

The grade is a set relation. Let $\Delta = \operatorname{diff}(\text{start}, \text{final})$, let $R$ be the required changes, and let $A$ be the allowed changes. The state criterion is

$$R \subseteq \Delta \subseteq A.$$

In code it reads `R ⊆ delta ⊆ A`. One half catches missing work, and the other catches unrequested work, as four example attempts show:

| Attempt | `R ⊆ delta` | `delta ⊆ A` | Reading |
|---------|-------------|-------------|---------|
| Correct row, correct value | yes | yes | Pass |
| Right item, wrong warehouse | no | no | Missing work and unauthorized work, reported separately |
| Did nothing | no | yes | Missing work only |
| Did the task, then added an unrequested comment | yes | no | Unauthorized work only |

Three rules kept the checker honest. First, derive target sets from the starting state. A task that names two example tickets may require all seven that match; hardcoded ids would pass on a small seed and miss 145 targets on a large one. Second, state is not enough. Final rows cannot show a read that never happened, a refused call, or a write that was reverted, so trace criteria sit beside state criteria. Third, a read is what was returned. A read-before-write criterion credits only rows the tool actually handed back; a page that was never fetched does not count.

The checker is pure and contains no model. Evaluating the same verified record twice gives the same grade.

## Replay before grading

A record stores the world's hash, every call with its arguments and result, the before and after image of every write, and the final touched rows. Before any criterion is evaluated, the calls are re-executed against a fresh session and compared. Evidence that does not replay is not graded, and uncertainty is never a pass.

This step exists because of a real defect. An early verifier did not compare recorded read results, and substituting one of them changed a failing attempt into a passing one. The fix was to compare the whole observation and the whole effect: results, typed errors, sequence numbers, write images, and final rows.

Two limits should be stated plainly. Replay proves consistency, not authenticity: a self-consistent record can still be fabricated. And a hash identifies an input; it does not archive it. If the file is gone, knowing its hash does not bring it back.

> **Try it:** Delete the replay check from `grade` and watch three tests fail. Then edit a recorded read result by hand and see a wrong attempt pass.

## Testing the harness

The supplied fixtures tested the world's behavior. None of them showed that a wrong answer receives a failing grade, and that needs its own evidence. Test the checker before you trust a score, using these techniques:

| Technique | What it proves |
|-----------|----------------|
| **Oracle** | The task is solvable and the checker accepts a correct solution |
| **Mutant** | The checker rejects one specific mistake, on the expected criterion |
| **Coverage rule** | Every automated criterion has at least one mutant that fails it |
| **Break it on purpose** | A test that has never failed has not been shown to work |
| **Fault injection** | Behavior at a crash, a short write, a lost response |

A correct scripted oracle must pass, and every deliberate mutant must fail. A criterion that no mutant can fail has never been tested. The original build ended with 13 oracles and 60 mutants across 69 automated criteria. That is evidence for the tested fault set; it is not a proof against every defect, and the write-up says so.

Two smaller lessons came from the same work. A passing fixture proves nothing until you see it fail, so a temporary defect was introduced to confirm that the relevant fixture could catch it. And benchmarks lie when the compiler deletes the work: the first allocation benchmark discarded its result and reported zero allocations, and keeping the result alive corrected it.

> **Try it:** Add a mutant of your own, predict which criteria it fails, then run it. When your prediction is wrong, find out whether the mutant or the checker is mistaken. In the original build one prediction was wrong, and the checker was right.

## Sampling and replication

With a trustworthy checker, the next question is how to run the agent so that results mean something. Sample the agent, never the checker: each sample gets a fresh session and its own seed, and the record is graded once.

Pin a revision before anything runs. Hash the world, the selected tasks, the system prompt, the driver settings, the sample count, the budgets, and the build. Executing the same run directory with a changed turn limit then produces a revision mismatch that names the component that changed. Some facts belong in the revision and others are only context:

| In the revision | Recorded as context only |
|-----------------|--------------------------|
| World and contract hashes | File paths |
| Task text and criteria | Timestamps and process ids |
| System prompt hash | Parallelism |
| Model, temperature, limits | |
| Build identity | |

Parallelism stays out of the revision but must be written down, because it changes error rates and latency.

Change one factor at a time. Between two live runs in this build, the code and the provider both varied, and the write-up says the difference cannot be attributed to either. And remember that small samples are small: two or three samples per task show that a task can pass and can fail. They do not give a pass rate.

Before reading on, write the evaluation protocol you would use for swapping one model for another, then compare it with the one this build planned before any comparison:

| Gate | Measure |
|------|---------|
| Quality | Automated success plus required human acceptance |
| Critical cases | Results per required task or failure group, because an average hides a severe defect |
| Coverage | Accepted tasks over all requested attempts, so refusing hard cases does not look like accuracy |
| Reliability | Errors, incomplete records, recovery attempts |
| Latency | Queue time and p50 and p95 completion |
| Total cost | All cost, including retries and failed samples, over accepted tasks |
| Uncertainty | Sample count and an interval for each comparison |

The arithmetic that motivates the cost gate is simple. A model that costs $10 for 100 accepted tasks is $0.10 each. A model that costs $6 and yields 40 accepted tasks is $0.15 each. The cheaper request was the more expensive result.

## What the live runs showed

The harness was run with a scripted oracle and with a small live model:

| Run | Driver | Passed |
|-----|--------|--------|
| Reference | Scripted oracle | 20 of 20 |
| Live, temperature 0, first build | Small model | 13 of 20 |
| Live, temperature 0, second build | Small model | 14 of 20, at $0.11 over 128 requests |
| Live, temperature 1, two worlds | Small model | 23 of 39 |

In the temperature 0 runs, every failed attempt ended with `completed`. None hit a turn, call, or budget limit. The agent stopped because it believed it was done.

Attribute every failure before you fix anything, to the agent, the world, or the task text:

| Source | Meaning | Share of failures in the analysed run |
|--------|---------|----------------------------------------|
| Agent | Tools worked, the agent chose badly | All but one |
| World | A wrong tool result or a missing capability | None |
| Task text | The instruction asked for something the tools could not express | One |

The task-text failure is the instructive one. The instruction said to create a ticket "describing" an issue. The create tool had no body field, so the model put the description in a comment, which the criteria forbade. Rewording the instruction took that task to 4 of 4. It also changed the task's hash, so the new result is not comparable with the old runs and is reported separately.

The agent failures fell into three recurring patterns, two failures each:

1. **Narrow search, then stop.** The agent takes the first result as the target. One attempt called a list with a limit of 1, received a total of 7 and an offset for the next page, used the first item, and stopped.
2. **Incomplete filter.** The task names two statuses, the filter accepts one value, so two calls are needed. The agent made one.
3. **Skipped or added step.** A requested reply was omitted, or an unrequested comment was added.

## Judges

A sampled, schema-constrained model judge was built for the prose criteria, and then removed. A model judge is an opinion, and the evidence showed why. One judgment split four to one; a later three-sample judgment was unanimous, and neither fact established accuracy. One of five samples approved an attempt that had made zero writes. And the judge read text written by the agent being judged, which is untrusted input.

The resulting rule: deterministic checks decide automated criteria, and prose criteria stay `not_reviewed` until a person reviews them. If a judge returns later, it triages immutable evidence for human review, its opinion is stored apart from the grade, and it is calibrated on held-out human labels. Unanimous models can share one error, so unanimous cases are audited too.

## Budgets and feasibility

Each kind of limit bounds something different:

| Limit | Bounds |
|-------|--------|
| Turns | Model round trips |
| Tool calls | Work done, which a large target set can dominate |
| Context | Request size, which a turn cap does not bound |
| Requests and money | Spend, reserved before dispatch |

Check feasibility before spending. Run the oracle for each selected task and count its calls and writes. One task's reference solution needed 42 calls, 38 of them writes, under a 40-turn limit, and a live run had already spent all 40 turns on it. The precheck refuses a run whose limit is below the minimum number of writes, and warns when only the reference solution does not fit.

Three more budget facts are easy to miss. Retries multiply: eight outer attempts over a client that retries twice is up to 24 HTTP requests, so give retries one owner. Reservations are estimates: a lost response leaves a charge of unknown size, which should be kept as uncertain exposure instead of treated as free. And context must be bounded explicitly: replacing old tool results with short notices cut one simulated attempt from 454k to 315k estimated prompt tokens, with the full trace kept separately for audit.

## Explaining a failure

A grade names the criterion that failed; it does not name the cause. Advisory observations read the trace and report what it shows: `unfollowed_page`, `no_writes`, `write_without_read`, `refused_calls`, and `ended_with_question`.

Two rules govern them. An observation never changes a grade, and a passing attempt can carry one. And you should report an observation's reach: these generic observations explained 2 of 5 failed live attempts. The rest needed the meaning of the instruction, which a generic rule does not have.

## Counterfactuals

To ask "what would the agent have done if...", the harness can fork a record in three ways:

| Fork | What runs | What it measures |
|------|-----------|------------------|
| Replay fork | Recorded calls, under a different session policy | Sensitivity to the policy. The model is not called |
| Live fork, edited call | The edited call runs for real, then the model continues | What the agent does after a different action |
| Live fork, edited result | The original call runs, the model is shown an invented result | Robustness to an observation. Marked synthetic |

Forks are bounded on depth, count, and budget, and they never count as attempts. Unlimited counterfactual search would select the branch that succeeded.

## Building it with coding agents

This harness was built with AI coding tools across many sessions, which creates its own evidence problem: decisions live in transcripts. Five practices handled it.

Keep a decision log with stable ids and a status. Each entry states the choice, the reason, an example, the limit, and the evidence. The status is one of built, partial, planned, deferred, or superseded, and when a choice changes, the id stays and the entry names its replacement.

Treat transcripts as searchable evidence. [agentspec](https://github.com/urmzd/agentspec) reads the session stores of several coding tools through one interface:

```sh
agentspec session search "judge" --project <name> --role user
agentspec session list --since 7d --format json
agentspec session export <source> <session-id>
```

`--role user` finds what was asked, as opposed to an assistant restating it.

Export what you will need. The tool reads each coding tool's own store and keeps no history itself; retention belongs to each tool, and some prune after about a month.

Do not share a checkout between sessions. An early commit swept up a concurrent session's change under an unrelated message. Use one worktree per session, stage files by name, and read the final diff before committing.

Disclose the assistance, and own the claims. Commands, source, fixtures, and saved records are the evidence, and the author remains responsible for verifying them.

## The build log

The harness was built in this order:

1. **Inspect the data first.** Count field shapes, absent against null, near-duplicate names, and target-set sizes on the large seed. Most design decisions came from this step.
2. **Validate the contract, then the seed.** Reject a broken world before any model runs. Zero tokens spent on an invalid setup.
3. **Build the world and the session layer**, with isolation tests.
4. **Make time and ids deterministic**, then order the tools.
5. **Write the checker as pure functions** over the diff and the trace.
6. **Write oracles and mutants.** This is where the checker's own bugs surfaced.
7. **Run live, and keep the failures.** Early provider failures stay in their original records.
8. **Audit the first grades.** The substituted-read defect was found here.
9. **Add replay verification**, then tighten it to cover observations.
10. **Try a judge, measure it, remove it.**
11. **Add revisions, budgets, and the feasibility check.**
12. **Add diagnostics and a read-only monitor** over the saved runs.

Measurement came before optimization at every step, and the harness was tested before its scores were believed.

## Techniques to take away

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

## Where this leads

The harness pattern, judge bias, and contamination are developed in [LLM evaluation](../../ai-platform-engineering/09-llm-evaluation/README.md), and the [Grounded SQL agent](../02-grounded-sql-agent/README.md) applies result-set scoring, bounded retry, and deterministic gates to a single query. A verified checker is also a reward function, which connects to cost per successful task, escalation, and reinforcement fine-tuning rewards in [Model routing and cascades](../../ai-platform-engineering/11-model-routing-and-cascades/README.md). What crash survival would require (journals, ownership, recovery) is the subject of [Durable orchestration and workers](../../ai-platform-engineering/05-durable-orchestration-and-workers/README.md), and committing evidence before acknowledging uses the idempotency and atomic claims of the [Exactly-once event API](../03-exactly-once-event-api/README.md). Testing the tests, with mutation testing and test doubles, is part of [The testing mentality](../../software-craftsmanship/03-testing-mentality/README.md).

Where the problem shows up: for agent evaluation and simulation vendors, isolated environments with verifiable grading are the core product; AI infrastructure vendors need reward functions for reinforcement fine-tuning, which means a checker that cannot be gamed; any team shipping agents needs regression gates on tool-using behavior built on outcome, evidence, and grade; and any staff-level AI role expects you to state limits and separate claims from evidence.

## Key terms

- **Outcome, evidence, grade**: how execution ended, whether its record is complete and consistent, and whether every criterion passed.
- **Session layer**: a private copy-on-write view of a shared, immutable world, one per attempt.
- **Required and allowed sets**: the changes a task must make and the changes it may make; the state criterion is $R \subseteq \Delta \subseteq A$.
- **Replay verification**: re-executing a record's calls against a fresh session and comparing every observation and effect before grading.
- **Oracle and mutant**: a correct scripted solution that must pass, and a deliberately wrong one that must fail on an expected criterion.
- **Revision**: a hash of everything that determines a run's result, pinned before the run.

## Summary

- Completed is not passed: keep outcome, evidence, and grade separate, and define the denominator in advance.
- Private sessions, a logical clock, and sequence-derived ids make attempts isolated and replayable; the provider's nondeterminism is sampled, never the checker's.
- The checker grades the set relation $R \subseteq \Delta \subseteq A$ plus trace criteria, only on records that replay.
- Oracles and mutants test the checker; 13 oracles and 60 mutants covered 69 criteria.
- Live failures were almost all the agent's and one was the task's; attribute before fixing, budget before spending, and keep model judges out of the grade.

## Review questions

1. Why must outcome, evidence, and grade be reported separately?
   <details><summary>Answer</summary>A run can complete cleanly after editing the wrong row, and a missing or inconsistent record says nothing about the agent. Collapsing the three makes a model mistake indistinguishable from a dead worker.</details>
2. What does each half of $R \subseteq \Delta \subseteq A$ catch?
   <details><summary>Answer</summary>$R \subseteq \Delta$ fails when required work is missing; $\Delta \subseteq A$ fails when unrequested or unauthorized work was done. An attempt that edits the wrong warehouse fails both, and they are reported separately.</details>
3. What defect made replay verification necessary, and what does replay not prove?
   <details><summary>Answer</summary>An early verifier did not compare recorded read results, and substituting one turned a failing attempt into a passing one. Replay proves a record is self-consistent, not that it is authentic.</details>
4. Why was the model judge removed?
   <details><summary>Answer</summary>Its judgments disagreed across samples, one sample in five approved an attempt with zero writes, and it read untrusted text written by the agent it was judging. Prose criteria now stay `not_reviewed` until a person reviews them.</details>
5. Why is the cheaper model in the cost-gate example the more expensive result?
   <details><summary>Answer</summary>Cost must be divided by accepted tasks: $10 for 100 accepted tasks is $0.10 each, while $6 for 40 accepted tasks is $0.15 each.</details>
