# Case Study: Grounded SQL Agent

The task sounds like a prompt template: turn a natural-language question into SQL against a database the model has never seen, run it, and return the rows. In practice it is three hard problems at once. The model does not know the schema, the SQL it writes is untrusted input that reaches a database, and "it looks right" is not a measurement. This case study builds an agent that handles all three, and its most important result is not an architecture but an ablation: measured on the same questions, schema discovery took an open model from 8 of 10 correct to 10 of 10 and matched a frontier model's accuracy at roughly a fifteenth of the cost.

The runnable piece is [`safety_gate.py`](safety_gate.py), the deterministic SQL classifier, standard library only. Run it with `python safety_gate.py`; no API key is needed. The rest of the study explains the agent around it, the evidence for each design choice, and what public benchmarks can and cannot tell you about a system like this.

> **Learning objectives**
> - Explain why grounding through schema tools, not model size, closed the accuracy gap, and what the benchmark could not measure.
> - Design bounded self-correction in which database errors become correction signals inside hard resource caps.
> - Order safety layers by strength and explain why a deterministic allowlist classifier, not the model, guards execution.
> - Score answers by result-set equivalence and trust the execution record over the model's narration.
> - Read text-to-SQL leaderboards critically, including the evidence that their gold answers are often wrong.

## Before you begin

You need SQL and a working idea of what an LLM tool call is. The evaluation theory behind this study is in [LLM evaluation](../29-evaluation/README.md). Run the safety gate first, then try to get a write past it; the `BLOCKED` list in the file is a catalog of the ways a plausible classifier fails.

## The problem

A user asks "what are the top 5 best-selling genres by total sales?" against an arbitrary relational database. The system must produce SQL, run it, and return rows. Three things make it harder than a prompt template:

1. The model does not know the schema. Table and column names are the entire difficulty, and guessing them is the dominant failure mode.
2. Generated SQL is untrusted input that reaches a database.
3. "It looks right" is not a measurement. Something has to score it.

Read the ablation in the next section before the architecture. It is the evidence that justifies most of the design, and without it the schema tools look like over-engineering.

## Grounding: the measured case

The same agent was run two ways over the same ten questions: *raw*, with one query tool and no information about the tables, and *grounded*, with schema discovery tools available. The results:

| Configuration | Executed | Correct | p50 latency | Cost/query |
|---------------|----------|---------|-------------|------------|
| Open 120B, raw | 9/10 | 8/10 | 6086 ms | $0.0012 |
| Open 120B, grounded | 10/10 | **10/10** | 3640 ms | $0.0014 |
| Frontier model, raw | 9/10 | 9/10 | 3998 ms | $0.0113 |
| Frontier model, grounded | 10/10 | **10/10** | 3820 ms | $0.0202 |

There are two readings, and the second matters more. First, grounding is what closes the gap: the open model gains two questions and gets *faster*, because it stops burning attempts on guessed column names. Second, the benchmark flatters the raw configurations. These questions ran against a well-known public sample schema that both models have partly memorized, and the raw runs lean on that memory, which a private customer schema will not provide. The honest conclusion is not "raw is nearly as good" but "this benchmark cannot tell you how bad raw is."

That second point is the transferable one: know which part of your benchmark your system is cheating on. A public dataset measures capability and leaks memorization at the same time, and only the ablation exposes which one you measured. The economic conclusion holds regardless. A grounded open model matched a grounded frontier model on accuracy at about 1/15 the cost per query.

## Schema discovery as tools, not context

Grounding is delivered through two deliberately separate tools. `list_schemas()` browses table names and row counts; on a small database it returns full detail in the first call. `fetch_schemas(tables)` returns full columns, types, primary keys, and foreign keys for the named tables.

The alternative, stuffing the whole schema into the system prompt, works until it does not: at hundreds of tables it exhausts the context and buries the relevant tables in noise. The tool split lets the model narrow first and fetch detail second.

This design also has a known breaking point. At warehouse scale, `list_schemas()` is itself too large. The next step is a retrieval layer over table and column descriptions that fetches only the top-k relevant tables per question, which turns schema discovery into a retrieval problem and hands it to [Retrieval and RAG](../27-retrieval-and-rag/README.md). Knowing the point at which your design breaks is part of the design.

## Bounded self-correction

Grounding prevents most bad guesses; self-correction handles the rest. When generated SQL fails, the error goes back to the model as a tool error, and the model tries again within a fixed budget:

```
generate -> classify -> execute
                |            |
                |            +-- database error --> return to model as tool error
                +-- blocked ---> return the reason as a tool error
                                          |
                                   retry, up to max_retries
```

Two properties make this converge rather than loop. The error is specific: "No such column: Genre.Title" is a usable correction signal, while "Query failed" is not. And the budget is hard: retries, model requests, tool calls, total tokens, result rows, per-request timeout, and a wall-clock deadline are all capped. Without a ceiling, a confused agent is an unbounded bill.

In the ablation, the single raw-configuration failure was exactly this budget doing its job. The model exhausted its retries guessing at column names, and the request was cut off rather than allowed to spiral. Returning a database error inside a fixed retry budget is what turns a failed query into a successful one without any prompt engineering.

## Safety as layers, with the model outside all of them

A prompt saying "read only" is a preference. A classifier that refuses to execute anything else is a control. The agent has three layers, ordered by strength:

| Layer | What it does | Can an application bug bypass it? |
|-------|--------------|-----------------------------------|
| Deterministic classifier | Blocks writes, admin, multi-statement; caps rows | Yes |
| Read-only connection | Re-checks at the data boundary (`PRAGMA query_only`, read-only DSN) | Yes |
| `SELECT`-only database role | Refuses writes at the engine | **No** |

The database role is the only real control. The other two exist to fail fast, to give the model a correctable error, and to protect deployments where the role cannot be set. Ordering them by strength, rather than treating them as three equal "safety features," is what keeps the design honest.

The classifier contains no model, and that is the point: it cannot be prompt-injected, and it behaves identically every run. Its interesting content is lexical, not logical. Each of these queries defeats a keyword-matching classifier:

```sql
SELECT * FROM t WHERE name = 'Begin Again'    -- contains BEGIN, is a read
SELECT 'a;b' AS x                             -- contains ';', is ONE statement
SELECT * INTO archive FROM users              -- leads with SELECT, WRITES
SELECT * FROM (SELECT ... LIMIT 5)            -- has LIMIT, is NOT bounded
```

The implementation therefore scans the query once into two parallel strings of equal length: a *cleaned* copy that is safe to execute, and a *masked* copy in which the contents of literals are blanked, so that text inside a string can never lie to a check. And it is an **allowlist**: anything not positively recognized as a read is refused. A denylist of bad keywords fails open on the first syntax nobody anticipated.

## Evaluation by result-set equivalence

With execution safe, the remaining question is whether the answer is right. The evaluator compares **returned rows** against gold rows, not SQL text against gold SQL. It ignores column naming and row order where those carry no meaning, and accepts projection differences only when every gold value is reproduced with the same row count.

This matters because there are many correct queries for one question. Two correct queries rarely look alike, so scoring SQL strings measures whether the model writes SQL the way you do; scoring result sets measures whether it answered the question.

Alongside correctness, every answer records rows returned, attempts, latency, token usage, model calls, and estimated cost. A prompt change that buys two points of accuracy for triple the cost is then visible as exactly that trade.

## Trust the execution, not the narration

The executed SQL and the rows it returned are stored by the tool and treated as the source of truth. The model's final structured message is useful metadata and nothing more. An agent that reports "I found 5 genres" after its query returned 3 rows is reporting an intention. Systems that trust the narration are the ones that confidently return numbers nobody ran.

## What the public benchmarks can and cannot tell you

Ten questions on a sample schema prove an approach. They do not predict accuracy on a customer's warehouse, and neither do the public leaderboards. The figures in this section were reviewed on September 26, 2026.

### The scores

The two benchmarks most often cited are:

| Benchmark | What it is | Reported results |
|-----------|-----------|------------------|
| [BIRD](https://bird-bench.github.io/) | 12,751 question and SQL pairs, 95 databases, 37 domains, scored by execution accuracy | Human baseline 92.96%. Gemini-SQL2, announced June 12, 2026 on Gemini 3.1 Pro, leads the single-model track at 80.04% |
| [Spider 2.0](https://arxiv.org/abs/2411.07763) | 632 enterprise workflow tasks; schemas often over 1,000 columns; BigQuery and Snowflake dialects | At release an o1-preview agent solved 21.3%, against 91.2% on Spider 1.0 and 73.0% on BIRD |

Two details in the BIRD results matter more than the top line. General frontier models trail: on the single-model track, GPT-5.5 and Claude Opus 4.6 were reported at roughly 70 to 73%, twenty points or more under the human baseline. And specialized mid-size models sit above them: three tuned 32B systems from data platform vendors were reported between about 74% and 76%. A narrow model trained for the task beats a larger general one, which is the same economic argument as the grounding ablation, arriving from a different direction.

Gemini-SQL2 was announced with a score and no paper, method, or public model ID. A number without a method is a claim to track, not a technique to copy.

### Why models fail on real schemas

The Spider 2.0 authors and the BIRD results point at the same causes:

| Cause | What it looks like |
|-------|--------------------|
| Schema linking at scale | The right three columns out of three thousand |
| Business logic outside the schema | "Active customer" is defined in a wiki, not a table |
| Value grounding | The filter is `'BR'`, the question said "Brazil" |
| Dialect | A function that exists in one engine and not another |
| Ambiguity | The question has two defensible readings |
| Multi-step workflows | Several dependent queries, over 100 lines |

None of these is fixed by a larger model alone. They are retrieval, grounding, and specification problems, which is why this study spends its effort on schema tools and an eval harness instead of on model choice.

### The gold answers are often wrong

An audit published at CIDR 2026, [*Text-to-SQL Benchmarks are Broken*](https://www.vldb.org/cidrdb/papers/2026/p5-jin.pdf), found annotation errors in 52.8% of BIRD Mini-Dev problems and in 66.1% of the 121 Spider 2.0-Snow problems with published gold SQL. Re-scoring five leading agents on a corrected sample of 100 BIRD Dev problems moved results by -2 to +19 points and reordered the ranking: one agent went from 62% to 81%, and from fourth to first. The audit classified the errors into four patterns:

| Error pattern | Meaning | Share in BIRD Mini-Dev |
|---------------|---------|------------------------|
| E1 | Gold SQL does not match the question's logic | 29.3% |
| E2 | Gold SQL misreads the data or schema | 57.8% |
| E3 | Gold SQL contradicts domain knowledge | 10.7% |
| E4 | The question is ambiguous | 29.7% |

Read leaderboards with that in mind. Matching a flawed answer key is not the same as answering correctly, and a system tuned to the key can score above a system that is right. The Spider 2.0-Snow leaderboard now lists entries above 96%; hold that next to the audit before drawing a conclusion from either.

### Clean data moved the ceiling more than pipeline engineering did

A follow-up from the same group, [*Human-Level Text-to-SQL via Reinforcement Learning on Verified Data*](https://arxiv.org/abs/2603.20004), corrected 61% of a 2.5k-instance training sample and reports reaching the 92.96% human baseline on an expert-verified version of BIRD by fine-tuning with verifiable rewards, without a multi-stage pipeline. It also names two failure modes that apply to any harness built like the one in this study.

> **Warning:** Result-based rewards have false positives. A wrong query can return the right rows on a small database, so result-set equivalence is necessary but not sufficient. Add a second check on hard cases.

> **Warning:** Models ignore supplied knowledge. Given a definition alongside the question, the model often does not use it. Score whether the hint was applied, not only whether the rows matched.

### What to do with this

Four practices follow from the evidence:

1. Build a private eval set on the customer's own schema. Public sets leak into training and carry errors you cannot see.
2. Audit your own gold answers. If published benchmarks are half wrong, a hand-written set of fifty is not exempt.
3. Report the denominator. "10/10 on ten public questions" is an honest sentence. "100% accurate" is not.
4. Treat a verified eval set as the asset. It is the regression gate today and the reward function for fine-tuning later; see [tuning the router](../24-model-routing/04-escalation-economics-and-evaluation.md#tuning-the-router-a-ladder) in *AI Engineering*.

## The build log

The agent was assembled in this order:

1. **The eval set first.** Ten questions with gold answers, and a scorer that compares result sets. Before any agent code, decide what "right" means.
2. **The safety gate second**, with its own test suite. It is pure and deterministic, so it is the cheapest part of the system to get right, and everything downstream can assume it.
3. **A minimal agent**: one `execute_query` tool, no schema access. This is the raw baseline, and it exists to be beaten.
4. **Measure the baseline.** 8/10 on an open model. Without this number, every later improvement is a story rather than a result.
5. **Add schema tools** and re-run the same eval. 10/10, and faster.
6. **Add bounded retry** so tool errors become correction signals rather than failures, with hard caps on every resource.
7. **Add the read-only connection layer**, re-checking at the data boundary.
8. **Run the ablation grid**: raw versus grounded, open versus frontier, same questions. This is what turned "grounding seems better" into a cost argument.
9. **Record the artifacts.** Eval report, latency benchmark, per-question results, committed. A claim without a committed artifact is a memory.

The sequence is the lesson: the measurement infrastructure was built before the thing being measured, and every architectural claim here is backed by a number that existed before the architecture did. Before writing agent code for your own domain, design its eval harness. If you cannot state how you would score an answer, you cannot tell whether a change helped.

## Techniques to take away

The techniques generalize to most tool-using agents:

| Technique | When to apply |
|-----------|---------------|
| Ablate the grounding | Any RAG or tool-augmented system, to prove retrieval earns its cost |
| Ask what the benchmark leaks | Any public dataset a model may have memorized |
| Audit the gold answers | Any eval set, public or your own, before trusting a score |
| Deterministic gate before execution | Any generated code, query, or command that will run |
| Allowlist, not denylist | Any classifier where unknown input must fail closed |
| Lex before you match keywords | Any rule applied to code or query text |
| Tool errors as correction signals | Any agent that can retry |
| Hard resource ceilings | Every agent loop, without exception |
| Score outputs, not artifacts | Any task with many correct forms |
| Trust the execution record | Any agent that both acts and narrates |
| Cost and latency per answer | Any LLM feature that will run more than once |

## Where this leads

The theory this study applies, including eval harnesses, scoring, and LLM-as-judge, is in [LLM evaluation](../29-evaluation/README.md). Schema discovery becomes retrieval at warehouse scale in [Retrieval and RAG](../27-retrieval-and-rag/README.md). Bounded retry generalizes to escalation across model tiers, and eval sets become tuning rewards, in [Model routing and cascades](../24-model-routing/README.md). Why the database role is the real control is a matter of least privilege and defense in depth, covered in [Authorization and access control](../../software-engineering/10-authorization/README.md). The delivery half of the same agent, with tool loops, streaming, and token budgets, is [Streaming and SSE](../../systems/04-streaming/README.md). And testing a system with a stochastic component through test doubles and deterministic tests is part of [The testing mentality](../../software-engineering/03-testing/README.md).

Where the problem shows up: AI infrastructure vendors argue grounded open models against frontier APIs on cost; data platforms offer natural-language querying over customer schemas, where safety and schema scale dominate; any agent product needs tool loops, retry budgets, and injection resistance; and any staff-level AI role expects you to measure before you architect.

## Key terms

- **Grounding**: giving the model the facts it needs, here the schema, through tools rather than relying on its memory.
- **Ablation**: running the same evaluation with one component removed to measure what that component contributes.
- **Bounded self-correction**: returning errors to the model as tool errors and retrying within hard caps on every resource.
- **Allowlist classifier**: a gate that refuses anything not positively recognized as safe.
- **Result-set equivalence**: scoring an answer by comparing returned rows with gold rows rather than comparing SQL text.
- **Execution record**: the SQL actually run and the rows actually returned, treated as the source of truth.

## Summary

- Grounding through schema tools took an open model from 8/10 to 10/10 and matched a frontier model at about 1/15 the cost, while the public schema hid how bad the raw configuration really is.
- Specific errors plus hard caps turn failed queries into corrections without letting a confused agent spiral.
- Safety layers are ordered by strength; the `SELECT`-only role is the only real control, and the deterministic allowlist classifier lexes before it matches.
- Answers are scored by result sets and recorded with cost and latency; the execution record, not the narration, is the truth.
- Public text-to-SQL benchmarks carry large gold-answer error rates, so a private, audited eval set is the real asset.

## Review questions

1. Why does the grounding ablation flatter the raw configurations?
   <details><summary>Answer</summary>The questions ran against a well-known public sample schema that both models have partly memorized, so the raw runs leaned on memory that a private customer schema will not provide. The benchmark cannot tell you how bad raw is.</details>
2. Why split schema discovery into `list_schemas()` and `fetch_schemas(tables)` instead of putting the schema in the prompt, and where does that design break?
   <details><summary>Answer</summary>A full schema in the prompt exhausts context and buries relevant tables at hundreds of tables; the split lets the model narrow first and fetch detail second. At warehouse scale `list_schemas()` is itself too large, and schema discovery becomes a retrieval problem over table and column descriptions.</details>
3. Explain why `SELECT * INTO archive FROM users` and `SELECT * FROM (SELECT ... LIMIT 5)` defeat a keyword classifier, and how the gate avoids that.
   <details><summary>Answer</summary>The first leads with SELECT but writes; the second contains LIMIT but is not bounded at the outer level. The gate lexes the query into a cleaned copy and a masked copy with literal contents blanked, applies structural checks, and is an allowlist, so anything not positively recognized as a bounded read is refused.</details>
4. Which safety layer is the real control, and what are the others for?
   <details><summary>Answer</summary>The `SELECT`-only database role, because an application bug cannot bypass it. The classifier and read-only connection fail fast, give the model a correctable error, and protect deployments where the role cannot be set.</details>
5. Why is result-set equivalence necessary but not sufficient?
   <details><summary>Answer</summary>Many correct queries exist for one question, so comparing rows is the right test of correctness. But a wrong query can return the right rows on a small database, so hard cases need a second check.</details>
