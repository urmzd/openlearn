# review.01 problems: design review of the tracer and the serving platform

Answer every question in `solve/review.01.toml` (written by `ol start review.01`).
The tag after each question is its answer type: `[number]` is a value, exact
where the question says so (`23040`, `1/50`), and `[proof]` a document in
`solve/review.01/` graded against its rubric. q1 to q6 are the numbers a
capacity section of your design must get right; q7 is the design document
and q8 the review of it.

### Capacity and budgets

Use SmolLM2-135M: 30 layers, 9 query heads, 3 key/value heads, head dimension 64.
The KV cache is format v1, float16 (2 bytes per value), and stores one key and
one value vector per layer, key/value head, and token.

**q1.** How many bytes of KV cache does one token of one sequence occupy? `[number, exact]`

**q2.** The engine's `runtime.toml` sets `kv_blocks = 2048` and `block_size = 16` (tokens per block). (a) How many bytes does the whole KV pool hold? (b) How many sequences of 512 tokens each (prompt plus output) fit in the pool at once? `[number, exact]`

**q3.** Requests arrive at 4 per second. Each streams 32 output tokens, with time to first token 0.5 s and 0.06 s between consecutive output tokens. By Little's law, how many requests are in flight on average? `[number]`

**q4.** The availability SLO is 99.5% of requests without a 5xx over 30 days. Expressed as a full outage, how many minutes of error budget is that? `[number, exact]`

**q5.** A fast-burn alert fires at a burn rate of 14.4 sustained over its 1 hour long window. What fraction of the 30-day error budget has that hour spent? `[number, exact]`

**q6.** At a constant burn rate of 14.4, how many hours until the whole 30-day budget is gone? `[number, exact]`

### The design and its review

**q7.** Write the design document of your system as it stands at the end of Pass 7: the tracer of Pass 1 and the serving platform built on it (engine, gateway, load generator, deployment, observability). Graded against `course/rubrics/design-review.md`: the problem, users, and constraints first; at least two options compared on the same criteria with numbers; what the chosen option gives up; the contract of every interface; failure modes with detection and degradation; capacity and latency budgets with the arithmetic (q1 to q6 are part of it); rollout and rollback, including contract and data migrations; open questions with an owner. `[proof]`

**q8.** Review q7 as a reviewer who did not write it (a peer, or you a day later): answer each line of the design-review rubric, record every "no" and every number you recomputed as a comment, and end with a decision. Graded against its own rubric (chapter section 4). `[proof]`
