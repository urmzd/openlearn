# Case Studies (4-8 days)

~4-6 hours per study. Each is a worked build: read the README, run the
implementation, then break it deliberately and watch the tests catch you.
Best taken *after* the track that carries the theory, as the applied pass.

| Day | Study | Do this |
|-----|-------|---------|
| 1-2 | [Order Book Matching](../../case-studies/01-order-book-matching) | Write the flat-list version yourself first, then do the BUD analysis before reading the heap layer |
| 3-4 | [Grounded SQL Agent](../../case-studies/02-grounded-sql-agent) | Try to get a write past the safety gate; design an eval harness for a domain of your own |
| 5-6 | [Exactly-Once Event API](../../case-studies/03-exactly-once-event-api) | Replace the constraint dedup with check-then-act and watch the concurrency test fail |
| 7-8 | [K-Means Optimization](../../case-studies/04-kmeans-optimization) | Predict the timing sweep before running it; explain why dropping `sqrt` is safe |

Pairs with [Lessons from Practice](../../tracks/software-craftsmanship/04-lessons-from-practice),
which catalogues the failure modes these builds were avoiding.

---
