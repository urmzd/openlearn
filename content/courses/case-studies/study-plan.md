# Study Plan

The case studies are best read after the books that carry their theory, as the applied pass: you will get more from the order book once heaps and amortized analysis are familiar, and more from the SQL agent once you know what an evaluation harness is for. Each study rewards about half a day of focused work, or two shorter sittings, and an afternoon is enough if you only read. The pattern for every study is the same: read the concepts, run the implementation, then break it deliberately and watch the tests catch you.

Start with [Order book matching](01-order-book-matching/README.md). Before reading past the three rules, write the flat-list version yourself; it takes about fifteen minutes and gives you something to judge the later layers against. Then do the bottleneck, unnecessary-work, and duplicated-work analysis out loud before you read the heap layer, so that the heap is something you derived rather than something you were told.

Move on to the [Grounded SQL agent](02-grounded-sql-agent/README.md). Spend real effort trying to get a write past the safety gate, because the queries that nearly work are the lesson. Then sketch an evaluation harness for a domain of your own before rereading the build log, and notice that the measurement came before the architecture.

The [Exactly-once event API](03-exactly-once-event-api/README.md) is next. Design the endpoints yourself first, then replace the constraint dedup with a check-then-act version and watch the concurrency test fail. Seeing that failure is worth more than any amount of reading about races.

With [K-means optimization](04-kmeans-optimization/README.md), predict the timing sweep before you run it, then explain out loud why dropping `sqrt` is safe. If the monotonicity argument does not come immediately, that is the gap worth closing before moving on.

Finish with the [Agent evaluation harness](05-agent-eval-harness/README.md), the longest study and the one that draws most on the others. Read the table the implementation prints, add a mutant of your own and predict which criteria it fails, and write your own protocol for a model swap before reading the one in the study.

Throughout, keep [Lessons from practice](../software-craftsmanship/04-lessons-from-practice/README.md) from *Software Craftsmanship* nearby. It catalogs the failure modes these builds were avoiding, and reading the two side by side shows each safeguard alongside the incident it prevents.
