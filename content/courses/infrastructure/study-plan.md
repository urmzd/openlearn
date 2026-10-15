# Infrastructure (5-7 weeks)

~8-10 hours per week. How distributed systems actually run. Topics 1-3 stay anchored to the Streamflow event-driven platform.

### Weeks 1-2: Containers, Kubernetes & Workloads

| Week | Focus | Reference |
|------|-------|-----------|
| 1 | Containers (namespaces/cgroups/layers), multi-stage + distroless, 12-factor | Docker / 12factor.net |
| 2 | K8s as a reconcile loop; objects; stateful vs stateless; caveats (limits/OOM, probes, SIGTERM, PDB); scaling (HPA/VPA/Cluster Autoscaler/KEDA) | Kubernetes docs |

### Weeks 3-4: Messaging & Distributed Workers

| Week | Focus | Reference |
|------|-------|-----------|
| 3 | Queue vs log, push/pull/long-poll, pub-sub vs competing consumers, delivery semantics, Little's Law; ZooKeeper vs KRaft vs Redpanda | Kafka docs + DDIA Ch 11 |
| 4 | Consumer groups, partition cap, rebalancing, idempotency/outbox, DLQ, lag → KEDA scaling | Kafka consumer docs |

### Weeks 5-6: Search & the Apache Stack

| Week | Focus | Reference |
|------|-------|-----------|
| 5 | Inverted index, TF-IDF → BM25, Lucene segments/merges, Elasticsearch vs OpenSearch vs Solr | Lucene / ES Guide |
| 6 | The Apache data ecosystem map: messaging, storage/tables, OLAP, search, orchestration | Apache docs + DDIA |

---
