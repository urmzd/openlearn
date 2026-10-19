# Course Pass 1: The Tracer

A thin system through every layer, before any layer is deep. The pass introduces Python, Rust, HTTP and SSE, Go, and containers as they are needed. The solution has a Python matmul and a byte-level bigram fitted by counting, a Rust engine that streams completions from that checkpoint, a Go gateway that checks a key and streams without buffering, images and Helm charts on kind, one trace across gateway and engine in Jaeger, a first incident drill with its runbook, and a first architecture decision record. C remains optional depth later in the course.

**Part of**: [the course](../course/). About 5 weeks at 10 to 12 hours a week. **Needs**: [Pass 0](../course-p00-setup/).

**After this pass**: [what the system does](milestone.md): every layer exists and runs end to end.

The stages, their chapters, and what "done" means for each are in [`path.tsv`](path.tsv). The engine serves the checkpoint the `tinyllm` CLI wrote.

Stages 8 to 11 run on a local kind cluster. Everything before stage 8 runs as local processes.
