# Course Pass 1: The Tracer

A thin system through every layer, before any layer is deep. You learn Python, Rust, HTTP and SSE, Go, and containers as you need them. You build a Python matmul and a byte-level bigram fitted by counting, a Rust engine that streams completions from your checkpoint, a Go gateway that checks a key and streams without buffering, images and Helm charts on kind, one trace across gateway and engine in Jaeger, your first incident drill with its runbook, and your first architecture decision record. C remains optional depth later in the course.

**Part of**: [the course](../course/). About 5 weeks at 10 to 12 hours a week. **Needs**: [Pass 0](../course-p00-setup/) and its gate MS-P0.

**Gate**: [MS-P1](milestone.md), every layer yours and running end to end.

The stages, their chapters, and what "done" means for each are in [`path.tsv`](path.tsv). The engine serves the checkpoint your CLI wrote.

```bash
practice/bin/ol learn course-p01-tracer         # stages and progress
practice/bin/ol learn course-p01-tracer next    # read the next unfinished stage
```

Pass 1 needs a local kind cluster for stages 8 to 11 (`ol doctor --pass 1`). Everything before stage 8 runs as local processes.
