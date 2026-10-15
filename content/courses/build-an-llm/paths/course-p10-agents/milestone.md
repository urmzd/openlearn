# Pass 10 milestones

**Pass result**: Go agent SDK on their engine's tool calls, RAG over their docs, eval runner with judge and A/B, usage policy at the gateway, agent chart; optional post-training.

The path includes the milestone stages below. Run each component milestone after its modules pass, then run the pass gate. The component gates run before the pass gate, which also reruns the smoke steps of earlier passes.

| Gate | Specification | What it covers |
|---|---|---|
| `MS-agent` | [`MS-agent.toml`](../../../../../course/milestones/MS-agent.toml) | A durable agent answers with citations and passes evals and policy checks. Requires `ag.01`, `ag.02`, `ag.03`, `ag.04`, `ag.05`, `ag.06`, `ag.07`, `ag.08`, `ag.09`, `ag.10`, `ag.11`, `ag.12`, `ethics.05`, `gw.08`, `dep.07`, `craft.23`. |
| `MS-P10` | [`MS-P10.toml`](../../../../../course/milestones/MS-P10.toml) | Agents, usage policy, durable runs, and agent deployment. Requires `ag.01`, `ag.02`, `ag.03`, `ag.04`, `ag.05`, `ag.06`, `ag.07`, `ag.08`, `ag.09`, `ag.10`, `ag.11`, `ag.12`, `ethics.05`, `gw.08`, `dep.07`, `craft.23`, `S-M10b`. |
| `MS-C2` | [`MS-C2.toml`](../../../../../course/milestones/MS-C2.toml) | Post-training improves the TinyStories model on held-out preference tasks. Requires `C2`, `L12.1`, `L12.2`, `L12.3`, `L12.4`, `S-M10b`. |


## Component gate details

## MS-C2: Post-training improves the TinyStories model on held-out preference tasks

Optional local post-training capstone. The v1.0.0 core course ends at C1.

Run a gate with `practice/bin/ol milestone <ID> --smoke`; omit `--smoke` for its full local and cluster steps.
