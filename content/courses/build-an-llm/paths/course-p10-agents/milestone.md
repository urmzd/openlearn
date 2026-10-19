# What the system does after Pass 10

**Pass result**: Go agent SDK on their engine's tool calls, RAG over their docs, eval runner with judge and A/B, usage policy at the gateway, agent chart; optional post-training.

Each capability below is built from the listed modules. The sections that follow give the commands the finished solution runs, what each one produces, and why each piece is needed. Everything from earlier passes keeps working.

| What the system does | Built from |
|---|---|
| A durable agent answers with citations and passes evals and policy checks. | `ag.01`, `ag.02`, `ag.03`, `ag.04`, `ag.05`, `ag.06`, `ag.07`, `ag.08`, `ag.09`, `ag.10`, `ag.11`, `ag.12`, `ethics.05`, `gw.08`, `dep.07`, `craft.23` |
| Agents, usage policy, durable runs, and agent deployment. | `ag.01`, `ag.02`, `ag.03`, `ag.04`, `ag.05`, `ag.06`, `ag.07`, `ag.08`, `ag.09`, `ag.10`, `ag.11`, `ag.12`, `ethics.05`, `gw.08`, `dep.07`, `craft.23`, `S-M10b` |
| Post-training improves the TinyStories model on held-out preference tasks. | `C2`, `L12.1`, `L12.2`, `L12.3`, `L12.4`, `S-M10b` |


## Details

## Post-training improves the TinyStories model on held-out preference tasks

Optional local post-training capstone. The v1.0.0 core course ends at C1.