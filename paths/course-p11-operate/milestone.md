# Pass 11 milestones

**Pass result**: v1.0.0: migrations (KV v2, API v2 as build modules), upgrades, bisect, chaos, security, docs, reviews, mock engagement, defense.

The path includes the milestone stages below. Run each component milestone after its modules pass, then run the pass gate. The component gates run before the pass gate, which also reruns the smoke steps of earlier passes.

| Gate | Specification | What it covers |
|---|---|---|
| `MS-ops` | [`MS-ops.toml`](../../course/milestones/MS-ops.toml) | Core incident drills are detected, resolved, and documented. Requires `ops.01`, `ops.02`, `ops.03`, `ops.04`, `ops.05`, `ops.06`, `ops.07`, `ops.08`, `ops.09`, `ops.10`, `ops.11`, `ops.12`. |
| `MS-P11` | [`MS-P11.toml`](../../course/milestones/MS-P11.toml) | v1.0.0 is released, migrated, reviewed, and handed off. Requires `craft.09`, `craft.10`, `craft.11`, `craft.12`, `craft.13`, `craft.14`, `craft.15`, `craft.16`, `craft.17`, `craft.18`, `craft.19`, `ops.04`, `ops.05`, `ops.06`, `ops.07`, `ops.08`, `ops.09`, `ops.10`, `ops.11`, `ops.12`, `review.02`, `review.03`, `ethics.06`, `field.01`, `field.02`, `field.03`, `field.04`, `field.05`, `field.06`, `field.07`, `iv.01`. |


## Component gate details



Run a gate with `practice/bin/ol milestone <ID> --smoke`; omit `--smoke` for its full local and cluster steps.
