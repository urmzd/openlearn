# Changelog

Notable user-facing changes are recorded here. Entries describe observable behavior and link to the migration or recovery procedure when an operator must act.

## [1.0.0] - 2026-10-09

First stable release of the TinyLLM learning platform and reference implementation.

### Added

- OpenAI-subset completion and embedding APIs at the gateway and engine.
- Python training and corpus tools, a Rust inference engine, and Go gateway and durable services.
- Versioned model, tokenizer, KV, corpus, and workflow file formats.
- Runbooks, telemetry, and rollback guidance for serving and durable workloads.

### Compatibility

- This release establishes the v1 API and KV envelope compatibility windows described in [the deprecation policy](../docs/DEPRECATION.md).

### Verification

- Release source is the commit carrying the annotated `v1.0.0` tag.
- The tag is created only after the required CI workflow passes on that commit.
