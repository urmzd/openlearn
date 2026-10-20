# Build Your Own LLM System: solutions and contracts

This directory holds the complete reference system that the [Build Your Own LLM System](../content/courses/build-an-llm/README.md) chapters explain. Nothing here is graded. Each chapter quotes the parts of these files that carry its idea, explains why each line is there, and links the full file.

| Directory | What it holds |
|---|---|
| [`solutions/python/`](solutions/python/) | `tinyllm`: numerics, autograd, tokenizers, models, training, data, and evals |
| [`solutions/rust/`](solutions/rust/) | The candle inference engine, the fast tokenizer, and the OpenAI-compatible server |
| [`solutions/go/`](solutions/go/) | The control plane: gateway, durable workflow engine, agent SDK, and load generator |
| [`solutions/c/`](solutions/c/) | Standalone kernels and systems data structures |
| [`solutions/entry/`](solutions/entry/) | Entry points, configuration, deployment manifests, and scripts that run the whole system |
| [`solutions/docs/`](solutions/docs/) | Model answers for the written modules: ADRs, runbooks, postmortems, model and data cards, threat model |
| [`solutions/learner-tests/`](solutions/learner-tests/) | Model test suites for the modules about testing |
| [`solutions/history/`](solutions/history/) | Earlier versions of units that a later module rewrote, so a chapter can show the version it teaches |
| [`solutions/primers/`](solutions/primers/) | Solutions to the language and tool primers |
| [`solutions/testkit/`](solutions/testkit/) | The fault and determinism kit (fake clock, failpoints, kill loop, chaos proxy, OTLP sink) the model test suites import |
| [`solutions/fixtures/`](solutions/fixtures/) | Small data files the solution tests read through `TINYLLM_FIXTURES` |
| [`contracts/`](contracts/) | The interfaces every component implements: Python stubs, Rust traits, Go interfaces, C headers, `.proto`, OpenAPI, and file-format specs |

## How to use it

Read a chapter first. Its principles and worked example derive the design, and its solution section walks through the code here in order. Open the full file when you want the surrounding detail: input validation, error types, and the helpers the walkthrough skips.

Before running a solution, predict its output or state changes and name the invariant it should preserve. Choose a boundary case or controlled failure that could challenge your explanation. Compare the result with your prediction, then validate it against the contract and the original requirement. Generated code needs the same checks: ease of producing an implementation does not establish that it is correct. See [How to read these books](../content/references/how-to-read.md) for the shared method.

The contracts are the stable boundary between components. When a chapter says a later module "takes over" a unit, the contract stays fixed and the implementation behind it improves; `solutions/history/` keeps the earlier version.
