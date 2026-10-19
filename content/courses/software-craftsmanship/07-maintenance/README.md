# Maintenance

Most engineering time is spent changing systems that already work. A system that cannot change safely is already failing slowly, so maintenance is a skill in its own right, with its own techniques: versioned formats, dual-read migrations, dependency hygiene, and bisection. This chapter applies them to the running system, after its first stable release.

Every migration here follows the same shape. Make the change additive, prove both sides work with tests, roll out behind a switch, and remove the old path only when telemetry shows nobody uses it. Concretely, an interface changes in phases: add the new version, read both, write the new, migrate the data, stop reading the old, and remove it. Each phase is deployable and reversible.

The first two sections are interface migrations carried out under the deprecation policy of the previous chapter. The KV cache format moves from f16 to fp8 E4M3 blocks with per-(layer, head) scales, with readers upgraded before writers and mixed versions negotiated during rollout. The public API moves from v1 to v2 under the [`openai-subset.v2`](../../../../course/contracts/openapi/openai-subset.v2.yaml) contract, served alongside v1 with `Deprecation` and `Sunset` headers. The last two sections deal with change you did not write and change you did not notice. A dependency upgrade is reviewed like any other change: pin it, read the changelog, run the full suite, and roll it out, checked against the allowlist in `allowed-deps.toml`. And a performance regression is found by search rather than by staring: a benchmark with a threshold plus `git bisect run` finds the commit in $\log_2 n$ steps.

Each section has a matching incident drill in [Incident Response and Chaos](../../systems/05-incident-response-and-chaos/), in *Systems and Architecture*, which rehearses the same procedure on the cluster under pressure.

> **Learning objectives**
> - Migrate a binary format and an HTTP API in reversible phases, with readers or both versions live before the switch.
> - Validate untrusted input completely before use and make offline conversions atomic and one-way.
> - Review a dependency upgrade by its effects, with the merge rule, owner, and rollback fixed in advance.
> - Bisect a performance regression with a consistent oracle and confirm the result.

## Before you begin

You need [Releases and Deprecation](../06-releases-and-deprecation/), whose policy each migration follows, and [The Testing Mentality](../03-testing-mentality/), especially the benchmarks and perf gates. The migrations change parts of the running system built in *Build Your Own LLM System*: the KV block pool from [Inference](../../build-an-llm/p08-inference/) and the serving stack from [Serving](../../build-an-llm/p10-serving/). The fp8 encoding behind KV v2 is explained in [FP8 and microscaling formats](../../mathematics-for-computing/09-numerical-methods-and-floating-point/04-fp8-and-microscaling-formats.md) in *Mathematics for Computing*. A useful exercise before the last section is to practise `git bisect run` on a small repository with a planted regression.

## In this chapter

1. [Interface migration: KV format v1 to v2](01-kv-format-v1-to-v2.md): a reader for both formats, per-slab fp8 scales worked by hand, complete validation with checked arithmetic, and an atomic one-way converter.
2. [Interface migration: API v1 to v2](02-api-v1-to-v2.md): header-based version selection in the engine and the gateway, deprecation headers, a hard sunset, and per-version metering.
3. [Dependency upgrades](03-dependency-upgrades.md): a model review of an OpenTelemetry family upgrade, built on evidence from the resolved graph, tests, vulnerability scans, and paired benchmarks.
4. [Performance regression bisect](04-performance-regression-bisect.md): a reproduction envelope, a predeclared regression rule, and a `git bisect run` gate that respects the skip and abort exit codes.

## Further reading

- *Software Engineering at Google*, ch. 21, [Dependency Management](https://abseil.io/resources/swe-book/html/ch21.html), and ch. 22, [Large-Scale Changes](https://abseil.io/resources/swe-book/html/ch22.html) (free).
- Michael Feathers, *Working Effectively with Legacy Code*.
- Martin Kleppmann, *Designing Data-Intensive Applications*, ch. 4, on encoding and evolution.
- [`git bisect`](https://git-scm.com/docs/git-bisect) documentation (free).
- [Protocol Buffers: updating a message type](https://protobuf.dev/programming-guides/proto3/#updating) (free).
