# Course build handoff

## Status update (2026-10-11, Claude subagent): read this first

Items 1 to 4 of the wrap-up brief are done: PR #22 (squash-merged as d561db0) was green on every CI job; the engine and milestone follow-ups were pushed directly to main (36b4217 to c960c88). B14 multimodal is **deferred** past its first slice: the design is merged into DESIGN.md and group M12 with S-M12 is on main; the vision and speech builds are not. Pick up from here:

- **Side modules:** L8.3 (Python paged KV cache) and load.02 (run comparison gate) are `kind = "side"`: no module calls them. MS-L8, MS-L10, and MS-prod no longer require them; dep.05's perf-gate job is checked only when present (course/deviations/L8.3.md, load.02.md).
- **Harness speed:** `ol check --all` reuses a pass recorded on the same code (`repo_hash`), the same trees of the module and its deps (`closure_hash`), course commit, and harness; deps that passed in the same run skip their cumulative smoke rerun; `--fresh` checks everything. Mutation results are content-keyed (closure reference code included) and `OL_MUTATION_CACHE` shares them; CI keeps the file between runs. On CI the first check --all of the e2e job now takes about 23 to 27 minutes, MS-P0's rerun about 20 s.
- **Milestones:** a pass gate runs each contributing milestone's steps on a freshly started service stack; a PR gate skips earlier gates' nightly steps (MS-C1). `{endpoints.*}` placeholders; suite steps take per-case `params`; new matchers `logits-close`, `disagg`, `greedy-equal`, `effects`; `trace` polls and reads Jaeger or Tempo; step keys `chaos` and `sink = "effects"`; drills run `[workload]` and grade `[[resolve.run]]`. Practice checks always skip their cluster tier under OL_SMOKE.
- **Reference system:** new entry points ctl (wf, data build, train, usage, agent ask, eval, rag), loadgen (run, compare), agent worker, and the config-form gateway (keys, limits, cache, routes, ledger, admin API, gw.08 policy, engine heartbeat); services durable, worker, agent, engine-spec, engine-prefill, engine-decode-a/b; the engine serves tiny-llama-2l-1k (same weights, 1024 positions) so tool prompts fit; reference answers for every solve set. Engine fixes: radix-held blocks count as cached; speculative decoding scores every fed position and streams; prefill and decode roles serve disaggregation; OpenAPI v2 side by side with v1 (craft.14 upgrades server.rs).
- **CI:** at e0abe0a the PR e2e (180 min, check --all 90 min) and "MS-P1 on kind" (`ol verify course --kind MS-P1`) were green; since then e2e and kind run nightly and PR shards sample 4 mutants with a kept kill cache.
- **MS-durable and MS-L10 (feat/course-2):** the 500-append kill loop and TrainRun under kills pass (not smoke); `disagg` and `greedy-equal` matchers; the engine serves prefill and decode roles and speculative decoding scores every fed position, so MS-L10's disaggregated and speculative steps pass.
- **Still open:** MS-durable-ha's kill loop (leader kills, partitions, linearizability, ports allocated before start); gateway v2 routing (`VersionedAPI` not in the chain); EngineControl Cancel/Drain stubs, no heartbeat client on the engine, resumed requests decoded outside the continuous batch; MS-C1 needs the C1 trainer (the reference trainer runs bigram only); `[services.engine-instruct]` for MS-L10's nightly step; ops.05 drill against v2; open-items.txt lines on the B9-2 serve loop are stale.
- **B14 (deferred):** on main: the DESIGN.md merge (licenses: SmolVLM-256M-Instruct and SigLIP Apache-2.0, whisper-tiny MIT/Apache-2.0, PD12M CC0, LibriSpeech CC BY 4.0, Common Voice unclear so a LibriSpeech stand-in), group M12 with S-M12, and the 56 MiB fixture budget. M12.2, M12.3, M12.5 and M12.6 fail verify check 9 (no call site) until their L13/L14/data callers exist; make them side modules or build the callers. Partial L13.1 to L13.5 and L14.1, L14.2, L14.5 builds are on the local branch `b14-wip` in the course worktree.
- **Runner notes:** give every agent its own scratch dir (`.scratchpad/w/<name>`); `uv run` at the worktree root may bump uv.lock's project version (revert it); an e2e or kind job is cancelled by any push to the branch.

You are taking over the build of the openlearn course from a Claude Code session. Finish everything below, in order. This file is the brief; `course/DESIGN.md` is the spec.

Repo: this git worktree (`.worktrees/course`, branch `feat/course-2`; earlier PRs #19 and #22 are merged). Work only inside it.

## 1. What the course is (read first)

One course where the learner builds one system end to end, module by module, and owns all of it. Read `course/DESIGN.md` sections 1, 2, 3, 5, 6, 9. Built modules are the exemplars: study `course/modules/L0.0.toml`, `L10.5.toml`, `gw.04.toml`, their references in `course/ref/`, tests in `course/tests/`, mutants in `course/mutants/`, and their chapters (path in each registry file's `chapter` field). The harness is `practice/bin/ol` plus the Python package in `course/harness` (`practice/README.md` documents the commands). Deviations already taken are in `course/DEVIATIONS.md`; append new ones.

## 2. Status

| Area | State |
|---|---|
| Design | `course/DESIGN.md` Draft v2, 307 modules in 13 batches |
| B1 harness + tracer | Done and verified |
| Foundation for B2-B13 (contracts, consolidation, harness) | Done |
| Module groups | 35 of 48 done (B2 to B9, B10-3). 240 registry files exist |
| Remaining groups | 13, listed in `course/handoff/remaining-groups.md`. B10-1, B10-2, B11-1, B11-2, B12-1, B12-2 were interrupted mid-run: their partial files are on disk, continue them rather than starting over |
| Open items from finished groups | 442 lines in `course/handoff/open-items.txt` (deferred verifications, incomplete ids, and SHARED changes other groups requested). Integration resolves them |
| B14 multimodal | Designed in `course/design/B14-multimodal.md`, not built, not yet merged into DESIGN.md |

## 3. Decisions made AFTER DESIGN.md was written (they override it)

1. **No FFI anywhere.** No ctypes, no PyO3, no Rust calling C (`extern "C"`, `tl-sys`). Languages meet only at process and file boundaries: HTTP, gRPC, subprocess, and files (safetensors, tokenizer.json, fixtures). Reason from the user: FFI is hard and teaches nothing meaningful.
2. **Custom kernels are optional depth, not core.** L9.1 to L9.6 (C kernels), ds.01 to ds.04 and rt.02 to rt.04 (C data structures and runtime), M09.5 and M09.6 stay as optional, skippable, standalone C modules: each has its own C test binary and is parity-checked against fixture files generated by the Python reference. No engine backend switch (that would be FFI). No pass gate or core milestone may depend on them. No Triton or CUDA part.
3. **The Rust engine runs on candle** (`candle-core`, `candle-nn`; Metal on macOS, CPU in CI), not on course C kernels. `candle-transformers` is forbidden so the learner still writes every layer. Rust's purpose is the production serving layer: scheduler, continuous batching, KV block manager, prefix cache, HTTP/SSE, tokenizer.
4. **Python and Rust exchange data through files only.** Model weights go Python to Rust via safetensors. The Rust tokenizer (L1.5) serves the engine; Python keeps its own BPE; parity is checked through shared fixture files.
5. **Four languages:** Python, Rust, Go, and C (optional kernels only).
6. **B14 Q12 resolved:** Python never decodes images through Rust. Fixtures ship pre-decoded `.npy` pixels; Python uses Pillow for everyday decoding; the engine decodes with the Rust `image` crate and one test checks it against the `.npy` pixels within a small tolerance.

## 4. Work to do, in order

### Step A: finish the 13 remaining module groups
Per `remaining-groups.md`. For each module: registry toml, reference with markers, course tests in exemplar style (each test says why it exists), mutants, fixtures, solve sets for `S-*`, milestone definitions for `MS-*`, and a six-beat chapter at its chapter home. Apply the decisions in section 3 (for example the Go durable and agent code must not depend on any FFI). Prove each: `ol start <ID>` gives a stub that compiles and fails, the reference passes, `ol verify course <ids>` passes, `ol lint` is clean.

### Step B: restructure batch (applies section 3 to already built modules)
Modules that touch FFI today: `ds.04 L0.0 L10.1 L1.5 L10.6 L10.0 L10.4 L10.8 L9.1 L9.5 M03.1 L9.2 L9.6 L9.3 L8.3 lang.03 L9.7 M09.6 L9.4 M09.5 rt.03 rt.01` (find them with `grep -rlE "ctypes|pyo3|PyO3|extern \"C\"|tl-sys|cdll" course/modules`).
- B1 tracer: the Python bigram (L0.0) uses numpy, no C matmul. Retire rt.01 (C ABI + ctypes). M03.1 becomes a math chapter with a Python matmul; its C matmul becomes an optional standalone exercise.
- L1.5: drop the PyO3 binding; parity with Python BPE through fixture files.
- L8.3: paged KV cache in pure Python.
- L9.1 to L9.6, ds.01 to ds.04, rt.02 to rt.04, M09.5, M09.6: mark optional, standalone C test binaries, parity via fixture files from the Python reference. Retire L9.7.
- L10.x: engine on candle; retire tl-sys and every Rust-to-C call; keep scheduler, batching, KV block manager, prefix cache, HTTP/SSE, disaggregation, metrics, spec decoding, tool calls.
- Contracts: shrink `tinyllm.h` to what the optional C modules need; remove PyO3 stubs; update `course/DESIGN.md` (sections 1.5 decisions, 2, 4, 5, 7, 9) so the spec matches.
- Harness: remove the PyO3 farm build, the C overlay's two-build ctypes path, and symbol-drift checks that only existed for FFI. Keep the C test path for the optional modules.
- Pass gates and milestones (MS-P6 and later): must not require optional modules.

### Step C: merge and build B14
Merge `course/design/B14-multimodal.md` into `course/DESIGN.md` (with decision 6 above and its optional C kernels standalone, no `kernels = "c"` switch). Use these defaults for its open questions: Pass 12 is core but after v1.0.0; raise the committed-fixture cap to 56 MiB; accept no cross-request batching in the encoder; WAV-only audio input for v1. Verify the licenses (SmolVLM-256M-Instruct, whisper-tiny, PD12M, LibriSpeech, Common Voice) before committing any of their assets; if a license is unclear, use a synthetic stand-in and record it. Then build all B14 modules.

### Step D: integrate
Apply every SHARED change in `open-items.txt` that is consistent with the design and section 3. Regenerate `course/modules.tsv`. Fill `paths/course-p02` through `course-p12` and the role paths with the check column. Link chapters from track index READMEs. CI jobs per DESIGN 5.14 in `.github/workflows`. sr.yaml artifacts. Final STUDY-PLAN.md, CS-CURRICULUM.md, README.md.

### Step E: adversarial verification and fixes
Per layer (math; early ML; late ML; Rust; Go; ops and practices; multimodal): act as a learner and implement a few modules from the chapter alone; plant wrong implementations and confirm the tests catch them; check chapters for the six-beat template, correct math, first-principles completeness, and no em dashes. Fix every real defect.

### Step F: final gate (all must pass, report real output)
```bash
practice/bin/ol lint && practice/bin/ol lint --links && practice/bin/ol learn --verify
practice/bin/ol verify course
practice/bin/ol verify course --e2e
(cd course/harness && uv run pytest -q)
PATH=~/.local/share/fnm/aliases/default/bin:$PATH practice/bin/ol verify predict
(cd site && pnpm build)   # if site/ exists on this branch
practice/bin/ol milestone <each pass gate MS-P0 .. MS-P12> --smoke
```
kind is NOT installed and must not be installed: kind-only tiers run through `--smoke` and are reported as blocked.

## 5. Runner rules (mandatory: the machine crashed under load before)

- Run mutants one at a time, never in parallel.
- Start every subprocess you launch from scripts (builds, test runs, mutant runs, servers) with `start_new_session=True`.
- Kill the whole process group (`os.killpg`) if its total RSS goes above 4 GB or it runs longer than 120 s; report it as a failure, do not retry in a loop.
- Before each batch of runs: delete `.course-verify` under your `OL_SCRATCH`, and delete stale `$TMPDIR/tmp*` entries (not modified in 30 minutes).
- Before each batch of runs: check free disk with `df -g .`; while it is under 30 GB, wait instead of running. Delete `.scratchpad/w/<name>` dirs once their work is finished.
- If you run work in parallel, give each worker its own `OL_SCRATCH`, `OL_COURSE_HOME`, `OL_CACHE` under `.scratchpad/w/<name>` and run at most 6 at once.

## 6. Rules for commits and content

- Prose has no em dashes. Chapters follow DESIGN 6.2.
- Never publish anything: no release assets, no package publishes. Downloads only from official sources with permissive licenses.
- Fixtures must not contain strings shaped like real secrets (GitHub push protection rejected fake Slack tokens once; the generator now uses `xoxb-EXAMPLE-...`). Keep fake keys obviously fake.
- Commit in logical checkpoints with conventional commit messages, no AI attribution lines, and push to `origin feat/course` (PR #19) after each step (A to F). Do not commit `uv.lock` changes that only bump the project version, or anything under `.scratchpad/`.
- Report honestly: if a check fails, say so with the output.

## 7. Done means

Every core module in DESIGN.md (including B14) is registered, has a reference, tests, mutants, and a chapter, and passes `ol verify course`; optional modules pass their standalone checks; no FFI remains on any path; the final gate passes except kind-only tiers; everything is pushed to PR #19 with a summary comment listing what passed and what is blocked.
