# From learning signals to inference at scale

Follow one model from a learning objective to a measured service. This route connects the science of learning with the engineering that makes a trained artifact useful. Each stage ends with evidence you can inspect; a successful command is only one part of that evidence.

## Choose a depth

For the high-level explanation, read [how models learn](19-training-methods/08-how-models-learn.md), [architecture and performance](20-inference-engines/09-architecture-and-performance.md), and [runtime controls](20-inference-engines/10-runtime-controls-and-flags.md). For hands-on work, continue through the stages below, using CPU examples first and supported GPU runtimes when available. The model-building route in [the worked LLM system](paths/README.md) supplies implementation depth.

## The route and the evidence

| Stage | Read and work | Evidence to produce |
|---|---|---|
| Understand learning | [Signals and the update loop](19-training-methods/08-how-models-learn.md), [learning foundations](01-statistical-learning/README.md) | Explain the target, loss, trainable parameters, and held-out task metric |
| Choose/build the architecture | [Foundation-model families](05-foundation-models/README.md), [decoder block](13-decoder-block/README.md) | Tiny forward-pass checks; parameter, compute, and state counts |
| Train locally | [Frameworks](16-training-frameworks/README.md), [training-step comparison](19-training-methods/07-training-step-and-validation.md), [TinyStories](capstones/01-tinystories.md) | Correct target shift/masks; parameter changes; train and validation curves |
| Fit and scale training | [Precision/accumulation](17-training-at-scale/01-mixed-precision-accumulation-and-checkpointing.md), [parallel layouts](17-training-at-scale/04-parallelism-and-topology.md), [recovery](17-training-at-scale/05-training-failures-and-recovery.md) | Matched global-batch checks, peak memory, useful tokens/sec, interrupted-run recovery |
| Adapt with the available signal | [SFT](18-post-training/01-supervised-fine-tuning.md), [DPO](18-post-training/02-direct-preference-optimization.md), [SFT recipes](18-post-training/10-sft-recipes-and-data-strategies.md), [preference variants](18-post-training/11-preference-optimization-variants.md), [method scoping](19-training-methods/06-evaluation-and-fine-tune-scope.md) | Frozen comparison with the base model and task-specific regressions |
| Learn from actions when justified | [Policy/data distinctions](18-post-training/05-on-policy-and-off-policy.md), [PPO and GAE](18-post-training/06-ppo-with-gae.md), [reward design](18-post-training/07-rewards-and-preference-models.md), [GRPO](18-post-training/03-grpo-with-verifiable-rewards.md) | Behavior probabilities, valid advantages, independent success, reference and old/current KL |
| Scale rollout learning | [Actor/learner systems and GRPO variants](18-post-training/08-rollout-systems-and-grpo-variants.md) | Revision/lag accounting, reward audit, accepted fresh tokens and improvement per GPU-hour |
| Compress if needed | [KL distillation](18-post-training/04-distillation.md), [distillation data/deployment](18-post-training/09-distillation-data-and-deployment.md), [quantization](20-inference-engines/01-quantization/README.md) | Student/base/teacher task comparison; export and precision regression checks |
| Serve and validate locally | [Local vLLM, SGLang, llama.cpp, and in-process lab](20-inference-engines/14-local-serving-and-validation.md), [model loading](20-inference-engines/03-model-loading/README.md) | Artifact identity, template/token equivalence, API/stream tests, held-out service quality |
| Optimize execution | [Architecture constraints](20-inference-engines/09-architecture-and-performance.md), [control mechanisms and flags](20-inference-engines/10-runtime-controls-and-flags.md), [batch invariance](20-inference-engines/13-batch-invariance-and-reproducibility.md) | Fixed-workload ablations, numerical/replay criteria, measured memory and latency |
| Deploy at scale | [Serving layouts](20-inference-engines/11-serving-layouts-and-disaggregation.md), [capacity and load](20-inference-engines/04-serving-and-load/README.md), [GPU deployment](20-inference-engines/07-gpu-kubernetes-deployment/README.md) | Open-loop goodput curve, startup and failure capacity, bounded admission, recovery behavior |
| Operate and revisit | [Performance gates](20-inference-engines/12-performance-experiments-and-validation.md), [evaluation](29-evaluation/README.md), [observability](../systems/11-observability/README.md) | Versioned reports, workload drift checks, regressions, and cost per successful task |

This is a branching route. SFT, preference learning, reward optimization, and distillation can solve different problems; running them all in sequence is not a requirement. Architecture, data rights, privacy, and evaluation constraints apply throughout. [Data Engineering](../data-engineering/README.md) covers corpus construction and provenance; the responsible-deployment chapters in this book cover release obligations.

## A reusable experiment record

```text
Question and expected mechanism:
Baseline and candidate revisions:
Data/workload and split hashes:
Learning objective or serving control changed:
Hardware, runtime versions, resolved configuration:
Correctness, numerical, quality, latency, and cost criteria:
Observed results, failures, uncertainty, and raw records:
Conclusion under these assumptions; untested conditions:
```

## How the material is verified

The [source register](../../references/sources.md#training-to-inference-science-and-engineering) records checked primary papers and runtime documentation. [Executable learning checks](19-training-methods/code/learning_checks.py) recompute the worked arithmetic, while the local and distributed labs specify experiments to run on the actual stack. Source verification, arithmetic verification, a trained model's behavior, and a GPU benchmark are distinct forms of evidence; each supports only its own claims.
