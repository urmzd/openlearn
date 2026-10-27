# Source Register and Validation

Research pass: **2026-10-06**. Scope: the new [CS curriculum](../courses/computer-science/README.md), foundational math/data books, and the identity/access of the AI engineering book. This is not a completed audit of every claim, link, benchmark, or implementation in the repository.

## Evidence and access

Prefer the author's, publisher's, university's, or standards body's own page. A free companion repository is not a free textbook. “Free to read” does not imply permission to redistribute or relicense. Consult the source's license before copying content; this register links to resources rather than reproducing them.

“Checked” below means the cited official page or document was readable during this pass and supported the listed identity or scope. It does not mean every exercise, download, chapter, or factual statement was independently verified. Live course pages can change; record the semester/version when using them.

## Mathematics and data books

| Source | Author / edition | Access and evidence | Curriculum use |
|---|---|---|---|
| [Calculus, Volumes 1-3](https://openstax.org/books/calculus-volume-1/pages/preface) | Gilbert Strang and Edwin Herman, senior contributing authors | Checked official preface: free web/PDF; lists scope of all three volumes | Single-variable differentiation/integration, series, multivariable/vector calculus |
| [Linear Algebra](https://hefferon.net/linearalgebra/) | Jim Hefferon | Checked author page: free text and supporting resources | Systems, vector spaces, maps, eigenvalues; pair with numerical labs |
| [Book of Proof](https://richardhammack.github.io/BookOfProof/) | Richard Hammack, third edition | Checked author page; older VCU URL failed retrieval | Logic and proof before algorithm correctness |
| [Discrete Mathematics: An Open Introduction](https://discrete.openmathbooks.org/dmoi4.html) | Oscar Levin, fourth edition | Checked official book page; free online | Counting, sequences, logic and graphs; match exercises to edition |
| [Introduction to Probability](https://math.dartmouth.edu/~prob/prob/prob.pdf) | Charles M. Grinstead and J. Laurie Snell | Checked Dartmouth-hosted full PDF | Discrete/continuous probability and stochastic reasoning |
| [Introductory Statistics 2e](https://openstax.org/books/introductory-statistics-2e/pages/preface) | Barbara Illowsky and Susan Dean | Checked publisher preface; free web/PDF | Applied inference and regression; algebra-based, not a mathematical-statistics replacement |
| [Mathematics for Machine Learning](https://mml-book.github.io/) | Marc Peter Deisenroth, A. Aldo Faisal, Cheng Soon Ong; 2020 | Checked authors' companion site with free PDF | Bridge from linear algebra, calculus, probability and optimization to ML |
| [An Introduction to Statistical Learning](https://www.statlearning.com/) | James, Witten, Hastie, Tibshirani; Taylor also on Python edition | Checked authors' site: R second edition (2021), Python edition (2023), free PDFs | Regression, classification, resampling and statistical learning labs |
| [Deep Learning](https://www.deeplearningbook.org/) | Ian Goodfellow, Yoshua Bengio, Aaron Courville | Checked official book site; free HTML | Mathematical foundations and neural networks |
| [Designing Data-Intensive Applications](https://dataintensive.net/) | Use author/publisher metadata for the chosen edition | Checked official landing page; commercial book | Storage, replication, transactions and distributed data; existing chapter references must be checked against the edition used |
| [Fundamentals of Data Engineering](https://www.oreilly.com/library/view/fundamentals-of-data/9781098108298/) | Joe Reis and Matt Housley | Checked publisher listing; paid/subscription full book | Data lifecycle, architecture and operational decisions; optional companion to local docs/course routes |

## Computer science foundations

| Source | Owner / author | Access and evidence | Curriculum use |
|---|---|---|---|
| [CS2023](https://csed.acm.org/) and [knowledge areas](https://csed.acm.org/knowledge-areas/) | ACM / IEEE-CS / AAAI | Checked official index and endorsement metadata | Breadth cross-check; not evidence of accreditation or full outcome alignment |
| [Composing Programs](https://composingprograms.com/) | Official textbook site | Checked root, which redirects readers to third edition; detailed edition content not audited | Programming foundations |
| [Open Data Structures](https://opendatastructures.org/) | Pat Morin | Checked author site; free book | Data structures before advanced algorithm analysis |
| [Algorithms](https://jeffe.cs.illinois.edu/teaching/algorithms/) | Jeff Erickson, first edition, 2019 | Checked author page; free electronic book and course archives | Proof-based algorithms; explicitly assumes discrete math and basic data structures |
| [Nand to Tetris](https://www.nand2tetris.org/) | Noam Nisan and Shimon Schocken | Checked official course: free project/lecture route; book and certificates are separate | Architecture and hardware/software construction |
| [Operating Systems: Three Easy Pieces](https://pages.cs.wisc.edu/~remzi/OSTEP/) | Remzi H. and Andrea C. Arpaci-Dusseau | Checked authors' site; free chapters | Virtualization, concurrency, persistence |
| [An Introduction to Computer Networks](https://intronetworks.cs.luc.edu/) | Peter L. Dordal | Checked university-hosted book | Network fundamentals before application protocols |
| [CS186](https://cs186berkeley.net/) | UC Berkeley | Checked course page; public materials, no assumed access to enrolled-student grading | Database foundations and internals |
| [6.045J, Spring 2011](https://ocw.mit.edu/courses/6-045j-automata-computability-and-complexity-spring-2011/) | MIT OpenCourseWare | Checked archived course | Automata, computability and complexity |
| [CS161 textbook](https://textbook.cs161.org/) | UC Berkeley | Checked public textbook | Security foundations |
| [6.831, Spring 2011](https://ocw.mit.edu/courses/6-831-user-interface-design-and-implementation-spring-2011/) | MIT OpenCourseWare | Checked archived course | HCI and usability methods; use current platform accessibility guidance for implementation |
| [CS184, Spring 2026](https://cs184.eecs.berkeley.edu/sp26/) | UC Berkeley | Checked semester-specific course page | Computer graphics |

## Chip Huyen's books

The likely intended reference is **Chip Huyen**, not “Nyugen”: [AI Engineering: Building Applications with Foundation Models](https://github.com/chiphuyen/aie-book), 2025. The official repository identifies the author/year and links to purchase options, chapter summaries, notes and the [table of contents](https://github.com/chiphuyen/aie-book/blob/main/ToC.md). These are free supporting materials, not the complete commercial book.

The same author describes *Designing Machine Learning Systems* as a companion focused on traditional ML applications, feature engineering and training. Use AIE for foundation-model applications; use DMLS for the broader production ML lifecycle. Both are optional paid references here.

| Reading theme | Local application | Evidence to produce |
|---|---|---|
| Evaluation | [LLM evaluation](../courses/ai-engineering/29-evaluation/) | Frozen test set, baseline, error taxonomy, uncertainty |
| Retrieval and context | [Retrieval & RAG](../courses/ai-engineering/27-retrieval-and-rag/) | Retrieval metrics and end-to-end answer assessment separately |
| Data quality | [Data engineering](../courses/data-engineering/) | Provenance, deduplication and leakage checks |
| Inference and application architecture | [LLM systems](../courses/ai-engineering/20-inference-engines/) | Quality, latency, throughput and cost under a stated workload |

## Engineering acronyms

References for [Engineering Acronyms](../courses/software-engineering/13-engineering-acronyms/README.md), checked 2026-10-25. The numerical scenarios in that chapter are teaching examples, not measurements of this project.

| Primary source | Scope supported |
|---|---|
| Rust project, [The Rust RFC Book](https://rust-lang.github.io/rfcs/) | RFC proposals, discussion, acceptance, and implementation as distinct stages |
| ADR GitHub organization, [Architectural Decision Records](https://adr.github.io/) | Recording a decision with rationale, tradeoffs, and consequences |
| Google re:Work, [Set goals with OKRs](https://rework.withgoogle.com/intl/en/guides/set-goals-with-okrs) | Objectives and measurable key results; goals are more than a task list |
| Chris Jones, John Wilkes, Niall Murphy, and Cody Smith, [Service Level Objectives](https://sre.google/sre-book/service-level-objectives/), *Site Reliability Engineering*, chapter 4 | Distinguishing SLIs, SLOs, and SLAs; defining measurements and agreement consequences |
| AWS Well-Architected Reliability Pillar, [Disaster Recovery objectives](https://docs.aws.amazon.com/wellarchitected/latest/reliability-pillar/disaster-recovery-dr-objectives.html) | RTO as acceptable restoration delay and RPO as acceptable data loss expressed in time |

## Training-to-inference science and engineering

Primary sources checked on **2026-10-26** for the [training-to-inference route](../courses/ai-engineering/training-to-inference.md). Papers establish algorithms and their reported experiments; runtime references establish documented controls for their stated versions. The curriculum's analytical examples are teaching scenarios, not measurements of local GPU performance.

| Source, edition, and locator | Claims supported | Local application and verification boundary |
|---|---|---|
| Goodfellow, Bengio, Courville, [*Deep Learning*](https://www.deeplearningbook.org/), 2016, chapters 5–8 | Learning signals, empirical objectives, gradient-based optimization | High-level learning loop; the chapter-5 HTML exceeded the retrieval size limit, so the official index/edition was checked |
| Sutton and Barto, [*Reinforcement Learning*, 2nd-edition draft hosted by Stanford](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf), 2014/2015 draft, section 5.5 | Behavior versus target policies; importance sampling and coverage | Draft section checked; the author's final-edition page timed out. Finite-action expectation independently recomputed |
| Schulman et al., [PPO](https://arxiv.org/abs/1707.06347), 2017, equations 7 and 9; [GAE](https://arxiv.org/abs/1506.02438), section 3 | Clipped surrogate, actor/value/entropy formulation, advantage recurrence | Four clipping sign cases, finite-difference slopes, GAE and terminal/truncation checks in `learning_checks.py` |
| Ouyang et al., [InstructGPT](https://arxiv.org/abs/2203.02155), 2022, section 3 | SFT, comparison reward modeling, and PPO as a documented RLHF recipe | Pairwise reward loss; not a claim about every current provider's pipeline |
| Shao et al., [DeepSeekMath](https://arxiv.org/html/2402.03300v3), 2024, section 4.1 | GRPO group baseline, token objective, reference-KL estimator | Corrected inclusive-baseline, zero-advantage, and KL-estimator distinctions; independent finite-action baseline check |
| Espeholt et al., [IMPALA](https://arxiv.org/abs/1802.01561), 2018 | Actor/learner lag and V-trace off-policy correction | Explains why stale PPO batches need an explicit estimator, not a renamed buffer |
| Yu et al., [DAPO](https://arxiv.org/abs/2503.14476), 2025; Liu et al., [Understanding R1-Zero-Like Training](https://arxiv.org/abs/2503.20783), 2025 | GRPO-related normalization, clipping, sampling, and bias analysis | Variants are recipe/objective changes; no reproduced convergence claim |
| Hugging Face, [TRL GRPO 0.23.1](https://huggingface.co/docs/trl/v0.23.1/grpo_trainer) | Versioned trainer loss/reward-scaling choices | Configuration vocabulary; defaults must be checked for the installed version |
| Hinton, Vinyals, Dean, [Distillation](https://arxiv.org/abs/1503.02531), 2015; Agarwal et al., [On-policy distillation](https://arxiv.org/html/2306.13649v3), 2024 | Soft targets, temperature, student-generated prefixes and distribution mismatch | KL arithmetic and distinction between prefix source and divergence direction |
| Narayanan et al., [Megatron-LM](https://arxiv.org/abs/2104.04473), 2021; Rajbhandari et al., [ZeRO](https://arxiv.org/abs/1910.02054), 2020 | Tensor/pipeline/data layouts and model-state sharding | Global-batch and pipeline-bubble calculations; actual scaling requires profiling |
| PyTorch [FSDP2](https://docs.pytorch.org/docs/2.9/distributed.fsdp.fully_shard.html), [distributed checkpoint](https://docs.pytorch.org/docs/2.9/distributed.checkpoint.html), [reproducibility](https://docs.pytorch.org/docs/2.9/notes/randomness.html), 2.9 | Sharding groups, checkpoint API constraints, scoped reproducibility | Distributed/restart lab and numerical acceptance criteria; no multi-GPU run claimed |
| Ainslie et al., [GQA](https://arxiv.org/abs/2305.13245), 2023; DeepSeek-AI, [DeepSeek-V2](https://arxiv.org/abs/2405.04434), 2024; Gu and Dao, [Mamba](https://arxiv.org/abs/2312.00752), 2024 | Shared KV heads, latent attention/MoE, recurrent state-space execution | Architecture-specific memory accounting; reported speedups are not universal |
| Kwon et al., [PagedAttention](https://arxiv.org/abs/2309.06180), 2023; Zhong et al., [DistServe](https://arxiv.org/abs/2401.09670), 2024 | Paged cache/sharing and prefill/decode interference/disaggregation | Cache budget and illustrative KV-transfer bound; deployment-specific benefit must be measured |
| vLLM [serve CLI](https://docs.vllm.ai/en/v0.11.0/cli/serve.html), [optimization guide](https://docs.vllm.ai/en/v0.11.0/configuration/optimization.html), archived 0.11.0 | Documented spelling and scope of scheduler, cache, precision, placement, and graph controls | Versioned mapping, not a recommendation of that release or its defaults |
| vLLM [batch invariance](https://docs.vllm.ai/en/stable/features/batch_invariance/), rolling stable snapshot | `VLLM_BATCH_INVARIANT` and supported execution restrictions | Separate from archived CLI examples; pin runtime before reproducing |
| SGLang [installation](https://docs.sglang.io/docs/get-started/install), [server arguments](https://docs.sglang.io/docs/advanced_features/server_arguments), [deterministic inference](https://docs.sglang.io/docs/advanced_features/deterministic_inference), rolling snapshot | Local launch, memory/chunk/request controls, current graph-option deprecation, deterministic mode | Commands checked against docs; no local GPU launch or performance result claimed |
| ggml-org, [llama.cpp server](https://github.com/ggml-org/llama.cpp/tree/master/tools/server), rolling repository snapshot | GGUF local server, loopback bind, alias and context controls | CPU/accelerator local route; pin a build before comparing |
| Horace He / Thinking Machines Lab, [Defeating Nondeterminism in LLM Inference](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/), 2025-09-10 | Fixed-shape determinism versus batch invariance; reduction-dependent kernels | Independent float-association/argmax examples plus a proposed actual-engine test matrix |

The [learning checks](../courses/ai-engineering/19-training-methods/code/learning_checks.py) independently recompute objective values, clipping derivatives, GAE boundaries, importance correction/ESS, finite-group baseline bias, KL direction, cache sizes, and numerical counterexamples. The [local lab](../courses/ai-engineering/20-inference-engines/14-local-serving-and-validation.md) separates API, numerical, task-quality, and load evidence. Hardware-dependent recipes remain experiments for the reader's supported stack, rather than claims of tested GPU results in this repository.

### SFT and preference-optimization variants

Checked 2026-10-26 for [SFT recipes](../courses/ai-engineering/18-post-training/10-sft-recipes-and-data-strategies.md) and [preference variants](../courses/ai-engineering/18-post-training/11-preference-optimization-variants.md). These sources identify objectives and data assumptions; no universal quality ranking or local reproduction of every paper is claimed.

| Primary source | Claims and verification boundary |
|---|---|
| Hugging Face, [TRL SFT 0.23.1](https://huggingface.co/docs/trl/v0.23.1/sft_trainer) | Completion/assistant masks and packing vocabulary; parameter updates are a separate axis |
| Wu et al., [On the Generalization of SFT](https://arxiv.org/abs/2508.05629), 2025 | DFT probability-dependent token update scaling; versioned implementation exposes a distinct loss type |
| Azar et al., [IPO](https://arxiv.org/abs/2310.12036), 2023 | Finite relative-score target; not a hard global KL constraint |
| Chowdhury et al., [Robust DPO](https://arxiv.org/abs/2403.00409), 2024 | Noise-correction assumptions; distinguish corrected estimators from label-smoothed mixtures |
| Ethayarajh et al., [KTO](https://arxiv.org/abs/2402.01306), 2024, revised 2026 | Individually desirable/undesirable labels and prospect-theoretic utility |
| Hong et al., [ORPO](https://arxiv.org/abs/2403.07691), 2024 | Reference-free odds-ratio penalty combined with supervised likelihood |
| Meng et al., [SimPO](https://arxiv.org/abs/2405.14734), 2024 | Length-averaged implicit reward and target margin |
| Guo et al., [Online AI-feedback alignment](https://arxiv.org/abs/2402.04792), 2024 | Online candidate collection and direct preference fitting |
| Ahmadian et al., [Back to Basics](https://arxiv.org/abs/2402.14740), 2024 | REINFORCE-style learning with a leave-one-out baseline |
| Hugging Face, [TRL DPO 0.23.1](https://huggingface.co/docs/trl/v0.23.1/dpo_trainer) | IPO normalization; cDPO/noise controls; EXO, NCA, BCO, SPPO, AOT, APO, WPO, reference refresh, length weighting, and combined-loss implementation vocabulary |

Independent checks compute smoothing's stationary margin, summed/averaged score reversal, a zero DPO pair margin with nonzero global KL, and leave-one-out baselines in `learning_checks.py`. These mathematical checks establish the examples, not production convergence.

## Cross-validation protocol

For future source additions, capture: stable source URL, title, authors, edition/date, access category, topic, prerequisite, specific claim or learning outcome, check date, and reviewer. For technical claims, record a section/page or named result and assumptions. Independently recompute quantitative examples or compare with an independent implementation. Two model answers agreeing is not independent evidence when both repeat the same source.

For datasets used in exercises, also record origin, license, collection dates, version/hash, units, schema, missingness, sampling, transformations, train/test split rules and known limitations. Keep raw inputs separate from derived data. Check leakage, duplicate records, out-of-range values, and whether the sample supports the stated conclusion.

When another system reviews a claim, exchange a concrete record:

```text
claim_id:
local_file_and_section:
claim_or_learning_outcome:
primary_source_and_locator:
edition_or_dataset_version:
independent_check_and_result:
disagreement_or_limitation:
reviewer_and_date:
status: pending | supported | corrected | unresolved
```

## Findings and remaining work

| Finding | Resolution / status |
|---|---|
| AI Engineering attribution and access | Official repository supports Chip Huyen (2025); paid text and free companion materials distinguished |
| Degree breadth versus interview tracks | Added explicit core sequence, external courses, coverage boundaries, and capstone requirements |
| Math workload | Paired-course schedule now budgets hours per subject; intensive track is distinguished from full semester study |
| Free access versus open license | Removed the math index's blanket assertion that every linked resource is openly licensed |
| Other-system coordination | Pending: system identity/contact or review output has not been supplied; no second-system agreement is claimed |
| Full repository correctness | Pending: existing technical claims, time estimates, outdated links and edition-specific chapter references still need individual audits |

Retrieval failures during research: the complete CS2023 HTML exceeded the browser's content-size limit; the older Hammack VCU page returned 502. The official CS2023 index and Hammack's current author page were used instead. Crafting Interpreters and a Cornell course URL could not be retrieved and were not promoted to checked sources. These retrieval failures do not establish that the websites are unavailable to other readers.
