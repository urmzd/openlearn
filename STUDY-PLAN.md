# Study Plans

Structured schedules for self-paced learning. Pick the plan that matches your goals.
For a degree-shaped sequence with assessment and specialization, use the [CS curriculum](CS-CURRICULUM.md). These shorter schedules are introductory passes or refreshers.

## Build Your Own LLM System (about 67 weeks)

Follow the [course path](paths/course/) at roughly 10 to 12 hours per week. Each pass ends at a runnable milestone. P0 to P11 are staged; P12 is a design plan whose modules and implementation are not built yet.

| Pass | Focus | Path | Approx. time |
|---|---|---|---:|
| 0 | Repository, Python, shell, Git, Make, and CI | [Setup](paths/course-p00-setup/) | 1.5 weeks |
| 1 | Byte bigram, Rust engine, Go gateway, and deployment | [Tracer](paths/course-p01-tracer/) | 5 weeks |
| 2 | Autograd and training foundations | [Foundations](paths/course-p02-foundations/) | 7.5 weeks |
| 3 | Tokenizers and data | [Tokens and data](paths/course-p03-tokens-and-data/) | 7.5 weeks |
| 4 | Recurrent and sequence-to-sequence models | [Sequence models](paths/course-p04-sequence-models/) | 4 weeks |
| 5 | Transformer models and fine-tuning | [Transformer](paths/course-p05-transformer/) | 6 weeks |
| 6 | Python inference and optional standalone C exercises | [Inference and kernels](paths/course-p06-inference-and-kernels/) | 7 weeks |
| 7 | Serving platform and observability | [Serving platform](paths/course-p07-serving-platform/) | 8 weeks |
| 8 | Durable data and training workflows | [Durable](paths/course-p08-durable/) | 4.5 weeks |
| 9 | Training and evaluating the capstone model | [Capstone training](paths/course-p09-capstone-training/) | 5 weeks |
| 10 | Agents, retrieval, and usage policy | [Agents](paths/course-p10-agents/) | 5 weeks |
| 11 | Operability, security, and release readiness | [Operate](paths/course-p11-operate/) | 5.5 weeks |
| 12 | Planned image and audio extensions | [Multimodal plan](paths/course-p12-multimodal/) | Not estimated |

Python and Rust exchange model and tokenizer data through files; the Rust engine uses candle. C modules are optional standalone exercises and are not required by core pass gates.

## Math Foundations (16 weeks)

~10-12 hours per active subject per week, or ~20-24 hours while two subjects are paired. At 10-12 hours total, take subjects sequentially and extend the schedule.

### Weeks 1-4: Calculus 1 + Linear Algebra

| Week | Calculus 1 | Linear Algebra |
|------|-----------|----------------|
| 1 | Limits and continuity | Linear systems, Gauss's method |
| 2 | Derivatives (rules and computation) | Vector spaces, subspaces |
| 3 | Applications of derivatives (optimization, MVT) | Linear transformations, matrices |
| 4 | Integration and FTC | Determinants, eigenvalues intro |

### Weeks 5-8: Calculus 2 + Discrete Math 1

| Week | Calculus 2 | Discrete Math 1 |
|------|-----------|-----------------|
| 5 | Integration techniques (by parts, partial fractions) | Sets, logic, truth tables |
| 6 | Applications of integration | Direct proof, contrapositive |
| 7 | Sequences and convergence tests | Proof by contradiction, induction |
| 8 | Power series, Taylor series | Relations, functions, cardinality |

### Weeks 9-12: Calculus 3 + Discrete Math 2

| Week | Calculus 3 | Discrete Math 2 |
|------|-----------|-----------------|
| 9 | Vectors in space, dot/cross product | Graph theory fundamentals |
| 10 | Partial derivatives, gradient | Graph coloring, Euler/Hamilton |
| 11 | Multiple integrals | Number theory, modular arithmetic |
| 12 | Vector calculus (Green's, Stokes') | Recurrence relations, generating functions |

### Weeks 13: Review & Catch-up

Revisit the hardest topics from each subject. Work through challenge problems.

### Weeks 14-16: Probability & Statistics

| Week | Focus |
|------|-------|
| 14 | Probability axioms, conditional probability, Bayes' theorem, discrete distributions |
| 15 | Continuous distributions, joint distributions, expectation & variance |
| 16 | Law of large numbers, CLT, hypothesis testing, confidence intervals, regression |

### Daily Routine (Math)

1. **Read** (30 min) -- textbook sections for the day's concept
2. **Work problems** (60-90 min) -- essential problems first, then challenge problems
3. **Review** (30 min) -- revisit previous day's concepts, rework one problem from memory
4. **Connect** (15 min) -- read the "Connections to CS" section, think about how the math applies

---

## Algorithm Mastery (21 days)

~2-3 hours per day. Each day targets specific patterns and problems.

### Week 1: Foundations

| Day | Focus | Problems | Patterns |
|-----|-------|----------|----------|
| **1** | Arrays & Hashing | [Two Sum](algorithms/01-arrays-hashing/two-sum.js), [Contains Duplicate](algorithms/01-arrays-hashing/contains-duplicate.js), [Missing Number](algorithms/01-arrays-hashing/missing-number.js), [Product Except Self](algorithms/01-arrays-hashing/product-of-array-expect-self.js) | [Hash map complement lookup, frequency counting, index-as-hash](algorithms/01-arrays-hashing/README.md) |
| **2** | Two Pointers & Sliding Window | [3Sum](algorithms/02-two-pointers-sliding-window/3sum.js), [Container With Most Water](algorithms/02-two-pointers-sliding-window/container-with-most-water.js) | [Opposite-end pointers, fast/slow pointers, window expand/shrink](algorithms/02-two-pointers-sliding-window/README.md) |
| **3** | Binary Search | [Find Min in Rotated Array](algorithms/03-binary-search/find-minimum-in-rotated-sorted-array.js), [Search in Rotated Array](algorithms/03-binary-search/search-in-rotated-sorted-array.js), [LISS](algorithms/03-binary-search/liss.js) | [Invariant-based search, search on answer space](algorithms/03-binary-search/README.md) |
| **4** | Linked Lists | [Add Two Numbers](algorithms/04-linked-lists/add-two-numbers.js) | [Runner technique, dummy head, reversal](algorithms/04-linked-lists/README.md) |
| **5** | Trees | [AVL + Range Query](algorithms/05-trees/example-7.py) | [DFS/BFS traversal, path problems, BST properties](algorithms/05-trees/README.md) |
| **6** | Graphs (traversal) | [Number of Islands](algorithms/06-graphs/number-of-islands.js), [Clone Graph](algorithms/06-graphs/clone-graph.js) | [DFS/BFS on grids, graph copying](algorithms/06-graphs/README.md) |
| **7** | Graphs (dependencies) | [Course Schedule](algorithms/06-graphs/course-schedule.js), [Pacific Atlantic](algorithms/06-graphs/pacific-atlantic-water-flow.ts) | [Topological sort, multi-source BFS](algorithms/06-graphs/README.md) |

### Week 2: Optimization & Search

| Day | Focus | Problems | Patterns |
|-----|-------|----------|----------|
| **8** | DP: 1D basics | [Climbing Stairs](algorithms/07-dynamic-programming/climbing-stairs.js), [House Robber](algorithms/07-dynamic-programming/house-robber.js), [House Robber II](algorithms/07-dynamic-programming/rob-houses-pt-2.js), [Coin Change](algorithms/07-dynamic-programming/change-coin.js) | [Fibonacci-style, take/skip, unbounded knapsack](algorithms/07-dynamic-programming/README.md) |
| **9** | DP: sequences | [Decode Ways](algorithms/07-dynamic-programming/decode-ways.js), [Word Break](algorithms/07-dynamic-programming/word-break.js), [LCS](algorithms/07-dynamic-programming/lcs.js), [Max Subarray](algorithms/07-dynamic-programming/maximum-subarray.js) | [Partition DP, two-string DP, Kadane's](algorithms/07-dynamic-programming/README.md) |
| **10** | DP: advanced | [Max Product Subarray](algorithms/07-dynamic-programming/maximum-product-subarray.js), [Unique Paths](algorithms/07-dynamic-programming/unique-paths.js), [Knapsack](algorithms/07-dynamic-programming/example-1.py), [Edit Distance](algorithms/07-dynamic-programming/example-2.py) | [Grid DP, min/max tracking, 0/1 knapsack](algorithms/07-dynamic-programming/README.md) |
| **11** | DP: hard variants | [Kadane](algorithms/07-dynamic-programming/example-3.py), [DP on Graph](algorithms/07-dynamic-programming/example-4.py), [Optimal BST](algorithms/07-dynamic-programming/example-5.c), [Alignment](algorithms/07-dynamic-programming/example-8.c) | [Interval DP, Knuth optimization, reconstruction](algorithms/07-dynamic-programming/README.md) |
| **12** | Greedy | [Buy/Sell Stock](algorithms/08-greedy/best-time-to-buy-and-sell-stock.js), [Jump Game](algorithms/08-greedy/jump-game.js), [Huffman Coding](algorithms/08-greedy/huffman-coding/), [Examples 1-3](algorithms/08-greedy/example-1.py) | [Greedy choice property, exchange argument, priority queues](algorithms/08-greedy/README.md) |
| **13** | Backtracking & Search | [Combination Sum](algorithms/09-backtracking/combination-sum.js), [Map Coloring](algorithms/09-backtracking/example-2.py) | [Choice/explore/unchoose, pruning, constraint propagation](algorithms/09-backtracking/README.md) |
| **14** | Math, Bits & Recursion | [Count Bits](algorithms/10-math-bit/count-number-of-bits.js), [1 Bits](algorithms/10-math-bit/number-of-1-bits.js), [Reverse Bits](algorithms/10-math-bit/reverse-bits.js), [Sum Without +](algorithms/10-math-bit/sum-of-two-integers.js), [Tree Path](algorithms/11-recursion-divide-conquer/example-1.py), [Max Subarray D&C](algorithms/11-recursion-divide-conquer/example-3.py) | [Bit tricks, divide & conquer](algorithms/10-math-bit/README.md), [Recursion patterns](algorithms/11-recursion-divide-conquer/README.md) |

### Week 3: Specialized Tracks & Interview Prep

| Day | Focus | Problems | Patterns |
|-----|-------|----------|----------|
| **15** | Concurrency & Systems | [C Miner project](archive/algorithms/12-concurrency-systems/miner/) | [Thread pools, producer-consumer, lock-free structures](archive/algorithms/12-concurrency-systems/README.md) |
| **16** | Functional Programming | [BST](archive/algorithms/13-functional-programming/bst.scm), [Visitors](archive/algorithms/13-functional-programming/list-visitor.scm), [Iterators](archive/algorithms/13-functional-programming/new-sqrt-iterator.scm), [Primes](archive/algorithms/13-functional-programming/IsPrime.scm) | [Immutable data, higher-order functions, recursion schemes](archive/algorithms/13-functional-programming/README.md) |
| **17** | ML & Statistics (models) | [Linear Regression](archive/algorithms/14-ml-statistics/linear-regression.py), [Logistic Reg & NNs](archive/algorithms/14-ml-statistics/ml-logistic-regression-and-neural-nets.py), [PyTorch](archive/algorithms/14-ml-statistics/learn-pytorch.py) | [Gradient descent, loss functions, backprop](archive/algorithms/14-ml-statistics/README.md) |
| **18** | ML (unsupervised) + Probabilistic | [Clustering](archive/algorithms/14-ml-statistics/clustering.py), [Naive Bayes & GMM](archive/algorithms/14-ml-statistics/ml-naive-bayes-and-gmm.py), [CNN & RNN](archive/algorithms/14-ml-statistics/ml-clustering-cnn-rnn.py), [Bloom Filter](algorithms/15-probabilistic-structures/bloom.py) | [Clustering, probabilistic models](archive/algorithms/14-ml-statistics/README.md), [Bloom filters, HyperLogLog, skip lists](algorithms/15-probabilistic-structures/README.md) |
| **19** | Practice: implement from scratch | [K-Means](archive/algorithms/14-ml-statistics/k-means.py), [TF-IDF Vector Search](archive/algorithms/14-ml-statistics/tf-idf-vector-search.py) | Re-read weakest topic READMEs |
| **20** | Company-targeted review | Pick 2-3 companies from [interviews/](interviews/README.md) and study their guides | [Shared concepts across companies](interviews/shared-concepts/README.md) |
| **21** | Mock interview day | Revisit 1 problem from each of topics 01-11 under timed conditions (45 min each) | Review all pattern READMEs as quick reference |

### Daily Routine (Algorithms)

1. **Read patterns first** (15 min) -- read the topic README before touching code
2. **Solve problems** (60-90 min) -- attempt each problem for 20 min before looking at the solution
3. **Review and annotate** (30 min) -- understand the solution, note edge cases, trace through examples
4. **Spaced review** (15 min) -- re-solve one problem from a previous day without looking at code

---

## ML & AI (18-37 weeks)

~10-12 hours per week. Requires math foundations (linear algebra, calculus, probability).

### Weeks 1-5: Statistical Learning

| Week | Focus | Reference |
|------|-------|-----------|
| 1 | Statistical learning framework, bias-variance | ISLR Ch 2 |
| 2 | Linear regression, model selection | ISLR Ch 3 |
| 3 | Classification (logistic regression, LDA, Naive Bayes) | ISLR Ch 4 |
| 4 | Resampling, regularization (ridge, lasso) | ISLR Ch 5-6 |
| 5 | Trees, random forests, boosting, SVMs, unsupervised | ISLR Ch 8-9, 12 |

### Weeks 6-13: Deep Learning

| Week | Focus | Reference |
|------|-------|-----------|
| 6 | ML basics, feedforward networks, backpropagation | Goodfellow Ch 5-6 |
| 7 | Regularization, optimization (SGD, Adam) | Goodfellow Ch 7-8 |
| 8 | CNNs -- architectures (LeNet to ResNet), applications | Goodfellow Ch 9 |
| 9 | RNNs, LSTMs, sequence-to-sequence | Goodfellow Ch 10 |
| 10 | Practical methodology, debugging, hyperparameter tuning | Goodfellow Ch 11 |
| 11 | Autoencoders, generative models (VAE, GAN) | Goodfellow Ch 14, 20 |
| 12 | Transformers, attention, BERT, GPT, scaling laws | Beyond book -- papers |
| 13 | Representation learning, self-supervised, transfer learning | Goodfellow Ch 15 + papers |

### Weeks 14-19: Reinforcement Learning

| Week | Focus | Reference |
|------|-------|-----------|
| 14 | Bandits, MDPs, Bellman equations | Sutton & Barto Ch 2-3 |
| 15 | DP, Monte Carlo, TD learning, Q-learning | Sutton & Barto Ch 4-6 |
| 16 | Function approximation, DQN | Sutton & Barto Ch 9-10 |
| 17 | Policy gradient, actor-critic, A2C/A3C | Sutton & Barto Ch 13 |
| 18 | PPO, DDPG, TD3, SAC | Spinning Up |
| 19 | RLHF, alignment, model-based RL | Research papers |

### Weeks 20-25: LLM Systems & Inference

| Week | Focus | Reference |
|------|-------|-----------|
| 20 | Transformer serving view, prefill vs decode, TTFT/TPOT | ML Systems + vLLM docs |
| 21 | KV cache, PagedAttention, prefix caching | vLLM (PagedAttention) paper |
| 22 | Continuous batching, FlashAttention, kernel optimization | TGI + FlashAttention papers |
| 23 | Quantization (GPTQ/AWQ/FP8), speculative decoding | Lilian Weng inference blog |
| 24 | Tensor/pipeline/expert parallelism, multi-GPU serving | TensorRT-LLM docs |
| 25 | GPU clusters, K8s deployment, autoscaling, SLOs | K8s + KServe/Ray Serve docs |

### Weeks 26-30: Foundation Models & Architectures

| Week | Focus | Reference |
|------|-------|-----------|
| 26 | Foundation models, the AI-engineering shift, adaptation ladder | Huyen *AI Engineering* Ch 1-2 |
| 27 | Transformer architecture variants (RMSNorm, RoPE, SwiGLU, MoE) | Llama/Gemma tech reports |
| 28 | Attention mechanisms (MHA/MQA/GQA/MLA, sliding-window, sparse) | DeepSeek + Mistral papers |
| 29 | Modalities: text vs audio vs vision, multimodal fusion (CLIP, VLMs) | HF Transformers + Whisper/CLIP |
| 30 | Diffusion models (DDPM, latent diffusion, DiT, flow matching) | HF Diffusers + Lilian Weng |

### Weeks 31-33: Neural Architectures & DL History

Foundations deep dive — best done alongside Deep Learning (weeks 6-9). Derive each architecture from its math and code it from scratch.

| Week | Focus | Reference |
|------|-------|-----------|
| 31 | History (perceptron → XOR winter → backprop); foundational math; code perceptron + MLP backprop | CS231n + Rumelhart 1986 |
| 32 | CNNs: convolution math, pooling, LeNet→AlexNet→VGG→ResNet; AlexNet deep dive | CS231n + AlexNet/ResNet papers |
| 33 | RNNs, BPTT, vanishing gradients; LSTM/GRU gate equations; CNNs vs RNNs vs Transformers | LSTM paper + Goodfellow Ch 10 |

### Weeks 34-37: Training & Post-Training

How a model is made before it is served. Pairs with RL (weeks 14-19) for the RLHF and GRPO sections.

| Week | Focus | Reference |
|------|-------|-----------|
| 34 | Pretraining: cross entropy, 6ND FLOPs, Chinchilla vs inference-optimal overtraining, MFU | Chinchilla + `code/memory_calc.py` |
| 35 | Training memory and parallelism: Adam state, ZeRO/FSDP2, TP/PP/CP/EP, JAX sharding | ZeRO + Megatron papers, PyTorch/JAX docs |
| 36 | Post-training: SFT, PPO-RLHF, DPO derivation, GRPO and verifiable rewards | InstructGPT, DPO, DeepSeekMath papers |
| 37 | PEFT and scoping: LoRA/QLoRA/DoRA math, run a LoRA fine-tune, eval base vs tuned | LoRA, QLoRA papers + `code/lora_from_scratch.py` |

---

## Role Path: Superstar FDE (14 weeks)

~10-12 hours per week. Assumes Python and basic linear algebra. The order and the done-when criteria live in [`paths/`](paths/): the [Superstar FDE](paths/superstar-fde/) path includes six parts, each also a path on its own. Run `practice/bin/ol learn superstar-fde` to follow it. If you already work in inference, do the Field Engineering weeks first.

| Week | Part | Focus |
|------|------|-------|
| 1 | Role brief, LLM Foundations 1 | What an inference cloud sells; history from n-grams through RNN/LSTM, attention, Transformer, scaling, RLHF, MoE, reasoning |
| 2 | LLM Foundations 2-3 | Transformer math: attention, backprop, FLOPs per token, memory; architecture variants from `config.json` |
| 3 | Frameworks and Models | PyTorch and JAX side by side, Keras 3 as a front end; loading from the Hub, safetensors, chat templates |
| 4-5 | Training | RL foundations; pretraining, post-training, LoRA; run a LoRA fine-tune and evaluate it |
| 6-8 | Inference Performance | Engine internals, quantization, the framework map; benchmark vLLM vs SGLang, latency curves, cost per 1M tokens |
| 9 | AI Full Stack | Streaming, retrieval, evals against your own endpoint, routing |
| 10-12 | Field Engineering | Discovery, sizing, performance testing engagements, POC, migration, commercials and security, escalation |
| 13-14 | Capstone | One mock customer through every part: eleven artifacts, then a 20-minute readout |

## Systems & Architecture (11-15 weeks)

~8-10 hours per week.

### Weeks 1-5: System Design

| Week | Focus | Reference |
|------|-------|-----------|
| 1 | Scalability, load balancing, caching, CDN | ByteByteGo |
| 2 | Database design, SQL vs NoSQL, CAP theorem, replication | ByteByteGo + DDIA |
| 3 | Distributed systems primitives (consensus, distributed transactions) | ByteByteGo |
| 4 | Messaging, event systems, API design | ByteByteGo |
| 5 | Classic problems (URL shortener, chat, news feed, etc.) | ByteByteGo |

### Weeks 6-8: Software Architecture

| Week | Focus | Reference |
|------|-------|-----------|
| 6 | Layered, event-driven, microkernel patterns | Software Architecture Patterns |
| 7 | Microservices, space-based architecture | Software Architecture Patterns |
| 8 | ADRs, quality attributes, trade-off analysis | Software Architecture Patterns |

### Weeks 9-12: Cloud Native

| Week | Focus | Reference |
|------|-------|-----------|
| 9 | Containers, Docker, multi-stage builds | Cloud Native DevOps with K8s |
| 10 | Kubernetes fundamentals (pods, deployments, services) | Cloud Native DevOps with K8s |
| 11 | Advanced K8s (StatefulSets, CRDs, operators, RBAC) | Cloud Native DevOps with K8s |
| 12 | Helm, GitOps, CI/CD, service mesh | Cloud Native DevOps with K8s |

### Weeks 13-15: Observability

| Week | Focus | Reference |
|------|-------|-----------|
| 13 | Observability vs monitoring, instrumentation, OpenTelemetry | Observability Engineering |
| 14 | SLOs, error budgets, debugging with observability | Observability Engineering + SRE Book |
| 15 | Production excellence, incident response, chaos engineering | Observability Engineering |

---

## Data Engineering (10-14 weeks)

~8-10 hours per week. Requires SQL and system design basics.

### Weeks 1-3: Foundations

| Week | Focus | Reference |
|------|-------|-----------|
| 1 | Data engineering lifecycle, OLTP vs OLAP, ETL vs ELT | Data Engineering Cookbook + DDIA Ch 1-2 |
| 2 | Storage formats (Parquet/Avro/Arrow), schemas, compression | Fundamentals of Data Engineering |
| 3 | Build an end-to-end mini pipeline (ingest → store → transform → query) | Hands-on |

### Weeks 4-7: Storage & Warehousing

| Week | Focus | Reference |
|------|-------|-----------|
| 4 | Storage engines (LSM vs B-tree), columnar execution | DDIA Ch 3 |
| 5 | Warehouse / lake / lakehouse, separation of storage & compute | Iceberg/Delta docs |
| 6 | Open table formats, partitioning, clustering, file sizing | Iceberg docs |
| 7 | Dimensional modeling (star schema, grain, SCDs) | The Data Warehouse Toolkit |

### Weeks 8-11: Batch & Streaming

| Week | Focus | Reference |
|------|-------|-----------|
| 8 | Distributed processing, partitioning, shuffle, Spark | Spark docs |
| 9 | Kafka -- log, partitions, consumer groups, replay | Kafka docs |
| 10 | Event time, watermarks, windowing, the Dataflow model | Streaming Systems |
| 11 | Delivery semantics, exactly-once, checkpointing, Flink | Flink docs + DDIA Ch 11 |

### Weeks 12-14: Orchestration & Modeling

| Week | Focus | Reference |
|------|-------|-----------|
| 12 | Orchestration (Airflow/Dagster), DAGs, idempotency, backfills | Airflow docs |
| 13 | dbt -- models, ref(), tests, materializations, lineage | dbt docs |
| 14 | Data quality, contracts, governance, data observability | Great Expectations docs |

---

## AI Platform Engineering (25-35 weeks)

~8-10 hours per week. **Patterns-first**: the goal is to internalize the fundamental patterns (like math, everything downstream is derivative) and treat each trendy tool as an instance. Requires [Deep Learning](ml/02-deep-learning/) and [LLM Systems](ml/04-llm-systems/), plus [System Design](systems/01-system-design/) and [Data Engineering](data-engineering/) foundations (OLTP vs OLAP, replication). Tip: [Coding & Design Patterns](ai-platform-engineering/06-coding-and-design-patterns/) can be read first as vocabulary.

### Weeks 1-6: Training & Frameworks

| Week | Focus | Reference |
|------|-------|-----------|
| 1 | PyTorch deep-dive: eager, autograd, `nn.Module`, `torch.compile` | PyTorch docs |
| 2 | JAX: `jit`/`grad`/`vmap`/`pmap`, Flax/Optax, PyTorch vs JAX | JAX docs |
| 3 | Distributed training I: data parallel (DDP), all-reduce, mixed precision | PyTorch DDP + Ultra-Scale Playbook |
| 4 | Distributed training II: FSDP/ZeRO, tensor/pipeline/expert parallelism, Megatron/DeepSpeed | FSDP tutorial + DeepSpeed docs |
| 5 | Embedding models: contrastive fine-tuning, hosting (TEI), vector stores | Sentence-Transformers + pgvector |
| 6 | Small language models: distillation, LoRA/QLoRA (PEFT), quantize, serve with vLLM | HF PEFT + vLLM docs |

### Weeks 7-9: RPC & Protocols

| Week | Focus | Reference |
|------|-------|-----------|
| 7 | RPC model, partial failure, serialization formats (JSON/Protobuf/Avro/Thrift) | DDIA Ch 4 + protobuf.dev |
| 8 | gRPC: HTTP/2, the four call types, streaming, interceptors, gateways | gRPC docs |
| 9 | Schema evolution, choosing the protocol per boundary, Arrow Flight | Avro spec + gRPC docs |

### Weeks 10-11: Streaming & SSE

| Week | Focus | Reference |
|------|-------|-----------|
| 10 | SSE protocol, `EventSource`, SSE vs WebSockets vs long-polling vs gRPC streaming | MDN SSE + HTML spec |
| 11 | End-to-end token streaming, cancellation/backpressure, proxies and scaling | OpenAI streaming + vLLM docs |

### Weeks 12-15: Distributed Data & Caching

| Week | Focus | Reference |
|------|-------|-----------|
| 12 | OLTP vs OLAP, sharding (hash/range/consistent hashing), hot shards, rebalancing | DDIA Ch 5-6 + Citus/Vitess |
| 13 | In-memory stores & caching patterns (cache-aside/through/back), invalidation | Redis docs + Scaling Memcache |
| 14 | Caching failure modes (stampede, herd, penetration); Cassandra (LSM, ring, quorum) | Redis + Cassandra docs |
| 15 | Knowledge graphs (Neo4j, Apache AGE on Postgres), bitemporal/temporal data | AGE + Neo4j docs |

### Weeks 16-18: Durable Orchestration, Workers & Patterns

| Week | Focus | Reference |
|------|-------|-----------|
| 16 | Durable execution & checkpointing, event-sourced replay, the worker pattern | Temporal + DBOS docs |
| 17 | Reliability patterns (idempotency, saga/compensation, outbox); distributed observability + profiling | Temporal docs + OpenTelemetry + Gregg |
| 18 | Coding & design patterns: decorator, facade, closures, currying, composition | Refactoring Guru + Mostly Adequate Guide |

### Weeks 19-25: Retrieval, Authorization & Evaluation

| Week | Focus | Reference |
|------|-------|-----------|
| 19 | Encoders (bi- vs cross-encoder), embedding retrieval, ANN/HNSW | RAG paper + HNSW + sbert |
| 20 | Chunking patterns (fixed/recursive/semantic, parent-child), metadata | saige RAG + pgvector |
| 21 | Lexical retrieval (BM25, bag-of-words, TF-IDF); hybrid fusion (RRF) + reranking | BM25 review + RRF paper |
| 22 | Multimodal handling (Document→Section→Variant), GraphRAG, production retrieval | GraphRAG + saige |
| 23 | Authorization: RBAC/ABAC/ReBAC/NGAC, the pushdown complexity ladder, Zanzibar | NIST + Zanzibar + OpenFGA |
| 24 | Secure RAG (permission-filtered retrieval, multi-tenancy); PDP/PEP, policy-as-code | NIST ABAC + OPA |
| 25 | LLM evaluation: cross-entropy, perplexity, **bits-per-byte**; benchmarks & contamination | MacKay + HELM + The Pile |

### Weeks 26-28: Edge, Realtime & On-Device Inference

| Week | Focus | Reference |
|------|-------|-----------|
| 26 | The deployment spectrum; **llama.cpp + GGUF** end-to-end (convert → quantize → `llama-server`); Ollama; MLX/ExecuTorch on-device | llama.cpp + GGUF + Ollama |
| 27 | Realtime/streaming encoders: offline vs causal/chunked; speech (Whisper offline, Conformer/RNN-T streaming); real-time factor | Whisper + Conformer papers |
| 28 | Efficiency architectures (the Mistral clinic): sliding-window attention, GQA, Mixtral MoE, Codestral Mamba/SSM, Voxtral | Mistral 7B + Mixtral + Mamba papers |

### Weeks 29-30: Model Routing & Cascades

| Week | Focus | Reference |
|------|-------|-----------|
| 29 | The three routing decisions; router families (rules, predictive, cascade, verifier-gated, session-level); oracle vs deployed router; model recall | RouteLLM + FrugalGPT + LLMRouterBench |
| 30 | Cache-aware switching, escalation policy (category, verifier, confidence), token-basis economics, the SFT/DPO/RFT ladder, evaluating a router, System One decision models, specialised models in the pool | FireRouter docs + RFT docs + TypeSafe docs |

### Optional deep dive (weeks 31-35)

LLM-as-judge & Elo arenas, RAG metrics (faithfulness, context precision/recall, NDCG/MRR), agent metrics (TTFT/TTLT, tool success), and building a composable `Scorer` eval harness gated in CI. References: Ragas, Chatbot Arena, lm-evaluation-harness, saige `eval/`.

### Daily Routine (AI Platform Engineering)

1. **Name the pattern, then the tool** — for every tool, ask "what pattern is this an instance of?" That's the transferable knowledge.
2. **Build, don't just read** — train it, shard it, cache it, stream it, orchestrate it, retrieve it, authorize it, score it.
3. **Measure** — GPU memory, tokens/sec, bytes-on-wire, TTFT, cache hit rate, retrieval recall, bits-per-byte; numbers beat intuition.
4. **Break it on purpose** — kill a worker mid-workflow, expire a hot cache key, reuse a Protobuf tag, leak a doc past an ACL; learn the failure modes.

---

## Diagramming & Documentation (2 weeks)

~6-8 hours per week. Pairs with Software Craftsmanship (the principles behind treating artifacts like source) and Infrastructure (the systems you draw). Uses the Streamflow event-driven platform as the running example.

### Week 1: Diagramming & the C4 Model

| Day | Focus | Reference |
|-----|-------|-----------|
| 1 | C4's four levels (Context/Container/Component/Code); the map analogy | c4model.com |
| 2 | Install `d2`; render the Streamflow C4 set; edit and `git diff` a change | D2 docs |
| 3 | Mermaid inline (sequence, state, ER); pick the diagram by the question | Mermaid docs |
| 4-5 | Redraw a system you know at L1+L2; set up render-in-CI for living diagrams | Hands-on |

### Week 2: Documentation & Technical Writing

| Day | Focus | Reference |
|-----|-------|-----------|
| 1 | Diátaxis: tutorial / how-to / reference / explanation -- don't mix them | diataxis.fr |
| 2 | Google Technical Writing One; active voice, BLUF, tables over prose | Google Tech Writing |
| 3 | Docs-as-code: version, review, test samples in CI, deprecate | SWE at Google Ch 10 |
| 4-5 | Write one ADR + one runbook for a real system | ADR / Write the Docs |

---

## Programming Languages (2-3 weeks)

~8-10 hours per week. The math first (lambda-calculus ladder → Hindley-Milner inference), then the *same* concept across five type systems. Run the `code/` files as you read.

### Week 1: The Math of Types

| Day | Focus | Reference |
|-----|-------|-----------|
| 1 | What a type system is: judgments `Γ ⊢ e : τ`, soundness = progress + preservation, Curry-Howard | TAPL Ch 1-9 / Software Foundations |
| 2 | The polymorphism taxonomy (Strachey, Cardelli-Wegner): parametric, ad-hoc, subtype, bounded | Cardelli-Wegner paper |
| 3 | The lambda-calculus ladder: STLC → System F (parametric polymorphism, parametricity) | TAPL Ch 22-23 |
| 4-5 | Hindley-Milner + Algorithm W: unification, the occurs check, `let`-generalization, principal types. Read & run `code/hindley_milner.py` | Damas-Milner paper |

### Week 2: The Same Concept, Five Mechanisms

| Day | Focus | Reference |
|-----|-------|-----------|
| 1 | Structural typing + variance + type-level programming — `code/polymorphism.ts` | TS Handbook |
| 2 | Go generics: type sets/unions, GC-shape stenciling — `code/generics.go` | Go generics tutorial |
| 3 | Rust traits: static (monomorphized) vs dynamic (`dyn`) dispatch, associated types — `code/traits.rs` | Rust Book Ch 10, 17 |
| 4 | Haskell typeclasses (dictionary passing) + higher-kinded `Functor` — `code/typeclasses.hs` | Wadler-Blott paper |
| 5 | OCaml HM inference + modules/functors — `code/inference.ml`; compare its principal types to Week 1 | Real World OCaml |

### Week 3 (optional): Depth

| Day | Focus | Reference |
|-----|-------|-----------|
| 1-2 | Subtyping & variance in depth: declaration- vs use-site, the array-covariance soundness hole | TAPL Ch 15-16 |
| 3 | How generics compile: monomorphization vs erasure vs dictionaries — the runtime-cost trade-off | — |
| 4-5 | The frontier: HKTs, associated types, GADTs, dependent types (the lambda cube top) | PLFA |

---

## Software Craftsmanship (3-4 weeks)

~6-8 hours per week. Individual craft → organizational craft → the testing mentality that ties them together → the failure modes that show what happens without them.

### Weeks 1-2: Pragmatic Programmer & SWE at Google

| Week | Focus | Reference |
|------|-------|-----------|
| 1 | Pragmatic philosophy; DRY, orthogonality, reversibility, tracer bullets, design by contract | The Pragmatic Programmer |
| 2 | HRT & culture, code review, Hyrum's Law, deprecation budgets, build/CI as leverage | SWE at Google |

### Week 3: The Testing Mentality & Lessons from Practice

| Day | Focus | Reference |
|-----|-------|-----------|
| 1 | Test pyramid; test behavior not implementation; fakes > mocks | SWE at Google Ch 11-14 |
| 2 | Property-based testing + fuzzing; add one property test | Hypothesis / proptest |
| 3 | Testing infra, data, docs, diagrams (the mentality everywhere) | Conftest / dbt tests |
| 4-5 | Canary, synthetic monitoring, chaos experiment design; SLOs | Principles of Chaos |
| 6 | Naming, data modeling, reliability in small code (Themes 1-2) | Lessons from Practice |
| 7 | Judgment signals, scope discipline, evidence and closure (Themes 3-5); audit one of your own repos against the catalog | Lessons from Practice |

---

## Infrastructure (5-7 weeks)

~8-10 hours per week. How distributed systems actually run. Topics 1-3 stay anchored to the Streamflow event-driven platform.

### Weeks 1-2: Containers, Kubernetes & Workloads

| Week | Focus | Reference |
|------|-------|-----------|
| 1 | Containers (namespaces/cgroups/layers), multi-stage + distroless, 12-factor | Docker / 12factor.net |
| 2 | K8s as a reconcile loop; objects; stateful vs stateless; caveats (limits/OOM, probes, SIGTERM, PDB); scaling (HPA/VPA/Cluster Autoscaler/KEDA) | Kubernetes docs |

### Weeks 3-4: Messaging & Distributed Workers

| Week | Focus | Reference |
|------|-------|-----------|
| 3 | Queue vs log, push/pull/long-poll, pub-sub vs competing consumers, delivery semantics, Little's Law; ZooKeeper vs KRaft vs Redpanda | Kafka docs + DDIA Ch 11 |
| 4 | Consumer groups, partition cap, rebalancing, idempotency/outbox, DLQ, lag → KEDA scaling | Kafka consumer docs |

### Weeks 5-6: Search & the Apache Stack

| Week | Focus | Reference |
|------|-------|-----------|
| 5 | Inverted index, TF-IDF → BM25, Lucene segments/merges, Elasticsearch vs OpenSearch vs Solr | Lucene / ES Guide |
| 6 | The Apache data ecosystem map: messaging, storage/tables, OLAP, search, orchestration | Apache docs + DDIA |

---

## Case Studies (4-8 days)

~4-6 hours per study. Each is a worked build: read the README, run the
implementation, then break it deliberately and watch the tests catch you.
Best taken *after* the track that carries the theory, as the applied pass.

| Day | Study | Do this |
|-----|-------|---------|
| 1-2 | [Order Book Matching](case-studies/01-order-book-matching/) | Write the flat-list version yourself first, then do the BUD analysis before reading the heap layer |
| 3-4 | [Grounded SQL Agent](case-studies/02-grounded-sql-agent/) | Try to get a write past the safety gate; design an eval harness for a domain of your own |
| 5-6 | [Exactly-Once Event API](case-studies/03-exactly-once-event-api/) | Replace the constraint dedup with check-then-act and watch the concurrency test fail |
| 7-8 | [K-Means Optimization](case-studies/04-kmeans-optimization/) | Predict the timing sweep before running it; explain why dropping `sqrt` is safe |

Pairs with [Lessons from Practice](software-craftsmanship/04-lessons-from-practice/),
which catalogues the failure modes these builds were avoiding.

---

## Combined Path (40+ weeks)

The full journey from foundations through PhD-level depth and Staff+ engineering expertise.

| Weeks | Focus | Details |
|-------|-------|---------|
| 1-4 | Calculus 1 + Linear Algebra | Parallel math study |
| 5-8 | Calculus 2 + Discrete Math 1 | Parallel math study |
| 9-12 | Calculus 3 + Discrete Math 2 | Parallel math study |
| 13 | Math review & catch-up | Revisit hardest topics |
| 14-16 | Probability & Statistics | Completes math track |
| 17-19 | Algorithm Mastery | 21-day plan (one week per study-plan week) |
| 20-21 | Competitive Programming | Advanced techniques, contest practice |
| 22-26 | Statistical Learning + Deep Learning | ISLR then Goodfellow |
| 27-30 | Deep Learning (continued) + Info Theory | Research-level depth |
| 31-34 | Reinforcement Learning | Sutton & Barto + Spinning Up |
| 35-39 | Systems & Architecture | System design through observability |
| 40-46 | AI Platform Engineering | Patterns-first: training, RPC, streaming, data & caching, orchestration, design patterns, RAG, authorization, evaluation |
| 47-48 | Case Studies | Four worked builds, applied pass over the theory |
| 49-51 | Interview preparation | Company-targeted study, mock interviews |

---

## PhD Research Track (20+ weeks)

For those pursuing research depth in ML/AI. Assumes math foundations and statistical learning are complete.

| Weeks | Focus | Reading |
|-------|-------|---------|
| 1-8 | Deep Learning (full book) | Goodfellow Parts I-III, all 20 chapters |
| 9-10 | Information Theory | MacKay -- entropy, KL divergence, channel coding |
| 11-14 | Reinforcement Learning | Sutton & Barto full + Spinning Up key papers |
| 15-17 | Transformer architectures | Attention Is All You Need, BERT, GPT, scaling laws papers |
| 18-19 | Alignment & Safety | Constitutional AI, RLHF, DPO papers |
| 20+ | Research exploration | Pick a subfield, read 10+ papers, implement one |

### How to Use These Plans

1. **Pick your starting point** -- if you have a math background, skip to algorithms or ML
2. **Parallelize where possible** -- math topics are designed for parallel study
3. **Don't skip the connections** -- the cross-references between tracks are where real understanding lives
4. **Adjust the pace** -- these timelines assume ~10 hrs/week; scale up or down as needed
5. **Use interview guides last** -- they're most effective after building deep understanding
