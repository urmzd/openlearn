# Technology Foundations: Explore and Build

An open learning guide across programming, mathematics, systems, data, AI, software engineering, and the human side of technology. It is for people from all walks of life: no computing degree or professional title is required. Choose an area that interests you, check its prerequisites, and build toward a project that matters to you. You do not need to complete every subject.

Use the [source register](../../references/sources.md) for textbook identity, access, and validation evidence. The suggested sequence, projects, and assessment rules below are learning aids, not degree requirements.

## Entry requirements and pacing

Prerequisites depend on the subjects you choose. For calculus, start with algebra, functions, exponentials/logarithms, and trigonometry. No prior programming is required for the first programming stage. Learn terminal basics and Git alongside programming; you do not need to finish the mathematics sequence before you begin.

The eight stages below suggest an order for learners who want a broad foundation. You can follow an individual subject instead. Budget roughly 12-15 weeks per stage and 8-12 hours per active subject each week; three subjects imply 24-36 hours per week before a capstone. These are planning estimates. At 10 hours total per week, take one subject at a time. Advance on demonstrated competence, not elapsed weeks.

## Suggested learning sequence

| Stage | Subjects and readings | Prerequisites | Required evidence |
|---|---|---|---|
| 1 | Programming with [Composing Programs](https://composingprograms.com/); [Discrete Math 1](../../tracks/math/05-discrete-math-1/), Hammack; [Calculus 1](../../tracks/math/01-calculus-1/), OpenStax | Entry algebra; trigonometry for calculus | Tested command-line program; direct, contradiction, and induction proofs; derivative and integral problems |
| 2 | Data structures with [Open Data Structures](https://opendatastructures.org/) and [practice](../../../practice/); [Discrete Math 2](../../tracks/math/06-discrete-math-2/), Levin; [Linear Algebra](../../tracks/math/03-linear-algebra/), Hefferon | Stage 1 programming and proofs | Implement a hash table and graph traversal; prove a loop invariant; solve systems and explain rank/nullity |
| 3 | [Algorithms](../../tracks/algorithms/), Erickson after basic data structures; architecture with [Nand to Tetris](https://www.nand2tetris.org/); [Calculus 2](../../tracks/math/02-calculus-2/) | Stages 1-2 | Correctness proof and complexity analysis; logic gates through CPU and assembler; numerical integration with error comparison |
| 4 | OS with [Operating Systems: Three Easy Pieces](https://pages.cs.wisc.edu/~remzi/OSTEP/); [networking](https://intronetworks.cs.luc.edu/); [Probability & Statistics](../../tracks/math/07-probability-statistics/) | Architecture, programming, discrete math; Calculus 2 for continuous probability | Shell/process exercise; concurrent queue with race tests; socket protocol handling partial reads; simulation and interval estimates |
| 5 | Databases with [Berkeley CS186](https://cs186berkeley.net/); theory with [MIT 6.045J](https://ocw.mit.edu/courses/6-045j-automata-computability-and-complexity-spring-2011/); [Calculus 3](../../tracks/math/04-calculus-3/) | Algorithms, proofs, OS; Calculus 2 and linear algebra | SQL and query-plan analysis; transaction anomaly reproduction; automata and reduction proofs; gradient checking |
| 6 | [Programming Languages](../../tracks/software-craftsmanship/), including parsing, interpreters and type checking; [Software Craftsmanship](../../tracks/software-craftsmanship/); security with [Berkeley CS161](https://textbook.cs161.org/) | Programming, theory, OS and networks | Small interpreter and type checker; team review and CI; threat model and security regression cases |
| 7 | HCI with [MIT 6.831](https://ocw.mit.edu/courses/6-831-user-interface-design-and-implementation-spring-2011/); graphics with [Berkeley CS184](https://cs184.eecs.berkeley.edu/sp26/); introductory [AI/ML](../../tracks/ml/01-statistical-learning/) plus search in [graphs](../../tracks/algorithms/06-graphs/) | Programming and software engineering; linear algebra, calculus and statistics for graphics/ML | Usability study; transformed and rendered scene; search baseline and supervised model with held-out evaluation |
| 8 | [Distributed systems](../../tracks/systems/), [data engineering](../../tracks/data-engineering/), specialization, and the [Build Your Own LLM System course](../build-an-llm/paths/course/) as one possible systems capstone | OS, networking, databases, software engineering; specialization-specific math | Failure-injection report; reproducible pipeline; defended capstone with results and limitations. The course provides a staged Python, Rust, and Go build; optional C exercises are standalone and outside the core gates |

Practice technical writing, accessibility, privacy, attribution, and professional responsibility throughout. Include a stakeholder-impact analysis in each substantial project. Study concurrency on one machine before distributed execution. A type-systems survey alone does not complete a programming-languages course.

## Breadth audit

These notes describe available material and where additional study is needed. “Local” means relevant material exists, not that a full semester or an independently graded assessment is available. External courses fill gaps in this plan; their existence does not mean their content has been imported.

| Area | Current route | Coverage boundary |
|---|---|---|
| Software development fundamentals | Composing Programs + local practice | Introductory teaching supplied externally |
| Algorithmic foundations | Local algorithms + Erickson + MIT theory | Formal computability/complexity supplied externally |
| Mathematical and statistical foundations | Seven local math topics + applied labs below | Add numerical optimization for data/ML depth |
| Architecture and organization | Nand to Tetris | External course; no dedicated local architecture track |
| Systems fundamentals | Architecture, OS, local C/C++ practice, and the [end-to-end LLM systems course](../build-an-llm/paths/course/) | Cross-course route; the course is a specialization project, not a complete OS or architecture sequence |
| Operating systems | OSTEP + local concurrency practice | Full OS course supplied externally |
| Networking and communication | Dordal + local RPC/streaming | Network fundamentals supplied externally |
| Data management | CS186 + local data engineering | Database internals supplied externally |
| Foundations of programming languages | Local type systems + interpreter assignment | Parsing/runtime/compiler depth needs further study |
| Software engineering | Local craftsmanship, testing, architecture, documentation | Team review must be arranged by learner |
| Parallel and distributed computing | Local concurrency, systems, AI platform | Requires measured failure and scaling experiments |
| Artificial intelligence | Local algorithms/search, ML, RL and foundation models | Logic, planning and knowledge representation need additional depth |
| Security | CS161 + local authorization | Security foundations supplied externally |
| Human-computer interaction | MIT 6.831 + project usability study | External course; no local HCI track |
| Graphics and interactive techniques | CS184 + linear algebra | External course; no local graphics track |
| Society, ethics, and the profession | Impact analysis across projects | Needs independent case discussion and review |
| Specialized platform development | Local cloud, infrastructure and edge inference | Mobile/embedded breadth remains an elective gap |

## Mathematics must produce working artifacts

For each lab: state assumptions, derive the method, implement a small version, compare against a known answer or independent library, and explain where it fails. These are assignment specifications, not prebuilt graded exercises.

| Foundation | Application | Acceptance evidence |
|---|---|---|
| Logic, sets, relations, induction | Dependency scheduler and permissions graph | Correct topological order or cycle witness; invariant proof; disconnected and cyclic cases |
| Counting, recurrences, modular arithmetic | Dynamic programming and hashing | Derive recurrence and complexity; enumerate small cases independently; distinguish collisions from equality |
| Linear systems, projections, eigenvectors, SVD | Least squares and PCA | Compare QR/SVD solution with a library; measure residual and reconstruction error; test rank deficiency; avoid explicit normal-equation inversion |
| Derivatives and Taylor approximation | Root finding and one-dimensional optimization | Compare analytic derivatives with central differences over several step sizes; report nonconvergence and cancellation |
| Integration and series | Numerical quadrature and probability normalization | Compare against an analytic integral; measure error as resolution changes; state convergence assumptions |
| Multivariable calculus and chain rule | Logistic regression or a tiny neural network | Derive gradients; finite-difference check on a smooth objective; show training loss and held-out performance separately |
| Probability and inference | Queue simulation and A/B experiment | Seeded replications; estimated uncertainty; assumptions about sampling and independence; false-positive simulation |
| Optimization and numerical analysis | Regularized regression and gradient descent | Compare with closed-form or library reference; learning-rate and conditioning experiment; stopping criterion |

Use [Mathematics for Machine Learning](https://mml-book.github.io/) to connect linear algebra, vector calculus, probability, and optimization to regression, PCA, and mixture models. It complements the foundational books. It is not a replacement for learning proof or elementary calculus.

## Specializations after the core

| Area | Ordered route | Capstone |
|---|---|---|
| Data engineering | Databases → [storage](../../tracks/data-engineering/02-storage-warehousing/) → [batch/streaming](../../tracks/data-engineering/03-batch-streaming/) → [modeling](../../tracks/data-engineering/04-orchestration-modeling/) | Versioned dataset with provenance, schema contracts, replay/deduplication, lineage, and tested quality constraints |
| AI/ML engineering | Probability + linear algebra + multivariable calculus → [statistical learning](../../tracks/ml/01-statistical-learning/) → [deep learning](../../tracks/ml/02-deep-learning/) → [AI platform](../../tracks/ai-platform-engineering/) | Compare a simple baseline and model-based system on a frozen test set; report uncertainty, latency and cost |
| Systems and infrastructure | Architecture → OS → networks → [systems](../../tracks/systems/) → [infrastructure](../../tracks/infrastructure/) | Storage or service system tested under concurrency, overload, restart, and partial failure |
| Languages and theory | Discrete proofs → algorithms → automata/computability → [type systems](../../tracks/software-craftsmanship/) | Interpreter or compiler with semantics, type rules, positive and negative tests |
| Graphics and scientific computing | Linear algebra → calculus → numerical methods → CS184 | Renderer or simulation with convergence/error measurements and performance profile |
| Security and human-centered software | Networks + OS → CS161 → HCI → [authorization](../../tracks/ai-platform-engineering/08-authorization-and-access-control/) | Accessible application with usability evidence, threat model, and adversarial permission tests |

For AI engineering, read **Chip Huyen, AI Engineering: Building Applications with Foundation Models (2025)** as an optional paid companion. Her [official repository](https://github.com/chiphuyen/aie-book) provides free supporting materials. Use its evaluation, RAG, data, and inference topics with the local platform modules. Read *Designing Machine Learning Systems* for the broader traditional ML lifecycle. Neither book replaces math, algorithms, or systems foundations.

## Putting your learning into practice

Keep a portfolio containing solved unseen problems, proofs, executable projects, raw measurements, and short technical reports. For each subject, explain one failure case without consulting the solution. Have an independent reviewer assess correctness, assumptions, and reproducibility; record unresolved disagreements. Automated checks verify behavior but do not establish conceptual understanding.

The final capstone must include a research question, source and dataset provenance, a baseline, an experiment plan fixed before final evaluation, reproducible commands, tests, limitations, and a presentation. Reserve a held-out dataset or workload. Do not tune on it and then report it as an independent test. Include writing, collaboration, and the social context of your work as part of your learning.
