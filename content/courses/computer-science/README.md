# Technology Foundations: Explore and Build

This is a reading guide, not a book of its own. It owns no chapters. Instead it recommends an order through the other books in the library, and through a handful of free external textbooks and courses, for anyone who wants a broad foundation across programming, mathematics, systems, data, AI, software engineering, and the human side of technology. It is for people from all walks of life: no computing degree or professional title is required, and you do not need to complete every subject. Choose an area that interests you, check its prerequisites in the [prerequisite map](prerequisites.md), and build toward a project that matters to you. The [study plan](study-plan.md) suggests a pace.

The textbooks named here are recorded, with their access and validation evidence, in the [source register](../../references/sources.md). The sequence, projects, and assessment ideas below are learning aids, not degree requirements.

## Before you begin

What you need depends on what you choose. For calculus, start with algebra, functions, exponentials and logarithms, and trigonometry; [Precalculus](../mathematics-for-computing/01-precalculus/README.md) in *Mathematics for Computing* covers them. No prior programming is required for the first programming stage. Learn terminal basics and Git alongside programming, with [Shell, git, make, and processes](../programming-languages/02-language-and-tool-primers/02-shell-git-make.md) in *Programming Languages*. You do not need to finish the mathematics before you begin programming; the two run side by side.

On pacing: the stages below are sized at roughly twelve to fifteen weeks each, with eight to twelve hours a week for each active subject, so three subjects at once means 24 to 36 hours a week before a capstone. These are planning estimates. At ten hours a week in total, take one subject at a time. Advance on demonstrated competence, not elapsed weeks.

## A suggested order

The eight stages suggest an order for readers who want a broad foundation. You can follow a single subject instead. Each stage names what to read, what it assumes, and the evidence that shows you have it.

### Programming, proof, and the first calculus

Begin with programming in [Composing Programs](https://composingprograms.com/), proofs in [Discrete Math 1](../mathematics-for-computing/06-discrete-math-1/README.md) (with Hammack's book), and [Calculus 1](../mathematics-for-computing/02-calculus-1/README.md) (with OpenStax). The stage assumes entry-level algebra, and trigonometry for calculus. You have it when you can write a tested command-line program, write direct, contradiction, and induction proofs, and work derivative and integral problems.

### Data structures and the second layer of mathematics

Next come data structures with [Open Data Structures](https://opendatastructures.org/) and the library's [practice problems](../../../practice/README.md), [Discrete Math 2](../mathematics-for-computing/07-discrete-math-2/README.md) (with Levin), and [Linear Algebra](../mathematics-for-computing/04-linear-algebra/README.md) (with Hefferon). This stage assumes the programming and proofs of the first. The evidence is implementing a hash table and a graph traversal, proving a loop invariant, and solving linear systems while explaining rank and nullity.

### Algorithms and how a computer works

With data structures in hand, read [Algorithms](../algorithms/README.md) alongside Erickson's textbook, build a computer from logic gates up with [Nand to Tetris](https://www.nand2tetris.org/), and continue with [Calculus 2](../mathematics-for-computing/03-calculus-2/README.md). The evidence is a correctness proof with a complexity analysis, logic gates carried through a CPU and an assembler, and a numerical integration compared against its error.

### Operating systems, networks, and probability

Then study operating systems with [Operating Systems: Three Easy Pieces](https://pages.cs.wisc.edu/~remzi/OSTEP/), networking with [An Introduction to Computer Networks](https://intronetworks.cs.luc.edu/), and [Probability and Statistics](../mathematics-for-computing/08-probability-statistics/README.md). This stage assumes architecture, programming, and discrete mathematics, plus Calculus 2 for continuous probability. The evidence is a shell and process exercise, a concurrent queue with race tests, a socket protocol that handles partial reads, and a simulation with interval estimates.

### Databases, theory of computation, and multivariable calculus

Study databases with [Berkeley CS186](https://cs186berkeley.net/), automata, computability, and complexity with [MIT 6.045J](https://ocw.mit.edu/courses/6-045j-automata-computability-and-complexity-spring-2011/), and [Calculus 3](../mathematics-for-computing/05-calculus-3/README.md). This stage assumes algorithms, proofs, and operating systems, plus Calculus 2 and linear algebra. The evidence is SQL with query-plan analysis, a reproduced transaction anomaly, automata and reduction proofs, and a gradient check.

### Languages, craft, and security

Turn to programming languages, including parsing, interpreters, and type checking, starting from [Type systems and polymorphism](../programming-languages/01-type-systems/README.md) in *Programming Languages*; to [Software Craftsmanship](../software-craftsmanship/README.md); and to security with [Berkeley CS161](https://textbook.cs161.org/). This stage assumes programming, theory, operating systems, and networks. The evidence is a small interpreter and type checker, a team code review with CI, and a threat model with security regression cases. Note that a type-systems survey alone does not complete a programming-languages course.

### People, pictures, and learning machines

Study human-computer interaction with [MIT 6.831](https://ocw.mit.edu/courses/6-831-user-interface-design-and-implementation-spring-2011/), graphics with [Berkeley CS184](https://cs184.eecs.berkeley.edu/sp26/), and introductory machine learning in [Statistical learning](../machine-learning/01-statistical-learning/README.md), with search from [Graphs](../algorithms/06-graphs/README.md). This stage assumes programming and software engineering, plus linear algebra, calculus, and statistics for graphics and machine learning. The evidence is a usability study, a transformed and rendered scene, and a search baseline next to a supervised model with held-out evaluation.

### Distributed systems, data, and a capstone

The last stage covers distributed systems in [Systems and Architecture](../systems/README.md), [Data Engineering](../data-engineering/README.md), a specialization of your choice, and a capstone. One possible systems capstone is [the course path through *Build Your Own LLM System*](../build-an-llm/paths/course/README.md), a staged build in Python, Rust, and Go; its optional C exercises are standalone and outside the core path. This stage assumes operating systems, networking, databases, software engineering, and whatever mathematics your specialization needs. The evidence is a failure-injection report, a reproducible pipeline, and a defended capstone with results and limitations.

Throughout every stage, practice technical writing, accessibility, privacy, attribution, and professional responsibility, and include a stakeholder-impact analysis in each substantial project. Study concurrency on one machine before distributed execution.

## What the library covers, and what it does not

The library does not cover every area of computing on its own, and some stages lean on external courses. The table records, for each area, the current route and where its coverage stops. "Local" means relevant material exists in the library, not that a full semester or an independently graded assessment is available. External courses fill gaps in this plan; their existence does not mean their content has been imported.

| Area | Current route | Coverage boundary |
|---|---|---|
| Software development fundamentals | Composing Programs + local practice | Introductory teaching supplied externally |
| Algorithmic foundations | Local algorithms + Erickson + MIT theory | Formal computability/complexity supplied externally |
| Mathematical and statistical foundations | Seven local math topics + applied labs below | Add numerical optimization for data/ML depth |
| Architecture and organization | Nand to Tetris | External course; no dedicated local architecture track |
| Systems fundamentals | Architecture, OS, local C/C++ practice, and the [end-to-end LLM systems course](../build-an-llm/paths/course/README.md) | Cross-course route; the course is a specialization project, not a complete OS or architecture sequence |
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

Mathematics sticks when it produces something that runs. For each of the labs below, state your assumptions, derive the method, implement a small version, compare it against a known answer or an independent library, and explain where it fails. These are assignment specifications, not prebuilt graded exercises:

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

[Mathematics for Machine Learning](https://mml-book.github.io/) connects linear algebra, vector calculus, probability, and optimization to regression, PCA, and mixture models. It complements the foundational books; it is not a replacement for learning proof or elementary calculus. Within the library, [Optimization](../mathematics-for-computing/11-optimization/README.md) and [Numerical methods and floating point](../mathematics-for-computing/10-numerical-methods-and-floating-point/README.md) supply the numerical-optimization depth that data and machine-learning work needs.

## Specializations after the core

Once the core is in place, each specialization has an ordered route and a capstone that shows you have it.

For **data engineering**, go from databases to [storage and warehousing](../data-engineering/02-storage-warehousing/README.md), then [batch and streaming](../data-engineering/03-batch-streaming/README.md), then [orchestration and modeling](../data-engineering/04-orchestration-modeling/README.md). The capstone is a versioned dataset with provenance, schema contracts, replay and deduplication, lineage, and tested quality constraints.

For **AI and ML engineering**, take probability, linear algebra, and multivariable calculus, then [statistical learning](../machine-learning/01-statistical-learning/README.md), [deep learning](../machine-learning/02-deep-learning/README.md), and [AI Platform Engineering](../ai-platform-engineering/README.md). The capstone compares a simple baseline with a model-based system on a frozen test set and reports uncertainty, latency, and cost. As an optional paid companion, read Chip Huyen's *AI Engineering: Building Applications with Foundation Models* (2025); her [official repository](https://github.com/chiphuyen/aie-book) provides free supporting materials. Use its evaluation, RAG, data, and inference topics alongside the platform chapters, and read her *Designing Machine Learning Systems* for the broader traditional ML lifecycle. Neither book replaces mathematics, algorithms, or systems foundations.

For **systems and infrastructure**, go from architecture to operating systems, networks, [Systems and Architecture](../systems/README.md), and [Infrastructure](../infrastructure/README.md). The capstone is a storage or service system tested under concurrency, overload, restart, and partial failure.

For **languages and theory**, go from discrete proofs to algorithms, automata and computability, and [type systems](../programming-languages/01-type-systems/README.md). The capstone is an interpreter or compiler with semantics, type rules, and positive and negative tests.

For **graphics and scientific computing**, go from linear algebra to calculus, [numerical methods](../mathematics-for-computing/10-numerical-methods-and-floating-point/README.md), and CS184. The capstone is a renderer or simulation with convergence and error measurements and a performance profile.

For **security and human-centered software**, go from networks and operating systems to CS161, HCI, and [authorization and access control](../ai-platform-engineering/08-authorization-and-access-control/README.md). The capstone is an accessible application with usability evidence, a threat model, and adversarial permission tests.

## Putting your learning into practice

Keep a portfolio of solved unseen problems, proofs, executable projects, raw measurements, and short technical reports. For each subject, explain one failure case without consulting the solution. Have an independent reviewer assess correctness, assumptions, and reproducibility, and record unresolved disagreements. Automated checks verify behavior, but they do not establish conceptual understanding.

The final capstone should include a research question, source and dataset provenance, a baseline, an experiment plan fixed before the final evaluation, reproducible commands, tests, limitations, and a presentation. Reserve a held-out dataset or workload; do not tune on it and then report it as an independent test. Treat writing, collaboration, and the social context of your work as part of what you are learning. When you are ready to apply, the [case studies](../case-studies/README.md) show the theory applied under deadline, and [Interview Preparation](../interview-preparation/README.md) turns it toward specific roles.
