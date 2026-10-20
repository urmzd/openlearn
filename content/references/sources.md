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
