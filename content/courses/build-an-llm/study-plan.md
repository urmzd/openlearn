# Study plan

*Build an LLM* is long, and it is best read the way the system is built: in passes, each ending with the whole system running end to end. This plan paces those passes at roughly 10 to 12 hours a week, which puts the core of the book, passes 1 to 11, at about 60 weeks part-time and the whole route, setup included, at about 67. Treat the week counts as a guide to proportion, not a schedule. The [course path](paths/course/README.md) lists the sections to read in each pass and what you should be able to explain afterwards, and each pass's milestone page describes what the system does once the pass is complete and why each piece is needed. Before starting, read [the system the solutions build](paths/course/SYSTEM.md).

## How to read each section

Do not read the book's directories in order: the parts interleave with mathematics, systems, and operations chapters from other books, and the course path puts them in the order the system needs them. Within a section, read the motivation, the principles, and the worked example first, then the interface, then the solution walkthrough. Before each step of the walkthrough, try to predict what the solution must do next; the prediction is what turns reading into understanding. Finish by answering the review questions without looking at the answers.

## The passes

The first two passes set the shape of everything after them. [Setup](paths/course-p00-setup/README.md) takes about a week and a half and leaves you with a repository whose CI lints commits and runs the native and Python test suites, along with fluency in Python, the shell, Git, Make, and CI. [The tracer](paths/course-p01-tracer/README.md), about five weeks, is the most important pass in the book: a byte bigram trained by the system's CLI, served by the Rust engine, behind the Go gateway, deployed on kind, with one trace and one runbook. Every later pass upgrades a component of this system without breaking its contracts.

The model passes come next. [Foundations](paths/course-p02-foundations/README.md), about seven and a half weeks, builds autograd and the training loop and retrains the bigram with them. [Tokens and data](paths/course-p03-tokens-and-data/README.md), also about seven and a half weeks, adds trained tokenizers and the data pipeline. [Sequence models](paths/course-p04-sequence-models/README.md), about four weeks, covers recurrent and sequence-to-sequence models and the origins of attention. [Transformer](paths/course-p05-transformer/README.md), about six weeks, builds transformer models, their objectives, fine-tuning, and the modern decoder block.

Then the system learns to run fast and stay up. [Inference and kernels](paths/course-p06-inference-and-kernels/README.md), about seven weeks, covers Python inference and the optional standalone C exercises. [Serving platform](paths/course-p07-serving-platform/README.md), about eight weeks, rebuilds the engine for real traffic and adds observability. [Durable](paths/course-p08-durable/README.md), about four and a half weeks, makes data and training into durable workflows.

The final passes put the system to work. [Capstone training](paths/course-p09-capstone-training/README.md), about five weeks, trains and evaluates the [capstone model](capstones/01-tinystories.md). [Agents](paths/course-p10-agents/README.md), about five weeks, adds agents, retrieval, and usage policy, with the optional [post-trained capstone](capstones/02-post-trained.md) alongside. [Operate](paths/course-p11-operate/README.md), about five and a half weeks, covers operability, security, and release readiness. [Multimodal](paths/course-p12-multimodal/README.md) is a plan for image and audio extensions; its design is in place but its modules and implementation are not built yet, so it carries no estimate.

For reference, the same pacing in one table:

| Pass | Focus | Approx. time |
|---|---|---:|
| [0, Setup](paths/course-p00-setup/README.md) | Repository, Python, shell, Git, Make, and CI | 1.5 weeks |
| [1, The tracer](paths/course-p01-tracer/README.md) | Byte bigram, Rust engine, Go gateway, and deployment | 5 weeks |
| [2, Foundations](paths/course-p02-foundations/README.md) | Autograd and training foundations | 7.5 weeks |
| [3, Tokens and data](paths/course-p03-tokens-and-data/README.md) | Tokenizers and data | 7.5 weeks |
| [4, Sequence models](paths/course-p04-sequence-models/README.md) | Recurrent and sequence-to-sequence models | 4 weeks |
| [5, Transformer](paths/course-p05-transformer/README.md) | Transformer models and fine-tuning | 6 weeks |
| [6, Inference and kernels](paths/course-p06-inference-and-kernels/README.md) | Python inference and optional standalone C exercises | 7 weeks |
| [7, Serving platform](paths/course-p07-serving-platform/README.md) | Serving platform and observability | 8 weeks |
| [8, Durable](paths/course-p08-durable/README.md) | Durable data and training workflows | 4.5 weeks |
| [9, Capstone training](paths/course-p09-capstone-training/README.md) | Training and evaluating the capstone model | 5 weeks |
| [10, Agents](paths/course-p10-agents/README.md) | Agents, retrieval, and usage policy | 5 weeks |
| [11, Operate](paths/course-p11-operate/README.md) | Operability, security, and release readiness | 5.5 weeks |
| [12, Multimodal](paths/course-p12-multimodal/README.md) | Planned image and audio extensions | Not estimated |

## A note on languages

Python and Rust exchange model and tokenizer data through files, and the Rust engine uses Candle; no language boundary uses a foreign-function interface. The C sections are optional standalone exercises: you can skip them without breaking any later pass, and come back to them when you want to see how the kernels underneath work.
