# Build Your Own LLM System (about 67 weeks)

Follow the [course path](paths/course) at roughly 10 to 12 hours per week. Each pass ends at a runnable milestone. P0 to P11 are staged; P12 is a design plan whose modules and implementation are not built yet.

| Pass | Focus | Path | Approx. time |
|---|---|---|---:|
| 0 | Repository, Python, shell, Git, Make, and CI | [Setup](paths/course-p00-setup) | 1.5 weeks |
| 1 | Byte bigram, Rust engine, Go gateway, and deployment | [Tracer](paths/course-p01-tracer) | 5 weeks |
| 2 | Autograd and training foundations | [Foundations](paths/course-p02-foundations) | 7.5 weeks |
| 3 | Tokenizers and data | [Tokens and data](paths/course-p03-tokens-and-data) | 7.5 weeks |
| 4 | Recurrent and sequence-to-sequence models | [Sequence models](paths/course-p04-sequence-models) | 4 weeks |
| 5 | Transformer models and fine-tuning | [Transformer](paths/course-p05-transformer) | 6 weeks |
| 6 | Python inference and optional standalone C exercises | [Inference and kernels](paths/course-p06-inference-and-kernels) | 7 weeks |
| 7 | Serving platform and observability | [Serving platform](paths/course-p07-serving-platform) | 8 weeks |
| 8 | Durable data and training workflows | [Durable](paths/course-p08-durable) | 4.5 weeks |
| 9 | Training and evaluating the capstone model | [Capstone training](paths/course-p09-capstone-training) | 5 weeks |
| 10 | Agents, retrieval, and usage policy | [Agents](paths/course-p10-agents) | 5 weeks |
| 11 | Operability, security, and release readiness | [Operate](paths/course-p11-operate) | 5.5 weeks |
| 12 | Planned image and audio extensions | [Multimodal plan](paths/course-p12-multimodal) | Not estimated |

Python and Rust exchange model and tokenizer data through files; the Rust engine uses candle. C modules are optional standalone exercises and are not required by core pass gates.
