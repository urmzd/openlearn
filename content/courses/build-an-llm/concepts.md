# Concepts and techniques

This overview gives you the whole of *Build Your Own LLM System* in one sitting: the single idea the book is organized around, what the thinnest working system contains, how the later parts upgrade it, and where the book connects to the rest of the library. Read it after the [book's front matter](README.md) and alongside [the system the solutions build](paths/course/SYSTEM.md), which is the component-and-contract map of the same story.

## The core insight

A language model is a function from a prefix of token ids to a distribution over the next id. Everything else in the stack is plumbing that makes that function cheap to train, cheap to run, and safe to expose, and each piece of plumbing has a contract you can test in isolation.

That framing explains the book's method. An LLM system is a stack of contracts: a checkpoint format between training and serving, file formats between Python and Rust, an HTTP API between the engine and everything in front of it. If you understand both sides of each contract, nothing in the stack is magic. So the book starts with the thinnest model that exercises every contract and then upgrades one component at a time while the contracts stay fixed. And because numbers are only right if they are checked, every section lists the cases a correct solution must handle and shows how its solution handles each one.

## The tracer: the thinnest system that works

The first turn of the spiral, the tracer, puts a real model behind a real engine with as little model as possible. Four ideas carry it.

The **byte tokenizer** has 256 ids, needs no training, and has no unknown tokens. Its one subtlety is on the way out: decoding a stream of bytes must handle incomplete UTF-8, because a multi-byte character can be split across two generated tokens.

The **count bigram** estimates the next byte from the current one by counting with add-$\alpha$ smoothing:

$$
P(b \mid a) = \frac{c_{ab} + \alpha}{\sum_x c_{ax} + 256\alpha}.
$$

Its negative log-likelihood on held-out text is the baseline every later model must beat. The reference implementation computes the bigram's logits in Python.

The **checkpoint** is the first cross-language contract. Python writes `model.safetensors`, and the Rust engine, built on Candle, reads the same weights. Shared fixture files check that both languages behave the same way.

You build all of this in [the byte bigram](01-foundations/01-byte-bigram.md) and serve it from [your first endpoint](11-serving/01-your-first-endpoint.md).

## The spine: upgrading one component at a time

After the tracer, each part of the book replaces or extends one component behind the same contracts. In order, the parts are autograd and training ([Foundations](01-foundations/README.md)), [tokenizers](02-tokenizers/README.md), [statistical language models](03-statistical-lm/README.md), [recurrent networks](04-recurrent/README.md), [attention origins](05-attention-origins/README.md), [the 2017 transformer](06-transformer-2017/README.md), [objectives and adaptation](07-objectives/README.md), [the modern decoder block](08-modern-block/README.md), [inference](09-inference/README.md), [optional standalone C kernels](10-kernels/README.md), [serving in Rust](11-serving/README.md), [training at scale](12-training-at-scale/README.md), and an optional [post-training](13-post-training/README.md) part. [Vision foundations](14-vision/README.md) and [speech recognition](15-speech/README.md) then extend the system to images and audio, and the [capstones](capstones/README.md) run it on one goal.

Because the system is built in passes that interleave these parts with chapters from other books, each part belongs to one or more passes of the [course path](paths/course/README.md). The table gives the mapping.

| Part | Course pass |
|---|---|
| [Foundations](01-foundations/README.md) | 1 (the tracer), 2 |
| [Tokenizers](02-tokenizers/README.md) | 3 |
| [Statistical language models](03-statistical-lm/README.md) | 3 |
| [Recurrent networks](04-recurrent/README.md) | 4 |
| [Attention origins](05-attention-origins/README.md) | 4 |
| [The transformer (2017)](06-transformer-2017/README.md) | 5 |
| [Objectives and adaptation](07-objectives/README.md) | 5 |
| [The modern decoder block](08-modern-block/README.md) | 5 |
| [Inference](09-inference/README.md) | 6 |
| [Optional standalone C kernels](10-kernels/README.md) | 6 |
| [Serving](11-serving/README.md) | 1 (the tracer), 7 |
| [Training at scale](12-training-at-scale/README.md) | 9 |
| [Post-training (optional)](13-post-training/README.md) | 10 |
| [Vision foundations](14-vision/README.md) | 12 |
| [Speech recognition](15-speech/README.md) | 12 |
| [Capstones](capstones/README.md) | 9, and 10 for the optional second |

## Connections to other books

The model stack in this book meets several other books at specific sections. In *Mathematics for Computing*, [vectors, matrices, and matmul](../mathematics-for-computing/04-linear-algebra/01-vectors-matrices-and-matmul.md) is the matrix multiply every logit goes through. In *AI Platform Engineering*, the [tracer gateway](../ai-platform-engineering/12-gateway/01-streaming-proxy.md) is the front door to the engine. In *Infrastructure*, the [tracer deploy](../infrastructure/01-containers-kubernetes/01-tracer-deploy.md) builds the images and charts the engine runs in. And [LLM systems and inference](../machine-learning/04-llm-systems/README.md) in *Machine Learning and AI* describes the production systems this spine rebuilds in miniature.

## The same ideas in industry

None of the design choices here are academic; each mirrors a widely used open-source system.

| Project | Practice it shares with this book |
|---|---|
| Hugging Face | safetensors, `transformers` checkpoints, `tokenizers` |
| llama.cpp / ggml | C kernels behind a stable ABI, served over an OpenAI-compatible HTTP API |
| vLLM, SGLang | Python model code over custom kernels, a separate serving engine |

## Further reading

- Karpathy, [*Neural Networks: Zero to Hero*](https://karpathy.ai/zero-to-hero.html) (free).
- Jurafsky and Martin, [*Speech and Language Processing*](https://web.stanford.edu/~jurafsky/slp3/) (free), chapters 3 (n-gram LMs) and 9 (transformers).
- Raschka, *Build a Large Language Model (From Scratch)*.
