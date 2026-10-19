# Concepts and techniques

This overview gives you the whole of *Build an LLM* in one sitting: the single idea the book is organized around, what the thinnest working system contains, how the later parts upgrade it, and where the book connects to the rest of the library. Read it after the [book's front matter](README.md) and alongside [the system the solutions build](paths/course/SYSTEM.md), which is the component-and-contract map of the same story.

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

You build all of this in [the byte bigram](p00-foundations/00-byte-bigram.md) and serve it from [your first endpoint](p10-serving/00-your-first-endpoint.md).

## The spine: upgrading one component at a time

After the tracer, each part of the book replaces or extends one component behind the same contracts. In order, the parts are autograd and training ([Foundations](p00-foundations/README.md)), [tokenizers](p01-tokenizers/README.md), [statistical language models](p02-statistical-lm/README.md), [recurrent networks](p03-recurrent/README.md), [attention origins](p04-attention-origins/README.md), [the 2017 transformer](p05-transformer-2017/README.md), [objectives and adaptation](p06-objectives/README.md), [the modern decoder block](p07-modern-block/README.md), [inference](p08-inference/README.md), [optional standalone C kernels](p09-kernels/README.md), [serving in Rust](p10-serving/README.md), [training at scale](p11-training-at-scale/README.md), and an optional [post-training](p12-post-training/README.md) part. [Vision foundations](p13-vision/README.md) and [speech recognition](p14-speech/README.md) then extend the system to images and audio, and the [capstones](capstones/README.md) run it on one goal.

Because the system is built in passes that interleave these parts with chapters from other books, each part belongs to one or more passes of the [course path](paths/course/README.md). The table gives the mapping.

| Part | Course pass |
|---|---|
| [Foundations](p00-foundations/README.md) | 1 (the tracer), 2 |
| [Tokenizers](p01-tokenizers/README.md) | 3 |
| [Statistical language models](p02-statistical-lm/README.md) | 3 |
| [Recurrent networks](p03-recurrent/README.md) | 4 |
| [Attention origins](p04-attention-origins/README.md) | 4 |
| [The transformer (2017)](p05-transformer-2017/README.md) | 5 |
| [Objectives and adaptation](p06-objectives/README.md) | 5 |
| [The modern decoder block](p07-modern-block/README.md) | 5 |
| [Inference](p08-inference/README.md) | 6 |
| [Optional standalone C kernels](p09-kernels/README.md) | 6 |
| [Serving](p10-serving/README.md) | 1 (the tracer), 7 |
| [Training at scale](p11-training-at-scale/README.md) | 9 |
| [Post-training (optional)](p12-post-training/README.md) | 10 |
| [Vision foundations](p13-vision/README.md) | 12 |
| [Speech recognition](p14-speech/README.md) | 12 |
| [Capstones](capstones/README.md) | 9, and 10 for the optional second |

## Connections to other books

The model stack in this book meets several other books at specific sections. In *Mathematics for Computing*, [vectors, matrices, and matmul](../mathematics-for-computing/03-linear-algebra/01-vectors-matrices-and-matmul.md) is the matrix multiply every logit goes through. In *AI Platform Engineering*, the [tracer gateway](../ai-platform-engineering/12-gateway/00-streaming-proxy.md) is the front door to the engine. In *Infrastructure*, the [tracer deploy](../infrastructure/01-containers-kubernetes/00-tracer-deploy.md) builds the images and charts the engine runs in. And [LLM systems and inference](../machine-learning/04-llm-systems/README.md) in *Machine Learning* describes the production systems this spine rebuilds in miniature.

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
