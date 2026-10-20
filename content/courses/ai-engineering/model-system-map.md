# Concepts and techniques

This overview gives you the whole of *AI Engineering* in one sitting: the single idea the book is organized around, what the thinnest working system contains, how the later parts upgrade it, and where the book connects to the rest of the library. Read it after the [book's front matter](README.md) and alongside [the system the solutions build](paths/course/SYSTEM.md), which is the component-and-contract map of the same story.

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

You build all of this in [the byte bigram](06-numerics-and-autograd/01-byte-bigram.md) and serve it from [your first endpoint](23-model-serving/01-your-first-endpoint.md).

## The spine: upgrading one component at a time

After the tracer, each part of the book replaces or extends one component behind the same contracts. In order, the parts are autograd and training ([Foundations](06-numerics-and-autograd/README.md)), [tokenizers](07-tokenizers/README.md), [statistical language models](08-statistical-language-models/README.md), [recurrent networks](09-recurrent-networks/README.md), [attention origins](10-attention/README.md), [the 2017 transformer](11-transformer/README.md), [objectives and adaptation](12-objectives-and-adaptation/README.md), [the modern decoder block](13-decoder-block/README.md), [inference](21-inference/README.md), [optional standalone C kernels](22-kernels/README.md), [serving in Rust](23-model-serving/README.md), [training at scale](17-training-at-scale/README.md), and an optional [post-training](18-post-training/README.md) part. [Vision foundations](14-vision/README.md) and [speech recognition](15-speech/README.md) then extend the system to images and audio, and the [capstones](capstones/README.md) run it on one goal.

Because the system is built in passes that interleave these parts with chapters from other books, each part belongs to one or more passes of the [course path](paths/course/README.md). The table gives the mapping.

| Part | Course pass |
|---|---|
| [Foundations](06-numerics-and-autograd/README.md) | 1 (the tracer), 2 |
| [Tokenizers](07-tokenizers/README.md) | 3 |
| [Statistical language models](08-statistical-language-models/README.md) | 3 |
| [Recurrent networks](09-recurrent-networks/README.md) | 4 |
| [Attention origins](10-attention/README.md) | 4 |
| [The transformer (2017)](11-transformer/README.md) | 5 |
| [Objectives and adaptation](12-objectives-and-adaptation/README.md) | 5 |
| [The modern decoder block](13-decoder-block/README.md) | 5 |
| [Inference](21-inference/README.md) | 6 |
| [Optional standalone C kernels](22-kernels/README.md) | 6 |
| [Serving](23-model-serving/README.md) | 1 (the tracer), 7 |
| [Training at scale](17-training-at-scale/README.md) | 9 |
| [Post-training (optional)](18-post-training/README.md) | 10 |
| [Vision foundations](14-vision/README.md) | 12 |
| [Speech recognition](15-speech/README.md) | 12 |
| [Capstones](capstones/README.md) | 9, and 10 for the optional second |

## Connections to other books

The model stack in this book meets several other books at specific sections. In *Mathematics*, [vectors, matrices, and matmul](../mathematics/09-linear-algebra/01-vectors-matrices-and-matmul.md) is the matrix multiply every logit goes through. In *AI Engineering*, the [tracer gateway](25-ai-gateway/01-streaming-proxy.md) is the front door to the engine. In *Systems Engineering*, the [tracer deploy](../systems/10-containers-kubernetes/01-tracer-deploy.md) builds the images and charts the engine runs in. And [LLM systems and inference](20-inference-engines/README.md) in *AI Engineering* describes the production systems this spine rebuilds in miniature.

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
