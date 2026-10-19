# A reading plan for Machine Learning and AI

This plan paces the book over roughly 18 to 37 weeks at ten to twelve hours a week, depending on how many chapters you read. It is a suggestion, not a schedule to keep: the [front matter](README.md) offers shorter reading paths for specific goals, and you can skip or reorder chapters to match them. The plan assumes you have the mathematical foundations in hand, meaning linear algebra, calculus, and probability from *Mathematics for Computing*.

## Statistical learning (about five weeks)

Begin with [Statistical Learning](01-statistical-learning/README.md), reading *An Introduction to Statistical Learning* alongside it. The first week is the statistical learning framework and the bias-variance tradeoff (chapter 2 of ISLR). The second is linear regression and model selection (chapter 3), and the third is classification with logistic regression, LDA, and naive Bayes (chapter 4). In the fourth week cover resampling and regularization with ridge and lasso (chapters 5 and 6), and in the fifth, trees, random forests, boosting, SVMs, and unsupervised learning (chapters 8, 9, and 12).

## Deep learning (about eight weeks)

[Deep Learning](02-deep-learning/README.md) takes the next eight weeks, following Goodfellow, Bengio, and Courville. Start with the machine-learning basics, feedforward networks, and backpropagation (chapters 5 and 6), then regularization and optimization with SGD and Adam (chapters 7 and 8). Spend a week on CNNs and their architectures from LeNet to ResNet (chapter 9), and another on RNNs, LSTMs, and sequence-to-sequence models (chapter 10). The fifth week is practical methodology, debugging, and hyperparameter tuning (chapter 11), and the sixth is autoencoders and generative models, VAEs and GANs (chapters 14 and 20). The last two weeks go beyond the book into the papers: transformers, attention, BERT, GPT, and scaling laws, and then representation learning, self-supervised learning, and transfer learning (with chapter 15).

If you are interested in foundations, this is the time to read [Neural Architectures and the History of Deep Learning](06-neural-architectures/README.md) alongside it; see below.

## Reinforcement learning (about six weeks)

[Reinforcement Learning](03-reinforcement-learning/README.md) follows Sutton and Barto. Week one is bandits, MDPs, and the Bellman equations (chapters 2 and 3); week two is dynamic programming, Monte Carlo methods, TD learning, and Q-learning (chapters 4 to 6); week three is function approximation and DQN (chapters 9 and 10); and week four is policy gradients, actor-critic methods, and A2C/A3C (chapter 13). Use OpenAI's Spinning Up for the fifth week on PPO, DDPG, TD3, and SAC, and the research papers for the sixth on RLHF, alignment, and model-based RL.

## LLM systems and inference (about six weeks)

[LLM Systems and Inference](04-llm-systems/README.md) moves from models to serving, with *Machine Learning Systems* and the vLLM docs as companions. Start with the serving view of a transformer: prefill versus decode and the TTFT and TPOT metrics. Then study the KV cache, PagedAttention, and prefix caching with the vLLM PagedAttention paper, followed by continuous batching, FlashAttention, and kernel optimization with the TGI and FlashAttention papers. The fourth week is quantization (GPTQ, AWQ, FP8) and speculative decoding, for which Lilian Weng's inference blog post is a good guide, and the fifth is tensor, pipeline, and expert parallelism and multi-GPU serving, with the TensorRT-LLM docs. Finish with GPU clusters, Kubernetes deployment, autoscaling, and SLOs, reading the Kubernetes, KServe, and Ray Serve docs.

## Foundation models and architectures (about five weeks)

[Foundation Models and Architectures](05-foundation-models/README.md) opens with foundation models, the shift to AI engineering, and the adaptation ladder (chapters 1 and 2 of Huyen's *AI Engineering*). The second week is the transformer architecture variants, RMSNorm, RoPE, SwiGLU, and MoE, read through the Llama and Gemma technical reports; the third is attention mechanisms (MHA, MQA, GQA, MLA, sliding-window, and sparse attention) with the DeepSeek and Mistral papers. In the fourth week compare modalities, text, audio, and vision, and multimodal fusion with CLIP and VLMs, using the Hugging Face Transformers docs and the Whisper and CLIP papers. The last week is diffusion models: DDPM, latent diffusion, DiT, and flow matching, with the Hugging Face Diffusers docs and Lilian Weng's post.

## Neural architectures and the history of deep learning (about three weeks)

[Neural Architectures and the History of Deep Learning](06-neural-architectures/README.md) is a foundations deep dive that is best read alongside the first four weeks of Deep Learning. Derive each architecture from its math and code it from scratch. The first week is the history, from the perceptron through the XOR winter to backpropagation, along with the foundational math and coding a perceptron and an MLP with backprop (CS231n and Rumelhart et al. 1986). The second is CNNs: convolution math, pooling, and the line from LeNet through AlexNet and VGG to ResNet, with a deep dive on AlexNet (CS231n and the AlexNet and ResNet papers). The third is RNNs, backpropagation through time, and vanishing gradients, the LSTM and GRU gate equations, and a comparison of CNNs, RNNs, and transformers (the LSTM paper and chapter 10 of Goodfellow).

## Training and post-training (about four weeks)

[Training and Post-Training](07-training-and-post-training/README.md) shows how a model is made before it is served, and pairs with the reinforcement learning weeks for its RLHF and GRPO material. In the first week work through pretraining: cross entropy, the $6ND$ FLOP count, Chinchilla versus inference-optimal overtraining, and MFU, with the Chinchilla paper and the chapter's `code/memory_calc.py`. In the second, study training memory and parallelism, from Adam's optimizer state through ZeRO and FSDP2, TP, PP, CP, and EP, to JAX sharding, reading the ZeRO and Megatron papers and the PyTorch and JAX docs. The third week is post-training: SFT, PPO-based RLHF, the derivation of DPO, and GRPO with verifiable rewards, with the InstructGPT, DPO, and DeepSeekMath papers. Finish with parameter-efficient fine-tuning and scoping: the LoRA, QLoRA, and DoRA math, running a LoRA fine-tune yourself, and evaluating base against tuned, with the LoRA and QLoRA papers and `code/lora_from_scratch.py`.

## Going further

To see the whole stack built end to end, continue with *[Build Your Own LLM System](../build-an-llm/README.md)*, following its [course path](../build-an-llm/paths/course/README.md), which walks through a worked solution for every layer from autograd to serving and post-training.
