# ML & AI (18-37 weeks)

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
