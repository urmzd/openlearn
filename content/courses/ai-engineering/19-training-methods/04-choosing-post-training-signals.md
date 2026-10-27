# Choosing post-training signals

This part of [Choosing Training and Adaptation Methods](README.md) develops the mechanisms below. Work the numerical examples or trace the code before checking the answer.

> **Learning objectives**
> - Match demonstrations, preferences, rewards, and teacher distributions to objectives.

## Post-training

Post-training adapts a base model to particular response, preference, reward, or deployment goals. Base-model capability, data coverage, retrieval, and serving behavior remain relevant. The three main tools need three different kinds of data: SFT teaches format from demonstrations, preference optimization such as DPO teaches taste from comparisons, and RL with verifiable rewards such as GRPO teaches skills that a grader can check. The key papers are [InstructGPT](https://arxiv.org/abs/2203.02155), [DPO](https://arxiv.org/abs/2305.18290), [DeepSeekMath](https://arxiv.org/abs/2402.03300), [DeepSeek-R1](https://arxiv.org/abs/2501.12948), and [Tulu 3](https://arxiv.org/abs/2411.15124).

### Supervised fine-tuning

SFT is next-token training on demonstrations, with one important change: **loss masking**. You train only on the response tokens $R$, labeling prompt and system positions `-100` (the `ignore_index`), so the loss is

$$L = -\frac{1}{|R|}\sum_{t \in R} \log p(y_t \mid y_{<t}, x).$$

Training on prompts would waste capacity on imitating users. [`train_step_torch.py`](code/train_step_torch.py) implements the masking.

The model also learns the exact special-token format of its **chat template** (`<|start_header_id|>`, `<|im_start|>`, and so on). Train and serve with the *same* template, taken from the tokenizer via `apply_chat_template`.

> **Warning:** Template mismatch between training and serving can silently degrade a fine-tune. Nothing crashes; the model just behaves worse.

To avoid wasting compute on padding, **packing** concatenates short examples into full-length sequences. Pair it with **document masking**, a position-ID reset plus block-diagonal attention (for example FlashAttention's varlen kernels), so packed examples do not attend to each other.

Typical hyperparameters are 1-3 epochs, a learning rate around 1e-5 for full fine-tuning and 1e-4 to 2e-4 for LoRA, with cosine decay and warmup. Quality beats quantity: [LIMA](https://arxiv.org/abs/2305.11206) got strong results from 1,000 curated examples.

### RLHF with PPO

Classic RLHF, as in InstructGPT, has three parts. First, train a **reward model** on preference pairs with the **Bradley-Terry** loss, where $y_w$ is the preferred response and $y_l$ the rejected one:

$$L_{RM} = -\log \sigma\big(r_\phi(x, y_w) - r_\phi(x, y_l)\big)$$

Second, use RL to maximize that reward with a KL penalty that keeps the policy near the SFT reference:

$$\max_\pi\ \mathbb{E}_{x,\, y \sim \pi}\big[r_\phi(x, y)\big] - \beta \cdot \mathrm{KL}\big(\pi(\cdot \mid x)\,\|\,\pi_{\text{ref}}(\cdot \mid x)\big)$$

In practice the KL term is applied per token as $r_t = -\beta\big(\log \pi(y_t \mid \cdot) - \log \pi_{\text{ref}}(y_t \mid \cdot)\big)$, with $r_\phi$ added at the final token.

Third, optimize with **PPO** ([Schulman et al.](https://arxiv.org/abs/1707.06347)), whose clipped surrogate uses the ratio $\rho_t = \pi_\theta / \pi_{\text{old}}$:

$$L = \mathbb{E}_t\Big[\min\big(\rho_t A_t,\ \mathrm{clip}(\rho_t, 1-\epsilon, 1+\epsilon)\, A_t\big)\Big]$$

with advantages $A_t$ computed by GAE from a learned **value model**, the critic.

Count the actual resident tensors and optimizer state: actor, reference, reward function, and critic are distinct roles but can share a backbone, cache scores, or be swapped. Generation sits inside the learning loop. The [PPO and GAE part](../18-post-training/06-ppo-with-gae.md) derives the masks, targets, and controls needed to evaluate this design.

### Deriving DPO

DPO is RLHF with the reward model solved away. Start from the same KL-constrained objective. Its optimum has a closed form, a Gibbs distribution:

$$\pi^*(y \mid x) = \frac{\pi_{\text{ref}}(y \mid x)\, \exp\big(r(x,y)/\beta\big)}{Z(x)}, \qquad Z(x) = \sum_y \pi_{\text{ref}}(y \mid x)\, \exp\big(r(x,y)/\beta\big)$$

Solve it for the reward:

$$r(x,y) = \beta \log \frac{\pi^*(y \mid x)}{\pi_{\text{ref}}(y \mid x)} + \beta \log Z(x)$$

Now substitute this into the Bradley-Terry model, $p(y_w \succ y_l \mid x) = \sigma\big(r(x,y_w) - r(x,y_l)\big)$. The intractable $\beta \log Z(x)$ appears in both terms and **cancels**. Parametrize $\pi^*$ by the policy $\pi_\theta$ and maximize the likelihood of the observed preferences:

$$L_{\text{DPO}}(\theta) = -\mathbb{E}_{(x, y_w, y_l)}\left[\log \sigma\left(\beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}\right)\right]$$

Call the bracketed margin $h$. The gradient is

$$\nabla L = -\beta\, \sigma(-h)\,\big[\nabla \log \pi_\theta(y_w \mid x) - \nabla \log \pi_\theta(y_l \mid x)\big].$$

The weight $\sigma(-h)$ is what makes DPO sensible: it focuses learning on pairs that the implicit reward $\beta \log \pi_\theta / \pi_{\text{ref}}$ currently ranks wrongly. And at initialization, when $\pi_\theta = \pi_{\text{ref}}$, the margin is zero and the loss is exactly $\log 2$. Both facts are asserted in [`dpo_loss.py`](code/dpo_loss.py), which checks the gradient against central finite differences.

The practical consequences follow from the derivation. DPO is offline: no reward model, no sampling, and only two models, the policy and a frozen reference whose log-probs can be precomputed. Its memory cost is that of SFT plus one forward pass. The coefficient $\beta$ changes the preference-loss parameterization inherited from reference regularization; it does not enforce a hard measured-KL bound. Tune it against held-out behavior and drift. Its known failure modes are likelihood displacement, where both the chosen and the rejected log-probs fall, and verbosity bias. Variants include IPO, KTO (which works on unpaired thumbs up/down), ORPO, and SimPO (reference-free). Online or iterative preference collection can reduce mismatch between the dataset and the current model, but its benefit depends on label quality, coverage, and cost. Compare it with fixed-pair training on the same held-out suite.

### GRPO and RL with verifiable rewards

DPO removes the reward model; **GRPO** ([DeepSeekMath](https://arxiv.org/abs/2402.03300)) keeps the RL loop but removes the critic. For each prompt, sample a **group** of $G$ outputs (8-64), score each, and use the group itself as the baseline. Every token of output $i$ gets the same advantage:

$$\hat{A}_i = \frac{r_i - \mathrm{mean}(r_1, \dots, r_G)}{\mathrm{std}(r_1, \dots, r_G)+\varepsilon}$$

The objective is the PPO surrogate averaged per output and per token, with a KL penalty:

$$J(\theta) = \mathbb{E}\left[\frac{1}{G}\sum_i \frac{1}{|o_i|}\sum_t \Big(\min\big(\rho_{i,t}\hat{A}_i,\ \mathrm{clip}(\rho_{i,t}, 1-\epsilon, 1+\epsilon)\,\hat{A}_i\big) - \beta\, \mathrm{KL}_t(\pi_\theta \,\|\, \pi_{\text{ref}})\Big)\right]$$

The per-token penalty can use the pointwise nonnegative k3 estimator below. Its expectation is forward KL when actions are sampled from the current policy with matching support; stale or old-policy samples need the stated correction or approximation:

$$\mathrm{KL}_t = \frac{\pi_{\text{ref}}}{\pi_\theta} - \log \frac{\pi_{\text{ref}}}{\pi_\theta} - 1$$

The group baseline avoids learning a value function. Inclusive group means and sampled standard deviations affect the gradient estimator; they are not generally an unbiased action-independent baseline. Memory savings depend on critic sharing, group size, reward scoring, and generation layout. See [rollout systems and variants](../18-post-training/08-rollout-systems-and-grpo-variants.md).

GRPO pairs naturally with **RLVR**, RL with verifiable rewards (named in Tulu 3), where the reward is a program rather than a model: the math answer matches, the unit tests pass, the JSON validates against a schema, the format regex matches. Removing a learned reward model removes one source of reward-model error, while incomplete tests, parser bugs, and format shortcuts remain exploitable. **DeepSeek-R1** ([2501.12948](https://arxiv.org/abs/2501.12948)) is the landmark result. R1-Zero ran GRPO with rule-based accuracy and format rewards directly on the base model, and long chain-of-thought and self-verification emerged. R1 itself added a cold-start SFT, then RL, then rejection-sampling SFT, then a final RL stage, and was distilled into small dense models.

Three follow-ups are worth naming: DAPO ([2503.14476](https://arxiv.org/abs/2503.14476)), with clip-higher, dynamic sampling, and a token-level loss; Dr. GRPO ([2503.20783](https://arxiv.org/abs/2503.20783)), which removes the length and std normalization biases; and GSPO ([2507.18071](https://arxiv.org/abs/2507.18071)), which uses sequence-level ratios and is stabler for MoE. Fireworks and others sell this as **RFT**, reinforcement fine-tuning, where the customer supplies the grader.

The hard part of RL is a systems problem: RL is mostly **generation**. Rollouts run in vLLM or SGLang, training runs in FSDP or Megatron, and weights sync between them every step. Asynchronous, off-policy rollouts trade staleness for throughput.

### Distillation

The last post-training tool trains a student on a teacher's outputs. **Sequence-level** distillation is SFT on teacher generations, which is how R1's distilled models were built, and needs only text. **Logit-level** distillation minimizes KL to the teacher's distribution ([Hinton et al.](https://arxiv.org/abs/1503.02531)) and needs a shared tokenizer and the teacher's logits. **On-policy distillation** (GKD) scores the student's own samples with the teacher. For a customer, "distill a frontier model's behavior on your task into an 8B" is often the best cost play, subject to the teacher's license terms.

### Learner and inference runtime responsibilities

A trainer owns differentiation, optimizer state, objective masks/reductions, data scheduling, and checkpoints. A rollout runtime such as vLLM or SGLang owns generation and its behavior probabilities. The orchestration layer owns revision-consistent weight transfer, scoring, queueing, and policy-lag limits. Some packages bundle several roles, but the contracts remain separate.

Use a trainer only after checking that its installed version implements the intended objective and normalization. The [versioned TRL GRPO reference](https://huggingface.co/docs/trl/v0.23.1/grpo_trainer) illustrates how the same trainer exposes several recipe choices. Distributed and adapter support must be verified with the actual model and topology. The [local lab](../20-inference-engines/14-local-serving-and-validation.md) starts with the shared training-to-serving validation loop.

## Summary

Match demonstrations, preferences, rewards, and teacher distributions to objectives. State the assumptions before applying the result to a different workload.

## Review questions

1. What does a falling reward loss fail to establish?

   <details><summary>Answer</summary>

   It does not establish better held-out task behavior, calibrated rewards, or a policy that serves correctly.

   </details>

[Chapter contents](README.md) · [Next: Parameter-efficient fine-tuning](05-parameter-efficient-fine-tuning.md)
