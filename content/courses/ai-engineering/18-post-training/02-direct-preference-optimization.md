# Direct preference optimization and IPO

[Supervised fine-tuning](01-supervised-fine-tuning.md) teaches a model the form of a response, but it does not encode which of two valid responses a user prefers. Paired comparisons (a chosen and a rejected completion for the same prompt) carry exactly that signal. The classical way to use them, RLHF, fits a separate reward model and then runs reinforcement learning against it. Direct preference optimization (DPO) shows that, for the KL-regularized objective RLHF optimizes, you can skip both and train the policy on the pairs with a single classification-style loss.

This section derives DPO from that objective, computes it by hand, and implements it together with its variant IPO, which regresses the supplied pair score to a finite target. Neither objective imposes a hard global KL bound. The derivation uses the KL divergence of [Entropy, cross-entropy, KL, JS, and the k3 estimator](../../mathematics/17-information-theory/01-entropy-cross-entropy-and-kl.md) and the closed-form optimum of the KL-regularized objective worked in [Optimization problem set, part b](../../mathematics/16-optimization/06-problem-set-b.md), both in *Mathematics*. The solution is [`dpo.py`](../../../../course/solutions/python/tinyllm/post/dpo.py) (`dpo_loss`, `ipo_loss`), and the contract is [`dpo.pyi`](../../../../course/contracts/py/tinyllm/post/dpo.pyi).

> **Learning objectives**
> - Derive the DPO loss from the KL-regularized reward objective and the Bradley-Terry preference model.
> - Explain why the partition function cancels and why the reference policy must appear in the loss.
> - Compute DPO and IPO losses by hand and implement them stably.
> - Contrast DPO's margin incentive with IPO's finite fitted target, and distinguish either from a global KL constraint.

## The objective

Write $\pi$ for the trainable policy, $\pi_{\text{ref}}$ for the frozen reference policy (the SFT model), $\beta$ for the preference strength, which doubles as the KL scale, and $\sigma$ for the logistic sigmoid. For a chosen completion $y^+$ and a rejected completion $y^-$ of the same prompt $x$, define

$$\Delta_\pi = \log \pi(y^+ \mid x) - \log \pi(y^- \mid x), \qquad \Delta_{\text{ref}} = \log \pi_{\text{ref}}(y^+ \mid x) - \log \pi_{\text{ref}}(y^- \mid x).$$

The DPO loss for one pair is

$$\mathcal{L}_{\text{DPO}} = -\log \sigma\big(\beta(\Delta_\pi - \Delta_{\text{ref}})\big).$$

The reference policy is fixed. When the policy equals the reference, the margin is zero and the per-pair loss is $\log 2$. The loss falls only as the policy prefers the chosen answer more than the reference already did.

## Where the loss comes from

The loss is not a guess; it falls out of the objective DPO replaces in four steps.

Start from RLHF, which maximizes $\mathbb{E}[r(x, y)] - \beta\, \mathrm{KL}(\pi \,\|\, \pi_{\text{ref}})$. Its optimum has a closed form:

$$\pi^*(y \mid x) = \frac{\pi_{\text{ref}}(y \mid x)\, e^{r(x, y)/\beta}}{Z(x)}.$$

Invert it. Solving for the reward gives $r(x, y) = \beta \log \frac{\pi^*(y \mid x)}{\pi_{\text{ref}}(y \mid x)} + \beta \log Z(x)$: the reward is a log-ratio against the reference, plus a term that depends only on the prompt.

Now use a pairwise preference model. Bradley-Terry says $P(y^+ \succ y^-) = \sigma\big(r(x, y^+) - r(x, y^-)\big)$. In the difference, $\beta \log Z(x)$ cancels, and that is the whole trick: no partition function, no reward model.

What is left is the margin, $r^+ - r^- = \beta\big((\log \pi(y^+) - \log \pi(y^-)) - (\log \pi_{\text{ref}}(y^+) - \log \pi_{\text{ref}}(y^-))\big) = \beta(\Delta_\pi - \Delta_{\text{ref}})$. Maximizing the log-likelihood of the observed preferences is minimizing $-\log \sigma(\beta(\Delta_\pi - \Delta_{\text{ref}}))$.

## IPO: same margin, different shape

DPO's loss keeps decreasing as the margin grows, so on deterministic preferences it pushes the margin toward infinity. IPO keeps the margin $z = \Delta_\pi - \Delta_{\text{ref}}$ but regresses it to a finite target $1/(2\beta)$ with a squared loss, $(z - 1/(2\beta))^2$, which gives a finite target for that observed pair score. It does not bound full-policy KL or behavior on unseen outputs. So IPO is a different loss shape, not just a different learning rate.

## Working it by hand

With $\Delta_\pi = 1.2$, $\Delta_{\text{ref}} = 0.2$, and $\beta = 0.1$, the scaled margin is 0.1, so the DPO loss is $-\log \sigma(0.1) = \log(1 + e^{-0.1}) \approx 0.6444$. If the two policies are equal, both deltas match and the loss is $\log 2 \approx 0.6931$.

IPO on the same pair uses the unscaled margin $z = \Delta_\pi - \Delta_{\text{ref}} = 1.0$ and the target $1/(2\beta) = 5$, so the loss is $(1 - 5)^2 = 16$.

## Designing the API

`dpo_loss(policy_chosen, policy_rejected, reference_chosen, reference_rejected, beta=0.1)` and `ipo_loss(...)`, with the same arguments, take completion log probabilities (summed over each completion's tokens), one entry per pair, and return the mean loss over pairs as a `float`. The reference log probabilities are inputs, computed once by the frozen reference model for this exercise; the reference weights are never updated. The helper accepts supplied sequence sums for both losses. Original-style IPO recipes may average token log probabilities instead; state and match that convention before transferring hyperparameters.

## Building it step by step

Two implementation concerns remain once the derivation is done. The first is stability: $-\log \sigma(z) = \log(1 + e^{-z})$ overflows for large negative $z$ if written literally, while `logaddexp(0, -z)` computes the same value without overflow. The second is alignment: each array entry is one pair, so all four arrays must have the same shape.

The shared margin checks shapes, because broadcasting a scalar reference against a batch would silently give every pair the same reference:

```python
def _margin(pc, pr, rc, rr):
    pc, pr, rc, rr = (np.asarray(x, dtype=np.float64) for x in (pc, pr, rc, rr))
    if not (pc.shape == pr.shape == rc.shape == rr.shape):
        raise ValueError("preference log-probability shapes differ")
    # (policy chosen - policy rejected) - (reference chosen - reference rejected) = Δπ - Δref
    return (pc - pr) - (rc - rr)
```

DPO scales the margin by $\beta$ and applies the logistic loss:

```python
def dpo_loss(policy_chosen, policy_rejected, reference_chosen, reference_rejected, beta=0.1):
    if beta <= 0:
        raise ValueError("beta must be positive")
    z = beta * _margin(policy_chosen, policy_rejected, reference_chosen, reference_rejected)
    # -log σ(z) = log(e^0 + e^-z), computed without overflow
    return float(np.mean(np.logaddexp(0.0, -z)))
```

IPO uses the unscaled margin and a squared distance to $1/(2\beta)$:

```python
def ipo_loss(policy_chosen, policy_rejected, reference_chosen, reference_rejected, beta=0.1):
    if beta <= 0:
        raise ValueError("beta must be positive")
    z = _margin(policy_chosen, policy_rejected, reference_chosen, reference_rejected)
    return float(np.mean((z - 1.0 / (2.0 * beta)) ** 2))
```

The full solution is [`dpo.py`](../../../../course/solutions/python/tinyllm/post/dpo.py).

### Why this design

Both losses are $O(\text{pairs})$ once the four log probabilities are known. The expensive part is the forward passes: two through the policy, and two through the reference, which can be computed once per dataset and cached because the reference is frozen.

Compared with a reward model plus PPO, DPO replaces the reward model and the RL loop with one classification-style loss. The price is that DPO only learns from the pairs it is given; it cannot explore. Within the implementation, `logaddexp` beats the literal `-log(sigmoid(z))`, which rounds `sigmoid(z)` to 0 for large negative $z$ and returns `inf`; `logaddexp` remains numerically stable over a much wider range. And IPO trades DPO's unbounded push for a fixed target margin: with $\beta = 0.1$ the target is 5 nats, a concrete fitted target for the chosen-over-rejected score difference, not a hard bound on all policy changes.

## What can go wrong

Three checks pin the implementation: the worked DPO loss must come out near 0.6444; a policy equal to the reference must give $\log 2$, which verifies the reference correction; and the IPO variant must give 16.0 on the worked pair.

The two classic bugs both live in the margin. Reversing chosen and rejected flips the sign of every update and trains the model toward the rejected answers; the solution subtracts rejected from chosen for both models in one place, `_margin`. Omitting the reference log probabilities removes the tie to the SFT model; the reference difference is part of `_margin`, shared by DPO and IPO.

> **Warning:** Without the reference terms the loss becomes $-\log \sigma(\beta \Delta_\pi)$, which rewards any increase in the chosen-over-rejected log ratio, including the part the SFT model already had. Nothing then keeps the policy near the reference.

## Where this leads

The [post-trained capstone](../capstones/02-post-trained.md) applies DPO to preference pairs after supervised fine-tuning. The next section, [Group relative policy optimization with verifiable rewards](03-grpo-with-verifiable-rewards.md), takes the other route: when answers can be checked automatically, it trains on sampled rollouts instead of fixed pairs. Production preference pipelines add handling of annotator disagreement and pair provenance, and online preference evaluation adds position controls and outcome measures beyond training loss (see the TRL preference trainers).

Compare [preference optimization variants](11-preference-optimization-variants.md) for IPO score normalization, noisy labels, KTO, ORPO, SimPO, online data, and a counterexample to treating a reference score as a trust-region guarantee.

### Further reading

- Rafailov et al., [*Direct Preference Optimization*](https://arxiv.org/abs/2305.18290) (2023).
- Azar et al., [*A General Theoretical Paradigm to Understand Learning from Human Preferences*](https://arxiv.org/abs/2310.12036) (IPO, 2023).
- Hugging Face, [TRL 0.23.1 DPO trainer](https://huggingface.co/docs/trl/v0.23.1/dpo_trainer).

## Key terms

- **reference policy**: the frozen model (usually the SFT checkpoint) against which the policy's log-ratios are measured.
- **preference margin**: $\Delta_\pi - \Delta_{\text{ref}}$, how much more the policy prefers the chosen completion than the reference does.
- **Bradley-Terry model**: $P(y^+ \succ y^-) = \sigma(r^+ - r^-)$, a pairwise preference probability from scalar rewards.
- **DPO**: minimizing $-\log \sigma(\beta \cdot \text{margin})$ over preference pairs, equivalent to fitting the KL-regularized reward objective.
- **IPO**: regressing the margin to the target $1/(2\beta)$ with a squared loss.

## Summary

- DPO optimizes preference odds relative to a frozen reference policy, with no reward model and no RL loop.
- The prompt-only term $\beta \log Z(x)$ cancels in the pairwise difference, which is why the loss needs only four log probabilities.
- The reference defines relative pair scores; it does not enforce a global KL bound. Policy equal to reference gives loss $\log 2$, but that loss alone does not prove equality of the full policies.
- IPO changes the loss shape, regressing the margin to $1/(2\beta)$, and is not just a different learning rate.

## Review questions

1. Why does the loss equal $\log 2$ when the policy equals the reference?

   <details><summary>Answer</summary>

   Then $\Delta_\pi = \Delta_{\text{ref}}$, the margin is 0, and $-\log \sigma(0) = -\log(1/2) = \log 2$. The loss starts there and falls only as the policy prefers the chosen answer *more than the reference does*.

   </details>

2. Why does $\log Z(x)$ not appear anywhere in the code?

   <details><summary>Answer</summary>

   It depends only on the prompt, so it is the same for the chosen and the rejected completion and cancels in the reward difference that Bradley-Terry uses.

   </details>

3. What happens if you drop the reference terms?

   <details><summary>Answer</summary>

   The loss becomes $-\log \sigma(\beta \Delta_\pi)$, which rewards any increase in the chosen-over-rejected log ratio, including the part the SFT model already had. Nothing ties the policy to the reference, so it can drift arbitrarily far; the worked example's loss would also change from 0.6444 to $-\log \sigma(0.12) \approx 0.6349$.

   </details>

4. Why is IPO's margin not multiplied by $\beta$?

   <details><summary>Answer</summary>

   $\beta$ enters through the target $1/(2\beta)$ instead. Larger $\beta$ gives a smaller target margin under this score convention. That changes the fitted relative preference, but does not enforce a hard global KL constraint.

   </details>
