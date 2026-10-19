# Distillation with forward and reverse KL

A deployment can often use a smaller student model, provided it preserves the useful behavior of a larger teacher. Distillation trains the student to match the teacher's output distribution token by token, which gives a measurable alternative to the preference- and reward-based training of the previous sections. The objective is a KL divergence between the two distributions, and the one real design decision is which direction of the KL to use: the two directions want different things from a student that cannot represent everything the teacher does.

This section defines both divergences, computes them by hand, and implements them in log space with temperature scaling. It builds on [Entropy, cross-entropy, KL, JS, and the k3 estimator](../../mathematics-for-computing/12-information-theory/01-entropy-cross-entropy-and-kl.md) in *Mathematics for Computing*, and for on-policy distillation on the samplers of [Sampling and logit processors](../09-inference/01-sampling-and-logit-processors.md). The solution is [`distill.py`](../../../../course/solutions/python/tinyllm/post/distill.py) (`forward_kl`, `reverse_kl`, and the shared `_logsoftmax`), and the contract is [`distill.pyi`](../../../../course/contracts/py/tinyllm/post/distill.pyi).

> **Learning objectives**
> - Compute forward and reverse KL between a teacher and a student by hand and say which is mass-covering and which mode-seeking.
> - Implement both divergences stably from logits with a max-subtracted log-softmax.
> - Explain why temperature is applied to both models and why the loss is multiplied by $T^2$.
> - Choose a direction for a given student and training setup.

## Two divergences

Write $p$ for the teacher's probabilities, $q$ for the student's, and $T$ for the distillation temperature. Forward KL is

$$D_{\mathrm{KL}}(p \,\|\, q) = \sum_i p_i \log \frac{p_i}{q_i},$$

and reverse KL is

$$D_{\mathrm{KL}}(q \,\|\, p) = \sum_i q_i \log \frac{q_i}{p_i}.$$

Both are nonnegative and zero when the distributions agree. At temperature $T$, the softmax uses logits divided by $T$, and the loss is commonly multiplied by $T^2$ to compensate for the gradient scaling that temperature introduces.

The direction decides whose probabilities weight the log ratio. Forward KL weights by the teacher $p$: wherever the teacher puts mass and the student does not, $\log(p/q)$ is large, so the student is pushed to cover every teacher mode (mass-covering). Reverse KL weights by the student $q$: regions the student ignores cost nothing, so it can settle on one teacher mode (mode-seeking), emphasizing the modes it already selects.

## Temperature and the $T^2$ factor

Temperature applies to both models before the divergence. Dividing both sets of logits by $T > 1$ flattens both distributions, exposing the teacher's ranking of wrong answers, the so-called dark knowledge. Temperature therefore changes the entropy of the target.

It also changes the gradient. The gradient of forward KL with respect to the student's logits is $(q - p)/T$, and for large $T$ the difference $q - p$ itself shrinks like $1/T$. The gradient therefore scales like $1/T^2$, and multiplying the loss by $T^2$ keeps its magnitude comparable across temperatures, so the learning rate does not need retuning.

## Working it by hand

For $p = [0.8, 0.2]$ and $q = [0.5, 0.5]$, forward KL is $0.8 \log 1.6 + 0.2 \log 0.4 \approx 0.193$. Reverse KL is $0.5 \log 0.625 + 0.5 \log 2.5 \approx 0.223$. The two directions disagree on the same pair, and both become zero for equal distributions.

## Designing the API

`forward_kl(student_logits, teacher_logits, temperature=1.0)` and `reverse_kl(...)` take logits of the same shape, `[..., V]` with the vocabulary last, and return the mean divergence over all leading positions, multiplied by $T^2$, as a `float`. In on-policy distillation the caller samples prefixes from the student, runs the teacher on those same prefixes, and passes both models' logits at each position.

## Building it step by step

Both divergences need log probabilities, not probabilities, because $\log(p_i/q_i) = \log p_i - \log q_i$. Computing `softmax` and then `log` underflows a tiny probability to 0 and returns `-inf`; computing the log-softmax directly keeps every term finite. Log-softmax is stable after subtracting the max: $\log \mathrm{softmax}(y)_i = y_i - m - \log \sum_j e^{y_j - m}$ for any $m$, and choosing $m = \max_j y_j$ makes the largest exponent $e^0 = 1$, so the sum cannot overflow and is at least 1.

```python
def _logsoftmax(x, temperature):
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    y = np.asarray(x, dtype=np.float64) / temperature       # temperature first, on the logits
    m = np.max(y, axis=-1, keepdims=True)                   # per position, over the vocabulary
    # y - m - log Σ exp(y - m): exact, and exp never sees a positive argument
    return y - m - np.log(np.exp(y - m).sum(axis=-1, keepdims=True))
```

Forward KL weights the log ratio by the teacher:

```python
def forward_kl(student_logits, teacher_logits, temperature=1.0):
    log_s, log_t = _logsoftmax(student_logits, temperature), _logsoftmax(teacher_logits, temperature)
    if log_s.shape != log_t.shape:
        raise ValueError("student and teacher shapes differ")
    p = np.exp(log_t)                                         # teacher probabilities are the weights
    # Σ_i p_i (log p_i - log q_i) per position, mean over positions, times T²
    return float(np.mean(np.sum(p * (log_t - log_s), axis=-1)) * temperature**2)
```

Reverse KL is the same computation with the roles of the weights and the ratio swapped:

```python
def reverse_kl(student_logits, teacher_logits, temperature=1.0):
    log_s, log_t = _logsoftmax(student_logits, temperature), _logsoftmax(teacher_logits, temperature)
    if log_s.shape != log_t.shape:
        raise ValueError("student and teacher shapes differ")
    q = np.exp(log_s)                                         # student probabilities are the weights
    return float(np.mean(np.sum(q * (log_s - log_t), axis=-1)) * temperature**2)
```

On `[1000, -1000]`, `_logsoftmax` gives `[0, -2000]`, so $p = [1, e^{-2000}] = [1, 0]$ and every term is finite: the divergence of the input against itself is exactly 0. The full solution is [`distill.py`](../../../../course/solutions/python/tinyllm/post/distill.py).

### Why this design

Each direction costs $O(\text{positions} \times V)$, dominated by the exponentials. The teacher forward pass is the real cost: for off-policy distillation its logits can be cached, while for on-policy distillation they must be recomputed on each batch of student samples.

Log space beats probability space because computing `p * log(p/q)` from probabilities fails exactly when distillation matters most: peaked teachers whose minority probabilities underflow. Log-softmax keeps the arithmetic exact over the full range of logits.

As for direction, forward KL is what classic logit distillation uses and is the right default when the student should hedge like the teacher. Reverse KL suits a small student that cannot represent every teacher mode and should commit to one, which is also what on-policy distillation optimizes when the teacher scores the student's own samples. The $T^2$ factor changes no minimizer, only the gradient scale; leaving it out silently shrinks the effective learning rate by $T^2$.

## What can go wrong

Three checks pin the implementation. The worked forward KL must be `0.19274...`, which fixes both the arithmetic and the direction. Reverse KL on the same pair must be `0.22314...`, different from forward KL, and zero for identical inputs. And extreme logits must stay finite: real logits can be in the hundreds, and `[1000, -1000]` against itself must give 0, not NaN.

The two direction bugs are easy to make and hard to notice, because both still produce a nonnegative number that goes down during training. Swapping teacher and student in forward KL silently computes reverse KL; `forward_kl` weights by `p = exp(log_t)`, the teacher's log-softmax. Weighting reverse KL by the teacher instead of the student computes forward KL; `reverse_kl` weights by `q = exp(log_s)`, the student's log-softmax.

> **Warning:** Leaving out the max subtraction in `_logsoftmax` makes `exp(1000)` overflow to `inf`, and the extreme-logit case returns NaN instead of 0.

## Where this leads

The [post-trained capstone](../capstones/02-post-trained.md) may use distillation instead of DPO or GRPO; it evaluates on the same held-out suite and records the teacher's identity. In production, on-policy sequence-level distillation transfers behavior on student-generated prefixes (see the Hugging Face TRL distillation recipes), and teacher-student calibration tunes temperature, which changes both target entropy and gradient magnitude.

### Further reading

- Hinton, Vinyals, and Dean, *Distilling the Knowledge in a Neural Network* (2015).
- Agarwal et al., *On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes* (2024).
- Hugging Face TRL, the distillation trainer recipes.

## Key terms

- **distillation**: training a student model to match a teacher model's output distribution.
- **forward KL**: $D_{\mathrm{KL}}(p \,\|\, q)$, weighted by the teacher; mass-covering.
- **reverse KL**: $D_{\mathrm{KL}}(q \,\|\, p)$, weighted by the student; mode-seeking.
- **distillation temperature**: a divisor $T$ applied to both models' logits before the softmax, flattening both distributions.
- **on-policy distillation**: distillation in which the teacher is evaluated on prefixes the student generated.

## Summary

- Forward KL penalizes teacher-supported modes the student misses.
- Reverse KL emphasizes modes the student already selects.
- Both are computed from a max-subtracted log-softmax, which stays finite for extreme logits.
- Temperature changes both target entropy and gradient scale; the $T^2$ factor restores the scale.
- On-policy distillation evaluates the teacher on student-generated prefixes.

## Review questions

1. Why does forward KL punish a student that misses a teacher mode, while reverse KL does not?

   <details><summary>Answer</summary>

   In forward KL the term for token $i$ is weighted by $p_i$. If the teacher gives $i$ mass and the student gives it almost none, $\log(p_i/q_i)$ is huge and weighted by a nonzero $p_i$. In reverse KL the same term is weighted by $q_i \approx 0$, so the miss costs almost nothing.

   </details>

2. What exactly would break if `_logsoftmax` did not subtract the max?

   <details><summary>Answer</summary>

   `exp(1000)` overflows to `inf`, the normalizer becomes `inf`, and the log probabilities become `-inf` or NaN (`inf - inf`). The extreme-logit case would return NaN instead of 0.

   </details>

3. Why is temperature applied to the teacher as well as the student?

   <details><summary>Answer</summary>

   The divergence compares two distributions at the same temperature. Softening only the teacher would ask the student, at $T = 1$, to match a flatter target than the one it is evaluated against; softening both compares the two models' rankings on equal terms.

   </details>
