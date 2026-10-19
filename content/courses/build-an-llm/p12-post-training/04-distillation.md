# Distillation with forward and reverse KL

## Overview

| | |
|---|---|
| **Module** | `L12.4` · optional build · Python · Pass 10 |
| **Solution** | [`distill.py`](../../../../course/solutions/python/tinyllm/post/distill.py): `forward_kl`, `reverse_kl` (and the shared `_logsoftmax`) |
| **Contract** | [`distill.pyi`](../../../../course/contracts/py/tinyllm/post/distill.pyi) |
| **Needs** | `M11.1` entropy and KL |
| **Used by** | `C2` alternative training recipe |
| **Optional depth** | On-policy sequence-level distillation |

## Key Takeaways

- Forward KL penalizes teacher-supported modes the student misses.
- Reverse KL emphasizes modes the student already selects.
- Temperature changes both target entropy and gradient scale.
- On-policy distillation evaluates the teacher on student-generated prefixes.

## How to read this chapter

Sections 1 to 3 define both divergences and compute them by hand. Section 4 fixes the interface. Section 5 walks through `distill.py`, derives why it works in log space and multiplies by `T²`, and explains when to pick each direction. Try to predict each step of the solution before reading it.

## 1. Why now

The agent deployment can use a smaller student if it preserves useful behavior from a larger teacher. Token-level distillation gives a measurable alternative to preference-based training.

## 2. Principles

For teacher distribution `p` and student distribution `q`, forward KL is `D_KL(p||q)=Σ p_i log(p_i/q_i)`. Reverse KL is `D_KL(q||p)=Σ q_i log(q_i/p_i)`. Both are nonnegative and zero when the distributions agree. At temperature `T`, softmax uses logits divided by `T`; the common `T²` multiplier compensates for gradient scaling when training.

| Symbol | Meaning |
|---|---|
| `p` | teacher probabilities |
| `q` | student probabilities |
| `T` | distillation temperature |

## 3. Worked example by hand

For `p=[0.8,0.2]` and `q=[0.5,0.5]`, forward KL is `0.8 log(1.6)+0.2 log(0.4) ≈ 0.193`. Reverse KL is `0.5 log(0.625)+0.5 log(2.5) ≈ 0.223`. The two directions disagree on the same pair, and both become zero for equal distributions.

## 4. The interface

`forward_kl(student_logits, teacher_logits, temperature=1.0)` and `reverse_kl(...)` take logits of the same shape, `[..., V]` with the vocabulary last, and return the mean divergence over all leading positions, multiplied by `T²`, as a `float`. In on-policy distillation the caller samples prefixes from the student (`L8.1`), runs the teacher on those same prefixes, and passes both models' logits at each position.

### What a correct solution must handle

| Case | Why it exists | Expected result |
|---|---|---|
| the hand forward KL | Pins forward KL arithmetic and direction | `0.19274...` for the worked pair |
| reverse KL on the same pair | Makes divergence direction explicit | `0.22314...`, different from forward KL; zero for identical inputs |
| extreme logits | Real logits can be in the hundreds | `[1000, -1000]` against itself gives a finite 0, not NaN |

## 5. The solution, step by step

### From first principles

1. **Both divergences need log probabilities, not probabilities.** `log(p_i/q_i) = log p_i - log q_i`. Computing `softmax` and then `log` underflows a tiny probability to 0 and returns `-inf`; computing the log-softmax directly keeps every term finite.
2. **Log-softmax is stable after subtracting the max.** `log softmax(y)_i = y_i - m - log Σ_j exp(y_j - m)` for any `m`; choosing `m = max_j y_j` makes the largest exponent `exp(0) = 1`, so the sum cannot overflow and is at least 1.
3. **The direction decides whose probabilities weight the log ratio.** Forward KL weights by the teacher `p`: wherever the teacher puts mass and the student does not, `log(p/q)` is large, so the student is pushed to cover every teacher mode (mass-covering). Reverse KL weights by the student `q`: regions the student ignores cost nothing, so it can settle on one teacher mode (mode-seeking).
4. **Temperature applies to both models before the divergence.** Dividing both sets of logits by `T > 1` flattens both distributions, exposing the teacher's ranking of wrong answers ("dark knowledge").
5. **Temperature shrinks the gradient, so multiply by `T²`.** The gradient of forward KL with respect to the student's logits is `(q - p)/T`, and for large `T` the difference `q - p` itself shrinks like `1/T`. The gradient therefore scales like `1/T²`; multiplying the loss by `T²` keeps its magnitude comparable across temperatures, so the learning rate does not need retuning.

### The code, annotated

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

On `[1000, -1000]`, `_logsoftmax` gives `[0, -2000]`, so `p = [1, e^-2000] = [1, 0]` and every term is finite: the divergence of the input against itself is exactly 0.

Full solution: [`distill.py`](../../../../course/solutions/python/tinyllm/post/distill.py)

### Why this solution

- **Cost.** `O(positions × V)` for each direction, dominated by the exponentials. The teacher forward pass is the real cost; for off-policy distillation its logits can be cached, for on-policy they must be recomputed on each batch of student samples.
- **Log-space versus probability-space.** Computing `p * log(p/q)` from probabilities fails exactly when distillation matters most: peaked teachers whose minority probabilities underflow. Log-softmax keeps the arithmetic exact over the full range of logits.
- **Forward versus reverse.** Forward KL is what classic logit distillation uses and is the right default when the student should hedge like the teacher. Reverse KL suits a small student that cannot represent every teacher mode and should commit to one, which is also what on-policy distillation optimizes when the teacher scores the student's own samples.
- **The `T²` factor.** It changes no minimizer, only the gradient scale. Leaving it out silently shrinks the effective learning rate by `T²`.

### Explain it back

**Why does forward KL punish a student that misses a teacher mode, while reverse KL does not?**

<details><summary>Answer</summary>

In forward KL the term for token `i` is weighted by `p_i`. If the teacher gives `i` mass and the student gives it almost none, `log(p_i/q_i)` is huge and weighted by a nonzero `p_i`. In reverse KL the same term is weighted by `q_i ≈ 0`, so the miss costs almost nothing.

</details>

**What exactly would break if `_logsoftmax` did not subtract the max?**

<details><summary>Answer</summary>

`exp(1000)` overflows to `inf`, the normalizer becomes `inf`, and the log probabilities become `-inf` or NaN (`inf - inf`). The extreme-logit case would return NaN instead of 0.

</details>

**Why is temperature applied to the teacher as well as the student?**

<details><summary>Answer</summary>

The divergence compares two distributions at the same temperature. Softening only the teacher would ask the student, at `T = 1`, to match a flatter target than the one it is evaluated against; softening both compares the two models' rankings on equal terms.

</details>

## 6. Pitfalls

| Pitfall | How the solution avoids it |
|---|---|
| Swap teacher and student in forward KL | `forward_kl` weights by `p = exp(log_t)`, the teacher's log-softmax |
| Weight reverse KL by the teacher instead of the student | `reverse_kl` weights by `q = exp(log_s)`, the student's log-softmax |

## 7. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `M11.1` | Defines the forward and reverse KL measures. |
| Forward | `C2` | May use distillation instead of DPO or GRPO, evaluates on the same held-out suite, and records teacher identity. |

## Going further

| Solution piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| token-level KL | on-policy sequence distillation | Transfers behavior on student-generated prefixes | Hugging Face TRL distillation recipes |
| temperature scaling | teacher/student calibration | Changes target entropy and gradient magnitude | Hinton et al., Distilling the Knowledge in a Neural Network |
