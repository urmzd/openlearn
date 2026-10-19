# Learning-rate schedules (cosine, WSD, Noam) and gradient clipping

## Overview

| | |
|---|---|
| **Module** | `M10.4` · build · Python · Pass 2 · 2 to 3 h |
| **Solution** | [`course/solutions/python/tinyllm/optim/schedule.py`](../../../../course/solutions/python/tinyllm/optim/schedule.py): `cosine_with_warmup`, `wsd`, `noam`, `clip_grad_norm_` |
| **Contract** | [`course/contracts/py/tinyllm/optim/schedule.pyi`](../../../../course/contracts/py/tinyllm/optim/schedule.pyi) · the training spec's `schedule` and `grad_clip` fields: [`formats/train-spec.schema.json`](../../../../course/contracts/formats/train-spec.schema.json) |
| **Needs** | no code dependency. Reading: `M00.3` (sequences and the geometric sum), `M10.2` (the optimizer whose `lr` a schedule sets) |
| **Used by** | `L0.5` sets `opt.lr` from a schedule and clips before every step · later `L3.6` (clipping for recurrent networks), `L5.5` (Noam), `C1` (WSD) · later: `L11.1`, `L6.1` |
| **Optional depth** | Loshchilov and Hutter, "SGDR: Stochastic Gradient Descent with Warm Restarts" (2017); Vaswani et al., "Attention Is All You Need" (2017), section 5.3; Hägele et al., "Scaling Laws and Compute-Optimal Training Beyond Fixed Training Durations" (2024); Pascanu, Mikolov, and Bengio, "On the difficulty of training recurrent neural networks" (2013), section 3.2 |

## Key Takeaways

- A **schedule** is a pure function from the step number to a learning rate, so a resumed run gets the same rate from the step count alone.
- **Warmup** ramps the rate linearly from 0 while Adam's second moment is still unreliable; **cosine decay** then lowers it smoothly to a floor at a fixed `total`.
- **Warmup-stable-decay** holds the peak rate and decays only at the end, so one run can be stopped and annealed at any length.
- The **Noam** schedule rises linearly and decays like $1/\sqrt{t}$, peaking at $(d_{\text{model}} \cdot \text{warmup})^{-1/2}$.
- **Global-norm clipping** rescales all gradients by one factor when their joint norm exceeds a limit: it caps the step size and keeps the direction.

## How to read this chapter

Sections 1 to 3 derive the three schedules and global-norm clipping and evaluate each by hand. Section 4 fixes the interface. Section 5 walks through the solution function by function, with the phase boundaries and edge cases that decide each comparison. Try to predict each branch from section 2 before reading it.

---

## 1. Why now

With `M10.3` the optimizer takes steps of about $\eta$ per coordinate, but it takes them with the same $\eta$ from the first step to the last. That fails at both ends. At step 1, Adam's second moment has seen one gradient, so its normalization is noisy, and a full-size step from random initial weights can push a Transformer into a region it never recovers from (the loss jumps and stays high). At the end, a constant rate keeps the weights bouncing around the minimum at a distance proportional to $\eta$, and the loss stops falling. The fix is a **schedule**: start small, run at the peak, end small. A second failure is a single bad batch: one gradient a thousand times larger than usual, common in recurrent networks (`L3.6`), which a normalized optimizer still follows for a step and momentum for many more. **Gradient clipping** bounds it. `L0.5` builds the train loop next; it calls a schedule before every step and clips before every update, and the `C1` training spec names both (`schedule.name = "wsd"`, `grad_clip = 1.0`).

## 2. Principles

| Symbol | Meaning | Type / shape |
|---|---|---|
| $t$ | the step: the number of optimizer updates already taken, 0 for the first | `int`, `step` |
| $\eta(t)$ | the learning rate used for update number $t + 1$ | `float` |
| $\eta_{\max}, \eta_{\min}$ | the peak rate and the floor | `float`, `lr_max`, `lr_min` |
| $W$ | warmup length in steps | `int`, `warmup` |
| $T$ | the step at which the cosine reaches the floor | `int`, `total` |
| $S, D$ | WSD's stable and decay lengths | `int`, `stable`, `decay` |
| $p$ | progress through a phase, in $[0, 1]$ | `float` |
| $d_{\text{model}}$ | the Transformer's model width (Noam) | `int` |
| $g^{(i)}$ | the gradient of parameter tensor $i$ | ndarray, `p.grad` |
| $\lVert g \rVert$ | the global norm, $\sqrt{\sum_i \sum_k (g^{(i)}_k)^2}$ | `float` |
| $c$ | the clipping limit | `float`, `max_norm` |

### 2.1 A schedule is a function of the step

The train loop (`L0.5`) does `opt.lr = schedule(t, ...)` and then `opt.step()`, for $t = 0, 1, 2, \dots$ Because the rate depends only on $t$ and the configuration, a run resumed from a checkpoint at step 5000 recomputes exactly the rate it would have used: no scheduler object has to be saved. The step is the number of updates **already taken**, the convention of HF's `LambdaLR` schedulers, so the values equal HF's when read the same way. Every warmup ramps linearly from 0: $\eta(t) = \eta_{\max}\, t / W$ for $t < W$. Update 1 therefore runs at rate 0 and changes nothing except the optimizer's moments, which is harmless and keeps the formulas exact at the phase boundaries.

### 2.2 Cosine decay with warmup

After warmup, the rate follows half a period of a cosine from $\eta_{\max}$ down to $\eta_{\min}$:

$$\eta(t) = \eta_{\min} + (\eta_{\max} - \eta_{\min})\, \frac{1 + \cos(\pi p)}{2}, \qquad p = \frac{t - W}{T - W}, \qquad W \le t \le T,$$

and $\eta(t) = \eta_{\min}$ for $t \ge T$. At $p = 0$, $\cos 0 = 1$ gives $\eta_{\max}$; at $p = 1$, $\cos \pi = -1$ gives $\eta_{\min}$; at $p = 1/2$ the rate is halfway. The curve is flat at both ends (its derivative in $p$ is $-\frac{\pi}{2}(\eta_{\max} - \eta_{\min}) \sin(\pi p)$, zero at $p = 0$ and $p = 1$), so the peak lasts a while and the final steps change the weights very little. The weakness is $T$: it must be fixed before training starts, and stopping early leaves the rate high.

### 2.3 Warmup, stable, decay

WSD (warmup-stable-decay, used by MiniCPM and many recent models) splits the run into three phases: the linear warmup over $W$ steps, a **stable** phase at $\eta_{\max}$ for $S$ steps, and a short **decay** over the last $D$ steps, here a straight line:

$$\eta(t) = \eta_{\max} - (\eta_{\max} - \eta_{\min})\, p, \qquad p = \frac{t - W - S}{D}, \qquad W + S \le t < W + S + D,$$

and $\eta_{\min}$ from $t = W + S + D$ on. The stable phase can be extended at will, and a decay can branch off any stable checkpoint, so one long run yields models at many lengths. Most of the loss improvement of a WSD run arrives during the short decay, which `C1` lets you see in its loss curve. (HF's `get_wsd_schedule` with a floor starts its warmup at the floor instead of 0; this one starts at 0 like the cosine, so the two agree from the end of warmup on.)

### 2.4 The Noam schedule

The 2017 Transformer used

$$\eta(t) = d_{\text{model}}^{-1/2} \cdot \min\!\left(t^{-1/2},\ t \cdot W^{-3/2}\right).$$

For $t < W$ the second term is smaller (because $t \cdot W^{-3/2} < t^{-1/2}$ exactly when $t^{3/2} < W^{3/2}$), so the rate grows linearly; for $t > W$ it decays like $1/\sqrt{t}$. The two meet at $t = W$, where $\eta = d_{\text{model}}^{-1/2} W^{-1/2} = (d_{\text{model}} W)^{-1/2}$, the peak. At $t = 0$ the formula reads $\min(\infty, 0) = 0$, and that is the value to return: Python raises `ZeroDivisionError` on `0 ** -0.5`. The factor $d_{\text{model}}^{-1/2}$ ties the peak to the width: wider models get smaller rates. `L5.5` trains with $d_{\text{model}} = 512$, $W = 4000$: peak $\approx 6.99 \times 10^{-4}$.

### 2.5 Clipping by the global norm

Treat all gradients of the model as one long vector and take its Euclidean length, the **global norm** $\lVert g \rVert$. If it exceeds the limit $c$, scale every gradient by the same factor:

$$g^{(i)} \leftarrow g^{(i)} \cdot \min\!\left(1,\ \frac{c}{\lVert g \rVert + 10^{-6}}\right).$$

One factor for all tensors keeps the direction of the full gradient and only shortens it, so the clipped step is still a descent direction; clipping each tensor separately would change the direction. The $10^{-6}$ is torch's guard against dividing by a zero norm. Clipping never scales up: below the limit, the gradients are untouched. The function returns the norm **before** clipping, which the train loop logs: a norm that keeps hitting the limit is the first sign of an unstable run. If the norm is infinite or NaN (an overflow, common under fp16 in `L11.1`), clipping cannot repair it, so the gradients are left alone and the caller skips the update. With Adam (`M10.3`), clipping matters less for the step size, since Adam normalizes per coordinate, but it still bounds what one outlier batch writes into $m$ and $v$.

## 3. Worked example by hand

**Cosine**, $W = 2$, $T = 6$, $\eta_{\max} = 1$, $\eta_{\min} = 0.1$:

| $t$ | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| phase | warmup | warmup | $p = 0$ | $p = 1/4$ | $p = 1/2$ | $p = 3/4$ | $p = 1$ | after |
| $\eta(t)$ | 0 | 0.5 | 1 | $0.1 + 0.9 \cdot \frac{1 + \cos(\pi/4)}{2} \approx 0.86820$ | 0.55 | $0.1 + 0.9 \cdot \frac{1 - \cos(\pi/4)}{2} \approx 0.23180$ | 0.1 | 0.1 |

This is the first case in section 4's table.

**WSD**, $W = 2$, $S = 2$, $D = 4$, $\eta_{\max} = 1$, $\eta_{\min} = 0.2$: steps 0 and 1 ramp (0, 0.5); steps 2 and 3 are stable (1, 1); the decay runs from step 4 ($p = 0$: 1) through steps 5, 6, 7 ($p = 1/4, 1/2, 3/4$: 0.8, 0.6, 0.4); step 8 and after are at the floor, 0.2.

**Noam**, $d_{\text{model}} = 4$, $W = 4$, so $d^{-1/2} = 1/2$ and $W^{-3/2} = 1/8$: $\eta(0) = 0$; $\eta(1) = \frac12 \min(1, \frac18) = \frac1{16}$; $\eta(2) = \frac12 \min(0.707, \frac14) = \frac18$; $\eta(4) = \frac12 \min(\frac12, \frac12) = \frac14$, the peak $(4 \cdot 4)^{-1/2}$; $\eta(16) = \frac12 \cdot \frac14 = \frac18$, half the peak at four times the step.

**Clipping**: two tensors with gradients $[3, 4]$ and $[12]$. $\lVert g \rVert = \sqrt{9 + 16 + 144} = \sqrt{169} = 13$. With $c = 6.5$ the factor is $6.5 / 13.000001 \approx 0.49999996$, so the gradients become about $[1.5, 2]$ and $[6]$, and the call returns 13.

## 4. The interface

```python
def cosine_with_warmup(step: int, warmup: int, total: int, lr_max: float, lr_min: float) -> float
def wsd(step: int, warmup: int, stable: int, decay: int, lr_max: float, lr_min: float) -> float
def noam(step: int, d_model: int, warmup: int) -> float
def clip_grad_norm_(params, max_norm: float) -> float   # in place; returns the norm before clipping
```

`params` is any iterable of objects with a `grad` attribute (an ndarray or `None`), the same parameters `M10.3` steps. The schedules raise `ValueError` for a negative step, a warmup longer than `total`, a zero decay length, or `lr_min` outside $[0, \eta_{\max}]$; `clip_grad_norm_` raises it for `max_norm <= 0`. The trailing underscore follows torch: the function changes its argument in place.

### What a correct solution must handle

| Case | Kind of check | Checks | Why it matters downstream |
|---|---|---|---|
| the hand-worked cosine | unit | section 3's cosine row, steps 0 to 7 | everyone agrees on the step convention |
| the hand-worked WSD | unit | section 3's WSD values, steps 0 to 9 | the `C1` schedule |
| the hand-worked Noam | unit | section 3's Noam values | the `L5.5` schedule |
| the hand-worked clip | unit | norm 13 clipped to 6.5 | the train loop's clip |
| cosine against HF | golden | HF's cosine schedules (with and without a floor and warmup) at every step | comparability with HF-trained baselines |
| WSD against HF | golden | HF's linear WSD, including steps past the end | same |
| clipping against torch | golden | `torch.nn.utils.clip_grad_norm_` on four gradient sets, including `None` and float32 | same clip as every reference run |
| schedules hit their corners | property | over random configurations: peak exactly at the end of warmup, never above it, never rising after it, never below the floor, the floor from the end on | every phase-boundary off-by-one |
| cosine after `total` stays at the floor | boundary | steps past `total` | runs that train a little longer than planned |
| Noam's peak and inverse-square-root decay | property | peak $(d W)^{-1/2}$ at $t = W$; $\eta\sqrt{t}$ constant after it; linear before | the 2017 recipe |
| Noam at step 0 is 0 | boundary | `noam(0, ...) == 0.0` with no exception | the first update |
| clipping below the limit is a no-op | boundary | small gradients unchanged bit for bit | clipping only shrinks |
| clipping is global and in place | property | one factor for all tensors; the norm after is `max_norm`; the arrays are the same objects | the direction is kept; the optimizer sees the change |
| a non-finite norm leaves the gradients alone | boundary | an `inf` gradient returns `inf` and changes nothing | `L11.1` skips overflowed steps |
| missing gradients are skipped | boundary | `grad = None` is skipped and stays `None`; no grads gives 0 | unused parameters |
| bad arguments | boundary | impossible configurations raise `ValueError` | errors at the first step, not a rising "decay" |

## 5. The solution, step by step

### From first principles

1. **Pure functions of the step.** A resumed run must get the same rate from the step count alone (section 2.1), so each schedule is a function of `step` and its configuration, with no hidden state to checkpoint.
2. **Fix the step convention, then the boundaries follow.** `step` is the number of updates already taken. Warmup is `lr_max * step / warmup` for `step < warmup`, so step 0 gives 0 and step `warmup` is the first step of the next phase, at exactly `lr_max`.
3. **Phases as a chain of half-open intervals.** Testing `step < warmup`, then `step < warmup + stable`, then `step < warmup + stable + decay`, each `<`, assigns every step to exactly one phase, and the value at each boundary is the start of the next phase. Everything past the end is the floor, so overrunning the plan never raises the rate.
4. **Progress is measured within the phase.** Cosine progress is $(t - W)/(T - W)$ and WSD's is $(t - W - S)/D$, each running from 0 to 1 across its own phase only.
5. **Noam needs its own step 0.** $\min(t^{-1/2}, tW^{-3/2})$ at $t = 0$ is $\min(\infty, 0) = 0$ mathematically, but Python cannot evaluate `0 ** -0.5`; return 0 explicitly.
6. **Clip one vector, not many.** The global norm is the length of all gradients concatenated; computing it in float64 avoids overflow and rounding for large models, and one scale factor for every tensor preserves the direction.
7. **Only shrink, and never repair an overflow.** Scale only when the factor is below 1; if the norm is not finite, return it unchanged so the caller can skip the step.

### The code, annotated

```python
def _check_lrs(lr_max: float, lr_min: float) -> None:
    if not 0.0 <= lr_min <= lr_max:                  # a floor above the peak would "decay" upward
        raise ValueError(...)


def cosine_with_warmup(step, warmup, total, lr_max, lr_min) -> float:
    if step < 0 or warmup < 0 or total < 1 or warmup > total:
        raise ValueError(...)
    _check_lrs(lr_max, lr_min)
    if step < warmup:
        return lr_max * step / warmup                # 0 at step 0; never divides by 0 here
    if step >= total:
        return lr_min                                # past the plan: stay at the floor
    progress = (step - warmup) / (total - warmup)    # 0 at the end of warmup, < 1 before total
    return lr_min + (lr_max - lr_min) * 0.5 * (1.0 + math.cos(math.pi * progress))
```

`step < warmup` is only reached when `warmup > 0`, so the division is safe, and `total - warmup > 0` whenever the cosine branch runs (`step < total` and `step >= warmup`). Trace section 3 with $W = 2$, $T = 6$: step 1 gives $1 \cdot 1/2 = 0.5$; step 3 has progress $1/4$ and gives $0.1 + 0.9 \cdot 0.5(1 + \cos(\pi/4)) \approx 0.86820$; step 6 hits `step >= total` and returns 0.1.

```python
def wsd(step, warmup, stable, decay, lr_max, lr_min) -> float:
    if step < 0 or warmup < 0 or stable < 0 or decay < 1:
        raise ValueError(...)
    _check_lrs(lr_max, lr_min)
    if step < warmup:
        return lr_max * step / warmup
    if step < warmup + stable:
        return lr_max                                # the stable phase starts after warmup
    if step < warmup + stable + decay:
        progress = (step - warmup - stable) / decay  # 0 at the first decay step
        return lr_min + (lr_max - lr_min) * (1.0 - progress)   # = lr_max - (lr_max - lr_min) p
    return lr_min


def noam(step, d_model, warmup) -> float:
    if step < 0 or d_model < 1 or warmup < 1:
        raise ValueError(...)
    if step == 0:
        return 0.0  # min(inf, 0); Python raises on 0 ** -0.5
    return d_model**-0.5 * min(step**-0.5, step * warmup**-1.5)
```

For WSD with $W = 2$, $S = 2$, $D = 4$: steps 2 and 3 return the peak, step 4 starts the decay at progress 0 (still 1), steps 5 to 7 give 0.8, 0.6, 0.4, and step 8 falls through to the floor. `decay >= 1` is required because a zero-length decay would jump from peak to floor and the progress would divide by 0. For Noam with $d = 4$, $W = 4$: step 4 gives $\frac12 \min(\frac12, 4 \cdot \frac18) = \frac14$, both branches of the `min` equal, which is why the peak is exactly at $t = W$.

```python
CLIP_EPS = 1e-6  # torch's guard against dividing by a zero norm


def clip_grad_norm_(params, max_norm: float) -> float:
    if not max_norm > 0.0:
        raise ValueError(f"max_norm must be > 0, got {max_norm}")
    grads = [p.grad for p in params if p.grad is not None]   # skip unused parameters
    if not grads:
        return 0.0
    # one norm over every tensor, accumulated in float64
    total = math.sqrt(sum(float(np.sum(np.square(g, dtype=np.float64))) for g in grads))
    if not math.isfinite(total):
        return total                       # inf or nan: leave grads, let the caller skip
    coef = max_norm / (total + CLIP_EPS)
    if coef < 1.0:                         # only ever shrink
        for g in grads:
            g *= coef                      # in place: the optimizer reads these arrays
    return total                           # the norm before clipping, for logging
```

Trace section 3: gradients $[3, 4]$ and $[12]$ give $\sqrt{9 + 16 + 144} = 13$; `coef` $= 6.5/13.000001 < 1$, so both arrays are scaled by about $0.49999996$ and the function returns 13. `g *= coef` keeps each array's identity and dtype (a float32 gradient stays float32), which is what makes the change visible to the optimizer. `np.square(g, dtype=np.float64)` squares in float64 so a float32 gradient with entries near $10^{20}$ does not overflow to `inf` before the sum.

Full solution: [`schedule.py`](../../../../course/solutions/python/tinyllm/optim/schedule.py).

### Why this solution

- **Cost.** Each schedule is $O(1)$ per step. Clipping is one pass over all gradients to compute the norm and, when it clips, one more to scale; memory is $O(1)$ beyond a float64 square per tensor.
- **Functions over scheduler objects.** torch's `lr_scheduler` classes hold state that must be saved and restored with the checkpoint. A pure function of `step` needs only the step count, which the optimizer state already stores; the cost is that composing phases is done with `if` chains rather than by stacking scheduler objects.
- **HF's step convention.** Reading the rate for update $t + 1$ from $t$ updates already taken matches HF's `LambdaLR` schedules value for value, so baselines trained with HF are comparable. The price is one update at rate 0 at the very start, which changes nothing but Adam's moments.
- **Global over per-tensor clipping.** Per-tensor clipping is simpler to fuse but changes the update direction; global clipping keeps the direction and only shortens the step, so the clipped update is still a descent direction.
- **Skipping over repairing non-finite norms.** Clipping an infinite norm gives a factor of 0 and silently zeroes the update; returning the infinite norm lets the loop (or `L11.1`'s grad scaler) detect the overflow and skip the step.

### Explain it back

**1. Why does warmup return `lr_max * step / warmup` rather than `lr_max * (step + 1) / warmup`?**

<details><summary>Answer</summary>

`step` counts the updates already taken, so the peak should arrive exactly at `step == warmup`, the first step of the next phase. With `step + 1` the ramp reaches the peak one step early, and every value of the run is shifted by one relative to HF's schedules.

</details>

**2. Why is cosine progress divided by `total - warmup` and not `total`?**

<details><summary>Answer</summary>

The cosine phase runs from step `warmup` to step `total`, a length of `total - warmup`. Dividing by `total` makes progress reach only `(total - warmup)/total < 1` at the end, so the rate at `total` is still above the floor and then jumps down to `lr_min`.

</details>

**3. Why does `noam` special-case step 0?**

<details><summary>Answer</summary>

The formula $\min(t^{-1/2}, tW^{-3/2})$ at $t = 0$ is $\min(\infty, 0) = 0$, but Python evaluates `0 ** -0.5` by raising `ZeroDivisionError`. Returning 0 explicitly gives the mathematically correct value for the first update without the exception.

</details>

**4. Why must clipping use one factor for all tensors?**

<details><summary>Answer</summary>

Scaling every tensor by the same factor multiplies the full gradient vector by a positive scalar, which keeps its direction and only shortens it. Clipping each tensor by its own norm rescales the components differently, so the combined update points in a different direction; small layers would never be clipped while large ones always would.

</details>

**5. What would happen if `clip_grad_norm_` clipped when the norm is infinite?**

<details><summary>Answer</summary>

`max_norm / inf` is 0, so every gradient would be multiplied by 0: the update silently does nothing (or, with a NaN norm, every gradient becomes NaN). Returning the non-finite norm without touching the gradients lets the caller see the overflow and skip the step.

</details>

## 6. Pitfalls

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| decaying to 0 and ignoring `lr_min` | the last part of the run does nothing; WSD and cosine runs end at different floors | both decays are written `lr_min + (lr_max - lr_min) * ...` |
| measuring cosine progress over `total` instead of `total - warmup` | the floor is never reached; the rate at `total` is above `lr_min` | `progress = (step - warmup) / (total - warmup)`; step 4 |
| warmup from `step + 1` | the rate reaches the peak one step early and every value differs from HF | `lr_max * step / warmup`; step 2 |
| letting the cosine run past `total` | the rate climbs back toward `lr_max` after the planned end | `if step >= total: return lr_min`; step 3 |
| counting WSD's stable phase from step 0 | the decay starts `warmup` steps early | the boundaries are `warmup + stable` and `warmup + stable + decay`; step 3 |
| evaluating Noam's formula at step 0 | `ZeroDivisionError` on the first update | `if step == 0: return 0.0`; step 5 |
| clipping each tensor by its own norm | the update direction changes; small layers are never clipped while large ones always are | one `total` over all gradients and one `coef`; step 6 |
| scaling small gradients up to `max_norm` | every step has the same length; early training diverges | scales only `if coef < 1.0`; step 7 |
| clipping an infinite norm | `max_norm / inf = 0` zeroes the update and the overflow goes unnoticed | returns early `if not math.isfinite(total)`; step 7 |

## 7. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `M00.3` | sequences and the geometric sum; the warmup is an arithmetic sequence (reading) |
| Back | `M10.2` | the optimizer protocol; a schedule writes `opt.lr` (reading) |
| Forward | `L0.5` | `train_step(..., clip=...)` clips, then the loop sets `opt.lr` from the spec's schedule and steps |
| Forward | `M10.3` | AdamW reads `opt.lr` on every step, and its decay scales with it |
| Forward | `L3.6` | clipping keeps truncated backpropagation through time stable |
| Forward | `L5.5` | the Noam schedule for the 2017 Transformer |
| Forward | `C1` | WSD with `decay_frac` and `grad_clip = 1.0` from `formats/train-spec.schema.json` |
| Forward | `L6.1` | sets its learning rate from these schedules |
| Forward | `L11.1` | clips before every update and skips steps whose norm is not finite |

## Going further

| This solution | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| schedule functions | `torch.optim.lr_scheduler` (`LambdaLR`, `SequentialLR`, `CosineAnnealingWarmRestarts`) | stateful schedulers that compose phases and save their own state | `torch/optim/lr_scheduler.py` |
| `wsd` | HF `get_wsd_schedule` | cosine and 1-sqrt decay shapes, warmup shapes | `transformers/optimization.py` |
| fixed schedules | schedule-free AdamW | averages the iterates instead of decaying the rate, so no `total` is needed | Defazio et al., "The Road Less Scheduled" (2024) |
| `clip_grad_norm_` | torch `foreach` norms, FSDP `clip_grad_norm_` | one fused kernel per dtype; a norm reduced across data-parallel shards | `torch/nn/utils/clip_grad.py`; `torch/distributed/fsdp` |
| skip on a non-finite norm | AMP `GradScaler` | lowers the loss scale and skips the step when gradients overflow; the course's `L11.1` | `torch/amp/grad_scaler.py` |
