# Training loop and the autograd bigram

## Overview

| | |
|---|---|
| **Module** | `L0.5` · build · Python · Pass 2 · 2 to 3 h |
| **Solution** | [`loop.py`](../../../../course/solutions/python/tinyllm/train/loop.py): `DataLoader`, `train_step`, `evaluate` · [`bigram.py`](../../../../course/solutions/python/tinyllm/lm/bigram.py), taken over from `L0.0` (its earlier version is [`history/L0.0/`](../../../../course/solutions/history/L0.0/)): `BigramLogits` (the table as a trainable `Module`), `to_lm`, and `BigramLM.sample` on PCG32 · the `train` verbs in [`cli_train.py`](../../../../course/solutions/entry/python/tinyllm/cli_train.py) |
| **Contract** | [`course/contracts/py/tinyllm/train/loop.pyi`](../../../../course/contracts/py/tinyllm/train/loop.pyi) · [`course/contracts/py/tinyllm/lm/bigram.pyi`](../../../../course/contracts/py/tinyllm/lm/bigram.pyi) |
| **Needs** | `L0.1` `no_grad` · `L0.2` `F.embedding` · `L0.3` the losses the loop trains with · `L0.4` `Module`, `Linear` · `M06.3` PCG32 · `M10.2` SGD · `M10.3` AdamW · `M10.4` `clip_grad_norm_` · `M02.2` `EMA` (the smoothed loss) · `rt.01` and `M03.1` (the C matmul of Pass 1) · reading: `L0.0` |
| **Used by** | `L10.0` the engine serves the table this module trains (inherited from `L0.0` with `bigram.py`) · `L0.6` keeps `L0.0`'s behaviour of `bigram.py` as its regression · later: `L2.2`, `L3.6`, `L6.1`, and the capstone train with this loop · later: `L5.5`, `L6.7`. Together with `L0.6`, this module closes Part 0 end to end: the autograd bigram reaches the count MLE, and the digits MLP trains through the loader |
| **Optional depth** | Karpathy, "A Recipe for Training Neural Networks" (2019, free); Goodfellow, Bengio, Courville, *Deep Learning*, ch. 8 |

## Key Takeaways

- One training step is zero the gradients, compute the loss, refuse a non-finite loss, backpropagate, clip, update: in that order, every time.
- Data order comes from the seeded PCG32 through the spec's Fisher-Yates shuffle, a fresh permutation each epoch, so two runs with one seed are bitwise identical.
- Evaluation weights every batch by its rows, runs in eval mode under `no_grad`, and always hands the model back in the mode it found it.
- The bigram is a convex problem whose minimum is the unsmoothed count model; your autograd reaches its NLL within $10^{-3}$ nats.
- `sample` now draws exactly as your Rust engine does, so a seed gives the same text in Python and Rust.

## How to read this chapter

Sections 1 to 3 build the training step and the bigram's optimum from first principles and work both by hand. Section 4 fixes the interface and lists what a correct solution must handle. Section 5 walks through `loop.py` and the new parts of `bigram.py` and explains why each choice is the right one. Try to predict each step of the solution before reading it.

The solution's command-line entry point, [`cli_train.py`](../../../../course/solutions/entry/python/tinyllm/cli_train.py), uses this module for two verbs: `train bigram --method autograd --data F --out DIR` (full-batch AdamW on every byte pair of `F`, then the same model directory as Pass 1, from the trained table) and `train mlp --data digits.npz --hidden 64 --epochs 30 --ckpt DIR` (a two-layer ReLU MLP on the UCI digits through `DataLoader`, with each pixel standardized by the training set's mean and standard deviation). `--method counts` stays the Pass 1 verb.

---

## 1. Why now

`L0.1` to `L0.4` give you every piece of a training run and no run. The digits MLP that closes Part 0 needs batches drawn in a reproducible order, a step that does the five things of a step in the right order, and an evaluation that does not train. And the tracer model the engine has served since Pass 1 is still a table of counts: it cannot be improved, only recounted. This module writes the loop every later model trains with, then retrains the tracer's own table with it. If the loop is right, gradient descent finds the count model again (the bigram problem has one minimum, and the counts are it), which makes the bigram the most precise end-to-end test of the autograd there is: the answer is known to many digits.

## 2. Principles

| Symbol | Meaning | Type / shape |
|---|---|---|
| $\theta$ | every parameter of the model | list of `Tensor` |
| $B$ | batch size; $n$ rows in the dataset | ints |
| $\mathcal{L}(\theta; \text{batch})$ | the loss `loss_fn` returns, one element | `Tensor` |
| $\eta$ | learning rate | float |
| $c$ | gradient clipping threshold, global norm (`M10.4`) | float |
| $\lVert g \rVert = \sqrt{\sum_p \sum_i g_{p,i}^2}$ | global gradient norm over all parameters | float |
| $W \in \mathbb{R}^{V \times V}$ | the bigram table, row $i$ = logits after token $i$ | `float32[V, V]` |
| $C_{ij}$, $R_i = \sum_j C_{ij}$ | bigram counts and row totals (`L0.0`) | |
| $N = T - 1$ | predictions in a text of $T$ tokens | int |
| $u$ | one uniform draw in $[0, 1)$ from PCG32 | float |

**The loader.** `DataLoader(arrays, batch_size, shuffle, rng)` holds named arrays that share their first dimension $n$ and yields dictionaries of rows. Without shuffle the order is $0, \dots, n - 1$. With shuffle, when each epoch **starts** it builds the list $0, \dots, n - 1$ and calls `rng.shuffle` on it: `spec/pcg32.md`'s Fisher-Yates (`M06.3`), going from the end and swapping $i$ with `below(i + 1)`. Each epoch advances the generator, so epoch 2 has a new permutation and a seeded run repeats exactly. `drop_last` drops a final partial batch, so every step sees $B$ rows; `len(loader)` says how many batches an epoch has. Arrays of different lengths, or a shuffle without a generator, are errors: one pairs inputs with the wrong labels, the other hides an unseeded order.

**The step.** `train_step(model, batch, loss_fn, opt, clip)`:

1. `opt.zero_grad()`. Gradients **accumulate** across `backward` calls (`L0.1`), so without this step $k$ trains on the sum of the gradients of steps $1, \dots, k$.
2. `loss = loss_fn(model, batch)`: a one-element Tensor, or a pair `(loss, {"acc": ...})` of the loss and extra metrics. More than one element is an error: `backward()` on a vector would need an upstream gradient nobody chose.
3. If the loss is not finite, raise `FloatingPointError` **before** any update: one `nan` step turns every weight into `nan` and the run is lost; raising first lets the caller skip the batch.
4. `loss.backward()`.
5. With `clip`, `clip_grad_norm_(parameters, c)` (`M10.4`) rescales all gradients together when $\lVert g \rVert > c$; report the norm **before** clipping, because that is the number that tells you training is unstable.
6. `opt.step()`, then return `{"loss": ..., "grad_norm": ...}` plus the extra metrics.
7. With `ema` (an `M02.2` `EMA`), feed the loss to `ema.update` and report `ema.value_debiased()` as `"loss_ema"`. One batch's loss is noise; the debiased EMA is the curve a training log plots, and without the correction its first steps sit near zero. A non-finite loss raised in step 3, so it never reaches the average.

**Evaluation.** `evaluate(model, loader, loss_fn)` switches the model to eval mode (dropout off, `L0.4`), runs every batch under `no_grad` (no graph, no stored activations, `L0.1`), and averages the loss and metrics **weighted by rows**: a last batch of 2 rows counts as 2 rows, not as a full batch. It puts the model back in the mode it found it, in a `finally`, so an exception in a batch cannot leave a training run with dropout silently off.

**The same model, trained.** `BigramLogits(V)` is a `Module` with one parameter, `weight` of shape `[V, V]`, starting at zeros (every next token equally likely, NLL $\ln V$). Its forward is a row lookup, `F.embedding(weight, ids)`: the logits `onehot(ids) @ W` of `L0.0` without the multiplications by zero, for ids of any shape (the trainer feeds `[B, T]` windows). Backward adds each position's $p - e_{\text{next}}$ into its row.

**Why gradient descent finds the counts.** The mean NLL of the table on a text is

$$\mathcal{L}(W) = \frac{1}{N}\sum_{i,j} C_{ij}\,\big(\log\textstyle\sum_k e^{W_{ik}} - W_{ij}\big),$$

a sum over rows of convex functions (log-sum-exp is convex, $-W_{ij}$ is linear). Setting the gradient of row $i$ to zero gives $R_i\,\mathrm{softmax}(W_i) = C_i$, so at the minimum $\mathrm{softmax}(W_i)_j = C_{ij}/R_i$: the **unsmoothed** count model. Its NLL, $-\frac{1}{N}\sum_{ij} C_{ij}\log(C_{ij}/R_i)$, is the floor no bigram can beat on that text. Pass 1's add-one model sits slightly above it; the trained table gets within $10^{-3}$ nats of the floor, never below it. Pairs that never occur push their logits toward $-\infty$ forever, so the loss approaches the floor without reaching it; AdamW (`M10.3`) with a large learning rate gets there in a few hundred full-batch steps.

**Sampling like the engine.** `BigramLM.sample` keeps its signature and switches its generator: one `PCG32(seed)` (stream 54) per call, one `uniform()` per token, and the Rust engine's draw (`L10.0`): weights $w_j = e^{z_j - \max z}$ of $z = \text{row}/\tau$ in float64, summed in id order, and the first $j$ whose running sum exceeds $u \cdot \sum_j w_j$. Same arithmetic, same order, same generator: `generate --seed S` and the engine's completion with seed `S` give the same ids. `to_lm()` turns the trained module into a `BigramLM` holding a float32 **copy** of the table, the object the CLI saves as `bigram.weight`.

## 3. Worked example by hand

**One training step.** `Linear(1, 1)` with $w = 2$, $b = 0$; the batch $x = [[1], [2]]$, $y = [[3], [5]]$; loss `mse`; SGD with $\eta = 0.1$.

1. Predictions $wx + b = [2, 4]$; residuals $[2 - 3, 4 - 5] = [-1, -1]$; loss $((-1)^2 + (-1)^2)/2 = 1$.
2. Gradients of $\frac{1}{2}\sum r_k^2$: $\partial/\partial w = \frac{2}{2}\sum r_k x_k = -1 \cdot 1 - 1 \cdot 2 = -3$, and $\partial/\partial b = \sum r_k = -2$.
3. Update: $w = 2 - 0.1 \cdot (-3) = 2.3$, $b = 0 - 0.1 \cdot (-2) = 0.2$.
4. A second step must use only the new gradients: residuals $[2.5 - 3, 4.8 - 5] = [-0.5, -0.2]$, so $\partial/\partial w = -0.5 - 0.4 = -0.9$, $\partial/\partial b = -0.7$. A loop that forgot `zero_grad` would add the first step's $-3$ and $-2$.
5. With `clip=1`, the global norm is $\sqrt{3^2 + 2^2} = \sqrt{13} = 3.606$; it is reported, and both gradients are scaled by $1/3.606$ before the update.

**The bigram, from zeros.** `BigramLogits(3)` on the text `abbacab` (ids `[0, 1, 1, 0, 2, 0, 1]`, $N = 6$ predictions):

1. Every row of zeros has softmax $[1/3, 1/3, 1/3]$, so the NLL is $\ln 3 = 1.0986$.
2. Row `a` makes three predictions (the next tokens are `b`, `c`, `b`). Each contributes $p - e_{\text{next}}$, and the mean divides by 6: $\big(3 \cdot [\tfrac13, \tfrac13, \tfrac13] - [0, 2, 1]\big)/6 = [\tfrac16, -\tfrac16, 0]$. Descent raises $W_{ab}$ and lowers $W_{aa}$; $W_{ac}$ is already right on average.
3. The floor: row `a` goes to $b$ twice and $c$ once ($2/3$, $1/3$), row `b` to $a$ and $b$ once each, row `c` to $a$ always. The NLL is $\frac{1}{6}\big(2\ln\tfrac32 + \ln 3 + 2\ln 2 + 0\big) = \frac{3\ln 3}{6} = 0.5493$, below the add-one model's $0.8351$ of `L0.0`.

These are `test_hand_example_train_step`, `test_zero_grad_every_step`, `test_clip_reports_the_norm_and_clips`, and `test_hand_example_bigram_gradient`.

## 4. The interface

```python
# python/tinyllm/train/loop.py
class DataLoader:
    def __init__(self, arrays: Mapping[str, NDArray], batch_size: int, shuffle: bool, rng, drop_last: bool = True)
    def __iter__(self) -> Iterator[dict[str, NDArray]]; def __len__(self) -> int
def train_step(model, batch, loss_fn, opt, clip: Optional[float] = None,
               ema: Optional[EMA] = None) -> dict[str, float]   # EMA from tinyllm.num.ema (M02.2)
def evaluate(model, loader, loss_fn) -> dict[str, float]          # mean loss, mean metrics, "n"

# python/tinyllm/lm/bigram.py (taken over from L0.0; BigramLM keeps its v0 API)
class BigramLogits(Module):
    def __init__(self, vocab: int = 256); def forward(self, ids) -> Tensor; def to_lm(self) -> BigramLM
```

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `test_hand_example_train_step` | unit | section 3: loss 1, gradients $-3$ and $-2$, then $w = 2.3$, $b = 0.2$ | you and the test agree on a step |
| `test_zero_grad_every_step` | unit | the second step uses $-0.9$ and $-0.7$ alone | gradients accumulate (`L0.1`) |
| `test_clip_reports_the_norm_and_clips` | unit | `grad_norm` is $\sqrt{13}$, the update uses clipped gradients | long runs stay stable (`M10.4`) |
| `test_nonfinite_loss_stops_before_the_update` | boundary | a `nan` loss raises and no weight moves | one bad batch cannot poison the run |
| `test_loss_fn_extras_and_shape` | boundary | `(loss, metrics)` pairs are reported; a vector loss is an error | accuracy beside the loss |
| `test_loss_ema_is_the_debiased_average` | unit | with $\beta = 0.9$ the losses 1 and 0.145 report `loss_ema` 1.0 and 0.55; a `nan` loss never reaches the EMA | the smoothed curve every later training log plots (`M02.2`) |
| `test_dataloader_batches_in_order` | unit | in-order batches, `drop_last`, `len` | evaluation sees every row once |
| `test_dataloader_shuffle_is_the_spec_permutation` | property | the order is the spec's Fisher-Yates, new each epoch | the Go and Rust ports shuffle the same way |
| `test_dataloader_rejects_bad_input` | boundary | unequal lengths, shuffle without a generator | silent mispairing |
| `test_evaluate_weights_by_rows_and_restores_mode` | unit | a 2-row batch counts 2 rows; eval mode, `no_grad`, mode restored | honest validation numbers (`L6.7`) |
| `test_evaluate_restores_mode_after_an_error` | boundary | an exception still restores training mode | dropout cannot stay off |
| `test_learns_xor` | learning | a 2-8-1 tanh MLP solves XOR and reaches the reference loss bar | every piece of `L0.1` to `L0.5` on one path |
| `test_learns_two_moons` | learning | minibatch AdamW with clipping reaches the bar and 95% accuracy | a real minibatch run |
| `test_same_seed_bitwise_identical` | property | same seed, identical weights and metrics; another seed differs | resuming (`L0.6`) and regression tests |
| `test_hand_example_bigram_gradient` | unit | section 3: NLL $\ln 3$, row `a`'s gradient $[1/6, -1/6, 0]$ | the bigram trains the right rows |
| `test_forward_is_a_row_gather` | unit | logits are table rows for 1-D and `[B, T]` ids; bad ids fail | the trainer feeds token windows |
| `test_autograd_bigram_reaches_the_count_mle` | property | within $10^{-3}$ nats of the floor, never below | `MS-L0` step 2 in miniature |
| `test_to_lm_serves_the_same_table` | unit | `to_lm()` gives the same logits through your C matmul, as a copy | the engine serves what you trained |
| `test_sample_draws_like_the_engine` | unit | ids equal the engine's PCG32 draw at three temperatures | Python and Rust agree on a seed |

The two learning tests compare against bars in `course/fixtures/ref-thresholds.tsv`: the reference's mean plus three standard deviations over five seeds.

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| 1. no `zero_grad` at the start of the step | the loss falls, then oscillates: every step adds all earlier gradients | `test_zero_grad_every_step` (mutant `s01`) |
| 2. evaluation that averages batches, leaves dropout on, or restores the mode only on success | validation loss biased by the last batch; noisy evaluations; training after a failed eval with dropout off | `test_evaluate_weights_by_rows_and_restores_mode` (mutants `s10`, `s12`, `s14`), `test_evaluate_restores_mode_after_an_error` (mutant `s13`) |
| 3. shuffling by sorting uniforms, or one permutation for every epoch | runs are not reproducible across languages, or every epoch sees the same order | `test_dataloader_shuffle_is_the_spec_permutation` (mutants `s07`, `s08`) |
| 4. checking the loss for `nan` after the update | the run is already lost when the error is raised | `test_nonfinite_loss_stops_before_the_update` (mutant `s05`) |
| 5. a new generator for every token, or a stream other than the engine's | Python and Rust disagree on the same seed; repeated draws of the same $u$ | `test_sample_draws_like_the_engine` (mutants `s20`, `s21`) |

## 6. Where it's used next
| Forward | `L2.2` | Registered call site uses this module. |
| Forward | `L3.6` | Registered call site uses this module. |
| Forward | `L5.5` | Registered call site uses this module. |
| Forward | `L6.1` | Registered call site uses this module. |
| Forward | `L6.7` | Registered call site uses this module. |

| Direction | Module | How it uses this |
|---|---|---|
| Back | `L0.1` | `evaluate` runs under `no_grad`; the loss is a `Tensor` |
| Back | `L0.2` | `BigramLogits.forward` is `F.embedding` |
| Back | `L0.3` | the tests (and your CLI) train on `cross_entropy`, `mse`, `bce_with_logits` |
| Back | `L0.4` | `BigramLogits` is a `Module`; the tests train `Linear` stacks |
| Back | `M06.3` | `PCG32`: the loader's shuffle and the sampler's draws |
| Back | `M10.2` | `SGD` steps the hand example |
| Back | `M10.3` | `AdamW` trains the MLPs and the bigram |
| Back | `M10.4` | `clip_grad_norm_` in `train_step` |
| Back | `M02.2` | `EMA.update` and `value_debiased` give `train_step`'s `loss_ema` |
| Back | Python reference | This model has no native library boundary. |
| Back | `M03.1` | `tl_matmul_f32` computes `BigramLM.logits` |
| Forward | `L10.0` | your Rust engine serves the table `to_lm()` produces, unchanged (`tl_arch = bigram`) |
| Forward | `L0.6` | takes over `safetensors.py`; its regression run of `L0.0`'s suite exercises your `bigram.py` too |

Later passes (`L2.2`, `L3.6`, `L6.1`, the capstone) call `train_step` and `evaluate` as they are. If you skip this module, `ol milestone MS-L0` cannot train either model.

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `DataLoader` | `torch.utils.data.DataLoader` | worker processes, pinned memory, samplers, a `generator` argument for seeded shuffles | `torch/utils/data/dataloader.py` |
| `train_step` | HF `Trainer.training_step`, Lightning `LightningModule` | mixed precision, gradient accumulation, distributed reduction, callbacks | `transformers/trainer.py` |
| non-finite guard | `torch.cuda.amp.GradScaler` | skips steps with `inf` gradients and lowers the loss scale (`L11.1`) | `torch/amp/grad_scaler.py` |
| PCG32 sampling shared with Rust | vLLM's per-request seeded sampling | a generator per request on GPU, the same draws across batch compositions | vLLM `v1/sample/sampler.py` |
