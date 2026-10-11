<!-- ol:module L3.1 -->
# Vanilla RNN with manual BPTT and truncation

## Overview

| | |
|---|---|
| **Module** | `L3.1` · build · Python · Pass 4 · 4 to 5 h, plus your graded tests (rung R3) |
| **You build** | `python/tinyllm/rnn/manual.py`: `RNNCache`, `rnn_forward`, `rnn_backward`, `tbptt_windows`, `tbptt_grads`, `gradient_flow`; and your own tests in `python/tests/l3-1-rnn/`, written first |
| **Contract** | [`course/contracts/py/tinyllm/rnn/manual.pyi`](../../../course/contracts/py/tinyllm/rnn/manual.pyi) |
| **Tests** | `course/tests/L3.1/` (what they check: section 4), a golden case from torch 2.14 in `course/fixtures/L3.1/rnn_torch.npz`; your tests are graded by mutation, threshold 0.70 plus one required fault, with a red-then-green journal |
| **Needs** | `M08.3` `matmul_vjp` (each step's backward) · `M03.4` `spectral_radius` (the diagnostic) · `L0.1` `Tensor` and `L0.2` the op library (the tests differentiate the same forward with your autograd) · reading: `M04.2` VJPs and the chain rule, `craft.03` red then green (or `--ref-deps`) |
| **Used by** | later: `L3.6` runs `cell='rnn'` as one fused autograd op over `rnn_forward` and `rnn_backward` |
| **Milestone** | `MS-L3` (the RNN language model's `rnn` row) |
| **Optional depth** | Werbos, "Backpropagation Through Time: What It Does and How to Do It" (1990); Williams and Peng, "An Efficient Gradient-Based Algorithm for On-Line Training of Recurrent Network Trajectories" (1990); Pascanu, Mikolov, Bengio, "On the difficulty of training recurrent neural networks" (2013) |

## Key Takeaways

- Backpropagation through time is reverse mode on the unrolled loop: the gradient reaching $h_t$ is what the loss sends it directly plus what step $t + 1$ sends back through $W_{hh}$ (`test_hand_example_two_steps`, `test_gradcheck_full_bptt`).
- Every step's backward is two matmul VJPs and the tanh derivative $1 - h_t^2$, read from the saved output, so the forward must keep every hidden state (`test_matches_autograd`, `test_matches_torch_rnn`).
- Truncated BPTT keeps the forward running but cuts the gradient: every $k_1$ steps, the losses since the last cut are differentiated back $k_2$ steps; with $k_2 \ge T$ it is full BPTT (`test_tbptt_windows_examples`, `test_tbptt_with_long_reach_is_full_bptt`).
- The gradient into an early state is the late one multiplied by $W_{hh}^\top$ once per step, so its norm grows or shrinks like $\rho(W_{hh})^{T-t}$: exploding and vanishing gradients, predicted by the spectral radius (`test_gradient_flow_explodes_and_vanishes`).

## How to work this chapter

```bash
ol start L3.1              # stubs manual.py; prints your test path and rung (R3)
ol tests L3.1              # the course tests
# write ONE test in python/tests/l3-1-rnn/, then:
ol tdd red L3.1            # must FAIL against your current code: records the red
# make it pass, then:
ol tdd green L3.1          # must PASS with the same test files: records the green
# repeat for each test; then:
ol check L3.1              # course tests, the red-then-green journal, the mutation grade
ol mutate L3.1             # the full grade, cached by your test files' hash
```

---

## 1. Why now

Every model so far sees a fixed window: the bigram one byte, the n-gram model (`L2.1`) $n - 1$ tokens, the neural probabilistic LM (`L2.2`) a fixed context of embeddings. Text has dependencies longer than any window you can afford: a quote opened 200 characters ago must close. A recurrent network keeps a state $h_t$ that it updates once per token, so in principle it sees the whole past. In practice it forgets, and the reason is in the gradient: to learn that a quote must close, the gradient of the closing step's loss has to travel back 200 steps through the recurrence. This module writes that backward pass by hand, so you can see where the gradient goes, why it vanishes or explodes, and why training cuts it short (truncation). `L3.2` and `L3.3` then fix the vanishing with gates, and `L3.6` trains all three as language models on the same text.

## 2. Principles

| Symbol | Meaning | Type / shape |
|---|---|---|
| $T, B, D, H$ | time steps, batch size, input size, hidden size | integers |
| $x_t$ | the input at step $t$ (a row per batch element) | `float64[B, D]` |
| $h_t$ | the hidden state after step $t$; $h_{-1} = h_0$ is the given start | `float64[B, H]` |
| $W_{xh}$, $W_{hh}$, $b_h$ | input-to-hidden, hidden-to-hidden weights, bias | `[D, H]`, `[H, H]`, `[H]` |
| $a_t = x_t W_{xh} + h_{t-1} W_{hh} + b_h$ | the pre-activation | `[B, H]` |
| $L$ | a scalar loss computed from the hidden states | real |
| $\bar h_t = \partial L/\partial h_t$ | the total gradient reaching $h_t$ | `[B, H]` |
| $g_t$ (`dh_all[t]`) | the part of $\bar h_t$ that comes directly from the loss at step $t$ | `[B, H]` |
| $\bar a_t$ | the gradient of the pre-activation | `[B, H]` |
| $\rho(W)$ | spectral radius, $\max \lvert\lambda\rvert$ over $W$'s eigenvalues (`M03.4`) | real |
| $k_1, k_2$ | truncation: cut every $k_1$ steps, backpropagate $k_2$ steps | integers |

**The recurrence.** $h_t = \tanh(a_t)$, $a_t = x_t W_{xh} + h_{t-1} W_{hh} + b_h$, for $t = 0, \dots, T - 1$. The same three parameters are used at every step: the network is one cell applied $T$ times. We use rows (one row per batch element) and right-multiplied weights, so `x @ Wxh` reads like the math. torch stores the transposes, `weight_ih_l0` $= W_{xh}^\top$ and `weight_hh_l0` $= W_{hh}^\top$, and two biases whose sum is $b_h$; the golden test maps between the two.

**BPTT is reverse mode on the unrolled loop.** Unroll the loop and you have a deep feed-forward graph whose layers share weights. Reverse mode (`M08.2`) walks it from the last node back. $h_t$ feeds two places: the loss at step $t$ (through whatever reads it, an output layer in `L3.6`), and step $t + 1$ through $a_{t+1}$. So its total gradient is the sum of both paths:

$$\bar h_t = g_t + \bar a_{t+1} W_{hh}^\top, \qquad \bar h_{T-1} = g_{T-1} + (\text{gradient from after the sequence, } \texttt{dh\_next}).$$

The second term is the **carry**: walking $t = T-1, \dots, 0$, keep it in a variable and add it to $g_t$ before anything else. Then through the tanh, with $\tanh'(a) = 1 - \tanh^2(a) = 1 - h_t^2$ (read from the saved output, so $a_t$ need not be kept):

$$\bar a_t = \bar h_t \odot (1 - h_t \odot h_t).$$

**Each step is two matmul VJPs.** $a_t$ contains $x_t W_{xh}$ and $h_{t-1} W_{hh}$, both of the form $Y = AB$, whose VJP (`M08.3`) is $\bar A = \bar Y B^\top$, $\bar B = A^\top \bar Y$. So

$$\bar x_t = \bar a_t W_{xh}^\top,\quad \bar W_{xh} \mathrel{+}= x_t^\top \bar a_t,\quad \bar h_{t-1} \text{ (the next carry)} = \bar a_t W_{hh}^\top,\quad \bar W_{hh} \mathrel{+}= h_{t-1}^\top \bar a_t,\quad \bar b_h \mathrel{+}= \textstyle\sum_{\text{rows}} \bar a_t.$$

The weight gradients **accumulate** over time (the same weight is used at every step), and $\bar W_{hh}$ pairs $\bar a_t$ with the **previous** state $h_{t-1}$, the one that was multiplied. After $t = 0$ the carry is $\partial L/\partial h_0$.

**Exploding and vanishing gradients.** If the loss only reads the last state, the carry from step $T-1$ back to step $t$ is multiplied by $\operatorname{diag}(1 - h_s^2)\,W_{hh}^\top$ once per step. Near the linear regime ($h$ small, $1 - h^2 \approx 1$) that is $W_{hh}^\top$ applied $T - 1 - t$ times, and the component along the top eigenvector grows or shrinks by $\rho(W_{hh})$ per step. So the gradient norm behaves like $\rho^{T-1-t}$: for $\rho = 1.25$ it is $1.25^{29} \approx 650$ times larger 30 steps back, for $\rho = 0.8$ about 1500 times smaller. Saturated tanh units ($|h| \to 1$) shrink it further. `gradient_flow` returns the per-step norms and $\rho$ (from `M03.4`'s QR algorithm), the diagnostic you print when training stalls. Note the transpose: walking back multiplies by $W_{hh}^\top$, and for a non-normal $W_{hh}$ the spectral norm (largest singular value) can be large while $\rho$ is small; the long-run growth is set by $\rho$.

**Truncated BPTT.** Full BPTT over a 100 000-token stream needs every hidden state in memory and a backward pass as long as the stream. Williams and Peng's TBPTT($k_1$, $k_2$) runs the forward straight through and, every $k_1$ steps, differentiates the losses of the last $k_1$ steps back through at most $k_2$ steps. The window ending at $e$ starts at $s = \max(0, e - k_2)$; the state entering it, $h_{s-1}$, is a **constant** (the cut), and only the losses of steps $[e_{\text{prev}}, e)$ enter (earlier steps of an overlapping window were already counted at their own cut). `tbptt_windows` lists the cuts, `tbptt_grads` sums the window gradients. Two checks pin the bookkeeping: with $k_2 \ge T$ every loss reaches its whole history exactly once, which is full BPTT; with $k_1 = k_2 = 1$ no gradient crosses $W_{hh}$ at all. `L3.6` uses $k_1 = k_2 = k$ with the state carried across batches (stateful TBPTT).

## 3. Worked example by hand

$D = H = B = 1$, $W_{xh} = 0.5$, $W_{hh} = 0.8$, $b_h = 0$, $h_0 = 0$, inputs $x = [1, 0]$, and the loss $L = h_1$ (the state after the second step), so $g = [0, 1]$.

**Forward.** $h_0 = \tanh(1 \cdot 0.5 + 0) = 0.462117$; $h_1 = \tanh(0 \cdot 0.5 + 0.462117 \cdot 0.8) = \tanh(0.369694) = 0.353724$.

**Backward**, $t = 1$: the carry is 0, so $\bar h_1 = g_1 = 1$. $\bar a_1 = 1 \cdot (1 - 0.353724^2) = 0.874879$. Then $\bar W_{xh} \mathrel{+}= x_1 \bar a_1 = 0$, $\bar W_{hh} \mathrel{+}= h_0 \bar a_1 = 0.462117 \times 0.874879 = 0.404297$, $\bar b_h \mathrel{+}= 0.874879$, $\bar x_1 = 0.874879 \times 0.5 = 0.437440$, and the carry becomes $\bar a_1 W_{hh} = 0.699904$.

**Backward**, $t = 0$: $\bar h_0 = g_0 + \text{carry} = 0 + 0.699904$. $\bar a_0 = 0.699904 \times (1 - 0.462117^2) = 0.550438$. $\bar W_{xh} \mathrel{+}= x_0 \bar a_0 = 0.550438$, $\bar W_{hh} \mathrel{+}= h_{-1} \bar a_0 = 0$ (the start state is 0), $\bar b_h \mathrel{+}= 0.550438$, so $\bar b_h = 1.425317$; $\bar x_0 = 0.275219$, and $\partial L/\partial h_{-1} = \bar a_0 W_{hh} = 0.440350$.

Without the carry, $\bar a_0$ would be 0 and $\bar W_{xh}$ would be 0: the first input would look useless although it decides $h_1$.

**Truncation windows.** $T = 10$, $k_1 = 4$, $k_2 = 6$: cuts at 4, 8, and 10 (the last, partial chunk), reaching back to $\max(0, 4 - 6) = 0$, $8 - 6 = 2$, and $10 - 6 = 4$: `[(0, 4), (2, 8), (4, 10)]`. The window $(2, 8)$ backpropagates only the losses of steps 4 to 7.

These are `test_hand_example_two_steps` and `test_tbptt_windows_examples`.

## 4. The interface

```python
# python/tinyllm/rnn/manual.py
class RNNCache(NamedTuple): x; h0; h; Wxh; Whh               # what the backward needs
def rnn_forward(x, h0, Wxh, Whh, bh) -> tuple[NDArray, RNNCache]      # h [T, B, H]
def rnn_backward(dh_all, cache, dh_next=None) -> dict[str, NDArray]  # keys x, h0, Wxh, Whh, bh
def tbptt_windows(T, k1, k2) -> list[tuple[int, int]]
def tbptt_grads(x, h0, Wxh, Whh, bh, dh_all, k1, k2) -> dict[str, NDArray]
def gradient_flow(cache, dh_last) -> tuple[NDArray, float]            # (norms [T], rho)
```

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `test_hand_example_two_steps` | unit | section 3, every number | you and the test agree on the carry and $1 - h^2$ |
| `test_gradcheck_full_bptt` | gradcheck | all five gradients against central differences | the backward is the derivative of the forward |
| `test_dh_next_enters_at_the_last_step` | gradcheck | a gradient arriving from after the sequence | stacking and windows (`L3.6`) |
| `test_matches_autograd` | differential | the same forward through your `Tensor` gives the same gradients | `L3.6` fuses this as one autograd op |
| `test_matches_torch_rnn` | golden | `torch.nn.RNN` outputs and gradients with transposed weights and summed biases | the convention is torch's |
| `test_tbptt_windows_examples` | unit | section 3's windows, tiling, the partial chunk, bad arguments | the trainer's schedule (`L3.6`) |
| `test_tbptt_with_long_reach_is_full_bptt` | property | $k_2 \ge T$ equals full BPTT for every $k_1$ | each loss counted exactly once |
| `test_tbptt_cuts_long_paths` | unit | $k_1 = k_2 = 1$: no gradient crosses $W_{hh}$ | what truncation gives up |
| `test_gradient_flow_explodes_and_vanishes` | property | norms follow $\rho^{T-1-t}$ along the top eigenvector; $\rho$ reported | diagnosing a stalled run |
| `test_gradient_flow_uses_whh_transpose` | unit | a non-normal $W_{hh}$: $W_{hh}^\top$ per step, $\rho = 0.5$ not the norm 2.13 | the diagnostic predicts the real gradient |
| `test_shapes_and_bad_arguments` | boundary | $T = 0$; mismatched shapes raise | errors at the call site |

### Your graded tests (rung R3)

Rung R3 gives you the interface (above) and one test; you write the rest **before** the code that makes each pass. The given test:

```python
# python/tests/l3-1-rnn/test_rnn.py
import numpy as np
from tinyllm.rnn.manual import rnn_backward, rnn_forward

def test_hand_example_two_steps():
    """Wxh = 0.5, Whh = 0.8, x = [1, 0], h0 = 0, L = h_1: dWxh = 0.550438, dWhh = 0.404297, dh0 = 0.440350."""
    h, cache = rnn_forward(np.array([[[1.0]], [[0.0]]]), [[0.0]], [[0.5]], [[0.8]], [0.0])
    g = rnn_backward(np.array([[[0.0]], [[1.0]]]), cache)
    assert np.allclose(g["Wxh"], [[0.5504375878410427]])
    assert np.allclose(g["Whh"], [[0.4042968189107551]])
    assert np.allclose(g["h0"], [[0.4403500702728342]])
```

Then, one at a time, red then green: every gradient against your own central differences (with a `dh_next`); the windows of `tbptt_windows`, including a final partial chunk; `tbptt_grads` with $k_2 \ge T$ equal to `rnn_backward`; one-step windows equal to the immediate terms only; `gradient_flow` on the non-normal $W_{hh}$ of section 4's table; and shape errors. Import only `tinyllm.rnn.manual`. `ol check L3.1` requires a red record before each green, a mutation score of at least 0.70, and the required fault killed: it is pitfall 1.

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| 1. not carrying the gradient from step $t + 1$ into $h_t$ | early inputs get zero gradient; the network only learns one-step dependencies | `test_hand_example_two_steps`, `test_gradcheck_full_bptt` (mutant `s01`) |
| 2. tanh derivative as $1 - h$, or from the wrong saved value | gradients off by a factor that grows with $\lvert h\rvert$ | `test_hand_example_two_steps` (mutant `s02`) |
| 3. pairing $\bar a_t$ with $h_t$ instead of $h_{t-1}$ in $\bar W_{hh}$, or overwriting instead of accumulating | wrong recurrent gradient; training diverges or stalls | `test_hand_example_two_steps`, `test_gradcheck_full_bptt` (mutants `s03`, `s04`) |
| 4. TBPTT bookkeeping: reaching back $k_1$ instead of $k_2$, counting a loss in two windows, dropping the last partial chunk | a gradient that is neither full nor truncated, larger than full BPTT | `test_tbptt_windows_examples`, `test_tbptt_with_long_reach_is_full_bptt` (mutants `s05`, `s06`, `s07`) |
| 5. diagnosing with $W_{hh}$ instead of $W_{hh}^\top$, or with the spectral norm | the diagnostic predicts explosion where the gradient vanishes | `test_gradient_flow_uses_whh_transpose`, `test_gradient_flow_explodes_and_vanishes` (mutants `s08`, `s12`) |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `M08.3` | `matmul_vjp` is each step's backward, twice |
| Back | `M03.4` | `spectral_radius` is the explosion predictor |
| Back | `L0.1` | the tests build the same forward on `Tensor` |
| Back | `L0.2` | `F.tanh`, `F.matmul`, `F.stack` for the autograd comparison |
| Forward | `L3.2` | the LSTM's cell state keeps the gradient from vanishing |
| Forward | `L3.6` | `cell='rnn'` wraps `rnn_forward` and `rnn_backward` as one autograd op and trains with stateful TBPTT |

If you skip this module, `ol check L3.6` stops with `L3.6 needs L3.1`: build it, or rerun with `--ref-deps`.

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `rnn_forward`/`rnn_backward` | cuDNN RNN (`cudnnRNNForward`, `cudnnRNNBackwardData`) | the input matmul for all steps as one GEMM, persistent kernels keeping $W_{hh}$ in registers | `aten/src/ATen/native/cudnn/RNN.cpp` |
| `tbptt_grads` | Keras `stateful=True`, fairseq's `detach` between chunks | stateful batching where row $b$ of each batch continues row $b$ of the previous one | `keras/layers/rnn/base_rnn.py` |
| `gradient_flow` | gradient-norm logging and clipping (`clip_grad_norm_`, `M10.4`) | clipping caps the explosion this diagnostic predicts | `torch/nn/utils/clip_grad.py` |
| the vanilla cell | linear recurrences (S4, Mamba, RWKV) | recurrences with $\rho$ held below 1 by construction, trained in parallel over time | `mamba_ssm/ops/selective_scan_interface.py` |
