# Vanilla RNN with manual BPTT and truncation

Every model so far sees a fixed window: the bigram model sees one byte, the [n-gram model](../03-statistical-lm/01-ngram-kneser-ney.md) sees $n - 1$ tokens, and [Bengio's neural probabilistic language model](../03-statistical-lm/02-bengio-nplm.md) sees a fixed context of embeddings. Text has dependencies longer than any window you can afford: a quote opened 200 characters ago must close. A recurrent network keeps a state $h_t$ that it updates once per token, so in principle it sees the whole past. In practice it forgets, and the reason is in the gradient. To learn that a quote must close, the gradient of the closing step's loss has to travel back 200 steps through the recurrence.

In this section you write that backward pass by hand, so you can see where the gradient goes, why it vanishes or explodes, and why training cuts it short (truncation). The [LSTM](02-lstm.md) and [GRU](03-gru.md) sections then fix the vanishing with gates, and the [RNN language model](06-rnn-language-model-stateful-tbptt.md) trains all three cells as language models on the same text. There, `cell='rnn'` runs as one fused autograd op over the `rnn_forward` and `rnn_backward` you build here, and it is the `rnn` row of the recurrent language-model comparison.

This section builds on [Matrix differentials, the trace trick, and closed-form VJPs](../../mathematics-for-computing/09-matrix-calculus-and-autodiff/03-matrix-differentials-and-vjps.md) from *Mathematics for Computing*, whose `matmul_vjp` is each step's backward, and on [Eigenvalues, power iteration, and the spectral radius](../../mathematics-for-computing/04-linear-algebra/04-eigenvalues-and-power-iteration.md), whose `spectral_radius` is the diagnostic. The cross-check runs the same forward through the [`Tensor`](../01-foundations/02-tensor-and-broadcasting-backward.md) and [op library](../01-foundations/03-op-library-and-gradcheck.md) of Part 0 and compares gradients with autograd. Useful background reading is [Jacobians, the multivariable chain rule, and numeric JVP/VJP](../../mathematics-for-computing/05-calculus-3/02-jacobians-and-the-chain-rule.md) and, from *Software Craftsmanship*, [TDD, unit tests, and mutation testing](../../software-craftsmanship/03-testing-mentality/01-tdd-unit-tests-and-mutation-grading.md). The solution is [`manual.py`](../../../../course/solutions/python/tinyllm/rnn/manual.py), with `RNNCache`, `rnn_forward`, `rnn_backward`, `tbptt_windows`, `tbptt_grads`, and `gradient_flow`; its contract is [`manual.pyi`](../../../../course/contracts/py/tinyllm/rnn/manual.pyi) and its model tests are [`test_rnn.py`](../../../../course/solutions/learner-tests/L3.1/python/tests/l3-1-rnn/test_rnn.py).

> **Learning objectives**
> - Derive backpropagation through time as reverse mode on the unrolled loop, including the carry from step $t + 1$.
> - Write each step's backward as two matmul VJPs and the tanh derivative $1 - h_t^2$, accumulating the shared weight gradients.
> - Implement truncated BPTT($k_1$, $k_2$) as bookkeeping on windows, and check it against full BPTT.
> - Predict exploding and vanishing gradients from the spectral radius $\rho(W_{hh})$, and explain why the diagnostic multiplies by $W_{hh}^\top$.

## The recurrence and its notation

The table below fixes the symbols used throughout the section, with their shapes.

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
| $\rho(W)$ | spectral radius, $\max \lvert\lambda\rvert$ over $W$'s eigenvalues | real |
| $k_1, k_2$ | truncation: cut every $k_1$ steps, backpropagate $k_2$ steps | integers |

The recurrence is $h_t = \tanh(a_t)$ with $a_t = x_t W_{xh} + h_{t-1} W_{hh} + b_h$, for $t = 0, \dots, T - 1$. The same three parameters are used at every step: the network is one cell applied $T$ times. We use rows (one row per batch element) and right-multiplied weights, so `x @ Wxh` reads like the math. torch stores the transposes, `weight_ih_l0` $= W_{xh}^\top$ and `weight_hh_l0` $= W_{hh}^\top$, and two biases whose sum is $b_h$; the golden test maps between the two conventions.

## BPTT is reverse mode on the unrolled loop

Unroll the loop and you have a deep feed-forward graph whose layers share weights. Reverse mode, as in [Scalar reverse mode: the Value graph](../../mathematics-for-computing/09-matrix-calculus-and-autodiff/02-scalar-reverse-mode.md), walks it from the last node back. The state $h_t$ feeds two places: the loss at step $t$ (through whatever reads it, an output layer in the [RNN language model](06-rnn-language-model-stateful-tbptt.md)), and step $t + 1$ through $a_{t+1}$. So its total gradient is the sum of both paths:

$$\bar h_t = g_t + \bar a_{t+1} W_{hh}^\top, \qquad \bar h_{T-1} = g_{T-1} + (\text{gradient from after the sequence, } \texttt{dh\_next}).$$

The second term is the **carry**. Walking $t = T-1, \dots, 0$, you keep it in a variable and add it to $g_t$ before anything else. Then you pass through the tanh, using $\tanh'(a) = 1 - \tanh^2(a) = 1 - h_t^2$, read from the saved output so that $a_t$ need not be kept:

$$\bar a_t = \bar h_t \odot (1 - h_t \odot h_t).$$

From there each step is two matmul VJPs. The pre-activation $a_t$ contains $x_t W_{xh}$ and $h_{t-1} W_{hh}$, both of the form $Y = AB$, whose VJP is $\bar A = \bar Y B^\top$, $\bar B = A^\top \bar Y$. So

$$\bar x_t = \bar a_t W_{xh}^\top,\quad \bar W_{xh} \mathrel{+}= x_t^\top \bar a_t,\quad \bar h_{t-1} \text{ (the next carry)} = \bar a_t W_{hh}^\top,\quad \bar W_{hh} \mathrel{+}= h_{t-1}^\top \bar a_t,\quad \bar b_h \mathrel{+}= \textstyle\sum_{\text{rows}} \bar a_t.$$

The weight gradients **accumulate** over time, because the same weight is used at every step, and $\bar W_{hh}$ pairs $\bar a_t$ with the **previous** state $h_{t-1}$, the one that was multiplied. After $t = 0$ the carry is $\partial L/\partial h_0$.

## Exploding and vanishing gradients

That carry is also where the trouble starts. If the loss only reads the last state, the carry from step $T-1$ back to step $t$ is multiplied by $\operatorname{diag}(1 - h_s^2)\,W_{hh}^\top$ once per step. Near the linear regime ($h$ small, $1 - h^2 \approx 1$) that is $W_{hh}^\top$ applied $T - 1 - t$ times, and the component along the top eigenvector grows or shrinks by $\rho(W_{hh})$ per step. So the gradient norm behaves like $\rho^{T-1-t}$: for $\rho = 1.25$ it is $1.25^{29} \approx 650$ times larger 30 steps back, and for $\rho = 0.8$ about 1500 times smaller. Saturated tanh units ($|h| \to 1$) shrink it further.

`gradient_flow` returns the per-step norms and $\rho$ (computed with the QR algorithm from the spectral-radius section), the diagnostic to print when training stalls. Note the transpose: walking back multiplies by $W_{hh}^\top$, and for a non-normal $W_{hh}$ the spectral norm (largest singular value) can be large while $\rho$ is small. The long-run growth is set by $\rho$.

## Truncated BPTT

Full BPTT over a 100 000-token stream needs every hidden state in memory and a backward pass as long as the stream. Williams and Peng's TBPTT($k_1$, $k_2$) runs the forward straight through and, every $k_1$ steps, differentiates the losses of the last $k_1$ steps back through at most $k_2$ steps. The window ending at $e$ starts at $s = \max(0, e - k_2)$. The state entering it, $h_{s-1}$, is a **constant** (the cut), and only the losses of steps $[e_{\text{prev}}, e)$ enter, since earlier steps of an overlapping window were already counted at their own cut. `tbptt_windows` lists the cuts and `tbptt_grads` sums the window gradients.

Two checks pin the bookkeeping. With $k_2 \ge T$ every loss reaches its whole history exactly once, which is full BPTT. With $k_1 = k_2 = 1$ no gradient crosses $W_{hh}$ at all. The [RNN language model](06-rnn-language-model-stateful-tbptt.md) uses $k_1 = k_2 = k$ with the state carried across batches (stateful TBPTT).

## Working it by hand

Take $D = H = B = 1$, $W_{xh} = 0.5$, $W_{hh} = 0.8$, $b_h = 0$, $h_0 = 0$, inputs $x = [1, 0]$, and the loss $L = h_1$ (the state after the second step), so $g = [0, 1]$.

**Forward.** $h_0 = \tanh(1 \cdot 0.5 + 0) = 0.462117$; $h_1 = \tanh(0 \cdot 0.5 + 0.462117 \cdot 0.8) = \tanh(0.369694) = 0.353724$.

**Backward at $t = 1$.** The carry is 0, so $\bar h_1 = g_1 = 1$. Then $\bar a_1 = 1 \cdot (1 - 0.353724^2) = 0.874879$. That gives $\bar W_{xh} \mathrel{+}= x_1 \bar a_1 = 0$, $\bar W_{hh} \mathrel{+}= h_0 \bar a_1 = 0.462117 \times 0.874879 = 0.404297$, $\bar b_h \mathrel{+}= 0.874879$, $\bar x_1 = 0.874879 \times 0.5 = 0.437440$, and the carry becomes $\bar a_1 W_{hh} = 0.699904$.

**Backward at $t = 0$.** $\bar h_0 = g_0 + \text{carry} = 0 + 0.699904$, so $\bar a_0 = 0.699904 \times (1 - 0.462117^2) = 0.550438$. Then $\bar W_{xh} \mathrel{+}= x_0 \bar a_0 = 0.550438$, $\bar W_{hh} \mathrel{+}= h_{-1} \bar a_0 = 0$ (the start state is 0), and $\bar b_h \mathrel{+}= 0.550438$, so $\bar b_h = 1.425317$. Finally $\bar x_0 = 0.275219$ and $\partial L/\partial h_{-1} = \bar a_0 W_{hh} = 0.440350$.

Without the carry, $\bar a_0$ would be 0 and $\bar W_{xh}$ would be 0: the first input would look useless although it decides $h_1$.

**Truncation windows.** With $T = 10$, $k_1 = 4$, $k_2 = 6$, the cuts fall at 4, 8, and 10 (the last, partial chunk), reaching back to $\max(0, 4 - 6) = 0$, $8 - 6 = 2$, and $10 - 6 = 4$: `[(0, 4), (2, 8), (4, 10)]`. The window $(2, 8)$ backpropagates only the losses of steps 4 to 7.

These two hand cases become unit tests. The model tests in [`test_rnn.py`](../../../../course/solutions/learner-tests/L3.1/python/tests/l3-1-rnn/test_rnn.py) pin them exactly: with $W_{xh} = 0.5$, $W_{hh} = 0.8$, $x = [1, 0]$, $h_0 = 0$, and $L = h_1$, they expect $\bar W_{xh} = 0.550438$, $\bar W_{hh} = 0.404297$, and $\partial L / \partial h_{-1} = 0.440350$. They add central differences with a `dh_next`, the windows with a partial chunk, $k_2 \ge T$ against `rnn_backward`, one-step windows, and `gradient_flow` on a non-normal $W_{hh}$.

## Designing the interface

The module exposes a cache type and five functions:

```python
# python/tinyllm/rnn/manual.py
class RNNCache(NamedTuple): x; h0; h; Wxh; Whh               # what the backward needs
def rnn_forward(x, h0, Wxh, Whh, bh) -> tuple[NDArray, RNNCache]      # h [T, B, H]
def rnn_backward(dh_all, cache, dh_next=None) -> dict[str, NDArray]  # keys x, h0, Wxh, Whh, bh
def tbptt_windows(T, k1, k2) -> list[tuple[int, int]]
def tbptt_grads(x, h0, Wxh, Whh, bh, dh_all, k1, k2) -> dict[str, NDArray]
def gradient_flow(cache, dh_last) -> tuple[NDArray, float]            # (norms [T], rho)
```

`rnn_forward` returns all hidden states and the cache; `rnn_backward` takes the per-step loss gradients `dh_all`, the cache, and an optional `dh_next` arriving from after the sequence, and returns a dictionary of gradients keyed by input name. `tbptt_windows` and `tbptt_grads` implement truncation, and `gradient_flow` takes a gradient arriving at the last state and returns the per-step norms and $\rho$.

## Building it step by step

Before reading the code, it helps to see the reasoning chain that fixes its shape.

1. **The backward needs every state.** Each step's VJP needs its input $x_t$, the previous state $h_{t-1}$ (for $\bar W_{hh}$), and $1 - h_t^2$; all are functions of the saved $h$. So the forward stores the whole `[T, B, H]` tensor and the cache holds `x`, `h0`, `h`, and the two weight matrices. The pre-activation is not stored, because $\tanh' = 1 - h^2$.
2. **Walk time in reverse with one carry.** $h_t$ has two consumers (the loss at $t$ and step $t + 1$), so its gradient is a sum; processing $t = T - 1$ down to 0, the second term is exactly what step $t + 1$ just produced. One variable, initialised from `dh_next`, holds it.
3. **Each step is two matmuls, so two VJPs.** `matmul_vjp(da, x_t, Wxh)` gives $(\bar x_t, \bar W_{xh}\text{ contribution})$ and `matmul_vjp(da, h_prev, Whh)` gives (the next carry, the $\bar W_{hh}$ contribution). Reusing the tested primitive means no transposes are written by hand, which is where sign and orientation bugs live.
4. **Shared weights accumulate.** The same $W_{xh}$, $W_{hh}$, $b_h$ are used at every step, so their gradients are sums over steps; per-step inputs ($\bar x_t$) are written once.
5. **Truncation is bookkeeping on windows.** Each cut at $e$ differentiates the losses since the previous cut through at most $k_2$ steps, treating the entering state as a constant. That means: run `rnn_backward` on the window with `h0` set to $h_{s-1}$, a gradient that is zero for steps already counted, and sum the results.
6. **Diagnose with the actual backward map.** The carry is multiplied by $\operatorname{diag}(1 - h_s^2) W_{hh}^\top$ per step; the diagnostic must apply exactly that, and report $\rho(W_{hh})$ as the long-run growth rate.

### The forward

The forward is one loop that keeps every state.

```python
def rnn_forward(x, h0, Wxh, Whh, bh):
    x, h0, Wxh, Whh, bh = (_f64(a) for a in (x, h0, Wxh, Whh, bh))   # float64 throughout
    _check(x, h0, Wxh, Whh, bh)                    # x [T,B,D], Wxh [D,H], Whh [H,H], h0 [B,H], bh [H]
    T, B, _ = x.shape
    h = np.empty((T, B, Whh.shape[0]), dtype=np.float64)
    prev = h0
    for t in range(T):
        prev = np.tanh(x[t] @ Wxh + prev @ Whh + bh)   # rows times right-multiplied weights
        h[t] = prev
    return h, RNNCache(x, h0, h, Wxh, Whh)            # everything the backward reads
```

### The backward

The backward has four ingredients: the carry, the tanh derivative from the saved output, two VJPs, and accumulation.

```python
def rnn_backward(dh_all, cache, dh_next=None):
    x, h0, h, Wxh, Whh = cache
    g = _f64(dh_all)                               # g[t]: the loss's direct gradient into h_t
    # ... shape checks ...
    carry = np.zeros_like(h0) if dh_next is None else _f64(dh_next)   # from after the sequence
    dx = np.zeros_like(x)
    dWxh = np.zeros_like(Wxh)
    dWhh = np.zeros_like(Whh)
    dbh = np.zeros(Whh.shape[0], dtype=np.float64)
    for t in range(h.shape[0] - 1, -1, -1):
        # Both paths into h_t: the loss at step t, and step t + 1 through Whh.
        dh = g[t] + carry
        da = dh * (1.0 - h[t] * h[t])  # tanh'(a_t) = 1 - tanh(a_t)^2 = 1 - h_t^2
        h_prev = h[t - 1] if t > 0 else h0          # the state that was multiplied by Whh
        dx[t], dW = matmul_vjp(da, x[t], Wxh)       # a_t contains x_t @ Wxh
        dWxh += dW                                  # shared weight: accumulate
        carry, dW = matmul_vjp(da, h_prev, Whh)     # a_t contains h_{t-1} @ Whh: next carry
        dWhh += dW
        dbh += da.sum(axis=0)                       # the bias is broadcast over the batch
    return {"x": dx, "h0": carry, "Wxh": dWxh, "Whh": dWhh, "bh": dbh}   # final carry = dL/dh0
```

Trace the hand example through it. At $t = 1$, `carry` is 0, `dh` is 1, and `da` is 0.874879; the first VJP gives $\bar x_1 = 0.437440$ and adds 0 to `dWxh` (since $x_1 = 0$); the second gives the carry $0.874879 \times 0.8 = 0.699904$ and adds $h_0 \bar a_1 = 0.404297$ to `dWhh`. At $t = 0$, `dh` is the carry, `da` is 0.550438, `dWxh` becomes 0.550438, `h_prev` is the start state 0, so `dWhh` stays 0.404297, and the final carry is $0.440350$.

### Truncation windows

The window list has ends every $k_1$ steps plus a last partial chunk, and each window reaches back $k_2$.

```python
def tbptt_windows(T: int, k1: int, k2: int) -> list[tuple[int, int]]:
    if T < 0 or k1 < 1 or k2 < k1:
        raise ValueError(...)                      # a window must cover the losses it owns
    ends = list(range(k1, T + 1, k1))
    if T > 0 and (not ends or ends[-1] != T):
        ends.append(T)                             # never drop the last partial chunk
    return [(max(0, e - k2), e) for e in ends]     # reach back k2, not k1
```

`k2 < k1` is refused because the window $(e - k_2, e)$ would not contain all the losses $[e - k_1, e)$ it is responsible for. The hand example, $T = 10$, $k_1 = 4$, $k_2 = 6$, gives ends 4, 8, 10 and windows `[(0, 4), (2, 8), (4, 10)]`.

### Truncated gradients

With the windows in hand, `tbptt_grads` does one full forward, then one `rnn_backward` per window with the entering state as a constant and only the window's own losses.

```python
h, cache = rnn_forward(x, h0, Wxh, Whh, bh)       # the forward is never truncated
# ... g, shape check, zero-initialised out ...
prev_end = 0
for start, end in tbptt_windows(h.shape[0], k1, k2):
    # The state entering the window is a constant: the cut.
    h_in = cache.h0 if start == 0 else h[start - 1]
    wcache = RNNCache(cache.x[start:end], h_in, h[start:end], cache.Wxh, cache.Whh)
    gw = np.zeros_like(h[start:end])
    # Only the losses since the last cut; earlier ones were counted then.
    gw[prev_end - start :] = g[prev_end:end]
    grads = rnn_backward(gw, wcache)
    out["x"][start:end] += grads["x"]
    if start == 0:
        out["h0"] += grads["h0"]                   # only a window that reaches t = 0 touches h0
    for k in ("Wxh", "Whh", "bh"):
        out[k] += grads[k]
    prev_end = end
return out
```

Building the window's cache from slices of the full forward means the window sees exactly the states the untruncated run produced; only the gradient is cut. With $k_2 \ge T$ every window starts at 0, so each loss is backpropagated through its entire history exactly once and the sum equals `rnn_backward` on the whole sequence. With $k_1 = k_2 = 1$ each window is one step whose entering state is a constant, so `dWhh` only receives $h_{t-1}^\top \bar a_t$ from the immediate loss and no gradient travels through $W_{hh}$ to an earlier step.

### Gradient flow

The diagnostic applies the true backward map to a gradient arriving at the last state and records the norm at each earlier state.

```python
def gradient_flow(cache: RNNCache, dh_last) -> tuple[NDArray, float]:
    h, Whh = cache.h, cache.Whh
    T = h.shape[0]
    # ... T >= 1, dh_last [B, H] ...
    norms = np.empty(T, dtype=np.float64)
    norms[T - 1] = np.linalg.norm(g)
    for t in range(T - 1, 0, -1):
        g = (g * (1.0 - h[t] * h[t])) @ Whh.T       # diag(1 - h_t^2), then W_hh transposed
        norms[t - 1] = np.linalg.norm(g)
    return norms, float(spectral_radius(Whh))       # rho, not the spectral norm
```

The full solution is [`manual.py`](../../../../course/solutions/python/tinyllm/rnn/manual.py).

### Why this design

Forward and backward both cost $O(T B (D H + H^2))$ time. Memory is $O(T B H)$ for the saved states, which is what full BPTT costs and what truncation bounds: TBPTT($k_1$, $k_2$) runs `rnn_backward` over $\lceil T / k_1 \rceil$ windows of at most $k_2$ steps, $O((T / k_1) k_2 B (DH + H^2))$ time, and a streaming implementation needs only $k_2$ states at once.

The cache saves $h$ and not $a$ because $\tanh'(a) = 1 - h^2$, so $h$ is enough, and $h$ is needed anyway: it is the output, and $h_{t-1}$ feeds $\bar W_{hh}$. Saving $a$ too would double the memory for nothing. The backward reuses `matmul_vjp` because the transposes in $\bar A = \bar Y B^\top$, $\bar B = A^\top \bar Y$ are the most common source of silent errors; one tested primitive used twice per step removes them and makes the backward read like the math.

Why write a backward by hand at all, when autograd exists? The [RNN language model](06-rnn-language-model-stateful-tbptt.md) wraps this pair as one fused autograd op: one graph node per sequence instead of a handful per step, and a backward with no Python-level graph traversal. Writing it also exposes the carry, which is where vanishing and exploding gradients come from. TBPTT is rebuilt from `rnn_backward` because windows are just short sequences with a constant entering state and a masked loss gradient, so the same tested backward serves both; the property "$k_2 \ge T$ equals full BPTT" then checks the bookkeeping, not a second backward.

The design accepts some tradeoffs: Python loops over time (the per-step matmuls dominate for useful $H$); float64 for exactness over speed; and a diagnostic that uses one gradient direction, so it shows the growth of that direction, bounded in the long run by $\rho$.

## What can go wrong

A correct solution has to survive a range of checks, from the hand example to golden comparisons with torch. The table lists them with the reason each matters downstream.

| Case | Kind of check | Checks | Why it matters downstream |
|---|---|---|---|
| The two-step hand example | unit | every number of the hand example | the carry and $1 - h^2$ are right |
| Gradcheck of full BPTT | gradcheck | all five gradients against central differences | the backward is the derivative of the forward |
| `dh_next` enters at the last step | gradcheck | a gradient arriving from after the sequence | stacking and windows in the language model |
| Matches autograd | differential | the same forward through `Tensor` gives the same gradients | the language model fuses this as one autograd op |
| Matches torch's RNN | golden | `torch.nn.RNN` outputs and gradients with transposed weights and summed biases | the convention is torch's |
| TBPTT window examples | unit | the hand windows, tiling, the partial chunk, bad arguments | the trainer's schedule |
| TBPTT with long reach is full BPTT | property | $k_2 \ge T$ equals full BPTT for every $k_1$ | each loss counted exactly once |
| TBPTT cuts long paths | unit | $k_1 = k_2 = 1$: no gradient crosses $W_{hh}$ | what truncation gives up |
| Gradient flow explodes and vanishes | property | norms follow $\rho^{T-1-t}$ along the top eigenvector; $\rho$ reported | diagnosing a stalled run |
| Gradient flow uses $W_{hh}^\top$ | unit | a non-normal $W_{hh}$: $W_{hh}^\top$ per step, $\rho = 0.5$ not the norm 2.13 | the diagnostic predicts the real gradient |
| Shapes and bad arguments | boundary | $T = 0$; mismatched shapes raise | errors at the call site |

The bugs these checks catch fall into five families.

> **Warning:** Forgetting to carry the gradient from step $t + 1$ into $h_t$ gives early inputs zero gradient, so the network only learns one-step dependencies. The solution writes `dh = g[t] + carry`, with the carry from the second VJP.

> **Warning:** Taking the tanh derivative as $1 - h$, or computing it from the wrong saved value, puts gradients off by a factor that grows with $\lvert h\rvert$. The solution uses `da = dh * (1.0 - h[t] * h[t])` with the same step's output.

> **Warning:** Pairing $\bar a_t$ with $h_t$ instead of $h_{t-1}$ in $\bar W_{hh}$, or overwriting instead of accumulating, gives a wrong recurrent gradient, and training diverges or stalls. The solution uses `h_prev = h[t - 1] if t > 0 else h0` and `dWhh += dW`.

> **Warning:** TBPTT bookkeeping errors (reaching back $k_1$ instead of $k_2$, counting a loss in two windows, dropping the last partial chunk) give a gradient that is neither full nor truncated, possibly larger than full BPTT. The solution uses `max(0, e - k2)`, `gw[prev_end - start:]`, and the appended final end.

> **Warning:** Diagnosing with $W_{hh}$ instead of $W_{hh}^\top$, or with the spectral norm, makes the diagnostic predict explosion where the gradient vanishes. `gradient_flow` multiplies by `Whh.T` and reports `spectral_radius`.

## Where this leads

Looking back, this section leaned on `matmul_vjp` from [Matrix differentials, the trace trick, and closed-form VJPs](../../mathematics-for-computing/09-matrix-calculus-and-autodiff/03-matrix-differentials-and-vjps.md), used twice per step; on `spectral_radius` from [Eigenvalues, power iteration, and the spectral radius](../../mathematics-for-computing/04-linear-algebra/04-eigenvalues-and-power-iteration.md) as the explosion predictor; and on [`Tensor`](../01-foundations/02-tensor-and-broadcasting-backward.md) and the [op library](../01-foundations/03-op-library-and-gradcheck.md) (`F.tanh`, `F.matmul`, `F.stack`) for the autograd cross-check. Looking forward, the [LSTM](02-lstm.md)'s cell state keeps the gradient from vanishing, and the [RNN language model](06-rnn-language-model-stateful-tbptt.md) wraps `rnn_forward` and `rnn_backward` as one autograd op for `cell='rnn'` and trains it with stateful TBPTT. Gradient clipping, from [Learning-rate schedules and gradient clipping](../../mathematics-for-computing/11-optimization/04-schedules-and-gradient-clipping.md), caps the explosion this diagnostic predicts.

Each piece here has a production counterpart worth reading:

| This piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `rnn_forward`/`rnn_backward` | cuDNN RNN (`cudnnRNNForward`, `cudnnRNNBackwardData`) | the input matmul for all steps as one GEMM, persistent kernels keeping $W_{hh}$ in registers | `aten/src/ATen/native/cudnn/RNN.cpp` |
| `tbptt_grads` | Keras `stateful=True`, fairseq's `detach` between chunks | stateful batching where row $b$ of each batch continues row $b$ of the previous one | `keras/layers/rnn/base_rnn.py` |
| `gradient_flow` | gradient-norm logging and clipping (`clip_grad_norm_`) | clipping caps the explosion this diagnostic predicts | `torch/nn/utils/clip_grad.py` |
| the vanilla cell | linear recurrences (S4, Mamba, RWKV) | recurrences with $\rho$ held below 1 by construction, trained in parallel over time | `mamba_ssm/ops/selective_scan_interface.py` |

### Further reading

- Werbos, "Backpropagation Through Time: What It Does and How to Do It" (1990).
- Williams and Peng, "An Efficient Gradient-Based Algorithm for On-Line Training of Recurrent Network Trajectories" (1990).
- Pascanu, Mikolov, and Bengio, "On the difficulty of training recurrent neural networks" (2013).

## Key terms

- **Backpropagation through time (BPTT)**: reverse-mode differentiation of the recurrence unrolled over time, with weights shared across steps.
- **Carry**: the gradient $\bar a_{t+1} W_{hh}^\top$ that step $t + 1$ sends back into $h_t$, added to the loss's direct gradient $g_t$.
- **Truncated BPTT, TBPTT($k_1$, $k_2$)**: run the forward straight through and, every $k_1$ steps, backpropagate the latest $k_1$ losses through at most $k_2$ steps, treating the entering state as a constant.
- **Spectral radius $\rho(W)$**: the largest eigenvalue magnitude of $W$, which sets the long-run growth or decay rate of repeated multiplication by $W$.
- **Exploding and vanishing gradients**: gradient norms that grow or shrink like $\rho^{T-1-t}$ as they travel back through the recurrence.

## Summary

- BPTT is reverse mode on the unrolled loop: the gradient reaching $h_t$ is what the loss sends it directly plus what step $t + 1$ sends back through $W_{hh}$.
- Every step's backward is two matmul VJPs and the tanh derivative $1 - h_t^2$, read from the saved output, so the forward must keep every hidden state.
- Shared weights accumulate their gradients over steps, and $\bar W_{hh}$ pairs $\bar a_t$ with the previous state $h_{t-1}$.
- Truncated BPTT keeps the forward running but cuts the gradient: every $k_1$ steps, the losses since the last cut are differentiated back $k_2$ steps; with $k_2 \ge T$ it is full BPTT.
- The gradient into an early state is the late one multiplied by $W_{hh}^\top$ once per step, so its norm grows or shrinks like $\rho(W_{hh})^{T-t}$: exploding and vanishing gradients, predicted by the spectral radius.

## Review questions

1. In the hand example, why would $\bar W_{xh}$ be 0 without the carry?
   <details><summary>Answer</summary>The loss reads only $h_1$, and $x_1 = 0$, so step 1 contributes $x_1 \bar a_1 = 0$ to $\bar W_{xh}$. The only nonzero contribution is $x_0 \bar a_0$, and $\bar a_0$ comes entirely from the carry (the direct gradient $g_0$ is 0). Dropping the carry makes the first input, which decides $h_1$, look useless.</details>
2. Why does $\bar W_{hh}$ at step $t$ pair $\bar a_t$ with $h_{t-1}$ and not $h_t$?
   <details><summary>Answer</summary>The term in $a_t$ is $h_{t-1} W_{hh}$; the VJP of $Y = AB$ with respect to $B$ is $A^\top \bar Y$, and $A$ here is $h_{t-1}$, the state that was multiplied. At $t = 0$ that is the start state $h_0$.</details>
3. In `tbptt_grads`, why is `gw` zero before `prev_end - start`?
   <details><summary>Answer</summary>Windows overlap when $k_2 > k_1$. The losses of steps before `prev_end` were already backpropagated at the previous cut; including them again would count them twice and make the truncated gradient larger than full BPTT.</details>
4. A non-normal $W_{hh}$ has spectral norm 2.13 but $\rho = 0.5$. Do gradients explode or vanish over long spans, and why does `gradient_flow` multiply by `Whh.T`?
   <details><summary>Answer</summary>They vanish: repeated multiplication grows like $\rho^k$ in the long run, even though a few steps can grow by up to the norm. The backward of $h_{t-1} W_{hh}$ multiplies the gradient by $W_{hh}^\top$, and a diagnostic that used $W_{hh}$ itself would follow a different map.</details>
5. Why does truncation not change the forward?
   <details><summary>Answer</summary>Truncation is an approximation of the gradient, not of the model. The states are those of the real recurrence over the whole stream; only the backward stops at $h_{s-1}$, which is treated as a constant. `tbptt_grads` makes this explicit by running one full forward and slicing it.</details>
