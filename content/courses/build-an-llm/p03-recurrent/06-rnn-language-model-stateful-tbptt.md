# RNN language model with stateful TBPTT

## Overview

| | |
|---|---|
| **Module** | `L3.6` · build · Python · Pass 4 · 2 h reading |
| **Solution** | [`python/tinyllm/rnn/rnnlm.py`](../../../../course/solutions/python/tinyllm/rnn/rnnlm.py): `ElmanRNN` (L3.1's forward and backward as one autograd op), `RNNLM` (`forward`, `init_state`, `detach_state`, `nll`, `generate`), `tbptt_batches`, `train_tbptt`, `save_rnnlm`, `load_rnnlm`; model property tests in [`test_rnnlm.py`](../../../../course/solutions/learner-tests/L3.6/python/tests/l3-6-rnnlm/test_rnnlm.py) |
| **Contract** | [`course/contracts/py/tinyllm/rnn/rnnlm.pyi`](../../../../course/contracts/py/tinyllm/rnn/rnnlm.pyi) |
| **Needs** | `L3.1` `rnn_forward`, `rnn_backward` · `L3.2` `LSTM` · `L3.3` `GRU` · `L0.5` `train_step` · `L0.3` `cross_entropy` · `L0.4` `Embedding`, `Linear` · `L0.2` ops · `L0.1` `Tensor`, `from_op`, `no_grad` · `L0.6` safetensors I/O · `M07.3` `xavier_uniform` · `M03.3` `orthogonal_init` · `M06.3` `PCG32` · training: `L1.1` `CharTokenizer`, `M10.3` `AdamW` · reading: `M10.4` (clipping), `M11.2` (bits per character) |
| **Used by** | later: `L6.7` the zoo's `rnnlm` bits-per-byte rows (joins the registry with B7); end of the recurrent stage: `{tinyllm} train rnnlm --cell {rnn,lstm,gru}` trains all three cells, with LSTM and GRU reaching the calibrated bar and the plain RNN worse than the LSTM |
| **Optional depth** | Mikolov et al., "Recurrent neural network based language model" (Interspeech 2010); Zaremba, Sutskever, and Vinyals, "Recurrent Neural Network Regularization" (2014), section 4; Williams and Peng, "An efficient gradient-based algorithm for on-line training of recurrent network trajectories" (1990) |

## Key Takeaways

- An RNN language model is an embedding, a recurrent layer, and a linear read-out; the three cells share one interface, and the Elman cell is `L3.1`'s hand-written BPTT wrapped as a single autograd op.
- Stateful training cuts the stream into contiguous lanes and walks them window by window: the hidden state flows from one window to the next.
- The gradient stops at the window boundary: the carried state is **detached**, not reset.
- Each new epoch starts from zeros, and clipping guards every step.
- A character LSTM trained this way for 150 steps beats the unigram entropy of the text by more than half a bit per character.

## How to read this chapter

Sections 1 to 3 build stateful truncated BPTT from first principles and work the lanes and windows by hand. Section 4 fixes the interface and what a correct solution must handle. Section 5 walks through the solution line by line and explains why each choice is the right one. Try to predict each step of the solution before reading it.

---

## 1. Why now

The n-gram model (`L2.1`) and the NPLM (`L2.2`) predict the next token from a fixed window: whatever happened 20 characters ago is invisible to them. The recurrent layers of `L3.1` to `L3.3` have a state that can, in principle, carry information across any distance. This module turns them into a language model and trains it the way recurrent LMs were trained in practice: on one long stream, with a state that persists across training windows. The result is the first model in the zoo (`L6.7`) whose context is not bounded by its architecture, and its bits per byte can be compared directly with the NPLM's on the same text.

## 2. Principles

| Symbol | Meaning | Type / shape |
|---|---|---|
| $V$, $d_e$, $d_h$ | vocabulary size, embedding width, hidden width | `int` |
| $x_t$ | token id at position $t$ | `int` |
| $E$ | embedding table | `float32[V, d_e]` |
| $h_t$ | recurrent state after reading $x_t$ ($h$ and $c$ for an LSTM) | `float32[n_layers, B, d_h]` |
| $W_o$, $b_o$ | read-out | `float32[V, d_h]`, `[V]` |
| $N$ | stream length | `int` |
| $B$ | number of lanes (`batch`) | `int` |
| $L = \lfloor N / B \rfloor$ | lane length | `int` |
| $k$ | window length (truncation) | `int` |
| $\mathrm{bpc}$ | bits per character: mean NLL in nats divided by $\ln 2$ | `float` |

### 2.1 Three cells, one interface

$$h_t = \mathrm{cell}(E[x_t], h_{t-1}), \qquad \text{logits}_t = W_o h_t + b_o, \qquad \mathcal{L} = \frac{1}{T}\sum_t -\log \operatorname{softmax}(\text{logits}_t)[x_{t+1}] .$$

`cell` is `L3.2`'s `LSTM`, `L3.3`'s `GRU`, or `ElmanRNN`, all with the same `forward(x, state) -> (out, state)`. `ElmanRNN` is the vanilla $h_t = \tanh(x_t W_{ih}^\top + h_{t-1} W_{hh}^\top + b_{ih} + b_{hh})$ in torch's parameter names; its forward over a whole sequence is **one** autograd node built with `from_op`: forward calls `L3.1`'s `rnn_forward`, backward calls `rnn_backward` on the saved cache. That is how cuDNN runs recurrent layers (one fused kernel per layer, not one graph node per time step), and it makes the hand-written BPTT of `L3.1` the engine of a real model. Two details: L3.1 uses $W_{xh} = W_{ih}^\top$, so the weight gradients are transposed back; the two biases are summed in the forward, so each receives the full bias gradient.

### 2.2 Stateful training

A long text could be backpropagated through in one piece only with memory proportional to its length. **Truncated BPTT** runs windows of $k$ tokens. The key choice is what state starts each window:

- **Stateless**: zeros every time. Simple, but the model never sees context from before the window, so it cannot learn dependencies longer than $k$.
- **Stateful**: the state the previous window ended with. The forward pass sees unbounded context; only the **gradient** is truncated at the window start.

Stateful training needs the state to be a **value** at the boundary: `detach_state` keeps the arrays and drops the graph, so backward on window $j + 1$ stops there instead of running on into window $j$'s graph (whose gradients were already applied).

### 2.3 Lanes and windows

For the carried state to make sense, row $b$ of window $j + 1$ must continue row $b$ of window $j$. So the stream is cut into $B$ **contiguous lanes**, lane $b$ = `stream[b L : (b + 1) L]` (the tail $N - BL$ is dropped), and window $j$ is columns $[jk, jk + k)$ of every lane as inputs, shifted by one as targets. A window needs $k + 1$ tokens, so the windows start at $p = 0, k, 2k, \dots$ while $p + k + 1 \le L$. After the last window the lanes start over and the state is reset to zeros: carrying the end of a lane into its own beginning would condition text on what never precedes it.

### 2.4 Clipping

The exploding gradients `L3.1` diagnosed are real in training: one bad window can produce a gradient that throws the weights far away. `train_tbptt` passes `clip` to `L0.5`'s `train_step`, which rescales the global gradient norm to at most `clip` (`M10.4`) before every step.

### 2.5 Evaluation and sampling

`nll(ids)` scores a held-out stream in one lane, chunk by chunk under `no_grad`, carrying the state, so its result does not depend on the chunk size; the mean divided by $\ln 2$ is bits per character, and per byte with `M11.2`'s accumulator. `generate` feeds each sampled token back in, drawing from `PCG32(seed)` exactly as `L0.5`'s bigram sampler does.

## 3. Worked example by hand

The stream $0, 1, \dots, 19$, $B = 2$ lanes, $k = 3$. $L = 10$: lane 0 is $0 \dots 9$, lane 1 is $10 \dots 19$. A window needs $k + 1 = 4$ tokens, so it starts at $p = 0, 3, 6$ ($p = 9$ would need token 12 of a 10-token lane):

| window | inputs | targets |
|---|---|---|
| 0 | `[[0 1 2] [10 11 12]]` | `[[1 2 3] [11 12 13]]` |
| 1 | `[[3 4 5] [13 14 15]]` | `[[4 5 6] [14 15 16]]` |
| 2 | `[[6 7 8] [16 17 18]]` | `[[7 8 9] [17 18 19]]` |

The state after reading `0 1 2` starts the read of `3 4 5`: lane 0 continues itself. Tokens 9 and 19 are only targets. With interleaved lanes (`reshape(L, B).T`), lane 0 would be $0, 2, 4, \dots$ and the state would carry across text that is not contiguous. A correct `tbptt_batches` returns exactly these three windows.

## 4. The interface

```python
class ElmanRNN(Module):
    def __init__(self, d_in, d_h, num_layers=1, rng=None): ...
    def forward(self, x, state=None, lengths=None) -> tuple[Tensor, Tensor]: ...
class RNNLM(Module):
    def __init__(self, vocab, d_emb, d_h, cell, n_layers=1, rng=None): ...
    def forward(self, ids, state=None) -> tuple[Tensor, Any]: ...    # ids [B, T] -> logits [B, T, V]
    def init_state(self, batch): ...; def detach_state(self, state): ...
    def nll(self, ids, chunk=256) -> NDArray: ...; def generate(self, prefix, n, temperature, seed) -> list[int]: ...
def tbptt_batches(stream, k, batch) -> list[tuple[NDArray, NDArray]]: ...
def train_tbptt(model, stream, k, batch, opt, clip, steps) -> list[float]: ...
def save_rnnlm(model, dir, tokenizer="bytes") -> None: ...; def load_rnnlm(dir) -> RNNLM: ...
```

### What a correct solution must handle

| Case | Kind of check | Checks | Why it matters downstream |
|---|---|---|---|
| hand example: windows | unit | section 3's three windows | the lanes the state flows along |
| windows drop the tail and validate | boundary | the tail is dropped; too-short lanes raise | no empty epochs |
| Elman layer is one fused op over L3.1 | differential | output and every gradient equal `rnn_forward` and `rnn_backward` | the hand-written BPTT drives a real model |
| gradcheck of the whole model | gradcheck | every parameter of a 2-layer model, Elman and LSTM cells, float64 | the model trains through all its layers |
| state carries across calls | property | two calls with the carried state equal one call | the definition of stateful |
| gradient is cut at the window boundary | differential | step 2's gradients equal window 2 alone from a constant state | truncation is where it is meant to be |
| state flows into the next window | differential | step 2's loss equals the run from window 1's state, not from zeros | context crosses windows |
| epoch wraps and resets the state | unit | step `len(windows)` repeats step 0 exactly | each epoch starts clean |
| clip is applied | unit | the optimizer sees a gradient norm of at most `clip` | the guard against explosions |
| NLL does not depend on the chunk | property | chunk 1, 7, 256 agree and equal one pass | evaluation of long held-out text |
| generate is seeded, greedy is argmax | unit | same seed, same ids; greedy feeds back the argmax | `{tinyllm} generate` |
| save and load roundtrip | unit | `config.json`, torch key names, identical logits | the zoo's checkpoint contract |
| validation | boundary | unknown cell, bad ids, lengths for the Elman layer, negative steps | caller bugs fail loudly |
| LSTM learns characters | learning | 150 steps: held-out bpc at the reference bar and 0.5 below the unigram entropy | the model learns |

### The properties, as tests

The solution rests on **properties** that hold for any seed: two calls with a carried state equal one call; `nll` does not depend on the chunk; with an optimizer that only records gradients (a small class with `zero_grad` and `step`), step 2's gradients equal those of window 2 from a constant state, the step `len(windows)` loss equals step 0's, and every recorded gradient norm is at most `clip`. The model suite [`test_rnnlm.py`](../../../../course/solutions/learner-tests/L3.6/python/tests/l3-6-rnnlm/test_rnnlm.py) writes these as tests against the contract only, together with the section 3 windows, the Elman layer against `rnn_forward`, greedy generation, and a save and load.

## 5. The solution, step by step

### From first principles

1. **The state carried between windows must belong to the same text.** So row $b$ of every window must be the continuation of row $b$ of the previous one. That forces $B$ contiguous lanes, `stream[:BL].reshape(B, L)`, and windows as column slices of that matrix (section 3).
2. **A window of $k$ inputs needs $k + 1$ tokens.** Targets are inputs shifted by one. Starts are $0, k, 2k, \dots$ with $p + k + 1 \le L$, which is `range(0, L - k, k)`. In the worked example $L = 10$, $k = 3$ gives starts $0, 3, 6$.
3. **Carry the value, cut the graph.** The next window's forward must start from the previous window's final state (stateful), but its backward must stop there, because the previous window's gradients have already been applied. `Tensor.detach()` gives exactly that: same array, no parents.
4. **Reset at the epoch boundary.** After the last window the lanes restart from their beginnings, so the carried state (from the lane's end) does not precede that text. Starting from zeros is the honest choice.
5. **The Elman layer can reuse `L3.1` wholesale.** `rnn_forward`/`rnn_backward` already compute a full sequence and its BPTT. Wrapping them as one `from_op` node needs only two conversions: torch stores $W_{ih}$, $W_{hh}$ as the transposes of `L3.1`'s $W_{xh}$, $W_{hh}$, and torch's two biases enter as their sum, so each gets the full bias gradient.
6. **Evaluation must be chunk-invariant.** Scoring a long held-out stream in chunks is fine as long as the state flows from chunk to chunk; then the result equals one pass, whatever the chunk size.

### The code, annotated

Lanes and windows:

```python
def tbptt_batches(stream, k, batch):
    s = np.asarray(stream)
    # ... (1-D integer stream, k >= 1, batch >= 1)
    L = len(s) // batch
    if L < k + 1:
        raise ValueError(f"each of {batch} lanes has {L} tokens; a window needs k + 1 = {k + 1}")
    # Lane b is one contiguous stretch of the stream, so the state that lane b
    # carries from window j into window j + 1 belongs to the same text.
    lanes = s[: L * batch].astype(np.int64).reshape(batch, L)      # tail N - BL dropped
    return [(lanes[:, p : p + k], lanes[:, p + 1 : p + k + 1]) for p in range(0, L - k, k)]
```

`reshape(batch, L)` is the whole point: with `reshape(L, batch).T` lane 0 would be $0, 2, 4, \dots$ (pitfall 3).

The training loop keeps the carried state in a small dict that `loss_fn` closes over, because `L0.5`'s `train_step` calls `loss_fn(model, batch)` and owns zero_grad, backward, clipping and the optimizer step:

```python
def train_tbptt(model, stream, k, batch, opt, clip, steps):
    # ... (steps >= 0)
    windows = tbptt_batches(stream, k, batch)
    carry = {"state": None}

    def loss_fn(m, b):
        logits, state = m(b["x"], carry["state"])   # stateful: start from the last window's end
        # Keep the values, cut the graph: the next window starts here.
        carry["state"] = m.detach_state(state)
        V = logits.shape[-1]
        return cross_entropy(F.reshape(logits, (-1, V)), b["y"].reshape(-1))

    losses = []
    model.train()
    j = 0
    while len(losses) < steps:
        if j == 0:
            carry["state"] = None  # a new epoch starts from zeros
        x, y = windows[j]
        losses.append(train_step(model, {"x": x, "y": y}, loss_fn, opt, clip=clip)["loss"])
        j = (j + 1) % len(windows)                  # wrap: the next epoch
    return losses
```

Detaching happens *inside* the forward of window $j$, before window $j$'s backward. That is safe because the detached copy is only used by window $j + 1$; window $j$'s own graph still runs through the undetached `state`. `detach_state` recurses into the LSTM's `(h, c)` tuple:

```python
def detach_state(self, state):
    if isinstance(state, tuple):
        return tuple(self.detach_state(s) for s in state)   # LSTM: (h, c)
    return state.detach() if isinstance(state, Tensor) else state
```

The Elman layer as one autograd node:

```python
def _rnn_layer(x, h0, w_ih, w_hh, b_ih, b_hh):
    """One Elman layer over the whole sequence as a single autograd op."""
    # L3.1 takes Wxh = W_ih^T, Whh = W_hh^T and one bias.
    h, cache = rnn_forward(x.data, h0.data, w_ih.data.T, w_hh.data.T, b_ih.data + b_hh.data)

    def vjp(g):
        d = rnn_backward(g, cache)                  # L3.1's BPTT over all T steps
        db = d["bh"].astype(b_ih.dtype)
        # torch's weights are the transposes of L3.1's Wxh and Whh; the two
        # biases are summed in the forward, so each gets the same gradient.
        return (d["x"].astype(x.dtype), d["h0"].astype(h0.dtype),
                d["Wxh"].T.astype(w_ih.dtype), d["Whh"].T.astype(w_hh.dtype),
                db, db.copy())

    return from_op(h.astype(x.dtype), (x, h0, w_ih, w_hh, b_ih, b_hh), vjp, "elman_rnn")
```

`ElmanRNN.forward` stacks layers by feeding each layer's output sequence to the next, and returns the last time step of each layer as `h_n`, shape `[n_layers, B, d_h]`, the same convention as torch and `L3.2`/`L3.3`. It rejects `lengths`, because a language-model stream has no padding.

The model itself is the three lines of section 2.1, with a transpose on each side because the recurrent layers are time first:

```python
def forward(self, ids, state=None):
    a = self._ids(ids)
    B, T = a.shape
    if state is None:
        state = self.init_state(B)                  # zeros; (h, c) for an LSTM
    x = self.emb(a.T)  # time first for the recurrent layer: [T, B, d_emb]
    out, new_state = self.rnn(x, state)
    logits = self.out(F.transpose(out, 0, 1))  # back to batch first: [B, T, V]
    return logits, new_state
```

Chunked evaluation carries one `state` across all chunks and computes a stable log-softmax in float64:

```python
def nll(self, ids, chunk=256):
    a = np.asarray(ids)
    # ... (1-D, at least 2 ids, chunk >= 1)
    out = np.empty(len(a) - 1, dtype=np.float64)
    state = None
    with no_grad():
        for s in range(0, len(a) - 1, chunk):
            inp = a[s : min(s + chunk, len(a) - 1)]          # the last id is only a target
            logits, state = self.forward(inp[None, :], state)
            z = logits.data[0].astype(np.float64)
            m = z.max(axis=-1, keepdims=True)
            ls = z - m - np.log(np.exp(z - m).sum(axis=-1, keepdims=True))
            tgt = a[s + 1 : s + 1 + len(inp)]
            out[s : s + len(inp)] = -ls[np.arange(len(inp)), tgt]
    return out
```

`generate` runs the prefix once, then feeds back each chosen token with the carried state: greedy (`argmax`) at temperature 0, otherwise `L0.5`'s inverse-CDF draw on one `rng.uniform()` per token from `PCG32(seed)`. `save_rnnlm` writes `config.json` (architecture, cell, sizes and the tokenizer name) and `model.safetensors` with torch's key names from `state_dict()`; `load_rnnlm` rebuilds the model from the config and loads strictly. Full solution: [`rnnlm.py`](../../../../course/solutions/python/tinyllm/rnn/rnnlm.py).

### Why this solution

- **Cost.** Each step is $O(k B (d_e d_h + d_h^2 + d_h V))$ time and $O(k B d_h)$ activation memory, independent of the stream length $N$. Full BPTT would need $O(N d_h)$ memory per lane.
- **Stateless windows** would be simpler (no carry, no detach) but cap the usable context at $k$. Stateful training gives unbounded forward context for the same memory; only the gradient horizon is $k$.
- **Overlapping (sliding) windows** would score every token with long context but cost $k$ times more compute per epoch. Non-overlapping windows plus a carried state get most of the context at no extra cost.
- **One fused op vs one node per time step.** Building the Elman layer from per-step `Tensor` ops would also work, but creates $T$ graph nodes per layer; the fused op mirrors cuDNN and reuses already-verified BPTT.
- **Tradeoff accepted.** Truncation means the gradient never connects a token to context more than $k$ steps back, so long dependencies can be *used* (via the carried state) but are learned only indirectly.

### Explain it back

1. Why does `train_tbptt` detach the state instead of resetting it to zeros?
   <details><summary>Answer</summary>Resetting would make training stateless: each window would see no context from before it. Detaching keeps the values, so the forward pass still sees all earlier text in the lane, and only removes the graph so backward stops at the window boundary.</details>
2. What goes wrong if the carried state is not detached at all?
   <details><summary>Answer</summary>Window $j + 1$'s backward would run on into window $j$'s graph, whose gradients were already applied. The earlier windows' gradients get counted again, and memory grows with every step.</details>
3. Why does `tbptt_batches` use `reshape(batch, L)` and not `reshape(L, batch).T`?
   <details><summary>Answer</summary>The carried state of row $b$ must describe the text right before row $b$'s next window. `reshape(batch, L)` makes each row one contiguous stretch; the transposed form interleaves tokens, so row 0 would be $0, 2, 4, \dots$ and the state would carry across text that never appears together.</details>
4. Why does the Elman vjp return the same bias gradient twice?
   <details><summary>Answer</summary>The forward uses $b_{ih} + b_{hh}$. The derivative of a sum with respect to each term is 1, so both biases receive the full gradient `d["bh"]`.</details>
5. Why is `nll` independent of `chunk`?
   <details><summary>Answer</summary>The recurrence is a function of the whole prefix through the state alone. Carrying `state` from chunk to chunk makes the chunked computation identical to one pass; only resetting it per chunk would make the score depend on the chunk size.</details>

## 6. Pitfalls

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| 1. the carried state not detached | backward runs into the previous window's graph again; gradients double-count | `loss_fn` stores `m.detach_state(state)` in `carry` (step 3) |
| 2. "detaching" by resetting to zeros every window | stateless training: no context beyond $k$ | the detached state keeps its values; only the graph is dropped |
| 3. interleaved lanes | the state carries across non-contiguous text | `reshape(batch, L)`: row $b$ is one contiguous stretch |
| 4. no clipping | one exploding window ruins the run | `train_step(..., clip=clip)` on every window |
| bias gradient to `bias_ih` only | `bias_hh` never trains | the vjp returns `db` and `db.copy()` |
| weight gradients not transposed back | wrong updates (or a crash when $d_e \ne d_h$) | the vjp returns `d["Wxh"].T` and `d["Whh"].T` |
| `forward` ignores its state | no statefulness at all | `forward` passes `state` into `self.rnn` and returns the new state |
| no reset at the epoch start | the first window is conditioned on the lane's end | `if j == 0: carry["state"] = None` |
| `nll` restarting the state per chunk | the score depends on the chunk size | one `state` variable carried across all chunks |
| `generate` feeding back the wrong token | greedy text differs from the argmax loop | the next forward is fed `[[nxt]]`, the token just chosen |
| `tokenizer` ignored by `save_rnnlm` | the zoo loads the wrong tokenizer | `"tl_tokenizer": tokenizer` in `config.json` |

## 7. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `L3.1` | `rnn_forward` and `rnn_backward` are the Elman layer's forward and backward |
| Back | `L3.2` | `LSTM`, `cell = "lstm"` |
| Back | `L3.3` | `GRU`, `cell = "gru"` |
| Back | `L0.5` | `train_step` runs each window (zero_grad, backward, clip, step) |
| Back | `L0.3` | `cross_entropy` over the window |
| Back | `L0.4` | `Embedding` and `Linear` |
| Back | `L0.2` | `F.transpose`, `F.stack`, `F.reshape` |
| Back | `L0.1` | `Tensor`, `from_op`, `no_grad` |
| Back | `L0.6` | `save_safetensors` and `load_safetensors` for the model directory |
| Back | `M07.3` | `xavier_uniform` for the Elman input weights |
| Back | `M03.3` | `orthogonal_init` for the Elman recurrent weights |
| Back | `M06.3` | `PCG32` for initialization and sampling |
| Back | `L1.1` | `CharTokenizer` encodes the corpus in the learning test |
| Back | `M10.3` | `AdamW` trains the learning test |
| Forward | `L6.7` | the zoo trains `rnnlm` checkpoints and reports their bits per byte next to the n-gram and the transformer |
| Forward | `L3.5` | ELMo's biLM is two of these language models, one per direction |

## Going further

| This piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `train_tbptt` | PyTorch's word language model example | the same `batchify` lanes and `repackage_hidden` detach | `pytorch/examples`, `word_language_model/main.py` |
| `ElmanRNN` fused op | cuDNN RNN kernels | one kernel per layer, weights packed for both directions | `torch/nn/modules/rnn.py`, `_VF.rnn_tanh` |
| LSTM LM | AWD-LSTM | weight-dropped hidden matrices, variational dropout, averaged SGD | Merity, Keskar, and Socher (2017), `salesforce/awd-lstm-lm` |
| truncation | RWKV, Mamba | recurrent models trained in parallel over the whole sequence, run as RNNs at inference | Peng et al. 2023; Gu and Dao 2023 |
