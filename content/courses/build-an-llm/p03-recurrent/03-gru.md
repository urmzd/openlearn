# GRU (torch gate order)

## Overview

| | |
|---|---|
| **Module** | `L3.3` · build · Python · Pass 4 |
| **Solution** | [`gru.py`](../../../../course/solutions/python/tinyllm/rnn/gru.py): `gru_cell`, `GRUCell`, `GRU` (stacked layers, inter-layer dropout, `lengths` for padded batches), with `_init` and `_mask` · model tests [`test_gru.py`](../../../../course/solutions/learner-tests/L3.3/python/tests/l3-3-gru/test_gru.py) |
| **Contract** | [`course/contracts/py/tinyllm/rnn/gru.pyi`](../../../../course/contracts/py/tinyllm/rnn/gru.pyi) |
| **Needs** | `L0.1` `Tensor` · `L0.2` the op library · `L0.4` `Module` and `Dropout` · `M07.3` `xavier_uniform` · `M03.3` `orthogonal_init` · `M06.3` the default PCG32 stream · reading: `L3.2` the LSTM (same layout, same masking), `M01.3`, `craft.03` |
| **Used by** | `L3.4` two GRUs make a bidirectional layer · later: `L3.6` `cell='gru'`, whose language model is one row of the recurrent comparison, and `L4.1` the default encoder and decoder cell |
| **Optional depth** | Cho et al., "Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation" (2014); Chung et al., "Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling" (2014); the cuDNN developer guide, RNN formulas |

## Key Takeaways

- A GRU has one state and two gates: the update gate $z$ interpolates, $h' = (1 - z) \odot n + z \odot h$, so $z \to 1$ copies the state and its gradient unchanged.
- The reset gate multiplies the recurrent part **after** its bias, $n = \tanh(W_{in}x + b_{in} + r \odot (W_{hn}h + b_{hn}))$: torch's and cuDNN's form, not Cho's original, and weights trained one way do not load the other.
- Gate blocks are stacked $r, z, n$ with torch's names, so torch GRUs load by name and give torch's outputs and gradients, padded batches included.
- Three gate blocks instead of four: three quarters of the LSTM's parameters for a similar memory, which is why `L4.1` defaults to it.

## How to read this chapter

Sections 1 and 2 derive the GRU as a one-state LSTM and pin the one detail that breaks checkpoints, the reset gate's position. Section 3 runs one cell step and its state gradient by hand, in torch's form and in Cho's. Section 4 fixes the interface. Section 5 walks through the solution and explains why each line is the right one. Try to predict each step of the solution before reading it.

---

## 1. Why now

The LSTM (`L3.2`) solved vanishing gradients with two states and four gates. Cho et al. found in 2014 that one state and two gates do about as well: merge the cell and hidden state, and let a single update gate decide how much of the old state to keep, with the complement going to the new candidate. The seq2seq models of Part 4 (`L4.1`) use the GRU by default because it is a quarter smaller and a little faster. It also brings the course's most common weight-loading bug: the reset gate's position. Papers write it one way, cuDNN and torch compute it another, and a checkpoint only works with the form it was trained in. The solution builds torch's form and checks it against torch.

## 2. Principles

| Symbol | Meaning | Type / shape |
|---|---|---|
| $x_t$, $h_t$ | input and state | `float32[B, D]`, `float32[B, H]` |
| $W_{ih}$, $W_{hh}$ | input and recurrent weights, three gate blocks stacked | `[3H, D]`, `[3H, H]` |
| $b_{ih}$, $b_{hh}$ | the two biases | `[3H]` each |
| $a = x W_{ih}^\top + b_{ih}$ | the input part of all three pre-activations | `[B, 3H]` |
| $u = h W_{hh}^\top + b_{hh}$ | the recurrent part, bias included | `[B, 3H]` |
| $r, z, n$ | reset gate, update gate, candidate | `[B, H]` each |
| $\sigma$, $\tanh$, $\odot$ | sigmoid, tanh, elementwise product | |
| $\ell_b$ | length of sequence $b$ in a padded batch | integer in $1..T$ |

**The cell.** Split $a$ and $u$ into blocks of $H$ columns, in torch's order $r, z, n$:

$$r = \sigma(a_{[0:H]} + u_{[0:H]}),\qquad z = \sigma(a_{[H:2H]} + u_{[H:2H]}),$$
$$n = \tanh\bigl(a_{[2H:3H]} + r \odot u_{[2H:3H]}\bigr),\qquad h' = (1 - z) \odot n + z \odot h.$$

$r$ decides how much of the old state the candidate may read: $r = 0$ makes $n$ a function of the input alone, a reset. $z$ decides how much of the old state survives: $h'$ is a convex combination of $n$ and $h$, so it stays in $(-1, 1)$ without an extra tanh.

**Where the reset gate goes.** Cho's paper computes $n = \tanh(W_{in}x + W_{hn}(r \odot h) + b_n)$: reset first, then the matmul. cuDNN computes $r \odot (W_{hn}h + b_{hn})$: matmul first (so the recurrent matmul for all three gates is one GEMM before any gate is known), then the reset, which also scales the recurrent bias $b_{hn}$. torch follows cuDNN. The two forms are different functions with the same parameter shapes, so nothing fails at load time; the outputs are just wrong. The worked example shows the gap: 0.635 against 0.848.

**Why it does not vanish.** $\partial h'/\partial h = \operatorname{diag}(z) + (\text{terms through } n, r, z)$. With $z \to 1$ the first term is the identity and the others vanish ($1 - z \to 0$ and $\sigma'(\cdot) \to 0$), so the state and its gradient pass through a step unchanged, the GRU's version of the LSTM's forget gate. Saturating $z$ with a large bias gives $\partial h_T/\partial h_0 = 1$ to within rounding over 15 steps.

**Layout, initialization, stacking, padding.** Everything else follows `L3.2` with three blocks instead of four: torch's names `weight_ih_l{k}, weight_hh_l{k}, bias_ih_l{k}, bias_hh_l{k}` in that order; per layer, three Xavier input blocks (`M07.3`) then three orthogonal recurrent blocks (`M03.3`), both biases 0 (no gate here plays the forget gate's role at initialization: $z$ starts at $\sigma(0) = 0.5$); dropout between layers in training mode only; and the `lengths` mask that freezes a finished sequence's state and zeroes its outputs.

## 3. Worked example by hand

One unit, $x = 1$, $h = 0.5$. Weights: $W_{ih} = 0$, $b_{ih} = [0, \ln 3, 0]$, $W_{hh} = [0, 0, 1]^\top$ (only the candidate's recurrent part reads $h$), $b_{hh} = [0, 0, 1]$.

1. $a = [0, \ln 3, 0]$; $u = [0, 0, 0.5 \times 1 + 1] = [0, 0, 1.5]$.
2. $r = \sigma(0) = 0.5$; $z = \sigma(\ln 3) = 0.75$.
3. $n = \tanh(0 + 0.5 \times 1.5) = \tanh(0.75) = 0.635149$.
4. $h' = 0.25 \times 0.635149 + 0.75 \times 0.5 = 0.158787 + 0.375 = 0.533787$.
5. $\partial h'/\partial h = z + (1 - z)(1 - n^2)\,r\,W_{hn} = 0.75 + 0.25 \times 0.596586 \times 0.5 = 0.824573$ (the gates do not read $h$ here, so no other terms).

Cho's form would compute $n = \tanh(0 + (0.5 \times 0.5) \times 1 + 1) = \tanh(1.25) = 0.848284$; swapping the interpolation would give $h' = 0.75n + 0.25h = 0.601362$.

This is the first hand case of section 4.

## 4. The interface

```python
# python/tinyllm/rnn/gru.py
def gru_cell(x, h, w_ih, w_hh, b_ih, b_hh) -> Tensor                     # h'
class GRUCell(Module):   # weight_ih [3H, D], weight_hh [3H, H], bias_ih, bias_hh [3H]
    def __init__(self, d_in, d_h, rng=None); def forward(self, x, h=None)
class GRU(Module):       # weight_ih_l{k}, weight_hh_l{k}, bias_ih_l{k}, bias_hh_l{k}
    def __init__(self, d_in, d_h, num_layers=1, dropout=0.0, rng=None)
    def forward(self, x, state=None, lengths=None) -> tuple[Tensor, Tensor]
    # x [T, B, d_in] -> out [T, B, H], h_n [num_layers, B, H]
```

### What a correct solution must handle

| Case | Kind of check | Checks | Why it matters downstream |
|---|---|---|---|
| The hand example cell | unit | section 3: $h' = 0.533787$, $\partial h'/\partial h = 0.824573$ | the gates, their order, and the reset placement are right |
| Matches torch's cell | golden | `torch.nn.GRUCell` outputs and every gradient | torch weights load and run |
| Matches torch's sequence | golden | one layer with a state, two layers, a packed padded batch | `L4.1` loads and trains GRUs |
| State-dict names match torch | golden | torch's names, order, and shapes | the safetensors key contract |
| Gradcheck of the cell | gradcheck | autograd through the cell against central differences | the backward is right without torch |
| The update gate near one keeps the state | property | saturated $z$: $h_T = h_0$ and $\partial h_T/\partial h_0 = 1$ | the reason the GRU remembers |
| Orthogonal recurrence and zero bias | unit | zero biases, orthogonal $W_{hh}$ blocks, Xavier bound, seeding | trainable from step 0 |
| Lengths keep padding out | boundary | padded outputs 0; final state equals running alone; no gradient into padding | batches of sentences (`L4.1`) |
| Dropout between layers only | unit | dropout between layers, in training mode only | evaluation is deterministic |
| Default state and bad arguments | boundary | no state means zeros; bad shapes, lengths, sizes raise | errors at the call site |

The model tests in [`test_gru.py`](../../../../course/solutions/learner-tests/L3.3/python/tests/l3-3-gru/test_gru.py) start from section 3 (`gru_cell` with $W_{hh} = [0, 0, 1]^\top$, $b_{ih} = [0, \ln 3, 0]$, $b_{hh} = [0, 0, 1]$, and $h = 0.5$ must give $0.25 \tanh(0.75) + 0.375$). The case that tells the two reset placements apart is the cell against a numpy formula with random weights and $b_{hn} \ne 0$; the others check the `state_dict` keys, orthogonal blocks, a padded batch, a 2-layer GRU against stacked numpy cells from a nonzero $h_0$, dropout ignored by a 1-layer GRU, and bad arguments.

## 5. The solution, step by step

### 5.1 From first principles

1. **One state, an interpolating update.** Merge the LSTM's cell and hidden state, and let one gate split the mass: $h' = (1 - z) \odot n + z \odot h$. With $z \to 1$ the step is the identity, so the state and its gradient pass unchanged; this replaces the forget gate. Because $h'$ is a convex combination of values in $(-1, 1)$, no output tanh is needed.
2. **A reset gate for the candidate.** The candidate should be able to ignore the old state (a reset). So $r$ scales the recurrent contribution inside $n$'s tanh.
3. **Where $r$ multiplies is a choice of function.** Placing $r$ after the recurrent matmul (and its bias) lets the recurrent GEMM for all three gates run before any gate is known, which is why cuDNN and torch use it. Cho's original places $r$ before the matmul. Same shapes, different functions: the solution must use torch's form to load torch weights.
4. **Everything else is the LSTM's machinery.** Three blocks in torch's order $r, z, n$ with two biases, Xavier input and orthogonal recurrent blocks, layers registered with torch's names, dropout between layers, and the same padding mask. Autograd derives the backward.
5. **No special bias at initialization.** No gate plays the forget gate's role at step 0: $z = \sigma(0) = 0.5$ already keeps half the state, so both biases start at 0.

### 5.2 The code, annotated

**Parameters.** Three Xavier input blocks, then three orthogonal recurrent blocks, from one generator; zero biases.

```python
def _init(d_in, d_h, rng):
    r = rng if rng is not None else PCG32(0).substream("init")
    w_ih = np.concatenate([xavier_uniform((d_h, d_in), 1.0, r) for _ in range(GATES)])  # [3H, D]
    w_hh = np.concatenate([orthogonal_init((d_h, d_h), 1.0, r) for _ in range(GATES)])  # [3H, H]
    return w_ih, w_hh, np.zeros(GATES * d_h), np.zeros(GATES * d_h)
```

**The cell.** The input part `a` and the recurrent part `b` are each computed once for all three gates, biases included; the gates add the matching slices, and the candidate scales the recurrent slice by $r$ after its bias.

```python
def gru_cell(x, h, w_ih, w_hh, b_ih, b_hh):
    a = F.matmul(x, F.transpose(w_ih, 0, 1)) + b_ih      # [B, 3H]: x W_ih^T + b_ih
    b = F.matmul(h, F.transpose(w_hh, 0, 1)) + b_hh      # [B, 3H]: h W_hh^T + b_hh, one GEMM
    H = a.shape[-1] // GATES
    r = F.sigmoid(a[..., 0:H] + b[..., 0:H])             # reset gate
    z = F.sigmoid(a[..., H : 2 * H] + b[..., H : 2 * H]) # update gate
    # torch's (and cuDNN's) form: r scales h @ w_hn^T + b_hn, bias included.
    n = F.tanh(a[..., 2 * H : 3 * H] + r * b[..., 2 * H : 3 * H])
    return (1.0 - z) * n + z * h                          # z -> 1 keeps h
```

On section 3: `a` is $[0, \ln 3, 0]$, `b` is $[0, 0, 1.5]$, so $r = 0.5$, $z = 0.75$, $n = \tanh(0.75) = 0.635149$, and $h' = 0.533787$. In Cho's form the candidate would read $\tanh(W_{hn}(r \odot h) + b_n) = \tanh(1.25)$, because the bias would not be scaled by $r$.

The last line uses `h` itself, not `h.data`: the $z \odot h$ term is the state's own path back through time, and cutting it from the graph would leave a correct forward with a weaker gradient.

**The modules.** `GRUCell` registers `weight_ih`, `weight_hh`, `bias_ih`, `bias_hh`; with no `h` it starts from zeros. `GRU` registers `weight_ih_l{k}, weight_hh_l{k}, bias_ih_l{k}, bias_hh_l{k}` per layer, drawing every layer from one generator, exactly as the LSTM does. Its forward has one state instead of two:

```python
h0 = _as_tensor(np.zeros((L, B, H), dtype=x.dtype) if state is None else state, x.dtype)
mask = _mask(lengths, T, B)                   # bool [T, B, 1], true at real steps
seq = x
h_n = []
for k in range(L):
    w = [getattr(self, f"{n}_l{k}") for n in ("weight_ih", "weight_hh", "bias_ih", "bias_hh")]
    h = h0[k]                                 # this layer's own initial state
    outs = []
    for t in range(T):
        h_new = gru_cell(seq[t], h, *w)
        if mask is None:
            h = h_new
            outs.append(h_new)
        else:
            m = mask[t]
            h = F.where(m, h_new, h)          # past its length: keep the state
            outs.append(F.where(m, h_new, 0.0))   # and output 0
    seq = F.stack(outs, axis=0)
    if k < L - 1:
        seq = self.drop(seq)                  # between layers only
    h_n.append(h)
return seq, F.stack(h_n, axis=0)              # h_n [num_layers, B, H]
```

Full solution: [`gru.py`](../../../../course/solutions/python/tinyllm/rnn/gru.py).

### 5.3 Why this solution

- **Complexity.** Per step and layer, two matmuls of cost $O(B \cdot 3H (D + H))$: three quarters of an LSTM of the same $H$. Parameters per layer are $3H(D + H) + 6H$; memory under autograd is $O(T L B H)$.
- **Why torch's reset placement?** It is the form torch and cuDNN compute, so it is the only form in which torch checkpoints are correct. It is also the faster form: the recurrent GEMM `b` does not depend on $r$, so it is computed once for all three gates.
- **Why compute `a` and `b` separately instead of one $z$ like the LSTM?** The LSTM adds input and recurrent parts for all four gates, so one sum suffices. The GRU's candidate needs the recurrent part on its own (to scale it by $r$), so the two halves must stay separate until the gates are formed.
- **Why no forget-style bias?** The interpolation already keeps half of $h$ at $z = 0.5$; there is no separate memory that would be wiped at the start. Zero biases match the contract and torch-style training.
- **GRU or LSTM?** Similar quality on most sequence tasks (Chung et al. 2014), with fewer parameters and one state to carry; the LSTM's separate cell can help on very long dependencies. `L4.1` takes the GRU for its size and speed.
- **Tradeoffs accepted.** As for the LSTM: Python loops, masking rather than packing, no fused kernel.

### 5.4 Explain it back

1. In section 3, where does each of the two terms of $\partial h'/\partial h = 0.75 + 0.0746$ come from?
   <details><summary>Answer</summary>$0.75 = z$ is the direct path through $z \odot h$. $0.0746 = (1 - z)(1 - n^2)\, r\, W_{hn} = 0.25 \times 0.596586 \times 0.5 \times 1$ is the path through the candidate. The gates do not read $h$ here ($W_{hh}$ is zero in the $r$ and $z$ blocks), so there are no further terms.</details>
2. Why does Cho's form give $\tanh(1.25)$ instead of $\tanh(0.75)$ on section 3's weights?
   <details><summary>Answer</summary>Cho's form computes $W_{hn}(r \odot h) + b_n = 1 \times 0.25 + 1 = 1.25$: the bias is outside the reset. torch's form computes $r \odot (W_{hn} h + b_{hn}) = 0.5 \times 1.5 = 0.75$: the reset scales the bias too. The only difference is whether $r$ multiplies $b_{hn}$ and where it meets the matmul.</details>
3. What goes wrong with the interpolation swapped, $h' = z \odot n + (1 - z) \odot h$?
   <details><summary>Answer</summary>The meaning of $z$ inverts: a value near 1 now overwrites the state instead of keeping it. Trained from scratch this is just a relabelling, but a torch checkpoint loaded into it forgets what it should keep (section 3 would give 0.601362 instead of 0.533787).</details>
4. Why is a test with $b_{hn} \ne 0$ the one that separates the two reset placements?
   <details><summary>Answer</summary>With $b_{hn} = 0$ and one unit the two forms are identical ($r w h$ either way), and with zero biases they differ only through how $r$ mixes across units, which a small test can miss. With $b_{hn} \ne 0$, torch's form scales the bias by $r$ and Cho's does not, so the candidates differ by $(1 - r) b_{hn}$ inside the tanh even for one unit.</details>

## 6. Pitfalls

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| 1. the gates in the order $z, r$ | trains from scratch; torch weights give garbage | slices `a` and `b` as $r, z, n$ |
| 2. the reset gate in Cho's position, or applied to $W_{hn}h$ but not to $b_{hn}$ | close to torch, never equal; a loaded checkpoint loses accuracy for no visible reason | `r * b[..., 2H:3H]`, where `b` already includes `b_hh` |
| 3. the interpolation swapped, $z \odot n + (1 - z) \odot h$ | the update gate's meaning inverted: a torch checkpoint forgets what it should keep | returns `(1.0 - z) * n + z * h` |
| 4. padding leaking into the state or the outputs | a short sentence's encoding depends on its batch-mates | `F.where` on the state and the output with the `_mask` |
| 5. a sigmoid candidate | $n$ cannot be negative; the state drifts to positive values | `n = F.tanh(...)` |
| 6. stacking mistakes: dropout after the last layer, layer $k$ starting from layer $k - 1$'s state | 1-layer train and eval differ; 2-layer outputs differ from torch | `if k < L - 1` around dropout; `h = h0[k]` |
| 7. part of the step computed on raw arrays (`h.data`), so $z \odot h$ carries no gradient | the forward is right and training is quietly worse: the state's own path back through time is cut | the whole cell is `F` ops on the `h` Tensor |

## 7. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `L0.1` | parameters are Tensors; `x[t]` and `h0[k]` are getitem |
| Back | `L0.2` | `F.matmul`, `F.transpose`, `F.sigmoid`, `F.tanh`, `F.where`, `F.stack` |
| Back | `L0.4` | `Module` registration order is the key order; `Dropout` between layers |
| Back | `M07.3` | `xavier_uniform` for the input blocks |
| Back | `M03.3` | `orthogonal_init` for the recurrent blocks |
| Back | `M06.3` | `PCG32(0).substream("init")` when no `rng` is given |
| Forward | `L3.4` | `bidirectional(GRU, GRU, x, lengths)`, the encoder of `L4.1` |
| Forward | `L3.6` | `RNNLM(cell='gru')`, compared with the vanilla RNN and the LSTM on the same text |
| Forward | `L4.1` | the default encoder and decoder cell of the seq2seq model |

## Going further

| This piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `gru_cell` | cuDNN's GRU (`CUDNN_GRU`) | the "linear before reset" form the solution implements, chosen so the recurrent GEMM runs before the gates | NVIDIA cuDNN API reference, `cudnnRNNMode_t` |
| `GRU` | `torch.nn.GRU` | packed sequences, bidirectional layers, projection sizes | `torch/nn/modules/rnn.py` |
| the original form | Keras `GRU(reset_after=False)` | Cho's placement, kept for old checkpoints; `reset_after=True` is the cuDNN form | `keras/layers/rnn/gru.py` |
| gating | minGRU (2024) | gates that read only the input, so the recurrence becomes a parallel scan | Feng et al., "Were RNNs All We Needed?" |
