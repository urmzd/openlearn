# LSTM (torch gate order)

## Overview

| | |
|---|---|
| **Module** | `L3.2` · build · Python · Pass 4 |
| **Solution** | [`lstm.py`](../../../../course/solutions/python/tinyllm/rnn/lstm.py): `lstm_cell`, `LSTMCell`, `LSTM` (stacked layers, inter-layer dropout, `lengths` for padded batches), with `_init` and `_mask` · model tests [`test_lstm.py`](../../../../course/solutions/learner-tests/L3.2/python/tests/l3-2-lstm/test_lstm.py) |
| **Contract** | [`course/contracts/py/tinyllm/rnn/lstm.pyi`](../../../../course/contracts/py/tinyllm/rnn/lstm.pyi) |
| **Needs** | `L0.1` `Tensor` · `L0.2` the ops the cell is written in · `L0.4` `Module`, `load_state_dict`, `Dropout` · `M07.3` `xavier_uniform` · `M03.3` `orthogonal_init` · `M06.3` the default PCG32 stream · reading: `M01.3` sigmoid and tanh, `L3.1` BPTT, `craft.03` |
| **Used by** | `L3.4` two LSTMs make a bidirectional layer · later: `L3.6` `cell='lstm'`, whose language model must beat the vanilla RNN on the same text, and the `L4.1` encoder option |
| **Optional depth** | Hochreiter and Schmidhuber, "Long Short-Term Memory" (1997); Gers, Schmidhuber, Cummins, "Learning to Forget" (2000); Jozefowicz, Zaremba, Sutskever, "An Empirical Exploration of Recurrent Network Architectures" (2015); Olah, *Understanding LSTM Networks* (2015) |

## Key Takeaways

- An LSTM keeps a second state, the cell $c_t$, updated **additively**: $c_t = f_t \odot c_{t-1} + i_t \odot g_t$. Along it the gradient is multiplied only by the forget gate, so with $f$ near 1 it survives many steps.
- The four gate blocks are stacked in torch's order $i, f, g, o$, with two biases, so torch weights load by name and give torch's outputs and gradients.
- Initialization matters at step 0: a forget bias of 1 makes the cell remember by default, and orthogonal recurrent blocks keep the state's scale.
- In a padded batch each sequence stops at its own length: its outputs past it are 0 and its final state is the state after its last real token.

## How to read this chapter

Sections 1 and 2 derive the LSTM from the vanishing gradient of `L3.1`, then fix torch's layout, the initialization, stacking, and padding. Section 3 runs one cell step and its cell-path gradient by hand. Section 4 fixes the interface. Section 5 walks through the solution and explains why each line is the right one. Try to predict each step of the solution before reading it.

---

## 1. Why now

`L3.1` showed the vanilla RNN's problem in numbers: the gradient into a state 30 steps back is the late gradient times $W_{hh}^\top$ thirty times, and with $\rho(W_{hh}) = 0.8$ that is a factor of about $10^{-3}$. The vanilla RNN language model (`L3.6`) learns which character comes next but not that a quote opened a line ago must close. The LSTM fixes this with one structural change, a memory cell updated by addition instead of by a squashing matmul, plus three gates that decide what to write, what to keep, and what to show. It is also the first module whose weights must match a framework's layout exactly: torch-trained LSTMs are everywhere, and the solution loads torch's weights by name. With autograd from Part 0 the solution writes only the forward; the backward through time comes for free.

## 2. Principles

| Symbol | Meaning | Type / shape |
|---|---|---|
| $x_t$ | input at step $t$ | `float32[B, D]` |
| $h_t$, $c_t$ | hidden state (what the next layer sees) and cell state (the memory) | `float32[B, H]` each |
| $W_{ih}$, $W_{hh}$ | input and recurrent weights, four gate blocks stacked | `[4H, D]`, `[4H, H]` |
| $b_{ih}$, $b_{hh}$ | the two biases (torch keeps both) | `[4H]` each |
| $z_t$ | all four pre-activations, $x_t W_{ih}^\top + b_{ih} + h_{t-1} W_{hh}^\top + b_{hh}$ | `[B, 4H]` |
| $i_t, f_t, g_t, o_t$ | input gate, forget gate, candidate, output gate | `[B, H]` each |
| $\sigma(v) = 1/(1 + e^{-v})$ | the logistic sigmoid, values in $(0, 1)$ | function |
| $\odot$ | elementwise product | |
| $\ell_b$ | the length of sequence $b$ in a padded batch | integer in $1..T$ |

**The cell.** One matmul per weight computes every gate at once, and the result is split into four blocks of $H$ columns, **in torch's order**:

$$i = \sigma(z_{[0:H]}),\quad f = \sigma(z_{[H:2H]}),\quad g = \tanh(z_{[2H:3H]}),\quad o = \sigma(z_{[3H:4H]})$$
$$c_t = f \odot c_{t-1} + i \odot g, \qquad h_t = o \odot \tanh(c_t).$$

The gates are sigmoids because they are fractions: how much of the old memory to keep ($f$), how much of the candidate to write ($i$), how much of the memory to show ($o$). The candidate $g$ is a tanh because it is content, centered at 0 and able to subtract. The output goes through one more tanh so $h$ stays in $(-1, 1)$ however large $c$ grows.

**Why it does not vanish.** Differentiate the cell update: $\partial c_t/\partial c_{t-1} = \operatorname{diag}(f_t)$, plus terms through the gates' dependence on $h_{t-1}$. Along the cell path the gradient from $c_T$ to $c_0$ is $\prod_t f_t$, elementwise, with no weight matrix and no tanh slope in it. If the network wants to remember, it sets $f \approx 1$ and the gradient flows back almost unchanged (Hochreiter's "constant error carousel"). With $W_{hh} = 0$ the cell path is the only path, and $\partial c_T / \partial c_0$ is exactly $\prod_t f_t$.

**torch's layout.** `torch.nn.LSTMCell` and `torch.nn.LSTM` store `weight_ih` $[4H, D]$ and `weight_hh` $[4H, H]$ (one row per gate unit, so the products use the transpose, like `L0.4`'s `Linear`), and two biases `bias_ih` and `bias_hh`. The two biases only ever appear as a sum, so a single bias is mathematically enough, but the checkpoint has both keys and both receive the same gradient. A stacked `LSTM` names its parameters per layer: `weight_ih_l0, weight_hh_l0, bias_ih_l0, bias_hh_l0, weight_ih_l1, ...`, registered in that order (`L0.4`: assignment order is `state_dict` order). Layer $k > 0$ reads layer $k - 1$'s output sequence, so its `weight_ih` is $[4H, H]$, and each layer starts from its own slice of the initial state.

**Initialization.** The contract draws, per layer, each of the four $[H, D]$ input blocks with `xavier_uniform` (`M07.3`, gain 1) and each of the four $[H, H]$ recurrent blocks with `orthogonal_init` (`M03.3`): an orthogonal matrix has every singular value 1, so the recurrence neither inflates nor shrinks the state at step 0 (the `L3.1` analysis with $\rho = 1$). The forget-gate block of $b_{ih}$ is 1 and every other bias is 0, so $f$ starts near $\sigma(1) = 0.73$: the cell remembers by default (Gers 2000, Jozefowicz 2015). torch's own default draws everything from $U(-1/\sqrt{H}, 1/\sqrt{H})$; comparisons with torch load torch's weights, so the initializations need not match.

**Stacking and dropout.** torch applies dropout to each layer's output sequence **except the last layer's**, and only in training mode; the dropout here is `L0.4`'s `Dropout`, so `eval()` turns it off.

**Padded batches.** Batching sequences of different lengths pads them to $T$ steps. torch's `pack_padded_sequence` makes the RNN skip padding; the equivalent here is a mask: at a step $t \ge \ell_b$, sequence $b$ keeps its $(h, c)$ unchanged (`F.where(mask, new, old)`) and outputs 0. Then the returned final state $(h_n, c_n)$ is each sequence's state after its own last real step, and nothing the padding contains reaches any output or receives any gradient.

## 3. Worked example by hand

One unit ($D = H = B = 1$), $x = 1$, $h_{t-1} = 0$, $c_{t-1} = 0.5$. Weights: $W_{ih} = [0, 0, \ln 2, 0]^\top$ (only the candidate reads $x$), $W_{hh} = 0$, $b_{ih} = [0, \ln 3, 0, \ln 3]$, $b_{hh} = 0$. So $z = [0, \ln 3, \ln 2, \ln 3]$.

1. $i = \sigma(0) = 0.5$; $f = \sigma(\ln 3) = 3/4 = 0.75$; $g = \tanh(\ln 2) = (4 - 1)/(4 + 1) = 0.6$; $o = \sigma(\ln 3) = 0.75$.
2. $c_t = 0.75 \times 0.5 + 0.5 \times 0.6 = 0.375 + 0.3 = 0.675$.
3. $h_t = 0.75 \times \tanh(0.675) = 0.75 \times 0.588259 = 0.441194$.
4. Backward from $\partial L/\partial h_t = 1$: $\partial h_t/\partial c_t = o\,(1 - \tanh^2 c_t) = 0.75 \times 0.653952 = 0.490463$, and through the cell path $\partial c_t/\partial c_{t-1} = f = 0.75$, so $\partial L/\partial c_{t-1} = 0.367847$.

With the gate order $i, f, o, g$ the same weights give $g = \tanh(\ln 3) = 0.8$ and $o = \sigma(\ln 2) = 2/3$, so $c_t = 0.775$ and $h_t = 0.433$: one swapped block, every number different.

This is the first hand case of section 4.

## 4. The interface

```python
# python/tinyllm/rnn/lstm.py
def lstm_cell(x, h, c, w_ih, w_hh, b_ih, b_hh) -> tuple[Tensor, Tensor]       # (h', c')
class LSTMCell(Module):   # weight_ih [4H, D], weight_hh [4H, H], bias_ih, bias_hh [4H]
    def __init__(self, d_in, d_h, rng=None); def forward(self, x, state=None)
class LSTM(Module):       # weight_ih_l{k}, weight_hh_l{k}, bias_ih_l{k}, bias_hh_l{k}
    def __init__(self, d_in, d_h, num_layers=1, dropout=0.0, rng=None)
    def forward(self, x, state=None, lengths=None) -> tuple[Tensor, tuple[Tensor, Tensor]]
    # x [T, B, d_in] -> out [T, B, H], (h_n, c_n) [num_layers, B, H]
```

### What a correct solution must handle

| Case | Kind of check | Checks | Why it matters downstream |
|---|---|---|---|
| The hand example cell | unit | section 3: $c' = 0.675$, $h' = 0.441194$, $\partial L/\partial c = 0.367847$ | the gates are in the right order with the right squashing |
| Matches torch's cell | golden | `torch.nn.LSTMCell` outputs and every gradient | torch weights load and run |
| Matches torch's sequence | golden | one layer with a state, two layers, a packed padded batch: outputs, $(h_n, c_n)$, all gradients | `L3.6` and `L4.1` load torch checkpoints |
| State-dict names match torch | golden | torch's parameter names, order, and shapes | the safetensors key contract (`L0.6`) |
| Gradcheck of the cell | gradcheck | autograd through the cell against central differences, float64 | the backward is right without torch |
| The cell state is a gradient highway | property | with $W_{hh} = 0$, $\partial c_T/\partial c_0 = \prod_t f_t$ | the reason the LSTM remembers |
| Forget bias and orthogonal recurrence | unit | forget bias 1, others 0; orthogonal $W_{hh}$ blocks; Xavier bound; seeding | trainable from step 0 (`L3.6`) |
| Lengths keep padding out | boundary | padded outputs 0; final state equals running the sequence alone; padding gets no gradient | batches of sentences (`L4.1`) |
| Dropout between layers only | unit | dropout between layers, in training mode only | evaluation is deterministic |
| Default state and bad arguments | boundary | no state means zeros; bad shapes, lengths, sizes raise | errors at the call site |

The model tests in [`test_lstm.py`](../../../../course/solutions/learner-tests/L3.2/python/tests/l3-2-lstm/test_lstm.py) start from section 3 (`lstm_cell` with $W_{ih} = [0, 0, \ln 2, 0]^\top$ and $b_{ih} = [0, \ln 3, 0, \ln 3]$ must give $c' = 0.675$ and $h' = 0.75 \tanh 0.675$), then compare the cell with a numpy formula under random weights and two nonzero biases, check the `state_dict` keys of a 2-layer LSTM, the forget bias and orthogonal blocks, a padded batch, a 2-layer LSTM against stacked numpy cells from a nonzero $(h_0, c_0)$, dropout ignored by a 1-layer LSTM, and bad arguments.

## 5. The solution, step by step

### 5.1 From first principles

1. **Give the gradient a path without a squashing matmul.** In `L3.1` the carry is multiplied by $\operatorname{diag}(1 - h^2) W_{hh}^\top$ every step. A state updated as $c_t = f \odot c_{t-1} + (\text{new content})$ has $\partial c_t / \partial c_{t-1} = \operatorname{diag}(f)$ along that path, so the network controls the decay directly through $f$.
2. **Each decision is a fraction or a content.** Keep, write, and show are amounts in $(0, 1)$: sigmoids. The candidate is signed content: a tanh. $h = o \odot \tanh(c)$ keeps the visible state bounded however large $c$ grows.
3. **Compute all gates with one matmul per weight.** The four gates read the same $x_t$ and $h_{t-1}$, so stacking their weights into $[4H, \cdot]$ and slicing the result is four matmuls in one, and fixes a block order. Matching torch means torch's order ($i, f, g, o$), torch's orientation ($[4H, D]$, used transposed), and torch's two biases.
4. **Autograd does the backward.** Everything is `F.matmul`, `F.transpose`, slicing, `F.sigmoid`, `F.tanh`, and elementwise ops, all with VJPs from `L0.2`, so BPTT through any number of steps and layers comes from the graph.
5. **Initialise for step 0.** Orthogonal recurrent blocks have every singular value 1, so the state's scale is preserved; a forget bias of 1 starts $f$ at $\sigma(1) \approx 0.73$ so memory is kept by default.
6. **Padding must be invisible.** Past its length, a sequence must neither change its state nor emit output, and nothing computed at a padded step may receive gradient. A per-step mask selecting between the new and the old state does all three.

### 5.2 The code, annotated

**One layer's parameters.** Drawn block by block from one generator in the contract's order: four Xavier input blocks, then four orthogonal recurrent blocks.

```python
def _init(d_in, d_h, rng):
    r = rng if rng is not None else PCG32(0).substream("init")
    w_ih = np.concatenate([xavier_uniform((d_h, d_in), 1.0, r) for _ in range(GATES)])  # [4H, D]
    w_hh = np.concatenate([orthogonal_init((d_h, d_h), 1.0, r) for _ in range(GATES)])  # [4H, H]
    b_ih = np.zeros(GATES * d_h)
    b_ih[d_h : 2 * d_h] = 1.0  # forget gate open at the start: remember by default
    b_hh = np.zeros(GATES * d_h)
    return w_ih, w_hh, b_ih, b_hh
```

Each recurrent block is orthogonal on its own; the stacked $[4H, H]$ matrix is not square, so drawing per block is what gives every gate's recurrence singular values of 1. The forget block is the second slice, $[H, 2H)$, because of the gate order.

**The cell.** Both biases are added (they only ever appear as a sum, but torch has two keys and both must receive gradient), then the four slices in torch's order.

```python
def lstm_cell(x, h, c, w_ih, w_hh, b_ih, b_hh):
    z = (F.matmul(x, F.transpose(w_ih, 0, 1)) + b_ih
         + F.matmul(h, F.transpose(w_hh, 0, 1)) + b_hh)     # [B, 4H]
    H = z.shape[-1] // GATES
    i = F.sigmoid(z[..., 0:H])               # input gate: how much to write
    f = F.sigmoid(z[..., H : 2 * H])         # forget gate: how much to keep
    g = F.tanh(z[..., 2 * H : 3 * H])        # candidate: signed content
    o = F.sigmoid(z[..., 3 * H : 4 * H])     # output gate: how much to show
    c_new = f * c + i * g                    # additive: dc_new/dc = f, no matmul, no tanh slope
    return o * F.tanh(c_new), c_new          # h stays in (-1, 1)
```

On section 3, $z = [0, \ln 3, \ln 2, \ln 3]$, so $(i, f, g, o) = (0.5, 0.75, 0.6, 0.75)$, $c' = 0.675$, $h' = 0.441194$. The `h` and `c` arguments may be plain arrays (a zero start state); `F` ops treat them as constants.

**The single cell module.** Torch's names, registered in torch's order, so `state_dict` matches `torch.nn.LSTMCell`.

```python
w_ih, w_hh, b_ih, b_hh = _init(d_in, d_h, rng)
self.weight_ih = _param(w_ih)     # float32, requires_grad
self.weight_hh = _param(w_hh)
self.bias_ih = _param(b_ih)
self.bias_hh = _param(b_hh)
```

**The padding mask.** A boolean `[T, B, 1]` that is true at real steps; the trailing axis broadcasts over $H$.

```python
def _mask(lengths, T, B):
    if lengths is None:
        return None
    n = np.asarray(lengths)
    # ... B integers, each in 1..T ...
    return (np.arange(T)[:, None] < n[None, :])[:, :, None]   # step t of sequence b is real
```

**The stacked LSTM.** Layers are created with one shared generator and registered as `weight_ih_l{k}, weight_hh_l{k}, bias_ih_l{k}, bias_hh_l{k}`; layer $k > 0$ reads $H$ inputs.

```python
r = rng if rng is not None else PCG32(0).substream("init")
for k in range(num_layers):
    w_ih, w_hh, b_ih, b_hh = _init(d_in if k == 0 else d_h, d_h, r)
    # torch's names, in torch's order: the safetensors key contract.
    setattr(self, f"weight_ih_l{k}", _param(w_ih))
    setattr(self, f"weight_hh_l{k}", _param(w_hh))
    setattr(self, f"bias_ih_l{k}", _param(b_ih))
    setattr(self, f"bias_hh_l{k}", _param(b_hh))
self.drop = Dropout(dropout)
```

The forward runs layer by layer over the whole sequence. Each layer starts from its own slice `h0[k], c0[k]`, and the mask keeps a finished sequence's state while zeroing its output.

```python
mask = _mask(lengths, T, B)
seq = x
h_n, c_n = [], []
for k in range(L):
    w = [getattr(self, f"{n}_l{k}") for n in ("weight_ih", "weight_hh", "bias_ih", "bias_hh")]
    h, c = h0[k], c0[k]                       # this layer's own initial state
    outs = []
    for t in range(T):
        h_new, c_new = lstm_cell(seq[t], h, c, *w)
        if mask is None:
            h, c = h_new, c_new
            outs.append(h_new)
        else:
            # Past its length a sequence keeps its state and outputs 0.
            m = mask[t]
            h = F.where(m, h_new, h)
            c = F.where(m, c_new, c)
            outs.append(F.where(m, h_new, 0.0))
    seq = F.stack(outs, axis=0)               # [T, B, H]: the next layer's input
    if k < L - 1:
        seq = self.drop(seq)                  # between layers only, and only in train()
    h_n.append(h)
    c_n.append(c)
return seq, (F.stack(h_n, axis=0), F.stack(c_n, axis=0))
```

`F.where` routes the gradient only to the branch it selected, so a padded step's `h_new` and `c_new`, and the padded inputs behind them, receive zero gradient. Because $\ell_b \ge 1$ and the mask is monotone in $t$, once a sequence ends its `h, c` are frozen at the last real step, which is the final state returned.

Full solution: [`lstm.py`](../../../../course/solutions/python/tinyllm/rnn/lstm.py).

### 5.3 Why this solution

- **Complexity.** Per step and layer, two matmuls of cost $O(B \cdot 4H \cdot (D + H))$, four times a vanilla RNN of the same $H$; parameters are $4H(D + H) + 8H$ per layer. Autograd keeps every intermediate of every step, $O(T L B H)$ memory.
- **Why autograd here when `L3.1` wrote the backward?** The LSTM's backward is long (four gates, two states, two biases) and error-prone; autograd gets it from tested primitives, and the gradcheck and torch comparison confirm it. `L3.1`'s manual backward was about seeing the carry; here the lesson is the cell path.
- **Why torch's layout?** Checkpoints are interchangeable only if names, shapes, gate order, and the two biases match. A single bias computes the same function, but a torch checkpoint then has an extra key and strict loading fails or drops half the bias.
- **Why masking instead of packing?** `pack_padded_sequence` saves compute by shrinking the batch; masking computes padded steps and discards them. Masking is simpler, keeps shapes fixed, and gives the same outputs, states, and gradients.
- **Why forget bias 1 and orthogonal blocks?** They make the network trainable from step 0: without them the cell halves its memory each step at the start ($\sigma(0) = 0.5$) and long dependencies are learned late or never.
- **Tradeoffs accepted.** Python loops over time and layers; float32 parameters (the gradcheck uses float64 inputs); no fused kernel.

### 5.4 Explain it back

1. In section 3, $\partial L / \partial c_{t-1} = 0.367847$. Which two factors make it, and why is there no weight matrix in it?
   <details><summary>Answer</summary>$\partial h_t / \partial c_t = o (1 - \tanh^2 c_t) = 0.490463$ and, along the cell path, $\partial c_t / \partial c_{t-1} = f = 0.75$. The cell update $c_t = f \odot c_{t-1} + i \odot g$ multiplies $c_{t-1}$ only by $f$ (and $W_{hh} = 0$ here, so the gates do not depend on $h_{t-1}$), so no matrix appears.</details>
2. Why does $\tanh$ appear on $c_t$ for the output but must not appear on $c_{t-1}$ in the update?
   <details><summary>Answer</summary>The output tanh bounds $h$, which feeds matmuls. Squashing the old cell ($f \odot \tanh c_{t-1}$) would put a factor $1 - \tanh^2$ on the cell path at every step, which is exactly the vanishing factor the LSTM was built to remove.</details>
3. A model trained with gate order $i, f, o, g$ is loaded as $i, f, g, o$. Why does it not fail loudly?
   <details><summary>Answer</summary>The shapes are identical, so every key loads. The third and fourth blocks swap roles: the candidate goes through a sigmoid and the output gate through a tanh. Every number changes (section 3: $c' = 0.775$ instead of 0.675) but nothing errors; only a golden comparison catches it.</details>
4. Why does `F.where(m, h_new, h)` make padding receive no gradient?
   <details><summary>Answer</summary>The VJP of `where` sends the incoming gradient to the selected branch and zero to the other. At padded steps the old `h` is selected, so `h_new`, the cell computation behind it, and the padded input get zero gradient; the output there is the constant 0.</details>
5. Why is dropout applied to every layer's output except the last?
   <details><summary>Answer</summary>That is torch's convention: dropout regularises what one layer passes to the next. The last layer's output goes to whatever the caller stacks on top, which applies its own regularisation; applying it inside would make a 1-layer LSTM differ between `train()` and `eval()` and differ from torch.</details>

## 6. Pitfalls

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| 1. the gates in another order ($i, f, o, g$ is common in papers and other frameworks) | trains fine from scratch; torch weights load and produce garbage | slices `z` as $i, f, g, o$ |
| 2. the wrong squashing: $g$ through a sigmoid, $h = o \odot c$ without the tanh, the input gate dropped, or the old cell squashed ($f \odot \tanh c$) | the cell can only add, $h$ grows without bound, or the gradient highway gains a tanh slope per step and vanishes again | `g = F.tanh(...)`, `c_new = f * c + i * g`, `h = o * F.tanh(c_new)` |
| 3. forget bias 0, or the $+1$ on the wrong block | the cell forgets half its memory per step at the start; long dependencies learned late or never | `_init` sets `b_ih[d_h:2*d_h] = 1.0`, the forget block |
| 4. padding leaking into the state or the outputs | the final state of a short sentence depends on how long its batch-mates are | `F.where` on `h`, `c`, and the output with the `_mask` |
| 5. one bias instead of two | the model computes the same function but a torch checkpoint has an extra key, and loading it drops half the bias | registers and adds both `bias_ih` and `bias_hh` |
| 6. stacking mistakes: dropout after the last layer, layer $k$ starting from layer $k - 1$'s final state | training and eval outputs differ in a 1-layer model; 2-layer outputs differ from torch | `if k < L - 1` around dropout; `h, c = h0[k], c0[k]` |

## 7. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `L0.1` | parameters are Tensors; `x[t]` and `h0[k]` are getitem |
| Back | `L0.2` | `F.matmul`, `F.transpose`, `F.sigmoid`, `F.tanh`, `F.where`, `F.stack` |
| Back | `L0.4` | `Module` registration order is the key order; `Dropout` between layers |
| Back | `M07.3` | `xavier_uniform` for the input blocks |
| Back | `M03.3` | `orthogonal_init` for the recurrent blocks |
| Back | `M06.3` | `PCG32(0).substream("init")` when no `rng` is given |
| Forward | `L3.4` | `bidirectional(LSTM, LSTM, x, lengths)` |
| Forward | `L3.6` | `RNNLM(cell='lstm')`, trained with stateful TBPTT; must beat the vanilla RNN on the same text |
| Forward | `L4.1` | the seq2seq encoder's LSTM option |

## Going further

| This piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `lstm_cell` | cuDNN's fused LSTM | the input projection for all $T$ steps as one GEMM, the four gates fused into one kernel | `aten/src/ATen/native/cudnn/RNN.cpp` |
| `lengths` masking | `torch.nn.utils.rnn.pack_padded_sequence` | sorts by length and shrinks the batch as sequences end, so no padded step is computed | `torch/nn/utils/rnn.py` |
| forget bias 1 | Keras `unit_forget_bias=True` | the same default, on by default | `keras/layers/rnn/lstm.py` |
| the LSTM itself | xLSTM (2024) | exponential gating and a matrix memory, parallelizable over time | `NX-AI/xlstm` |
