<!-- ol:module M00.2 -->
# Trig, unit circle, 2D rotations, complex numbers, Euler's formula

## Overview

| | |
|---|---|
| **Module** | `M00.2` · build · Python · Pass 2 · 3 to 4 h |
| **You build** | `python/tinyllm/num/rotation.py`: `rotation_matrix`, `rotate_pairs`, `as_complex`, `as_real` |
| **Contract** | [`course/contracts/py/tinyllm/num/rotation.pyi`](../../course/contracts/py/tinyllm/num/rotation.pyi) |
| **Tests** | `course/tests/M00.2/test_rotation.py` (what they check: section 4) |
| **Needs** | nothing to call. Reading: `M00.1` (exponents and $e$), `lang.01` (numpy slicing and broadcasting) |
| **Used by** | `L7.3` RoPE rotates query and key pairs with `rotate_pairs` · `L5.4` sinusoidal positional encoding · `M07.0` Box-Muller normals use $\cos$ and $\sin$ of $2\pi u$ (none is authored yet: section 6) |
| **Milestone** | `MS-P2` (the Pass 2 gate) |
| **Optional depth** | OpenStax, *Precalculus 2e* (free), ch. 5 to 7 (trigonometric functions and identities) and 8.5 (polar form of complex numbers); 3Blue1Brown, "Euler's formula with introductory group theory" (video); Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding" (2021), section 3.2 |

## Key Takeaways

- A point on the **unit circle** at angle $\theta$ (in radians) is $(\cos\theta, \sin\theta)$, so $\cos^2\theta + \sin^2\theta = 1$ (`test_rotation_matrix_hand_values`).
- **Rotating** the plane by $\theta$ is the matrix $R(\theta) = \begin{pmatrix}\cos\theta & -\sin\theta \\ \sin\theta & \cos\theta\end{pmatrix}$. It keeps every length, and **angles add**: $R(a)R(b) = R(a + b)$ (`test_length_is_preserved`, `test_angles_add`).
- Because angles add, the dot product of two rotated vectors depends only on the **difference** of their angles. RoPE turns token positions into angles and gets attention scores that depend only on relative position (`test_dot_product_sees_only_relative_angle`).
- Reading $(x, y)$ as the complex number $x + iy$, **Euler's formula** $e^{i\theta} = \cos\theta + i\sin\theta$ makes the same rotation one multiplication (`test_euler_formula_is_the_rotation`, `test_golden_torch_polar`).
- Pairs are **interleaved**, $(x_{2i}, x_{2i+1})$. Pairing $x_i$ with $x_{i+k}$ instead is a different model with the same shapes (`test_layout_is_interleaved`).

## How to work this chapter

```bash
ol start M00.2              # stubs python/tinyllm/num/rotation.py into your repo
ol tests M00.2              # read the test catalog first: rung R0, you write no tests here
ol check M00.2              # exit code is the verdict
ol diff  M00.2              # after passing: your code against the reference
```

---

## 1. Why now

Your bigram model sees one previous byte, so the order of a text matters to it only through adjacent pairs. The models of Passes 4 and 5 read whole contexts with attention, and attention on its own is blind to order: it scores every pair of positions with a dot product of two vectors, and shuffling the tokens shuffles the scores without changing any of them. "The dog bit the man" and "the man bit the dog" would get the same representation. Two fixes in the course add position back, and both are rotations. `L5.4`, the 2017 Transformer's sinusoidal encoding, writes $\sin$ and $\cos$ of position-dependent angles into the input. `L7.3`, RoPE, the scheme in Llama and SmolLM2, rotates each pair of coordinates of every query and key by an angle proportional to its position. This module builds rotations from the unit circle up, proves the property RoPE depends on, and gives you `rotate_pairs`, which `L7.3` calls. `M07.0` uses the same $\cos$ and $\sin$ to turn uniform random numbers into normal ones.

## 2. Principles

| Symbol | Meaning | Type / shape |
|---|---|---|
| $\theta, a, b$ | angles in radians | `float` |
| $\pi$ | half the circumference of a circle of radius 1, $3.14159\ldots$ | `math.pi` |
| $\cos\theta, \sin\theta$ | the coordinates of the point at angle $\theta$ on the unit circle | `float` |
| $u, v$ | vectors in the plane, $u = (u_0, u_1)$ | `float[2]` |
| $\langle u, v\rangle$ | dot product $u_0 v_0 + u_1 v_1$ | `float` |
| $\lVert u\rVert$ | length $\sqrt{u_0^2 + u_1^2}$ | `float` |
| $R(\theta)$ | the $2 \times 2$ rotation matrix by $\theta$ | `float[2, 2]` |
| $i$ | the imaginary unit, $i^2 = -1$ | `1j` |
| $z = x + iy$ | a complex number with real part $x$ and imaginary part $y$ | `complex` |
| $\bar z$ | the conjugate $x - iy$ | `complex` |
| $\lvert z\rvert$ | the modulus $\sqrt{x^2 + y^2}$ | `float` |
| $k$ | the number of pairs in a vector of length $2k$ | `int` |
| $m, n$ | token positions | `int` |
| $\omega$ | an angular frequency: radians per position | `float` |

### 2.1 Angles in radians

Walk along the circle of radius 1 centred at the origin, starting at $(1, 0)$ and going counterclockwise. The **radian** measure of an angle is the distance walked. A full turn is the whole circumference, $2\pi$; a half turn is $\pi$; a right angle is $\pi/2$. Degrees convert by $180° = \pi$, so $60° = \pi/3$. Every function in this module, and numpy's `np.cos` and `np.sin`, takes radians. Negative angles walk clockwise, and adding $2\pi$ lands on the same point, so trigonometric functions repeat with **period** $2\pi$.

### 2.2 The unit circle

The point reached at angle $\theta$ has coordinates $(\cos\theta, \sin\theta)$: that is the definition of cosine and sine. Because the point is at distance 1 from the origin, Pythagoras gives

$$\cos^2\theta + \sin^2\theta = 1 .$$

Reflection across the horizontal axis gives $\cos(-\theta) = \cos\theta$ and $\sin(-\theta) = -\sin\theta$. A few values come from familiar triangles:

| $\theta$ | 0 | $\pi/6$ | $\pi/4$ | $\pi/3$ | $\pi/2$ | $\pi$ |
|---|---|---|---|---|---|---|
| $\cos\theta$ | 1 | $\sqrt3/2$ | $\sqrt2/2$ | $1/2$ | 0 | $-1$ |
| $\sin\theta$ | 0 | $1/2$ | $\sqrt2/2$ | $\sqrt3/2$ | 1 | 0 |

### 2.3 Rotations of the plane

Rotating the plane by $\theta$ about the origin is **linear**: rotating a sum is the sum of the rotations, and rotating $c\,u$ is $c$ times the rotation of $u$. So it is enough to know where the two basis vectors go. $(1, 0)$ goes to $(\cos\theta, \sin\theta)$ by definition, and $(0, 1)$, a quarter turn further, goes to $(\cos(\theta + \pi/2), \sin(\theta + \pi/2)) = (-\sin\theta, \cos\theta)$. Those images are the columns of the matrix:

$$R(\theta) = \begin{pmatrix}\cos\theta & -\sin\theta \\ \sin\theta & \cos\theta\end{pmatrix}, \qquad R(\theta)\begin{pmatrix}x\\y\end{pmatrix} = \begin{pmatrix}x\cos\theta - y\sin\theta\\ x\sin\theta + y\cos\theta\end{pmatrix}.$$

A rotation keeps lengths. Expanding, $(x\cos\theta - y\sin\theta)^2 + (x\sin\theta + y\cos\theta)^2 = (x^2 + y^2)(\cos^2\theta + \sin^2\theta) = x^2 + y^2$, because the cross terms $\mp 2xy\cos\theta\sin\theta$ cancel. It also keeps dot products: $\langle R u, R v\rangle = \langle u, v\rangle$. In matrix terms $R(\theta)^\top R(\theta) = I$, and $R(\theta)^\top = R(-\theta)$ undoes the rotation.

### 2.4 Composition: angles add

Rotating by $b$ and then by $a$ is a rotation by $a + b$, so $R(a)R(b) = R(a + b)$. Multiplying the two matrices out and comparing the first column with $R(a + b)$'s gives the **angle-addition formulas**:

$$\cos(a + b) = \cos a\cos b - \sin a\sin b, \qquad \sin(a + b) = \sin a\cos b + \cos a\sin b .$$

With $b = -a$: $R(a)R(-a) = R(0) = I$.

### 2.5 Why RoPE rotates pairs

Rotate $u$ by $\alpha$ and $v$ by $\beta$, then take the dot product. Using $\langle R u, w\rangle = \langle u, R^\top w\rangle$ and composition,

$$\langle R(\alpha)u,\ R(\beta)v\rangle = \langle u,\ R(-\alpha)R(\beta)v\rangle = \langle u,\ R(\beta - \alpha)v\rangle .$$

The result depends on $\alpha$ and $\beta$ only through $\beta - \alpha$. RoPE sets $\alpha = m\omega$ for a query at position $m$ and $\beta = n\omega$ for a key at position $n$, so the attention score $\langle R(m\omega)q, R(n\omega)k\rangle$ depends only on $n - m$, the distance between the tokens: shifting a sentence along the context changes nothing. One frequency $\omega$ would make positions $2\pi/\omega$ apart indistinguishable, so RoPE splits a $2k$-dimensional query into $k$ pairs and rotates pair $i$ by $m\omega_i$ with a ladder of frequencies $\omega_i$ (built in `M00.3`). `rotate_pairs(x, theta)` is exactly that operation, with `theta[..., i]` the angle of pair $i$, and pair $i$ is $(x_{2i}, x_{2i+1})$, the **interleaved** layout of the RoFormer paper and Meta's Llama code. Hugging Face's Llama pairs $x_i$ with $x_{i+k}$ instead (the "half" layout of `rotate_half`); `L7.3` implements both and converts between them.

### 2.6 Complex numbers and Euler's formula

A **complex number** $z = x + iy$ is a point $(x, y)$ of the plane with a multiplication: expand as polynomials in $i$ and replace $i^2$ by $-1$.

$$(x + iy)(c + is) = (xc - ys) + i(xs + yc) .$$

Compare with section 2.3: with $c = \cos\theta$ and $s = \sin\theta$, the right-hand side is exactly $R(\theta)(x, y)$. **Multiplying by $\cos\theta + i\sin\theta$ rotates by $\theta$.** The modulus $\lvert z\rvert = \sqrt{x^2 + y^2}$ is the length, and $z\bar z = \lvert z\rvert^2$.

Euler's formula names that rotating number:

$$e^{i\theta} = \cos\theta + i\sin\theta .$$

Why an exponential? Because it obeys the law of exponents from `M00.1`: $e^{ia}e^{ib} = e^{i(a + b)}$ is the composition rule of section 2.4, now one line. `M02.1` proves the formula by comparing the Taylor series of $e^x$, $\cos x$, and $\sin x$. Consequences used in the solve set: $e^{i\pi} = -1$, $e^{i\pi/2} = i$, $(e^{i\theta})^n = e^{in\theta}$ (De Moivre), $\cos\theta = (e^{i\theta} + e^{-i\theta})/2$, and $\operatorname{Re}(e^{ia}\,\overline{e^{ib}}) = \cos(a - b)$, the relative-angle property of section 2.5 in complex form.

`as_complex` reads a real vector of length $2k$ as $k$ complex numbers $z_i = x_{2i} + i\,x_{2i+1}$, the numpy analogue of `torch.view_as_complex`; `as_real` reads them back. With them, `rotate_pairs(x, t)` equals `as_real(as_complex(x) * np.exp(1j * t))`.

## 3. Worked example by hand

**Two pairs, two angles.** $x = [3, 4, 1, 0]$ has pairs $(3, 4)$ and $(1, 0)$. Rotate pair 0 by the angle $t_0$ with $\cos t_0 = 3/5$ and $\sin t_0 = 4/5$ (so $t_0 = \operatorname{atan2}(4, 3) \approx 0.9273$ radians, about $53.13°$), and pair 1 by $\pi/2$.

| pair | $(x, y)$ | $\cos, \sin$ | $x\cos - y\sin$ | $x\sin + y\cos$ |
|---|---|---|---|---|
| 0 | $(3, 4)$ | $0.6, 0.8$ | $1.8 - 3.2 = -1.4$ | $2.4 + 2.4 = 4.8$ |
| 1 | $(1, 0)$ | $0, 1$ | $0 - 0 = 0$ | $1 + 0 = 1$ |

So `rotate_pairs([3, 4, 1, 0], [t0, pi/2])` is $[-1.4, 4.8, 0, 1]$ (`test_hand_example_rotation`; numpy gives $6 \times 10^{-17}$ instead of the exact 0, because $\cos(\pi/2)$ in floating point is not exactly 0). Length check: $\sqrt{1.96 + 23.04} = 5 = \sqrt{9 + 16}$.

**The same with complex numbers.** `as_complex([3, 4, 1, 0])` is $[3 + 4i,\ 1 + 0i]$. Then

$$(3 + 4i)(0.6 + 0.8i) = 1.8 + 2.4i + 2.4i + 3.2\,i^2 = -1.4 + 4.8i, \qquad (1 + 0i)\cdot i = i ,$$

and `as_real` returns $[-1.4, 4.8, 0, 1]$: `test_hand_example_as_complex_product`.

**Composition.** $R(\pi/2)R(\pi/3)$: multiply $\begin{pmatrix}0 & -1\\ 1 & 0\end{pmatrix}\begin{pmatrix}1/2 & -\sqrt3/2\\ \sqrt3/2 & 1/2\end{pmatrix} = \begin{pmatrix}-\sqrt3/2 & -1/2\\ 1/2 & -\sqrt3/2\end{pmatrix}$, whose first column $(\cos, \sin) = (-\sqrt3/2, 1/2)$ is the point at $5\pi/6 = \pi/2 + \pi/3$.

## 4. The interface

```python
# python/tinyllm/num/rotation.py
def rotation_matrix(theta: float) -> NDArray: ...           # [[cos, -sin], [sin, cos]], float64
def rotate_pairs(x: ArrayLike, theta: ArrayLike) -> NDArray: ...  # pair i of the last axis by theta[..., i]
def as_complex(x: ArrayLike) -> NDArray: ...                # [..., 2k] -> complex [..., k]
def as_real(z: ArrayLike) -> NDArray: ...                   # complex [..., k] -> [..., 2k]
```

`theta` broadcasts to `x.shape[:-1] + (k,)`: a `[k]` theta rotates every row alike, a `[T, k]` theta gives each position its own angles (what `L7.3` passes). The result is a new array; float32 input stays float32, everything else comes back float64. An odd last axis or a theta that does not broadcast raises `ValueError`.

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `test_hand_example_rotation` | unit, smoke | section 3: $[3, 4, 1, 0]$ becomes $[-1.4, 4.8, 0, 1]$ | you and the tests agree on direction and layout |
| `test_hand_example_as_complex_product` | unit, smoke | $(3 + 4i)(0.6 + 0.8i) = -1.4 + 4.8i$ through `as_complex` and `as_real` | the complex view of the same example |
| `test_rotation_matrix_hand_values` | unit | $R(\pi/2)$, $R(t_0)$, $R(0) = I$ | the columns are the images of the basis vectors |
| `test_rotate_pairs_matches_rotation_matrix` | differential | each pair equals $R(\theta_i)$ times the pair | your two functions agree |
| `test_length_is_preserved` | property | every pair keeps its length | RoPE never rescales queries or keys |
| `test_angles_add` | property | $R(a)R(b) = R(a + b)$, and rotating twice adds the angles | the law behind relative positions |
| `test_negative_angle_undoes` | property | rotating by $-\theta$ returns the input | $R(\theta)^\top = R(-\theta)$ |
| `test_dot_product_sees_only_relative_angle` | property | $\langle R(m\omega)q, R(n\omega)k\rangle$ is unchanged when $m$ and $n$ shift together | the RoPE property `L7.3` is built on |
| `test_euler_formula_is_the_rotation` | differential | `as_complex(rotate_pairs(x, t)) == as_complex(x) * exp(1j t)` | Euler's formula, checked on random data |
| `test_layout_is_interleaved` | boundary | pairs are $(x_{2i}, x_{2i+1})$, not halves | the layout bug that crashes nothing |
| `test_as_real_roundtrip` | property | `as_real(as_complex(x)) == x` exactly, float32 stays complex64 | lossless views |
| `test_golden_torch_polar` | golden | three cases against `torch.view_as_complex(x) * torch.polar(1, theta)`, one in float32 | the exact op Meta's Llama code runs |
| `test_theta_broadcasts` | unit | a `[k]` theta applies to every row of a batch | `L5.4` and `L7.3` shapes |
| `test_dtype_follows_input` | boundary | float32 in, float32 out; integers promoted to float64 | `L7.3` keeps activations in float32 |
| `test_input_not_modified` | boundary | `x` is unchanged after the call | keys are reused for every query |
| `test_odd_or_bad_shapes_rejected` | boundary | odd lengths, bad theta, and 0-d input raise `ValueError` | no silently dropped coordinate |

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| 1. the sign on the wrong $\sin$ term | rotates by $-\theta$; lengths and relative positions still look right, every value is wrong | `test_hand_example_rotation`, `test_golden_torch_polar` (mutant `s01`) |
| 2. pairing halves, $(x_i, x_{i+k})$, instead of neighbours | shapes match, outputs differ from the reference model; with HF weights this is the classic RoPE layout bug | `test_layout_is_interleaved` (mutants `s02`, `s05`) |
| 3. angles in degrees | $\cos(90)$ in radians is $-0.448$; every rotation is by a meaningless angle | `test_hand_example_rotation` (mutant `s03`) |
| 4. rotating in place | the first call is right and corrupts `x`; the second use of the same keys is rotated twice | `test_input_not_modified` (mutant `s04`) |
| 5. computing the second output from the already-rotated first | `y = x' sin + y cos` uses the new $x$; lengths change and nothing matches | `test_length_is_preserved` (mutant `s08`) |
| 6. writing $R(\theta)$ transposed | `rotation_matrix` rotates clockwise; `rotate_pairs` may still be right, so the two disagree | `test_rotation_matrix_hand_values` (mutant `s06`) |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `M00.1` | the exponential and the laws of exponents behind Euler's formula (reading) |
| Back | `lang.01` | numpy strided slicing (`x[..., 0::2]`) and broadcasting (reading) |
| Forward | `L7.3` | RoPE: `rotate_pairs(q, positions[:, None] * inv_freq)` for queries and keys, in both layouts |
| Forward | `L5.4` | the sinusoidal encoding: $\mathrm{PE}(p + j)$ is a fixed rotation of $\mathrm{PE}(p)$, the property its tests check |
| Forward | `M07.0` | Box-Muller: $\sqrt{-2\ln(1 - u_1)}\,\cos(2\pi u_2)$ turns two uniforms into a normal |
| Forward | `M00.3` | builds the frequency ladder $\omega_i$ that RoPE feeds into the angles |

None of the forward modules is authored yet, so `ol verify course M00.2` reports no call site (check 9) until `M07.0` or `L7.3` lands; see `course/DEVIATIONS.md` row M00-01.

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `rotate_pairs` (interleaved) | Meta's Llama reference `apply_rotary_emb` | precomputes $e^{im\omega_i}$ once (`freqs_cis`) and multiplies complex views | `llama/model.py` in `meta-llama/llama` |
| the "half" layout | Hugging Face `apply_rotary_pos_emb` with `rotate_half` | caches `cos` and `sin` per position, permutes the weights instead of the activations | `transformers/models/llama/modeling_llama.py` |
| rotations in Python | vLLM `RotaryEmbedding` | a fused CUDA kernel over queries and keys, `is_neox_style` selects the layout | `vllm/model_executor/layers/rotary_embedding.py` |
| rotations on the CPU | llama.cpp `ggml_rope_ext` | both layouts (`GGML_ROPE_TYPE_NEOX`), context-extension scaling (`L7.4`) | `ggml/src/ggml-cpu/ops.cpp` |
