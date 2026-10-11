"""Recompute every auto-checked answer of S-M12 independently (see _confirm.py).

Output sizes by enumerating window start positions; receptive fields by
propagating index sets through the layers; correlation, convolution, the
Toeplitz matrix, and both gradients by explicit loops (the gradients by
SymPy differentiation of the loss); im2col copy counts by enumeration;
parameter and MAC counts by summing over the weight tensor's index ranges;
strides from the memory-order address formula; aliasing by folding the
frequency into [0, sr/2]; the Keys kernel evaluated in exact rationals;
DFTs as explicit SymPy sums; frame counts by enumerating frame starts;
the Hann COLA sum by summing the shifted squared windows; mel and decibel
values from their formulas; InfoNCE and SigLIP by SymPy; and WER by a
Levenshtein DP over words.

    uv run --project course/harness python course/oracle/solve/S-M12.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import sympy as sp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _confirm import confirm  # noqa: E402

R = sp.Rational


def out_len(n: int, k: int, s: int = 1, p: int = 0, d: int = 1) -> int:
    count, start = 0, -p
    while (
        start + d * (k - 1) <= n - 1 + p
    ):  # the window's last tap inside the padded input
        count += 1
        start += s
    return count


def receptive(layers: list[tuple[int, int]]) -> int:
    # Inputs reaching output 0 of the last layer, walking the layers backwards.
    idx = {0}
    for k, s in reversed(layers):
        idx = {o * s + r for o in idx for r in range(k)}
    return max(idx) - min(idx) + 1


def vec(xs) -> str:
    return "[" + ", ".join(str(sp.nsimplify(v)) for v in xs) + "]"


def edit(a: list[str], b: list[str]) -> int:
    dp = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, len(b) + 1):
            cur = min(dp[j] + 1, dp[j - 1] + 1, prev + (a[i - 1] != b[j - 1]))
            prev, dp[j] = dp[j], cur
    return dp[-1]


def main() -> int:
    c: dict[str, str] = {}
    c["q1.a"] = str(out_len(224, 7, 2, 3))
    c["q1.b"] = str(out_len(out_len(224, 7, 2, 3), 3, 2, 1))
    c["q1.c"] = str(out_len(out_len(3000, 3, 1, 1), 3, 2, 1))
    c["q1.d"] = str(out_len(32, 3, 1, 0, 4))
    k, d = sp.symbols("k d", positive=True)
    span = sp.interpolate(
        [(1, 1), (2, 1 + d), (3, 1 + 2 * d)], k
    )  # taps at 0, d, 2d, ...
    c["q2.a"] = str(sp.expand(span))
    pk = sp.solve(
        sp.Eq(sp.Symbol("n") + 2 * sp.Symbol("p") - k + 1, sp.Symbol("n")),
        sp.Symbol("p"),
    )[0]
    c["q2.b"] = str(pk)
    c["q3.a"] = str(receptive([(3, 1)] * 3))
    c["q3.b"] = str(receptive([(3, 2), (3, 1)]))

    x, w = [1, 2, 3, 4], [1, 0, -1]
    corr = [sum(w[r] * x[i + r] for r in range(3)) for i in range(2)]
    conv = [sum(w[2 - r] * x[i + r] for r in range(3)) for i in range(2)]
    c["q4.a"], c["q4.b"] = vec(corr), vec(conv)
    T = [[0] * 4 for _ in range(2)]
    for i in range(2):
        for r in range(3):
            T[i][i + r] = w[r]
    c["q5.a"] = "[" + ", ".join(vec(row) for row in T) + "]"
    g = [1, 1]
    xs = sp.symbols("x0:4")
    ws = sp.symbols("w0:3")
    loss = sum(g[i] * sum(ws[r] * xs[i + r] for r in range(3)) for i in range(2))
    sub = {**dict(zip(xs, x)), **dict(zip(ws, w))}
    dx = [sp.diff(loss, v).subs(sub) for v in xs]
    c["q5.b"] = vec(dx)
    full = [
        sum(g[j] * w[i - j] for j in range(2) if 0 <= i - j < 3) for i in range(4)
    ]  # numpy.convolve
    c["q5.c"] = str(full == [int(v) for v in dx]).lower()
    c["q6.a"] = vec([sp.diff(loss, v).subs(sub) for v in ws])
    copies = sum(
        1
        for p in range(2)
        for q in range(2)
        for r in range(2)
        for s in range(2)
        if (p + r, q + s) == (1, 1)
    )
    c["q6.b"] = str(copies)

    c["q8.a"] = str(
        sum(1 for _ in range(128) for _ in range(64) for _ in range(9)) + 128
    )
    c["q8.b"] = str(64 * 1 * 3 * 3 + 64)
    c["q8.c"] = str(math.prod([128, 64, 3, 3]) * out_len(56, 3, 1, 1) ** 2)
    npatch = out_len(224, 16, 16) ** 2
    c["q9.a"] = str(npatch)
    c["q9.b"] = str(math.prod([768, 3, 16, 16]) + 768)
    c["q9.c"] = str(npatch + 1)

    N, C, H, W, e = 2, 3, 4, 5, 4

    def addr_nchw(n, cc, h, ww):
        return e * (((n * C + cc) * H + h) * W + ww)

    def addr_nhwc(n, cc, h, ww):
        return e * (((n * H + h) * W + ww) * C + cc)

    unit = [(1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)]
    c["q10.a"] = vec([addr_nchw(*u) for u in unit])
    c["q10.b"] = vec([addr_nhwc(*u) for u in unit])
    c["q10.c"] = str(addr_nhwc(1, 2, 3, 4))
    c["q10.d"] = str(3 * 14 * 14)

    def fold(f, sr):
        f = f % sr
        return min(f, sr - f)

    c["q11.a"] = str(16000 // 2)
    c["q11.b"] = str(fold(7000, 10000))
    c["q11.c"] = str(fold(12000, 16000))
    c["q11.d"] = str(fold(25000, 16000))
    alias = fold(R(4, 3), 1)  # cycles per output pixel after keeping every 4th
    c["q12.a"] = str(1 / alias)
    c["q12.b"] = str(R(4, 3) < R(1, 2)).lower()
    c["q12.c"] = str(R(1, 2) / 4)
    cb = [[(i + j) % 2 for j in range(8)] for i in range(8)]
    kept = {cb[i][j] for i in range(0, 8, 2) for j in range(0, 8, 2)}
    c["q12.d"] = "b" if len(kept) == 1 else "a"

    def keys(t, a):
        t = abs(t)
        if t < 1:
            return (a + 2) * t**3 - (a + 3) * t**2 + 1
        if t < 2:
            return a * t**3 - 5 * a * t**2 + 8 * a * t - 4 * a
        return 0

    c["q13.a"] = str(keys(R(1, 2), R(-1, 2)))
    c["q13.b"] = str(keys(R(3, 2), R(-1, 2)))
    c["q13.c"] = str(keys(R(3, 2), R(-3, 4)))
    sup = 1 * max(R(1000, 250), 1)
    c["q14.a"] = str(sup)
    c["q14.b"] = str(2 * sp.ceiling(sup) + 1)
    c["q14.c"] = str(3 * max(R(2), 1))

    def dft(v):
        n = len(v)
        return [
            sp.nsimplify(
                sp.simplify(
                    sum(v[j] * sp.exp(-2 * sp.pi * sp.I * j * kk / n) for j in range(n))
                )
            )
            for kk in range(n)
        ]

    c["q15.a"] = vec(dft([1, 0, 0, 0]))
    c["q15.b"] = vec(dft([1, 1, 1, 1]))
    c["q15.c"] = vec(dft([0, 1, 0, 0]))
    c["q15.d"] = str(sp.conjugate(2 - sp.I))
    X = dft([1, 2, 0, -1])
    c["q16.a"] = str(sp.simplify(sum(sp.Abs(v) ** 2 for v in X)))
    a4, b4 = [1, 2, 0, 0], [1, 1, 0, 0]
    c["q16.b"] = vec([sum(a4[j] * b4[(i - j) % 4] for j in range(4)) for i in range(4)])
    c["q16.c"] = str(300 + 100 - 1)
    even = [3, 1, 2, 1]  # x[j] = x[(4 - j) % 4]
    c["q16.d"] = str(all(sp.im(v) == 0 for v in dft(even))).lower()
    radices, n = [], 400
    for p in (4, 4, 5, 5):
        assert n % p == 0
        radices.append(p)
        n //= p
    c["q17.a"] = str(len(radices))
    c["q17.b"] = str(400 * 400)
    c["q17.c"] = str((1024 // 2) * int(math.log2(1024)))
    c["q17.d"] = str(400 & (400 - 1) == 0).lower()

    samples, n_fft, hop = 30 * 16000, 400, 160
    padded = samples + 2 * (n_fft // 2)
    c["q18.a"] = str(sum(1 for t in range(0, padded, hop) if t + n_fft <= padded))
    c["q18.b"] = str(R(16000, n_fft))
    c["q18.c"] = str(n_fft // 2 + 1)
    nw = 16
    hann = [R(1, 2) - R(1, 2) * sp.cos(2 * sp.pi * j / nw) for j in range(nw)]
    m = 2 * nw  # a sample far from the edges
    total = sum(
        hann[m - t * (nw // 4)] ** 2
        for t in range(m // (nw // 4) + 1)
        if 0 <= m - t * (nw // 4) < nw
    )
    c["q18.d"] = str(sp.nsimplify(sp.simplify(total)))
    c["q18.e"] = str(R(1000 * n_fft, 16000))
    c["q18.f"] = vec(
        [sp.nsimplify(R(1, 2) - R(1, 2) * sp.cos(2 * sp.pi * j / 4)) for j in range(4)]
    )

    def slaney(f):
        f = sp.nsimplify(f)
        return (
            f / R(200, 3)
            if f < 1000
            else 15 + sp.log(f / 1000) / (sp.log(R(32, 5)) / 27)
        )

    c["q19.a"] = str(sp.simplify(slaney(1000)))
    c["q19.b"] = str(sp.simplify(slaney(6400)))
    c["q19.c"] = str(2595 * sp.log(1 + R(700, 700), 10))
    c["q19.d"] = str(sp.simplify(10 * sp.log(1000, 10)))
    L = max(sp.log(max(R(1, 10**12), R(1, 10**10)), 10), 1 - 8)
    c["q19.e"] = str(sp.nsimplify((L + 4) / 4))
    c["q19.f"] = str(sp.nsimplify((sp.log(R(1, 10**10), 10) + 4) / 4))

    s = sp.symbols("s1:5")
    info = -sp.log(sp.exp(s[0]) / sum(sp.exp(v) for v in s))
    c["q20.a"] = str(sp.simplify(info.subs({s[0]: 2, s[1]: 0, s[2]: 0, s[3]: 0})))
    Nn = sp.Symbol("N", positive=True)
    c["q20.b"] = str(sp.simplify(-sp.log(sp.exp(0) / (Nn * sp.exp(0)))))
    c["q20.c"] = str(sp.log(1024) - 2)
    sv, t, b, z = sp.symbols("s t b z", real=True)
    sig = -sp.log(1 / (1 + sp.exp(-z * (t * sv + b))))
    c["q20.d"] = str(sp.simplify(sp.diff(sig, sv).subs({z: 1, t: 10, b: -10, sv: 1})))
    ref, hyp = "hello".split(), "oh hello there".split()
    c["q23"] = str(R(edit(ref, hyp), len(ref)))
    return confirm("S-M12", c)


if __name__ == "__main__":
    sys.exit(main())
