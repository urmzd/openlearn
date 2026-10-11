# /// script
# requires-python = ">=3.12"
# dependencies = ["torch==2.14.1", "numpy==2.5.3"]
# ///
"""Maintainer generator for the M12.1 torch golden fixture (conv1d, conv2d, unfold, fold).

For each conv case, float64 inputs x, weight w, an optional bias b, and an
upstream gradient g of the output; records torch.nn.functional.conv1d or
conv2d's output and the autograd gradients of sum(out * g) with respect to
x, w, and b (the backward of a conv is a conv: the course test rebuilds them
from im2col and col2im). For each unfold case, F.unfold(x) and F.fold of a
random column tensor (the adjoint, which sums overlapping patches).

  conv2d.plain     x [2, 3, 7, 6],  w [4, 3, 3, 3], bias
  conv2d.strided   x [1, 4, 9, 8],  w [6, 2, 3, 2], stride (2, 1), pad (1, 2), dilation (1, 2), groups 2, bias
  conv2d.depthwise x [2, 6, 8, 8],  w [6, 1, 3, 3], stride 2, pad 1, groups 6
  conv2d.patch     x [1, 3, 8, 8],  w [5, 3, 4, 4], stride 4 (a ViT patch embedding), bias
  conv1d.plain     x [2, 3, 11],    w [4, 3, 3], pad 1, bias
  conv1d.dilated   x [1, 4, 20],    w [4, 2, 3], stride 2, pad 1, dilation 2, groups 2
  conv1d.stem      x [1, 5, 12],    w [6, 5, 3], stride 2, pad 1, bias (Whisper's second stem conv)
  unfold.a         x [2, 3, 6, 5], kernel (2, 3), stride 1, pad 0, dilation 1
  unfold.b         x [1, 2, 9, 7], kernel (3, 2), stride (2, 1), pad (1, 1), dilation (2, 1)

    uv run --offline --python 3.12 --script course/oracle/M12.1/conv_torch.py

Run from the repo root, then update the MANIFEST.tsv row it prints.
"""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as Fn

OUT = Path("course/fixtures/M12.1/conv_torch.npz")
gen = np.random.default_rng(20261201)


def conv(arrs, name, xs, ws, bias, stride, pad, dilation, groups):
    x, w = gen.standard_normal(xs), gen.standard_normal(ws)
    b = gen.standard_normal(ws[0]) if bias else None
    xt, wt = torch.tensor(x, requires_grad=True), torch.tensor(w, requires_grad=True)
    bt = None if b is None else torch.tensor(b, requires_grad=True)
    fn = Fn.conv2d if len(xs) == 4 else Fn.conv1d
    out = fn(xt, wt, bt, stride=stride, padding=pad, dilation=dilation, groups=groups)
    g = gen.standard_normal(out.shape)
    (out * torch.from_numpy(g)).sum().backward()
    arrs.update(
        {
            f"{name}.x": x,
            f"{name}.w": w,
            f"{name}.g": g,
            f"{name}.y": out.detach().numpy(),
            f"{name}.dx": xt.grad.numpy(),
            f"{name}.dw": wt.grad.numpy(),
            f"{name}.params": np.array(
                [
                    *np.atleast_1d(stride),
                    *np.atleast_1d(pad),
                    *np.atleast_1d(dilation),
                    groups,
                ]
            ),
        }
    )
    if b is not None:
        arrs[f"{name}.b"] = b
        arrs[f"{name}.db"] = bt.grad.numpy()


def unfold(arrs, name, xs, k, stride, pad, dilation):
    x = gen.standard_normal(xs)
    cols = Fn.unfold(
        torch.from_numpy(x), k, dilation=dilation, padding=pad, stride=stride
    )
    c = gen.standard_normal(tuple(cols.shape))
    folded = Fn.fold(
        torch.from_numpy(c), xs[2:], k, dilation=dilation, padding=pad, stride=stride
    )
    arrs.update(
        {
            f"{name}.x": x,
            f"{name}.cols": cols.numpy(),
            f"{name}.c": c,
            f"{name}.folded": folded.numpy(),
            f"{name}.params": np.array([*k, *stride, *pad, *dilation]),
        }
    )


def main() -> None:
    arrs: dict[str, np.ndarray] = {}
    conv(
        arrs,
        "conv2d.plain",
        (2, 3, 7, 6),
        (4, 3, 3, 3),
        True,
        (1, 1),
        (0, 0),
        (1, 1),
        1,
    )
    conv(
        arrs,
        "conv2d.strided",
        (1, 4, 9, 8),
        (6, 2, 3, 2),
        True,
        (2, 1),
        (1, 2),
        (1, 2),
        2,
    )
    conv(
        arrs,
        "conv2d.depthwise",
        (2, 6, 8, 8),
        (6, 1, 3, 3),
        False,
        (2, 2),
        (1, 1),
        (1, 1),
        6,
    )
    conv(
        arrs,
        "conv2d.patch",
        (1, 3, 8, 8),
        (5, 3, 4, 4),
        True,
        (4, 4),
        (0, 0),
        (1, 1),
        1,
    )
    conv(arrs, "conv1d.plain", (2, 3, 11), (4, 3, 3), True, 1, 1, 1, 1)
    conv(arrs, "conv1d.dilated", (1, 4, 20), (4, 2, 3), False, 2, 1, 2, 2)
    conv(arrs, "conv1d.stem", (1, 5, 12), (6, 5, 3), True, 2, 1, 1, 1)
    unfold(arrs, "unfold.a", (2, 3, 6, 5), (2, 3), (1, 1), (0, 0), (1, 1))
    unfold(arrs, "unfold.b", (1, 2, 9, 7), (3, 2), (2, 1), (1, 1), (2, 1))
    meta = {
        "generator": "course/oracle/M12.1/conv_torch.py",
        "torch": torch.__version__,
        "numpy": np.__version__,
        "params": "conv: stride.., pad.., dilation.., groups; unfold: kh, kw, sh, sw, ph, pw, dh, dw",
    }
    arrs["__meta__"] = np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrs)
    b = buf.getvalue()
    OUT.write_bytes(b)
    print(
        f"{OUT}\t{hashlib.sha256(b).hexdigest()}\t{len(b)}\tcourse/oracle/M12.1/conv_torch.py\t"
        f"torch=={torch.__version__.split('+')[0]},numpy=={np.__version__}\t-\tApache-2.0"
    )


if __name__ == "__main__":
    main()
