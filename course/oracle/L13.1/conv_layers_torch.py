# /// script
# requires-python = ">=3.12"
# dependencies = ["torch==2.14.1", "numpy==2.5.3"]
# ///
"""Maintainer generator for the L13.1 golden fixtures (torch.nn layers, float64).

Forward and backward of torch.nn.Conv2d, torch.nn.Conv1d,
F.max_pool2d, and F.avg_pool2d with an upstream gradient g, so the
learner's hand-written vector-Jacobian products are compared against
autograd:

  conv2d.lenet    Conv2d(1, 6, 5, pad 2) on 2x1x8x8 (LeNet's first layer)
  conv2d.grouped  Conv2d(4, 6, (3, 2), stride (2, 1), pad (1, 2), dilation (1, 2), groups 2)
  conv2d.patch    Conv2d(3, 8, 4, stride 4) on 1x3x8x8 (a ViT patch embedding)
  conv1d.stem1    Conv1d(5, 6, 3, pad 1) (Whisper's first stem conv)
  conv1d.stem2    Conv1d(6, 6, 3, stride 2, pad 1) (its second, halving the frames)
  maxpool.k2      max_pool2d(k 2) on 2x3x6x6
  maxpool.k3s2    max_pool2d(k 3, stride 2) on 1x2x7x7 (overlapping windows)
  maxpool.ties    max_pool2d(k 2) on integer-valued input with many ties
  avgpool.k2      avg_pool2d(k 2) on 2x3x6x6
  avgpool.k3s2    avg_pool2d(k 3, stride 2) on 1x2x7x7

Keys: <case>.x, .w, .b (convs), .g, .y, .dx, .dw, .db (convs), .params.

    uv run --offline --python 3.12 --script course/oracle/L13.1/conv_layers_torch.py

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

OUT = Path("course/fixtures/L13.1/conv_layers_torch.npz")
gen = np.random.default_rng(20261310)


def conv(arrs, name, xs, cin, cout, k, stride, pad, dilation, groups, bias=True):
    dims = len(xs) - 2
    layer = (torch.nn.Conv2d if dims == 2 else torch.nn.Conv1d)(
        cin,
        cout,
        k,
        stride=stride,
        padding=pad,
        dilation=dilation,
        groups=groups,
        bias=bias,
    ).double()
    with torch.no_grad():
        layer.weight.copy_(
            torch.from_numpy(gen.standard_normal(tuple(layer.weight.shape)))
        )
        if bias:
            layer.bias.copy_(torch.from_numpy(gen.standard_normal(cout)))
    x = gen.standard_normal(xs)
    xt = torch.tensor(x, requires_grad=True)
    y = layer(xt)
    g = gen.standard_normal(tuple(y.shape))
    (y * torch.from_numpy(g)).sum().backward()

    def flat(v):
        # 2-D cases always store (row, column) pairs; 1-D cases one int.
        a = list(np.atleast_1d(v))
        return a * 2 if dims == 2 and len(a) == 1 else a

    arrs.update(
        {
            f"{name}.x": x,
            f"{name}.w": layer.weight.detach().numpy().copy(),
            f"{name}.g": g,
            f"{name}.y": y.detach().numpy(),
            f"{name}.dx": xt.grad.numpy(),
            f"{name}.dw": layer.weight.grad.numpy(),
            f"{name}.params": np.array(
                [
                    cin,
                    cout,
                    *flat(k),
                    *flat(stride),
                    *flat(pad),
                    *flat(dilation),
                    groups,
                ]
            ),
        }
    )
    if bias:
        arrs[f"{name}.b"] = layer.bias.detach().numpy().copy()
        arrs[f"{name}.db"] = layer.bias.grad.numpy()


def pool(arrs, name, fn, xs, k, stride, ints=False):
    x = gen.integers(0, 3, xs).astype(np.float64) if ints else gen.standard_normal(xs)
    xt = torch.tensor(x, requires_grad=True)
    y = fn(xt, k, stride)
    g = gen.standard_normal(tuple(y.shape))
    (y * torch.from_numpy(g)).sum().backward()
    arrs.update(
        {
            f"{name}.x": x,
            f"{name}.g": g,
            f"{name}.y": y.detach().numpy(),
            f"{name}.dx": xt.grad.numpy(),
            f"{name}.params": np.array([k, stride]),
        }
    )


def main() -> None:
    torch.manual_seed(0)
    arrs: dict[str, np.ndarray] = {}
    conv(arrs, "conv2d.lenet", (2, 1, 8, 8), 1, 6, 5, 1, 2, 1, 1)
    conv(arrs, "conv2d.grouped", (1, 4, 9, 8), 4, 6, (3, 2), (2, 1), (1, 2), (1, 2), 2)
    conv(arrs, "conv2d.patch", (1, 3, 8, 8), 3, 8, 4, 4, 0, 1, 1)
    conv(arrs, "conv1d.stem1", (1, 5, 12), 5, 6, 3, 1, 1, 1, 1)
    conv(arrs, "conv1d.stem2", (1, 6, 12), 6, 6, 3, 2, 1, 1, 1)
    pool(arrs, "maxpool.k2", Fn.max_pool2d, (2, 3, 6, 6), 2, 2)
    pool(arrs, "maxpool.k3s2", Fn.max_pool2d, (1, 2, 7, 7), 3, 2)
    pool(arrs, "maxpool.ties", Fn.max_pool2d, (1, 2, 6, 6), 2, 2, ints=True)
    pool(arrs, "avgpool.k2", Fn.avg_pool2d, (2, 3, 6, 6), 2, 2)
    pool(arrs, "avgpool.k3s2", Fn.avg_pool2d, (1, 2, 7, 7), 3, 2)
    meta = {
        "generator": "course/oracle/L13.1/conv_layers_torch.py",
        "torch": torch.__version__,
        "numpy": np.__version__,
        "params": "conv: c_in, c_out, k.., stride.., pad.., dilation.., groups; pool: k, stride",
    }
    arrs["__meta__"] = np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrs)
    b = buf.getvalue()
    OUT.write_bytes(b)
    print(
        f"{OUT}\t{hashlib.sha256(b).hexdigest()}\t{len(b)}\tcourse/oracle/L13.1/conv_layers_torch.py\t"
        f"torch=={torch.__version__.split('+')[0]},numpy=={np.__version__}\t-\tApache-2.0"
    )


if __name__ == "__main__":
    main()
