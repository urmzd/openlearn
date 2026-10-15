# contracts/py/tinyllm/sig/conv.pyi (M12.1): convolution as a linear map
# chapter: content/tracks/math/12-signals-and-images/01-convolution-as-a-linear-map.md
#
# Shapes: 1-D x [N, C, L] with w [K, C/groups, R]; 2-D x [N, C, H, W]
# (NCHW) with w [K, C/groups, R, S] (KCRS). "Convolution" here is what deep
# learning means by it: cross-correlation, no kernel flip, the semantics of
# torch.nn.functional.conv1d and conv2d. stride, pad, and dilation are an
# int or a (row, column) pair of ints; padding is zeros. Results are
# float64; inputs are never modified. Every function raises ValueError on a
# shape that does not fit (a kernel larger than the padded input, groups
# that do not divide the channels, a weight with the wrong channel count).
from numpy.typing import NDArray

def conv_out_size(n: int, k: int, stride: int = 1, pad: int = 0, dilation: int = 1) -> int:
    """Output length along one axis: floor((n + 2 pad - dilation (k - 1) - 1) / stride) + 1.
    ValueError unless n, k, stride, dilation >= 1, pad >= 0, and the dilated
    kernel, dilation (k - 1) + 1 samples, fits in the padded input n + 2 pad."""

def conv1d_direct(x: NDArray, w: NDArray, b: NDArray | None, stride=1, pad=0, dilation=1, groups=1) -> NDArray:
    """y[n, k, o] = b[k] + sum over c in k's group and r of
    w[k, c, r] x[n, g C/groups + c, o stride + r dilation - pad], as F.conv1d.
    Input channels split into `groups` equal groups; output channel k sees only
    group k // (K / groups). Returns [N, K, conv_out_size(L, R, ...)]."""

def conv2d_direct(x: NDArray, w: NDArray, b: NDArray | None, stride=(1, 1), pad=(0, 0), dilation=(1, 1), groups=1) -> NDArray:
    """The 2-D version of conv1d_direct over NCHW and KCRS, as F.conv2d:
    returns [N, K, P, Q] with P = conv_out_size(H, R, ...) and
    Q = conv_out_size(W, S, ...)."""

def correlate_vs_convolve(x: NDArray, w: NDArray) -> tuple[NDArray, NDArray]:
    """For 2-D x [H, W] and kernel w [R, S], the 'valid' cross-correlation
    (what conv2d_direct computes) and the true convolution, which is the
    cross-correlation with w flipped on both axes. Both [H - R + 1, W - S + 1]."""

def im2col(x: NDArray, kh: int, kw: int, stride=(1, 1), pad=(0, 0), dilation=(1, 1)) -> NDArray:
    """Every receptive field of x [N, C, H, W] as a column: [N, C*kh*kw, P*Q],
    rows ordered (c, r, s) and columns (p, q) row-major, exactly
    F.unfold(x, (kh, kw), dilation, padding, stride). Then a conv is a matmul:
    conv2d_direct(x, w, None) == (w.reshape(K, -1) @ im2col(x, R, S)).reshape(N, K, P, Q)
    for groups = 1."""

def col2im(cols: NDArray, x_shape: tuple[int, int, int, int], kh: int, kw: int, stride=(1, 1), pad=(0, 0), dilation=(1, 1)) -> NDArray:
    """The adjoint of im2col for an input of shape x_shape: scatter each
    column back to the pixels it was copied from and SUM where windows
    overlap (entries that fell in the padding are dropped), exactly F.fold.
    <im2col(x), y> == <x, col2im(y)> for every x and y. ValueError when
    cols is not [N, C*kh*kw, P*Q] for x_shape."""

def conv_matrix(in_shape: tuple[int, int], w: NDArray, stride=(1, 1), pad=(0, 0)) -> NDArray:
    """The doubly block Toeplitz matrix T [P*Q, H*W] of the single-channel
    cross-correlation of an [H, W] input with the 2-D kernel w (zero padding,
    no dilation): T @ x.ravel() == conv2d_direct(x[None, None], w[None, None],
    None, stride, pad).ravel(). T's transpose is the backward pass with
    respect to x."""
