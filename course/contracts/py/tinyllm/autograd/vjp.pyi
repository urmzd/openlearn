# contracts/py/tinyllm/autograd/vjp.pyi (M08.3): closed-form vector-Jacobian products
# chapter: content/courses/mathematics/14-matrix-calculus-and-autodiff/03-matrix-differentials-and-vjps.md
#
# For an op Y = f(X) inside a scalar loss L, the VJP maps the upstream
# gradient G = dL/dY to dL/dX. Every function here is pure: it reads its
# arguments (never writes them) and returns new float arrays of the input's
# shape. The forward pass saves what the VJP needs (y, xhat, rstd), so the
# VJP never recomputes it.
from numpy.typing import ArrayLike, NDArray

def unbroadcast(g: ArrayLike, shape: tuple[int, ...]) -> NDArray:
    """Sum g down to `shape`, undoing numpy broadcasting: sum over the
    leading axes g has and shape lacks, then over every axis where shape has
    1 and g does not (keeping it as size 1). ValueError when g's shape could
    not have come from broadcasting `shape`."""

def matmul_vjp(g: ArrayLike, A: ArrayLike, B: ArrayLike) -> tuple[NDArray, NDArray]:
    """Y = A @ B with A [..., m, k] and B [..., k, n] (numpy matmul,
    batch axes broadcast): dA = G B^T, dB = A^T G, each unbroadcast to its
    operand's shape. ValueError when A or B has fewer than 2 axes."""

def softmax_vjp(g: ArrayLike, y: ArrayLike, axis: int = -1) -> NDArray:
    """y = softmax(x, axis): dx = y * (g - sum(g * y, axis, keepdims=True))."""

def log_softmax_vjp(g: ArrayLike, y: ArrayLike, axis: int = -1) -> NDArray:
    """y = log_softmax(x, axis) (log-probabilities):
    dx = g - exp(y) * sum(g, axis, keepdims=True)."""

def layernorm_vjp(
    g: ArrayLike, xhat: ArrayLike, rstd: ArrayLike, gamma: ArrayLike
) -> tuple[NDArray, NDArray, NDArray]:
    """LayerNorm over the last axis (size D):
        mu = mean(x), var = mean((x - mu)^2), rstd = 1 / sqrt(var + eps),
        xhat = (x - mu) * rstd, y = gamma * xhat + beta.
    rstd has shape x.shape[:-1] or x.shape[:-1] + (1,). With d = g * gamma:
        dx = rstd * (d - mean(d) - xhat * mean(d * xhat))   (means over D)
        dgamma = sum of g * xhat over every leading axis, dbeta = sum of g.
    Returns (dx, dgamma, dbeta); dgamma and dbeta have gamma's shape (D,)."""

def rmsnorm_vjp(
    g: ArrayLike, x: ArrayLike, rstd: ArrayLike, w: ArrayLike
) -> tuple[NDArray, NDArray]:
    """RMSNorm over the last axis (size D):
        rstd = 1 / sqrt(mean(x^2) + eps), y = w * x * rstd.
    rstd has shape x.shape[:-1] or x.shape[:-1] + (1,). With d = g * w:
        dx = rstd * (d - x * rstd^2 * mean(d * x))
        dw = sum of g * x * rstd over every leading axis.
    Returns (dx, dw)."""

def cross_entropy_vjp(logits: ArrayLike, targets: ArrayLike, ignore_index: int = -100) -> NDArray:
    """Gradient of L = mean over the rows t != ignore_index of
    -log_softmax(logits)[row, t] (logits [..., V], integer targets [...]):
        dlogits = (softmax(logits) - onehot(t)) / n_valid on valid rows,
        0 on ignored rows; all zeros when every row is ignored.
    softmax is M09.2's (finite for logits near 1e4). ValueError when a
    target other than ignore_index lies outside [0, V)."""
