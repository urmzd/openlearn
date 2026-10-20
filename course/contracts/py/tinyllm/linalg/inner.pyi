# contracts/py/tinyllm/linalg/inner.pyi (M03.6)
# chapter: content/courses/mathematics/09-linear-algebra/06-inner-products-projections-cosine-and-top-k.md
#
# Inner products, the projections and similarities built on them, and exact
# top-k retrieval by cosine similarity. Vectors lie along `axis` (default the
# last); every other axis broadcasts with numpy's rules. Results are float64.
from numpy.typing import ArrayLike, NDArray

def normalize(x: ArrayLike, axis: int = -1, eps: float = 1e-12) -> NDArray:
    """x / max(||x||, eps) along axis: a unit vector, or (for ||x|| < eps)
    x scaled by 1 / eps, so the zero vector stays zero instead of NaN."""

def cosine_sim(a: ArrayLike, b: ArrayLike, axis: int = -1, eps: float = 1e-8) -> NDArray:
    """cos(a, b) = (a . b) / (max(||a||, eps) * max(||b||, eps)) along axis,
    each norm clamped separately (torch.nn.functional.cosine_similarity's
    rule). In [-1, 1] up to rounding; 0 when either vector is zero. The
    reduced axis is dropped. ValueError when the vector lengths differ."""

def project(x: ArrayLike, v: ArrayLike, axis: int = -1) -> NDArray:
    """The orthogonal projection of x onto the line spanned by v:
    ((x . v) / (v . v)) v along axis, broadcast. x - project(x, v) is
    orthogonal to v. ValueError when some v is the zero vector or the
    vector lengths differ."""

def topk_cosine(query: ArrayLike, matrix: ArrayLike, k: int) -> tuple[NDArray, NDArray]:
    """Exact top-k rows of matrix [n, d] by cosine similarity with query [d]
    (or with each row of a batch [q, d]). Returns (indices int64, scores
    float64), each [k] (or [q, k]), scores as cosine_sim with eps = 1e-8,
    ordered by score descending, ties to the lowest row index (the D11
    rule), so the result is deterministic. ValueError unless
    1 <= k <= n and the vector lengths agree."""
