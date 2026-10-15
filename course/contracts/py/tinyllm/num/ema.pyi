# contracts/py/tinyllm/num/ema.pyi (M02.2)
# chapter: content/tracks/math/02-calculus-2/02-series-and-the-ema.md
#
# The exponential moving average m_t = beta m_{t-1} + (1 - beta) x_t, m_0 = 0,
# is the weighted sum sum_{i=1}^{t} (1 - beta) beta**(t - i) x_i: a truncated
# geometric series whose weights add up to 1 - beta**t, not 1. Dividing by
# that sum removes the bias toward m_0 = 0 (Adam's bias correction, M10.3).
# The weights and their total come from tinyllm.num.series (M00.3).
from numpy.typing import ArrayLike, NDArray

def ema_weights(beta: float, t: int) -> NDArray:
    """float64 [t]: the weight (1 - beta) * beta**(t - i) of x_i in m_t, for
    i = 1..t (oldest first, so the last weight is 1 - beta). Their sum is
    1 - beta**t. t = 0 gives an empty array.
    ValueError unless 0 <= beta < 1 and t >= 0."""

class EMA:
    def __init__(self, beta: float) -> None:
        """m_0 = 0 at t = 0. ValueError unless 0 <= beta < 1 (and beta is finite)."""

    @property
    def t(self) -> int:
        """The number of updates so far."""

    @property
    def value(self) -> float | NDArray:
        """m_t, the biased average (0.0 before any update)."""

    def update(self, x: ArrayLike) -> float | NDArray:
        """m_t = beta * m_{t-1} + (1 - beta) * x; t += 1; return m_t.
        x is a number (the result is a Python float) or an array (the result
        is a new float64 array; the average is elementwise). Every update after
        the first must have the first one's shape: ValueError otherwise, and
        for a non-finite x, with the state left unchanged."""

    def value_debiased(self) -> float | NDArray:
        """m_t / (1 - beta**t): exact for a constant input at every t.
        With beta = 0 it is the last x. RuntimeError before the first update
        (0 / 0: nothing has been averaged yet)."""
