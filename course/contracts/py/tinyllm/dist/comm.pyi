# contracts/py/tinyllm/dist/comm.pyi (L11.2, optional): collectives over processes
# chapter: content/courses/build-an-llm/12-training-at-scale/02-collectives-over-processes.md
#
# `world` processes, numbered 0 .. world - 1 (the rank), each holding a Comm.
# Point to point: one full-duplex pipe per pair of ranks (multiprocessing
# Pipe); a message is a numpy array, delivered in order. Collectives are
# built from send and recv on a ring (rank r sends to (r + 1) % world and
# receives from (r - 1) % world) and must be called by every rank in the same
# order with arrays of the same shape and dtype.
#
# Chunks: a flat array of n entries splits into `world` contiguous chunks by
# chunk_bounds (the sizes of numpy.array_split: the first n % world chunks
# hold one entry more). Rank r owns chunk r.
#
# Deadlock rule: a send of a large array blocks until the receiver reads it.
# In every ring step, even ranks send then receive and odd ranks receive then
# send, so a ring of any size makes progress.
from typing import Any, Callable, Literal

from numpy.typing import ArrayLike, NDArray

def chunk_bounds(n: int, world: int) -> list[tuple[int, int]]:
    """[(start, end)] of the world chunks of n entries, in order, as
    numpy.array_split: sizes n // world + 1 for the first n % world chunks,
    n // world for the rest. ValueError for n < 0 or world < 1."""

class Comm:
    rank: int
    world: int
    bytes_sent: int  # payload bytes (array.nbytes) this rank has sent, over its lifetime

    def send(self, x: ArrayLike, dst: int) -> None:
        """Send a copy of x to rank dst and count x.nbytes. ValueError for
        dst == rank or dst outside [0, world)."""

    def recv(self, src: int) -> NDArray:
        """The next array rank src sent to this rank (blocks until it arrives)."""

    def reduce_scatter(self, x: ArrayLike) -> NDArray:
        """Ring reduce-scatter of the flattened x: returns chunk `rank` of the
        elementwise sum over ranks (1-D, x's dtype). In step s = 0 .. world - 2,
        send chunk (rank - s - 1) % world to the right neighbour, receive chunk
        (rank - s - 2) % world from the left one, and add it into that chunk.
        Each rank sends world - 1 chunks: about (world - 1) / world of x."""

    def all_gather(self, x: ArrayLike) -> NDArray:
        """Ring all-gather: x is this rank's chunk (1-D; chunks may differ in
        size), the result the concatenation of every rank's chunk in rank
        order. In step s = 0 .. world - 2, send chunk (rank - s) % world to
        the right neighbour and receive chunk (rank - s - 1) % world."""

    def all_reduce(self, x: ArrayLike, op: Literal["sum", "mean"] = "sum") -> NDArray:
        """reduce_scatter, then all_gather, reshaped to x's shape; "mean"
        divides the sum by world. Every rank gets a bit-identical result, and
        each sends 2 (world - 1) / world of x's bytes when world divides x.size
        (the bandwidth-optimal ring). ValueError for another op."""

    def broadcast(self, x: ArrayLike, src: int) -> NDArray:
        """Rank src's x on every rank (the x other ranks pass is ignored),
        passed along the ring from src: every rank but the one just before
        src sends it once, to its right neighbour."""

    def barrier(self) -> None:
        """Return only when every rank has called barrier."""

def spawn(fn: Callable[..., Any], world: int, *args: Any, timeout: float = 60.0) -> list[Any]:
    """Run fn(comm, *args) in `world` new processes (multiprocessing's
    "spawn" start method, so fn must be a module-level function), one Comm
    per rank, and return their return values by rank. A rank that raises
    stops the run: the other processes are terminated and RuntimeError
    carries that rank's traceback. TimeoutError when they do not all finish
    within timeout seconds (a deadlock), after terminating them.
    ValueError for world < 1."""
