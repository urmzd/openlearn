# contracts/py/tinyllm/autograd/mode.pyi (L0.1): grad mode
# chapter: content/courses/build-an-llm/p00-foundations/01-tensor-and-broadcasting-backward.md
#
# Grad mode decides whether an op records a graph node. It is on by default,
# per thread, and `no_grad` turns it off for the body of a `with` block:
# evaluation, sampling, and optimizer updates build no graph and keep no
# intermediate arrays alive.
from contextlib import AbstractContextManager

def is_grad_enabled() -> bool:
    """True unless the current thread is inside a `no_grad()` block."""

def no_grad() -> AbstractContextManager[None]:
    """Context manager: grad mode is off inside the block and restored to its
    previous value on exit, also when the block raises. Blocks nest. Ops run
    inside it return tensors with requires_grad False and no parents."""
