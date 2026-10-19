#!/usr/bin/env python3
"""Pull vs push: why a pull loop gives backpressure for free.

Standalone, no external deps:  python3 backpressure_demo.py

Model (Streamflow `orders`):
  - Producer offers messages at arrival rate lambda (msgs/tick).
  - One worker has a service rate mu (msgs/tick), mu < lambda (overload).

PUSH: the broker shoves `lambda` msgs/tick at the worker regardless of pace.
  The worker can only durably handle `mu`; the surplus has nowhere to live in
  the worker, so it is dropped (or, in a real push broker, buffered in an
  unbounded in-flight set until the worker OOMs / the broker's prefetch fills).

PULL: the worker calls poll() and only ever asks for what it can handle (`mu`).
  The unconsumed surplus stays *in the log* as a growing backlog (lag). Nothing
  is lost and nothing overwhelms the worker -- the backlog is the backpressure
  signal an autoscaler (KEDA, topic 01) reacts to.

We also show Little's Law (L = lambda * W) and that queue depth is the integral
of (arrival - service) over time.
"""

from __future__ import annotations

TICKS = 20
LAMBDA = 10.0  # arrivals per tick
MU = 6.0  # the single worker's service rate per tick


def simulate_push() -> tuple[int, int]:
    """Broker pushes LAMBDA/tick. Worker drains MU/tick; the rest is dropped."""
    handled = 0
    dropped = 0
    for _ in range(TICKS):
        arrived = LAMBDA
        done = min(arrived, MU)  # worker can only durably process MU
        handled += int(done)
        dropped += int(arrived - done)  # surplus has nowhere safe to go
    return handled, dropped


def simulate_pull() -> tuple[int, int]:
    """Worker pulls <= MU/tick. Surplus accumulates as backlog (lag), not loss."""
    handled = 0
    backlog = 0.0  # = integral of (arrival - service); this is consumer lag
    peak_backlog = 0.0
    for _ in range(TICKS):
        backlog += LAMBDA  # producer appends to the log
        pulled = min(backlog, MU)  # poll() asks only for what we can handle
        backlog -= pulled  # rest stays in the log -> lag grows
        handled += int(pulled)
        peak_backlog = max(peak_backlog, backlog)
    return handled, int(peak_backlog)


def littles_law(arrival_rate: float, latency_ticks: float) -> float:
    """L = lambda * W : expected in-flight count."""
    return arrival_rate * latency_ticks


def main() -> None:
    p_handled, p_dropped = simulate_push()
    l_handled, l_backlog = simulate_pull()

    print(
        f"params: lambda={LAMBDA}/tick  mu={MU}/tick  ticks={TICKS}  (overload: lambda>mu)\n"
    )

    print("PUSH (broker sets the pace):")
    print(f"  handled durably : {p_handled}")
    print(f"  DROPPED/overrun : {p_dropped}   <- worker overwhelmed, surplus lost\n")

    print("PULL (consumer sets the pace via poll()):")
    print(f"  handled durably : {l_handled}")
    print(f"  backlog (lag)   : {l_backlog}   <- nothing lost; lag is the signal\n")

    assert p_dropped > 0, "push model should overrun under overload"
    assert l_handled == p_handled, "both drain at mu; pull just keeps the surplus"

    # Little's Law: at steady state with avg service latency W, in-flight L = lambda*W.
    w = 1.0 / MU  # avg ticks to service one message
    print(
        f"Little's Law  L = lambda * W = {LAMBDA} * {w:.3f} = "
        f"{littles_law(LAMBDA, w):.2f} in-flight (steady-state estimate)"
    )
    print(
        "queue depth(t) = integral_0^t (arrival(s) - service(s)) ds  ->  "
        "lag is the running area between the two curves"
    )
    print("\nOK")


if __name__ == "__main__":
    main()
