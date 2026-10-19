"""Limit order book with a matching engine, implemented three ways.

Standard library only. Run me: ``python matching_engine.py``.

THE PROBLEM
-----------
Orders arrive one at a time. Each has a side (buy/sell), a limit price, and a
quantity. On arrival, an order matches against resting orders on the opposite
side while the prices are compatible; whatever quantity is left rests in the
book. Three rules define the whole system:

    1. Match on arrival. The remainder rests.
    2. Best price first; within a price, oldest first (price-time priority).
    3. The execution prints at the RESTING order's price, not the arriving one.

THE ONE INVARIANT
-----------------
After every event, the book is never crossed: the best bid is strictly below
the best ask. If a compatible pair could rest simultaneously, rule 1 was
violated somewhere. ``Engine.check_invariant`` asserts this after every order,
which is what makes a deterministic replay a real test rather than a demo.

THE SHAPE
---------
Two orderings appear in rule 2, so the storage layer needs two structures:

    * one that ranks PRICES  (which level is best)
    * one that ranks TIME    (which order within a level is oldest)

Everything below is a variation on how to hold those two orderings. The Engine
never changes -- that is the point of putting them behind one interface.

    Layer 1  ListBook    flat list, linear scan        best(): O(n)
    Layer 2  HeapBook    heap of prices + dict levels  best(): O(1) amortised
    Layer 3  LadderBook  tick-indexed array + pointer  best(): O(1) amortised,
                                                       no logarithm anywhere

Layer 1 is the one to write first: correct by inspection, gets the matching
loop under test, and gives you a working system to point at while discussing
the improvements. The jump to each next layer is earned by naming the
bottleneck, the unnecessary work, and the duplicated work in the previous one.
"""

from __future__ import annotations

import heapq
import itertools
from dataclasses import dataclass
from typing import Protocol

# Prices are integers (ticks or cents). Floats never enter the engine: float
# equality would silently break both price comparison and level bucketing.


@dataclass
class Order:
    id: int
    side: str  # "buy" or "sell"
    price: int  # buy: max willing to pay. sell: min willing to accept.
    qty: int  # remaining quantity; mutated as fills happen
    seq: int = 0  # arrival order, stamped by the Engine (the tie-break truth)


@dataclass(frozen=True)
class Fill:
    """One execution. Frozen on purpose: it captures price and quantity AT
    fill time, so later mutation of the Order objects cannot rewrite history."""

    buy_id: int
    sell_id: int
    price: int
    qty: int


class Book(Protocol):
    """One side's resting orders. Three verbs, and the Engine knows no more.

    add(order)        -- rest an order, joining the back of its price level
    best()            -- best-priced level's oldest order, or None
    fill(order, qty)  -- bookkeeping after an execution against a resting order
    """

    def add(self, order: Order) -> None: ...
    def best(self) -> Order | None: ...
    def fill(self, order: Order, qty: int) -> None: ...


def _better(side: str, a: int, b: int) -> bool:
    """Is price `a` more aggressive than `b` for this side?

    The only place the buy/sell asymmetry is written down. Buys want the
    highest price first, sells the lowest.
    """
    return a > b if side == "buy" else a < b


# ===========================================================================
# LAYER 1 -- ListBook. Brute force: one flat list, scan it.
#
# Write this first. Then justify the next layer out loud:
#   Bottleneck:  best() scans every resting order, O(n), and the Engine calls
#                best() once per fill iteration.
#   Unnecessary: it scans orders at prices that cannot possibly win.
#   Duplicated:  the best price rarely changes between calls, yet it is
#                recomputed from scratch every single time.
# ===========================================================================


class ListBook:
    def __init__(self, side: str) -> None:
        self.side = side
        self.orders: list[Order] = []

    def add(self, order: Order) -> None:
        self.orders.append(order)

    def best(self) -> Order | None:
        winner: Order | None = None
        for order in self.orders:
            if winner is None or _better(self.side, order.price, winner.price):
                winner = order
            elif order.price == winner.price and order.seq < winner.seq:
                winner = order  # same price: older wins
        return winner

    def fill(self, order: Order, qty: int) -> None:
        order.qty -= qty
        if order.qty == 0:
            self.orders.remove(order)


# ===========================================================================
# LAYER 2 -- HeapBook. One structure per ordering, which is the whole trick.
#
#   self.heap  ranks PRICES -- one entry per price LEVEL, not per order
#   each level is a dict, and dicts preserve insertion order = oldest first
#
# The heap holds levels, not orders, so it stays small (bounded by distinct
# live prices) and fill() never touches it. An emptied level leaves a stale
# price behind and best() sweeps it lazily -- deleting eagerly from a heap
# costs O(n) to find the entry, which is exactly the cost being avoided.
# ===========================================================================


class HeapBook:
    def __init__(self, side: str) -> None:
        self.side = side
        self.levels: dict[int, dict[int, Order]] = {}  # price -> {id: Order}
        self.heap: list[int] = []  # prices, negated for buys

    def _key(self, price: int) -> int:
        # heapq is min-only. Sells want the lowest price first: natural order.
        # Buys want the highest first: negate going in, negate coming out.
        return -price if self.side == "buy" else price

    def add(self, order: Order) -> None:
        if order.price not in self.levels:
            self.levels[order.price] = {}
            heapq.heappush(self.heap, self._key(order.price))  # once per level
        self.levels[order.price][order.id] = order  # back of the queue

    def best(self) -> Order | None:
        while self.heap:
            price = -self.heap[0] if self.side == "buy" else self.heap[0]
            level = self.levels.get(price)
            if level:
                return next(iter(level.values()))  # first inserted = oldest
            heapq.heappop(self.heap)  # dead level: sweep it and retry
        return None

    def fill(self, order: Order, qty: int) -> None:
        order.qty -= qty
        if order.qty == 0:
            level = self.levels[order.price]
            del level[order.id]
            if not level:
                del self.levels[order.price]  # heap entry stays; best() sweeps


# ===========================================================================
# LAYER 3 -- LadderBook. Drop the logarithm entirely.
#
# Real venues quote a bounded price range on a fixed tick, so "price" is a
# small integer index, not an arbitrary key. That turns the price ordering
# from a comparison problem into an ARRAY INDEX problem: levels live in a
# preallocated array and best() walks a pointer that only ever moves away
# from the top of the book, then gets pulled back on add().
#
# Amortised O(1) with no comparisons and no allocation on the hot path. The
# trade is memory proportional to the price range and a hard assumption that
# prices stay inside it -- state that assumption out loud, because it is the
# thing that makes this layer legitimate rather than a micro-optimisation.
# ===========================================================================


class LadderBook:
    def __init__(self, side: str, max_price: int = 1024) -> None:
        self.side = side
        self.max_price = max_price
        self.levels: list[dict[int, Order] | None] = [None] * (max_price + 1)
        # Pointer to the current best price. Starts at the worst possible end
        # and is dragged toward the top of the book by add().
        self.ptr = 0 if side == "buy" else max_price

    def _more_aggressive(self, price: int) -> bool:
        return _better(self.side, price, self.ptr)

    def add(self, order: Order) -> None:
        if not 0 <= order.price <= self.max_price:
            raise ValueError(
                f"price {order.price} outside ladder [0, {self.max_price}]"
            )
        level = self.levels[order.price]
        if level is None:
            level = self.levels[order.price] = {}
        level[order.id] = order
        if self._more_aggressive(order.price):
            self.ptr = order.price  # a new best: pull the pointer to it

    def best(self) -> Order | None:
        step = -1 if self.side == "buy" else 1
        while 0 <= self.ptr <= self.max_price:
            level = self.levels[self.ptr]
            if level:
                return next(iter(level.values()))
            self.ptr += step  # empty level: walk away from the top of book
        return None

    def fill(self, order: Order, qty: int) -> None:
        order.qty -= qty
        if order.qty == 0:
            level = self.levels[order.price]
            assert level is not None
            del level[order.id]
            if not level:
                self.levels[order.price] = None


# ===========================================================================
# ENGINE -- the one algorithm. It never changes across layers.
# ===========================================================================


class Engine:
    def __init__(self, book_factory=HeapBook) -> None:
        self.buys: Book = book_factory("buy")
        self.sells: Book = book_factory("sell")
        self.history: list[Fill] = []  # append-only; matching never reads it
        self._seq = itertools.count()

    def process(self, order: Order) -> list[Fill]:
        order.seq = next(self._seq)
        # The only routing in the system.
        opposite, own = (
            (self.sells, self.buys) if order.side == "buy" else (self.buys, self.sells)
        )

        fills: list[Fill] = []
        while order.qty > 0:
            resting = opposite.best()
            if resting is None or not self._compatible(order, resting):
                break
            qty = min(order.qty, resting.qty)  # partial fills, both directions
            buy, sell = (order, resting) if order.side == "buy" else (resting, order)
            # Rule 3: the price is the RESTING order's. The arriving order was
            # willing to do worse; the resting order set the terms by waiting.
            fill = Fill(buy.id, sell.id, price=resting.price, qty=qty)
            fills.append(fill)
            self.history.append(fill)
            opposite.fill(resting, qty)
            order.qty -= qty

        if order.qty > 0:  # zero-remainder guard: never rest a dead order
            own.add(order)
        return fills

    @staticmethod
    def _compatible(order: Order, resting: Order) -> bool:
        buy, sell = (order, resting) if order.side == "buy" else (resting, order)
        return buy.price >= sell.price

    def check_invariant(self) -> None:
        """The book is never crossed. Checked after every event."""
        best_bid, best_ask = self.buys.best(), self.sells.best()
        assert (
            best_bid is None or best_ask is None or best_bid.price < best_ask.price
        ), f"crossed book: best bid {best_bid.price} >= best ask {best_ask.price}"


# ===========================================================================
# VALIDATION = deterministic replay. A known input stream, expected fills,
# expected final book, invariant asserted after every single event. Every
# scenario runs against all three layers and must produce identical output.
# ===========================================================================


def replay(engine: Engine, orders: list[Order]) -> list[Fill]:
    out: list[Fill] = []
    for order in orders:
        out += engine.process(order)
        engine.check_invariant()
    return out


LAYERS = (ListBook, HeapBook, LadderBook)


def test_sweeps_levels_in_price_order() -> None:
    """An aggressive buy sweeps the cheapest ask first, then the next."""
    for layer in LAYERS:
        engine = Engine(layer)
        got = replay(
            engine,
            [
                Order(1, "sell", 10, 60),
                Order(2, "sell", 9, 40),
                Order(3, "buy", 10, 80),  # takes 40@9, then 40 of the 60@10
                Order(4, "buy", 8, 50),  # 8 < 10: nothing to hit, rests
            ],
        )
        assert got == [Fill(3, 2, 9, 40), Fill(3, 1, 10, 40)], layer.__name__
        assert engine.sells.best().price == 10 and engine.sells.best().qty == 20
        assert engine.buys.best().price == 8 and engine.buys.best().qty == 50


def test_same_price_is_first_in_first_out() -> None:
    """Two asks at one price: the OLDER fills first. This is rule 2's teeth."""
    for layer in LAYERS:
        engine = Engine(layer)
        got = replay(
            engine,
            [
                Order(1, "sell", 10, 30),
                Order(2, "sell", 10, 30),
                Order(3, "buy", 10, 30),
            ],
        )
        assert got == [Fill(3, 1, 10, 30)], layer.__name__  # id 1, not id 2
        assert engine.sells.best().id == 2


def test_partial_fill_keeps_queue_position() -> None:
    """A partially filled resting order stays at the FRONT of its level. It
    was filled, not resubmitted -- losing its place would be a real bug."""
    for layer in LAYERS:
        engine = Engine(layer)
        replay(
            engine,
            [
                Order(1, "sell", 10, 100),
                Order(2, "sell", 10, 50),
                Order(3, "buy", 10, 40),  # partial-fills order 1, 60 left
            ],
        )
        assert engine.sells.best().id == 1 and engine.sells.best().qty == 60


def test_exact_fill_leaves_no_ghost() -> None:
    """An exactly-filled order is removed. A zero-quantity ghost left at the
    front of a level blocks every future match at that price -- the classic
    bug this scenario exists to catch."""
    for layer in LAYERS:
        engine = Engine(layer)
        got = replay(
            engine,
            [
                Order(1, "sell", 10, 40),
                Order(2, "sell", 10, 40),
                Order(3, "buy", 10, 40),  # exactly kills order 1
                Order(4, "buy", 10, 40),  # must still reach order 2
            ],
        )
        assert got == [Fill(3, 1, 10, 40), Fill(4, 2, 10, 40)], layer.__name__
        assert engine.sells.best() is None


def test_empty_book_rests_everything() -> None:
    """Nothing to match: everything rests, with no special-case code."""
    for layer in LAYERS:
        engine = Engine(layer)
        got = replay(engine, [Order(1, "buy", 5, 10), Order(2, "buy", 7, 10)])
        assert got == [] and engine.buys.best().price == 7  # best bid = highest


def test_layers_agree_on_a_long_random_stream() -> None:
    """The real proof that the interface holds: a deterministic pseudo-random
    stream through all three layers must produce byte-identical fill logs."""
    import random

    rng = random.Random(7)
    stream = [
        Order(i, rng.choice(("buy", "sell")), rng.randint(80, 120), rng.randint(1, 50))
        for i in range(1, 400)
    ]
    logs = []
    for layer in LAYERS:
        engine = Engine(layer)
        # Fresh Order objects per layer: process() mutates quantities.
        replay(engine, [Order(o.id, o.side, o.price, o.qty) for o in stream])
        logs.append(engine.history)
    assert logs[0] == logs[1] == logs[2]
    assert logs[0], "the stream should produce at least one fill"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print(f"ok  {test.__name__}")
    print(f"\n{len(tests)} scenarios passed across {len(LAYERS)} book layers")
