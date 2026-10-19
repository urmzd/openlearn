# Jane Street Software Engineer Interview Guide

Comprehensive preparation for Jane Street SWE roles, with focus on systems engineering, quantitative thinking, and functional programming.

## Interview Process Overview

Timeline: **3-6 weeks**, **5-7 rounds**

| Round | Format | Duration | Focus |
|-------|--------|----------|-------|
| Recruiter Screen | Phone | 20 min | Background, interest in trading |
| Phone Screen 1 | Technical | 60 min | Problem-solving, coding |
| Phone Screen 2 | Technical | 60 min | Systems / math reasoning |
| Onsite 1 | Coding | 90 min | OCaml or systems problem |
| Onsite 2 | Systems Design | 90 min | Trading systems, real-time |
| Onsite 3 | Probability / Math | 60 min | Quantitative reasoning |
| Onsite 4 | Behavioral / Fit | 45 min | Culture, collaboration |

Jane Street interviews are **longer and more conversational** than typical tech interviews. They care about how you think, not just what answer you reach.

## Compensation

Jane Street pays significantly above Big Tech:

- **New Grad SWE**: ~$400-500K total comp (Year 1)
- **Senior SWE**: ~$600K-1.5M+ total comp
- **Principal/Senior**: $1M-3M+
- Compensation is heavily bonus-weighted (base is a fraction of total)
- No equity -- it's a partnership, comp is cash + bonus

## Key Themes

1. **Think out loud** -- Jane Street values the thought process above all. Silence is bad. Even wrong-but-reasoned approaches earn credit.
2. **OCaml is the primary language** -- Most production code is OCaml. You don't need to know it before interviewing, but familiarity with functional programming helps enormously.
3. **Quantitative reasoning** -- Probability, expected value, and mathematical thinking permeate everything.
4. **Real-time systems** -- Trading systems have microsecond-level latency requirements. Understand low-latency patterns.
5. **No leetcode** -- Problems are original and open-ended. They test reasoning, not pattern matching.
6. **Collaborative interviews** -- Interviewers actively help you. It's a conversation, not an exam.
7. **AI/ML in trading** -- Signal generation, execution optimization, anomaly detection. ML is a growing area.

## Problem-Solving Approach

Jane Street problems are designed to be solved **through discussion**. The interviewer will:
- Give hints if you're stuck
- Ask you to extend your solution
- Probe edge cases
- Ask "what if we change this constraint?"

**Your job**: Think clearly, communicate your reasoning, and be responsive to hints.

## Coding Rounds

### What to Expect

- **Language**: OCaml preferred, but Python/C++ accepted
- **Style**: Clean, functional style valued. Immutability, pattern matching, type safety.
- **Problems**: Open-ended, often with real-world trading flavor
- **Duration**: Longer sessions (60-90 min), deeper exploration of one problem

### OCaml Essentials

If you want to stand out, learn basic OCaml:

```ocaml
(* Pattern matching *)
let rec map f = function
  | [] -> []
  | x :: xs -> f x :: map f xs

(* Option types (no null!) *)
let find_first pred lst =
  match List.find_opt pred lst with
  | Some x -> x
  | None -> failwith "not found"

(* Records *)
type order = {
  symbol : string;
  price : float;
  quantity : int;
  side : [`Buy | `Sell];
}

(* Pipe operator for readability *)
let process orders =
  orders
  |> List.filter (fun o -> o.price > 100.0)
  |> List.sort (fun a b -> compare a.price b.price)
  |> List.map (fun o -> o.symbol)
```

### Reported Problem Types

#### 1. Order Book Implementation

```python
from collections import defaultdict
from sortedcontainers import SortedDict

class OrderBook:
    """Limit order book with price-time priority."""

    def __init__(self):
        self.bids = SortedDict()   # price -> list of (timestamp, quantity, order_id)
        self.asks = SortedDict()   # price -> list of (timestamp, quantity, order_id)
        self.orders = {}           # order_id -> (side, price)
        self._ts = 0

    def add_order(self, order_id: str, side: str, price: float, quantity: int) -> list:
        """Add an order. Returns list of fills [(price, quantity, passive_order_id)]."""
        self._ts += 1
        fills = []

        if side == "buy":
            fills = self._match(self.asks, price, quantity, ascending=True)
            remaining = quantity - sum(f[1] for f in fills)
            if remaining > 0:
                if price not in self.bids:
                    self.bids[price] = []
                self.bids[price].append((self._ts, remaining, order_id))
                self.orders[order_id] = ("buy", price)
        else:
            fills = self._match(self.bids, price, quantity, ascending=False)
            remaining = quantity - sum(f[1] for f in fills)
            if remaining > 0:
                if price not in self.asks:
                    self.asks[price] = []
                self.asks[price].append((self._ts, remaining, order_id))
                self.orders[order_id] = ("sell", price)

        return fills

    def _match(self, book, price, quantity, ascending) -> list:
        fills = []
        remaining = quantity
        keys_to_remove = []

        iterator = book.keys() if ascending else reversed(book.keys())
        for book_price in iterator:
            if ascending and book_price > price:
                break
            if not ascending and book_price < price:
                break
            if remaining <= 0:
                break

            level = book[book_price]
            while level and remaining > 0:
                ts, qty, oid = level[0]
                fill_qty = min(remaining, qty)
                fills.append((book_price, fill_qty, oid))
                remaining -= fill_qty
                if fill_qty == qty:
                    level.pop(0)
                    if oid in self.orders:
                        del self.orders[oid]
                else:
                    level[0] = (ts, qty - fill_qty, oid)

            if not level:
                keys_to_remove.append(book_price)

        for k in keys_to_remove:
            del book[k]

        return fills

    def cancel_order(self, order_id: str) -> bool:
        if order_id not in self.orders:
            return False
        side, price = self.orders.pop(order_id)
        book = self.bids if side == "buy" else self.asks
        if price in book:
            book[price] = [(ts, q, oid) for ts, q, oid in book[price] if oid != order_id]
            if not book[price]:
                del book[price]
        return True

    def best_bid(self) -> float | None:
        return self.bids.keys()[-1] if self.bids else None

    def best_ask(self) -> float | None:
        return self.asks.keys()[0] if self.asks else None

    def spread(self) -> float | None:
        bb, ba = self.best_bid(), self.best_ask()
        return ba - bb if bb is not None and ba is not None else None
```

#### 2. Market Data Parser / Feed Handler

```python
from dataclasses import dataclass
from typing import Callable

@dataclass
class MarketUpdate:
    symbol: str
    timestamp: int  # nanoseconds
    bid_price: float
    bid_size: int
    ask_price: float
    ask_size: int

class FeedHandler:
    """Process market data feed with gap detection and dedup."""

    def __init__(self):
        self.last_seq: dict[str, int] = {}
        self.callbacks: list[Callable] = []
        self.gap_buffer: dict[str, dict[int, MarketUpdate]] = {}

    def register(self, callback: Callable[[MarketUpdate], None]):
        self.callbacks.append(callback)

    def on_message(self, symbol: str, seq_num: int, update: MarketUpdate):
        expected = self.last_seq.get(symbol, 0) + 1

        if seq_num < expected:
            return  # Duplicate, ignore

        if seq_num > expected:
            # Gap detected -- buffer the message
            if symbol not in self.gap_buffer:
                self.gap_buffer[symbol] = {}
            self.gap_buffer[symbol][seq_num] = update
            return

        # seq_num == expected: process it and drain buffer
        self._dispatch(update)
        self.last_seq[symbol] = seq_num

        # Check if buffered messages can now be processed
        buf = self.gap_buffer.get(symbol, {})
        next_seq = seq_num + 1
        while next_seq in buf:
            self._dispatch(buf.pop(next_seq))
            self.last_seq[symbol] = next_seq
            next_seq += 1

    def _dispatch(self, update: MarketUpdate):
        for cb in self.callbacks:
            cb(update)
```

#### 3. Expression Evaluator / Mini Language

```python
from typing import Union

Token = Union[int, float, str]

def tokenize(expr: str) -> list[Token]:
    tokens = []
    i = 0
    while i < len(expr):
        if expr[i].isspace():
            i += 1
        elif expr[i].isdigit() or (expr[i] == '-' and (not tokens or tokens[-1] in '(+-*/')):
            j = i + 1
            while j < len(expr) and (expr[j].isdigit() or expr[j] == '.'):
                j += 1
            tokens.append(float(expr[i:j]))
            i = j
        elif expr[i] in '+-*/()':
            tokens.append(expr[i])
            i += 1
        else:
            raise ValueError(f"Unexpected character: {expr[i]}")
    return tokens

def evaluate(expr: str) -> float:
    """Evaluate arithmetic expression with operator precedence."""
    tokens = tokenize(expr)
    pos = [0]

    def parse_expr():
        result = parse_term()
        while pos[0] < len(tokens) and tokens[pos[0]] in ('+', '-'):
            op = tokens[pos[0]]
            pos[0] += 1
            right = parse_term()
            result = result + right if op == '+' else result - right
        return result

    def parse_term():
        result = parse_factor()
        while pos[0] < len(tokens) and tokens[pos[0]] in ('*', '/'):
            op = tokens[pos[0]]
            pos[0] += 1
            right = parse_factor()
            result = result * right if op == '*' else result / right
        return result

    def parse_factor():
        if tokens[pos[0]] == '(':
            pos[0] += 1
            result = parse_expr()
            pos[0] += 1  # skip ')'
            return result
        else:
            val = tokens[pos[0]]
            pos[0] += 1
            return val

    return parse_expr()
```

## Probability / Math Round

### What to Expect

- Brain teasers with rigorous mathematical reasoning
- Expected value calculations
- Probability and combinatorics
- Game theory scenarios
- Market-making scenarios

### Classic Problems

#### Expected Value

**"You flip a fair coin until you get heads. You get $2^n where n is the number of flips. What's the expected payout?"**

E = sum(2^n * (1/2)^n for n in 1..inf) = sum(1 for n in 1..inf) = infinity (St. Petersburg paradox)

Follow-up: "Would you pay $1M to play? Why not?" -- Risk aversion, utility theory, practical bounds.

#### Probability

**"You have 100 coins in a bag. 99 are fair, 1 is double-headed. You pick one and flip it 10 times, getting all heads. What's the probability it's the double-headed coin?"**

```
P(double | 10H) = P(10H | double) * P(double) / P(10H)
                = 1 * (1/100) / (1*(1/100) + (1/2)^10 * (99/100))
                = 0.01 / (0.01 + 99/102400)
                = 0.01 / (0.01 + 0.000967...)
                ≈ 0.912
```

#### Market Making

**"I'm going to flip a coin. If heads, this stock is worth $100. If tails, $0. Make a market."**

You bid $45, offer $55 (or tighter). Discuss:
- Why a spread? (Compensation for adverse selection risk)
- What if I flip and tell you the result, then ask again? (Update based on information)
- What if you must make a market 1000 times? (Law of large numbers, optimal spread)

### Topics to Study

- Bayes' theorem and conditional probability
- Expected value and variance
- Markov chains
- Basic combinatorics (permutations, combinations, stars and bars)
- Betting strategies and Kelly criterion
- Arbitrage detection

## System Design Round

### Trading System Design

#### Design a Low-Latency Trading System

```
[Market Data Feed] --> [Feed Handler] --> [Signal Generator]
                                                 |
                                          [Risk Engine]
                                                 |
                                          [Order Manager]
                                                 |
                                          [Exchange Gateway]
```

Key requirements:
- **Latency**: Tick-to-trade < 10 microseconds
- **Deterministic**: No GC pauses, no page faults
- **Fault tolerance**: Failover without losing orders

Design decisions:
- **Kernel bypass**: DPDK / RDMA for network I/O, bypass kernel TCP/IP stack
- **Lock-free queues**: SPSC (single producer, single consumer) ring buffers
- **Memory pre-allocation**: No malloc in hot path, pre-allocate all buffers
- **CPU pinning**: Pin threads to cores, isolate from OS scheduler
- **NUMA awareness**: Allocate memory close to the CPU that uses it
- **No logging in hot path**: Log asynchronously, buffer in shared memory

#### Design a Risk Management System

- **Pre-trade checks**: Position limits, order size limits, price bands
- **Real-time P&L**: Mark positions to market continuously
- **Exposure tracking**: Net exposure by asset, sector, geography
- **Circuit breakers**: Halt trading if losses exceed threshold
- **Latency requirement**: Risk check must add < 1 microsecond

#### Design a Market Data Distribution System

- **Multicast**: UDP multicast for market data (one-to-many, no TCP overhead)
- **Sequencing**: Sequence numbers for gap detection
- **Conflation**: During bursts, send only latest price (not every tick)
- **Recovery**: Request/response channel for filling gaps
- **Normalization**: Normalize different exchange formats into unified schema

## Behavioral / Fit

### Jane Street Culture

- **Intellectual curiosity** -- They want people who find hard problems genuinely fun
- **Collaborative** -- No lone wolves. Trading is inherently team-based.
- **Humble** -- Confident but not arrogant. Willing to say "I don't know."
- **Pragmatic** -- Simple, correct solutions over clever ones

### Common Questions

- "What's the most interesting technical problem you've worked on?"
- "Tell me about a time you were wrong about something technical."
- "How do you approach learning a new domain?"
- "What do you know about quantitative trading?"
- "Why Jane Street over a tech company?"

## AI/ML in Trading Context

For AI-focused roles at Jane Street:

- **Signal generation**: ML models for predicting price movements
- **Execution optimization**: Minimize market impact, optimal order splitting
- **Anomaly detection**: Identify unusual market behavior or system issues
- **Feature engineering**: Derive trading signals from raw market data
- **Online learning**: Models that adapt in real-time to market regime changes
- **Interpretability**: Understanding WHY a model makes predictions (regulatory and risk)

## Preparation Tips

1. **Learn functional programming** -- OCaml, Haskell, or at minimum, write Python in a functional style. Immutability, pattern matching, higher-order functions.
2. **Practice probability** -- Work through "A Practical Guide to Quantitative Finance Interviews" (the "green book") and Heard on the Street.
3. **Think out loud always** -- Practice verbalizing your thought process. Jane Street values this above answer correctness.
4. **Low-latency systems** -- Understand kernel bypass, lock-free programming, cache-friendly data structures.
5. **Study market microstructure** -- Order books, bid-ask spread, market making basics. You don't need a finance degree, but understand the fundamentals.
6. **Practice estimation** -- Fermi estimation, back-of-envelope calculations. "How many piano tuners in Chicago?"
7. **Be genuinely curious** -- Jane Street interviews are conversations. Ask questions about their problems. Show you find it interesting.

## Sources

- [Jane Street Careers](https://www.janestreet.com/join-jane-street/)
- [Jane Street Tech Blog](https://blog.janestreet.com/)
- [Jane Street Puzzles](https://www.janestreet.com/puzzles/)
- [Glassdoor - Jane Street SWE Interview Questions](https://www.glassdoor.com/Interview/Jane-Street-Software-Engineer-Interview-Questions-EI_IE255549.0,11_KO12,29.htm)
- [A Practical Guide to Quantitative Finance Interviews (Green Book)](https://www.amazon.com/Practical-Guide-Quantitative-Finance-Interviews/dp/1438236662)
- [Wall Street Oasis - Quant Firm Discussions](https://www.wallstreetoasis.com/forum/hedge-fund)
- r/cscareerquestions, Blind (community reports)
