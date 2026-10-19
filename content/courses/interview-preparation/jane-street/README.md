# Jane Street

Jane Street is a quantitative trading firm whose engineering culture is unlike almost anywhere else: most of its production code is written in OCaml, its interview problems are original rather than drawn from LeetCode, and its interviewers care more about how you think than about the answer you reach. Interviews are longer and more conversational than typical tech interviews, and they mix coding with probability, market-making puzzles, and the design of real-time trading systems.

This chapter prepares you for Jane Street SWE roles with a focus on systems engineering, quantitative thinking, and functional programming. It describes the process and compensation, explains how to approach a problem that is meant to be solved through discussion, and then works through the coding rounds (OCaml essentials, an order book, a feed handler, and an expression evaluator), the probability and market-making round, trading system design, and the behavioral round.

> **Learning objectives**
> - Explain how Jane Street's conversational interview style changes the way you should work a problem.
> - Read and write basic OCaml: pattern matching, options, records, and the pipe operator.
> - Implement a limit order book with price-time priority, a feed handler with gap detection, and a recursive-descent expression evaluator.
> - Solve expected-value, Bayesian, and market-making problems and explain the reasoning aloud.
> - Design a low-latency trading pipeline, a risk system, and a market data distribution system.

## Before you begin

Comfort with conditional probability and expected value is assumed. Low-latency systems concepts recur from the [Hudson River Trading](../hrt/README.md) chapter, which goes deeper into hardware, and [Distributed systems](../shared-concepts/distributed-systems.md) covers the sequencing and recovery ideas behind market data feeds.

## The interview process

Expect three to six weeks and five to seven rounds:

| Round | Format | Duration | Focus |
|-------|--------|----------|-------|
| Recruiter Screen | Phone | 20 min | Background, interest in trading |
| Phone Screen 1 | Technical | 60 min | Problem-solving, coding |
| Phone Screen 2 | Technical | 60 min | Systems / math reasoning |
| Onsite 1 | Coding | 90 min | OCaml or systems problem |
| Onsite 2 | Systems Design | 90 min | Trading systems, real-time |
| Onsite 3 | Probability / Math | 60 min | Quantitative reasoning |
| Onsite 4 | Behavioral / Fit | 45 min | Culture, collaboration |

Jane Street pays significantly above Big Tech. A new-grad SWE earns about $400-500K in total compensation in the first year, a senior SWE about $600K-1.5M or more, and principal and senior staff $1M-3M or more. Compensation is heavily bonus-weighted, with base salary a fraction of the total. There is no equity: the firm is a partnership, so compensation is cash plus bonus.

## What Jane Street looks for

The single most important theme is to think out loud. Jane Street values the thought process above all; silence is bad, and even a wrong but well-reasoned approach earns credit. The rest follows from the firm's work. OCaml is the primary language for most production code; you do not need to know it before interviewing, but familiarity with functional programming helps enormously. Quantitative reasoning (probability, expected value, mathematical thinking) permeates everything. Trading systems have microsecond-level latency requirements, so you should understand low-latency patterns. There is no LeetCode: problems are original and open-ended, testing reasoning rather than pattern matching. The interviews are collaborative, with interviewers actively helping you, more conversation than exam. And AI and ML are a growing area in trading, for signal generation, execution optimization, and anomaly detection.

That collaborative style deserves a word on its own, because it changes how you should work. Jane Street problems are designed to be solved through discussion. The interviewer will give hints if you are stuck, ask you to extend your solution, probe edge cases, and ask "what if we change this constraint?" Your job is to think clearly, communicate your reasoning, and respond to the hints. A candidate who ignores a hint to defend their first idea does worse than one who takes it and runs.

## Coding rounds

OCaml is preferred, but Python and C++ are accepted. A clean, functional style is valued: immutability, pattern matching, and type safety. Problems are open-ended and often have a real-world trading flavour, and the sessions are long (60-90 minutes), exploring one problem deeply rather than several shallowly.

### OCaml essentials

If you want to stand out, learn basic OCaml. The listing shows the four features you will use most: recursive functions defined by pattern matching on lists, option types in place of null, records (here an order with a polymorphic variant for its side), and the pipe operator `|>`, which reads a transformation top to bottom.

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

Three problem types are reported often enough to practise. All three are given here in Python, since that is the most common fallback language.

### An order book

The classic problem is a limit order book with price-time priority. Bids and asks are each a sorted map from price to a queue of resting orders, ordered by arrival timestamp. An incoming buy matches against asks from the lowest price upward, as long as the ask is at or below its limit; an incoming sell matches against bids from the highest price downward. Whatever quantity is left rests in the book at the order's own price. Each fill is reported as `(price, quantity, passive_order_id)`, at the resting order's price.

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

Notice two details that interviewers like to probe. Emptied price levels are collected in `keys_to_remove` and deleted after the loop, because deleting from a `SortedDict` while iterating over its keys is unsafe. And the `orders` index lets `cancel_order` find an order's level directly, though removal within the level is still a linear scan; a natural follow-up is to make cancellation $O(1)$ with an intrusive linked list per level. Using `list.pop(0)` for the front of a level is also $O(n)$, and a `deque` would fix it.

### A market data feed handler

The second problem is a feed handler that processes a sequenced market data feed per symbol. Each message carries a sequence number. A number below the expected one is a duplicate and is dropped. A number above it means a gap: messages were lost or reordered, so the handler buffers the early message rather than dispatching it out of order. When the expected message arrives, it is dispatched and the buffer is drained for as long as the next sequence number is present.

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

The obvious extension is what happens if the missing message never arrives. A production handler would start a timer on gap detection and, when it fires, request the missing range over a recovery channel or resynchronise from a snapshot; this is the "recovery" piece of the market data distribution design later in the chapter.

### An expression evaluator

The third problem is a small language: evaluate arithmetic with the usual precedence. The tokenizer treats a `-` as part of a number when it appears at the start or after an operator or opening parenthesis, which handles unary minus on literals. The evaluator is a recursive-descent parser with one function per precedence level: `parse_expr` handles `+` and `-` over terms, `parse_term` handles `*` and `/` over factors, and `parse_factor` handles a parenthesised expression or a number. Because each level loops left to right, the operators are left-associative.

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

> **Warning:** The tokenizer's check `tokens[-1] in '(+-*/'` only works because the previous token is a string. When the previous token is a number, `float in str` raises a `TypeError`, so an input like `3 - 2` fails at the minus sign. Checking `isinstance(tokens[-1], str)` first fixes it. The parser also skips `)` without checking it, so unbalanced input is not reported. These are exactly the edge cases a Jane Street interviewer will ask you to find.

## The probability and math round

This round consists of brain teasers that demand rigorous mathematical reasoning: expected value calculations, probability and combinatorics, game theory, and market-making scenarios. Three classic problems show the range.

### Expected value: the St. Petersburg paradox

"You flip a fair coin until you get heads. You get $\$2^n$, where $n$ is the number of flips. What's the expected payout?" The game ends on flip $n$ with probability $(1/2)^n$, so

$$E = \sum_{n=1}^{\infty} 2^n \left(\tfrac{1}{2}\right)^n = \sum_{n=1}^{\infty} 1 = \infty.$$

The follow-up is the interesting part: "Would you pay $1M to play? Why not?" The answer turns on risk aversion and utility theory (the expected utility of the payout is finite for a concave utility), and on practical bounds, since no counterparty can pay out an arbitrarily large sum.

### Bayesian updating: the double-headed coin

"You have 100 coins in a bag. 99 are fair, 1 is double-headed. You pick one and flip it 10 times, getting all heads. What's the probability it's the double-headed coin?" Apply Bayes' theorem:

```
P(double | 10H) = P(10H | double) * P(double) / P(10H)
                = 1 * (1/100) / (1*(1/100) + (1/2)^10 * (99/100))
                = 0.01 / (0.01 + 99/102400)
                = 0.01 / (0.01 + 0.000967...)
                ≈ 0.912
```

Ten heads are about a thousand times likelier from the double-headed coin, which overwhelms its 1-in-100 prior.

### Market making

"I'm going to flip a coin. If heads, this stock is worth $100. If tails, $0. Make a market." The fair value is $50, so you might bid $45 and offer $55, or tighter. Then discuss the follow-ups. Why a spread at all? It is compensation for adverse selection risk: the person trading with you may know something you do not. What if I flip and tell you the result, then ask again? You update on the information, and the market collapses to $0 or $100. What if you must make a market 1000 times? The law of large numbers lets you narrow the spread, and the question becomes what spread is optimal.

To prepare for this round, study Bayes' theorem and conditional probability, expected value and variance, Markov chains, basic combinatorics (permutations, combinations, stars and bars), betting strategies and the Kelly criterion, and arbitrage detection.

## The system design round

System design at Jane Street is trading system design, where the constraints are latency, determinism, and never losing an order.

### A low-latency trading system

```
[Market Data Feed] --> [Feed Handler] --> [Signal Generator]
                                                 |
                                          [Risk Engine]
                                                 |
                                          [Order Manager]
                                                 |
                                          [Exchange Gateway]
```

The requirements are a tick-to-trade latency under 10 microseconds, deterministic behaviour with no GC pauses or page faults, and failover without losing orders. Each design decision serves those requirements. Network I/O uses kernel bypass (DPDK or RDMA) to skip the kernel's TCP/IP stack. Stages communicate through lock-free single-producer single-consumer ring buffers. All buffers are preallocated, with no `malloc` in the hot path. Threads are pinned to cores and isolated from the OS scheduler, and memory is allocated NUMA-aware, close to the CPU that uses it. There is no logging in the hot path; logs are written asynchronously from a buffer in shared memory.

### A risk management system

A risk system performs pre-trade checks (position limits, order size limits, price bands), computes real-time P&L by marking positions to market continuously, tracks net exposure by asset, sector, and geography, and trips circuit breakers that halt trading if losses exceed a threshold. Because it sits in the order path, the risk check must add less than a microsecond.

### Market data distribution

Distributing market data internally uses UDP multicast, which is one-to-many without TCP overhead. Sequence numbers allow gap detection, as in the feed handler above. During bursts, conflation sends only the latest price rather than every tick. A separate request/response channel recovers missed messages, and a normalisation layer converts different exchange formats into one unified schema.

## The behavioral round

Jane Street's culture has four traits that this round looks for: intellectual curiosity, meaning you find hard problems genuinely fun; collaboration, since trading is inherently team-based and there are no lone wolves; humility, being confident but not arrogant and willing to say "I don't know"; and pragmatism, preferring simple, correct solutions over clever ones.

Common questions include "What's the most interesting technical problem you've worked on?", "Tell me about a time you were wrong about something technical.", "How do you approach learning a new domain?", "What do you know about quantitative trading?", and "Why Jane Street over a tech company?"

## AI and ML in trading

For AI-focused roles, be ready to discuss how ML fits into trading: signal generation with models that predict price movements; execution optimisation that minimises market impact through optimal order splitting; anomaly detection for unusual market behaviour or system issues; feature engineering that derives signals from raw market data; online learning, with models that adapt in real time to regime changes; and interpretability, understanding why a model makes a prediction, which matters for regulators and for risk.

## Preparing

Learn functional programming: OCaml, Haskell, or at minimum Python written in a functional style, with immutability, pattern matching, and higher-order functions. Practise probability with *A Practical Guide to Quantitative Finance Interviews* (the "green book") and *Heard on the Street*. Practise thinking out loud until it is habitual, because Jane Street values it above answer correctness. Learn the low-latency fundamentals (kernel bypass, lock-free programming, cache-friendly data structures) and the basics of market microstructure: order books, the bid-ask spread, and market making. You do not need a finance degree, but you should understand the fundamentals. Practise Fermi estimation and back-of-envelope calculation ("How many piano tuners in Chicago?"). And be genuinely curious: these interviews are conversations, so ask about the interviewer's problems and show that you find them interesting.

## Summary

- Interviews are long, collaborative conversations; reasoning aloud and using hints matters more than reaching the answer alone.
- OCaml is the house language, and a functional style is valued in any language.
- Practise the order book, the sequenced feed handler, and recursive-descent parsing, and know their edge cases.
- Probability problems reward exact setup: expected value sums, Bayes' theorem, and spread as compensation for adverse selection.
- Trading system design is about microsecond determinism: kernel bypass, lock-free queues, preallocation, and pinning.

## Further reading

- [Jane Street Careers](https://www.janestreet.com/join-jane-street/)
- [Jane Street Tech Blog](https://blog.janestreet.com/)
- [Jane Street Puzzles](https://www.janestreet.com/puzzles/)
- [Glassdoor: Jane Street SWE Interview Questions](https://www.glassdoor.com/Interview/Jane-Street-Software-Engineer-Interview-Questions-EI_IE255549.0,11_KO12,29.htm)
- [A Practical Guide to Quantitative Finance Interviews (Green Book)](https://www.amazon.com/Practical-Guide-Quantitative-Finance-Interviews/dp/1438236662)
- [Wall Street Oasis: Quant Firm Discussions](https://www.wallstreetoasis.com/forum/hedge-fund)
- Community reports on r/cscareerquestions and Blind.
