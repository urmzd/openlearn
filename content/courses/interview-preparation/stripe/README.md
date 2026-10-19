# Interviewing at Stripe

Stripe runs one of the most distinctive interview processes in tech. There is no LeetCode: no algorithm puzzles, no trick questions about tree rotations. Instead you fix bugs in a real codebase, build a feature against a real API, write practical code that resembles Stripe's own work, and design payment infrastructure. The reason is the product. Stripe moves money, and a bug can mean real financial loss for real businesses, so the company hires for correctness, production-quality code, and clear communication rather than puzzle speed.

This chapter follows the process from the first call to the last onsite round. After the pipeline and the compensation, it lays out the themes every round shares, then works through each round in turn: practical coding (with a payment retry system and a tiered rate limiter you should be able to defend line by line), the bug bash, payment-focused system design and API design, the integration round, and the values-based behavioral round. It ends with a preparation plan.

> **Learning objectives**
> - Describe Stripe's five-round process and what makes each round different from a typical big-tech loop.
> - Explain idempotency, retries with exponential backoff, and sliding-window rate limiting through working code, including where the reference code falls short.
> - Approach a bug bash systematically and recognize the common bug categories, especially financial ones.
> - Design a payment, subscription billing, or webhook delivery system around financial invariants: double-entry bookkeeping, idempotency, reconciliation, and at-least-once delivery.
> - Recognize the conventions that make Stripe's API a model of good API design.

## Before you begin

You will get the most from this chapter if you are comfortable writing clean Python and reading unfamiliar code. The system design material builds on the [Distributed Systems](../shared-concepts/distributed-systems.md) chapter of this book, especially its treatment of exactly-once processing and retries, and the [Software Craftsmanship](../../software-craftsmanship/README.md) book is good background for the code-quality bar Stripe sets.

## In this chapter

- [The interview process](#the-interview-process) and [Compensation](#compensation): five rounds over two to six weeks, and what a senior offer looks like.
- [What Stripe is looking for](#what-stripe-is-looking-for): six themes, starting with "no LeetCode" and "correctness is sacred".
- [The practical coding screen](#the-practical-coding-screen): a payment retry system and an API rate limiter.
- [The bug bash](#the-bug-bash): Stripe's signature round, and the categories of bugs to hunt for.
- [The system design round](#the-system-design-round): payment processing, subscription billing, webhooks, and Stripe-style API design.
- [The integration round](#the-integration-round): building a feature against an unfamiliar API.
- [The behavioral round](#the-behavioral-round) and [Preparing well](#preparing-well): Stripe's values and how to spend your preparation time.

## The interview process

The process takes two to six weeks (faster with a referral) and has five rounds after the recruiter screen:

| Round | Format | Duration | Focus |
|-------|--------|----------|-------|
| Recruiter Screen | Phone | 30 min | Background, role fit |
| Phone Screen | Coding (practical) | 60 min | Real-world coding problem |
| Onsite 1 | Bug Bash | 60 min | Debug a failing codebase |
| Onsite 2 | System Design | 60 min | Payment infrastructure design |
| Onsite 3 | Integration | 60 min | Build with Stripe API |
| Onsite 4 | Behavioral / Manager | 45 min | Values, collaboration |

> **Warning:** Using AI tools during Stripe interviews is strictly prohibited.

## Compensation

For a senior engineer (L3) in the US, reported base salary runs \$190-230K, RSUs add roughly \$200-400K a year, and the bonus is 10-15%. That puts total compensation around \$400-650K at L3 and \$600-900K+ at L4 (Staff).

## What Stripe is looking for

The first theme is the one candidates find hardest to believe: Stripe explicitly does not ask algorithm puzzles. Every problem is practical and resembles real Stripe engineering work, so time spent grinding puzzles is time taken away from what will actually be tested.

The second theme explains the rest. Correctness is sacred, because Stripe processes money; the priority order is correctness, then speed, then elegance. That drives the third theme, production quality: clean code, error handling, edge cases, and testability are non-negotiable even in a timed round. It also drives the fourth, financial invariants. Double-entry accounting, idempotency, and exactly-once processing come up in several rounds, and you should know them cold.

The last two themes are about how you work. Stripe values engineers who communicate: who think out loud, explain tradeoffs clearly, and ask good questions. And it values full-stack thinking; even backend roles may touch APIs, developer experience, and documentation.

## The practical coding screen

The phone screen gives you a real-world problem rather than a puzzle. You might build a small service, a data processing pipeline, or an API handler. The code must be clean, well structured, and handle edge cases, and the interviewer evaluates your approach and communication as much as the final code. Two reported problems show the flavor.

### Building a payment retry system

The task is to charge a payment through a gateway, retrying transient failures, without ever charging the customer twice. Three ideas carry the solution. First, model the payment's lifecycle explicitly with a status enum, so that "pending", "succeeded", "failed", and "requires retry" are distinct states rather than implied by flags. Second, distinguish failures the gateway says are retryable (a timeout, a temporary network issue) from final ones (a declined card), and back off exponentially between retries so you do not hammer a struggling dependency. Third, use an idempotency key: the processor remembers the result for each key and returns it on a repeat call, and it passes the same key to the gateway so that the gateway also deduplicates retries of the same charge.

```python
import time
from enum import Enum
from dataclasses import dataclass, field

class PaymentStatus(Enum):
    PENDING = "pending"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    REQUIRES_RETRY = "requires_retry"

@dataclass
class PaymentAttempt:
    payment_id: str
    amount_cents: int
    currency: str
    status: PaymentStatus = PaymentStatus.PENDING
    attempt_count: int = 0
    max_attempts: int = 3
    idempotency_key: str = ""
    error: str = ""

class PaymentProcessor:
    def __init__(self, gateway):
        self.gateway = gateway
        self.processed: dict[str, PaymentAttempt] = {}

    def process(self, payment: PaymentAttempt) -> PaymentAttempt:
        """Process a payment with retry logic and idempotency."""
        # Idempotency check
        if payment.idempotency_key and payment.idempotency_key in self.processed:
            return self.processed[payment.idempotency_key]

        while payment.attempt_count < payment.max_attempts:
            payment.attempt_count += 1
            try:
                result = self.gateway.charge(
                    amount=payment.amount_cents,
                    currency=payment.currency,
                    idempotency_key=payment.idempotency_key,
                )
                if result.success:
                    payment.status = PaymentStatus.SUCCEEDED
                    break
                elif result.retryable:
                    payment.status = PaymentStatus.REQUIRES_RETRY
                    # Exponential backoff
                    time.sleep(min(2 ** payment.attempt_count, 30))
                else:
                    payment.status = PaymentStatus.FAILED
                    payment.error = result.error_message
                    break
            except Exception as e:
                payment.error = str(e)
                if payment.attempt_count >= payment.max_attempts:
                    payment.status = PaymentStatus.FAILED

        if payment.idempotency_key:
            self.processed[payment.idempotency_key] = payment
        return payment
```

Note that the amount is stored as integer cents, never as a float of dollars, and the backoff is capped at 30 seconds so the delay does not grow without bound.

This listing is a solid first pass, and in a Stripe interview the follow-up is usually "what's wrong with it?" You should be able to name the gaps yourself. If every attempt returns a retryable failure, the loop exits with the status still `REQUIRES_RETRY` rather than `FAILED`, and it sleeps once more after the final attempt for no benefit. An exception retries immediately with no backoff. Catching every `Exception` also treats a programming error the same as a network blip. Finally, the idempotency record is written only after processing finishes, so two concurrent calls with the same key can both pass the check and both reach the gateway; a production version records the key as "in progress" atomically before charging (and relies on the gateway-side key as a second line of defense). Pointing these out, and fixing one or two, is exactly the kind of correctness thinking the round rewards.

### Building a rate limiter for API keys

The second reported problem is a rate limiter with tiered limits: each API key belongs to a tier with a per-minute and a per-day cap. The reference solution keeps a sliding log, a list of request timestamps per key, and checks both windows against it on every request.

```python
import time
from collections import defaultdict

class RateLimiter:
    """Sliding window rate limiter for API keys with tiered limits."""

    def __init__(self):
        self.windows: dict[str, list[float]] = defaultdict(list)
        self.tiers: dict[str, dict] = {
            "free":       {"requests_per_min": 20,  "requests_per_day": 1000},
            "standard":   {"requests_per_min": 100, "requests_per_day": 10000},
            "enterprise": {"requests_per_min": 1000, "requests_per_day": 100000},
        }
        self.api_key_tiers: dict[str, str] = {}

    def register(self, api_key: str, tier: str = "free"):
        self.api_key_tiers[api_key] = tier

    def allow(self, api_key: str) -> tuple[bool, dict]:
        """Check if request is allowed. Returns (allowed, metadata)."""
        tier = self.api_key_tiers.get(api_key)
        if tier is None:
            return False, {"error": "unknown_api_key"}

        limits = self.tiers[tier]
        now = time.monotonic()

        # Clean old entries
        window = self.windows[api_key]
        one_day_ago = now - 86400
        self.windows[api_key] = [t for t in window if t > one_day_ago]
        window = self.windows[api_key]

        # Check per-minute limit
        one_min_ago = now - 60
        recent = sum(1 for t in window if t > one_min_ago)
        if recent >= limits["requests_per_min"]:
            retry_after = window[-limits["requests_per_min"]] + 60 - now
            return False, {
                "error": "rate_limit_exceeded",
                "limit": "per_minute",
                "retry_after_seconds": max(0, retry_after),
            }

        # Check per-day limit
        if len(window) >= limits["requests_per_day"]:
            return False, {
                "error": "rate_limit_exceeded",
                "limit": "per_day",
                "retry_after_seconds": window[0] + 86400 - now,
            }

        window.append(now)
        return True, {
            "remaining_minute": limits["requests_per_min"] - recent - 1,
            "remaining_day": limits["requests_per_day"] - len(window),
        }
```

Walk through the decisions in order. An unknown key is rejected with an explicit error rather than silently allowed. The clock is `time.monotonic()`, which cannot jump backward when the wall clock is adjusted. Entries older than a day are pruned first, so the list is exactly the last 24 hours of accepted requests, in order.

The `retry_after` arithmetic is the part interviewers like to probe. When the per-minute check fails, at least `limit` requests fall in the last minute, and they are the newest entries in the list. The client can send again once enough of them age out that fewer than `limit` remain, which happens when the entry at index `-limit` (the oldest of the newest `limit` requests) passes the 60-second mark: at `window[-limit] + 60`. The daily case uses the same reasoning with the oldest entry, `window[0] + 86400`. Rejected requests are not appended, so being throttled does not extend the throttle.

The tradeoff to raise yourself is memory: a sliding log stores one timestamp per accepted request, up to 100,000 per enterprise key per day, and each call rescans the list. That is fine for an interview, but at scale you would move to a sliding-window counter or a token bucket, store state in a shared store such as Redis so that every API server sees the same counts, and make the check-and-increment atomic.

## The bug bash

The bug bash is Stripe's most distinctive round. You are given a real codebase, usually in Python, Ruby, or Go, with several failing tests. In 60 minutes you read the code, work out the intended behavior, find the bugs, and fix them, and there is more than one bug to find.

The interviewers are testing four things. Can you read unfamiliar code quickly? Do you debug methodically, reading the error messages and narrowing the search, rather than guessing? Do you notice details such as off-by-one errors, missing null checks, and type mismatches? And are your fixes clean rather than hacky patches that make one test pass?

It helps to carry a mental checklist of where bugs usually hide. Here is the one worth memorizing:

```
1. Off-by-one errors
   - Loop bounds (< vs <=)
   - Array indexing
   - String slicing

2. Null/None handling
   - Missing null checks
   - Optional values used without guards

3. Type coercion
   - String vs integer comparison
   - Float precision (money should be cents, not dollars)

4. Concurrency bugs
   - Race conditions
   - Missing locks
   - Deadlocks

5. Logic errors
   - Inverted conditions
   - Wrong operator (AND vs OR)
   - Missing break in switch/match

6. Edge cases
   - Empty collections
   - Single element
   - Boundary values (0, negative, max int)

7. Financial bugs (Stripe-specific)
   - Rounding errors (always use integer cents)
   - Currency conversion precision
   - Double-charging (missing idempotency)
```

The way to prepare is to practice the activity itself. Read open-source code and look for bugs. Do code review exercises on real pull requests. Practice debugging without a debugger, tracing the code in your head. Most of all, work through failing tests systematically: read the test, understand the behavior it expects, then trace the code path it exercises until you find where reality diverges. (The payment retry listing above makes a good warm-up: it has several of the bugs on this list.)

## The system design round

Stripe's system design round is lower-level and more practical than the usual big-tech version. Instead of drawing boxes for a social network at planetary scale, you design the API (RESTful endpoints, request and response schemas, error codes), trace how money flows through the system, and show how it maintains financial invariants (double-entry accounting, idempotency, reconciliation) and stays reliable, including exactly-once processing in a distributed system. Three prompts come up often.

### Designing a payment processing system

A charge flows from the merchant's API call through a gateway that checks idempotency first, so a retried request returns the cached result instead of charging again. New requests go through the payment service, a risk and fraud engine, and a router that picks the payment method. A card payment travels over the card network to the issuing bank, and the response is recorded in the ledger before a webhook tells the merchant what happened:

```
[Merchant API Call]
       |
  [API Gateway] --> [Idempotency Check]
       |                    |
  [Payment Service]    [Already processed? Return cached result]
       |
  [Risk/Fraud Engine]
       |
  [Payment Method Router]
       |
  +----+----+----+
  |    |    |    |
[Card] [Bank] [Wallet] [Crypto]
  |
[Card Network (Visa/MC)]
  |
[Issuing Bank]
  |
[Response] --> [Ledger Entry] --> [Webhook Dispatch]
```

Four financial concepts hold the design together. Double-entry bookkeeping records every transaction as matching debits and credits whose sum is always zero; a \$100 payment with a \$2.90 processing fee looks like this:

```
Payment of $100:
  Debit:  Merchant Receivable  +$100
  Credit: Platform Revenue     -$97.10
  Credit: Processing Fee       -$2.90
```

Because the entries must sum to zero, any bug that creates or destroys money shows up as an unbalanced ledger. Idempotency means every API call carries an idempotency key, and retrying with the same key returns the same result. Reconciliation periodically checks that the internal ledger matches external bank records. And settlement transfers funds to merchants in batches, daily or weekly.

### Designing a subscription billing system

Subscription billing is a state machine driven by a clock. A billing clock, a cron-like system, fires at each subscription's interval; an invoice is generated; a payment is attempted; and on failure, retry logic and dunning take over before the subscription's status is updated:

```
[Subscription Created] --> [Billing Clock]
                                |
                          [Invoice Generator]
                                |
                          [Payment Attempt]
                                |
                     [Success] or [Retry Logic]
                                |
                          [Dunning Management]
                          (email reminders, grace period)
                                |
                          [Subscription Status Update]
```

The details interviewers probe are proration (charging fairly when a customer changes plans mid-cycle), dunning (retrying failed payments on a schedule such as 1, 3, and 7 days, with email reminders and a grace period), and tax calculation, whose rules depend on the customer's region.

### Designing a webhook delivery system

Webhooks are how Stripe tells merchants that something happened, and their design is a study in unreliable networks. Delivery is at-least-once: retry with exponential backoff until the merchant acknowledges, which means merchants must handle duplicates. Ordering per resource is possible but complex, so treat it as optional. Each payload carries an HMAC signature so merchants can verify it came from Stripe. An endpoint that fails N times in a row is disabled and the merchant is notified, and merchants can replay events they missed.

### Designing a Stripe-style API

Stripe's API is widely considered one of the best designed in the industry, and the design round often touches its conventions. Resources are RESTful with consistent naming, actions on a resource are sub-paths, errors share one structured format, related objects can be expanded inline, writes accept an idempotency key header, and the API version is pinned by a header:

```python
# Good Stripe-style API design:

# 1. RESTful with consistent naming
POST   /v1/payments            # Create a payment
GET    /v1/payments/:id        # Retrieve a payment
POST   /v1/payments/:id/cancel # Action on a resource
GET    /v1/payments?limit=10   # List with pagination

# 2. Consistent error format
{
    "error": {
        "type": "card_error",
        "code": "card_declined",
        "message": "Your card was declined.",
        "param": "card_number",
        "decline_code": "insufficient_funds"
    }
}

# 3. Expandable objects
GET /v1/payments/py_123?expand[]=customer&expand[]=invoice

# 4. Idempotency
POST /v1/payments
Idempotency-Key: unique-key-123

# 5. Versioning via header
Stripe-Version: 2024-12-18
```

Each convention solves a real problem: the error format gives clients a machine-readable `type` and `code` plus the offending `param`; expansion saves round trips without bloating every response; and date-based version headers let Stripe evolve the API without breaking existing integrations.

## The integration round

In the integration round you get access to the Stripe API, or a mock of it, and implement a feature with it, such as adding subscription upgrade and downgrade support. You navigate an unfamiliar codebase, read the API documentation to find the endpoints you need, implement the feature, and write tests for it. The round tests API fluency, full-stack thinking from frontend to backend to external service, your instinct to test, and whether you can extend an existing codebase cleanly.

The single best preparation is to use the Stripe API before the interview. Sign up, get test keys, and make real calls. Build a small project, such as a checkout flow or subscription management, and practice reading API documentation and implementing against it.

## The behavioral round

The behavioral round is assessed against Stripe's values, and each one translates into something specific the interviewer is looking for:

| Value | What They Assess |
|-------|-----------------|
| Users first | Empathy for developers using Stripe |
| Move with urgency | Ship quickly without sacrificing quality |
| Think rigorously | First-principles thinking, data-driven |
| Trust and amplify | Collaborate, uplift teammates |
| Global optimization | What's best for Stripe, not just your team |

Prepare concrete stories for the common questions:

- "Tell me about a time you found and fixed a subtle bug in production."
- "Describe a system you built that had to be correct, not just fast."
- "How do you approach code review?"
- "Tell me about a time you simplified a complex system."
- "Describe a time you had to make a decision between two imperfect options."

## Preparing well

Take the "no LeetCode" message seriously and spend your time on practical coding instead. Use the Stripe API and build something real, so that payments, customers, invoices, and webhooks are familiar. Practice debugging: take open-source repositories, introduce bugs, and fix them, and get used to reading code you did not write. Learn the financial-systems fundamentals: double-entry bookkeeping, idempotency, ACID transactions, and reconciliation. Study Stripe's API as a model of design through its [API reference](https://stripe.com/docs/api), and read the infrastructure posts on the [Stripe engineering blog](https://stripe.com/blog/engineering). Above all, write clean code under time pressure: good names, real error handling, covered edge cases, and comments where the code is not obvious.

## Key terms

- **Idempotency key**: a client-chosen identifier that makes a repeated request return the original result instead of performing the operation again.
- **Exponential backoff**: retrying after delays that grow geometrically (here $2^n$ seconds, capped at 30) to avoid overloading a failing dependency.
- **Sliding log**: a rate-limiting method that stores each request's timestamp and counts those inside the window.
- **Double-entry bookkeeping**: recording every transaction as debits and credits that sum to zero.
- **Reconciliation**: checking the internal ledger against external records such as bank statements.
- **Dunning**: the schedule of retries and reminders that follows a failed subscription payment.
- **At-least-once delivery**: a guarantee that a message is delivered one or more times, so receivers must tolerate duplicates.

## Summary

- Stripe's five rounds are all practical: a coding screen, a bug bash, payment-focused system design, an API integration round, and a values-based behavioral round. No algorithm puzzles.
- Correctness comes first because Stripe moves money: integer cents, idempotency, and explicit handling of every failure mode.
- Be ready to critique your own code; the reference retry system has real gaps (final status, backoff on exceptions, concurrent duplicates).
- System design centers on financial invariants: double-entry ledgers, idempotency, reconciliation, settlement, and at-least-once webhooks.
- Prepare by using the Stripe API, practicing debugging, and studying Stripe's API conventions.

## Review questions

1. Why does the payment retry system pass the idempotency key to the gateway as well as checking its own `processed` dictionary?
   <details><summary>Answer</summary>The local dictionary only protects against repeat calls to this processor after a result has been stored. If a charge succeeds at the gateway but the response is lost (a timeout), the retry loop will call the gateway again; the gateway-side key lets the gateway recognize the retry and return the original result instead of charging twice.</details>
2. Name two bugs in the reference `PaymentProcessor.process` and how you would fix them.
   <details><summary>Answer</summary>(a) If all attempts return retryable failures, the final status is `REQUIRES_RETRY`, not `FAILED`; set `FAILED` after the loop when no success occurred, and skip the sleep after the last attempt. (b) Concurrent calls with the same key can both pass the idempotency check because the key is stored only at the end; atomically record the key as in progress before charging. Exceptions also retry with no backoff, which you can fix by applying the same backoff in the `except` branch.</details>
3. In the rate limiter, why is `retry_after` computed from `window[-limit]`?
   <details><summary>Answer</summary>The newest `limit` timestamps are all within the last minute. A new request is allowed once fewer than `limit` remain in the window, which happens when the oldest of those newest `limit` entries, `window[-limit]`, becomes more than 60 seconds old, at time `window[-limit] + 60`.</details>
4. Why should money be represented as integer cents?
   <details><summary>Answer</summary>Binary floating point cannot represent most decimal fractions exactly, so sums and conversions of dollar floats accumulate rounding errors. Integer cents make arithmetic exact, and rounding happens only at explicit, controlled points such as currency conversion.</details>
5. What does double-entry bookkeeping buy you in a payment system?
   <details><summary>Answer</summary>Every transaction's entries sum to zero, so the ledger as a whole must balance. Any bug that creates or loses money shows up as an imbalance, which makes errors detectable, and the ledger becomes the source of truth that reconciliation compares against bank records.</details>
6. A webhook system promises at-least-once delivery. What does that require of the merchant, and what does Stripe add for safety?
   <details><summary>Answer</summary>Merchants must handle duplicate events idempotently, since retries can redeliver an event that was already processed. Stripe signs each payload with HMAC so merchants can verify authenticity, disables endpoints after repeated failures with a notification, and lets merchants replay missed events.</details>

## Further reading

- [Stripe Software Engineer Interview Guide - Exponent](https://www.tryexponent.com/guides/stripe-swe-interview)
- [Stripe Software Engineer Interview Guide - InterviewQuery](https://www.interviewquery.com/interview-guides/stripe-software-engineer)
- [My Stripe Interview Experience (2025-2026) - Medium](https://medium.com/@diyaag2020/my-stripe-interview-experience-2025-2026-a-journey-to-the-final-round-19990fa6876a)
- [Stripe's Interview Process - interviewing.io](https://interviewing.io/stripe-interview-questions)
- [Stripe System Design Interview - System Design Handbook](https://www.systemdesignhandbook.com/guides/stripe-system-design-interview/)
- [Stripe API Documentation](https://stripe.com/docs/api)
- [Stripe Engineering Blog](https://stripe.com/blog/engineering)
- [Glassdoor - Stripe SWE Interview Questions](https://www.glassdoor.com/Interview/Stripe-Software-Engineer-Interview-Questions-EI_IE671932.0,6_KO7,24.htm)
- Community reports on r/cscareerquestions and Blind
