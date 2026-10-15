# Stripe Software Engineer Interview Guide

Comprehensive preparation for Stripe SWE roles. Stripe is known for one of the **most unique interview processes** in tech -- no leetcode, practical bugs, real API integration, and financial system design.

## Interview Process Overview

Timeline: **2-6 weeks** (faster with referral), **5 rounds**

| Round | Format | Duration | Focus |
|-------|--------|----------|-------|
| Recruiter Screen | Phone | 30 min | Background, role fit |
| Phone Screen | Coding (practical) | 60 min | Real-world coding problem |
| Onsite 1 | Bug Bash | 60 min | Debug a failing codebase |
| Onsite 2 | System Design | 60 min | Payment infrastructure design |
| Onsite 3 | Integration | 60 min | Build with Stripe API |
| Onsite 4 | Behavioral / Manager | 45 min | Values, collaboration |

**AI use in Stripe interviews is strictly prohibited.**

## Compensation (Senior L3, US)

- **Base**: $190-230K
- **Equity**: RSUs ~$200-400K/yr
- **Bonus**: 10-15%
- **Total Comp**: ~$400-650K (L3 Senior), ~$600K-900K+ (L4 Staff)

## Key Themes

1. **No leetcode** -- Stripe explicitly does NOT ask algorithm puzzles. Every problem is practical and resembles real Stripe engineering work.
2. **Correctness is sacred** -- Stripe processes money. A bug can mean real financial loss. Correctness > speed > elegance.
3. **Production quality** -- Clean code, error handling, edge cases, testability. This is non-negotiable.
4. **Financial invariants** -- Double-entry accounting, idempotency, exactly-once processing. Know these cold.
5. **Communication** -- Stripe values engineers who think out loud, explain trade-offs clearly, and ask good questions.
6. **Full-stack thinking** -- Even backend roles may touch APIs, developer experience, and documentation.

## Phone Screen: Practical Coding

### What to Expect

- A real-world problem, not an algorithm puzzle
- May involve building a small service, data processing pipeline, or API handler
- Code must be clean, well-structured, and handle edge cases
- Interviewer evaluates your approach, communication, and code quality

### Reported Problem Types

#### Build a Payment Retry System

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

#### Build a Rate Limiter for API Keys

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

## Onsite 1: Bug Bash

### What It Is

You're given a **real codebase with failing tests**. Your job is to find and fix the bugs. This is Stripe's most distinctive round.

### Format

- Pre-existing codebase (usually Python, Ruby, or Go)
- Several failing test cases
- You read the code, understand the expected behavior, find the bugs, and fix them
- Time pressure: 60 minutes, multiple bugs

### What They're Testing

- **Code reading ability** -- Can you quickly understand unfamiliar code?
- **Debugging methodology** -- Do you read error messages? Use a systematic approach?
- **Attention to detail** -- Off-by-one errors, null checks, type mismatches
- **Code quality** -- Your fixes should be clean, not hacky

### Bug Categories to Watch For

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

### Preparation Strategy

- Practice reading open-source code and finding bugs
- Do code review exercises (review PRs, find issues)
- Practice debugging without a debugger (read code, trace mentally)
- Work through failing test cases systematically: read the test, understand expected behavior, trace the code path

## Onsite 2: System Design

### Stripe-Specific Design Focus

Stripe system design is **lower-level and more practical** than typical system design. They focus on:

- **API design**: RESTful endpoints, request/response schemas, error codes
- **Data flow**: How does money move through the system?
- **Financial invariants**: Double-entry accounting, idempotency, reconciliation
- **Reliability**: Exactly-once processing in a distributed system

### Common Topics

#### Design a Payment Processing System

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

**Key financial concepts**:

- **Double-entry bookkeeping**: Every transaction has a debit and credit. The sum must always be zero.
  ```
  Payment of $100:
    Debit:  Merchant Receivable  +$100
    Credit: Platform Revenue     -$97.10
    Credit: Processing Fee       -$2.90
  ```

- **Idempotency**: Every API call includes an idempotency key. Retrying with the same key returns the same result.

- **Reconciliation**: Periodically verify internal ledger matches external bank records.

- **Settlement**: Batch funds transfer to merchants (daily or weekly).

#### Design a Subscription Billing System

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

- **Billing clock**: Cron-like system that fires at subscription intervals
- **Proration**: Handle mid-cycle plan changes fairly
- **Dunning**: Retry logic for failed payments (1 day, 3 days, 7 days)
- **Tax calculation**: Region-dependent tax rules

#### Design a Webhook Delivery System

- **At-least-once delivery**: Retry with exponential backoff
- **Ordering**: Deliver events in order per resource (optional, complex)
- **Signature verification**: HMAC signature for webhook authenticity
- **Failure handling**: Disable endpoint after N consecutive failures, notify merchant
- **Replay**: Allow merchants to replay missed events

### API Design Principles (Stripe-Specific)

Stripe's API is considered one of the best-designed APIs in the industry:

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

## Onsite 3: Integration Round

### What It Is

You're given access to the **Stripe API** (or a mock) and asked to implement a feature using it. This tests your ability to work with an unfamiliar API effectively.

### Format

- You'll navigate an unfamiliar codebase
- Read API documentation to understand available endpoints
- Implement a feature (e.g., "add subscription upgrade/downgrade support")
- Write tests for your implementation

### What They're Testing

- **API fluency**: Can you read docs and use an API effectively?
- **Full-stack thinking**: Frontend to backend to external service
- **Testing instinct**: Do you write tests?
- **Code integration**: Can you extend an existing codebase cleanly?

### Preparation

- **Use the Stripe API before your interview**: Sign up, get test keys, make real API calls
- Practice reading API documentation and implementing against it
- Build a small project that uses Stripe (checkout flow, subscription management)

## Behavioral Round

### Stripe Values

| Value | What They Assess |
|-------|-----------------|
| Users first | Empathy for developers using Stripe |
| Move with urgency | Ship quickly without sacrificing quality |
| Think rigorously | First-principles thinking, data-driven |
| Trust and amplify | Collaborate, uplift teammates |
| Global optimization | What's best for Stripe, not just your team |

### Common Questions

- "Tell me about a time you found and fixed a subtle bug in production."
- "Describe a system you built that had to be correct, not just fast."
- "How do you approach code review?"
- "Tell me about a time you simplified a complex system."
- "Describe a time you had to make a decision between two imperfect options."

## Preparation Tips

1. **No leetcode** -- Seriously. Stripe does not ask algorithm puzzles. Focus on practical coding.
2. **Use the Stripe API** -- Build something real. Understand payments, customers, invoices, webhooks.
3. **Practice debugging** -- Find open-source repos, introduce bugs, fix them. Practice reading unfamiliar code.
4. **Learn financial systems** -- Double-entry bookkeeping, idempotency, ACID transactions, reconciliation.
5. **API design** -- Study Stripe's API as a model. Read their API reference: stripe.com/docs/api
6. **Read the Stripe engineering blog** -- stripe.com/blog/engineering. Focus on infrastructure posts.
7. **Clean code above all** -- Variable naming, error handling, edge cases, comments where non-obvious.

## Sources

- [Stripe Software Engineer Interview Guide - Exponent](https://www.tryexponent.com/guides/stripe-swe-interview)
- [Stripe Software Engineer Interview Guide - InterviewQuery](https://www.interviewquery.com/interview-guides/stripe-software-engineer)
- [My Stripe Interview Experience (2025-2026) - Medium](https://medium.com/@diyaag2020/my-stripe-interview-experience-2025-2026-a-journey-to-the-final-round-19990fa6876a)
- [Stripe's Interview Process - interviewing.io](https://interviewing.io/stripe-interview-questions)
- [Stripe System Design Interview - System Design Handbook](https://www.systemdesignhandbook.com/guides/stripe-system-design-interview/)
- [Stripe API Documentation](https://stripe.com/docs/api)
- [Stripe Engineering Blog](https://stripe.com/blog/engineering)
- [Glassdoor - Stripe SWE Interview Questions](https://www.glassdoor.com/Interview/Stripe-Software-Engineer-Interview-Questions-EI_IE671932.0,6_KO7,24.htm)
- r/cscareerquestions, Blind (community reports)
