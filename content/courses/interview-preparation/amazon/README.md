# Interviewing at Amazon

Amazon's software engineering interview is unusual in one respect above all others: every round, without exception, includes behavioral questions on its Leadership Principles. You can solve every coding problem and still be rejected because your stories were thin. At the same time, the technical bar is real. The coding rounds are classic data structures and algorithms at medium-to-hard difficulty, and the design round assumes planetary scale: whatever you build, the follow-up is "how does this work at a million transactions per second?" This chapter prepares you for both halves, with particular attention to the AI and ML infrastructure roles that are a growing share of Amazon's hiring.

The chapter follows the loop. It starts with the process and compensation, then the themes that run through every round, and then spends the most time on the Leadership Principles and the STAR method, because that is where most of your preparation time should go. After that come the coding rounds (with a worked, Amazon-flavored order-routing problem), the services-oriented system design round, the Bar Raiser, and the AI and ML topics specific to AWS. It ends with a preparation plan.

> **Learning objectives**
> - Describe Amazon's five-to-six-round loop and the role of the Bar Raiser.
> - Prepare 10 to 15 STAR stories that map onto the 16 Leadership Principles, told in the first person with metrics.
> - Recognize the frequently reported coding problem families and solve a practical priority-queue routing problem, including its concurrency pitfall.
> - Design services-oriented systems with sagas, compensation, idempotency, and event-driven messaging, using AWS building blocks.
> - Discuss the AWS AI/ML stack and common ML design questions.

## Before you begin

The coding rounds assume fluency with standard data structures and algorithms, and the design round builds on [Distributed systems](../shared-concepts/distributed-systems.md) in this book. For AI and ML roles, read [ML infrastructure](../shared-concepts/ml-infrastructure.md) and [Full-stack AI engineering](../shared-concepts/full-stack-ai.md) too. The saga and compensation patterns used in the design section are developed in depth in [Durable Orchestration and Workers](../../ai-platform-engineering/05-durable-orchestration-and-workers/README.md) in *AI Platform Engineering*.

## The interview process

The loop takes two to four weeks and has five or six rounds:

| Round | Format | Duration | Focus |
|-------|--------|----------|-------|
| Recruiter Screen | Phone | 30 min | Background, role fit, LP preview |
| Online Assessment | HackerRank/CodeSignal | 90-120 min | 2 coding problems + work simulation |
| Phone Screen | Coding + LP | 60 min | 1-2 coding problems + behavioral |
| Onsite 1 | Coding + LP | 60 min | DS&A + Leadership Principles |
| Onsite 2 | Coding + LP | 60 min | DS&A + Leadership Principles |
| Onsite 3 | System Design + LP | 60 min | Large-scale design + LP |
| Onsite 4 | Bar Raiser + LP | 60 min | Cross-team assessment + LP deep dive |

Notice the last column: every round includes Leadership Principles questions. That is non-negotiable at Amazon.

## Compensation

For a senior SDE (L6) in the US, base salary is $175-190K and is capped. Equity comes as RSUs on a back-loaded schedule, vesting 5%, 15%, 40%, and 40% over four years. To offset the light early vesting, there is a front-loaded sign-on bonus of $50-150K. Total compensation is about $350-450K in the first year and rises to about $400-600K in years three and four as the equity ramps.

## What Amazon is looking for

Five themes run through the loop. The first dominates: **Leadership Principles are evaluated in every round**, and you cannot pass without strong stories. The second is that **coding is bread and butter**: classic data structures and algorithms, at medium-to-hard LeetCode level. The third is that **scale is assumed**; Amazon operates at planetary scale, so every answer should consider what happens at a million transactions per second. The fourth is **Bias for Action**: Amazon values shipping over perfection, so show that you can move fast while keeping high standards. And the fifth is that **AI and ML are growing**. AWS AI services (SageMaker, Bedrock, Titan), Alexa, recommendations, and robotics make ML infrastructure a major hiring area.

## The Leadership Principles

This is the most important part of your preparation. Amazon has 16 Leadership Principles, and you need two or three strong stories for each, told in STAR format. Because stories overlap, 10 to 15 well-chosen stories can cover all of them. The principles and the behavior each describes are:

| # | Principle | Key Behavior |
|---|-----------|-------------|
| 1 | Customer Obsession | Start with the customer, work backwards |
| 2 | Ownership | Think long-term, never say "that's not my job" |
| 3 | Invent and Simplify | Expect innovation, find ways to simplify |
| 4 | Are Right, A Lot | Good judgment, seek diverse perspectives |
| 5 | Learn and Be Curious | Never stop learning |
| 6 | Hire and Develop the Best | Raise the bar, develop others |
| 7 | Insist on the Highest Standards | Relentlessly high standards |
| 8 | Think Big | Create bold vision, think differently |
| 9 | Bias for Action | Speed matters. Calculated risks are okay. |
| 10 | Frugality | Do more with less |
| 11 | Earn Trust | Listen, speak candidly, self-critical |
| 12 | Dive Deep | Stay connected to details, audit frequently |
| 13 | Have Backbone; Disagree and Commit | Challenge decisions respectfully, then commit |
| 14 | Deliver Results | Focus on inputs, deliver with quality |
| 15 | Strive to be Earth's Best Employer | Safe, diverse, empathetic workplace |
| 16 | Success and Scale Bring Broad Responsibility | Better every day, for customers and community |

### Telling a STAR story

STAR gives each story a shape the interviewer can score:

```text
S - Situation: Set the scene (1-2 sentences)
T - Task: What was YOUR specific responsibility?
A - Action: What did YOU do? (Most time here, be specific)
R - Result: Quantifiable outcome + what you learned
```

Four Amazon-specific habits make the difference. Say "I," not "we," because the interviewer wants your contribution. Include specific metrics: "$2M saved," "latency reduced 40%," "99.95% to 99.99%." Prepare follow-up depth for questions like "What would you do differently?" and "What was the biggest risk?" And map each story to two or three principles, since interviewers assign credit by principle and a single story can earn it for several.

### Questions to expect

The most frequently reported questions cluster around five principles. For **Customer Obsession**: "Tell me about a time you went above and beyond for a customer," and "Describe a time you had to balance customer needs with technical constraints." For **Ownership**: "Tell me about a time you took ownership of something outside your role," and "Describe a project where you saw it through end to end, including the parts you didn't enjoy." For **Dive Deep**: "Tell me about a time you had to dig into data to find the root cause of a problem," and "Describe a time metrics told you one thing but reality was different." For **Have Backbone; Disagree and Commit**: "Tell me about a time you disagreed with a decision and what you did," and "Describe a time you had to commit to a plan you didn't fully agree with." And for **Bias for Action**: "Tell me about a time you made a decision with incomplete information," and "Describe a time you took a calculated risk."

## The coding rounds

There are two or three coding rounds. Each 60-minute round typically has one or two problems in about 45 minutes, with the remaining 15 reserved for Leadership Principles questions. The problems are standard data structures and algorithms (arrays, trees, graphs, dynamic programming, hash maps), solved on a whiteboard or in a shared editor, and clean code is expected: name variables well and handle edge cases.

### The problem families

The reported problems fall into five families, each with a characteristic technique:

| Family | Problem | Technique |
|--------|---------|-----------|
| Arrays and strings | Two Sum / Three Sum | Hash map, two pointers |
| | Longest Substring Without Repeating Characters | Sliding window |
| | Group Anagrams | Sorted string as hash key |
| | Product of Array Except Self | Prefix and suffix products |
| Trees | Lowest Common Ancestor | Recursion or parent pointers |
| | Serialize/Deserialize BST | Preorder traversal |
| | Validate BST | Inorder traversal or min/max bounds |
| | Level Order Traversal | BFS with a queue |
| Graphs | Number of Islands | DFS/BFS flood fill |
| | Course Schedule | Topological sort |
| | Shortest Path | BFS (unweighted), Dijkstra (weighted) |
| | Clone Graph | BFS/DFS with a hash map |
| Dynamic programming | Longest Increasing Subsequence | $O(n \log n)$ with patience sorting |
| | Coin Change | Bottom-up DP |
| | Edit Distance | 2D DP |
| | Word Break | DP with a hash set |
| Amazon favorites | LRU Cache | Doubly linked list plus hash map |
| | Design a Parking Lot | Object-oriented design |
| | Merge Intervals | Sort by start, merge overlaps |
| | Min Stack | Stack with min tracking |

### A worked problem: routing orders to warehouses

Amazon also asks practical problems in its own domain. A representative one: route each incoming order to a warehouse that has the inventory to fulfill it, and let each warehouse process its orders in priority order, with Prime ahead of expedited ahead of standard.

The design follows from the requirements. Each warehouse gets its own priority queue, a binary heap keyed by `(priority, timestamp)`, so higher-priority orders come first and ties go to the earlier order. Submitting an order scans the warehouses for one whose stock covers every item, counting duplicates, and pushes the order onto that warehouse's heap. Processing pops the highest-priority order and decrements the inventory.

```python
import heapq
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

class Priority(Enum):
    PRIME = 0
    EXPEDITED = 1
    STANDARD = 2

@dataclass(order=True)
class Order:
    priority: int
    timestamp: float
    order_id: str = field(compare=False)
    items: list = field(compare=False)
    warehouse: str = field(compare=False, default="")

class OrderRouter:
    """Route orders to nearest warehouse with available inventory."""

    def __init__(self):
        self.queues: dict[str, list] = defaultdict(list)  # warehouse -> priority queue
        self.inventory: dict[str, dict[str, int]] = {}     # warehouse -> {item: count}

    def add_warehouse(self, warehouse_id: str, inventory: dict[str, int]):
        self.inventory[warehouse_id] = inventory
        self.queues[warehouse_id] = []

    def submit_order(self, order: Order) -> Optional[str]:
        """Route order to a warehouse that can fulfill it. Returns warehouse ID."""
        for warehouse_id, stock in self.inventory.items():
            if self._can_fulfill(stock, order.items):
                order.warehouse = warehouse_id
                heapq.heappush(self.queues[warehouse_id],
                               (order.priority, order.timestamp, order))
                return warehouse_id
        return None  # No warehouse can fulfill

    def process_next(self, warehouse_id: str) -> Optional[Order]:
        """Process highest-priority order from a warehouse."""
        queue = self.queues[warehouse_id]
        if not queue:
            return None
        _, _, order = heapq.heappop(queue)
        # Decrement inventory
        stock = self.inventory[warehouse_id]
        for item in order.items:
            stock[item] -= 1
        return order

    def _can_fulfill(self, stock: dict[str, int], items: list[str]) -> bool:
        needed = defaultdict(int)
        for item in items:
            needed[item] += 1
        return all(stock.get(item, 0) >= count for item, count in needed.items())
```

`_can_fulfill` counts each item's required quantity before comparing with stock, so an order for two of the same item is checked correctly. Submission is $O(W \cdot k)$ for $W$ warehouses and $k$ items in the order, plus $O(\log n)$ for the heap push; processing is $O(\log n + k)$. Despite the docstring, the scan picks the first warehouse that can fulfill the order rather than the nearest, which is a natural follow-up: add a distance to each warehouse and scan in distance order.

> **Warning:** Inventory is checked at submission but decremented only at processing. Two orders submitted back to back can both pass the check against the same last unit, and the second will drive stock negative. A correct design reserves inventory at submission (decrement or hold it immediately) and releases the reservation if the order is cancelled. Pointing this out unprompted is exactly the kind of ownership the interviewers look for.

## The system design round

Amazon's system design round is grounded in its services-oriented architecture. Four principles shape a good answer. Build **services, not monoliths**: everything is a service with a well-defined API. **Work backwards from the customer**, starting with the customer experience. Assume **two-pizza teams**: each service is owned by a small, autonomous team. And address **operational excellence**: how do you monitor, alarm, and run on-call for this?

### An e-commerce order system

The most Amazon-flavored prompt is the order system itself. An order service coordinates inventory, payment, and fulfillment services, each owning its own store:

```text
[Client] --> [API Gateway] --> [Order Service]
                                    |
                    +---------------+---------------+
                    |               |               |
             [Inventory Svc]  [Payment Svc]  [Fulfillment Svc]
                    |               |               |
             [DynamoDB]       [Payment Gateway]  [Warehouse Svc]
                                                      |
                                                 [Shipping Svc]
```

Because no single database spans these services, an order is a **saga**: a distributed transaction carried out as a sequence of local steps, from order to payment to inventory to fulfillment. Each step needs a **compensation**: if payment fails after inventory has been reserved, release the inventory. Every API call must be **idempotent**, so retries are safe. And services communicate asynchronously through **events**, using SQS and SNS.

### A recommendation system

For "Design Amazon's recommendation system," the building blocks are user-item collaborative filtering, which at Amazon scale means matrix factorization or alternating least squares; "customers who bought X also bought Y," built from co-purchase frequency and item-item similarity; real-time personalization that combines batch-computed embeddings with real-time session features; and A/B testing through Weblab, Amazon's experimentation platform.

### A distributed cache

For "Design a distributed cache" (think ElastiCache), cover consistent hashing for key distribution; write-through versus write-behind strategies; cache invalidation by TTL, by events, or with versioned keys; and hot-key mitigation through local caching, key splitting, and request coalescing.

### A serverless platform

For "Design AWS Lambda," cover cold-start optimization with pre-warmed containers and snapshot-and-restore; concurrency control with per-function and account-level limits; event-source integration with SQS, Kinesis, and API Gateway triggers; and resource allocation, where CPU scales with the memory configuration.

### AWS building blocks

Using Amazon's own services naturally in your designs shows that you understand the ecosystem. These are the ones to know:

| Service | Purpose | When to Reference |
|---------|---------|------------------|
| DynamoDB | NoSQL key-value store | High-throughput, low-latency reads/writes |
| SQS | Message queue | Async decoupling between services |
| SNS | Pub/sub notification | Fan-out to multiple consumers |
| Kinesis | Real-time streaming | Event processing, analytics |
| S3 | Object storage | Large files, data lake |
| SageMaker | ML platform | Training, hosting, MLOps |
| Bedrock | Managed LLM service | AI application development |
| ECS/EKS | Container orchestration | Microservice deployment |
| CloudWatch | Monitoring | Metrics, logs, alarms |

The difference between SQS (a queue, where each message goes to one consumer) and SNS (pub-sub, where each message fans out to every subscriber) is worth being precise about; [Messaging and Distributed Queueing](../../infrastructure/02-messaging-and-queueing/README.md) in *Infrastructure* explains it, along with long-polling.

## The Bar Raiser

The **Bar Raiser** is an interviewer from a different team who holds veto power over the hire. Their job is to make sure every hire raises the bar. Expect the deepest Leadership Principles probing of the whole loop, possibly with a coding or design component as well. The question they are answering is whether you are better than half of the current Amazonians at your level, and their follow-up questions go three or four levels deep on each story.

Prepare for that by saving your best stories for this round. Expect "Tell me more" and "Why?" repeatedly. Have backup examples, because if they exhaust one story they will ask for another. And be ready to discuss failures and what you learned from them, which speaks to Earn Trust and Learn and Be Curious.

## AI and ML topics

For AI-focused roles, know the AWS AI/ML stack. **SageMaker** is the end-to-end ML platform: notebooks, training, hosting, and pipelines. **Bedrock** provides managed foundation models, including Claude, Titan, and Llama. **Titan** is Amazon's own family of foundation models. **Inferentia** and **Trainium** are Amazon's custom AI chips, an alternative to NVIDIA GPUs. And **Comprehend**, **Rekognition**, and **Textract** are pre-built AI services.

Common ML design questions include designing a fraud detection system, a product search ranking system, an ML feature store, and a model monitoring and retraining pipeline. [ML infrastructure](../shared-concepts/ml-infrastructure.md) covers the patterns behind each.

## Preparing well

Write your Leadership Principles stories first and spend about 60% of your preparation time on them, aiming for 10 to 15 strong stories that each map to several principles. Work 100 to 150 LeetCode problems, focusing on Amazon-tagged ones. Use Amazon's own services in your designs, referencing DynamoDB, SQS, and Kinesis naturally. Think like an owner: in every answer, show end-to-end thinking about how you would run the thing in production. Practice STAR out loud, recording yourself and keeping each answer to three or four minutes. Read about Amazon's tenets, since each team has them and understanding the tenet-driven culture shows depth. And always cover operational excellence (monitoring, alerting, runbooks, on-call) in any system design.

## Key terms

- **Leadership Principles (LPs)**: Amazon's 16 cultural principles, assessed in every interview round.
- **STAR**: Situation, Task, Action, Result; the structure for a behavioral story.
- **Bar Raiser**: an interviewer from outside the hiring team, with veto power, who ensures each hire raises the bar.
- **Saga**: a distributed transaction carried out as a sequence of local steps, each with a compensating action.
- **Compensation**: the action that undoes a completed saga step when a later step fails.
- **Weblab**: Amazon's internal A/B testing platform.

## Summary

- Every round tests the Leadership Principles; 10 to 15 metric-rich, first-person STAR stories mapped to several principles each are the core of your preparation.
- Coding is classic medium-to-hard data structures and algorithms in about 45 minutes per round, plus practical problems like order routing.
- In the order router, reserve inventory at submission, not at processing, or concurrent orders can oversell.
- System design is services-oriented: sagas with compensation, idempotent APIs, SQS/SNS events, and operational excellence, built from AWS services.
- The Bar Raiser probes your best stories three or four levels deep and can veto the hire.

## Review questions

1. Why does a story told with "we" score poorly at Amazon even if the project was a success?
   <details><summary>Answer</summary>Interviewers assign Leadership Principles credit for your individual actions. "We" hides what you specifically did, so there is nothing to credit; the Action part of STAR should be in the first person and specific.</details>
2. In the `OrderRouter`, two orders each needing the last unit of an item are submitted before either is processed. What happens, and how would you fix it?
   <details><summary>Answer</summary>Both pass <code>_can_fulfill</code> because stock is decremented only in <code>process_next</code>, so the second processed order drives stock to −1. Reserve inventory at submission time (decrement immediately or record a hold) and release it if the order is cancelled.</details>
3. In an order saga, payment fails after inventory was reserved. What must happen, and what property must every step have to make retries safe?
   <details><summary>Answer</summary>A compensating action releases the reserved inventory. Every service call must be idempotent, so retrying a step after a timeout does not reserve or charge twice.</details>
4. When would you choose SNS over SQS in a design?
   <details><summary>Answer</summary>When several independent services must each react to the same event (fan-out). SQS delivers each message to one consumer, which suits distributing work across a pool; the common pattern is SNS fanning out to one SQS queue per consuming service.</details>
5. What does the Bar Raiser assess that the other interviewers might not?
   <details><summary>Answer</summary>Whether you are better than half of current Amazonians at the level, judged by someone outside the hiring team with no stake in filling the role. They probe stories several levels deep and can veto the hire.</details>
6. Why does Amazon front-load the sign-on bonus?
   <details><summary>Answer</summary>RSUs vest 5/15/40/40 over four years, so equity income is small in the first two years. The sign-on bonus offsets that gap.</details>

## Further reading

- [Amazon Leadership Principles](https://www.amazon.jobs/content/en/our-workplace/leadership-principles)
- [How 12 Top Tech Companies Interview Software Engineers](https://www.tryexponent.com/blog/how-top-tech-companies-interview-software-engineers) (Exponent)
- [Glassdoor: Amazon SDE interview questions](https://www.glassdoor.com/Interview/Amazon-Software-Development-Engineer-Interview-Questions-EI_IE6036.0,6_KO7,36.htm)
- [levels.fyi: Amazon compensation data](https://www.levels.fyi/companies/amazon/salaries/software-engineer)
- [AWS documentation](https://docs.aws.amazon.com/)
- Community reports on r/cscareerquestions, r/ExperiencedDevs, and Blind.
