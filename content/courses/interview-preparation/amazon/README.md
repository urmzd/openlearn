# Amazon Software Engineer Interview Guide

Comprehensive preparation for Amazon SDE roles, with focus on AI/ML infrastructure and scalable systems.

## Interview Process Overview

Timeline: **2-4 weeks**, **5-6 rounds** (loop)

| Round | Format | Duration | Focus |
|-------|--------|----------|-------|
| Recruiter Screen | Phone | 30 min | Background, role fit, LP preview |
| Online Assessment | HackerRank/CodeSignal | 90-120 min | 2 coding problems + work simulation |
| Phone Screen | Coding + LP | 60 min | 1-2 coding problems + behavioral |
| Onsite 1 | Coding + LP | 60 min | DS&A + Leadership Principles |
| Onsite 2 | Coding + LP | 60 min | DS&A + Leadership Principles |
| Onsite 3 | System Design + LP | 60 min | Large-scale design + LP |
| Onsite 4 | Bar Raiser + LP | 60 min | Cross-team assessment + LP deep dive |

**Every round includes Leadership Principles behavioral questions.** This is non-negotiable at Amazon.

## Compensation (L6 Senior SDE, US)

- **Base**: $175-190K (capped)
- **Equity**: RSUs vest back-loaded (5/15/40/40 over 4 years)
- **Sign-on bonus**: $50-150K (front-loaded to offset equity vesting)
- **Total Comp Year 1**: ~$350-450K
- **Total Comp Year 3-4**: ~$400-600K (as equity ramps)

## Key Themes

1. **Leadership Principles dominate** -- LPs are evaluated in EVERY round. You cannot pass without strong LP stories.
2. **Coding is bread and butter** -- Classic DS&A problems. Medium-hard Leetcode level.
3. **Scale is assumed** -- Amazon operates at planetary scale. Everything is "how does this work at 1M TPS?"
4. **Bias for Action** -- Amazon values shipping over perfection. Show you can move fast with high standards.
5. **AI/ML is growing** -- AWS AI services (SageMaker, Bedrock, Titan), Alexa, recommendations, robotics. ML infrastructure is a major hiring area.

## Leadership Principles (THE Most Important Section)

Amazon has 16 Leadership Principles. You need **2-3 strong stories per principle**, told in **STAR format**.

### The 16 Principles

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

### STAR Method (Amazon Edition)

```
S - Situation: Set the scene (1-2 sentences)
T - Task: What was YOUR specific responsibility?
A - Action: What did YOU do? (Most time here, be specific)
R - Result: Quantifiable outcome + what you learned
```

**Amazon-specific tips**:
- Use "I" not "we" -- they want YOUR contribution
- Include specific metrics: "$2M saved", "latency reduced 40%", "99.95% to 99.99%"
- Prepare follow-up depth: "What would you do differently?" "What was the biggest risk?"
- Map each story to 2-3 LPs -- stories overlap, and interviewers will assign LP credit

### Common LP Questions

**Customer Obsession**:
- "Tell me about a time you went above and beyond for a customer."
- "Describe a time you had to balance customer needs with technical constraints."

**Ownership**:
- "Tell me about a time you took ownership of something outside your role."
- "Describe a project where you saw it through end-to-end, including the parts you didn't enjoy."

**Dive Deep**:
- "Tell me about a time you had to dig into data to find the root cause of a problem."
- "Describe a time metrics told you one thing but reality was different."

**Have Backbone; Disagree and Commit**:
- "Tell me about a time you disagreed with a decision and what you did."
- "Describe a time you had to commit to a plan you didn't fully agree with."

**Bias for Action**:
- "Tell me about a time you made a decision with incomplete information."
- "Describe a time you took a calculated risk."

## Coding Rounds (2-3 rounds)

### What to Expect

- 1-2 problems per 45-minute coding section (15 min reserved for LP questions)
- Standard DS&A: arrays, trees, graphs, DP, hash maps
- Clean code expected: name variables well, handle edge cases
- Typically on a whiteboard or shared editor

### Frequently Reported Problems

#### Arrays & Strings
- **Two Sum / Three Sum** -- Hash map, two-pointer
- **Longest Substring Without Repeating Characters** -- Sliding window
- **Group Anagrams** -- Sorted string as hash key
- **Product of Array Except Self** -- Prefix/suffix products

#### Trees
- **LCA (Lowest Common Ancestor)** -- Recursive or parent pointers
- **Serialize/Deserialize BST** -- Preorder traversal
- **Validate BST** -- Inorder traversal or min/max bounds
- **Level Order Traversal** -- BFS with queue

#### Graphs
- **Number of Islands** -- DFS/BFS flood fill
- **Course Schedule** -- Topological sort
- **Shortest Path** -- BFS (unweighted), Dijkstra (weighted)
- **Clone Graph** -- BFS/DFS with hash map

#### Dynamic Programming
- **Longest Increasing Subsequence** -- O(n log n) with patience sort
- **Coin Change** -- Bottom-up DP
- **Edit Distance** -- 2D DP
- **Word Break** -- DP with hash set

#### Amazon Favorites
- **LRU Cache** -- DLL + HashMap
- **Design a Parking Lot** -- OOP design
- **Merge Intervals** -- Sort by start, merge overlapping
- **Min Stack** -- Stack with min tracking

### Implementation Example: Order Processing System

A practical Amazon-flavored problem:

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

## System Design Round

### Approach (Amazon-Specific)

Amazon system design is grounded in their services-oriented architecture. Key principles:
- **Services, not monoliths** -- Everything is a service with a well-defined API
- **Work backwards from the customer** -- Start with the customer experience
- **Two-pizza teams** -- Services owned by small, autonomous teams
- **Operational excellence** -- How do you monitor, alarm, and on-call for this?

### Common Topics

#### Design an E-Commerce Order System

```
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

- **Saga pattern**: Distributed transaction across services (order -> payment -> inventory -> fulfillment)
- **Compensation**: If payment fails after inventory reserved, release inventory
- **Idempotency**: Every API call must be idempotent (retry-safe)
- **Event-driven**: SQS/SNS for async communication between services

#### Design Amazon's Recommendation System

- **User-item collaborative filtering**: At Amazon scale, use matrix factorization / ALS
- **"Customers who bought X also bought Y"**: Co-purchase frequency, item-item similarity
- **Real-time personalization**: Combine batch-computed embeddings with real-time session features
- **A/B testing**: Weblab (Amazon's experimentation platform)

#### Design a Distributed Cache (ElastiCache)

- **Consistent hashing** for key distribution
- **Write-through vs write-behind** strategies
- **Cache invalidation**: TTL, event-based, versioned keys
- **Hot key mitigation**: Local caching, key splitting, request coalescing

#### Design AWS Lambda (Serverless)

- **Cold start optimization**: Pre-warmed containers, snapshot/restore
- **Concurrency control**: Per-function and account-level limits
- **Event source integration**: SQS, Kinesis, API Gateway triggers
- **Resource allocation**: CPU scales with memory configuration

### Amazon Infrastructure to Know

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

## Bar Raiser Round

The **Bar Raiser** is an interviewer from a different team who has veto power. They ensure every hire raises the bar.

### What to Expect

- Heavy LP focus (often the deepest LP probing of the loop)
- May include a coding or design component
- They assess: "Is this person better than 50% of current Amazonians at this level?"
- Follow-up questions go 3-4 levels deep on your stories

### How to Prepare

- Your best LP stories should be reserved for this round
- Expect "Tell me more" and "Why?" repeatedly
- Have backup examples -- if they exhaust one story, they'll ask for another
- Be prepared to discuss failures and learnings (Earn Trust, Learn and Be Curious)

## AI/ML-Specific Topics

For AI-focused roles at Amazon:

### AWS AI/ML Stack
- **SageMaker**: End-to-end ML platform (notebooks, training, hosting, pipelines)
- **Bedrock**: Managed foundation models (Claude, Titan, Llama, etc.)
- **Titan**: Amazon's own foundation models
- **Inferentia/Trainium**: Custom AI chips (alternative to NVIDIA GPUs)
- **Comprehend/Rekognition/Textract**: Pre-built AI services

### Common ML Design Questions
- Design a fraud detection system
- Design a product search ranking system
- Design an ML feature store
- Design a model monitoring and retraining pipeline

## Preparation Tips

1. **LP stories first** -- Spend 60% of your prep time on LP stories. Have 10-15 strong stories that map to multiple principles.
2. **Leetcode**: 100-150 problems. Focus on Amazon-tagged problems on Leetcode.
3. **Use Amazon's own services in designs** -- Reference DynamoDB, SQS, Kinesis naturally. Shows you understand the ecosystem.
4. **Think like an owner** -- In every answer, show end-to-end thinking. "How would I own this in production?"
5. **Practice STAR out loud** -- Record yourself. Keep answers to 3-4 minutes. Time yourself.
6. **Read Amazon's tenets** -- Each team has tenets. Understanding the tenet-driven culture shows depth.
7. **Operational excellence** -- Always discuss monitoring, alerting, runbooks, and on-call for any system design.

## Sources

- [Amazon Leadership Principles](https://www.amazon.jobs/content/en/our-workplace/leadership-principles)
- [How 12 Top Tech Companies Interview Software Engineers - Exponent](https://www.tryexponent.com/blog/how-top-tech-companies-interview-software-engineers)
- [Glassdoor - Amazon SDE Interview Questions](https://www.glassdoor.com/Interview/Amazon-Software-Development-Engineer-Interview-Questions-EI_IE6036.0,6_KO7,36.htm)
- [levels.fyi - Amazon Compensation Data](https://www.levels.fyi/companies/amazon/salaries/software-engineer)
- [AWS Documentation](https://docs.aws.amazon.com/)
- r/cscareerquestions, r/ExperiencedDevs, Blind (community reports)
