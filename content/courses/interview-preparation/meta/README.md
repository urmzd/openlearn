# Meta Software Engineer Interview Guide

Comprehensive preparation for Meta SWE roles. Meta is one of the largest AI-focused employers globally, with a structured, well-documented interview process operating at planetary scale.

## Interview Process Overview

Timeline: **3-6 weeks**, **5-6 rounds**

| Round | Format | Duration | Focus |
|-------|--------|----------|-------|
| Online Assessment | CodeSignal (proctored) | 90 min | Progressive coding problem (4 stages) |
| Recruiter Screen | Phone | 30 min | Background, level calibration |
| Phone Screen | Coding (CoderPad) | 45 min | 2 LC medium problems |
| Onsite 1 | Coding | 45 min | Traditional DS&A |
| Onsite 2 | AI-Enabled Coding | 60 min | Coding with AI assistant (new 2025+) |
| Onsite 3 | System Design | 45 min | Large-scale distributed systems |
| Onsite 4 | Behavioral | 45 min | Meta core values alignment |

**New in 2025-2026**: Meta introduced an **AI-enabled coding round** where candidates have access to an AI assistant in CoderPad. This tests your ability to leverage AI tools effectively, not just raw coding ability.

## Compensation (E5 Senior, US)

- **Base**: $190-220K
- **Equity**: RSUs ~$250-500K/yr (vest quarterly after 1-year cliff)
- **Bonus**: 15-25% of base
- **Total Comp**: ~$450-700K (E5), ~$700K-1.2M+ (E6 Staff)
- Meta refreshes equity aggressively for top performers

## Key Themes

1. **Speed and volume** -- Meta interviews are fast-paced. 2 coding problems in 45 minutes means ~20 min per problem. No time for hesitation.
2. **Move fast** -- Meta's culture values velocity. In behavioral, show you bias toward action and iterate quickly.
3. **AI-first transformation** -- Meta is pivoting aggressively to AI (LLaMA, AI assistants, recommendation engines). AI knowledge is increasingly relevant.
4. **Scale is the default** -- Everything at Meta serves billions of users. "How does this work at 3B users?" is always the lens.
5. **Structured process** -- Meta's interview is well-documented and consistent. This makes it one of the most "preparable" interviews.

## Online Assessment (New Screening Step)

As of 2025, Meta uses CodeSignal with **full video and microphone proctoring**.

- **Format**: 1 complex problem divided into 4 progressive stages that unlock sequentially
- **Duration**: 90 minutes
- **Reported problems**:
  - In-memory database with key-value operations (SET, GET, DELETE, filtering, transactions)
  - Cloud-based file storage service (upload, download, permissions, versioning)
  - Task scheduling system with dependencies and priorities

This is very similar to Anthropic's OA format. Production quality matters.

## Coding Rounds (2 rounds)

### Traditional Coding Round (45 min)

2 problems, medium difficulty, 20 minutes each + 5 min intro.

**Meta draws heavily from these categories**:

| Category | Frequency | Classic Problems |
|----------|-----------|-----------------|
| Arrays/Strings | Very High | Two Sum, valid palindrome, move zeroes |
| Trees/BST | Very High | LCA, diameter, right side view, serialize |
| Graphs | High | Clone graph, course schedule, shortest path |
| Hash Maps | Very High | Group anagrams, two sum, subarray sum |
| BFS/DFS | Very High | Number of islands, word ladder, maze |
| Dynamic Programming | High | Longest palindromic substring, coin change |
| Binary Search | Medium | Search rotated array, find peak |
| Linked Lists | Medium | Merge sorted lists, reverse, cycle detection |
| Stacks/Queues | Medium | Valid parentheses, min stack |
| Intervals | Medium | Merge intervals, meeting rooms |

### Key Meta Coding Patterns

#### Two-Pass HashMap

```python
def two_sum(nums: list[int], target: int) -> list[int]:
    seen = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    return []
```

#### BFS with Level Tracking

```python
from collections import deque

def right_side_view(root) -> list[int]:
    """Binary tree right side view -- last node at each level."""
    if not root:
        return []
    result = []
    queue = deque([root])
    while queue:
        level_size = len(queue)
        for i in range(level_size):
            node = queue.popleft()
            if i == level_size - 1:
                result.append(node.val)
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
    return result
```

#### Graph Clone (DFS with Visited Map)

```python
def clone_graph(node):
    if not node:
        return None
    cloned = {}

    def dfs(n):
        if n in cloned:
            return cloned[n]
        copy = Node(n.val)
        cloned[n] = copy
        for neighbor in n.neighbors:
            copy.neighbors.append(dfs(neighbor))
        return copy

    return dfs(node)
```

#### Monotonic Stack

```python
def daily_temperatures(temperatures: list[int]) -> list[int]:
    """Days until warmer temperature."""
    result = [0] * len(temperatures)
    stack = []  # indices of decreasing temperatures
    for i, temp in enumerate(temperatures):
        while stack and temperatures[stack[-1]] < temp:
            prev = stack.pop()
            result[prev] = i - prev
        stack.append(i)
    return result
```

### AI-Enabled Coding Round (60 min, new)

This round gives you access to an AI coding assistant in the editor.

**What they're testing**:
- Can you effectively prompt the AI for help?
- Do you understand the code the AI generates?
- Can you debug AI-generated code?
- Do you know when AI output is wrong?
- Can you iterate and refine with AI assistance?

**Strategy**:
- Use the AI for boilerplate, syntax reminders, and edge case generation
- Always review and understand AI output before using it
- Explain your reasoning to the interviewer -- don't just copy-paste AI output
- The problems may be harder since you have AI help

## System Design Round

### Meta-Specific Focus Areas

Meta system design revolves around their products: News Feed, Messenger, Instagram, WhatsApp, and increasingly AI/ML systems.

### Approach Framework

```
1. Clarify requirements (3-5 min)
   - Functional: What does the system do?
   - Non-functional: Scale, latency, consistency, availability
   - Constraints: Budget, team size, timeline

2. API Design (5 min)
   - Define key endpoints
   - Request/response schemas

3. High-Level Design (10 min)
   - Draw component diagram
   - Data flow

4. Data Model (5 min)
   - Schema design
   - Storage choices (SQL vs NoSQL vs graph)

5. Deep Dive (15-20 min)
   - Interviewer picks area to probe
   - Scale, reliability, performance

6. Trade-offs & Wrap-up (5 min)
```

### Common System Design Questions

#### Design Facebook News Feed

```
[User Action] --> [Write Path]
                      |
               [Fan-out Service]
                      |
         +------------+------------+
         |            |            |
   [Feed Cache]  [Feed Cache]  [Feed Cache]
   (User A)      (User B)      (User C)
         |
   [Read Path] --> [Feed Ranking] --> [Response]
```

Key decisions:
- **Fan-out on write** (push model): Pre-compute feeds for followers when a post is created
  - Good for users with few followers
  - Celebrity problem: Fan-out to 100M followers is expensive
- **Fan-out on read** (pull model): Compute feed at read time
  - Good for celebrities
  - Higher read latency
- **Hybrid**: Push for normal users, pull for celebrities

Feed ranking:
- **Features**: Post age, engagement (likes, comments), relationship strength, content type
- **ML model**: Deep learning ranker trained on engagement signals
- **Diversity**: Ensure variety (don't show 10 posts from same friend)

#### Design Instagram Stories

- **Write path**: Upload video/image -> encode -> distribute to CDN
- **Read path**: Fetch stories from followed users, sorted by recency and engagement
- **Ephemeral storage**: Stories expire after 24 hours -- TTL in storage layer
- **Ring UI**: Client-side state management for viewed/unviewed

#### Design Facebook Messenger

- **Real-time delivery**: WebSocket connections for online users, push notifications for offline
- **Message storage**: Ordered by timestamp per conversation, sharded by conversation ID
- **Presence**: Distributed presence service (who's online now)
- **End-to-end encryption**: Key exchange, message encryption/decryption
- **Group messaging**: Fan-out within group, last-writer-wins for reactions

#### Design a Social Graph Service

```
[Query: "Friends of friends who like hiking"]
    |
    v
[Graph Query Engine]
    |
    +-- [Graph Store (TAO)]  -- Entity and relationship storage
    |       |
    |       +-- [Cache Layer]  -- Heavily cached, read-heavy workload
    |
    +-- [Index Service]  -- Inverted index for attribute search
```

- **Storage**: Adjacency list, sharded by entity ID
- **Caching**: Aggressive caching (social graph is read-heavy, ~1000:1 read:write)
- **TAO**: Meta's graph store (Trillions of edges, billions of nodes)
- **Consistency**: Eventual consistency for social graph, strong for auth/permissions

#### Design a Content Recommendation System (AI-Flavored)

```
[User Interaction Stream]
    |
    v
[Feature Service] --> [Candidate Retrieval] --> [Ranking Model]
                                                     |
                                              [Filtering & Safety]
                                                     |
                                              [Personalized Feed]
```

- **Two-tower model**: User embedding + item embedding, nearest neighbor retrieval
- **Ranking**: Multi-task learning (predict likes, comments, shares, time spent)
- **Exploration**: Epsilon-greedy or Thompson sampling for new content
- **Safety**: Content integrity classifier filters harmful/misleading content

### Numbers to Know (Meta Scale)

| Metric | Value |
|--------|-------|
| Daily active users | ~3.2 billion (family of apps) |
| Posts per day | ~1 billion |
| Photos uploaded per day | ~350 million |
| Messages per day | ~100 billion |
| Data stored | Exabytes |
| Cache hit rate (TAO) | >99.9% |
| News Feed reads/sec | Millions |

## Behavioral Round

### Meta Core Values

| Value | What They Assess |
|-------|-----------------|
| Move Fast | Bias toward action, iterative development |
| Be Bold | Take risks, think big |
| Focus on Long-Term Impact | Sustainable decisions, not quick fixes |
| Build Social Value | Care about user impact |
| Be Open | Transparent communication, share information |

### Common Questions

- "Tell me about a time you had to move fast under uncertainty."
- "Describe a time you had to make a decision with incomplete data."
- "Tell me about a project that failed. What did you learn?"
- "How do you handle disagreements with teammates?"
- "Describe a time you had to balance speed with quality."
- "Tell me about a time you influenced a decision without authority."

### STAR Format Tips for Meta

- Keep answers to **2-3 minutes** (Meta interviewers often have many questions)
- **Quantify impact** -- Meta loves metrics
- **Show velocity** -- How fast did you move? Why?
- **Demonstrate learning** -- What would you do differently?

## AI/ML at Meta

### Key Technologies
- **LLaMA**: Open-source LLM family (know architecture basics)
- **PyTorch**: Meta's ML framework (used everywhere)
- **FAISS**: Vector similarity search (know for recommendation design)
- **Custom silicon**: Meta is developing custom AI chips for inference

### ML Infrastructure
- **Feature Store**: Real-time features for recommendation models
- **Training infrastructure**: Large-scale distributed training on custom GPU clusters
- **Model serving**: Low-latency inference for ranking models across all products
- **A/B testing**: Massive experimentation platform (thousands of concurrent experiments)

## Preparation Tips

1. **Leetcode Meta-tagged problems** -- Focus on the top 50 most-asked. Meta draws from a known pool.
2. **Speed is critical** -- Practice solving mediums in 15-20 minutes. You need two in 45 minutes.
3. **Write clean code fast** -- No pseudocode. Working code with good variable names.
4. **Prepare for AI-enabled round** -- Practice using Copilot/Cursor effectively. Know when to trust and when to override.
5. **System design at Meta scale** -- Always think in billions. "How many servers?" "What's the storage cost?"
6. **Read Meta engineering blog** -- engineering.fb.com. Focus on TAO, News Feed ranking, and ML infrastructure.
7. **Behavioral is 25% of score** -- Don't under-prepare. Have 6-8 strong stories ready.

## Sources

- [Meta E5 Interview Guide - HelloInterview](https://www.hellointerview.com/guides/meta/e5)
- [Meta System Design Interview - IGotAnOffer](https://igotanoffer.com/blogs/tech/meta-system-design-interview)
- [Meta Engineering Blog](https://engineering.fb.com/)
- [Proven Meta Software Engineer Interview Guide - Prepfully](https://prepfully.com/interview-guides/meta-software-engineer)
- [Senior Engineer's Guide to Meta Interviews - interviewing.io](https://interviewing.io/guides/hiring-process/meta-facebook)
- [Glassdoor - Meta SWE Interview Questions](https://www.glassdoor.com/Interview/Meta-Software-Engineer-Interview-Questions-EI_IE40772.0,4_KO5,22.htm)
- [levels.fyi - Meta Compensation Data](https://www.levels.fyi/companies/meta/salaries/software-engineer)
- r/cscareerquestions, Blind (community reports)
