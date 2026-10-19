# Meta

Meta is one of the largest AI-focused employers in the world, and its interview process is among the most structured and best documented in the industry. That cuts both ways. Because the process is consistent and draws from a known pool of problems, it is one of the most "preparable" interviews you will face; because everyone else knows that too, the bar for speed is high. Two coding problems in 45 minutes leaves about twenty minutes each, with no time for hesitation.

The process has also changed recently. Since 2025 Meta screens with a proctored online assessment and has added an AI-enabled coding round in which you work with an AI assistant in the editor. This chapter covers both, along with the traditional coding round and its key patterns, the product-centred system design round with five common prompts and the numbers that describe Meta's scale, the behavioral round built on Meta's core values, and the AI and ML technologies worth knowing.

> **Learning objectives**
> - Describe the Meta interview loop, including the online assessment and the AI-enabled coding round.
> - Solve the core Meta coding patterns (hash-map lookups, level-order BFS, graph cloning, monotonic stacks) in under twenty minutes each.
> - Work productively with an AI assistant in an interview while showing you understand and can correct its output.
> - Design News Feed, Stories, Messenger, a social graph service, and a recommender, including the fan-out trade-off.
> - Prepare concise, quantified behavioral stories aligned to Meta's values.

## Before you begin

The system design material assumes the vocabulary in [Distributed systems](../shared-concepts/distributed-systems.md) (caching, sharding, consistency) and, for recommendation, [Full-stack AI](../shared-concepts/full-stack-ai.md). The online assessment closely resembles the [Anthropic online assessment](../anthropic/01-online-assessment.md), which is worth reading alongside this chapter.

## The interview process

Expect three to six weeks and five or six rounds:

| Round | Format | Duration | Focus |
|-------|--------|----------|-------|
| Online Assessment | CodeSignal (proctored) | 90 min | Progressive coding problem (4 stages) |
| Recruiter Screen | Phone | 30 min | Background, level calibration |
| Phone Screen | Coding (CoderPad) | 45 min | 2 LC medium problems |
| Onsite 1 | Coding | 45 min | Traditional DS&A |
| Onsite 2 | AI-Enabled Coding | 60 min | Coding with AI assistant (new 2025+) |
| Onsite 3 | System Design | 45 min | Large-scale distributed systems |
| Onsite 4 | Behavioral | 45 min | Meta core values alignment |

The AI-enabled round is new for 2025-2026. Candidates have access to an AI assistant in CoderPad, and the round tests your ability to use AI tools effectively, not just raw coding ability.

At E5 (senior) in the US, base salary is $190-220K, RSUs are about $250-500K per year (vesting quarterly after a one-year cliff), and the bonus is 15-25% of base. Total compensation is around $450-700K at E5 and $700K-1.2M or more at E6 (staff). Meta refreshes equity aggressively for top performers.

## What Meta looks for

Five themes run through the loop. The first is speed and volume: the interviews are fast-paced, and two coding problems in 45 minutes means about twenty minutes per problem. The second is the cultural counterpart, "move fast": Meta values velocity, so in behavioral answers show a bias toward action and quick iteration. The third is the AI-first transformation; Meta is pivoting aggressively to AI (LLaMA, AI assistants, recommendation engines), so AI knowledge is increasingly relevant. The fourth is that scale is the default: everything serves billions of users, and "how does this work at 3B users?" is always the lens. The fifth is the structured process itself, which is well documented and consistent and therefore rewards preparation.

## The online assessment

As of 2025, Meta screens with CodeSignal under full video and microphone proctoring. You get 90 minutes for one complex problem divided into four progressive stages that unlock sequentially. Reported problems include an in-memory database with key-value operations (SET, GET, DELETE, filtering, transactions), a cloud-based file storage service (upload, download, permissions, versioning), and a task scheduling system with dependencies and priorities. The format is very similar to Anthropic's online assessment, and as there, production quality matters: later stages are much easier if the early code was clean.

## Coding rounds

There are two coding rounds onsite: a traditional one and the AI-enabled one.

### The traditional round

The traditional round gives you two medium problems in 45 minutes, about twenty minutes each plus a five-minute introduction. Meta draws heavily from these categories:

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

Four patterns cover a large share of these, and you should be able to write each from memory.

The first is the one-pass hash map. For Two Sum you walk the array once, and for each number ask whether its complement has already been seen. Storing each number after the check means an element is never paired with itself.

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

The second is BFS with level tracking. Recording `len(queue)` at the start of each iteration tells you exactly how many nodes belong to the current level, so you can act on the first or last node of each level. For the right side view, the answer is the last node of every level.

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

The third is graph cloning with a visited map. The map from original node to copy does double duty: it prevents infinite recursion on cycles, and it ensures each node is copied exactly once so shared neighbours stay shared. The copy is registered in the map before its neighbours are visited, which is what makes cycles work.

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

The fourth is the monotonic stack. To find, for each day, how many days until a warmer one, keep a stack of indices whose temperatures are decreasing. Each new temperature pops every cooler day off the stack and resolves its answer. Each index is pushed and popped at most once, so the whole pass is $O(n)$.

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

### The AI-enabled round

The newer 60-minute round gives you an AI coding assistant in the editor. What it tests is not whether you can get the AI to write code, but whether you can work with it as an engineer: whether you can prompt it effectively, whether you understand the code it generates, whether you can debug that code, whether you can tell when its output is wrong, and whether you can iterate and refine with its help.

The strategy follows from that. Use the AI for boilerplate, syntax reminders, and generating edge cases. Always review and understand its output before you use it. Explain your reasoning to the interviewer as you go rather than copying and pasting. And expect the problems to be harder than in the traditional round, precisely because you have help.

## The system design round

Meta's system design questions revolve around its products, News Feed, Messenger, Instagram, and WhatsApp, and increasingly around AI and ML systems. The round follows a predictable shape, and keeping to the time budget for each phase matters:

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

Five prompts come up often.

### News Feed

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

The central decision is when to fan out. With fan-out on write (the push model), you precompute each follower's feed when a post is created. That is good for users with few followers, but it creates the celebrity problem: fanning out to 100M followers is expensive. With fan-out on read (the pull model), you compute the feed at read time, which suits celebrities but raises read latency. The usual answer is a hybrid: push for normal users and pull for celebrities, merging the two at read time.

Ranking then orders the candidate posts. Its features include post age, engagement (likes, comments), relationship strength, and content type; the model is a deep learning ranker trained on engagement signals; and a diversity pass ensures variety, so you do not see ten posts from the same friend.

### Instagram Stories

On the write path, an uploaded video or image is encoded and distributed to the CDN. On the read path, the client fetches stories from followed users, sorted by recency and engagement. Stories are ephemeral and expire after 24 hours, which is naturally expressed as a TTL in the storage layer. The ring UI's viewed and unviewed state is managed on the client.

### Messenger

Messenger needs real-time delivery, over WebSocket connections for online users and push notifications for offline ones. Messages are stored ordered by timestamp within each conversation and sharded by conversation ID. A distributed presence service tracks who is online. End-to-end encryption requires key exchange and per-message encryption and decryption. Group messaging fans out within the group, and reactions can use last-writer-wins.

### A social graph service

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

The graph is stored as adjacency lists sharded by entity ID. Because the social graph is read-heavy, around 1000:1 reads to writes, caching is aggressive. Meta's own store, TAO, holds trillions of edges and billions of nodes. The consistency model is split: eventual consistency is fine for the social graph, but authentication and permissions need strong consistency.

### A content recommendation system

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

Retrieval uses a two-tower model, a user embedding and an item embedding, with nearest-neighbour search. Ranking uses multi-task learning to predict likes, comments, shares, and time spent together. Exploration of new content uses epsilon-greedy or Thompson sampling. A content integrity classifier filters harmful or misleading content before anything is shown.

### Numbers to know

Meta-scale estimates start from these figures:

| Metric | Value |
|--------|-------|
| Daily active users | ~3.2 billion (family of apps) |
| Posts per day | ~1 billion |
| Photos uploaded per day | ~350 million |
| Messages per day | ~100 billion |
| Data stored | Exabytes |
| Cache hit rate (TAO) | >99.9% |
| News Feed reads/sec | Millions |

## The behavioral round

The behavioral round assesses alignment with Meta's core values:

| Value | What They Assess |
|-------|-----------------|
| Move Fast | Bias toward action, iterative development |
| Be Bold | Take risks, think big |
| Focus on Long-Term Impact | Sustainable decisions, not quick fixes |
| Build Social Value | Care about user impact |
| Be Open | Transparent communication, share information |

Common questions include "Tell me about a time you had to move fast under uncertainty.", "Describe a time you had to make a decision with incomplete data.", "Tell me about a project that failed. What did you learn?", "How do you handle disagreements with teammates?", "Describe a time you had to balance speed with quality.", and "Tell me about a time you influenced a decision without authority."

Use the STAR format, adapted for Meta. Keep each answer to two or three minutes, because interviewers often have many questions to get through. Quantify impact, since Meta loves metrics. Show velocity: how fast you moved, and why. And demonstrate learning by saying what you would do differently.

## AI and ML at Meta

Know the key technologies: LLaMA, Meta's open-source LLM family (know the architecture basics); PyTorch, Meta's ML framework, used everywhere; FAISS, for vector similarity search, which comes up in recommendation design; and Meta's custom AI chips for inference. On the infrastructure side, know the feature store that supplies real-time features to recommendation models, the large-scale distributed training on custom GPU clusters, low-latency serving of ranking models across all products, and the massive A/B testing platform that runs thousands of concurrent experiments.

## Preparing

Work the LeetCode Meta-tagged problems, focusing on the 50 most asked, since Meta draws from a known pool. Speed is critical: practise solving mediums in 15-20 minutes, because you need two in 45. Write clean code fast, with no pseudocode, just working code with good variable names. Prepare for the AI-enabled round by practising with Copilot or Cursor until you know when to trust the tool and when to override it. In system design, always think in billions: how many servers, what the storage costs. Read the Meta engineering blog at engineering.fb.com, focusing on TAO, News Feed ranking, and ML infrastructure. And do not under-prepare for behavioral, which is about 25% of the score; have six to eight strong stories ready.

## Summary

- Meta's loop is structured and preparable, but fast: about twenty minutes per coding problem.
- The proctored online assessment is a four-stage progressive problem where clean early code pays off.
- The AI-enabled round tests judgment about AI output, not the ability to paste it.
- News Feed design hinges on the push, pull, or hybrid fan-out trade-off; the social graph is read-heavy and cached.
- Behavioral answers should be short, quantified, and show velocity and learning.

## Further reading

- [Meta E5 Interview Guide (HelloInterview)](https://www.hellointerview.com/guides/meta/e5)
- [Meta System Design Interview (IGotAnOffer)](https://igotanoffer.com/blogs/tech/meta-system-design-interview)
- [Meta Engineering Blog](https://engineering.fb.com/)
- [Proven Meta Software Engineer Interview Guide (Prepfully)](https://prepfully.com/interview-guides/meta-software-engineer)
- [Senior Engineer's Guide to Meta Interviews (interviewing.io)](https://interviewing.io/guides/hiring-process/meta-facebook)
- [Glassdoor: Meta SWE Interview Questions](https://www.glassdoor.com/Interview/Meta-Software-Engineer-Interview-Questions-EI_IE40772.0,4_KO5,22.htm)
- [levels.fyi: Meta Compensation Data](https://www.levels.fyi/companies/meta/salaries/software-engineer)
- Community reports on r/cscareerquestions and Blind.
