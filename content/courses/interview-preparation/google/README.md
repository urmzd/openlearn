# Google

Google's software engineering interview is the template many other companies copied: several rounds of algorithmic coding on a plain editor, one open-ended system design round, and a behavioral round built around "Googleyness". What makes it hard is not any single trick but the combination of breadth and scale. Problems are drawn from a wide range of topics, you are expected to write correct code without an IDE, and every answer is eventually tested against the follow-up "how does this work with a billion rows?"

This chapter prepares you for Google SWE roles with an emphasis on full-stack AI and ML infrastructure. It covers the process and compensation, the themes interviewers assess, the coding rounds with three patterns worth mastering, the system design round with four common prompts and the latency numbers you should know, the Googleyness and leadership round, and the Google-specific AI infrastructure that comes up for ML-focused roles.

> **Learning objectives**
> - Describe the Google interview loop, including hiring committee and team matching.
> - Write BFS with extra search state, a constrained sliding window, and a top-K autocomplete trie without an IDE.
> - Structure a system design answer in six steps and apply it to search, video serving, recommendation, and model serving.
> - Quote the core latency numbers and use them in estimates.
> - Answer behavioral questions in STAR form with the intellectual humility Google values.

## Before you begin

The algorithm patterns here assume comfort with graphs, hashing, and heaps. For the distributed systems vocabulary behind the design round, read [Distributed systems](../shared-concepts/distributed-systems.md); for the ML-flavoured designs, read [Full-stack AI](../shared-concepts/full-stack-ai.md) and [ML infrastructure](../shared-concepts/ml-infrastructure.md). The [Google DeepMind](../deepmind/README.md) chapter builds on this one for research-heavy roles.

## The interview process

The process takes four to eight weeks, a range that is notoriously variable, and has five to seven rounds:

| Round | Format | Duration | Focus |
|-------|--------|----------|-------|
| Recruiter Screen | Phone | 30 min | Background, level calibration |
| Phone Screen | Coding (Google Meet) | 45 min | DS&A, one medium/hard problem |
| Onsite 1 | Coding | 45 min | Algorithms, data structures |
| Onsite 2 | Coding | 45 min | Algorithms, data structures |
| Onsite 3 | System Design | 45 min | Large-scale distributed systems |
| Onsite 4 | Behavioral (Googleyness & Leadership) | 45 min | Culture, collaboration, ambiguity |
| Onsite 5 (optional) | Coding or Design | 45 min | Tiebreaker round |

The interviews are not the end. A hiring committee reviews your packet, which takes one to three weeks, and then team matching takes another one to four weeks, so the total timeline can stretch significantly.

At L5 (senior) in the US, base salary is about $180-220K, equity is $200-400K per year in GSUs vesting over four years, and the bonus is 15-30% of base. Total compensation is around $400-650K at L5 and $600K-1M or more at L6 (staff).

## What Google looks for

Five themes shape the loop. Scale is the lens: every solution is evaluated at Google scale, billions of users and petabytes of data, and "how does this work with 1B rows?" is always the follow-up. Coding fluency matters: Google wants clean, compilable code in 20-25 minutes, and wants to see you think in code rather than pseudocode. System design is open-ended, with no single right answer; what is assessed is structured thinking, trade-off analysis, and the ability to go deep when probed. Googleyness is real, meaning interviewers evaluate how you handle ambiguity, disagree respectfully, and navigate without authority. And AI and ML are increasingly central, since Google is an AI-first company; expect ML-flavoured system design such as recommendation systems, search ranking, and model serving.

## Coding rounds

There are two or three coding rounds. Each 45-minute session has one or two problems, written in Google Docs or a simple shared editor with no autocomplete and no way to run code. The interviewer expects you to write correct code, not pseudocode, and follow-ups are common: optimise it, handle edge cases, extend it.

Topics appear with roughly this frequency, and most have a direct analogue in AI and ML work:

| Topic | Frequency | AI/ML Relevance |
|-------|-----------|-----------------|
| Arrays/Strings | Very High | Data preprocessing, tokenization |
| Graphs (BFS/DFS) | Very High | Knowledge graphs, dependency resolution |
| Dynamic Programming | High | Sequence models, Viterbi algorithm |
| Trees/Tries | High | Decision trees, prefix matching |
| Hash Maps | Very High | Feature stores, caching |
| Sliding Window | High | Streaming data, time-series |
| Binary Search | Medium | Hyperparameter tuning, sorted retrieval |
| Heaps/Priority Queues | Medium | Top-K, scheduling |
| Union-Find | Medium | Clustering, connected components |
| Concurrency | Medium-High (for infra) | Model serving, request handling |

Three patterns are worth mastering in detail, because they generalise to many of the reported problems.

### Graph search with extra state

Many grid and graph problems become simple BFS once you realise that the node is not just a position but a position plus some resource. Here the state is `(row, col, obstacles_remaining)`: you may walk through at most `max_obstacles` blocked cells, and the visited set must include the remaining budget, because reaching the same cell with more budget left is a genuinely different state.

```python
from collections import deque

def shortest_path_with_constraints(grid, start, end, max_obstacles):
    """BFS with state: (row, col, obstacles_remaining)."""
    rows, cols = len(grid), len(grid[0])
    queue = deque([(start[0], start[1], max_obstacles, 0)])  # r, c, obstacles_left, dist
    visited = set()
    visited.add((start[0], start[1], max_obstacles))

    while queue:
        r, c, obs, dist = queue.popleft()
        if (r, c) == end:
            return dist
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                new_obs = obs - (1 if grid[nr][nc] == 1 else 0)
                if new_obs >= 0 and (nr, nc, new_obs) not in visited:
                    visited.add((nr, nc, new_obs))
                    queue.append((nr, nc, new_obs, dist + 1))
    return -1
```

Because BFS explores in order of distance, the first time it pops the target it has found the shortest path. The state space is at most `rows * cols * (max_obstacles + 1)`, which bounds both time and memory.

### Sliding window over a stream

The sliding window pattern processes a sequence in one pass by maintaining a running aggregate over a window. In this variant you want the maximum sum of a size-`k` subarray whose elements are all at most `max_val`. An element over the limit cannot be in any valid window, so the window resets just past it.

```python
def max_sum_subarray_with_constraint(nums, k, max_val):
    """Max sum of subarray of size k where all elements <= max_val."""
    window_sum = 0
    max_sum = float('-inf')
    left = 0

    for right in range(len(nums)):
        if nums[right] > max_val:
            window_sum = 0
            left = right + 1
            continue
        window_sum += nums[right]
        if right - left + 1 == k:
            max_sum = max(max_sum, window_sum)
            window_sum -= nums[left]
            left += 1

    return max_sum if max_sum != float('-inf') else -1
```

Each element is added once and removed at most once, so the whole pass is $O(n)$. The function returns `-1` when no valid window exists.

### A trie for autocomplete

The third pattern trades memory for query speed. A trie stores words character by character, and this one also stores, at every node, the top ten results for that prefix. Insertion does a little extra work at each node; autocomplete then just walks the prefix and reads the precomputed list.

```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.top_results = []  # Pre-computed top-K results for this prefix

class AutocompleteTrie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: str, score: float):
        node = self.root
        for ch in word:
            if ch not in node.children:
                node.children[ch] = TrieNode()
            node = node.children[ch]
            # Maintain top-K at each node
            node.top_results.append((score, word))
            node.top_results.sort(reverse=True)
            node.top_results = node.top_results[:10]

    def autocomplete(self, prefix: str, k: int = 5) -> list[str]:
        node = self.root
        for ch in prefix:
            if ch not in node.children:
                return []
            node = node.children[ch]
        return [word for _, word in node.top_results[:k]]
```

A query costs $O(|\text{prefix}| + k)$ regardless of how many words share the prefix, which is the property a real autocomplete service needs. Be ready to discuss the cost: memory grows with the number of nodes times ten, and updating a word's score means touching every node on its path.

### Problems reported from Google interviews

These problems are reported often, each with the technique that cracks it:

- **Minimum window substring**: sliding window with a frequency map.
- **Word ladder**: BFS on the word graph.
- **Course schedule**: topological sort, which is cycle detection in a directed graph.
- **Median of two sorted arrays**: binary search in $O(\log(m+n))$.
- **Design a rate limiter**: sliding-window counter or token bucket.
- **Serialize and deserialize a binary tree**: BFS or preorder with null markers.
- **LRU cache**: doubly linked list plus hash map, the same problem asked at [Anthropic](../anthropic/README.md).
- **Merge K sorted lists**: a min-heap.
- **Trapping rain water**: two pointers or a stack.
- **Number of islands**: BFS or DFS flood fill.

## The system design round

Google interviewers want to see a particular arc. Start by gathering requirements, functional and non-functional, with numbers. Then design the API, as RESTful endpoints or RPC definitions. Draw the high-level architecture, the boxes and arrows, and the data model, with schema design and storage choices. The interviewer will then pick one or two areas for a deep dive. Finish on scale and reliability: how the design handles 10x traffic and what fails. Four prompts are common.

### Google Search

```
[Query] --> [Web Server] --> [Query Parser]
                                 |
                          [Index Servers (sharded)]
                                 |
                          [Ranking Service]
                                 |
                          [Ads Injection]
                                 |
                          [Result Aggregation] --> [Response]
```

The core is an inverted index sharded either by document ID or by term. PageRank is precomputed offline and used as one ranking signal. Before the index is queried, query understanding applies spelling correction, query expansion, and entity recognition. Freshness comes from running a real-time index for recent content beside a batch index for historical content, and caching is multi-tier: CDN, query cache, and result cache.

### YouTube and video serving

For video, the upload pipeline transcodes each video to multiple resolutions for adaptive bitrate streaming (HLS or DASH). A CDN provides edge caching and replicates hot content. Recommendation combines collaborative filtering with deep learning, typically a two-tower model. Videos live in blob storage and metadata in Bigtable or Spanner.

### A recommendation system

The AI-flavoured version of the design round is a recommender:

```
[User Action Stream] --> [Feature Store] --> [Candidate Generation]
                                                     |
                                              [Ranking Model (ML)]
                                                     |
                                              [Filtering & Diversity]
                                                     |
                                              [Results]
```

Candidate generation narrows millions of items to hundreds with approximate nearest neighbours (ANN) and collaborative filtering. A deep neural network then ranks those candidates using user and item features. The feature store serves both real-time features, such as the last ten actions, and batch features, such as the user profile. Embeddings are precomputed and served from a vector index (ScaNN at Google). Freshness is a balance between exploitation, showing known-good items, and exploration, discovering new ones.

### A model serving platform

For AI-focused roles this prompt is highly relevant. A model registry provides version control for models and supports A/B testing. The serving infrastructure handles GPU allocation, batching, and auto-scaling. Feature serving provides low-latency feature retrieval for online inference. Monitoring covers model drift detection and prediction quality tracking. Deployment is canaried: a gradual rollout with automated rollback.

### Numbers to know

Estimates in the design round rest on a handful of latency and throughput figures:

| Resource | Capacity |
|----------|----------|
| Single SSD read | ~100 μs |
| Single HDD read | ~10 ms |
| Network round trip (same DC) | ~0.5 ms |
| Network round trip (cross-continent) | ~100 ms |
| 1 GB over 1 Gbps network | ~10 sec |
| Bigtable write | ~5 ms |
| Spanner global write | ~10-100 ms |

## Googleyness and leadership

The behavioral round scores four attributes:

| Attribute | What They're Assessing |
|-----------|----------------------|
| Googleyness | Do the right thing, comfort with ambiguity, collaborative |
| Leadership | Influence without authority, mentoring, raising the bar |
| Role-Related Knowledge | Technical depth appropriate for level |
| General Cognitive Ability | Problem-solving, learning ability |

Common questions include "Tell me about a time you had to work with incomplete information.", "Describe a situation where you disagreed with your team's approach.", "Tell me about a time you had to influence a decision without having authority.", "Describe a project that failed. What did you learn?", and "Tell me about a time you helped someone else grow."

Answer in the STAR shape. Give the situation in one or two sentences of context, then the task, your specific responsibility. Spend most of the answer on the action: what you did, specifically, as distinct from the team. End with the result, quantified, and what you learned. Above all, Google values intellectual humility. Do not oversell; acknowledge mistakes and what you would do differently.

## AI and ML preparation

For full-stack AI roles, also prepare the Google stack around ML: TensorFlow Serving and Vertex AI for the model deployment pipeline, MapReduce and Dataflow for large-scale data processing, Bigtable and Spanner for ML features and metadata, Pub/Sub for event streaming into real-time features, and Kubernetes (GKE) for orchestrating model serving.

Know Google's own AI infrastructure as well. TPUs are Google's custom ML accelerators, and you should know the TPU-versus-GPU trade-offs. Pathways is Google's next-generation ML infrastructure for multi-task models. Gemini is its multimodal model, worth understanding at a high level. Vertex AI is the managed ML platform covering training, serving, and monitoring.

## Preparing

Work 150-200 LeetCode problems, focused on medium and hard, since Google draws from a wide range. Practise writing real code in Google Docs, without an IDE, until you are comfortable without autocomplete, and time yourself: 20 minutes per medium problem, 25 per hard. Practise system design with a friend on a whiteboard or shared document. Do at least three to five mock interviews before your onsite, through Pramp, interviewing.io, or peers. And read the Google engineering blog to understand the company's infrastructure philosophy.

## Summary

- The loop is five to seven rounds, followed by hiring committee and team matching, which can add weeks.
- Coding is judged on correct, compilable code written without an IDE, followed by optimisation and extension.
- Model the search state explicitly, keep windows $O(n)$, and precompute where queries must be fast.
- System design follows requirements, API, architecture, data model, deep dive, and scale; back estimates with real latency numbers.
- Googleyness rewards comfort with ambiguity, influence without authority, and intellectual humility.

## Further reading

- [How 12 Top Tech Companies Interview Software Engineers (Exponent)](https://www.tryexponent.com/blog/how-top-tech-companies-interview-software-engineers)
- [The Reality of Tech Interviews in 2025 (Pragmatic Engineer)](https://newsletter.pragmaticengineer.com/p/the-reality-of-tech-interviews)
- [Google SWE Interview (IGotAnOffer)](https://igotanoffer.com/blogs/tech/google-software-engineer-interview)
- [Google Engineering Blog](https://blog.research.google/)
- [Glassdoor: Google SWE Interview Questions](https://www.glassdoor.com/Interview/Google-Software-Engineer-Interview-Questions-EI_IE9079.0,6_KO7,24.htm)
- [levels.fyi: Google Compensation Data](https://www.levels.fyi/companies/google/salaries/software-engineer)
- Community reports on r/cscareerquestions and r/ExperiencedDevs.
