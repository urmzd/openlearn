# Google Software Engineer Interview Guide

Comprehensive preparation for Google SWE roles, with emphasis on full-stack AI/ML infrastructure.

## Interview Process Overview

Timeline: **4-8 weeks** (notoriously variable), **5-7 rounds**

| Round | Format | Duration | Focus |
|-------|--------|----------|-------|
| Recruiter Screen | Phone | 30 min | Background, level calibration |
| Phone Screen | Coding (Google Meet) | 45 min | DS&A, one medium/hard problem |
| Onsite 1 | Coding | 45 min | Algorithms, data structures |
| Onsite 2 | Coding | 45 min | Algorithms, data structures |
| Onsite 3 | System Design | 45 min | Large-scale distributed systems |
| Onsite 4 | Behavioral (Googleyness & Leadership) | 45 min | Culture, collaboration, ambiguity |
| Onsite 5 (optional) | Coding or Design | 45 min | Tiebreaker round |

After interviews: **Hiring Committee review** (1-3 weeks), then **Team Matching** (1-4 weeks). Total timeline can stretch significantly.

## Compensation (L5 Senior, US)

- **Base**: $180-220K
- **Equity**: $200-400K/yr (GSUs, vest over 4 years)
- **Bonus**: $15-30% of base
- **Total Comp**: ~$400-650K (L5), ~$600K-1M+ (L6 Staff)

## Key Themes

1. **Scale is the lens** -- Every solution is evaluated at Google scale (billions of users, petabytes of data). "How does this work with 1B rows?" is always the follow-up.
2. **Coding fluency matters** -- Clean, compilable code in 20-25 minutes. They want to see you think in code, not pseudocode.
3. **System design is open-ended** -- No single right answer. They want structured thinking, trade-off analysis, and the ability to go deep when probed.
4. **Googleyness is real** -- They evaluate how you handle ambiguity, disagree respectfully, and navigate without authority.
5. **AI/ML is increasingly central** -- Google is an AI-first company. Expect ML-flavored system design (recommendation systems, search ranking, model serving).

## Coding Rounds (2-3 rounds)

### What to Expect

- 1-2 problems per 45-minute session
- Google Docs or a simple shared editor (no autocomplete, no running code)
- Interviewer expects you to **write correct code**, not pseudocode
- Follow-ups are common: optimize, handle edge cases, extend

### Frequency Distribution of Topics

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

### Patterns to Master

#### Graph BFS/DFS with State

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

#### Sliding Window for Streaming

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

#### Trie for Autocomplete/Prefix Search

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

### Common Google Problems (Reported)

- **Minimum window substring** -- Sliding window + frequency map
- **Word ladder** -- BFS on word graph
- **Course schedule** -- Topological sort (cycle detection in DAG)
- **Median of two sorted arrays** -- Binary search, O(log(m+n))
- **Design a rate limiter** -- Sliding window counter or token bucket
- **Serialize/deserialize binary tree** -- BFS or preorder with null markers
- **LRU Cache** -- DLL + hashmap (same as Anthropic)
- **Merge K sorted lists** -- Min-heap
- **Trapping rain water** -- Two-pointer or stack
- **Number of islands** -- BFS/DFS flood fill

## System Design Round

### Approach (Google-Specific)

Google interviewers want to see:
1. **Requirements gathering** -- Functional and non-functional, with numbers
2. **API design** -- RESTful endpoints or RPC definitions
3. **High-level architecture** -- Draw the boxes and arrows
4. **Data model** -- Schema design, storage choices
5. **Deep dive** -- Interviewer picks 1-2 areas to probe
6. **Scale and reliability** -- How does it handle 10x traffic? What fails?

### Common Google System Design Topics

#### Design Google Search

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

Key points:
- **Inverted index** sharded by document ID or term
- **PageRank** precomputed offline, used as a ranking signal
- **Query understanding**: spelling correction, query expansion, entity recognition
- **Freshness**: real-time index for recent content, batch index for historical
- **Caching**: Multi-tier (CDN, query cache, result cache)

#### Design YouTube / Video Serving

- **Upload pipeline**: Transcoding to multiple resolutions, adaptive bitrate (HLS/DASH)
- **CDN**: Edge caching, hot content replication
- **Recommendation**: Collaborative filtering + deep learning (Two-Tower model)
- **Storage**: Blob storage for videos, metadata in Bigtable/Spanner

#### Design a Recommendation System (AI-Flavored)

```
[User Action Stream] --> [Feature Store] --> [Candidate Generation]
                                                     |
                                              [Ranking Model (ML)]
                                                     |
                                              [Filtering & Diversity]
                                                     |
                                              [Results]
```

- **Candidate generation**: Approximate nearest neighbors (ANN), collaborative filtering
- **Ranking**: Deep neural network (DNN) with user/item features
- **Feature store**: Real-time features (last 10 actions) + batch features (user profile)
- **Serving**: Pre-compute embeddings, serve with vector DB (ScaNN)
- **Freshness**: Balance between exploitation (show known-good) and exploration (discover new)

#### Design a Model Serving Platform

Highly relevant for AI-focused roles:

- **Model registry**: Version control for models, A/B testing support
- **Serving infrastructure**: GPU allocation, batching, auto-scaling
- **Feature serving**: Low-latency feature retrieval for online inference
- **Monitoring**: Model drift detection, prediction quality tracking
- **Canary deployment**: Gradual rollout with automated rollback

### Numbers to Know

| Resource | Capacity |
|----------|----------|
| Single SSD read | ~100 μs |
| Single HDD read | ~10 ms |
| Network round trip (same DC) | ~0.5 ms |
| Network round trip (cross-continent) | ~100 ms |
| 1 GB over 1 Gbps network | ~10 sec |
| Bigtable write | ~5 ms |
| Spanner global write | ~10-100 ms |

## Googleyness & Leadership (Behavioral)

### Core Attributes

| Attribute | What They're Assessing |
|-----------|----------------------|
| Googleyness | Do the right thing, comfort with ambiguity, collaborative |
| Leadership | Influence without authority, mentoring, raising the bar |
| Role-Related Knowledge | Technical depth appropriate for level |
| General Cognitive Ability | Problem-solving, learning ability |

### Common Questions

- "Tell me about a time you had to work with incomplete information."
- "Describe a situation where you disagreed with your team's approach."
- "Tell me about a time you had to influence a decision without having authority."
- "Describe a project that failed. What did you learn?"
- "Tell me about a time you helped someone else grow."

### STAR Framework (Adapted for Google)

- **Situation**: Brief context (1-2 sentences)
- **Task**: What was your specific responsibility?
- **Action**: What did YOU do? (Be specific about your individual contribution)
- **Result**: Quantifiable outcome + what you learned

**Key**: Google values **intellectual humility**. Don't oversell. Acknowledge mistakes and what you'd do differently.

## AI/ML-Specific Preparation

For full-stack AI roles at Google, also prepare:

- **TensorFlow Serving / Vertex AI**: Model deployment pipeline
- **MapReduce / Dataflow**: Large-scale data processing
- **Bigtable / Spanner**: Storage for ML features and metadata
- **Pub/Sub**: Event streaming for real-time ML features
- **Kubernetes (GKE)**: Container orchestration for model serving

### Google-Specific AI Infrastructure

- **TPUs**: Google's custom ML accelerators. Know TPU vs GPU trade-offs.
- **Pathways**: Google's next-gen ML infrastructure for multi-task models
- **Gemini**: Multimodal model architecture (understand at a high level)
- **Vertex AI**: Managed ML platform (training, serving, monitoring)

## Preparation Tips

1. **Leetcode**: 150-200 problems, focus on medium/hard. Google draws from a wide range.
2. **Write real code**: Practice in Google Docs (no IDE). Get comfortable without autocomplete.
3. **Time yourself**: 20 minutes per medium problem, 25 minutes per hard.
4. **System design**: Practice with a friend. Draw on a whiteboard or shared doc.
5. **Mock interviews**: At least 3-5 before your onsite. Pramp, interviewing.io, or peers.
6. **Read Google engineering blog**: Understand their infrastructure philosophy.

## Sources

- [How 12 Top Tech Companies Interview Software Engineers - Exponent](https://www.tryexponent.com/blog/how-top-tech-companies-interview-software-engineers)
- [The Reality of Tech Interviews in 2025 - Pragmatic Engineer](https://newsletter.pragmaticengineer.com/p/the-reality-of-tech-interviews)
- [Google SWE Interview - IGotAnOffer](https://igotanoffer.com/blogs/tech/google-software-engineer-interview)
- [Google Engineering Blog](https://blog.research.google/)
- [Glassdoor - Google SWE Interview Questions](https://www.glassdoor.com/Interview/Google-Software-Engineer-Interview-Questions-EI_IE9079.0,6_KO7,24.htm)
- [levels.fyi - Google Compensation Data](https://www.levels.fyi/companies/google/salaries/software-engineer)
- r/cscareerquestions, r/ExperiencedDevs (Reddit community reports)
