# Palantir

Palantir builds data integration and analysis platforms (Gotham, Foundry, and the newer AIP) for governments and large enterprises. Its interviews are shaped by that work. The coding rounds favour graph problems, because Palantir's products are fundamentally about entities and relationships. And Palantir has a signature round found almost nowhere else, the decomposition interview, in which you are handed a vague real-world problem ("help a hospital manage patient flow") and asked to break it into entities, workflows, and technical components.

This chapter prepares you for Palantir's Forward Deployed Engineer (FDE) and Backend Engineer roles, with a focus on data infrastructure and AI systems. It covers the two tracks and the process, the coding round with graph, data transformation, and timeline problems, a complete worked decomposition, the system design round on data integration, enterprise LLM assistants, and geospatial analytics, and the values round, including how to engage with the ethical questions Palantir's work raises.

> **Learning objectives**
> - Distinguish Palantir's FDE and backend tracks and describe the interview loop.
> - Implement BFS shortest path, connected components, and three-colour cycle detection on an adjacency-set graph.
> - Write a path-based transformer for nested data and a sweep-line merge of overlapping timelines.
> - Decompose an ambiguous scenario into an ontology, workflows, an architecture, and trade-offs within thirty minutes.
> - Design a data integration platform and an ontology-grounded LLM assistant with access control and audit.

## Before you begin

The graph algorithms assume familiarity with BFS and DFS. The design material draws on [Full-stack AI](../shared-concepts/full-stack-ai.md) for retrieval-augmented generation and on [Distributed systems](../shared-concepts/distributed-systems.md) for pipelines and storage. Data modelling practice, designing schemas for real domains, is the best general preparation for the decomposition round.

## The interview process

Expect three to five weeks and four or five rounds:

| Round | Format | Duration | Focus |
|-------|--------|----------|-------|
| Recruiter Screen | Phone | 30 min | Background, role fit |
| Phone Screen | Coding (HackerRank/Karat) | 60 min | DS&A |
| Onsite 1 | Coding | 45-60 min | Algorithms, data structures |
| Onsite 2 | Decomposition | 60 min | Break down ambiguous problems |
| Onsite 3 | System Design | 60 min | Data platform architecture |
| Onsite 4 | Behavioral / Values | 45 min | Mission, collaboration, impact |

Palantir has two main SWE tracks. Forward Deployed Engineers are client-facing and full-stack, adapting Palantir's platforms to each customer's needs. Backend and infrastructure engineers develop the core platforms: Gotham, Foundry, and AIP.

Reported total compensation is about $200-250K for new grads, $350-500K for senior SWEs, and $500-800K for staff and above. Equity is RSUs with four-year vesting, and since Palantir is public, the equity is liquid.

## What Palantir looks for

The decomposition round is unique to Palantir and is its signature interview, testing your ability to break an ambiguous, real-world problem into technical components. Behind it is data platform thinking: ontologies, data models, and workflows. Palantir is mission-driven, working with governments and large enterprises, and looks for candidates who care about impact and can handle the ethical complexity. It wants full-stack generalists, especially for FDE roles: engineers who can do frontend, backend, data engineering, and client communication. And its fastest-growing product, AIP (Artificial Intelligence Platform), integrates LLMs into enterprise workflows, so AI and ML platform knowledge is increasingly relevant.

## The coding round

The coding round is standard data structures and algorithms at medium to hard difficulty, on HackerRank for the phone screen or a whiteboard onsite. Clean code matters: variable naming, modularity, and edge cases. Three problem families are reported most often.

### Graph problems

Graph problems are very common, since Palantir's products are about entities and relationships. A single adjacency-set class covers the usual three questions. Shortest path in an unweighted graph is BFS, carrying the path along with each node and marking nodes visited when they are enqueued so none is queued twice. Connected components is repeated DFS from every unvisited node. Cycle detection in a directed graph uses three colours: white for unvisited, grey for on the current DFS path, black for finished. Meeting a grey neighbour means you have found a back edge, which is a cycle.

```python
from collections import defaultdict, deque

class Graph:
    def __init__(self):
        self.adj = defaultdict(set)

    def add_edge(self, u, v, directed=False):
        self.adj[u].add(v)
        if not directed:
            self.adj[v].add(u)

    def shortest_path(self, start, end) -> list:
        """BFS shortest path."""
        queue = deque([(start, [start])])
        visited = {start}
        while queue:
            node, path = queue.popleft()
            if node == end:
                return path
            for neighbor in self.adj[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        return []

    def connected_components(self) -> list[set]:
        """Find all connected components using DFS."""
        visited = set()
        components = []
        for node in self.adj:
            if node not in visited:
                component = set()
                stack = [node]
                while stack:
                    n = stack.pop()
                    if n not in visited:
                        visited.add(n)
                        component.add(n)
                        stack.extend(self.adj[n] - visited)
                components.append(component)
        return components

    def detect_cycle(self) -> bool:
        """Detect cycle in directed graph using DFS coloring."""
        WHITE, GRAY, BLACK = 0, 1, 2
        color = defaultdict(int)

        def dfs(node):
            color[node] = GRAY
            for neighbor in self.adj[node]:
                if color[neighbor] == GRAY:
                    return True
                if color[neighbor] == WHITE and dfs(neighbor):
                    return True
            color[node] = BLACK
            return False

        return any(dfs(n) for n in self.adj if color[n] == WHITE)
```

> **Warning:** Because `adj` is a `defaultdict`, merely reading `self.adj[v]` inserts `v` as a key. In a directed graph, a node that only ever appears as an edge target is not yet a key, so `detect_cycle` inserts it while iterating over `self.adj` and Python raises `RuntimeError: dictionary changed size during iteration`. Iterating over `list(self.adj)`, or adding both endpoints as keys in `add_edge`, fixes it. Copying `path + [neighbor]` at every BFS step also costs $O(\text{length})$ per enqueue; storing parent pointers and rebuilding the path at the end is the efficient version. The recursive DFS can also hit Python's recursion limit on deep graphs.

### Data transformation

The second family is ETL-style transformation of nested data. This transformer takes rules keyed by dot-separated paths, where `*` matches every element of a list, and applies each rule's function at the matching locations of a deep copy of the input, so the original is never mutated.

```python
from typing import Any

class DataTransformer:
    """Transform nested data structures with a rule-based pipeline."""

    def __init__(self):
        self.rules: list[tuple[str, callable]] = []

    def add_rule(self, path: str, transform_fn: callable):
        """Add a transformation rule for a dot-separated path."""
        self.rules.append((path, transform_fn))

    def apply(self, data: dict) -> dict:
        result = self._deep_copy(data)
        for path, fn in self.rules:
            self._apply_at_path(result, path.split('.'), fn)
        return result

    def _apply_at_path(self, data: Any, path_parts: list[str], fn: callable):
        if not path_parts:
            return

        key = path_parts[0]

        if key == '*' and isinstance(data, list):
            for item in data:
                if len(path_parts) == 1:
                    fn(item)
                else:
                    self._apply_at_path(item, path_parts[1:], fn)
        elif isinstance(data, dict) and key in data:
            if len(path_parts) == 1:
                data[key] = fn(data[key])
            else:
                self._apply_at_path(data[key], path_parts[1:], fn)

    def _deep_copy(self, obj):
        if isinstance(obj, dict):
            return {k: self._deep_copy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._deep_copy(item) for item in obj]
        return obj
```

Notice the asymmetry at the end of a path. For a dict key the function's return value replaces the value, but when the last segment is `*` the function is called on each list item and its return value is discarded, so it only has an effect if it mutates the item in place. A rule like `tags.*` with `str.upper` therefore does nothing. Assigning `data[i] = fn(item)` makes the two cases consistent, and an interviewer will likely ask whether you noticed.

### Timelines and intervals

The third family is interval problems. Here the task is to merge overlapping labelled events into non-overlapping segments, each tagged with the set of labels active during it. The sweep-line approach turns each event into an enter boundary and an exit boundary, sorts them by time, and walks them while keeping a count of active events per label. Whenever time advances and some label is active, the span since the previous boundary becomes a segment.

```python
def merge_timelines(events: list[tuple[int, int, str]]) -> list[tuple[int, int, set[str]]]:
    """Merge overlapping events into unified timeline segments.

    Input: [(start, end, label), ...]
    Output: [(start, end, {labels}), ...] where segments don't overlap
    """
    if not events:
        return []

    # Create boundary events
    boundaries = []
    for start, end, label in events:
        boundaries.append((start, 1, label))   # enter
        boundaries.append((end, -1, label))     # exit
    boundaries.sort(key=lambda x: (x[0], -x[1]))  # exits before enters at same time

    result = []
    active = {}  # label -> count
    prev_time = None

    for time, delta, label in boundaries:
        if prev_time is not None and prev_time < time and active:
            active_labels = {l for l, c in active.items() if c > 0}
            if active_labels:
                result.append((prev_time, time, active_labels))

        active[label] = active.get(label, 0) + delta
        if active[label] == 0:
            del active[label]
        prev_time = time

    return result
```

For example, `[(1, 5, "a"), (3, 8, "b")]` produces `[(1, 3, {"a"}), (3, 5, {"a", "b"}), (5, 8, {"b"})]`. Counting per label, rather than storing a set, handles two overlapping events with the same label. One detail to be precise about: the key `(x[0], -x[1])` actually sorts enters (`-1`) before exits (`1`) at the same timestamp, the opposite of what the comment says. It happens not to matter here, because no segment is emitted between boundaries at the same instant and the counts are correct once all of them are processed, but you should be able to say so. Sorting dominates, so the algorithm is $O(n \log n)$.

## The decomposition round

The decomposition round is Palantir's most distinctive interview. You are given a vague, real-world problem and asked to break it down into concrete technical components. The interviewer presents a high-level scenario, such as "help a hospital manage patient flow". You ask clarifying questions to understand the problem, identify the key entities, relationships, and workflows, sketch a technical solution (data model, services, UI components), and finish by discussing trade-offs and prioritisation. The best way to understand the round is to work one through.

### A worked decomposition: disaster response

Suppose the prompt is "Build a system to manage disaster response". Start by clarifying. What types of disasters: natural, industrial, others? Who are the users: emergency coordinators, field responders, logistics staff? What decisions must be made: resource allocation, evacuation routing, supply chain? And what data sources exist: satellite imagery, weather, population data, real-time reports?

With those answers, identify the entities. This is the ontology, and it maps directly to how Palantir's platforms model the world:

```
Entities:
- Incident (location, type, severity, status)
- Resource (type: personnel/vehicle/supply, location, status, capacity)
- Area (geography, population, infrastructure, risk level)
- Task (description, priority, assigned_to, status, dependencies)
- Report (source, timestamp, content, verified)
```

Next, identify the workflows: what people actually do with those entities, with the inputs, processing, and outputs of each.

```
1. Situation Assessment
   Input: Reports, sensor data, satellite imagery
   Process: Aggregate, verify, prioritize
   Output: Incident severity map

2. Resource Allocation
   Input: Available resources, incident priorities, geography
   Process: Optimization (minimize response time, maximize coverage)
   Output: Assignment plan

3. Communication
   Input: Decisions, updates
   Process: Route to relevant stakeholders
   Output: Notifications, orders

4. Tracking & Adjustment
   Input: Real-time field updates
   Process: Monitor progress, detect issues
   Output: Re-allocation recommendations
```

Only now sketch the technical architecture, layer by layer, from ingestion through fusion and analytics to the interfaces each user group needs:

```
[Data Ingestion Layer]
├── Satellite feed adapter
├── Weather API adapter
├── Field report API (mobile app)
└── Sensor data stream (IoT)
         |
[Data Fusion & Ontology]
├── Entity resolution (deduplicate reports)
├── Geospatial indexing
└── Temporal event correlation
         |
[Analytics / Decision Support]
├── Resource optimization solver
├── Route planning (with road damage awareness)
├── Demand forecasting
└── LLM-powered situation summarization
         |
[UI Layer]
├── Map-based situation dashboard
├── Resource management console
├── Mobile app for field responders
└── Notification system
```

Finish with the trade-offs. Real-time updates trade against accuracy, since faster updates may be less verified. A centralised design with a single coordinator trades against a distributed one with regional autonomy. And complexity trades against usability: more features versus ease of use under stress, which matters a great deal for people working in a disaster.

### Practising decomposition

Other scenarios reported include "Help a city manage its transportation system", "Build a system for tracking supply chain across multiple countries", "Design a platform for clinical trials management", "Help a military organization plan logistics", and "Build a fraud detection system for a bank".

Five habits make the difference. Ask lots of questions, because the problem is intentionally vague and clarifying it shows product thinking. Think in entities and relationships, which maps directly to Palantir's ontology model. Prioritise ruthlessly: you cannot build everything, so say what the MVP is. Consider the human: who uses this, what decisions they make, and what they need to see. And put the data model first, getting the ontology right before thinking about features.

## The system design round

Palantir's design questions are about data platforms, and three come up often.

### A data integration platform

This is literally what Palantir Foundry does:

```
[Source Systems] --> [Connectors] --> [Data Pipeline]
                                          |
                                    [Schema Mapping]
                                          |
                                    [Entity Resolution]
                                          |
                                    [Ontology Layer]
                                          |
                          +---------------+---------------+
                          |               |               |
                    [Search/Query]  [Analytics]    [Actions/Workflows]
```

Schema mapping maps heterogeneous sources onto the unified ontology. Entity resolution deduplicates entities across sources with fuzzy or ML-based matching. Lineage tracking records where every data point came from. Access control is row-level and column-level, based on each user's clearance. And versioning supports time-travel queries and an audit trail.

### An LLM-powered enterprise assistant (AIP)

```
[User Query] --> [Context Assembly]
                      |
              [Ontology-Grounded RAG]
                      |
              [LLM (Claude/GPT)]
                      |
              [Action Verification]
                      |
              [Tool Execution] or [Response]
```

What distinguishes this from a generic chatbot is grounding in the ontology: the LLM has access to the enterprise data ontology, and retrieval-augmented generation pulls relevant data objects, not just documents. An action framework lets the LLM propose actions, such as writing data or triggering a workflow, but a human approves them. The LLM's view is restricted to the user's own permissions, and every interaction is logged for compliance.

### Geospatial analytics

A geospatial system indexes with the H3 hexagonal grid or S2 geometry, runs time-series analysis on moving entities, visualises heatmaps, trajectories, and clusters on a map, and must scale to billions of location events per day.

## The behavioral and values round

Palantir builds for defence, intelligence, healthcare, and enterprise, and you should be comfortable with that. The values it looks for are mission focus, being impact-driven ("What's the most impactful thing you've worked on?"), intellectual rigour, caring about getting things right, and ownership, end to end from data model to deployment.

Common questions include "Why Palantir?", "Tell me about a time you had to make sense of a complex, messy dataset.", "Describe a project where you had to balance technical and non-technical stakeholders.", "What's the most complex system you've designed end-to-end?", and "How do you approach a problem you've never seen before?" Be genuine about the first one: if you are uncomfortable with defence work, Palantir may not be the right fit.

Palantir's work with government agencies is controversial, and you should expect to discuss it. Be prepared to articulate your personal stance honestly, show that you have thought about the ethical implications, discuss how technology can be used responsibly, and demonstrate that you can engage with hard questions maturely.

## AI and ML at Palantir

AIP integrates enterprise LLMs with ontology-grounded context, supports tool calling and action execution within enterprise workflows, fine-tunes on domain-specific data under privacy constraints, and performs RAG over structured enterprise data rather than just documents. On the ML infrastructure side, Foundry supports feature engineering on its data pipelines, model training and deployment within the platform, model monitoring with retraining triggers, and integration with external frameworks such as PyTorch and TensorFlow.

## Preparing

Practise decomposition by taking any real-world problem and breaking it into entities, workflows, and technical components in thirty minutes. Practise graph problems (BFS, DFS, shortest path, connected components, cycle detection), because Palantir loves them. Practise data modelling by designing schemas for real domains such as healthcare, supply chain, and finance. Read about Palantir's products until you understand Gotham, Foundry, and AIP at a conceptual level. Prepare impact stories, so that every behavioral answer highlights measurable impact. Think about data quality, since messy, incomplete, and conflicting data is the norm. And form an opinion on Palantir's mission: you will be asked, and a thoughtful perspective matters more than the specific stance.

## Summary

- Palantir hires FDEs and backend engineers, and both face a signature decomposition round.
- Graph problems dominate the coding round; know BFS, components, and three-colour cycle detection, and their Python pitfalls.
- A decomposition moves from clarifying questions to ontology, workflows, architecture, and trade-offs.
- System design centres on data integration: schema mapping, entity resolution, lineage, fine-grained access control, and ontology-grounded LLMs.
- Expect to discuss the ethics of Palantir's work honestly and maturely.

## Further reading

- [Palantir Careers](https://www.palantir.com/careers/)
- [Palantir Blog](https://blog.palantir.com/)
- [Palantir AIP Documentation](https://www.palantir.com/platforms/aip/)
- [Glassdoor: Palantir SWE Interview Questions](https://www.glassdoor.com/Interview/Palantir-Technologies-Software-Engineer-Interview-Questions-EI_IE236375.0,21_KO22,39.htm)
- [levels.fyi: Palantir Compensation Data](https://www.levels.fyi/companies/palantir-technologies/salaries/software-engineer)
- Community reports on r/cscareerquestions and Blind.
