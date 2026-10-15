# Palantir Software Engineer Interview Guide

Comprehensive preparation for Palantir Forward Deployed Engineer (FDE) and Backend Engineer roles, with focus on data infrastructure and AI systems.

## Interview Process Overview

Timeline: **3-5 weeks**, **4-5 rounds**

| Round | Format | Duration | Focus |
|-------|--------|----------|-------|
| Recruiter Screen | Phone | 30 min | Background, role fit |
| Phone Screen | Coding (HackerRank/Karat) | 60 min | DS&A |
| Onsite 1 | Coding | 45-60 min | Algorithms, data structures |
| Onsite 2 | Decomposition | 60 min | Break down ambiguous problems |
| Onsite 3 | System Design | 60 min | Data platform architecture |
| Onsite 4 | Behavioral / Values | 45 min | Mission, collaboration, impact |

Palantir has two main SWE tracks:
- **Forward Deployed Engineer (FDE)**: Client-facing, full-stack, adapts Palantir to customer needs
- **Backend / Infra Engineer**: Core platform development (Gotham, Foundry, AIP)

## Compensation (Reported Ranges)

- **New Grad**: ~$200-250K total comp
- **Senior SWE**: ~$350-500K total comp
- **Staff+**: ~$500-800K total comp
- RSUs with 4-year vesting; Palantir went public so equity is liquid

## Key Themes

1. **Decomposition is unique to Palantir** -- The "decomposition" round is their signature interview. It tests your ability to break an ambiguous, real-world problem into technical components.
2. **Data platform thinking** -- Palantir builds data integration and analysis platforms. Think about ontologies, data models, and workflows.
3. **Mission-driven** -- Palantir works with governments and large enterprises. They look for candidates who care about impact and can handle the ethical complexity.
4. **Full-stack generalists** -- Especially for FDE roles, they want engineers who can do frontend, backend, data engineering, and client communication.
5. **AI/ML platform** -- AIP (Artificial Intelligence Platform) integrates LLMs into enterprise workflows. This is their fastest-growing product.

## Coding Round

### What to Expect

- Standard DS&A, medium to hard difficulty
- Clean code matters -- variable naming, modularity, edge cases
- May be on HackerRank (phone) or whiteboard (onsite)

### Frequently Reported Problems

#### Graph Problems (Very Common)

Palantir loves graph problems because their products are fundamentally about entities and relationships.

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

#### Data Transformation / ETL

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

#### Interval / Timeline Problems

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

## Decomposition Round (Palantir-Specific)

### What It Is

You're given a vague, real-world problem and asked to break it down into concrete technical components. This is Palantir's most distinctive interview round.

### Format

1. Interviewer presents a high-level scenario (e.g., "Help a hospital manage patient flow")
2. You ask clarifying questions to understand the problem
3. You identify the key entities, relationships, and workflows
4. You sketch a technical solution: data model, services, UI components
5. You discuss trade-offs and prioritization

### Example: "Build a system to manage disaster response"

**Step 1: Clarify**
- What type of disasters? (Natural, industrial, etc.)
- Who are the users? (Emergency coordinators, field responders, logistics)
- What decisions need to be made? (Resource allocation, evacuation routing, supply chain)
- What data sources exist? (Satellite, weather, population data, real-time reports)

**Step 2: Identify entities (ontology)**
```
Entities:
- Incident (location, type, severity, status)
- Resource (type: personnel/vehicle/supply, location, status, capacity)
- Area (geography, population, infrastructure, risk level)
- Task (description, priority, assigned_to, status, dependencies)
- Report (source, timestamp, content, verified)
```

**Step 3: Identify workflows**
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

**Step 4: Technical architecture**
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

**Step 5: Trade-offs**
- Real-time vs. accuracy: Faster updates may be less verified
- Centralized vs. distributed: Single coordinator vs. regional autonomy
- Complexity vs. usability: More features vs. ease of use under stress

### Other Decomposition Scenarios

- "Help a city manage its transportation system"
- "Build a system for tracking supply chain across multiple countries"
- "Design a platform for clinical trials management"
- "Help a military organization plan logistics"
- "Build a fraud detection system for a bank"

### Tips for Decomposition

1. **Ask lots of questions** -- The problem is intentionally vague. Clarifying shows product thinking.
2. **Think in entities and relationships** -- This maps directly to Palantir's ontology model.
3. **Prioritize ruthlessly** -- You can't build everything. What's the MVP?
4. **Consider the human** -- Who uses this? What decisions do they make? What do they need to see?
5. **Data model first** -- Get the ontology right before thinking about features.

## System Design Round

### Common Topics

#### Design a Data Integration Platform

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

- **Schema mapping**: Map heterogeneous sources to unified ontology
- **Entity resolution**: Deduplicate entities across sources (fuzzy matching, ML-based)
- **Lineage tracking**: Know where every data point came from
- **Access control**: Row-level and column-level security based on user clearance
- **Versioning**: Time-travel queries, audit trail

#### Design an LLM-Powered Enterprise Assistant (AIP)

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

- **Ontology-grounded**: LLM has access to the enterprise data ontology
- **RAG (Retrieval Augmented Generation)**: Pull relevant data objects, not just documents
- **Action framework**: LLM can propose actions (write data, trigger workflow), but human approves
- **Access control**: LLM's view is restricted to the user's permissions
- **Audit trail**: Every LLM interaction logged for compliance

#### Design a Geospatial Analytics System

- **Indexing**: H3 hexagonal grid or S2 geometry for geospatial queries
- **Temporal**: Time-series analysis on moving entities
- **Visualization**: Heatmaps, trajectories, clustering on a map
- **Scale**: Billions of location events per day

## Behavioral / Values Round

### Palantir Values

- **Mission focus**: They build for defense, intelligence, healthcare, and enterprise. Be comfortable with this.
- **Impact-driven**: "What's the most impactful thing you've worked on?"
- **Intellectual rigor**: Deep thinkers who care about getting it right.
- **Ownership**: End-to-end, from data model to deployment.

### Common Questions

- "Why Palantir?" (Be genuine. If you're uncomfortable with defense work, this may not be the right fit.)
- "Tell me about a time you had to make sense of a complex, messy dataset."
- "Describe a project where you had to balance technical and non-technical stakeholders."
- "What's the most complex system you've designed end-to-end?"
- "How do you approach a problem you've never seen before?"

### Ethics / Controversial Topics

Palantir's work with government agencies is controversial. Be prepared to:
- Articulate your personal stance honestly
- Show you've thought about the ethical implications
- Discuss how technology can be used responsibly
- Demonstrate you can engage with hard questions maturely

## AI/ML at Palantir

### AIP (Artificial Intelligence Platform)
- Enterprise LLM integration with ontology-grounded context
- Tool calling and action execution within enterprise workflows
- Fine-tuning on domain-specific data with privacy constraints
- RAG with structured enterprise data (not just documents)

### ML Infrastructure
- Feature engineering on top of Foundry data pipelines
- Model training and deployment within the Foundry platform
- Model monitoring and retraining triggers
- Integration with external ML frameworks (PyTorch, TensorFlow)

## Preparation Tips

1. **Practice decomposition** -- Take any real-world problem and break it into entities, workflows, and technical components. Time yourself to 30 minutes.
2. **Graph problems** -- Palantir loves graphs. Practice BFS, DFS, shortest path, connected components, cycle detection.
3. **Data modeling** -- Practice designing schemas for real-world domains (healthcare, supply chain, finance).
4. **Read about Palantir's products** -- Understand Gotham, Foundry, and AIP at a conceptual level.
5. **Prepare impact stories** -- Every behavioral answer should highlight measurable impact.
6. **Think about data quality** -- Messy, incomplete, conflicting data is the norm. How do you handle it?
7. **Form an opinion on their mission** -- You'll be asked. Having a thoughtful perspective matters more than the specific stance.

## Sources

- [Palantir Careers](https://www.palantir.com/careers/)
- [Palantir Blog](https://blog.palantir.com/)
- [Palantir AIP Documentation](https://www.palantir.com/platforms/aip/)
- [Glassdoor - Palantir SWE Interview Questions](https://www.glassdoor.com/Interview/Palantir-Technologies-Software-Engineer-Interview-Questions-EI_IE236375.0,21_KO22,39.htm)
- [levels.fyi - Palantir Compensation Data](https://www.levels.fyi/companies/palantir-technologies/salaries/software-engineer)
- r/cscareerquestions, Blind (community reports)
