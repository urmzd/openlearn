# Round 5: Hiring Manager Interview

## Format

- **Duration**: 45 minutes
- **Setting**: Video call with an infrastructure team lead
- **Focus**: Past experience, technical judgment, cultural fit, leadership

## Structure

| Segment | Duration | Content |
|---------|----------|---------|
| Intro & rapport | 5 min | Background, why Anthropic |
| Project deep dive | 15-25 min | Present a past project end-to-end |
| Scenario questions | 10-15 min | "How would you approach..." |
| Your questions | 5 min | Ask about team, challenges, culture |

## Project Deep Dive

Prepare 1-2 projects to discuss in detail. The ideal project:

- **End-to-end ownership**: You drove it from design to production
- **Infrastructure-relevant**: Distributed systems, scaling, reliability, performance
- **Had real trade-offs**: You made decisions that weren't obvious
- **Had measurable impact**: Latency reduced by X%, cost saved $Y, uptime improved to Z%

### What They Want to Hear

**Problem framing**: Why did this project matter? What was the business/technical constraint?

**Technical depth**: Architecture decisions, why you chose approach A over B. Specific technologies and why. Data model, scaling considerations.

**Debugging process**: Walk through a hard bug you found. How did you narrow it down? What tools did you use? What was the root cause? How did you prevent recurrence?

**Scaling challenges**: What happened when load increased? How did you identify bottlenecks? What did you change? What would break next?

**Collaboration**: Who else was involved? How did you divide work? How did you handle disagreements?

### Example Talking Points

- "We needed sub-100ms p99 for the API. I profiled and found the bottleneck was serialization, not network. Switched from JSON to protobuf and added connection pooling, which brought p99 from 240ms to 60ms."
- "The system handled 10K RPS but we needed 100K. I redesigned the caching layer with a two-tier approach -- local L1 cache with short TTL and shared Redis L2. This reduced origin hits by 95%."
- "We had intermittent failures that only showed up under load. I added distributed tracing with correlation IDs, which revealed a connection pool exhaustion issue in a downstream service."

## Scenario Questions

### "Pick an Approach" Scenarios

These test your judgment. They're looking for you to **favor simplicity** and explain your reasoning.

**Example**: "You're designing a new service. Do you build it as a monolith or microservices?"

Good answer: "Start monolith. We don't yet know the right service boundaries, and a monolith is simpler to deploy, debug, and iterate on. Split into services when we have clear evidence of where the boundaries should be -- different scaling needs, different team ownership, or different deployment cadences."

**Example**: "Your team wants to adopt a new database. The current one works but is hitting limits. How do you approach the migration?"

Good answer: "First, quantify the limits -- are we hitting them in 3 months or 18 months? If there's time, I'd run a proof of concept with the new database on a non-critical workload. Dual-write to both during migration. Monitor both for correctness and performance before cutting over. Keep the rollback path open for at least 2 weeks."

### Debugging & Incident Response

**"Walk me through how you'd debug a production latency spike."**

1. Check dashboards: Is it all requests or a subset? When did it start?
2. Correlate with deployments: Was anything deployed in the last hour?
3. Check dependencies: Are downstream services healthy?
4. Look at resource metrics: CPU, memory, disk I/O, network
5. Sample traces: Find slow requests, identify where time is spent
6. Check for external factors: Traffic spike? Bad actor? DNS issues?

**"A service is returning 500 errors intermittently. How do you investigate?"**

1. Error rate: What percentage? Correlated with time/request type?
2. Logs: What's the actual error? Stack trace?
3. Is it one instance or all instances? (If one, likely host issue)
4. Recent changes: Deployments, config changes, dependency updates?
5. Resource exhaustion: Connection pools, file descriptors, memory?

## Behavioral Questions

### Cross-Functional Communication

- "Tell me about a time you explained a technical constraint to a non-technical stakeholder."
- "How do you handle when product wants a feature that's architecturally expensive?"
- Framework: State the situation, what you communicated, how you translated technical concepts, the outcome.

### Conflict Resolution

- "Tell me about a disagreement with a teammate about a technical approach."
- "How do you handle pushback on your designs?"
- Framework: Describe the disagreement (not the person), how you evaluated both approaches objectively, what you learned, the resolution.

### Pushing Back on Timelines

- "Your manager asks you to ship a feature in 2 weeks that you estimate takes 4. What do you do?"
- Good answer: Quantify the gap. Show what's possible in 2 weeks vs. 4 weeks. Propose a phased approach -- what's the MVP that delivers value in 2 weeks? What's deferred? Make the trade-offs visible.

## Cultural / Ethics Questions

Anthropic takes AI safety seriously. Expect questions about:

### AI Safety

- "Why does AI safety matter to you?"
- "What's your understanding of Constitutional AI?"
- Don't fake passion. Be honest about your level of knowledge. Show genuine curiosity and willingness to learn.

### Constitutional AI Basics

- Anthropic's approach to aligning AI: train the model with a set of principles (a "constitution") rather than relying purely on human feedback
- The model critiques and revises its own outputs based on these principles
- Reduces reliance on human labelers for safety, more scalable

### Security vs. Deadline Trade-offs

- "What would you do if you discovered a security vulnerability but fixing it would delay a launch?"
- Expected answer: Fix it. Security is non-negotiable at Anthropic. Communicate the delay clearly with context on the risk.

### Responsible Deployment

- "How would you handle a situation where a model is producing concerning outputs in production?"
- Think about: immediate mitigation (rate limiting, safety filter tightening), root cause investigation, transparent communication, post-mortem

## Questions to Ask Them

Good questions show you've done your homework:

- "What's the biggest infrastructure challenge the team is facing right now?"
- "How does the team balance shipping speed with reliability?"
- "How does the safety team interact with infrastructure? Is safety a consideration in system design reviews?"
- "What does on-call look like for the infra team?"
- "What's the team's approach to technical debt?"

Avoid generic questions ("What's the culture like?") in favor of specific, thoughtful ones that show genuine interest in the work.

## Preparation Tips

1. **Prepare your project story** -- Practice telling it in 15 minutes. Have a 5-minute version too.
2. **Quantify your impact** -- Numbers are memorable. "Reduced latency by 40%" beats "made it faster."
3. **Be honest about what you don't know** -- "I haven't worked with that specifically, but here's how I'd approach learning it" is a strong answer.
4. **Read Anthropic's research blog** -- Show awareness of their recent work beyond just Claude.
5. **Have genuine questions** -- This is also your interview of them.
