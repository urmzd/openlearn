# The Hiring Manager Interview

The last round is a 45-minute video call with an infrastructure team lead. It is the least technical round in format and one of the most consequential in practice, because it is where the team decides whether they want to work with you. The interviewer is assessing four things at once: your past experience, your technical judgment, your cultural fit, and your leadership.

This section follows the shape of the conversation. It starts with the structure of the 45 minutes, then spends most of its time on the project deep dive, the segment you can prepare for most directly. It then covers the scenario questions that test your judgment, the behavioral questions, the safety and ethics questions that are distinctive to Anthropic, and the questions you should ask in return.

> **Learning objectives**
> - Choose and structure a past project for a 15-to-25-minute deep dive, with quantified impact.
> - Answer "pick an approach" scenarios by favoring simplicity and making tradeoffs explicit.
> - Walk through a structured investigation of a production latency spike or intermittent 500 errors.
> - Answer cross-functional, conflict, and timeline questions with a clear situation-action-outcome frame.
> - Speak honestly about AI safety and Constitutional AI, and handle security-versus-deadline and responsible-deployment scenarios.

## The shape of the conversation

The 45 minutes usually break down as follows:

| Segment | Duration | Content |
|---------|----------|---------|
| Intro & rapport | 5 min | Background, why Anthropic |
| Project deep dive | 15-25 min | Present a past project end-to-end |
| Scenario questions | 10-15 min | "How would you approach..." |
| Your questions | 5 min | Ask about team, challenges, culture |

## The project deep dive

The deep dive is the heart of the interview, so prepare one or two projects to discuss in detail. The ideal project has four properties. You had **end-to-end ownership**, driving it from design to production. It is **infrastructure-relevant**, touching distributed systems, scaling, reliability, or performance. It involved **real tradeoffs**, decisions that were not obvious. And it had **measurable impact**: latency reduced by X%, cost saved by $Y, uptime improved to Z%.

As you tell the story, the interviewer is listening for five things. The first is **problem framing**: why did this project matter, and what business or technical constraint drove it? The second is **technical depth**: the architecture decisions and why you chose approach A over B, the specific technologies and why, the data model, and the scaling considerations. The third is your **debugging process**: walk through a hard bug you found, how you narrowed it down, which tools you used, what the root cause was, and how you prevented a recurrence. The fourth is **scaling challenges**: what happened when load increased, how you identified the bottlenecks, what you changed, and what would break next. The fifth is **collaboration**: who else was involved, how you divided the work, and how you handled disagreements.

Concrete talking points carry far more weight than generalities. Answers in this style work well:

- "We needed sub-100ms p99 for the API. I profiled and found the bottleneck was serialization, not network. Switched from JSON to protobuf and added connection pooling, which brought p99 from 240ms to 60ms."
- "The system handled 10K RPS but we needed 100K. I redesigned the caching layer with a two-tier approach: a local L1 cache with short TTL and a shared Redis L2. This reduced origin hits by 95%."
- "We had intermittent failures that only showed up under load. I added distributed tracing with correlation IDs, which revealed a connection pool exhaustion issue in a downstream service."

Practice telling your story in 15 minutes, and have a five-minute version ready too.

## Scenario questions

After the deep dive, the interviewer moves to hypotheticals. These come in two flavors.

### Picking an approach

"Pick an approach" scenarios test your judgment. The interviewers want to see you **favor simplicity** and explain your reasoning.

A typical question is: "You're designing a new service. Do you build it as a monolith or microservices?" A good answer is: "Start with a monolith. We don't yet know the right service boundaries, and a monolith is simpler to deploy, debug, and iterate on. Split into services when we have clear evidence of where the boundaries should be: different scaling needs, different team ownership, or different deployment cadences."

Another is: "Your team wants to adopt a new database. The current one works but is hitting limits. How do you approach the migration?" A good answer is: "First, quantify the limits: are we hitting them in 3 months or 18 months? If there's time, I'd run a proof of concept with the new database on a non-critical workload. Dual-write to both during the migration. Monitor both for correctness and performance before cutting over. Keep the rollback path open for at least two weeks."

Notice what both answers share: they defer irreversible decisions until there is evidence, and they keep a way back.

### Debugging and incident response

The second flavor asks you to investigate a hypothetical incident. What matters is a structured order of questions, from broad to narrow.

For "Walk me through how you'd debug a production latency spike," a good sequence is: check the dashboards to see whether it affects all requests or a subset, and when it started; correlate with deployments in the last hour; check whether downstream dependencies are healthy; look at resource metrics (CPU, memory, disk I/O, network); sample traces to find slow requests and see where their time is spent; and finally check for external factors such as a traffic spike, a bad actor, or DNS issues.

For "A service is returning 500 errors intermittently. How do you investigate?", start with the error rate: what percentage, and is it correlated with time or request type? Read the logs for the actual error and stack trace. Determine whether it is one instance or all of them; if it is one, a host issue is likely. Look at recent changes: deployments, configuration changes, dependency updates. And check for resource exhaustion: connection pools, file descriptors, memory.

## Behavioral questions

The behavioral questions fall into three themes, and the same frame serves all of them: the situation, what you did, and the outcome.

For **cross-functional communication**, expect "Tell me about a time you explained a technical constraint to a non-technical stakeholder" or "How do you handle it when product wants a feature that's architecturally expensive?" Describe the situation, what you communicated, how you translated the technical concepts, and the outcome.

For **conflict resolution**, expect "Tell me about a disagreement with a teammate about a technical approach" or "How do you handle pushback on your designs?" Describe the disagreement rather than the person, how you evaluated both approaches objectively, what you learned, and how it was resolved.

For **pushing back on timelines**, expect "Your manager asks you to ship a feature in two weeks that you estimate takes four. What do you do?" A good answer quantifies the gap, shows what is possible in two weeks versus four, and proposes a phased approach: what minimum version delivers value in two weeks, and what is deferred. The point is to make the tradeoffs visible rather than simply saying yes or no.

## Safety and ethics questions

Anthropic takes AI safety seriously, and this round is where that shows up most directly.

On **AI safety** itself, expect "Why does AI safety matter to you?" and "What's your understanding of Constitutional AI?" Do not fake passion. Be honest about your level of knowledge, and show genuine curiosity and willingness to learn.

You should know the basics of **Constitutional AI**. It is Anthropic's approach to aligning AI by training the model with a set of principles, a "constitution," rather than relying purely on human feedback. The model critiques and revises its own outputs according to those principles, which reduces reliance on human labelers for safety and scales better.

On **security versus deadlines**, you may be asked: "What would you do if you discovered a security vulnerability but fixing it would delay a launch?" The expected answer is to fix it. Security is non-negotiable at Anthropic. Communicate the delay clearly, with context on the risk.

On **responsible deployment**, you may be asked: "How would you handle a situation where a model is producing concerning outputs in production?" Think in terms of immediate mitigation (rate limiting, tightening safety filters), root-cause investigation, transparent communication, and a postmortem.

## Your questions for them

Good questions show you have done your homework, and this is also your interview of them. Specific questions work best:

- "What's the biggest infrastructure challenge the team is facing right now?"
- "How does the team balance shipping speed with reliability?"
- "How does the safety team interact with infrastructure? Is safety a consideration in system design reviews?"
- "What does on-call look like for the infra team?"
- "What's the team's approach to technical debt?"

Avoid generic questions such as "What's the culture like?" in favor of specific, thoughtful ones that show genuine interest in the work.

## Preparing for the round

Prepare your project story and practice it at both lengths. Quantify your impact, because numbers are memorable: "reduced latency by 40%" beats "made it faster." Be honest about what you do not know; "I haven't worked with that specifically, but here's how I'd approach learning it" is a strong answer. Read Anthropic's research blog so you are aware of their recent work beyond Claude. And arrive with genuine questions.

## Where this leads

The safety-versus-latency tradeoffs you may be asked to discuss are worked through technically in [the system design round](03-system-design.md). For a broader bank of questions about defending a system you have built, with the design records and postmortems that back each answer, see [Defend your system](../../06-defend-your-system/README.md).

## Key terms

- **Project deep dive**: a 15-to-25-minute end-to-end account of a past project, covering framing, depth, debugging, scaling, and collaboration.
- **Quantified impact**: a result stated as a measured change, such as p99 latency from 240 ms to 60 ms.
- **Dual write**: writing to both the old and new data stores during a migration so they can be compared before cutover.
- **Constitutional AI**: Anthropic's method of aligning a model by having it critique and revise its outputs against a written set of principles.

## Summary

- The 45 minutes are intro, a long project deep dive, scenarios, and your questions; prepare the deep dive most.
- Pick a project with end-to-end ownership, infrastructure relevance, real tradeoffs, and numbers.
- In scenarios, favor simplicity, defer irreversible decisions until there is evidence, and keep a rollback path.
- Investigate incidents from broad to narrow: scope, recent changes, dependencies, resources, traces, external factors.
- Be honest about AI safety, know the basics of Constitutional AI, and treat security as non-negotiable.

## Review questions

1. What four properties make a project a good choice for the deep dive?
   <details><summary>Answer</summary>End-to-end ownership from design to production, relevance to infrastructure (distributed systems, scaling, reliability, performance), real non-obvious tradeoffs, and measurable impact.</details>
2. Why is "start with a monolith" usually the stronger answer to the monolith-versus-microservices question?
   <details><summary>Answer</summary>Early on you do not know the right service boundaries, and a monolith is simpler to deploy, debug, and change. You split when evidence appears: different scaling needs, team ownership, or deployment cadences.</details>
3. One instance of a service is returning intermittent 500s and the others are healthy. What does that suggest, and what do you check?
   <details><summary>Answer</summary>A host-level problem is likely. Check that instance's logs and stack traces, its resource usage (memory, file descriptors, connection pools), and whether it received a different deployment or configuration.</details>
4. Your manager wants a four-week feature in two weeks. What does a good answer do?
   <details><summary>Answer</summary>It quantifies the gap, shows what can and cannot be delivered in two weeks, and proposes a phased plan with a minimum valuable version first and explicit deferrals, so the tradeoff is visible.</details>
5. You find a security vulnerability days before a launch. What do you do?
   <details><summary>Answer</summary>Fix it, and communicate the delay clearly with context on the risk. Security is not traded against a deadline.</details>
