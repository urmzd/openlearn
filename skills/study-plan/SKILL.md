---
name: study-plan
description: "Generate a personalized study plan based on goals, current level, available time, and target languages. Draws from all curriculum content including algorithms, math, ML, systems, and polyglot practice tracks."
invoke: user
arguments:
  - name: args
    description: "<goal> — e.g., 'interview-prep 4 weeks golang', 'systems-mastery rust c', 'full-stack 12 weeks', or 'assess' to start with a diagnostic"
---

# Custom Study Plan Generator

Create personalized study plans that adapt to the user's goals, experience, and available time.

## Instructions

1. Parse the goal and constraints from: `{args}`
2. If the argument is "assess" or unclear, start by asking the user:
   - What are your goals? (interview prep, language mastery, CS fundamentals, career growth)
   - What's your current level? (beginner, intermediate, advanced in each area)
   - How many hours per week can you dedicate?
   - Which languages do you want to practice in?
   - Any specific companies you're targeting? (reference interviews/ guides)
   - Timeline? (weeks until deadline)
3. Read the relevant curriculum READMEs to understand available content:
   - `README.md` — overall structure
   - `STUDY-PLAN.md` — existing structured plans
   - `algorithms/README.md` — algorithm topics
   - `math/README.md` — math foundations
   - `practice/README.md` — polyglot practice track
   - `interviews/README.md` — company guides
4. Generate a week-by-week plan

## Plan Types

### interview-prep [weeks] [languages...]
Optimized for technical interviews:
- Week 1-2: Core algorithms (arrays, trees, graphs, DP) in primary language
- Week 3: System design + company-specific patterns
- Week 4+: Mock interviews, weak-area drilling, secondary language practice
- Daily: 1 algorithm problem + 1 language-specific exercise from practice/
- Include company-specific guidance from interviews/{company}/

### language-mastery [languages...]
Deep fluency in target languages:
- Phase 1: Language essentials (read practice/{lang}/README.md, do exercises 01-03)
- Phase 2: Intermediate patterns (exercises 04-06, language idiom focus)
- Phase 3: Advanced topics (exercises 07-10, concurrency, metaprogramming)
- Cross-language comparison: implement the same algorithm in all target languages
- Daily: 1 exercise + compare with previous language implementation

### systems-mastery [languages...]
Systems programming depth:
- Phase 1: Memory management fundamentals (C exercises, then Rust comparison)
- Phase 2: Concurrency (pthreads → channels → async, across C/Rust/Zig/Go)
- Phase 3: Performance (profiling, cache-awareness, SIMD, zero-copy)
- Phase 4: Systems projects (allocator, network server, file system)
- Pair with the C runtime chapters in ml/08-tinyllm/p09-kernels/ (rt.02 arena, rt.03 thread pool)

### full-stack [weeks]
Comprehensive curriculum:
- Follow the Combined Path from STUDY-PLAN.md
- Add polyglot practice exercises alongside each algorithm topic
- Integrate company interview prep in final weeks
- Adjust pace based on user's starting level

### fundamentals [weeks]
CS foundations for career changers or refreshers:
- Math foundations (prioritize discrete math + probability)
- Core algorithms (focus on patterns, not memorization)
- One systems language (C or Rust) + one general purpose (Python or TypeScript)
- Build up to practice exercises progressively

## Plan Format

Generate plans as a structured weekly schedule:

```
## Week N: {Theme}

### Goals
- {specific, measurable goals for the week}

### Daily Schedule ({X} hrs/day)

| Day | Morning ({Y} min) | Afternoon ({Z} min) | Evening ({W} min) |
|-----|-------------------|---------------------|-------------------|
| Mon | {activity} | {activity} | {activity} |
| ... | ... | ... | ... |

### Exercises
- [ ] {specific exercise with link to file/README}
- [ ] {language practice with link to practice/{lang}/}

### Milestones
- By end of week: {what they should be able to do}
- Self-check: {how to verify understanding}
```

## Adaptation Rules

- If user knows algorithms well → skip to language practice + systems
- If user is new to a language → start with exercises 01-03 before anything complex
- If interview timeline < 4 weeks → focus on high-frequency patterns only
- If user targets quant firms → emphasize math, probability, C++ from practice/
- If user targets AI labs → emphasize ML track + Python + systems
- If user targets cloud companies → emphasize Go, Java, system design
- Always pair theory with implementation: every concept gets coded
- Schedule spaced review: revisit topics from 3 days ago, 1 week ago, 2 weeks ago
- Include use of `study-session` skill for active recall checkpoints

## Cross-References

Reference specific content from the curriculum:
- Algorithm patterns → `algorithms/{topic}/README.md`
- Practice exercises → `practice/{tier}/{lang}/README.md`
- Interview prep → `interviews/{company}/README.md`
- Study routines → `STUDY-PLAN.md` (daily routine sections)
- Math prerequisites → `math/README.md`
- Competitive programming → `competitive-programming/README.md`
- Worked builds → `case-studies/{study-dir}/README.md` (each has a runnable implementation)
