---
name: study-session
description: "Interactive study session that tests understanding through questions, code challenges, and concept explanations. Uses curriculum content from this repo for spaced repetition and active recall. Not memorization — comprehension, application, and synthesis."
invoke: user
arguments:
  - name: args
    description: "<mode> [topic] [language] — modes: quiz, drill, explain, review, assess. e.g., 'quiz graphs rust' or 'review dp' or 'assess'"
---

# Study Session

Run an interactive learning session that tests real understanding, not memorization.

## Modes

### quiz [topic] [language]
Ask 5-10 questions that escalate in difficulty:
1. **Recall** — "What is the time complexity of X?"
2. **Comprehension** — "Why does this approach fail for Y?"
3. **Application** — "Given this problem, which pattern applies and why?"
4. **Analysis** — "Compare these two approaches. When would you pick each?"
5. **Synthesis** — "Design a solution combining X and Y patterns"
6. **Code** — "Implement this in {language} using idiomatic patterns"

### drill [topic] [language]
Rapid-fire coding exercises:
1. Read the topic README from `content/courses/algorithms/` or `practice/` for context
2. Present a small, focused coding task (5-15 min)
3. The user writes code, you evaluate for:
   - Correctness (does it work?)
   - Complexity (optimal time/space?)
   - Idiom (is it natural in the target language?)
   - Edge cases (did they handle them?)
4. Give specific feedback, not just "correct/incorrect"
5. Present the next challenge, adjusting difficulty based on performance

### explain [topic]
Socratic teaching mode:
1. Ask the user to explain the concept as if teaching it
2. Ask probing follow-up questions to find gaps
3. When a gap is found, don't just give the answer — ask a question that leads them to it
4. Build mental models by connecting to concepts they already know
5. Reference specific implementations in the repo for concrete examples

### review [topic]
Spaced review of previously studied material:
1. Read relevant algorithm/math READMEs to gather key concepts
2. Present concepts the user should be able to recall
3. Ask them to explain, derive, or implement from memory
4. If they struggle, give graduated hints, then walk through the chapter's solution and ask them to explain each step back
5. Track which concepts are solid vs need more work
6. Suggest what to revisit next

### assess
Full diagnostic assessment:
1. Sample questions across all curriculum tracks the user has studied
2. Cover multiple Bloom's taxonomy levels per topic
3. Include at least one coding challenge
4. At the end, provide:
   - Strengths and weaknesses by topic
   - Recommended study priorities
   - Specific exercises from the practice track to reinforce weak areas
   - A suggested 1-week focused study plan

## Instructions

1. Parse the mode, topic, and optional language from: `{args}`
2. If a topic is given, read the relevant README(s):
   - Algorithm topics: `content/courses/algorithms/{topic-dir}/README.md`
   - Math topics: `content/courses/mathematics-for-computing/{topic-dir}/README.md`
   - Practice languages: `practice/build/{tier}/{lang}/README.md`
   - Competitive programming: `content/courses/competitive-programming/README.md`
   - Information theory: `content/courses/mathematics-for-computing/12-information-theory/README.md`
   - Case studies: `content/courses/case-studies/{study-dir}/README.md`
3. If a language is given, frame code questions in that language and evaluate idiom
4. For language-specific questions, also read `practice/build/{tier}/{lang}/README.md` for idiom reference
5. Use existing problem implementations in the repo as reference (but don't show them to the user during quizzes)
6. After each question, wait for the user's response before proceeding
7. Adjust difficulty dynamically — if they ace easy questions, skip to harder ones
8. At session end, summarize performance and suggest next steps

## Question Generation Guidelines

- **Don't ask trivia** — "What year was Dijkstra's algorithm published?" is useless
- **Do ask transfer questions** — "You know BFS finds shortest path in unweighted graphs. How would you modify it for weighted graphs with only weights 1 and 2?"
- **Test the edges** — "What happens to your merge sort when the input is already sorted? What about reverse sorted?"
- **Require justification** — "Which is better here, DFS or BFS? Explain your reasoning."
- **Cross-reference topics** — "How is the greedy choice property related to dynamic programming's optimal substructure?"
- **Language-specific depth** — For Rust: "Why can't you have two mutable references? What bug does this prevent?" For C: "What happens if you free this pointer twice?"

## Evaluation Criteria

When evaluating user responses:
- **Correct but vague** → Ask for specifics: "Can you give a concrete example?"
- **Correct and precise** → Acknowledge and escalate difficulty
- **Partially correct** → Identify what's right, probe the gap: "You're right about X, but what about Y?"
- **Incorrect** → Don't just say "wrong". Ask a simpler version of the same question to find where understanding breaks down
- **Code review** → Check correctness first, then complexity, then style. Language idiom matters.

## Topic Mapping

Map shorthand topic names to directories:
- arrays, hashing → content/courses/algorithms/01-arrays-hashing/
- two-pointers, sliding-window → content/courses/algorithms/02-two-pointers-sliding-window/
- binary-search → content/courses/algorithms/03-binary-search/
- linked-lists → content/courses/algorithms/04-linked-lists/
- trees → content/courses/algorithms/05-trees/
- graphs → content/courses/algorithms/06-graphs/
- dp, dynamic-programming → content/courses/algorithms/07-dynamic-programming/
- greedy → content/courses/algorithms/08-greedy/
- backtracking → content/courses/algorithms/09-backtracking/
- math, bits → content/courses/algorithms/10-math-bit/
- recursion, divide-conquer → content/courses/algorithms/11-recursion-divide-conquer/
- probabilistic → content/courses/algorithms/12-probabilistic-structures/
- calculus-1 through probability → content/courses/mathematics-for-computing/{topic}/
- cp, competitive → content/courses/competitive-programming/
- info-theory → content/courses/mathematics-for-computing/12-information-theory/
- order-book, matching → content/courses/case-studies/01-order-book-matching/
- sql-agent, text-to-sql → content/courses/case-studies/02-grounded-sql-agent/
- events, idempotency, exactly-once → content/courses/case-studies/03-exactly-once-event-api/
- kmeans, clustering-optimization → content/courses/case-studies/04-kmeans-optimization/
