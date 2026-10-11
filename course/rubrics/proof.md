# Proof rubric

Every `type = "proof"` question in a solve key is self-graded against its own
`rubric` lines; a question that names no lines uses this list. `ol check`
prints each line and you answer y or n. Every line must be a yes for the
proof to count, and the verdict is tagged `self` (DESIGN 5.5).

- [ ] The claim is stated exactly, with every symbol defined before it is used.
- [ ] The proof names its method (direct, contradiction, contrapositive, induction, cases).
- [ ] Every step follows from the previous steps, a definition, or a named earlier result.
- [ ] Every case the argument splits into is handled, including the boundary cases.
- [ ] An induction states the hypothesis for n = k and uses it, not the conclusion.
- [ ] The last line says what was proved, and it is the original claim.
