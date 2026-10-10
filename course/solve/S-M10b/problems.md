# S-M10b: constrained optimization, duality, and DPO

Enter exact expressions where possible: write a vector or a pair as `[a, b]`, $e$ as `E`, and a multiplier $\lambda$ inside an answer as `lam`. For KKT questions, use the convention `g(x) <= 0` and multipliers `lambda >= 0`. Mark rubric proofs as attested only after writing the requested steps.

## Lagrange multipliers and KKT (10)

**q1.** Minimize `x^2 + y^2` subject to `x + y = 2`. Give the minimizer.
**q2.** For the same problem, give the equality multiplier using `L=f+lambda*(x+y-2)`.
**q3.** Minimize `x^2` subject to `x >= 1`. Give the minimizer.
**q4.** Write the nonnegative multiplier for constraint `1-x <= 0` at the minimizer.
**q5.** For problem 3, give the Lagrangian gradient stationarity residual at `x=1`, `lambda=2`.
**q6.** Is complementary slackness satisfied at `x=1`, `lambda=2`? Answer bool.
**q7.** Minimize `(x-3)^2` subject to `x <= 1`. Give the minimizer.
**q8.** Give the nonnegative multiplier for `x-1 <= 0` in problem 7.
**q9.** At a strict feasible point, what value must an inactive inequality multiplier take?
**q10.** State whether `x=0, lambda=0` is feasible for `x >= 1`, with `g=1-x <= 0`.

## Duality (4)

**q11.** Minimize `x^2` subject to `x >= 1`. Give the primal optimal value.
**q12.** For `L=x^2+lambda*(1-x)`, give the dual function `inf_x L`.
**q13.** Give the dual optimal value for problem 11.
**q14.** Is the duality gap zero? Answer bool.

## KL-regularized objective and DPO (4)

**q15.** For rewards `[1,0]`, reference probabilities `[1/2,1/2]`, and `beta=1`, give optimal probabilities proportional to `p_ref * exp(reward/beta)`.
**q16.** Give the log-odds of the optimal policy from problem 15.
**q17.** If chosen and rejected have equal policy/reference log odds, give the per-pair DPO loss.
**q18.** Derive in one or two sentences why subtracting the reference log ratio recovers the reward difference divided by beta at the optimum. (Rubric.)
