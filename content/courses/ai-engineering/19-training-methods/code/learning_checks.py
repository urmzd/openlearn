"""Dependency-free checks of the curriculum's worked examples.

Run from the repository root with:
    python3 content/courses/ai-engineering/19-training-methods/code/learning_checks.py

These are arithmetic and estimator checks, not a training or GPU benchmark.
"""

from math import exp, isclose, log, sqrt


def surrogate(ratio: float, advantage: float, clip: float = 0.2) -> float:
    """One action's maximized PPO surrogate, with no KL or entropy term."""
    clipped = min(max(ratio, 1.0 - clip), 1.0 + clip)
    return min(ratio * advantage, clipped * advantage)


def gae(rewards, values, bootstrap, trace, gamma=1.0, lam=0.5):
    """One collected fragment; masks separate bootstrap and trace boundaries."""
    n = len(rewards)
    if len(values) != n + 1 or len(bootstrap) != n or len(trace) != n:
        raise ValueError("values need n+1 entries; masks need n entries")
    result = [0.0] * n
    next_advantage = 0.0
    for t in reversed(range(n)):
        residual = rewards[t] + gamma * bootstrap[t] * values[t + 1] - values[t]
        next_advantage = residual + gamma * lam * trace[t] * next_advantage
        result[t] = next_advantage
    return result


def kl(p, q):
    """Exact finite-categorical KL; examples here have positive probabilities."""
    return sum(a * log(a / b) for a, b in zip(p, q, strict=True))


def run_checks():
    # Cross entropy and variable-token accumulation.
    assert isclose(-log(0.25), 1.3862943611198906)
    assert isclose(-log(0.5), 0.6931471805599453)
    assert (2 * 1 + 6 * 3) / 8 == 2.5
    assert (1 + 3) / 2 != 2.5

    # All four signs/directions of PPO clipping, including harmful moves.
    cases = [(1.5, 1, 1.2), (0.5, 1, 0.5), (1.5, -1, -1.5), (0.5, -1, -0.8)]
    for ratio, advantage, expected in cases:
        assert isclose(surrogate(ratio, advantage), expected)
    h = 1e-6
    favorable_slope = (surrogate(1.5 + h, 1) - surrogate(1.5 - h, 1)) / (2 * h)
    harmful_slope = (surrogate(1.5 + h, -1) - surrogate(1.5 - h, -1)) / (2 * h)
    assert favorable_slope == 0.0
    assert isclose(harmful_slope, -1.0, abs_tol=1e-8)
    assert isclose(-(surrogate(2, 1) + surrogate(0.5, -1)) / 2, -0.2)

    # GAE targets and the difference between terminal and artificial cutoff.
    advantage = gae([0, 1], [0.2, 0.4, 0], [1, 0], [1, 0])
    assert all(isclose(a, b) for a, b in zip(advantage, [0.5, 0.6], strict=True))
    assert all(
        isclose(a + v, target)
        for a, v, target in zip(advantage, [0.2, 0.4], [0.7, 1.0], strict=True)
    )
    assert gae([0], [0.2, 0.4], [0], [0]) == [-0.2]
    assert gae([0], [0.2, 0.4], [1], [0]) == [0.2]
    # A trace boundary prevents a following unrelated episode leaking backward.
    separated = gae([0, 100], [0.2, 0.4, 0], [1, 0], [0, 0])
    assert isclose(separated[0], 0.2)

    # Importance correction on a fully enumerable problem and ESS.
    behavior, target, rewards = [0.8, 0.2], [0.4, 0.6], [1, 3]
    ordinary = sum(m * r for m, r in zip(behavior, rewards, strict=True))
    corrected = sum(
        m * (p / m) * r for m, p, r in zip(behavior, target, rewards, strict=True)
    )
    assert isclose(ordinary, 1.4) and isclose(corrected, 2.2)
    weights = [1, 1, 1, 9]
    assert isclose(sum(weights) ** 2 / sum(w * w for w in weights), 12 / 7)

    # GRPO population-standard-deviation convention, including an equal group.
    r = [0, 1, 1]
    mean = sum(r) / len(r)
    std = sqrt(sum((x - mean) ** 2 for x in r) / len(r))
    adv = [(x - mean) / std for x in r]
    assert isclose(adv[0], -sqrt(2)) and isclose(adv[1], 1 / sqrt(2))
    assert isclose(sum(adv), 0, abs_tol=1e-12)
    assert [(x - 1) / 1e-8 for x in [1, 1, 1]] == [0, 0, 0]
    assert isclose(1 - 0.5**4 - (1 - 0.5) ** 4, 0.875)
    assert isclose(1 - 0.01**4 - (1 - 0.01) ** 4, 0.03940398)
    # k3 is pointwise nonnegative; signed log-ratio samples are not.
    for log_ratio in [-2, -0.1, 0, 0.1, 2]:
        assert exp(-log_ratio) + log_ratio - 1 >= -1e-15

    # Inclusive sample-mean baseline has a finite-group scale bias.
    # Enumerate two independent Bernoulli actions with p=0.25, reward=a,
    # score d log pi(a)/d p = a/p - (1-a)/(1-p).
    p = 0.25
    expected_inclusive_gradient = 0.0
    for a in [0, 1]:
        for b in [0, 1]:
            probability = (p if a else 1 - p) * (p if b else 1 - p)
            baseline = (a + b) / 2
            score_a = a / p - (1 - a) / (1 - p)
            score_b = b / p - (1 - b) / (1 - p)
            expected_inclusive_gradient += (
                probability * ((a - baseline) * score_a + (b - baseline) * score_b) / 2
            )
    assert isclose(expected_inclusive_gradient, 0.5)  # True d E[r]/d p is 1.

    # Exact KL direction and reward model examples.
    assert isclose(kl([0.8, 0.2], [0.5, 0.5]), 0.19274475702175753)
    assert isclose(kl([0.5, 0.5], [0.8, 0.2]), 0.22314355131420976)
    assert isclose(log(1 + exp(-2)), 0.1269280110429726)
    assert log(1 + exp(0)) == log(2)
    assert 100_000 * 512 * 32_000 * 2 == 3_276_800_000_000  # Cached logits

    # SFT target positions follow the causal shift and packed-document boundary.
    target_mask = [False, False, True, True]
    assert target_mask[1:] == [False, True, True]
    document_ids = [0, 0, 1, -1]
    eligible = [
        document_ids[j] >= 0 and document_ids[j] == document_ids[j - 1]
        for j in range(1, len(document_ids))
    ]
    assert eligible == [True, False, False]
    assert (1 * 1 + 3 * 3) / (1 + 3) == 2.5

    # Conservative label smoothing has a finite stationary *pair* margin.
    smoothing = 0.1
    optimum = log((1 - smoothing) / smoothing)
    sigmoid = 1 / (1 + exp(-optimum))
    assert isclose(optimum, log(9))
    assert isclose(sigmoid - (1 - smoothing), 0, abs_tol=1e-12)
    assert -4 > -6 and -4 / 2 < -6 / 6  # Sum and mean rank differently.
    reference, candidate = [0.1, 0.1, 0.8], [0.49, 0.49, 0.02]
    margin = log(candidate[0] / candidate[1]) - log(reference[0] / reference[1])
    assert margin == 0 and isclose(log(1 + exp(-margin)), log(2))
    assert 1.48 < kl(candidate, reference) < 1.49  # Zero margin is not zero KL.
    rewards = [0, 1, 1]
    baselines = [(sum(rewards) - r) / (len(rewards) - 1) for r in rewards]
    assert baselines == [1, 0.5, 0.5]

    # Standard-attention cache, roofline, DP batch, and pipeline bubble.
    kv_per_token = 2 * 32 * 8 * 128 * 2
    assert kv_per_token == 128 * 1024
    assert kv_per_token * 8192 == 1024**3
    assert 2 * 32 * 32 * 128 * 2 * 8192 == 4 * 1024**3
    assert 1e15 / 4e12 == 250
    assert 2 * 8 * 4 == 64  # microbatch * accumulation * independent replicas
    assert isclose((4 - 1) / (16 + 4 - 1), 3 / 19)
    assert 2 * 1024**3 / (50 * 10**9) > 0.04  # Illustrative KV transfer seconds

    # Numerical non-associativity and an argmax margin bound.
    a, b, c = 1e16, -1e16, 1.0
    assert (a + b) + c == 1 and a + (b + c) == 0
    original = [1.0000, 0.9999]
    perturbed = [original[0] - 1e-4, original[1] + 1e-4]
    assert original[0] > original[1] and perturbed[0] < perturbed[1]
    assert 0.01 > 2 * 0.001
    assert 480 / 60 == 8  # Compliant requests/second, not all submitted requests
    assert isclose((12 / 60) / 480, 1 / 2400)  # $12/hour, one minute, 480 passes
    assert (2.5 / 0.20) * 10**6 == 12_500_000  # Price per million, not per token
    print(
        "Learning checks passed: objectives, PPO/GAE, importance weights, GRPO, KL, capacity, and numerics."
    )


if __name__ == "__main__":
    run_checks()
