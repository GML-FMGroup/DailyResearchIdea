# Potential-Intent Learning: Decentralized Multi-Robot Learning from Human Feedback as a Potential Game

## Motivation

Existing human-in-the-loop multi-robot learning methods, such as the thesis work on Multi-Robot Learning from Human Feedback, demonstrate empirical success but lack any convergence guarantee when the human's feedback target shifts as teammates co-adapt. The structural root cause is that the human's latent intent is a function of the team's joint policy, so each agent's reward signal is non-stationary and simultaneous decentralized updates can cycle or diverge. This meta-gap recurs across multiple paper-trees: R2BC and MAPF-GPT also assume a fixed expert or reward target, and none provides a shared sufficient statistic that makes decentralized updates monotone under co-adaptation.

## Key Insight

The human's latent intent posterior is a joint potential whose gradient each agent can estimate from a locally computable shared sufficient statistic—the human's feedback likelihood ratio conditioned on the agent's own action—so simultaneous decentralized updates are monotone in this potential and converge to a Nash equilibrium rather than cycling.

## Method

**What it is.** Potential-Intent Learning (PIL) is a decentralized multi-robot learning algorithm that treats the human supervisor's latent intent posterior as the potential of a potential game. Each robot receives only its own action and a scalar human feedback signal (approve/disapprove or preference), maintains a local estimate of the shared sufficient statistic, and updates its policy by projected gradient ascent on its local estimate of the potential. Output: a joint policy that converges to a Nash equilibrium of the potential game.

**Load-bearing assumption (stated explicitly).** PIL assumes that the scalar human approve/disapprove signal f_t, multiplied by robot i's own log-likelihood ratio, yields an unbiased estimate of the gradient of a single shared scalar potential with respect to robot i's policy — i.e., the human's feedback is a monotone, additively-separable function of the joint action that each robot can recover from its own action alone. This assumption is operationalized by the counterfactual baseline in the pseudocode below: s_i is computed as f_t times the difference between robot i's log-likelihood ratio and the expected log-likelihood ratio under robot i's reference action marginalized over teammates' empirical action distribution (a COMA-style counterfactual), so the statistic estimates robot i's marginal effect on the potential rather than the raw joint-feedback correlation. The smoothing window T is a variance-reduction heuristic, not an identification argument: averaging a confounded signal over T episodes does not remove bias when the confounding is systematic (correlated teammate actions), only when it is zero-mean noise. The convergence claim is therefore stated conditionally on this de-confounded estimator being unbiased.

**How it works.**

```pseudocode
Input: N robots, human feedback oracle H (approve/disapprove), learning rate η, smoothing β, window T
Initialize: each robot i has policy π_i(a_i | o_i; θ_i), reference policy π_i^ref (frozen copy of initial π_i), local statistic buffer B_i = ∅, teammate action buffer A_{-i} = ∅
for episode t = 1, 2, ...:
    each robot i observes o_i, samples a_i ~ π_i(·|o_i; θ_i)
    team executes joint action a = (a_1,...,a_N)
    human observes joint action a and returns feedback f_t ∈ {+1, -1}
    # Shared sufficient statistic: counterfactual feedback likelihood ratio for robot i's action
    for each robot i:
        # Counterfactual baseline: expected log-likelihood ratio under reference action, marginalized over teammates' empirical actions
        # A_{-i} contains the last T joint actions of teammates (shared via the same window T)
        cf_i = (1/|A_{-i}|) Σ_{a_{-i} ∈ A_{-i}} log[ π_i^ref(a_i|o_i) / π_i^ref(a_i|o_i) ]  # placeholder; see concrete formula below
        # Concrete counterfactual statistic:
        # s_i = f_t · [ log π_i(a_i|o_i;θ_i) − log π_i^ref(a_i|o_i) ] − (1/|A_{-i}|) Σ_{a_{-i} ∈ A_{-i}} f_t · [ log π_i(a_i'|o_i;θ_i) − log π_i^ref(a_i'|o_i) ]
        # where a_i' is robot i's reference action sampled from π_i^ref(·|o_i), and the second term is the expected feedback under the reference action marginalized over teammates' empirical actions in A_{-i}.
        s_i = f_t · [ log π_i(a_i|o_i;θ_i) − log π_i^ref(a_i|o_i) ]
              − (1/|A_{-i}|) Σ_{a_{-i} ∈ A_{-i}} f_t · [ log π_i(a_i'|o_i;θ_i) − log π_i^ref(a_i'|o_i) ]
        append s_i to B_i; if |B_i| > T: drop oldest
        append a_{-i} to A_{-i}; if |A_{-i}| > T: drop oldest
        # Local estimate of potential gradient via smoothed finite difference
        g_i = (1/|B_i|) Σ_{s ∈ B_i} s · ∇_{θ_i} log π_i(a_i|o_i; θ_i)
        # Projected gradient ascent on local potential estimate
        θ_i ← Proj_Θ( θ_i + η · g_i )
    # Optional: anneal η and β (β = 1/√t)
```

**Why this design.** We chose a potential-game formulation over a general-sum game because the human's latent intent posterior is a scalar function of the joint policy that each agent's feedback likelihood ratio is an unbiased noisy gradient of; this makes simultaneous decentralized updates monotone in the potential, accepting the cost that the human's intent must be expressible as a single scalar potential (we handle multi-intent via a mixture over intents, each with its own potential). We chose a local sufficient statistic (feedback likelihood ratio conditioned on the agent's own action) over a centralized critic because it allows each robot to estimate the potential gradient from its own action and feedback only, accepting the cost that the statistic is noisy when the human's feedback depends on other robots' actions (we correct this via a smoothing window T and a reference policy π_i^ref). We chose projected gradient ascent with annealing over natural gradient or mirror descent because it has a known monotonicity property for potential games under unbiased gradient estimates, accepting the cost that convergence rate is slower than natural gradient (we compensate with a larger T). A domain expert would NOT describe PIL as a variant of MAPF-GPT or R2BC because those methods train on fixed expert demonstrations or a fixed reward, whereas PIL's training signal is the human feedback likelihood ratio that shifts with the team's joint policy, and the potential-game structure is what makes this non-stationarity convergent rather than cyclic.

**Why it measures what we claim.** The local statistic s_i = f_t · [log π_i(a_i|o_i;θ_i) − log π_i^ref(a_i|o_i)] minus the counterfactual baseline measures the human's latent intent posterior gradient with respect to robot i's policy because the human feedback f_t is an unbiased noisy estimate of the sign of the potential's gradient under the assumption that the human's approval probability is monotone in the potential and that the counterfactual baseline removes the confounding from teammates' actions; this assumption fails when the human's feedback is biased by other robots' actions (e.g., the human approves a bad action because a teammate's good action dominates), in which case s_i reflects a confounded signal rather than robot i's contribution. The smoothing window T measures the stationarity of the human's intent because averaging over T episodes reduces the variance of the non-stationary feedback under the assumption that the human's intent changes slowly relative to T; this assumption fails when the human's intent shifts abruptly (e.g., a new task), in which case the average reflects a mixture of intents rather than the current one. The projected gradient ascent step measures monotone improvement in the potential because the update direction is an unbiased estimate of the potential gradient under the assumption that the local statistic is an unbiased estimate of the shared sufficient statistic; this assumption fails when robots' actions are correlated (e.g., they coordinate), in which case the local estimate is biased and convergence may be to a correlated equilibrium rather than a Nash equilibrium.

## Contribution

["A decentralized multi-robot learning algorithm, Potential-Intent Learning (PIL), that formulates human-in-the-loop learning as a potential game whose joint potential is the human's latent intent posterior and derives per-agent update rules using a locally computable shared sufficient statistic (the feedback likelihood ratio).", "A design principle: the human feedback signal can serve as a shared sufficient statistic for the potential gradient if and only if it is conditioned on each agent's own action via a likelihood ratio, which makes simultaneous decentralized updates monotone in the potential under co-adaptation.", 'An empirical demonstration on simulated multi-robot tasks (e.g., cooperative navigation and predator-prey) that PIL converges to a stable joint policy while baseline decentralized human-feedback methods cycle or diverge, and that the local statistic is a low-variance estimate of the potential gradient.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Cooperative multi-robot tasks with scalar human approve/disapprove feedback: (1) 2-robot alternating-role task, (2) 3-robot box-pushing with one salient robot, (3) 4-robot formation with abrupt intent shift, (4) synthetic known-potential task (2 robots, 3 actions each, ground-truth potential defined as a 2D quadratic with known Nash) | Matches the feedback channel PIL actually consumes; synthetic task validates estimator bias directly |
| Primary metric | Team return relative to centralized potential-game Nash equilibrium (computed by exact solver for synthetic task; by centralized projected gradient ascent with full joint action for the other tasks, run to convergence with 10 random restarts, taking the best) | Directly measures convergence-to-equilibrium claim |
| Baseline | Independent policy gradient on raw feedback (no counterfactual baseline, no window) | Tests whether potential structure is necessary |
| Baseline | Centralized critic with shared human feedback (COMA-style centralized critic trained on joint action and scalar feedback) | Tests whether decentralization costs anything |
| Baseline | MAPF-GPT-style imitation from fixed demonstrations (behavior cloning on 10k expert episodes generated by the centralized Nash policy) | Tests non-stationary-feedback advantage |
| Ablation-of-ours | PIL without smoothing window (T=1) | Isolates the variance-reduction mechanism |
| Ablation-of-ours | PIL without counterfactual baseline (raw s_i = f_t · log ratio) | Isolates the de-confounding mechanism |

Episode budget: 50k episodes per task for PIL and baselines; 10k for synthetic task. Number of robots: 2, 3, 4 as listed. Compute: each run uses 1 GPU (NVIDIA V100) for policy networks (2-layer MLP, hidden=256, GeLU) and 1 CPU core for environment simulation; total estimated 120 GPU-hours for all tasks and seeds (5 seeds per configuration).

### Why this setup validates the claim

The central claim is that treating human intent as a potential and letting each robot ascend a local estimate of its gradient yields monotone convergence to a Nash equilibrium of that potential. The primary metric — team return normalized against a computed centralized Nash equilibrium return on the same task — is the right detector because the claim is about equilibrium convergence, not raw reward; a method that merely improves reward without approaching the potential's equilibrium would not validate the mechanism. The synthetic known-potential task directly measures estimator bias: we compute the ground-truth potential gradient at the current joint policy and compare it to the per-agent statistic s_i, reporting mean squared error; this tests the load-bearing assumption that s_i is an unbiased estimate of the potential gradient. The independent-policy-gradient baseline tests the sub-claim that the potential-game structure is what prevents cycling: if PIL only matches it, the structure is doing no work. The centralized-critic baseline tests the sub-claim that the local sufficient statistic is sufficient, isolating whether decentralization introduces a measurable gap. The imitation baseline tests the sub-claim that a feedback-likelihood-ratio signal that shifts with the joint policy beats a fixed demonstration signal under non-stationary intent. The T=1 ablation is the sharpest test of the variance-reduction role: removing the smoothing window should increase variance and degrade performance specifically where feedback is noisy, not uniformly. The no-counterfactual ablation tests the de-confounding mechanism: removing the counterfactual baseline should degrade performance specifically where teammate actions are correlated, confirming the confounding account.

### Expected outcome and causal chain

**vs. independent policy gradient on raw feedback** — On a case where two robots must alternate roles and the human's approval depends on the joint action, the baseline treats each robot's feedback as its own reward, so correlated teammate actions inject noise that makes updates oscillate and the team cycles between role assignments. Our method instead averages the counterfactual feedback likelihood ratio over a window and ascends a shared potential, so updates are monotone in the potential. We expect the baseline to show high variance in team return across episodes and persistent role-swapping, while PIL shows a tighter, upward-trending return curve approaching the Nash reference.

**vs. centralized critic with shared feedback** — On a case where the human's approval is dominated by one robot's salient action, the centralized critic still attributes credit correctly because it sees the joint action, but its centralized training cannot be executed by decentralized robots and its advantage estimate is biased when the feedback is a scalar sign rather than a reward. Our method instead derives each robot's gradient from its own action-conditioned likelihood ratio with a counterfactual baseline, so it stays decentralized and unbiased under the monotone-approval assumption. We expect near-parity between the two on well-conditioned tasks and a visible gap favoring PIL only where the centralized critic's scalar-sign bias compounds, confirming decentralization is not the bottleneck.

**vs. MAPF-GPT-style imitation from fixed demonstrations** — On a case where the human's intent shifts mid-training (a new task priority), the imitation baseline is anchored to its fixed demonstration distribution and cannot adapt, so its return plateaus at the old intent's optimum and degrades on the new one. Our method instead keeps consuming feedback whose likelihood ratio shifts with the current joint policy, so the potential tracks the new intent after roughly one smoothing window. We expect the imitation baseline to be competitive before the intent shift and to fall behind sharply after it, while PIL recovers within a window — the predicted signature of non-stationary-signal convergence.

### What would falsify this idea

If the T=1 ablation matches full PIL on the noisy-feedback subset (rather than degrading specifically there), the smoothing-window variance-reduction is doing nothing and the variance account is wrong. Likewise, if the no-counterfactual ablation matches full PIL on the correlated-teammate subset (rather than degrading specifically there), the counterfactual baseline is not removing confounding and the de-confounding account is wrong. Finally, if the synthetic known-potential task shows that s_i has high bias (mean squared error not decreasing with T), the load-bearing assumption that s_i is an unbiased estimate of the potential gradient is false.

## References

1. Multi-Robot Learning from Human Feedback
2. R2BC: Multi-Agent Imitation Learning from Single-Agent Demonstrations
3. MAPF-GPT: Imitation Learning for Multi-Agent Pathfinding at Scale
4. Transformer-Based Imitative Reinforcement Learning for Multirobot Path Planning
