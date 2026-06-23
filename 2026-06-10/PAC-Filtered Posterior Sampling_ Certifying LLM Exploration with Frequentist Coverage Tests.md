# PAC-Filtered Posterior Sampling: Certifying LLM Exploration with Frequentist Coverage Tests

## Motivation

The LLM-based PSRL method in 'Toward Efficient Exploration by Large Language Model Agents' achieves efficient exploration by using an LLM to sample from a posterior over environments, but it lacks theoretical regret guarantees because the LLM's posterior samples are unverified; any sample may be arbitrarily unfaithful to the true dynamics, causing overconfident exploration and unbounded regret. The root cause is the absence of a certification mechanism that can reject samples inconsistent with observed data, a gap that standard PSRL avoids by assuming exact posterior inference.

## Key Insight

A frequentist coverage test based on PAC-derived confidence bounds on next-state predictions provides a high-probability guarantee that a posterior sample is consistent with observed transitions, enabling a provable regret bound for LLM-based PSRL without requiring the LLM to represent the exact posterior.

## Method

**Method: PAC-Filtered Posterior Sampling (PFPS)**

(A) **What it is**: PFPS is a meta-algorithm that wraps any LLM-based posterior sampler (e.g., from 'Toward Efficient Exploration') and filters its samples using a PAC-derived confidence test. Input: an LLM that produces a sampled environment model θ~ ~ P(·|history). Output: either a filtered model θ that passes the test or, if all samples fail, a safe exploratory action. The method operates online during agent-environment interaction.

(B) **How it works** (pseudocode):
```python
def pfps_step(history H, LLM, delta=0.05, n_min=1, mixing_estimate=1.0):
    # H = sequence of (s_t, a_t, s_{t+1}, r_t) observed so far
    # Step 1: Sample candidate models
    theta_candidates = []
    while len(theta_candidates) < K:  # K = 10
        theta = LLM.sample_posterior(H)  # from the LLM
        theta_candidates.append(theta)
    
    # Step 2: Frequentist coverage test for each candidate
    filtered_thetas = []
    for theta in theta_candidates:
        # Partition H into training set H_train and test set H_test (last M=20 steps)
        H_train, H_test = split(H, ratio=0.8, sliding_window=M)
        # Compute martingale bound using Hoeffding-Azuma for dependent sequences
        # Assume transitions form a martingale difference sequence with bounded increments (range 1)
        # mixing_estimate = rough upper bound on the mixing time (here set to 1 for simplicity)
        bound = martingale_bound(len(H_test), delta, mixing_estimate)  # = sqrt((2 * variance * mixing_estimate * log(1/delta)) / n) 
        # where variance = 0.25 for 0-1 loss
        # Compute empirical 0-1 error: fraction of transitions where argmax_p(s'|s,a) != true s'
        err = empirical_error(theta, H_test, loss='zero_one')
        if err <= bound:
            filtered_thetas.append(theta)
    
    if len(filtered_thetas) == 0:
        # Fallback: perform random action
        action = sample_random_action()
    else:
        theta_chosen = random.choice(filtered_thetas)
        # Step 3: Act optimally under chosen model
        action = argmax_a Q_theta(s_current, a)  # using model-based planning with MCTS (horizon 10, 50 simulations)
    return action

# Hyperparameters: K=10, delta=0.05, M=20 (sliding window size), mixing_estimate=1.0
```

(C) **Why this design**: We chose a martingale bound over a standard PAC bound to account for temporal dependence in online RL transitions; this avoids the strict i.i.d. assumption that would be invalid. The bound uses a rough mixing-time estimate (default 1) to control dependence; if the process mixes faster, the bound is conservative but still correct. We chose to hold out a subset of recent transitions for testing (sliding window M=20) to detect non-stationarity and reduce overfitting to stale data; the cost is reduced sample size for the bound, but we compensate by using a martingale bound that still provides valid coverage under weak dependence. We chose random action as fallback instead of abstention because exploration is the goal; random actions still collect data, while abstention would require external supervision unavailable in standard RL. We set K=10 as a budget to limit LLM calls (each costing ~1s API time); increasing K improves filtering but linearly increases latency, so K=10 balances coverage and speed.

(D) **Why it measures what we claim**: The martingale bound `bound = martingale_bound(...)` measures the _maximum tolerated empirical error_ that the model can have while still being considered faithful to the true dynamics with high probability; this operationalizes _faithfulness_ because the bound is derived from the assumption that the model class (defined by the LLM's representation) has finite complexity and that transitions form a martingale difference sequence with bounded increments (range 1). If the assumption of bounded increments fails (e.g., unbounded rewards), the bound may be too small, causing the test to reject faithful models—making PFPS conservative, but still safe. The empirical error `err = empirical_error(theta, H_test)` measures _consistency_ of the sampled model with observed data under the 0-1 loss; this relies on the assumption that the LLM's posterior is concentrated on near-deterministic models (low-entropy predictions). If the LLM outputs high-entropy distributions, the max-prob next state may be a poor summary, and the test may reject many samples unnecessarily—then PFPS becomes conservative and falls back to random exploration, which degrades sample efficiency but still maintains safety (no overconfidence). Thus, the method directly measures and enforces the two key properties needed for a regret bound: consistency with data (err ≤ bound) and avoidance of overconfident samples (by filtering).

## Contribution

(1) A PAC-based filtering mechanism that certifies each LLM posterior sample by testing its predictive consistency against a held-out set of transitions from the environment, ensuring high-probability faithfulness. (2) A provable regret bound for LLM-based PSRL under the filtered sampling scheme, where the regret scales with the number of non-stationary changes in the environment rather than the LLM's representation error. (3) A design principle for combining frequentist confidence bounds with Bayesian posterior sampling, demonstrating how to obtain theoretical guarantees from an approximate, black-box posterior source.

## Experiment

### Evaluation Setup
| Role | Choice | Rationale |
|------|--------|-----------|
| Dataset | Natural language RL tasks (ALFWorld, MiniGrid) | Covers diverse NL-based decision-making |
| Primary metric | Sample efficiency (steps to 90% max reward) | Directly measures exploration quality |
| Baseline 1 | LLM-based PSRL (Toward Efficient Exploration) | Tests uncalibrated posterior sampling |
| Baseline 2 | PPO with LLM prior (Efficient RL with LLM Priors) | Tests static LLM knowledge without online model |
| Baseline 3 | Raw LLM interaction (Can foundation models...) | Tests no RL, just LLM innate exploration |
| Ablation | PFPS without PAC filter (replace with random action selection) | Isolates the effect of the filtering step |

### Why this setup validates the claim
This combination forms a falsifiable test of the central claim that PAC-filtering improves sample efficiency in LLM-based RL. The primary metric directly measures exploration efficiency, which PFPS aims to enhance. The baselines isolate each sub-claim: PSRL tests reliance on unfiltered LLM posterior, PPO+LLM tests reliance on static prior, and Raw LLM tests no RL mechanism. The ablation (no PAC filter) identifies the contribution of the filtering component itself. By comparing across these, we can attribute performance gains specifically to the PAC test, not merely the LLM posterior or RL framework. If PFPS outperforms PSRL but not the ablation, the filter is key. If it outperforms all, the method passes. The metric is sensitive to both quick learning and avoidance of overconfident errors.

### Expected outcome and causal chain
**vs. LLM-based PSRL** — On a non-stationary task where reward function changes midway (e.g., ALFWorld kitchen layout changes), PSRL's LLM posterior remains confident in the old model, causing it to repeat suboptimal actions and suffer high regret. Our method's martialingale-based test detects the mismatch on the sliding window and falls back to random exploration, which gathers data to correct the model. We expect PFPS to show a smaller regrowth of regret after the change (e.g., 30% less drop in reward) and faster recovery (fewer episodes to return to 90% max reward).

**vs. PPO with LLM prior** — On a task requiring novel dynamics not covered by the LLM prior (e.g., MiniGrid environment with new lava hazard), PPO+LLM trusts its initial Q-function and fails to explore effectively, plateauing at low reward. Our method samples candidate models from the LLM posterior, discards inconsistent ones via the martingale test, and plans optimistically; thus it discovers the hazard faster. We expect PFPS to achieve 2x sample efficiency (fewer steps to reach 90% max reward) and ultimately higher asymptotic reward (e.g., 20% higher) on such novel dynamics.

**vs. Raw LLM interaction** — On a task requiring long-term credit assignment (e.g., ALFWorld multi-step cooking), raw LLM takes random exploratory actions with no planning, leading to slow improvement and inconsistent performance. PFPS uses model-based rollouts (MCTS) from the filtered model to plan, so it efficiently makes progress toward the subgoal. We expect PFPS to have a smoother learning curve with lower variance (e.g., standard deviation of reward across episodes halved) and faster convergence (e.g., 3x fewer episodes to reach target recipe success rate).

### What would falsify this idea
If PFPS does not outperform LLM-based PSRL specifically on non-stationary tasks (i.e., the reward recovery gap is less than 10%), or if the ablation (no PAC filter) matches PFPS performance, then the PAC filter is not the source of improvement and the central claim is wrong.

## References

1. Toward Efficient Exploration by Large Language Model Agents
2. Efficient Reinforcement Learning with Large Language Model Priors
3. Can foundation models actively gather information in interactive environments to test hypotheses?
4. EXCALIBUR: Encouraging and Evaluating Embodied Exploration
5. Distilled Self-Critique of LLMs with Synthetic Data: a Bayesian Perspective
