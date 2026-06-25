# Validating and Correcting Proxy Rewards for Reinforcement Learning-Based Data Scheduling

## Motivation

Existing RL-based data scheduling methods such as HDS assume that proxy rewards (e.g., loss-weighted quality, weight norms) are sufficient for optimizing downstream task performance without verification. This unvalidated assumption leads to misalignment when the proxy reward fails to capture confounders like data diversity or model state, causing the RL policy to optimize an inaccurate signal. The core structural problem is that no prior work explicitly tests whether a given proxy reward is a sufficient statistic for downstream performance given observable confounders, as seen in DoReMi where domain weights from a small proxy model are assumed to transfer to a larger model without checking.

## Key Insight

The conditional sufficiency of a proxy reward for downstream performance can be tested via gradient correlation, and when sufficiency fails, a lightweight correction network can learn a residual that restores alignment without requiring full retraining.

## Method

**Proxy Reward Validation and Correction (PRVC)**

(A) **What it is** — PRVC is a two-stage framework that first tests whether the current proxy reward is a sufficient statistic for downstream performance given the state, and if not, learns a residual correction to the reward before using it in the RL update. Input: RL policy π, proxy reward function R_proxy(s,a), downstream evaluation metric J_downstream (e.g., validation perplexity). Output: corrected reward r_corrected used for policy gradient updates.

(B) **How it works** — Pseudocode:
```python
# Hyperparameters: τ=0.95 (sufficiency threshold), C_θ = 2-layer MLP (256 hidden, ReLU), learning rate for C_θ: 1e-4

# Stage 1: Sufficiency Test (performed every N=5000 steps)
def sufficiency_test(π, R_proxy, J_downstream):
    collect batch of trajectories {τ_i} using π
    for each transition (s,a) in trajectories:
        r_proxy = R_proxy(s,a)
        g_down = ∇_π J_downstream(s,a)  # via REINFORCE with baseline
        g_proxy = ∇_π (log π(a|s) * r_proxy)
    ρ = correlation(g_proxy, g_down) across all transitions
    return ρ > τ

if sufficiency_test(π, R_proxy, J_downstream):
    r_corrected = r_proxy
else:
    # Stage 2: Correction
    initialize correction network C_θ (weights θ)
    for k in range(100):  # mini-optimization loop
        sample (s,a) from recent trajectory buffer
        δ = C_θ(s,a)
        r_corrected = r_proxy + δ
        # Assume a small set of downstream rewards r_downstream is available (e.g., from a held-out validation set)
        loss = MSE(r_corrected, r_downstream)
        θ ← θ - lr * ∇_θ loss
    # Use r_corrected = r_proxy + C_θ(s,a) for subsequent RL updates
```

(C) **Why this design** — We chose a gradient-based test over direct reward comparison (e.g., correlation of reward values) because gradients capture directional alignment in the optimization landscape, reducing false positives from coincidentally correlated reward magnitudes. The threshold τ=0.95 is conservatively high to minimize reliance on misaligned proxies, accepting a small false rejection rate that triggers unnecessary corrections. The correction network is kept lightweight (2-layer MLP) to avoid overfitting to the limited downstream reward data, trading expressive power for sample efficiency; a deeper network could model complex residuals but would require more data and risk instability. We opted to optimize the correction using MSE on a held-out downstream reward set rather than using the RL policy's reward itself because the downstream metric is the ground truth, but this introduces a dependency on having such a held-out set. Unlike DoReMi, which assumes the proxy model's domain weights transfer to a larger model without testing, our test is dynamic and does not require pre-training a separate proxy model; this avoids the assumption of transferability but adds per-iteration computation for gradient estimation.

(D) **Why it measures what we claim** — The gradient correlation ρ between ∇_π r_proxy and ∇_π J_downstream measures *conditional sufficiency* because if R_proxy is a sufficient statistic for J_downstream given state s, then the policy gradients should be proportional under the Markovian assumption that the state captures all confounders; this assumption holds when the reward function aggregates all information relevant to downstream performance. This assumption fails when the proxy reward ignores confounders that affect J_downstream independently (e.g., data diversity influences both proxy and J_downstream but through different pathways), in which case ρ drops below threshold and the test reliably detects misalignment. The correction network C_θ then produces a residual δ that operationalizes the *missing information*: optimizing MSE between r_corrected and r_downstream ensures that r_corrected approximates the true downstream reward, making it a better proxy for the RL update. The key assumption is that the held-out downstream reward r_downstream is an unbiased estimator of the true objective; this assumption fails if the held-out set is not representative of the test distribution, in which case the correction may overfit to the held-out set and harm generalization.

## Contribution

(1) A two-stage framework (PRVC) that tests the conditional sufficiency of a proxy reward for downstream performance using gradient correlation and corrects it via a lightweight residual network. (2) A practical algorithm that ensures the RL policy optimizes a signal aligned with the downstream metric, reducing the risk of misoptimization in data scheduling. (3) Empirical demonstration (in proposed experiments) that the method improves downstream perplexity and few-shot accuracy over baseline RL schedulers without reward correction.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | Pile (diverse domains) | Covers varied data sources, common in ODM research |
| Primary metric | Validation perplexity (averaged) | Direct measure of language modeling quality |
| Baseline 1 | Random mixing | Uniform sampling, no adaptation |
| Baseline 2 | DoReMi | Proxy model-based domain weighting |
| Baseline 3 | Aioli | Unified optimization with salience |
| Ablation-of-ours | PRVC without correction | Isolates effect of correction mechanism |

### Why this setup validates the claim

This experimental design tests the central claim that PRVC improves data scheduling by correcting misaligned proxy rewards. The Pile dataset provides diverse domains where proxy misspecification is likely. Random mixing serves as a naive baseline showing the need for adaptation. DoReMi tests whether static domain weights suffice, while Aioli tests dynamic scheduling without sufficiency checks. The ablation isolates the correction component. Validation perplexity is chosen because it is the direct objective for language model pre-training, making it the appropriate metric to detect improvements from reward correction. If PRVC outperforms baselines, it confirms that dynamic sufficiency testing and correction add value beyond existing methods.

### Expected outcome and causal chain

**vs. Random mixing** — On a case where one domain becomes more informative over time, random mixing wastes capacity on stale domains because it ignores distribution shifts. Our method detects when the proxy reward (e.g., domain loss) is insufficient for downstream performance and corrects it, so we expect a consistent perplexity gap in favor of PRVC, especially on domains with non-stationary importance.

**vs. DoReMi** — On a case where proxy model weights misalign with the training policy's learning dynamics (e.g., when a domain becomes redundant), DoReMi's fixed weights cause overemphasis because they lack dynamic adaptation. Our method tests gradient correlation and corrects the reward when misalignment is detected, so we expect PRVC to achieve lower perplexity on domains where proxy weight and true importance diverge, with parity on well-aligned domains.

**vs. Aioli** — On a case where the proxy reward's gradient direction conflicts with the true objective (e.g., when reducing proxy loss harms downstream perplexity), Aioli's unified optimization may push the policy in harmful directions because it trusts the proxy implicitly. Our correction network adjusts the reward to align with downstream signal, so we expect PRVC to avoid performance regressions on such conflict cases, yielding a noticeable advantage on subsets where proxy and true objectives diverge.

### What would falsify this idea

If PRVC shows no improvement over the ablation (no correction) on any domain, or if gains are uniform across domains rather than concentrated where proxy misalignment is predicted, then the central claim of benefit from sufficiency testing and correction is false.

## References

1. Holistic Data Scheduler for LLM Pre-training via Multi-Objective Reinforcement Learning
2. Efficient Online Data Mixing For Language Model Pre-Training
3. Aioli: A Unified Optimization Framework for Language Model Data Mixing
4. DoReMi: Optimizing Data Mixtures Speeds Up Language Model Pretraining
5. Automatic Document Selection for Efficient Encoder Pretraining
