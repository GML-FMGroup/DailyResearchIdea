# DRIFT: Detecting Preference Drift via Cumulative Regret Monitoring in Long-Horizon Agent Interactions

## Motivation

Existing benchmarks like DeepPlanning assume static user preferences, but real-world interactions involve preference drift. TripScore and TripTailor evaluate plans against static criteria, ignoring changes. The root cause is that adaptation mechanisms either require costly continuous re-estimation or lack theoretical grounding for when to trigger re-planning.

## Key Insight

Under bounded drift frequency, the cumulative regret of a fixed policy against a consistently estimated preference model follows a sublinear rate; a deviation beyond a theoretical bound signals drift, enabling lightweight detection without re-estimation.

## Method

**(A) What it is**: DRIFT is a lightweight statistical detection mechanism that monitors cumulative regret relative to a static reference policy (with periodic recalibration). It takes as input the agent's action sequence, observed rewards, and an initial preference model. It outputs a binary drift signal when cumulative regret exceeds a theoretically derived threshold.

**(B) How it works (pseudocode)**
```python
# Hyperparameters: σ² (reward variance, estimated from initial 100 steps), δ=0.05 (confidence), C=0.1*reward_range (misspecification constant), W=100 (sliding window size)
1: Initialize preference model M0 from initial 500 user interactions.
2: Define static reference policy π_ref that selects actions maximizing M0's predicted reward.
3: Initialize R = 0, t = 0, buffer = []
4: For each interaction step:
   a. Agent selects action a_t.
   b. Observe reward r_t.
   c. Compute δ_t = M0.predict_reward(π_ref.action(context_t)) - r_t  # regret increment
   d. R += δ_t
   e. Compute threshold θ_t = sqrt(2 * σ² * t * log(1/δ)) + C
   f. If R > θ_t: output drift signal and trigger re-planning (re-estimate M0 from most recent W interactions).
   g. Every W steps (or on drift signal), update M0 using last W interactions (sliding window).
5: Continue.
```

**(C) Why this design**: We chose to use cumulative regret against a static reference policy (π_ref) rather than an adaptive oracle because a static baseline yields a known regret bound under no drift (sublinear), making detection straightforward; the cost is that any model misspecification in M0 adds constant bias to regret, which we accommodate with the C term. We use a Hoeffding-style threshold (θ_t) with a variance estimate σ² because it provides a distribution-free bound, avoiding parametric assumptions; the cost is that it may be conservative, delaying detection. We compute regret increment δ_t using M0's predicted reward for π_ref's action rather than waiting for actual user feedback on that action, which would require additional queries; this introduces potential misalignment if M0's predictions are inaccurate, but we mitigate by updating M0 only after drift is detected (or periodically every W steps). We set the confidence parameter δ to control false positives, trading off detection delay vs. false alarm rate. This design relies on the assumption that the initial preference model M0 remains sufficiently accurate for predicting rewards under π_ref over the entire evaluation horizon in the absence of drift. To mitigate this, we incorporate a periodic re-estimation of M0 using a sliding window of recent data (window size W) even when no drift is detected, ensuring the reference policy stays calibrated.

**(D) Why it measures what we claim**: The cumulative regret R measures the divergence between the agent's actual performance and the performance expected under static preferences because, under no drift, R grows sublinearly with t (due to random noise); when preferences drift, the true reward for π_ref's actions decreases, causing δ_t to systematically increase, so R grows faster. This equivalence relies on the assumption that M0's reward predictions remain calibrated for π_ref's actions in the no-drift regime; this assumption fails if M0 is poorly estimated initially, in which case R reflects model error rather than drift. The threshold θ_t captures the maximum plausible R under no drift, bounded by σ² and δ; it measures the confidence interval for R under the null hypothesis (no drift). This assumes rewards are bounded and temporally independent given context; when rewards are autocorrelated (e.g., user mood), the threshold may underestimate variance, causing false positives. Additionally, cumulative regret R measures preference drift only under the assumption that the agent's policy is at least as good as π_ref under no drift. If the agent is exploratory or suboptimal, δ_t may be negative even without drift, causing false positives. This equivalence fails when the agent's actions are worse than π_ref's actions, in which case R reflects agent suboptimality rather than drift.

## Contribution

['(1) A theoretically grounded drift detection mechanism for long-horizon agent interactions that monitors cumulative regret against a static reference policy, eliminating the need for continuous re-estimation.', '(2) The introduction of a Hoeffding-based threshold that provably controls false alarm rate under bounded drift frequency, providing a practical adaptation trigger.', '(3) A demonstration that the method can be integrated into existing planning benchmarks (e.g., DeepPlanning) to evaluate agent adaptability under preference drift.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|----------------------|
| Dataset | Synthetic Preference Drift Benchmark + MovieLens subset | Controlled drift and real-world validation |
| Primary metric | Drift Detection F1 Score | Balances precision and recall |
| Baseline 1 | Sliding Window Average | Simple threshold on mean reward |
| Baseline 2 | CUSUM | Cumulative sum change detection |
| Baseline 3 | Bayesian Changepoint | Probabilistic change point model |
| Baseline 4 | ADWIN (Adaptive Windowing) | Non-regret online change detection |
| Ablation 1 | DRIFT w/ periodic reference update (window W=100) | Replaces pure static with periodic |
| Ablation 2 | DRIFT w/ suboptimal agent (random policy) | Tests robustness to agent suboptimality |

### Why this setup validates the claim
This experimental design forms a falsifiable test of DRIFT's core claim: that cumulative regret against a static reference policy (with periodic recalibration) detects preference drift with bounded false positives. The synthetic dataset enables precise control over drift timing and magnitude, while the MovieLens dataset (user rating sequences over time) provides realistic, noisy preference shifts. The four baselines represent distinct detection philosophies (simple heuristic, nonparametric cumulative sum, probabilistic model, adaptive windowing), each probing a different weakness: sliding window fails on gradual drift, CUSUM struggles with misspecified variance, Bayesian methods require prior knowledge, ADWIN may have high false positives on noise. The ablations isolate the contribution of the static reference (by swapping for periodic update) and the assumption of agent optimality (by using a random policy). The F1 metric is appropriate because it captures both detection delay (via recall) and false alarms (via precision), directly reflecting the theoretical trade-off controlled by δ. Additionally, we will release a public implementation with the synthetic data generator to ensure reproducibility.

### Expected outcome and causal chain

**vs. Sliding Window Average** — On a case where drift is gradual (e.g., preference shifts linearly over 100 steps), the sliding window baseline averages recent rewards, so the mean changes slowly and detection is delayed or missed entirely. Our DRIFT method accumulates regret against a static reference; each step's negative regret increment adds up quickly because the reference's predicted reward becomes increasingly outdated, triggering the threshold earlier. We expect DRIFT to achieve a higher recall (e.g., >0.9 vs ~0.6) on gradual drift subsets, with comparable precision.

**vs. CUSUM** — On a case where reward variance is high and drift is abrupt (e.g., a single step jump), CUSUM's cumulative sum may be dominated by noise, causing false positives or delayed detection. Our method uses a variance-adaptive threshold derived from Hoeffding's inequality, which automatically widens the confidence band under high noise, reducing false alarms. However, this conservatism may increase detection delay on abrupt low-noise shifts. We expect DRIFT to have lower false positive rate (e.g., <0.05 vs >0.2) on high-noise subsets, with slightly longer delay on abrupt drift.

**vs. Bayesian Changepoint** — On a case where the drift occurs in a non-stationary pattern (e.g., multiple small shifts), Bayesian changepoint may overfit by inferring many changepoints or underfit if prior assumptions are wrong. Our method aggregates regret linearly, so it naturally smoothes over multiple small shifts, triggering only when cumulative divergence exceeds the threshold. We expect DRIFT to have more stable F1 across drift patterns, while Bayesian performance varies significantly with prior misspecification.

**vs. ADWIN** — On a case with abrupt drift and high noise, ADWIN's adaptive window may shrink too aggressively, causing false positives; DRIFT's variance-adaptive threshold provides more robustness. We expect DRIFT to have higher precision (e.g., >0.95 vs ~0.8) on high-noise subsets.

**Ablation: periodic reference update** — On a case with model misspecification in M0, the pure static DRIFT may have elevated false positives; the periodic update (window W=100) recalibrates π_ref, reducing false alarms. We expect the ablation to improve F1 on misspecified initial models (e.g., 0.85 vs 0.70).

**Ablation: suboptimal agent** — On a case with a suboptimal agent (random policy), DRIFT with static reference will produce false positives because regret increases from suboptimality; the suboptimal agent ablation will show lower F1 (e.g., <0.4) compared to the optimal agent case, confirming that the method is sensitive to agent quality.

### What would falsify this idea
If DRIFT's detection performance is uniformly inferior to all baselines across drift types, or if the ablation with adaptive reference shows no degradation, then the static reference is not a key component and the central claim of regret-based detection is invalid. Additionally, if the suboptimal agent ablation does not produce more false positives than the optimal agent case, then the claim that DRIFT measures drift rather than agent suboptimality is falsified.

## References

1. DeepPlanning: Benchmarking Long-Horizon Agentic Planning with Verifiable Constraints
2. TripScore: Benchmarking and rewarding real-world travel planning with fine-grained evaluation
3. TripTailor: A Real-World Benchmark for Personalized Travel Planning
4. NATURAL PLAN: Benchmarking LLMs on Natural Language Planning
5. TravelPlanner: A Benchmark for Real-World Planning with Language Agents
6. ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs
7. Reasoning with Language Model is Planning with World Model
8. Large Language Models Cannot Self-Correct Reasoning Yet
