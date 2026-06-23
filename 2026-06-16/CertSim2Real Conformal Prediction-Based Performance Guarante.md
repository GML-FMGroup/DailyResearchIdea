# CertSim2Real: Conformal Prediction-Based Performance Guarantees for Sim-to-Real Transfer via Dynamics-Invariant State Abstraction

## Motivation

Existing methods for sim-to-real transfer in robotic manipulation, such as those benchmarked in 'Experiences from Benchmarking Vision-Language-Action Models for Robotic Manipulation', rely on heuristic domain randomization or empirical evaluation on a limited set of real-world tasks, providing no formal performance guarantees. The root cause is that simulation is treated as a sufficient proxy without quantifying the distribution shift to real-world dynamics. This leaves a structural gap: no principled method to derive a certified lower bound on real-world performance from simulation data alone. We address this by leveraging conformal prediction on a learned state abstraction that is invariant across a simulator ensemble, enabling finite-sample guarantees.

## Key Insight

A dynamics-invariant state abstraction, learned via contrastive alignment of latent representations across multiple simulators, renders the distribution shift from simulation to real-world as a structured shift within the support of the training ensemble, allowing conformal prediction to produce calibrated real-world performance lower bounds.

## Method

**A) What it is**
CertSim2Real (C2R) is a certification framework that outputs a finite-sample lower bound on the real-world task success rate of a policy π, given access to an ensemble of simulators with different dynamics parameters and a small set of real-world calibration trajectories (optional). Its inputs are: a policy π, a set of K=5 simulators {S_k} with distinct dynamics (e.g., friction coefficient in [0.2,0.8], mass scaling in [0.5,1.5]), a task success indicator function R, and a small real-world dataset D_real (≥10 episodes). The output is a lower bound L such that P(R(π, real) ≥ L) ≥ 1-δ with δ=0.1.

**B) How it works**
```python
# Training Phase
Encoder φ: 4-layer CNN with BatchNorm, ReLU activations, output dim=64. Trained with Adam lr=1e-4, batch_size=64.
Train φ via contrastive learning across simulator ensemble:
  For each timestep, sample (s, s') from two different simulators with same task outcome.
  Loss = -log( exp( φ(s)^T φ(s') / τ ) / ( sum_negatives exp( φ(s)^T φ_neg / τ ) ) )   # InfoNCE, τ=0.1
  Use 256 negative samples per positive pair.

# Certification Phase
For each simulator S_k (k=1..5):
  Run policy π for N=100 episodes, collect abstract states z_t = φ(o_t) and binary success indicators g_t ∈ {0,1}.
  Compute nonconformity score A(z_t, g_t) = - ||z_t||^{-1} log( g_t + ε )   with ε=1e-6, lower is better.

Combine all scores from all simulators into set A_ensemble (size = 5*100*avg_episode_len).

# Conformal Prediction on observed real-world abstract states (if D_real available)
If real-world data is available:
  For each real episode i:
    z_i = φ(o_t) along rollout
    α_i = max_t A(z_i_t, g_i)   # worst-case score
  Calibration set: use real scores as calibration.
  Combine ensemble scores and real scores into pooled set: pool = A_ensemble ∪ {α_i}.
  Compute q̂ = quantile( pool, (1-δ)*(1+1/|pool|) ).
  Lower bound L = exp(-q̂ * max||z||) where max||z||=observed maximum norm of abstract states.
Else (no real data):
  Use only ensemble scores: q̂_ens = quantile(A_ensemble, 1-δ).
  Compute correction factor rho = max_{i,j} W_1( P_φ(S_i), P_φ(S_j) ) where W_1 is 1-Wasserstein distance approximated by Sinkhorn distance with regularization λ_S=0.1, using 1000 samples from each simulator.
  q̂_corrected = q̂_ens + λ * rho, with λ=0.5.
  L = exp(-q̂_corrected * max||z||).
```
**Load-bearing assumption (explicit):** The contrastive encoder φ learns a state representation that is invariant to simulator-specific dynamics, meaning that for any two simulators with the same task outcome, the distribution of φ(s) is exchangeable. This ensures that the real-world distribution of abstract states is contained within the convex hull of the simulator ensemble’s distribution. **This assumption is tested via a diagnostic check:** After training, compute the average nonconformity score per simulator for each of 10 bins of abstract state magnitude; if the maximum pairwise gap between simulator averages exceeds 0.2, the invariance is deemed insufficient and the certification is invalid (i.e., we do not output a bound).

**C) Why this design**
We chose contrastive learning over adversarial training for invariance because contrastive learning directly optimizes alignment between simulators without requiring a discriminator, which is more stable and scales to high-dimensional observations; the cost is that contrastive loss may not capture all invariant features if the negative sampling is insufficient. We chose an ensemble of simulators with varied dynamics (e.g., different friction, mass) rather than a single simulator with domain randomization because the ensemble provides a clear distributional support that can be used for conformal calibration; the trade-off is increased computational cost of running multiple simulators. We chose conformal prediction on abstract states rather than on raw observations because the abstraction reduces dimensionality and irrelevant variation, improving the efficiency of conformal sets; however, this depends on the quality of the encoder, and a poor encoder may discard task-relevant information. Hyperparameter λ for the correction factor was set by evaluating on a held-out set of simulator variants, accepting that this introduces a small dependence on held-out data.

**D) Why it measures what we claim**
The abstract state encoder φ measures dynamics-invariant features because contrastive learning pulls together states from different simulators that lead to the same task outcome, assuming that the outcome-determining features are shared across simulators; this assumption fails when the task outcome is influenced by simulator-specific artifacts (e.g., friction color), in which case φ may align on spurious correlations. The nonconformity score A(z, g) = -log(g+ε) measures the likelihood of failure given the abstract state; specifically, it is monotonic in the probability of failure under the implicit density of the ensemble. The conformal quantile q̂ on the pooled scores then provides a coverage guarantee for future test points under the assumption that the real-world distribution is exchangeable with the ensemble distribution when projected through φ; this assumption fails when the real-world dynamics lie outside the convex hull of the simulator ensemble, in which case the bound may be invalid or too conservative. Thus, the computed lower bound L reflects the minimal success rate achieved with probability at least 1-δ under the exchangeability assumption, ensuring that the certification is rigorous but contingent on the simulator ensemble covering real-world variation.

## Contribution

(1) A novel framework, CertSim2Real, that integrates dynamics-invariant state abstraction learned via contrastive learning across a simulator ensemble with conformal prediction to provide finite-sample lower bounds on real-world task success rates. (2) A design principle that performance certification requires the simulator ensemble to structurally cover plausible real-world dynamics shifts, operationalized by the invariance learning step. (3) (Optional) A methodology for deriving a correction factor when no real-world calibration data is available, using the maximal discrepancy between simulators.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
|------|--------|-----------|
| Dataset | LIBERO-10 manipulation tasks (10 tasks with known sim-to-real gaps) | Standardized sim-to-real benchmark with documented dynamics shifts |
| Primary metric | Lower bound L on success rate (target coverage 90%, δ=0.1) | Directly measures certification claim |
| Baseline 1 | Domain Randomization (DR) training on single simulator with random dynamics, no certification | Represents standard training without guarantees |
| Baseline 2 | Single Simulator (SS) certification using only one simulator (the best-matching one) + 10 real episodes calibration | Isolates ensemble benefit |
| Baseline 3 | No real calibration (NoReal): uses ensemble but no real-world data, correction factor rho | Tests value of real data |
| Ablation of ours | Without contrastive encoder: replace with randomly initialized φ (fixed) | Tests necessity of invariance learning |
| Additional diagnostic | Invariance gap test: max pairwise Wasserstein distance between simulator abstract state distributions (binned) | Check load-bearing assumption |

### Why this setup validates the claim

This experimental design directly tests the central claim that C2R produces a valid lower bound on real-world success rate. By using LIBERO-10, which features diverse manipulation tasks with known sim-to-real dynamics shifts (e.g., friction, mass, object stiffness), we can assess whether the ensemble of 5 simulators covers real-world variation. The baselines isolate key components: Domain Randomization represents standard training without certification, Single Simulator tests the ensemble's necessity, and No Real Calibration probes the value of real-world data. The primary metric L is the exact output claimed, so improvement over baselines demonstrates the certification's utility. The ablation of the contrastive encoder tests whether invariance learning is crucial for the abstraction. The diagnostic invariance gap test provides a concrete check on the load-bearing assumption: if the gap exceeds 0.2, the bound is not emitted, preventing false claims. Together, these choices form a falsifiable test: if C2R's L fails to be higher or valid where the identified failure modes occur, the idea is refuted.

### Expected outcome and causal chain

**vs. Domain Randomization (DR)** — On a task where the real-world friction differs significantly from the randomization distribution, DR's policy fails without any guarantee because it cannot detect the mismatch. Our method instead uses the ensemble to cover a range of dynamics, and conformal prediction yields a valid lower bound even if the real friction is at the edge of the ensemble. We expect C2R's L to be significantly higher than DR's (which is essentially zero) on such out-of-distribution tasks, while both may be similar on in-distribution tasks.

**vs. Single Simulator (SS)** — On a case where the real dynamics are well-represented by one simulator but not a single fixed one, SS produces an overly optimistic bound because its calibration is only valid for that simulator. Our method aggregates over the ensemble, so the bound remains valid even if the real dynamics lie between simulators. We expect C2R's L to be higher than SS's when the single simulator has a large sim-to-real gap, but comparable when the single simulator happens to match reality.

**vs. No real calibration (NoReal)** — On a task where the sim-to-real gap is large but the ensemble is diverse, NoReal's correction factor (using rho) may be too conservative or too optimistic because it relies on maximal Wasserstein discrepancy. Our method's optional real-world calibration directly aligns the conformal quantile with real data, yielding a tighter and correct bound. We expect C2R with real data to have higher L than NoReal on tasks with large sim-to-real shift (where rho correction is imprecise), while both may be similar when the shift is small or when the correction factor rho accurately captures the shift.

**vs. Ablation (without contrastive encoder)** — On a case where non-invariant features (e.g., background color) differ between simulators, the ablation's encoder may retain spurious correlations, causing the abstraction to map dissimilar states together and produce inflated nonconformity scores. Our contrastive encoder instead discards those features, making the scores more accurate and the bound tighter. We expect C2R to have higher L than the ablation on tasks with visual distractors (e.g., varying table texture), but similar on purely dynamics-variant tasks.

### What would falsify this idea

If C2R's lower bound L is not consistently above the baseline without contrastive encoder across tasks with visual distractors, or if L ever violates the coverage guarantee (i.e., the true real-world success rate falls below L on a held-out test set of at least 50 episodes), then the central claim that the framework provides a valid certification is falsified. Additionally, if the diagnostic invariance gap exceeds 0.2 for many tasks (indicating the load-bearing assumption fails), the method's applicability is severely limited.

## References

1. Experiences from Benchmarking Vision-Language-Action Models for Robotic Manipulation
2. π0: A Vision-Language-Action Flow Model for General Robot Control
3. RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control
4. ManiSkill2: A Unified Benchmark for Generalizable Manipulation Skills
5. Approximate convex decomposition for 3D meshes with collision-aware concavity and tree search
6. Fine-Tuning Vision-Language-Action Models: Optimizing Speed and Success
7. Language Conditioned Multi-Finger Dexterous Manipulation Enabled by Physical Compliance and Switching of Controllers
