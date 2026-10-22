# Identifiability-Gated Simulator Pretraining for Embodied Navigation

## Motivation

Prior work such as VLNVerse, GRUtopia, and Navigation World Models treats the simulator's transition kernel as a faithful oracle, generating pretraining signal without any quantitative check that simulated transitions match real embodied dynamics. This structural assumption means that any mismatch between the physics engine (or learned world model) and reality silently corrupts the pretraining targets, and no existing method gates simulator-generated data by measured sim-to-real fidelity of the transition kernel itself.

## Key Insight

A simulator transition is admitted as a pretraining target only if it is statistically indistinguishable from real transitions under a learned sufficient-statistic test, which makes the pretraining signal identifiable with respect to real dynamics rather than merely plausible under the simulator.

## Method

**What it is** — IGSP (Identifiability-Gated Simulator Pretraining) is a wrapper around any simulator-based pretraining pipeline for embodied navigation. It takes as input a simulator transition kernel and a small set of real embodied rollouts, and outputs a gated dataset of simulated transitions that pass a learned sufficient-statistic two-sample test against real dynamics.

**How it works** — We learn a sufficient statistic φ(s, a, s') via a contrastive objective that maximizes the likelihood ratio between real and simulated transitions. For each candidate simulated transition, we compute a two-sample test statistic T = ||φ(s,a,s') - μ_real||²_Σ⁻¹ where μ_real and Σ are estimated from held-out real transitions (n_real ≥ 500). A transition is admitted if T ≤ τ, where τ is set by a permutation test at significance α = 0.05. The gated dataset is then used for standard behavior cloning pretraining. The full procedure is:

```pseudocode
Input: simulator S, real rollouts R = {(s,a,s')}, candidate simulated transitions C
1. Fit φ via contrastive loss on R ∪ C (InfoNCE, temperature 0.1)
2. Estimate μ_real, Σ from φ(R) using held-out real data
3. For each c in C: compute T(c) = ||φ(c) - μ_real||²_Σ⁻¹
4. Determine τ via permutation test: shuffle real/sim labels 1000 times, set τ as 95th percentile of max T under null
5. Gate: D_gated = {c in C : T(c) ≤ τ}
6. Pretrain navigation policy on D_gated via behavior cloning
Output: pretrained policy π
```

**Why this design** — We chose a learned sufficient statistic φ over raw state-action pairs because raw pixel-level comparisons are dominated by nuisance factors (lighting, texture) that do not affect dynamics, accepting the cost that φ must be trained on limited real data and may overfit; we mitigate this by using a contrastive objective that explicitly separates real from simulated transitions. We chose a permutation-test threshold τ over a fixed margin because the null distribution of T under the hypothesis 'simulator is faithful' is unknown and simulator-dependent, accepting the cost of extra computation at calibration time. We chose to gate at the transition level rather than the trajectory level because pretraining signal is composed of individual transitions and a single bad transition can corrupt the policy, accepting the cost that we may discard some useful long-horizon structure. This design is not a variant of prior sim-to-real calibration methods (e.g., domain randomization or system identification) because those adjust the simulator parameters to match real data, whereas we leave the simulator unchanged and instead gate which of its outputs are admissible as training targets based on a statistical test of indistinguishability.

**Why it measures what we claim** — The test statistic T(c) measures the sim-to-real identifiability of transition c because under the assumption that φ is a sufficient statistic for the real-vs-simulated distinction, T(c) is a monotone transform of the likelihood ratio p_real(c)/p_sim(c); this assumption fails when φ is not sufficient (e.g., it ignores a dynamical feature that differs between real and simulated but is not predictive of the real/sim label in the training set), in which case T(c) reflects only the projection of the discrepancy onto the learned feature space and may admit transitions that differ in the ignored feature. The permutation-test threshold τ measures the fidelity of the simulator's transition kernel because it is calibrated to the null hypothesis that real and simulated transitions are exchangeable under φ; this assumption fails when the real data is not representative of the deployment distribution, in which case τ becomes too permissive or too strict relative to the true deployment fidelity.

## Contribution

['A novel identifiability-gated pretraining framework (IGSP) that wraps any simulator-based pretraining pipeline with a learned sufficient-statistic two-sample test to admit only statistically indistinguishable simulated transitions.', 'A concrete design principle: gating at the transition level with a permutation-test threshold yields pretraining datasets whose fidelity is quantitatively controlled, unlike prior sim-to-real methods that adjust simulator parameters rather than gate outputs.', 'An open-source implementation of the gating module and calibration procedure, compatible with existing embodied navigation simulators such as VLNVerse and GRUtopia.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | VLNVerse + real robot rollouts | Sim benchmark plus genuine real dynamics for gating |
| Primary metric | Success rate on real-world held-out episodes | Directly reflects deployment fidelity, not sim score |
| Baseline | Uni-NaVid (video VLA, no gating) | Tests whether gating beats strong sim-pretrained policy |
| Baseline | Domain-randomized sim pretraining | Tests gating against parameter-shifting calibration |
| Baseline | System-identification sim pretraining | Tests gating against simulator-matching calibration |
| Baseline | Density-ratio importance weighting (same φ) | Tests gating against reweighting, not thresholding |
| Ablation-of-ours | IGSP with fixed margin τ, no permutation test | Isolates contribution of calibrated threshold |
| Ablation-of-ours | IGSP with φ replaced by fixed dynamical features | Tests whether learned φ is necessary |

### Why this setup validates the claim

The central claim is that gating simulated transitions by a learned identifiability statistic yields a pretraining set whose dynamics are statistically indistinguishable from real, and that this improves real-world navigation. The dataset pairing (a rich simulated benchmark plus a modest set of real rollouts) is what lets us both pretrain at scale and calibrate the gate against true deployment dynamics. The primary metric—real-world success rate on held-out episodes—is the right detector because the mechanism predicts a gain specifically on transitions where sim and real diverge; a metric that rewards sim-like behavior (e.g., sim success) would be blind to the effect. Uni-NaVid tests whether gating improves over an ungated strong pretrained policy; domain randomization and system identification test whether gating beats the two dominant calibration philosophies that adjust the simulator rather than filter its outputs. The density-ratio importance-weighting baseline tests whether hard gating yields a qualitatively different admitted set and gain than reweighting with the same φ, establishing that the contribution is not just a thresholded version of known reweighting. The fixed-margin ablation tests whether the permutation-calibrated threshold, not merely filtering, drives the effect. The fixed-feature ablation tests whether the learned φ is necessary or whether simple dynamical statistics suffice. Together these form a falsifiable test: the gain must be present versus ungated pretraining, must not be explained away by simulator adjustment or by arbitrary thresholding, and must differ from reweighting.

### Expected outcome and causal chain

**vs. Uni-NaVid (video VLA, no gating)** — On a case where the simulator renders a corridor with different friction and lighting than the real deployment, Uni-NaVid pretrains on all simulated transitions and learns a policy that over-relies on simulator-specific visual-motion correlations, producing drift and collisions near textured floors. Our method instead admits only transitions whose φ-statistic falls inside the real exchangeability region, so the policy sees dynamics consistent with real rollouts. We expect a visible success-rate gap concentrated on real episodes with high sim-real visual/dynamic divergence, with near-parity on episodes where the simulator is already faithful.

**vs. Domain-randomized sim pretraining** — On a case where randomization broadens the simulator over lighting and friction but leaves the underlying transition kernel mismatched (e.g., a systematic slip model), the randomized baseline learns a policy robust to nuisance variation yet still calibrated to the wrong slip dynamics, producing overconfident turning on real slippery surfaces. Our method leaves the simulator unchanged and instead rejects transitions whose dynamics fall outside the real φ-distribution, so the pretraining targets track real slip behavior. We expect the gap to appear specifically on slip-sensitive episodes while both methods match on benign surfaces.

**vs. System-identification sim pretraining** — On a case where system identification fits simulator parameters to aggregate real statistics but misses a transient dynamic (e.g., a brief acceleration lag), the baseline matches average behavior yet propagates the transient mismatch into every pretraining transition, yielding delayed corrective actions on real starts and stops. Our method gates at the transition level, so transitions exhibiting the mismatched transient are excluded even though the aggregate simulator looks calibrated. We expect a success-rate advantage on episodes with abrupt motion changes, and near-parity on steady-state navigation.

**vs. Density-ratio importance weighting (same φ)** — On a case where the simulator produces many transitions with moderate mismatch and a few with severe mismatch, importance weighting assigns low but nonzero weight to all, so the policy still trains on mismatched dynamics, whereas hard gating excludes them entirely. We expect the gated policy to outperform reweighting on episodes requiring precise dynamics (e.g., narrow gaps), with the gap growing as the fraction of severely mismatched transitions increases.

**vs. IGSP with fixed margin τ** — On a case where the fixed margin is either too loose or too tight relative to the actual simulator-real discrepancy, the ablation either admits mismatched transitions or discards too many useful ones, producing a policy that is respectively miscalibrated or undertrained. Our permutation-calibrated τ adapts to the simulator-dependent null distribution, so the admitted set is neither inflated nor starved. We expect the ablation to show higher variance in real success across environments, while our method stays stable.

**vs. IGSP with fixed dynamical features** — On a case where the true mismatch involves a feature not captured by the fixed features (e.g., a subtle timing offset), the fixed-feature variant cannot detect it and admits mismatched transitions, while the learned φ can if the feature is predictive in the training data. We expect the learned φ to yield higher success on episodes with such mismatches, but we also expect both to fail on mismatches absent from the real data, highlighting the coverage limitation.

### What would falsify this idea

If the real-world success gain over ungated pretraining is uniform across episodes rather than concentrated on episodes where the predicted sim-real divergence is high, the claim that gating targets identifiability is wrong. Equivalently, if the fixed-margin ablation matches the calibrated threshold's performance, the permutation-test calibration contributes nothing and the mechanism reduces to arbitrary filtering.

## References

1. VLNVerse: A Benchmark for Vision-Language Navigation with Versatile, Embodied, Realistic Simulation and Evaluation
2. Uni-NaVid: A Video-based Vision-Language-Action Model for Unifying Embodied Navigation Tasks
3. GRUtopia: Dream General Robots in a City at Scale
4. Navigation World Models
5. Towards Long-Horizon Vision-Language Navigation: Platform, Benchmark and Method
6. SAME: Learning Generic Language-Guided Visual Navigation with State-Adaptive Mixture of Experts
7. Habitat 3.0: A Co-Habitat for Humans, Avatars and Robots
8. Qwen Technical Report
