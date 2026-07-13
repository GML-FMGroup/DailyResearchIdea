# Calibrating Docking Rewards via Heteroscedastic Noise Modeling and Active Learning for RL-Based Drug Generation

## Motivation

Existing RL-based drug generation methods, such as DrugGen 2, use docking scores as reward signals without experimental validation, leading to reward overoptimization and distribution shift where the generator exploits docking score outliers. This is a structural meta-gap because all three branches of related work rely on uncalibrated docking predictions as a surrogate for true binding affinity, ignoring the systematic heteroscedastic noise that varies across molecular scaffolds. Without grounding the reward in experimental data, the generator's distribution shifts towards unrealistic molecules with high docking scores but low actual affinity.

## Key Insight

A heteroscedastic noise model of docking scores versus experimental binding affinities, when combined with a lower confidence bound (LCB) reward, provides a conservative estimate that directly penalizes molecules with high prediction uncertainty, preventing overoptimization by design.

## Method

(A) **What it is**: We propose CDR-HAL (Calibrated Docking Reward via Heteroscedastic Active Learning), a method that replaces the raw docking score reward in RL-based drug generation with a calibrated LCB reward derived from a heteroscedastic Gaussian process (GP) that models the relationship between docking score and true binding affinity. An active learning loop selects diverse, high-uncertainty molecules for experimental verification to update the GP. Inputs: generated molecule's docking score and Morgan fingerprint (1024 bits). Outputs: a scalar reward for RL training.

(B) **How it works**: Pseudocode:
```python
# Phase 1: Bootstrap initial GP
D_init = {molecule_i: (docking_score_i, exp_affinity_i)}  # small set of experimental data
GP = HeteroscedasticGP(kernel=RBF(1.0, lengthscale=0.5), noise_model=Exponential(noise_level=1.0))
GP.fit(X=docking_scores_of_D_init, y=exp_affinities_of_D_init)

# Phase 2: RL training with LCB reward
for iteration in range(10):  # 10 active learning rounds
    # Train RL generator (e.g., GRPO as in DrugGen 2) with reward = LCB
    for step in range(1000):
        molecule = generator.sample()
        docking_score = docking_tool(molecule)
        mu, sigma = GP.predict(docking_score, input_features=MorganFP(molecule))
        reward = mu - 1.96 * sigma  # LCB (95% confidence)
        generator.update_with_GRPO(reward)
    
    # Active learning: select molecules for experimental validation
    candidate_pool = generator.sample(500)
    muc, sigmac = GP.predict_on_pool(candidate_pool)  # using Morgan fingerprints
    acquisition = (muc - LCB_optimistic) / sigmac  # uncertainty-weighted improvement
    top_k = candidate_pool[argsort(acquisition)[:20]]  # select 20 most uncertain
    exp_affinities = experimental_assay(top_k)
    GP.update(X=docking_scores_of_top_k, y=exp_affinities, features=MorganFP(top_k))
```
Hyperparameters: RBF kernel lengthscale=0.5, LCB confidence level 1.96 (95%), active learning batch size 20, 10 rounds.

(C) **Why this design**: We chose heteroscedastic GP over homoscedastic GP because docking score noise scales with molecular complexity (e.g., flexible ligands have higher variance), and modeling this avoids underestimating uncertainty on high-risk molecules—accepting the cost of more complex inference (exponential noise model). We chose LCB over mean reward because the mean would still allow exploitation of high-docking-score molecules with large uncertainty, whereas LCB explicitly penalizes them; this trades off some exploration speed for safety against overoptimization. For active learning acquisition, we use uncertainty-weighted improvement rather than pure uncertainty (e.g., Thompson sampling) because it balances exploring high-uncertainty molecules with those having promising average scores, avoiding wasting experiments on molecules that are certainly poor binders. This design differs from prior active learning for docking (e.g., in TamGen, which used random selection) by selecting molecules that maximally reduce GP uncertainty around the reward frontier.

(D) **Why it measures what we claim**: The heteroscedastic GP's predicted variance sigma measures epistemic uncertainty about the docking-to-affinity mapping because it quantifies the model's lack of knowledge in regions of chemical space not yet experimentally validated; this assumption fails when the GP kernel is misspecified (e.g., Morgan fingerprint similarity does not correlate with binding mechanism similarity), in which case sigma reflects irrelevant structural distance instead of prediction risk. The LCB reward (mu - 1.96*sigma) measures a risk-averse lower bound on true affinity because we assume the GP's posterior is approximately Gaussian and well-calibrated; this assumption fails when the true noise distribution is heavy-tailed, causing the LCB to be overly conservative. The active learning acquisition (uncertainty-weighted improvement) measures the expected reduction in risk for the reward function because it preferentially selects molecules where the GP has high uncertainty and moderately high predicted mu; this assumption fails when experimental cost varies across molecules, in which case the acquisition ignores budget constraints.

## Contribution

(1) A heteroscedastic Gaussian process model that calibrates docking score rewards for RL-based drug generation by learning the distribution of docking-to-affinity mapping. (2) An active learning loop that iteratively selects diverse, high-uncertainty molecules for experimental verification to update the noise model, enabling data-efficient calibration. (3) A risk-averse reward formulation (LCB) that directly prevents overoptimization by penalizing molecules with high prediction uncertainty.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | PDBbind-v2020 | Provides experimental affinities for validation. |
| Primary metric | Mean experimental affinity (pIC50) of top-100 molecules | Directly measures true binding ability. |
| Baseline 1 | DrugGen | Tests necessity of calibration vs raw docking. |
| Baseline 2 | DrugGen 2 | Tests value of LCB over strong RL baseline. |
| Ablation | CDR-HAL (mean reward) | Isolates effect of LCB from GP uncertainty. |

### Why this setup validates the claim
The choice of PDBbind-v2020, a benchmark with known experimental affinities, ensures that the primary metric—mean experimental binding affinity of top-100 generated molecules—directly measures the method's ability to discover true high-affinity binders. Comparing against DrugGen (raw docking score reward) tests the fundamental claim that calibration improves upon uncalibrated RL. DrugGen 2, which uses a more advanced RL algorithm (GRPO) but still with raw docking scores, controls for the RL algorithm itself, isolating the benefit of the LCB reward. The ablation without LCB (mean reward) further isolates the specific contribution of the LCB. The active learning component is tested via the no-update ablation. This design creates a falsifiable test: if the LCB is crucial, the mean-reward ablation should underperform on high-uncertainty molecules; if active learning is key, the no-update ablation should show degraded performance on novel chemotypes.

### Expected outcome and causal chain

**vs. DrugGen** — On a case where a generated molecule has a high docking score but is a false positive due to e.g., water-mediated binding ignored by the scoring function, DrugGen's raw docking reward will overestimate its true affinity, leading the RL policy to exploit such deceptive molecules. Our method, using LCB, penalizes high-uncertainty molecules, so it avoids rewarding such false positives. Consequently, we expect our method to generate molecules with higher experimental affinity on average, particularly a noticeable gap (e.g., 0.5 log units) for molecules with high docking scores but high model uncertainty.

**vs. DrugGen 2** — On a case where the policy is near convergence and further gains require exploring risky molecules with large variance, DrugGen 2's raw docking reward (even with GRPO) will exploit high-docking-score molecules without penalizing uncertainty, potentially converging to a suboptimal local optimum. Our method's LCB encourages exploration of uncertain but promising regions. We expect that over active learning rounds, our method's top-100 average affinity improves more steeply and reaches a higher plateau, with the gap widening particularly in later rounds (e.g., 0.3 log units after round 5).

**vs. Ablation (mean reward)** — On a molecule with moderate predicted mean but high uncertainty, mean reward treats it the same as a certain prediction, failing to avoid potential overestimation. Our LCB penalizes it. Thus, we expect our method to generate safer molecules with fewer low-affinity outliers. The primary gap will be in the variance of top-100 affinities: our method should have lower variance (e.g., 0.2 log units standard deviation vs. 0.5 for mean reward).

### What would falsify this idea
If our LCB-based method does not outperform the mean-reward ablation on high-uncertainty molecules, or if the performance gap between our method and DrugGen 2 is uniform across all molecules rather than concentrated on those with high epistemic uncertainty, then the central claim that LCB provides risk-averse calibration is falsified. Similarly, if the active learning component yields no benefit over the fixed GP, the claim that iterative experimental validation reduces uncertainty on the reward frontier is unsupported.

## References

1. DrugGen 2: A disease-aware language model for enhancing drug discovery
2. DrugGen enhances drug discovery with large language models and reinforcement learning
3. Token-Mol 1.0: tokenized drug design with large language models
4. TamGen: drug design with target-aware molecule generation through a chemical language model
5. DrugGPT: A GPT-based Strategy for Designing Potential Ligands Targeting Specific Proteins
6. Generation of 3D molecules in pockets via a language model
