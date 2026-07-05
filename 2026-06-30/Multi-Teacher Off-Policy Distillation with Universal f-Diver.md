# Multi-Teacher Off-Policy Distillation with Universal f-Divergence Correction

## Motivation

Existing on-policy distillation methods for language models, such as AsyncOPD, assume a single fixed teacher and KL divergence, limiting their applicability to ensemble distillation or scenarios where the teacher is iteratively updated. This structural limitation stems from the reliance on V-trace corrections designed for a single target policy and a specific divergence. We need a universal off-policy correction that supports multiple teachers and any f-divergence loss, enabling generalization to ensemble and evolving teacher setups.

## Key Insight

Using a geometric mean of teacher densities as the distillation target and a divergence-specific gradient multiplier derived from the density ratio provides a universal correction that works for any f-divergence and any number of teachers, because the geometric mean naturally aggregates multiple teacher distributions while the gradient multiplier adapts the correction to the chosen divergence.

## Method

## Multi-Teacher Off-Policy Distillation with Universal f-Divergence Correction

### (A) What it is
We propose **MTOPD** (Multi-Teacher Off-Policy Distillation), a universal off-policy correction mechanism for on-policy distillation. It takes a stale student policy `π_s`, a set of teacher policies `{π_t_i}`, and a choice of f-divergence `D_f`. It outputs a corrected gradient estimate that allows using stale rollouts for updating the student.

### (B) How it works
```pseudocode
Input: student policy π_s (parameterized), teacher policies π_t_1,...,π_t_k,
       stale rollout buffer B from π_s_old, f-divergence specified by convex function f
Output: corrected gradient ∇L

For each transition (s, a) in B:
  1. Compute density ratios r_i = π_t_i(a|s) / π_s_old(a|s) for each teacher i.
  2. Compute geometric mean target density ratio:
        r_geometric = (∏_i r_i^{w_i})^{1/∑ w_i}, where w_i are teacher weights (default uniform).
     This corresponds to the density ratio between the geometric mean teacher distribution and π_s_old.
     **Note:** The geometric mean of densities is not normalized; we assume the normalizing constant Z is close to 1 (i.e., teacher densities are well-aligned) and treat r_geometric as the density ratio up to a constant factor.
  3. Compute on-policy density ratio r_on = 1 (since data is from π_s_old).
  4. Compute importance weight ρ = r_geometric / r_on = r_geometric.
  5. Compute gradient multiplier for f-divergence: 
        g(ρ) = f'(ρ) - f'(1) (where f' is the derivative of f).
     For KL divergence (f(x)=x log x): g(ρ)=log ρ; for reverse KL (f(x)=-log x): g(ρ)=-1/ρ; for Jensen-Shannon divergence (f(x)=x log x - (x+1)log((x+1)/2)): g(ρ)=log(2ρ/(1+ρ)).
  6. The corrected gradient for the student log-probability is:
        ∇L = g(ρ) * ∇log π_s(a|s).

The overall loss is average over the batch. Truncation: clip ρ to [1-ε, 1+ε] to control variance (ε=0.2 default).
```

### (C) Why this design
We chose the geometric mean over arithmetic mean for combining teacher densities because the geometric mean corresponds to the target distribution that minimizes the average f-divergence to the teachers under a log-loss (i.e., it is the barycenter in the space of log-densities), ensuring that the combined target is a proper probability distribution. This is in contrast to arithmetic mean, which may produce a target that is not a valid density when teachers disagree. We derived the gradient multiplier g(ρ) directly from the derivative of the f-divergence, following the property that the gradient of D_f(π_s || π_target) with respect to π_s parameters is E_{π_s}[g(ρ)∇log π_s]. This is a theoretical grounding that ensures the update direction matches the chosen divergence exactly, unlike heuristic clipping. We introduce truncation of ρ to control variance at the cost of slightly biased gradient estimates, which is a necessary trade-off for stability in high-dimensional action spaces. The design is not a simple combination of V-trace and importance sampling; it generalizes V-trace by replacing the single target density ratio with a geometric mean and the V-trace truncation with f-divergence-specific multipliers.

### (D) Why it measures what we claim
The density ratio ρ = r_geometric measures **relative off-policyness** between the stale data distribution π_s_old and the combined teacher target distribution, because the geometric mean target density is the density that minimizes the average f-divergence to the teachers; this assumption holds when the teachers are independent proposals and the student is being distilled to match their consensus. The gradient multiplier g(ρ) measures **the direction and magnitude of the correction** for the chosen f-divergence, because it is derived from the first-order condition of the divergence—this equivalence relies on the assumption that the stale data distribution adequately covers the support of the target; when support mismatch occurs (e.g., stale data lacks some actions the teachers prefer), g(ρ) may become extreme or undefined, indicating that the correction is unreliable in low-density regions. The truncation of ρ measures **a bounded version of off-policyness** that trades off bias for variance; the failure mode is that when true ρ lies outside the clip range, the gradient direction becomes clipped to the boundary, effectively using a different divergence than intended. 

**Diagnostic for support coverage:** In practice, we measure the fraction of transitions where ρ is extreme (e.g., >2 or <0.5) to quantify support mismatch; if this fraction exceeds 10% for a batch, we flag that the correction may be unreliable and optionally reduce the learning rate or switch to on-policy data.

## Contribution

(1) A universal off-policy correction mechanism for on-policy distillation that supports multiple teachers and any f-divergence loss, via a geometric mean of teacher densities and a divergence-specific gradient multiplier. (2) A design principle that the gradient multiplier should be the derivative of the f-divergence evaluated at the density ratio, unifying previous ad-hoc corrections like V-trace. (3) A computational framework that enables ensemble distillation and iterative teacher improvement with stale rollouts.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | GSM8K math reasoning | Tests language model distillation under distribution shift. |
| Primary metric | Accuracy | Direct measure of policy quality improvement. |
| Baseline 1 | Synchronous OPD (no staleness) | Upper bound: no off-policy correction needed. |
| Baseline 2 | Stale OPD without correction | Baseline fails due to staleness mismatch. |
| Baseline 3 | V-trace with clipped IS | Off-policy correction using importance sampling. |
| Baseline 4 | Importance-weighted average of teacher gradients | Each teacher’s gradient (under forward KL) averaged with importance weights ρ_i = π_t_i/π_s_old, clipped to [0.8,1.2]. |
| Ablation | MTOPD with arithmetic mean teachers (ρ = (1/K) Σ r_i, g(ρ)=log ρ) | Tests importance of geometric mean. |

### Why this setup validates the claim
This combination forms a falsifiable test of MTOPD's central claim: that its off-policy correction via geometric mean teacher combination and f-divergence‑specific multipliers enables effective on-policy distillation from stale data. Synchronous OPD provides the ideal performance ceiling; stale uncorrected OPD isolates the staleness problem; V‑trace tests a competing correction method; the importance-weighted average baseline tests a simpler multi-teacher aggregation. The ablation removing the geometric mean tests whether that design choice is responsible for gains. GSM8K is sensitive to distribution shift because reasoning tasks require precise action likelihoods, and accuracy directly reflects policy improvement. If MTOPD outperforms V‑trace and uncorrected baselines, and the ablation degrades, the claim is supported.

### Expected outcome and causal chain

**vs. Synchronous OPD** — On a stale rollout where the student’s old policy assigns low probability to the correct reasoning step, synchronous OPD (using fresh on‑policy data) correctly updates because it has no distribution mismatch. Our method instead computes a corrected gradient via the geometric mean teacher ratio; since teachers remain accurate, the correction upweights the step, and the gradient multiplier log(ρ) (for forward KL) or -1/ρ (for reverse KL) applies the right magnitude. We expect accuracy within 2–3% of synchronous OPD, showing near‑on‑policy effectiveness despite staleness.

**vs. Stale OPD without correction** — On a stale rollout where the student’s old policy is outdated (e.g., after a parameter update that changed action preferences), uncorrected OPD treats the old data as on‑policy and applies a gradient away from the true target, causing oscillations or divergence. Our method reweights with the geometric mean teacher density ratio, aligning the gradient with the current teachers’ consensus. We expect a clear accuracy gap (e.g., +10–15%) on the stale condition, with uncorrected OPD plateauing or dropping over time while MTOPD steadily improves.

**vs. V-trace with clipped IS** — On a stale rollout where teachers disagree (e.g., one teacher prefers a reasoning step another dislikes), V‑trace uses a clipped importance weight from a single teacher (or average in multi‑teacher case), which can be unstable or biased. Our method uses the geometric mean teacher density ratio, which is a proper density and yields a smooth gradient via the f‑divergence multiplier; it handles disagreement by weighting each teacher equally in log‑space. We expect MTOPD to be more robust, with lower variance and faster convergence, showing a consistent 2–5% higher accuracy on sets with high teacher disagreement.

**vs. Importance-weighted average of teacher gradients** — This baseline averages individual teacher gradients weighted by their importance ratio ρ_i. When teachers disagree, the arithmetic average may point in a direction that does not correspond to any valid divergence; our geometric mean target defines a consistent f-divergence objective. We expect MTOPD to achieve 1–3% higher accuracy and lower gradient variance.

**Ablation: arithmetic mean teachers** — Replacing geometric mean with arithmetic mean in ρ and using the same f-divergence multiplier (log ρ) will produce a gradient direction that minimizes D_KL(π_s || mixture of teachers), which differs from our claimed target. This ablation will likely degrade performance (2–5% lower accuracy) because the arithmetic mean target is not the barycenter for f-divergence minimization.

### What would falsify this idea
If MTOPD’s gain over V‑trace is uniform across reasoning steps rather than concentrated on steps where teachers disagree or where the stale policy is particularly misaligned, then the geometric mean and f‑divergence multiplier are not addressing the predicted failure mode, and the claim would be falsified.

## References

1. AsyncOPD: How Stale Can On-Policy Distillation Be?
2. AReaL: A Large-Scale Asynchronous Reinforcement Learning System for Language Reasoning
3. Prosperity before Collapse: How Far Can Off-Policy RL Reach with Stale Data on LLMs?
4. HybridFlow: A Flexible and Efficient RLHF Framework
5. Asynchronous RLHF: Faster and More Efficient Off-Policy RL for Language Models
6. GEAR: A GPU-Centric Experience Replay System for Large Reinforcement Learning Models
7. DeepSpeed-Chat: Easy, Fast and Affordable RLHF Training of ChatGPT-like Models at All Scales
8. On Realization of Intelligent Decision-Making in the Real World: A Foundation Decision Model Perspective
