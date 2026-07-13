# Uncertainty-Adaptive Trust Region Policy Distillation

## Motivation

Existing policy distillation methods like Trust Region Policy Distillation (TRPD) and Simple Policy Optimization (SPO) enforce a fixed KL divergence constraint that is agnostic to the student's per-state uncertainty. This leads to instability in states where the student is uncertain about the teacher's action distribution, and unnecessarily slows learning in confident states, because the same constraint treats all states identically despite structural differences in student knowledge.

## Key Insight

The student's epistemic uncertainty about the teacher's action distribution provides a principled state-dependent signal that indicates where the trust region should be tightened (high uncertainty) or loosened (low uncertainty), enabling an automatic trade-off between stability and learning speed.

## Method

# UBPD (Uncertainty-Budgeted Policy Distillation)

(A) **What it is**: UBPD is an on-policy distillation algorithm that adapts the per-state KL divergence budget inversely proportional to the student's epistemic uncertainty. Inputs: teacher policy π_T, student policy π_S (2-layer MLP, hidden=256, ReLU activation), ensemble of K=5 student networks {π_S^i}_{i=1}^K (each same architecture with independent initializations), base budget δ=0.01, scaling coefficient α=1.0, penalty coefficient λ=10. Outputs: updated student policy π_S.

(B) **How it works**: Pseudocode.
```python
# UBPD algorithm
num_iterations = 1000
for iteration in range(num_iterations):
    batch = sample_student_rollouts(batch_size=64)
    for s in batch:
        # compute epistemic uncertainty as ensemble variance
        probs = [π_S_i(s) for i in range(K)]  # action probabilities from each ensemble member, shape (K, action_dim)
        u_s = variance(probs, axis=0).mean()   # scalar per state, averaged over action dimensions
        δ_s = δ * exp(-α * u_s)                # adaptive budget, high uncertainty → small budget
        # compute KL divergence
        kl_s = KL(π_T(s) || π_S(s))
        # constraint: if kl_s > δ_s, apply penalty
        loss_s = kl_s + λ * max(0, kl_s - δ_s) # λ=10
    # average loss and update student (and optionally uncertainty estimator)
    loss = mean(loss_s)
    update(π_S, loss)  # Adam optimizer, lr=1e-4
```

(C) **Why this design**: We chose exponential inverse scaling over linear scaling (e.g., δ_s = δ - α*u_s) because exponential ensures δ_s remains positive and provides a smooth gradient for α; we chose an ensemble-based uncertainty estimator over dropout-based Monte Carlo because ensemble variance is more stable and less noisy across training, though it doubles computation; we chose a penalty-based constraint enforcement (Lagrangian style) over hard clipping (as in TRPD) because it allows soft violation when uncertainty is low, accelerating learning, but can lead to constraint violations early in training. The trade-off for each: exponential vs. linear—smooth but harder to tune the base; ensemble vs. dropout—accuracy vs. cost; penalty vs. clipping—flexibility vs. strictness.

(D) **Why it measures what we claim**: The computational quantity δ_s = δ * exp(-α * u_s) operationalizes the concept of 'adaptive constraint tightness' because u_s (epistemic uncertainty) measures the student's lack of knowledge about the teacher's actions under the assumption that ensemble variance reflects approximation error of the student's distribution; this assumption fails when the teacher itself is noisy (aleatoric uncertainty dominates), in which case u_s may reflect teacher stochasticity rather than student ignorance, causing the budget to be too tight in genuinely stochastic states and potentially slowing learning. The penalty term λ * max(0, kl_s - δ_s) operationalizes 'constraint enforcement' because it adds loss only when the budget is exceeded, assuming that exceeding the budget is harmful; this assumption fails if the budget is mis-scaled, causing unnecessary penalties that distort learning. The ensemble variance as uncertainty measure operationalizes 'epistemic uncertainty' because it captures disagreement among student hypotheses, assuming the ensemble is diverse enough; this assumption fails if ensemble members collapse to the same policy, yielding zero variance and a falsely confident budget. Additionally, the adaptive budget relies on the load-bearing assumption that U_s measures epistemic uncertainty about the teacher's action distribution, independent of aleatoric noise in the teacher. This assumption is tested by replacing ensemble variance with a heuristic (e.g., state value) and checking performance degradation.

## Contribution

(1) A novel algorithm, UBPD, that adapts the trust region radius per state based on student epistemic uncertainty, eliminating the need for manual tuning of a global constraint in policy distillation.
(2) The principle that epistemic uncertainty, measured via ensemble variance, provides a theoretically grounded and computationally tractable signal for state-dependent constraint adaptation, linking uncertainty estimation to optimization stability.
(3) A demonstration (via experiments) that UBPD improves sample efficiency and training stability compared to fixed-radius baselines (TRPD, SPO) across continuous control and reasoning tasks.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | MuJoCo continuous control tasks (HalfCheetah-v2, Walker2d-v2, Hopper-v2, Humanoid-v2) | Standard benchmark for distillation; Humanoid added for a more challenging domain. |
| Primary metric | Average return over 10 evaluation episodes | Measures policy performance directly. |
| Baseline 1 | On-Policy Distillation (OPD) | No KL constraint, baseline for comparison. |
| Baseline 2 | Trust Region Policy Distillation (TRPD) | Fixed budget constraint (δ=0.01). |
| Baseline 3 | UBPD with state-value heuristic (UBPD-SV) | Replaces ensemble variance with state-value magnitude (|V(s)|) as budget: δ_s = δ * exp(-α * |V(s)|). Tests if uncertainty estimation is critical. |
| Ablation-of-ours | UBPD with MC dropout instead of ensemble (UBPD-Dropout) | Uses Monte Carlo dropout with 10 forward passes to estimate uncertainty. Tests ensemble variance importance. |

Additionally, we report training time (wall-clock hours on a single GPU) and peak GPU memory usage (in GB) to assess feasibility.

### Why this setup validates the claim
This setup tests UBPD's central claim: adaptive constraint based on epistemic uncertainty improves distillation. MuJoCo tasks involve varying state-dependent uncertainty (e.g., near obstacles vs. open spaces), making them ideal for detecting adaptation benefits. OPD (no constraint) and TRPD (fixed budget) isolate the effect of adaptation vs. no constraint and vs. static constraint. The ablations (UBPD-SV, UBPD-Dropout) test whether the specific uncertainty estimator (ensemble variance) is crucial or whether any state-dependent proxy suffices. Average return directly reflects the student's learned policy quality; if UBPD's adaptation works, it should achieve higher returns than baselines, especially on tasks with high state-dependent uncertainty (e.g., Humanoid). Training time and memory usage quantify the computational overhead of the ensemble. This combination creates a falsifiable test: if UBPD does not outperform TRPD on high-uncertainty states, the adaptive mechanism fails; if UBPD-SV matches UBPD, the uncertainty estimator is redundant.

### Expected outcome and causal chain

**vs. On-Policy Distillation (OPD)** — On a case where the teacher has high variance in action probabilities (e.g., near a cliff in HalfCheetah), OPD forces the student to match the teacher's distribution exactly, risking catastrophic forgetting of safe actions because it has no budget constraint. Our method instead applies a tight budget when student uncertainty is low (safe states) and loosens it when uncertainty is high (cliff), preserving performance. We expect UBPD to show a noticeable gain on safety-critical subtasks (e.g., higher return in cliff episodes) while matching OPD on trivial ones.

**vs. Trust Region Policy Distillation (TRPD)** — On a case where the student has high epistemic uncertainty initially (e.g., early training on Humanoid), TRPD enforces a fixed budget (δ=0.01), either too tight (slowing learning) or too loose (causing destructive updates depending on task). Our method instead adapts the budget inversely to uncertainty, allowing larger steps when student is confident (low uncertainty) and smaller when uncertain. We expect UBPD to converge faster (higher return at early iterations, e.g., after 200K steps) and achieve similar or better final performance, with the gap largest in the early phase and on complex tasks (Humanoid).

**vs. UBPD with state-value heuristic (UBPD-SV)** — On a case where aleatoric noise is high (e.g., stochastic transition dynamics in Hopper), state-value may not correlate with epistemic uncertainty about the teacher's actions. UBPD (ensemble) should outperform UBPD-SV because ensemble variance directly measures student disagreement about the teacher's distribution. We expect UBPD to have higher final return, especially on tasks with high aleatoric uncertainty.

**vs. UBPD with MC dropout (ablation)** — On a case where the student ensemble collapses (e.g., identical networks), UBPD (ensemble) still provides low variance, so budget is high; MC dropout might still show high variance due to noise, leading to overly tight budgets. Our method instead uses ensemble variance which is more stable, preventing unnecessary budget tightening. We expect UBPD (ensemble) to maintain higher return than the dropout variant on tasks with noisy teacher distributions (e.g., Hopper), while parity on low-noise tasks (e.g., early HalfCheetah).

### What would falsify this idea
If UBPD does not outperform TRPD on high-uncertainty states (e.g., early training on Humanoid) and instead shows similar return across all states, the adaptive mechanism is not beneficial. Also, if the dropout ablation or state-value heuristic matches UBPD on all tasks, the ensemble design is unnecessary or the uncertainty estimation is not critical.

## References

1. Trust Region Policy Distillation
2. Qwen3 Technical Report
3. Simple Policy Optimization
4. Qwen2.5 Technical Report
5. Qwen2.5-Coder Technical Report
6. Code Llama: Open Foundation Models for Code
7. Magicoder: Empowering Code Generation with OSS-Instruct
8. Qwen Technical Report
