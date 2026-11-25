# Trust-Region Delta Distillation: Monotone Competence Retention for Multi-Teacher Medical Language Model Integration

## Motivation

Multi-teacher on-policy distillation (MOPD^2) as used in KMed-31B integrates four category-specific medical specialists into a single student by summing per-teacher on-policy deltas, but this unconstrained summation provides no invariant that the student's per-category competence stays at or above each specialist's level — cross-category gradient interference can silently regress one category while another improves, and the abstract reports only aggregate benchmark lifts that cannot expose such regressions. The root cause is structural: the distillation objective is a single scalar sum over categories, so the update direction is determined by the aggregate, not by any per-category feasibility condition. This is the same failure mode that recurs across medical model-integration work — MediConfusion shows that aggregate benchmark scores mask severe per-capability failure modes in medical MLLMs, and VILA-M3 argues that domain-specialist knowledge is lost when folded into a generalist without explicit retention mechanisms — yet no prior composition method enforces per-category non-regression as a constraint.

## Key Insight

Per-category non-regression under multi-teacher distillation is enforceable as a sequence of per-category trust-region projections onto the intersection of half-spaces defined by each specialist's reward minus a tolerance, because on-policy distillation deltas are additive in parameter space and each specialist's reward is locally linear in that delta — making the feasible set convex and the projection closed-form.

## Method

**What it is.** Trust-Region Delta Distillation (TRDD) is a drop-in replacement for the unconstrained delta-summation step of MOPD^2. Its inputs are the four category-specific specialists {π_k}, the current student π_θ, a per-category on-policy rollout buffer, and a tolerance vector ε = (ε_1,…,ε_4); its output is an updated student parameter θ' whose per-category reward R_k(θ') satisfies R_k(θ') ≥ R_k(π_k) − ε_k for every category k.

**Load-bearing assumption (stated explicitly).** TRDD assumes that each category's reward R_k is locally linear in the parameter delta within the trust-region radius ρ, so that the first-order surrogate R̂_k(θ+Δ) = R_k(θ) + g_k^T Δ is an accurate feasibility test and the closed-form half-space projection actually guarantees R_k(θ') ≥ R_k(π_k) − ε_k after the update. This assumption is not asserted as true; it is measured and enforced by the feasibility-restoring loop below, which re-evaluates the true per-category reward on held-out category-k rollouts after each delta and backtracks until every previously accepted constraint is re-verified. The per-category constraint-violation rate after the full update is reported as a first-class output.

**How it works.**

```pseudocode
Input: student θ, specialists {π_k}_{k=1..4}, tolerance ε_k, step size η, linearization radius ρ
       held-out per-category eval sets E_k (disjoint from rollout buffers D_k), max backtracking steps B=5, shrink factor β=0.5
Initialize θ' ← θ; accepted constraints C ← ∅
For each category k in 1..4:
    # 1. On-policy delta from specialist k (MOPD^2 teacher delta)
    Sample rollouts D_k ~ π_k on category-k prompts
    Compute per-token log-ratio delta_k(θ) = ∇_θ E_{D_k}[ log π_k(a|s) − log π_θ(a|s) ]
    # 2. First-order reward surrogate for category k
    R̂_k(θ + Δ) = R_k(θ) + g_k^T Δ,  where g_k = ∇_θ R_k(θ) estimated by REINFORCE on D_k
    # 3. Trust-region projection onto the half-space {Δ : R̂_k(θ+Δ) ≥ R_k(π_k) − ε_k}
    If R̂_k(θ + η·delta_k) ≥ R_k(π_k) − ε_k:
        Δ_k ← η · delta_k                                  # unconstrained step is feasible
    Else:
        slack = R_k(θ) − (R_k(π_k) − ε_k)
        Δ_k ← η·delta_k + ( (slack − g_k^T(η·delta_k)) / ||g_k||^2 ) · g_k
    # 4. Clip to linearization radius ρ (trust region validity)
    Δ_k ← clip(Δ_k, ||Δ_k|| ≤ ρ)
    # 5. Feasibility-restoring loop: re-evaluate true reward on held-out E_k and on all previously accepted categories
    θ_cand ← θ' + Δ_k
    For b in 1..B:
        violations ← { j ∈ C ∪ {k} : R_j(θ_cand) < R_j(π_j) − ε_j }   # R_j measured on E_j
        If violations = ∅: break
        Δ_k ← β · Δ_k                                   # shrink step
        θ_cand ← θ' + Δ_k
    # 6. Re-linearize g_k at θ_cand if the step left the trust region (||Δ_k|| > ρ)
    If ||Δ_k|| > ρ:
        Recompute g_k = ∇_θ R_k(θ_cand) on D_k and re-run steps 3–5 once
    θ' ← θ_cand; C ← C ∪ {k}
    Record per-category violation indicator 1[R_j(θ') < R_j(π_j) − ε_j] for all j ∈ C
Return θ', per-category constraint-violation rate over the update
```

Hyperparameters: tolerance ε_k (default 0.01 × R_k(π_k), i.e. 1% relative slack), step size η (default 1e-5), linearization radius ρ (default 0.02 × ||θ||), max backtracking steps B=5, shrink factor β=0.5, held-out eval set size |E_k| = 256 category-k prompts. R_k(π_k) is estimated on the same held-out E_k with the same reward model used for R_k(θ), reported with a 95% bootstrap confidence interval over E_k; slack is a valid signed distance only when the E_k distributions for specialist and student match, which is enforced by using the identical E_k for both.

**Why this design.** We chose a closed-form half-space projection over a learned gating or penalty-term approach because the per-category non-regression condition is a single linear inequality in the distillation delta, so the projection has an exact solution and adds no trainable parameters — accepting the cost that the projection is only first-order accurate and may need re-linearization if the step leaves the trust region, which is why the feasibility-restoring loop re-evaluates true reward and backtracks rather than trusting the surrogate. We chose to project each specialist's delta sequentially rather than jointly because the four categories are trained on disjoint prompt distributions with disjoint reward models, so the per-category gradients are computed on separate rollout buffers and joint projection would require a coupled quadratic program over four constraints whose solution is not closed-form — accepting the cost that sequential projection can violate an earlier category's constraint if a later delta has a large component along g_k, which is why step 5 re-verifies all previously accepted constraints and reports the measured violation rate. We chose a per-category relative tolerance ε_k = 1% of the specialist's reward over a fixed absolute tolerance because medical reward scales differ by orders of magnitude across categories (rubric scores vs. exact-match QA), so a fixed absolute tolerance would be vacuous for some categories and binding for others — accepting the cost that the tolerance must be re-set if reward normalization changes. This is not a variant of standard KL-regularized distillation (e.g., the on-policy distillation of Agarwal et al. or the MOPD^2 delta summation itself): those methods regularize toward a teacher in expectation over the whole task mixture and provide no per-category feasibility guarantee, whereas TRDD treats each specialist's reward as a hard half-space constraint and projects the update onto the intersection one constraint at a time, with the true post-update reward measured on held-out data rather than assumed.

**Why this measures what we claim.** The quantity R̂_k(θ+Δ) = R_k(θ) + g_k^T Δ measures per-category competence retention for category k because the first-order Taylor expansion of the category-k reward around θ is exact when the reward is locally linear in the parameter delta, which holds when the delta norm is below the trust-region radius ρ; this assumption fails when the reward surface is strongly curved (e.g., near a category boundary where the rubric switches from partial to full credit), in which case R̂_k reflects the linear extrapolation rather than the true post-update reward, and the clip to ρ bounds the extrapolation error — and the feasibility-restoring loop in step 5 detects exactly this failure by re-evaluating R_k on held-out E_k and backtracking until the true reward satisfies the constraint. The slack term slack = R_k(θ) − (R_k(π_k) − ε_k) measures how far the current student is from the non-regression boundary for category k because it is exactly the signed distance (in reward units) to the half-space boundary; this assumption fails when R_k(π_k) is estimated on a different prompt distribution than the student's on-policy buffer, in which case slack reflects distribution shift rather than true competence gap — which is why R_k(π_k) and R_k(θ) are both estimated on the identical held-out E_k with the same reward model. The projection direction g_k / ||g_k||^2 measures the cheapest parameter-space direction that restores category-k feasibility because it is the minimum-norm correction satisfying the linear constraint; this assumption fails when g_k is estimated from too few rollouts (high variance), in which case the projection direction is noisy and the constraint may be violated on re-evaluation — which is why the measured per-category constraint-violation rate after the full update is reported as a first-class output, and why g_k is re-linearized at θ_cand when the step leaves the trust region.

## Contribution

["A constrained multi-teacher distillation algorithm, TRDD, that replaces MOPD^2's unconstrained delta summation with per-category half-space trust-region projections, giving a closed-form per-update feasibility check for each specialist's competence floor.", 'A design principle for medical model integration: per-category non-regression should be enforced as a hard constraint at each distillation update rather than monitored post hoc on aggregate benchmarks, because aggregate scores can mask category-level regressions.', 'A per-category reward-surrogate estimator built from specialist on-policy rollouts that supplies the linearization gradient g_k needed for the projection, usable as a diagnostic of which categories are near their feasibility boundary during training.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | KMed-31B four-category medical task suite (QA, diagnosis prediction, clinical inquiry, rubric-scored dialogue) | Matches the category structure TRDD constrains |
| Dataset (second setting) | Public rubric-scored medical dialogue set (e.g., HealthBench-style) combined with a radiology VQA specialist | Tests TRDD on a second multi-specialist integration setting |
| Primary metric | Worst-category reward retention vs. specialists, plus per-category constraint-violation rate after the full update | Directly tests the non-regression guarantee and the measured trust region |
| Secondary metric | Fraction of categories meeting a clinical safety floor (R_k ≥ R_k(π_k) − ε_k) | Deployment-relevant, reusable evaluation protocol |
| Baseline | MOPD^2 unconstrained delta summation | Tests whether constraints beat unconstrained aggregation |
| Baseline | KL-regularized on-policy distillation | Tests whether hard half-space projection beats soft regularization |
| Baseline | Per-category specialist (upper bound) | Anchors what feasibility means per category |
| Baseline | PCGrad-style gradient surgery | Tests whether projection onto per-category half-spaces beats pairwise gradient conflict resolution |
| Baseline | Task-arithmetic projection | Tests whether sequential half-space projection is not subsumed by parameter-space task merging |
| Ablation-of-ours | TRDD without trust-region clip (ρ→∞) | Isolates the linearization-radius contribution |
| Ablation-of-ours | TRDD without feasibility-restoring loop (single-pass projection) | Isolates the measured-trust-region contribution and quantifies the sequential-projection gap |
| Ablation-of-ours | Joint-projection QP variant (solved numerically) | Shows the sequential choice is a genuine design decision rather than a simplification |
| Ablation-of-ours | TRDD with fixed absolute tolerance (ε_k = 0.01 absolute) | Tests the relative-tolerance design choice |

### Why this setup validates the claim

The central claim is that sequential half-space projection with a measured trust region yields per-category non-regression that unconstrained delta summation and KL regularization cannot deliver. MOPD^2 is the right baseline because it is the exact method TRDD replaces; if TRDD's constraints matter, the gap must appear as reduced worst-category degradation on the categories where MOPD^2's summed deltas interfere. KL-regularized on-policy distillation tests a distinct sub-claim: that a hard feasibility boundary beats an expectation-only penalty, so its gap should concentrate on categories whose reward scale makes the KL weight mistuned. PCGrad-style gradient surgery and task-arithmetic projection test whether the per-category half-space projection is subsumed by existing constrained or merging methods; if TRDD's gain over these is uniform rather than concentrated on the categories predicted to suffer cross-category interference, the projection is not acting through the claimed feasibility mechanism. The specialist upper bound anchors the feasibility target, so a reader can see whether TRDD actually lands within the 1% slack rather than merely beating baselines. Worst-category retention is the correct primary metric because the claim is a per-category floor, not average improvement; averaging would hide exactly the cross-category interference the method targets. The per-category constraint-violation rate is a first-class output because it measures whether the first-order surrogate and sequential projection actually certify non-regression on held-out data, closing the method-to-retention causal chain. The no-clip and no-feasibility-loop ablations isolate whether the trust-region radius and the measured backtracking, rather than projection alone, are what bound extrapolation error and sequential-projection violations. The joint-projection QP variant tests whether the sequential choice is a genuine design decision rather than a simplification.

### Expected outcome and causal chain

**vs. MOPD^2 unconstrained delta summation** — On a case where a clinical-inquiry delta has a large component opposing the QA category's gradient, MOPD^2 sums all four deltas and the QA reward drops below its specialist level because nothing checks feasibility per category. Our method projects each delta onto its category's half-space before summing and re-verifies all previously accepted constraints on held-out data, so the QA constraint is restored by the minimum-norm correction and backtracking. We expect a visible worst-category gap favoring TRDD that concentrates on the two categories with opposing gradient directions, with near-parity on the aligned categories, and a lower per-category constraint-violation rate for TRDD.

**vs. KL-regularized on-policy distillation** — On a case where rubric-scored clinical inquiry and exact-match QA have reward scales differing by orders of magnitude, the KL baseline uses one global regularization weight, so it either over-constrains the small-scale category or lets the large-scale category drift because its penalty is dominated. Our method uses per-category relative tolerances, so each category's boundary is set in its own reward units. We expect TRDD to show retention near the specialist floor on both scale regimes, while the KL baseline shows a systematic shortfall on the category whose scale is mismatched to the weight.

**vs. Per-category specialist (upper bound)** — On a case where the student must serve all four categories from one parameter set, the specialist ensemble achieves each category's ceiling but cannot be deployed as a single model, which is the reference point rather than a competitor. Our method should approach this ceiling within the 1% slack on every category, so the observable signal is that TRDD's worst-category retention sits close to the specialist line while remaining a single student, and any large residual gap on one category flags that its tolerance or gradient estimate is binding.

**vs. PCGrad-style gradient surgery** — PCGrad resolves pairwise gradient conflicts by projecting conflicting components, but it provides no per-category reward floor and no measured post-update feasibility check. We expect TRDD to show higher worst-category retention and a lower constraint-violation rate on the categories where PCGrad's pairwise projections leave a residual conflict, with near-parity on categories without conflicts.

**vs. Task-arithmetic projection** — Task arithmetic merges specialist deltas by arithmetic combination and projection onto a task-vector subspace, but it does not enforce per-category non-regression. We expect TRDD to outperform task arithmetic on worst-category retention, especially on the category whose specialist delta has the largest opposing component, and to show a lower measured violation rate.

**vs. TRDD without trust-region clip (ρ→∞)** — Removing the clip lets the projection step exceed the linearization radius, so the first-order surrogate becomes inaccurate and the true reward may violate the constraint. We expect the no-clip ablation to show a higher per-category constraint-violation rate and worse worst-category retention, isolating the trust-region radius contribution.

**vs. TRDD without feasibility-restoring loop (single-pass projection)** — Removing the backtracking loop means later deltas can violate earlier constraints without re-verification. We expect the single-pass variant to show a measurable per-category constraint-violation rate after the full update, while full TRDD reduces it, quantifying the sequential-projection gap.

**vs. Joint-projection QP variant** — The QP variant solves all four constraints jointly and should achieve the lowest violation rate, but at higher compute cost and without a closed-form solution. We expect full TRDD to approach the QP variant's retention within a small margin, showing the sequential choice is a genuine design decision rather than a simplification.

**vs. TRDD with fixed absolute tolerance** — A fixed absolute tolerance is vacuous for small-scale categories and binding for large-scale ones. We expect the relative-tolerance TRDD to show more balanced per-category retention across reward scales, confirming the relative-tolerance design choice.

### What would falsify this idea

If TRDD's gain over MOPD^2 is uniform across all four categories rather than concentrated on the categories predicted to suffer cross-category interference, then the projection is not acting through the claimed feasibility mechanism. Likewise, if the no-clip ablation matches full TRDD, the trust-region radius is not doing the extrapolation-error work the argument assigns it, and if the single-pass variant matches full TRDD, the feasibility-restoring loop is not needed.

## References

1. KMed-31B Technical Report
