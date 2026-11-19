# Cross-Workload Identifiability of Per-Operator Effective Limits for NPU Design-Space Critics

## Motivation

CriticNPU's Roofline-style critic calibrates per-operator-type effective limits from scalar simulator observations collected on a single prefill/decode workload regime, so the limits it reports are entangled with that regime's scheduling and memory-access mix rather than being a property of the NPU configuration. This is a structural identifiability problem, not a data-volume problem: with observations from one workload mix, per-operator limits and workload-dependent scheduling effects are confounded, so the critic's bottleneck attribution silently absorbs the calibration workload and cannot be trusted for a held-out regime without re-identification. The same meta-level failure recurs across the Idea Cards: CATransformers and the multi-objective differentiable NAS work both fit device/objective-conditioned models whose parameters are only meaningful for the workload-and-objective distribution seen during search, with no test that the learned conditioning is stable under a distribution shift in the evaluation regime.

## Key Insight

Per-operator-type effective limits become identifiable as configuration-intrinsic quantities exactly when the calibration design spans multiple workload regimes whose operator-mix vectors are sufficiently non-collinear, because then the shared limit vector is the unique component that predicts every held-out regime while workload-specific scheduling residuals cannot be absorbed into it.

## Method

**Name and I/O.** We propose XCAL (Cross-workload Identifiable Calibration), a drop-in replacement for CriticNPU's online per-operator calibration. Inputs: a set of simulated NPU configurations, each with architecture parameters; simulator traces grouped into K=4 workload regimes (prefill-long, prefill-short, decode-batch-1, decode-batch-16); each trace yields per-operator-type runtimes and the regime's operator-mix vector. Outputs: one workload-invariant limit vector `L` per configuration (compute and memory effective limits per operator type), a per-regime scheduling residual `s_k`, an identifiability score `rho` that gates whether `L` may be used for bottleneck attribution on a new regime, and a decomposition of held-out error into L-attributable and s-attributable components.

**Load-bearing assumption (stated explicitly).** XCAL assumes that per-operator runtime decomposes additively as `r[c,k,o] = mix[k,o] * (L[c,o] + s[k,o])` with a regime-shared, configuration-intrinsic limit `L[c,o]` and a zero-mean regime-specific residual `s[k,o]`, so that the shared component is uniquely recoverable from a small set of non-collinear operator-mix regimes. This assumption is a modeling choice, not a physical law: Roofline-style performance models treat runtime as a max (or nonlinear overlap) of compute and memory bounds, so the effective per-operator limit is regime-dependent by construction (prefill compute-bound vs decode HBM-bound). XCAL therefore (i) verifies the assumption empirically via the rank/condition-number check and the synthetic ground-truth recovery experiment below, and (ii) reports the decomposition of held-out error so that a violation is visible rather than silently absorbed into `L`.

**How it works.**

```pseudocode
# Phase 0: mix-matrix rank/conditioning check (gate on design, not data volume)
M = [mix[k,o]]  # K x O matrix, K=4 regimes, O operator types
cond_M = condition_number(M)          # numpy.linalg.cond, 2-norm
rank_M = matrix_rank(M, tol=1e-6)     # numpy.linalg.matrix_rank
if rank_M < min(K, O) or cond_M > cond_max = 1e3:
    flag design as rank-deficient; report cond_M; L is not uniquely identified
    # fallback: drop the most collinear regime pair (smallest principal angle) and re-check

# Phase 1: multi-regime trace collection (replaces single-regime online calibration)
for config c in candidate pool:
    for regime k in K_regimes:            # K = 4 default
        trace[c,k] = simulator.run(c, regime=k, budget=B_trace)   # B_trace = 8 sims/regime

# Phase 2: joint latent-factor fit (one fit per configuration)
#   observed runtime r[c,k,o] for operator type o in regime k
#   model: r[c,k,o] = mix[k,o] * ( L[c,o] + s[k,o] ) + noise
#   L[c,o] = config-intrinsic limit, s[k,o] = workload scheduling residual
fit L[c,:], s[:,:] by constrained least squares (scipy.optimize.minimize, SLSQP):
    minimize  sum_{k,o} ( r[c,k,o] - mix[k,o]*(L[c,o]+s[k,o]) )^2
              + lambda * sum_{k,o} s[k,o]^2        # lambda = 0.1, shrink residuals
    subject to  L[c,o] > 0,  sum_k s[k,o] = 0      # zero-mean residual: L absorbs the shared component

# Phase 3: cross-workload identifiability test with error decomposition
for each held-out regime h (leave-one-regime-out):
    refit L on K-1 regimes, predict r[c,h,o]
    err_h = normalized prediction error on regime h
    # decomposition: err_h = err_L(h) + err_s(h)
    err_L(h) = normalized error of prediction using refit L with s_h set to its training-mean (0)
    err_s(h) = err_h - err_L(h)   # residual-attributable component
rho[c] = 1 - mean_h(err_h) / err_baseline_single_regime
rho_L[c] = 1 - mean_h(err_L(h)) / err_baseline_single_regime   # L-attributable identifiability

# Phase 4: constrained refit with drift penalty (only if rho[c] >= rho_min = 0.6)
refit L[c,:] with penalty  mu * sum_k || s[k,:] - s_bar[:] ||^2   # mu = 0.5, penalize per-workload drift
return L[c,:], rho[c], rho_L[c], per-regime bottleneck summary

# Phase 5: use in CriticNPU loop
if rho[c] >= rho_min and rho_L[c] >= rho_min:
    critic uses L[c,:] for pre-simulation filtering and post-sim attribution
else:
    flag configuration as non-identifiable; fall back to direct simulation ranking
```

**Why this design.** We chose a joint latent-factor model with a zero-mean residual constraint over per-regime independent fits because independent fits cannot distinguish a configuration-intrinsic limit from a regime-specific scheduling effect, accepting the cost that the model assumes additive separability of limit and residual. We chose the leave-one-regime-out identifiability score `rho` over a held-out random split because random splits leak the same regime's mix into training, accepting the cost that `rho` is pessimistic when only one regime is available. We chose a constrained refit with a drift penalty over unconstrained refitting because unconstrained refitting lets `L` drift toward whichever regime dominates the loss, accepting the cost of an extra hyperparameter `mu` that must be tuned per configuration family. We added the Phase 0 rank/conditioning check and the Phase 3 error decomposition because the additive model is only identified when the mix matrix is well-conditioned and when held-out error is attributable to `L` rather than to the regime-specific residual `s_h`; these checks make the load-bearing assumption testable rather than asserted. This differs from CriticNPU's online scalar calibration and from the hardware-conditioned hypernetwork of the multi-objective differentiable NAS work: a domain expert would not describe XCAL as a variant of either, because neither performs a cross-regime identifiability test nor separates a configuration-intrinsic limit from a workload residual.

**Why it measures what we claim.** The quantity `rho` measures cross-workload identifiability of the limit vector because the leave-one-regime-out refit is equivalent to asking whether `L` estimated without regime `h` predicts regime `h`; this equivalence assumes the held-out regime's operator-mix vector is not collinear with the training regimes' mix vectors and that the held-out error is attributable to `L` rather than to the regime-specific residual `s_h` (i.e., `s_h` is small or independently estimable), and it fails when all regimes share nearly the same mix, in which case `rho` reflects interpolation within one mix rather than transfer, or when a large stable regime-specific effect inflates `err_h` and deflates `rho` even though `L` is perfectly identified. The decomposition `err_h = err_L(h) + err_s(h)` makes this failure testable: `rho_L` isolates the L-attributable component, and a gap between `rho` and `rho_L` flags residual-dominated held-out error. The zero-mean residual constraint measures the configuration-intrinsic component because it forces `L` to carry only the regime-shared signal; this assumes scheduling effects are regime-specific and average out, and it fails when a scheduling effect is shared across all sampled regimes, in which case `L` absorbs it and overstates the intrinsic limit. The drift penalty `mu` measures stability of `L` under regime reweighting because it penalizes per-workload residual divergence; this assumes residual divergence is a proxy for limit instability, and it fails when a genuinely regime-specific effect is large but stable, in which case the penalty suppresses a real scheduling signal.

## Contribution

(1) XCAL, a calibration protocol that replaces single-regime online fitting with a joint latent-factor model plus a leave-one-regime-out identifiability score that separates configuration-intrinsic per-operator limits from workload-dependent scheduling residuals. (2) A design principle: per-operator effective limits are only attributable to an NPU configuration when the calibration regime set spans non-collinear operator mixes, and the identifiability score operationalizes when this condition holds. (3) An identifiability-gated interface for CriticNPU that flags non-identifiable configurations and falls back to direct simulation ranking, plus a released multi-regime trace protocol for NPU DSE.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | CriticNPU NPU-DSE traces, K=4 workload regimes (prefill-long, prefill-short, decode-batch-1, decode-batch-16) | Matches target task, provides per-regime operator mixes |
| Dataset (synthetic) | Synthetic ground-truth generator with known L and s, mix matrices spanning condition numbers 1 to 1e4 | Validates recovery mechanism against known truth |
| Primary metric | Sample efficiency to best-config (sims) | Directly tests the sample-efficiency claim |
| Secondary metric | rho and rho_L (identifiability scores) | Tests whether the gate tracks L-identifiability |
| Secondary metric | L recovery error (relative L2) on synthetic data vs. mix-matrix condition number | Tests the rank/conditioning mechanism |
| Baseline | CriticNPU online calibration | Tests whether cross-regime separation beats single-regime fit |
| Baseline | Independent per-regime fits | Tests whether joint latent model beats naive per-regime fitting |
| Baseline | Random search over configs | Lower bound on sample efficiency |
| Baseline | Off-the-shelf invariant-prediction estimator (anchor regression / IRM) on the same per-regime traces | Tests whether XCAL's zero-mean-residual constraint beats generic invariance |
| Ablation-of-ours | XCAL without identifiability gate (always use L) | Isolates contribution of the rho gate |
| Ablation-of-ours | XCAL without Phase 0 rank/conditioning check | Isolates contribution of the design gate |
| Ablation-of-ours | XCAL without error decomposition (rho only, no rho_L) | Isolates contribution of the L-attributable score |

Hyperparameters: lambda = 0.1, mu = 0.5, rho_min = 0.6, cond_max = 1e3, B_trace = 8 sims/regime. Sensitivity reported over lambda in {0.01, 0.1, 1.0} and mu in {0.1, 0.5, 1.0} on a held-out configuration family.

Compute budget: per configuration, 4 regimes × 8 sims = 32 simulator calls for calibration, plus leave-one-regime-out refits (4 refits, each a small SLSQP solve over O operator types, <1 s CPU). For a candidate pool of 500 configurations, total calibration = 16,000 simulator calls; wall-clock dominated by the simulator, estimated at 2–6 GPU-hours on the CriticNPU simulator. Fallback simulator if CriticNPU traces are unavailable: an open Roofline-style analytical NPU model with configurable compute and HBM limits, calibrated to published NPU specs.

### Why this setup validates the claim

The central claim is that separating a configuration-intrinsic limit from a workload-specific scheduling residual improves sample efficiency in NPU DSE, and that a cross-workload identifiability score correctly gates when this separation is trustworthy. CriticNPU online calibration tests the first sub-claim directly: if joint cross-regime fitting beats its single-regime online calibration at equal simulator budget, the separation is doing real work. Independent per-regime fits test whether the joint latent-factor structure (zero-mean residual, shared L) is what matters, rather than merely collecting more regimes of data. The off-the-shelf invariant-prediction baseline tests whether XCAL's specific zero-mean-residual constraint is doing work beyond generic invariance, addressing the novelty concern. Random search anchors the sample-efficiency axis so gains are not artifacts of a weak reference. The no-gate and no-rank-check ablations isolate the rho gate and the Phase 0 design gate: if gains persist without them, the gates are redundant; if gains collapse on non-identifiable configurations, the gates are doing their claimed job. The synthetic ground-truth experiment validates the mechanism directly: recovery error of L as a function of mix-matrix condition number shows whether the rank/conditioning argument is correct, and the rho vs. rho_L comparison shows whether the gate tracks L-identifiability or is confounded by residual magnitude. Sample efficiency to best-config is the right primary metric because it is the quantity CriticNPU itself optimizes and the quantity the identifiability argument is meant to improve, without confounding with simulator wall-clock or downstream accuracy.

### Expected outcome and causal chain

**vs. CriticNPU online calibration** — On a case where prefill and decode regimes are limited by different resources (compute vs. HBM), the baseline produces a single calibrated limit that averages the two regimes because its online scalar calibration has no way to attribute a residual to scheduling versus architecture. Our method instead fits L jointly across regimes with a zero-mean residual, so the regime-specific component is absorbed by s_k rather than contaminating L. We expect a visible gap in sims-to-best-config concentrated on configurations where prefill and decode bottlenecks diverge, with near-parity on configurations where one resource dominates both regimes.

**vs. Independent per-regime fits** — On a case where a configuration's true limit is stable but each regime's trace is short (8 sims), the baseline produces noisy per-regime limits that disagree, because independent fits cannot share statistical strength and cannot distinguish limit from scheduling effect. Our method instead pools across regimes through the shared L with a shrinkage penalty on s, so the limit estimate is stabilized by the cross-regime constraint. We expect a gap that widens as per-regime trace budget shrinks, and narrows toward parity when traces are long enough that independent fits are well-determined.

**vs. Random search over configs** — On a case where the candidate pool is large and most configurations are far from optimal, the baseline wastes simulator budget on weak candidates uniformly, because it has no mechanism to filter before simulation. Our method instead uses L for pre-simulation filtering and bottleneck attribution, so weak candidates are rejected before they consume a trace. We expect a large gap in sims-to-best-config on large pools, shrinking as the pool size approaches the budget, consistent with filtering doing most of the work.

**vs. Off-the-shelf invariant-prediction estimator (anchor regression / IRM)** — On the same per-regime traces, the generic estimator enforces invariance without the zero-mean-residual identification convention or the rank/conditioning check, so it can select a solution that is invariant but not the configuration-intrinsic limit when the mix matrix is ill-conditioned. Our method instead combines the zero-mean constraint with the Phase 0 rank check and the rho_L decomposition, so it reports when the limit is not identified rather than returning an arbitrary invariant solution. We expect XCAL to match or beat the generic estimator on well-conditioned mixes and to win by a larger margin on configurations where the mix matrix condition number exceeds 1e3, where the generic estimator's invariance is not sufficient for identification.

**vs. Synthetic ground truth** — On synthetic data with known L and s, we expect relative L2 recovery error to increase monotonically with mix-matrix condition number, with a sharp rise beyond cond_M ≈ 1e3, confirming the rank/conditioning mechanism. We expect rho_L to track recovery error more closely than rho, confirming that the decomposition isolates L-attributable error.

### What would falsify this idea

If the gain over CriticNPU online calibration is uniform across all configurations rather than concentrated where prefill/decode bottlenecks diverge, or if the no-gate ablation matches XCAL's gains on configurations that rho flags as non-identifiable, or if synthetic L recovery error does not increase with mix-matrix condition number, then the cross-regime separation, the identifiability gate, and the rank/conditioning mechanism are not the causal drivers and the central claim is wrong.

## References

1. Mechanistic Critics for Sample-Efficient NPU Design-Space Exploration
