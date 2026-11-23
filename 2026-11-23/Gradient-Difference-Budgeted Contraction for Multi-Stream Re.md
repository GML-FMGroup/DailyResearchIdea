# Gradient-Difference-Budgeted Contraction for Multi-Stream Residual Mixing

## Motivation

Multi-stream residual architectures such as cmHC constrain stream-mixing matrices to the Birkhoff polytope and justify a fixed geometric contraction by claiming bounds on cumulative gradient difference across layers, but the contraction radius is chosen geometrically (how deep inside the polytope the matrices live) with no mechanism tying that radius to the quantity the bound actually concerns — the accumulated layer-to-layer gradient difference. The result is that the geometric-only justification in cmHC (and its hyper-connection ancestors) cannot distinguish an under-mixed regime that keeps streams nearly independent from an over-mixed regime that collapses stream distinguishability, because the contraction strength is never coupled to the gradient-difference budget it is supposed to control. This is the shared residual gap: geometry constrains mixing coefficients without a feedback channel from the gradient-difference bound those coefficients are claimed to satisfy.

## Key Insight

The cumulative gradient-difference bound across layers is a monotone function of the realized contraction depth, so replacing a fixed geometric contraction with a per-layer contraction radius that is the exact solution of a budget equation makes the bound self-tightening rather than merely asserted.

## Method

**(A) What it is.** Gradient-Difference-Budgeted Contraction (GDBC) is a per-layer schedule for the cmHC stream-mixing matrices: given a residual-stream state, the current doubly stochastic mixing matrix, and a running gradient-difference accumulator, it outputs a contraction radius `r_t` and a contracted mixing matrix `M_t` whose depth inside the Birkhoff polytope is set by solving a scalar budget equation, not by a fixed constant.

**(B) How it works.**

```pseudocode
# per layer t, given: stream states S_{t-1} (n streams x d), incoming mixing M_raw_t in Birkhoff polytope B_n,
# running accumulated gradient-difference estimate G_hat (scalar), budget B_max, base rate eta, floor r_min,
# EMA rate rho = 0.1, Jacobian-probe count k = 4, calibration interval C = 500 steps

1. Project M_raw_t onto B_n via Sinkhorn (exact doubly stochastic, 20 iters, tol 1e-6) -> M_t
2. Compute stream-difference proxy: D_t = mean_{i<j} || S_{t-1}[i] - S_{t-1}[j] ||_2
3. Compute Jacobian-weighted gradient-difference proxy:
     J_t = finite-difference estimate of module Jacobian on S_{t-1} using k=4 Hutchinson probes
     dG_t = || M_t - M_{t-1} ||_F * || J_t S_{t-1} ||_2        # layer-to-layer mixing change times Jacobian-weighted stream separation
4. Budget update: G_hat <- (1 - rho) * G_hat + rho * dG_t                 # rho = 0.1 EMA
5. Solve scalar budget equation for r_t in [r_min, 1] using the exact finite-horizon sum with current EMA rate:
       G_hat * (1 - (1 - r_t)^T) / r_t = B_max          =>   r_t = clip(1 - G_hat / B_max, r_min, 1)   # closed form when T large; otherwise solve numerically
6. Contract toward interior: M_t <- (1 - r_t) * M_t + r_t * U_n   # U_n = uniform doubly stochastic matrix (1/n ones)
7. Apply M_t to streams: S_t = M_t @ S_{t-1} + F_t(S_{t-1})   # F_t = layer module output
8. Every C=500 steps, recompute exact cumulative gradient difference on a sampled subset of 8 layers and log correlation with G_hat; if correlation < 0.5, freeze r_t at r_min and flag calibration failure
9. Return S_t, M_t, r_t, G_hat
```

The contraction radius `r_t` is the closed-form solution of a geometric-series budget: if the per-layer mixing change contributes `dG_t` and the contraction shrinks the future accumulation by factor `(1 - r_t)`, the steady-state accumulated gradient difference is `G_hat / (1 - r_t)`, and setting that equal to `B_max` yields the schedule. `B_max` is a single scalar budget hyperparameter; `r_min` guards against degenerate full contraction. The load-bearing assumption is explicit: `dG_t` is a faithful, monotone surrogate for the accumulated layer-to-layer gradient difference that the cmHC bound controls **only when the module Jacobian is approximately identity across adjacent layers** (the residual-path assumption). To make this assumption testable, step 3 replaces the Jacobian-free proxy with a Jacobian-weighted finite-difference estimate, and step 8 adds a periodic calibration check that correlates `G_hat` with a recomputed exact bound on a sampled subset of layers; if the correlation falls below 0.5, the method flags a calibration failure and freezes the radius at `r_min` rather than driving an unvalidated proxy.

**(C) Why this design.** We chose a closed-form budget equation over a learned or scheduled contraction radius because the cmHC bound is a geometric series in the contraction factor, so the radius that saturates a target budget is algebraically determined rather than something a controller must discover; the cost is that the schedule inherits the bound's own slack, so `B_max` must be tuned per depth. We chose an EMA gradient-difference accumulator (`rho = 0.1`) over an exact per-layer recomputation of the cumulative bound because the exact bound requires retaining all historical mixing matrices, which is memory-prohibitive at depth; the cost is a lag of roughly `1/rho` layers in responding to abrupt mixing changes. We chose the uniform matrix `U_n` as the contraction target over the polytope barycenter computed per-layer because `U_n` is exactly doubly stochastic and keeps the contraction an exact Sinkhorn-consistent operation; the cost is that contraction is isotropic and cannot preferentially preserve informative stream directions. A domain expert would not describe GDBC as a variant of cmHC: cmHC fixes the contraction depth geometrically and reports gradient-difference bounds as post-hoc analysis, whereas GDBC inverts the direction — the bound is the input and the contraction depth is the output.

**(D) Why it measures what we claim.** The EMA accumulator `G_hat` measures accumulated layer-to-layer gradient difference because the mixing-change-times-Jacobian-weighted-stream-separation product `dG_t = ||M_t - M_{t-1}||_F * ||J_t S_{t-1}||_2` is a first-order surrogate for the change in the gradient of the loss with respect to stream features under the assumption that the module Jacobian is approximately identity across adjacent layers (the residual-path assumption); this assumption fails when a layer's module has large Jacobian norm, in which case `G_hat` reflects mixing-geometry change rather than true gradient-difference accumulation. The contraction radius `r_t = 1 - G_hat/B_max` measures realized contraction depth relative to the budget because the geometric-series identity `sum_k (1-r)^k dG = dG/r` holds exactly when the per-layer contribution is stationary; this assumption fails under non-stationary training (e.g., learning-rate warmup), in which case `r_t` reflects a stale budget rather than the current accumulation rate. The stream-difference proxy `D_t` measures residual-stream distinguishability because mean pairwise L2 distance is monotone in the subspace separation the cmHC bound controls under the assumption of bounded stream norms; this assumption fails when one stream dominates in norm, in which case `D_t` reflects norm imbalance rather than directional separation. The calibration check in step 8 directly tests the first assumption by correlating `G_hat` with a recomputed exact cumulative gradient difference on a sampled subset of layers; a correlation below 0.5 triggers a calibration-failure flag and freezes `r_t` at `r_min`.

## Contribution

(1) A gradient-difference-budgeted contraction schedule (GDBC) that replaces cmHC's fixed geometric contraction radius with the closed-form solution of a scalar budget equation coupling contraction strength to a running gradient-difference accumulator. (2) A design principle: contraction depth in multi-stream residual mixing should be a function of the realized gradient-difference accumulation rather than a geometric hyperparameter, which separates the under-mixing and over-mixing regimes by construction. (3) An open reference implementation of the budget-solved contraction layer compatible with Sinkhorn-projected doubly stochastic mixing.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | Deep Transformer LM pre-training (depth-scaled: 12, 24, 48 layers; fixed token budget 10B tokens; 3 seeds) | Depth is where mixing accumulation compounds |
| Primary metric | Downstream task accuracy at fixed compute (average over 5 tasks: HellaSwag, PIQA, WinoGrande, ARC-easy, LAMBADA) | Detects budget-saturated mixing benefit, not raw loss |
| Baseline | Fixed-radius cmHC (radius = 0.1, 0.3, 0.5; best per depth reported) | Tests whether adaptive radius beats constant depth |
| Baseline | Standard residual connections | Tests whether stream mixing helps at all |
| Baseline | Orthogonal residual update | Tests stream-separation mechanism against alternative |
| Baseline | Learned adaptive-radius controller (2-layer MLP, hidden=64, GeLU, mapping G_hat to r_t, trained jointly) | Tests whether closed-form budget equation is not merely a reparameterization of adaptive-depth ideas |
| Baseline | Per-layer hand-tuned constant radius (grid search over 5 values per depth) | Tests whether the budget equation adds value over per-layer tuning |
| Ablation-of-ours | GDBC with `B_max` frozen at init value | Isolates the budget-feedback loop's contribution |
| Ablation-of-ours | GDBC with Jacobian-free proxy (original `dG_t = ||M_t - M_{t-1}||_F * D_t`) | Tests whether Jacobian weighting matters |
| Ablation-of-ours | GDBC with calibration check disabled | Tests whether the calibration guard changes behavior |
| Diagnostic | Correlation between `G_hat` and exact cumulative gradient difference (recomputed every 500 steps on 8 sampled layers) | Empirically grounds the surrogate's validity |
| Non-LLM testbed | Deep GNN stack (20 layers, GCNII-style) on Cora and PubMed | Tests whether budget-feedback loop transfers to oversmoothing regime |

### Why this setup validates the claim

The central claim is that the contraction radius should be *derived from* an accumulated gradient-difference budget rather than fixed geometrically, and that doing so improves deep-stack training where mixing accumulation compounds. Depth-scaled Transformer pre-training is the right testbed because the geometric-series accumulation that GDBC budgets is a depth phenomenon: at shallow depth the difference between a fixed and adaptive radius is small, so a gain there would not be diagnostic. Fixed-radius cmHC is the load-bearing baseline — it shares GDBC's entire mechanism except the feedback loop, so any gap isolates the budget equation. Standard residual connections test the weaker sub-claim that multi-stream mixing helps at all, and orthogonal updates test whether our gain is really about stream separation rather than just any mixing change. The learned adaptive-radius controller and per-layer hand-tuned constant radius test whether the closed-form budget equation is more than a reparameterization of existing adaptive-depth ideas. The frozen-`B_max` ablation is the falsification lever: if the feedback loop matters, freezing it should reproduce fixed-radius behavior; if it does not, GDBC's advantage is an artifact of the contraction itself. The Jacobian-free ablation and the correlation diagnostic directly test the load-bearing assumption that `G_hat` tracks true accumulated gradient difference. The GNN testbed checks whether the budget-feedback loop transfers beyond LLMs. Downstream accuracy at fixed compute is the right metric because GDBC trades a small per-step overhead for better-conditioned deep stacks, and that trade should show up as quality at equal budget rather than as raw loss.

### Expected outcome and causal chain

**vs. Fixed-radius cmHC** — On a deep stack where mixing matrices drift toward the Birkhoff polytope vertices over many layers, the fixed-radius baseline under-mixes in early layers (radius too small to prevent near-independent streams) and over-mixes late (radius too large, streams collapse toward uniform), because its single constant cannot track the accumulating stream difference. Our method instead raises `r_t` when `G_hat` grows and relaxes it when accumulation is stable, because the budget equation converts measured accumulation into the needed contraction. We expect a modest but consistent accuracy gap that widens with depth and is near-parity at shallow depth.

**vs. Standard residual connections** — On a depth where the single residual stream must carry both persistent features and per-layer updates, the baseline forces one additive path to do both, so representation collapse and gradient vanishing trade off (the seesaw effect) and late-layer updates are dominated by stream direction. Our method instead maintains multiple streams whose mixing is budget-controlled, so new representational directions survive. We expect a clear gap on the deepest configurations and near-parity on shallow ones.

**vs. Orthogonal residual update** — On a case where the module output is largely aligned with the input stream, the orthogonal baseline discards the aligned component wholesale, which can starve layers that legitimately need to reinforce the stream, because orthogonality is enforced unconditionally. Our method instead modulates how much streams mix according to accumulated gradient difference, preserving reinforcement when it is informative. We expect our gain to be concentrated where stream separation is high but reinforcement still matters, with the orthogonal baseline possibly ahead on pure feature-novelty subsets.

**vs. Learned adaptive-radius controller** — The MLP controller must discover the mapping from `G_hat` to `r_t` from scratch and may overfit to early training phases, whereas our closed-form budget equation is algebraically determined by the geometric-series identity. We expect GDBC to match or slightly exceed the controller at lower parameter count and to be more stable across seeds, with the controller potentially catching up at the deepest depth if given enough training.

**vs. Per-layer hand-tuned constant radius** — Hand-tuning per layer requires a grid search that scales linearly with depth and cannot adapt to non-stationary accumulation during training. We expect GDBC to outperform the best per-layer constant at depth 48, where the optimal radius shifts across layers and training phases, and to be near-parity at depth 12.

**vs. GDBC with Jacobian-free proxy** — If the Jacobian weighting matters, the Jacobian-free ablation should show lower correlation with the exact bound and worse downstream accuracy, especially at depth 48 where module Jacobians deviate most from identity. We expect a measurable gap that grows with depth.

**vs. GDBC with calibration check disabled** — Disabling the calibration guard should have little effect when the surrogate is valid but may cause instability if the correlation drops below 0.5. We expect near-parity on average but higher variance across seeds for the disabled version.

**vs. GNN testbed** — On the 20-layer GCNII stack, we expect the budget-feedback loop to transfer if the gradient-difference accumulation mechanism is architecture-agnostic; if it does not transfer, the claim is scoped to Transformer LMs.

### What would falsify this idea

If the frozen-`B_max` ablation matches full GDBC across all depths, the budget-feedback loop contributes nothing and the central claim is wrong. If GDBC's gain over fixed-radius cmHC is uniform across shallow and deep stacks rather than concentrated at depth, the predicted accumulation mechanism is not what is producing the improvement. If the correlation between `G_hat` and the exact cumulative gradient difference falls below 0.5 across most layers and training phases, the surrogate is invalid and the budget framing should be dropped.

## References

1. cmHC: Balancing Stream Mixing in Hyper-Connections via Geometric Contraction
