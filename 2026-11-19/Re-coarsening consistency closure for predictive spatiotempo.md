# Re-coarsening consistency closure for predictive spatiotemporal weather downscaling

## Motivation

Across the downscaling literature the high-resolution product is treated as an output to be generated, while the coarse backbone state it must remain physically consistent with is only an input — no existing method makes cross-scale dynamical agreement an explicit training invariant. PROLA's fixed numerical scaffolds and lightweight residual heads correct the pretrained backbone locally but provide no mechanism tying residual-head outputs back to the backbone's multi-scale message passing, so a downscaled field can drift from the coarse node-edge state that supposedly generated it; OneForecast instead adds a densified regional grid and a neural nested grid as architectural add-ons, which mitigate boundary information loss but still leave consistency an emergent hope rather than an enforced property. The structural root cause is that the coarse-to-fine map and the fine-to-coarse pooling operator are never required to be mutually inverse on the backbone's own graph, so the residual head is free to produce high-resolution fields whose re-coarsened image diverges from the backbone state that conditioned it.

## Key Insight

Because the frozen backbone already owns a differentiable graph-pooling operator, re-coarsening the high-resolution prediction through that same operator and penalizing divergence from the backbone's coarse state turns cross-scale dynamical consistency into a trainable invariant defined in the backbone's own message-passing coordinates rather than in pixel space.

## Method

**What it is.** We introduce C3 (Cross-scale Consistency Closure), a training-time closure for PROLA-style predictive spatiotemporal weather downscaling. Inputs: a recent coarse state sequence, the frozen pretrained weather backbone with its multi-scale graph-pooling operators, PROLA-adapted weights, and the lightweight residual heads. Outputs: a high-resolution future field whose re-coarsened image, computed by the backbone's own pooling operator, matches the backbone's coarse node-edge state within a calibrated per-level tolerance that accounts for the pooling operator's irreducible information loss.

**Load-bearing assumption (stated explicitly).** C3 assumes that the backbone's graph-pooling operator P_l, applied to a high-resolution field, yields a coarse state that is directly comparable to the backbone's pretrained coarse node-edge states Z_l, so that penalizing the excess re-coarsening residual beyond a calibrated floor enforces genuine cross-scale dynamical consistency rather than penalizing legitimate fine-scale information that P_l was never designed to preserve. This assumption is not taken for granted: we calibrate the irreducible re-coarsening residual r_l = Pool(Y_ref) - Z_l on held-out high-resolution references Y_ref that are known-consistent, and penalize only the excess over E[||r_l||^2]. If P_l is not idempotent or not information-preserving for sub-grid structure, the calibration absorbs that intrinsic loss and the closure tests deviation beyond it.

**How it works.**

```pseudocode
# Frozen backbone B with multi-scale graph pooling P_1..P_L (level l -> l+1)
# PROLA-adapted weights W, residual heads H_spatial, H_temporal
# Inputs: coarse sequence X_coarse (t-k..t), lead time tau
# Calibration set: held-out high-resolution references Y_ref (N_cal = 512 samples)

1. Forward coarse pass (no grad on B):
   Z_0, messages_0 = B.forward(X_coarse, return_messages=True)
   # Z_l, messages_l for l = 1..L captured at every graph level
   Z_coarse = Z_L                      # coarsest node-edge state

2. Predict high-res trajectory:
   Y_hat = H_temporal(H_spatial(B.adapted_forward(X_coarse; W)), tau)

3. Re-coarsening closure (differentiable through P):
   Z_hat = Y_hat
   for l = L down to 1:
       Z_hat = P_l(Z_hat)              # backbone's OWN pooling operator
   # Z_hat now lives on the coarsest graph, same node set as Z_coarse

4. Calibrated scale-consistency loss:
   # Precompute per-level floor on calibration set:
   #   r_l = Pool_to_level(Y_ref, l) - Z_l.detach()
   #   floor_l = mean(||r_l||^2) over N_cal = 512 held-out samples
   L_cons = sum_l  w_l * max(0, || Pool_to_level(Z_hat, l) - Z_l.detach() ||^2 - floor_l)
   # w_l = 1/(1+l) geometric decay; Z_l detached (frozen target)

5. Edge-message consistency (optional term, same pass):
   L_edge = max(0, || EdgeAgg(Z_hat) - messages_L.detach() ||^2 - floor_edge)
   # floor_edge = mean(|| EdgeAgg(Y_ref) - messages_L.detach() ||^2) on calibration set

6. Total loss for residual-head training:
   L = L_pred + lambda * L_cons + lambda_e * L_edge
   # lambda = 0.1, lambda_e = 0.05, tuned on validation

7. Update only H_spatial, H_temporal (and PROLA low-rank factors);
   B and P_l stay frozen.
```

**Why this design.** We chose to reuse the backbone's own graph-pooling operator P_l as the re-coarsening map rather than training a separate learned coarsening network, because a learned inverse would only enforce consistency with itself, not with the backbone's multi-scale message passing, accepting the cost that P_l must be available and differentiable at every level. We chose to detach the coarse targets Z_l rather than backpropagate through the frozen backbone, because the closure should constrain the residual heads without perturbing the pretrained representation, accepting the cost that the target is a fixed snapshot rather than a jointly optimized quantity. We chose a per-level geometric decay w_l = 1/(1+l) rather than uniform weighting, because finest-level agreement is the hardest and noisiest to enforce and would dominate the gradient, accepting the cost that coarser-level consistency is enforced more weakly. We chose to add the closure as an auxiliary loss on top of PROLA's residual-head training rather than as a new architecture, because the diagnosis is an absent training invariant, not a missing module. We chose to calibrate the per-level floor floor_l on 512 held-out high-resolution references rather than demand exact inversion, because P_l is a learned aggregation that legitimately discards sub-grid structure, and penalizing only the excess over the irreducible residual isolates genuine inconsistency from information loss. A domain expert would not describe this as a variant of OneForecast's neural nested grid, because OneForecast adds a densified regional grid as an architectural component whereas C3 adds no parameters and instead imposes a differentiable invariant on the existing PROLA residual heads via the backbone's pooling operator.

**Why it measures what we claim.** `L_cons` measures cross-scale dynamical consistency because the backbone's pooling operator P_l is assumed to be the same operator that produced the coarse node-edge states during pretraining, so re-coarsening a consistent high-resolution field through P_l should reproduce those states up to the calibrated floor floor_l; this assumption fails when the high-resolution field contains fine-scale structure that P_l legitimately discards (e.g., orographic detail below the coarse grid spacing), in which case the calibrated `L_cons` reflects only the excess inconsistency beyond that information loss, and the floor_l term absorbs the intrinsic loss. `L_edge` measures message-level agreement because the edge aggregation is assumed to summarize the same node-edge interactions that condition the coarse forecast; this assumption fails when the residual head produces a field that matches nodes but reorganizes edges (e.g., shifted fronts), in which case `L_edge` flags a real dynamical mismatch rather than a numerical artifact, and the floor_edge term absorbs the baseline edge-aggregation loss. The geometric weighting w_l measures the claim that finest-level consistency is the binding constraint because finer levels are assumed to carry the downscaling signal; this assumption fails when the target resolution is only marginally finer than the coarsest level, in which case the weighting over-penalizes an uninformative scale, and the calibrated floor prevents that scale from dominating.

## Contribution

["C3, a training-time cross-scale consistency closure that re-coarsens high-resolution predictions through the frozen backbone's own graph-pooling operators and penalizes divergence from detached multi-scale node-edge states, requiring no new parameters.", "A design principle: cross-scale consistency in predictive spatiotemporal downscaling is best enforced in the backbone's message-passing coordinates via its native pooling operator rather than in pixel space via an auxiliary coarsening network.", 'A per-level geometric weighting scheme for the consistency loss that allocates penalty across graph levels according to the scale at which downscaling signal is carried.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | PSWD coarse-to-fine benchmark | Matches the task PROLA already targets |
| Primary metric | Calibrated re-coarsened closure error at finest level: mean over test set of max(0, ||Pool(Y_hat) - Z_L||^2 - floor_L), where floor_L is the per-level calibration floor from 512 held-out high-resolution references | Directly measures the invariant we impose, net of the pooling operator's irreducible information loss |
| Secondary metric | Decomposition of closure residual into P_l-information-loss component (measured on ground-truth high-res fields: mean ||Pool(Y_ref) - Z_L||^2) and genuine-inconsistency component (excess over floor_L) | Verifies that the primary metric measures the claimed invariant rather than an unstated proxy |
| Baseline | PROLA residual heads, no closure | Isolates the effect of the added invariant |
| Baseline | Learned-coarsening consistency (trained inverse) | Tests our claim that backbone pooling is essential |
| Baseline | Uniform-weight closure | Tests the geometric-decay design choice |
| Baseline | Cycle-consistency with separately trained inverse AND multi-scale skip-consistency loss (U-Net-style) | Tests that the distinguishing element is specifically the backbone's pretrained P_l coordinates, not consistency per se |
| Ablation-of-ours | C3 without L_edge term | Isolates the edge-message contribution |
| Ablation-of-ours | C3 with hard equality target (no calibration floor) | Tests whether the calibrated floor is necessary to avoid over-smoothing |

### Why this setup validates the claim

The central claim is that the missing ingredient in PROLA-style predictive spatiotemporal downscaling is a training invariant tying the high-resolution output back to the backbone's own coarse multi-scale state, and that imposing it via the backbone's frozen pooling operator improves high-resolution trajectory fidelity. The no-closure PROLA baseline tests the sub-claim that the invariant is absent and needed: if PROLA already implicitly satisfies closure, C3 gains nothing. The learned-coarsening baseline tests the sub-claim that reuse of the backbone's own P_l matters rather than any differentiable re-coarsening: a self-consistent learned inverse should satisfy its own closure trivially but fail to align with the backbone's pretrained multi-scale states, so it should show weaker gains. The cycle-consistency baseline with a separately trained inverse and multi-scale skip-consistency loss tests whether consistency per se, rather than consistency in the backbone's pretrained P_l coordinates, is sufficient; if C3 matches this baseline, the core commitment to backbone pooling is not what produces the effect. The uniform-weight baseline tests whether the geometric decay is doing real work rather than being cosmetic. The L_edge ablation tests whether message-level agreement adds anything beyond node-state agreement. The hard-equality ablation tests whether the calibrated floor is necessary to prevent the closure from biasing toward over-smoothed fields. The primary metric — calibrated re-coarsened closure error at the finest level, plus its correlation with downstream high-resolution field error — is the right detector because it is exactly the quantity the loss constrains, net of the pooling operator's irreducible information loss; the secondary decomposition into information-loss and genuine-inconsistency components verifies that the primary metric measures the claimed invariant rather than an unstated proxy. If the mechanism works, the genuine-inconsistency component should drop and high-resolution fidelity should track it, whereas a metric unrelated to the invariant could improve for reasons uncoupled from our causal story.

### Expected outcome and causal chain

**vs. PROLA residual heads, no closure** — On a case where the coarse sequence contains a developing front and the residual heads sharpen it into fine structure, the no-closure baseline produces a high-resolution field whose re-coarsening drifts from the backbone's coarse state because nothing during training penalizes that drift. Our method instead penalizes the excess re-coarsening mismatch beyond the calibrated floor through P_l, so the residual heads are pushed toward fields whose coarse projection stays anchored, and we expect a noticeable drop in calibrated finest-level closure error concentrated on frontogenesis cases, with near-parity on quiescent smooth-flow cases.

**vs. Learned-coarsening consistency** — On a case with orographic fine structure near the target region boundary, the learned-coarsening baseline produces a field consistent with its own trained inverse but misaligned with the backbone's pretrained coarse node-edge states because its inverse was optimized for self-consistency, not for matching the backbone's multi-scale representation. Our method instead re-coarsens through the backbone's own P_l, so the target is the pretrained coarse state itself (net of the calibrated floor), and we expect a larger gain than the learned-coarsening variant specifically on boundary-adjacent and multi-scale-rich subsets, with the gap shrinking on smooth interiors.

**vs. Cycle-consistency with separately trained inverse and multi-scale skip-consistency loss** — On a case where the high-resolution field contains fine-scale structure that a U-Net-style skip-consistency term would preserve, the cycle-consistency baseline enforces agreement in its own learned coordinates and may match C3 on smooth cases. Our method instead enforces agreement in the backbone's pretrained P_l coordinates, so we expect C3 to outperform this baseline specifically on cases where the backbone's multi-scale message passing carries dynamical information that a separately trained inverse does not capture (e.g., frontogenesis and boundary-adjacent subsets), with parity on cases where consistency per se is sufficient.

**vs. Uniform-weight closure** — On a case where the target resolution is only marginally finer than the coarsest level, the uniform-weight baseline produces gradients dominated by an uninformative coarse scale because every level is weighted equally, diluting the binding finest-level constraint. Our method instead decays weight geometrically so the finest level dominates, and we expect a modest but consistent advantage on marginal-refinement cases and parity on strongly-refined cases, which is the pattern the weighting is designed to produce.

**vs. C3 without L_edge term** — On a case where the residual head produces a field that matches coarse node states but reorganizes edges (e.g., a shifted front), the no-L_edge variant cannot detect the edge-level mismatch because it only constrains node-state agreement. Our method adds the edge-message consistency term, so we expect a larger gain on cases with edge reorganization and near-parity on cases where node and edge agreement co-vary.

**vs. C3 with hard equality target (no calibration floor)** — On a case with orographic fine structure below the coarse grid spacing, the hard-equality variant penalizes all re-coarsening residual including the pooling operator's legitimate information loss, biasing the residual heads toward over-smoothed fields. Our method instead subtracts the calibrated floor floor_l, so we expect the hard-equality variant to show higher fine-scale error (over-smoothing) and C3 to retain fine-scale fidelity while achieving comparable genuine-inconsistency reduction.

### What would falsify this idea

If the C3 gain over no-closure PROLA is uniform across subsets rather than concentrated on the frontogenesis, boundary-adjacent, and marginal-refinement cases where the predicted failure modes occur, the central claim that the closure addresses a specific cross-scale inconsistency is wrong. Equivalently, if the learned-coarsening baseline or the cycle-consistency baseline matches C3's gains, then reuse of the backbone's own pooling operator — the design's core commitment — is not what produces the effect.

## References

1. PROLA: Principal-Orthogonal Low-rank Adaptation for Predictive Spatiotemporal Weather Downscaling
