# Flux-Closure Regularization for Generative Urban Energy Maps

## Motivation

Generative urban energy models such as DySENSE condition only on static land use plus dynamic climate prompts, so the generated monthly building energy maps carry no guarantee of surface-atmosphere flux consistency; the model can produce spatially plausible but physically impossible energy fields whenever it is queried on morphologies outside its four training cities. The root cause is structural: DySENSE's conditional generator treats the energy map as an unconstrained image-like output, and no differentiable surface energy balance term enters training or inference, so land-use conditioning fully determines the output without any closure invariant. The same gap recurs across the field—UCformer shows that embedding physical priors improves urban climate emulation, yet that insight has not been transferred to generative energy synthesis, leaving energy maps decoupled from the fluxes they imply.

## Key Insight

A generated monthly energy map uniquely implies sensible and latent heat fluxes through a differentiable urban surface energy balance, so penalizing the residual of that balance imposes a closure invariant the generator cannot satisfy by memorizing land-use-to-energy mappings alone.

## Method

**What it is.** FluxCL (Flux-Closure Learning) augments a DySENSE-style conditional generator with a differentiable urban surface energy balance (SEB) head. Inputs are a static land-use map, a monthly climate prompt, and the generator's output energy map; outputs are a corrected energy map whose implied sensible/latent heat fluxes close the SEB residual within tolerance, plus a scalar closure residual reported per sample.

**Load-bearing assumption (stated explicitly).** FluxCL treats the generated monthly-mean building energy map as the anthropogenic flux term Q_anth in a surface energy balance whose remaining terms (net shortwave, net longwave, sensible, latent) are computable from land-use-derived morphology parameters and climate prompts alone. Driving the SEB residual to zero is therefore claimed to make the energy map physically admissible rather than merely forcing the generator to compensate for errors in the other, unvalidated terms. This assumption is operationalized and empirically bounded in Phase 2b below; it is not assumed away.

**How it works.**

```pseudocode
# Phase 1: generator (inherited DySENSE conditioning)
E_hat = G_theta(landuse, month_prompt, climate_prompt)   # W/m^2 monthly mean

# Phase 2: differentiable SEB head (physics-guided closure)
# landuse -> per-pixel morphology params: albedo a, roughness z0, building fraction bf, vegetation frac vf
# These are produced by a 2-layer MLP (hidden=128, GeLU) applied per pixel to the 6 land-use class probabilities
# plus normalized building height and vegetation index; output layer is sigmoid-scaled to physical ranges
# a in [0.05,0.40], z0 in [0.01,2.0] m, bf in [0,1], vf in [0,1].
Q_sw  = (1 - a) * S_down(month_prompt, climate_prompt)          # net shortwave
Q_lw  = eps * (L_down - sigma * T_surf^4)                        # net longwave
H     = rho*cp * (T_surf - T_air) / r_a(z0, bf)                 # sensible
LE    = rho*Lv * (q_sat(T_surf) - q_air) / (r_a + r_s(vf))       # latent
Q_anth = E_hat                                                   # anthropogenic = generated energy
residual = Q_sw + Q_lw - H - LE + Q_anth                        # W/m^2, target 0

# Phase 2b: SEB-head calibration and residual decomposition (executed before Phase 3)
# (i) Fit the morphology MLP and the SEB head to an independent reference: eddy-covariance flux-tower
#     observations (H, LE) and reanalysis downwelling radiation (S_down, L_down) for the four cities.
#     Loss: L_cal = mean((H_pred - H_obs)^2) + mean((LE_pred - LE_obs)^2) + 0.1*mean((Q_sw_pred - Q_sw_obs)^2)
#     + 0.1*mean((Q_lw_pred - Q_lw_obs)^2), Adam lr=1e-3, 200 epochs, batch=256 pixels.
# (ii) Report per-term residual budget (Q_sw, Q_lw, H, LE) against the reference; terms whose
#     reference residual exceeds 20 W/m^2 are flagged as weakly constrained.
# (iii) Decompose the total residual into generator-attributable and SEB-model-attributable components:
#     residual_total = residual_gen + residual_seb, where residual_seb is the SEB head's own residual
#     against the reference (computed with E_hat replaced by observed Q_anth) and residual_gen is the
#     remainder. Only residual_gen is used as the training signal in Phase 3; residual_seb is reported
#     as a diagnostic and is not optimized.

# Phase 3: training regularizer
L = L_gen(E_hat, E_gt) + lambda_flux * mean(residual_gen^2) + lambda_smooth * TV(E_hat)
# lambda_flux = 0.1, lambda_smooth = 1e-3, T_surf solved by 5 Newton iterations per pixel
# (vectorized batched solver; Jacobian cached on a 64x64 coarse grid and bilinearly interpolated)

# Phase 4: post-hoc consistency projection (inference only)
E_proj = argmin_E ||E - E_hat||^2  s.t.  |residual(E)| <= tau   # tau = 5 W/m^2
# closed-form: E_proj = E_hat - residual(E_hat) projected onto feasible set via one Lagrange step
```

**Why this design.** We chose a differentiable SEB head solved by fixed Newton iterations over an implicit neural flux surrogate because the balance is nonlinear in surface temperature and a closed-form inverse does not exist, accepting the cost of extra per-pixel compute during training. We chose to apply the closure as a soft training regularizer plus a hard post-hoc projection rather than only one because the regularizer shapes the generator's output distribution while the projection guarantees feasibility at inference, accepting the cost that projection can shift outputs away from the learned distribution by up to tau. We chose to derive morphology parameters from land use rather than learn them jointly because tying them to the conditioning variable makes the closure term a function of the same input the generator sees, so the residual cannot be reduced by an unconstrained free parameter; this is the key difference from UCformer, which embeds physical priors as architectural inductive biases rather than as an explicit closure residual on the output, and a domain expert would not describe FluxCL as a variant of UCformer because the mechanism is a balance-equation residual on generated fields, not a physics-guided Transformer. We added Phase 2b because the residual is a single scalar equation with many weakly constrained inputs, and without calibration the generator could lower it by absorbing bias in Q_sw, Q_lw, H, and LE rather than by becoming physically correct; Phase 2b fixes the non-anthropogenic terms against an independent reference and isolates the generator-attributable component residual_gen, so the training signal cannot be satisfied by compensation alone.

**Why it measures what we claim.** The SEB residual measures physical closure of the generated energy map because the balance equation is exact under the assumption that monthly-mean fluxes are stationary and that land-use-derived morphology parameters are accurate; this assumption fails when sub-monthly variability or unmodeled anthropogenic sources dominate, in which case the residual reflects model mismatch rather than generator inconsistency. Phase 2b bounds this failure mode by reporting the SEB head's own residual against flux-tower and reanalysis data and by decomposing the total residual into generator-attributable and SEB-model-attributable components; the training signal uses only the generator-attributable component, so a generator that memorizes land-use-to-energy mappings cannot satisfy it by construction. The post-hoc projection distance measures distributional closure under novel morphologies because projecting a sample onto the feasible SEB set is equivalent to asking whether the generator already lies near the physically admissible manifold; this fails when the feasible set is multi-modal, in which case small projection distance can still hide large semantic errors. The out-of-distribution morphology stress test measures generalization because novel land-use configurations are drawn from a held-out morphology prior, assuming that held-out configurations are representative of unseen cities; this fails when the held-out prior is too narrow, in which case the test overstates generalization.

## Contribution

(1) FluxCL, a differentiable urban surface energy balance closure module that couples generated monthly building energy maps to sensible/latent heat flux consistency through a training regularizer and a post-hoc projection. (2) A design principle showing that tying morphology parameters to the generator's conditioning variable prevents the closure residual from being absorbed by free parameters, yielding a physically grounded consistency invariant. (3) An out-of-distribution morphology stress test protocol for evaluating generative urban energy models under novel land-use configurations.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | DySENSE four-city monthly energy maps | Aligned static morphology plus dynamic monthly energy targets |
| Calibration reference | Eddy-covariance flux-tower H/LE and reanalysis S_down/L_down for the four cities | Bounds SEB-head parameter error before closure is trusted |
| Primary metric | Generator-attributable SEB closure residual (W/m^2) | Isolates generator inconsistency from SEB-model mismatch |
| Secondary metric | Per-term residual budget (Q_sw, Q_lw, H, LE) vs. reference | Shows which terms dominate and whether parameters are in bounds |
| Baseline | DySENSE generator (unregularized) | Tests whether physics closure is inherited for free |
| Baseline | UCformer-style physics-prior model | Tests architecture-prior vs explicit residual mechanism |
| Baseline | Post-hoc projection only | Isolates value of training-time closure regularizer |
| Ablation-of-ours | FluxCL without SEB head | Confirms gain comes from closure, not capacity |
| Ablation-of-ours | FluxCL with uncalibrated SEB head (no Phase 2b) | Tests whether calibration is what makes the residual valid |

### Why this setup validates the claim

The central claim is that a differentiable SEB residual head, applied both as a training regularizer and an inference projection, yields generated energy maps that are physically admissible without sacrificing fidelity to observed monthly energy. The unregularized DySENSE generator tests the null: if closure were already implicit in the conditioning, its residual would be as low as ours and the SEB head would be redundant. The UCformer-style baseline tests the mechanism claim specifically — it embeds physics as an architectural prior rather than as an output-space balance equation, so any advantage of FluxCL on residual concentrated at high-anthropogenic or high-vegetation-fraction pixels isolates the residual mechanism rather than generic physics awareness. The projection-only baseline separates the two halves of our design: if the training regularizer contributes nothing, projection-only should match FluxCL, falsifying the claim that shaping the output distribution matters. The ablation removes the SEB head entirely to confirm that any gain is attributable to closure rather than added parameters. The uncalibrated-SEB ablation tests the load-bearing assumption directly: if FluxCL's advantage disappears when the SEB head is not calibrated against flux-tower and reanalysis data, then the residual was measuring agreement with an unvalidated parameterization rather than generator inconsistency, and the primary metric is not doing the work claimed. The calibration reference and per-term residual budget are included so the reader can see which terms dominate the residual and whether the morphology-derived parameters are within stated error bounds. Generator-attributable SEB closure residual is the right primary metric because it is the quantity the method explicitly optimizes after removing the SEB head's own error against the reference; fidelity is tracked secondarily to detect the predicted trade-off, so a residual win accompanied by catastrophic fidelity loss would not count as success.

### Expected outcome and causal chain

**vs. DySENSE generator (unregularized)** — On a case where a dense commercial land-use tile receives a winter climate prompt, the baseline produces an energy map whose implied anthropogenic flux is inconsistent with the morphology-implied sensible and latent fluxes because it never evaluates a surface energy balance, so its generator-attributable residual is dominated by unmodeled longwave and storage terms. Our method instead pushes the generated field toward balance through the residual penalty, so we expect a large residual gap on high-building-fraction tiles and near-parity on sparse low-density tiles, where the balance terms are small.

**vs. UCformer-style physics-prior model** — On a case where a novel land-use configuration combines high vegetation fraction with high building fraction, the baseline produces plausible-looking but physically inconsistent outputs because its physics enters as an architectural inductive bias that cannot enforce a per-sample balance equation. Our method instead computes an explicit residual on the output and projects onto the feasible set, so we expect FluxCL to show a residual advantage concentrated on the mixed-morphology subset while matching the baseline on homogeneous tiles.

**vs. Post-hoc projection only** — On a case where the generator's raw output is far from the feasible set, the baseline's one-step projection produces a feasible but distributionally shifted map because the generator was never shaped toward the manifold during training. Our method instead reduces the pre-projection residual through the regularizer, so we expect smaller projection distances and better fidelity retention for FluxCL, with the gap widening as the raw residual grows.

**vs. FluxCL with uncalibrated SEB head (no Phase 2b)** — On a case where the SEB head's morphology parameters are biased (e.g., albedo overestimated on high-vegetation tiles), the uncalibrated variant can lower the total residual by pushing E_hat to absorb the Q_sw bias rather than by becoming physically correct, so its generator-attributable residual is artificially low while its per-term budget against the reference is large. Our calibrated method instead fixes the non-anthropogenic terms against flux-tower and reanalysis data and optimizes only the generator-attributable component, so we expect the uncalibrated variant to show a smaller total residual but a larger generator-attributable residual and a larger per-term budget error, with the gap concentrated on tiles where the reference residual exceeds 20 W/m^2.

### What would falsify this idea

If FluxCL's generator-attributable residual gain over the unregularized generator is uniform across morphology subsets rather than concentrated where the predicted balance-term mismatch is largest, the closure mechanism is not doing the causal work claimed. Likewise, if projection-only matches FluxCL on both residual and fidelity, or if the uncalibrated-SEB ablation matches the calibrated FluxCL on generator-attributable residual, the training-time regularizer or the calibration step is unnecessary and the central design claim fails.

## References

1. DySENSE: Dynamic Spatiotemporal ENergy Synthesis and Emulation with Satellite Imagery
