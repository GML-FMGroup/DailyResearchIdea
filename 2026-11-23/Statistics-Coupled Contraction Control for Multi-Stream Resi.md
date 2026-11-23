# Statistics-Coupled Contraction Control for Multi-Stream Residual Mixing

## Motivation

Multi-stream residual generalizations such as cmHC constrain stream-mixing matrices to the Birkhoff polytope and contract the polytope toward its interior to avoid under-mixing, but the contraction is applied as a fixed geometric schedule that presumes residual-stream statistics stay stationary across depth. Because real residual distributions shift with depth (as documented in the oversmoothing analysis of residual GNNs, E5), a static contraction radius can be too aggressive in layers where streams are already well-mixed and too weak where they remain near vertices, so the mixing behavior the method claims to control is decoupled from the quantities that actually govern it. This is the meta-level problem: residual-stream geometry is treated as a fixed prior rather than an observed, layer-varying quantity.

## Key Insight

The contraction radius can be made a closed-form function of the layer's own observed stream-difference and gradient-difference statistics, so that the polytope geometry is always derived from the current residual distribution rather than assumed stationary.

## Method

**What it is.** Statistics-Coupled Contraction (SCC) is a per-layer controller that replaces cmHC's fixed contraction radius with a radius computed in closed form from running estimates of residual-stream variance and gradient-difference accumulation, producing layer-specific doubly stochastic mixing matrices that remain on the Birkhoff polytope. The controller's load-bearing assumption, stated plainly: the normalized pairwise stream difference measured at the mixing-matrix output (before the per-stream nonlinearity) is a monotone proxy for the distance of the mixing matrix P_l from the Birkhoff barycenter, so that a radius rho_l = 1 - eta*(s_l+g_l) controls the mixing-matrix geometry it claims to control. This assumption is operationalized by measuring d_l on the mixing-matrix output h_l^mix = P_l^c h_{l-1} (i.e., before the per-stream MLP), not on the post-MLP stream h_l, and is calibrated against the directly computable distance of P_l^c from the barycenter (see step 3b).

**How it works.**
```pseudocode
# Per layer l, per training step t
# State: running estimates s_l (stream-difference), g_l (gradient-difference)
# Hyperparameters: beta=0.9 (EMA decay), eta=0.5 (coupling gain),
#                  r_min=0.05, r_max=0.9 (radius clamp), eps=1e-8

1. Forward: compute candidate mixing matrix M_l = softmax_rows(W_l)  # W_l learnable
2. Doubly-stochastic projection: P_l = sinkhorn(M_l, iters=20)
3. Measure stream difference at layer l on the MIXING-MATRIX OUTPUT (pre-MLP):
     h_l^mix = P_l h_{l-1}                      # output of mixing, before per-stream MLP
     d_l = mean_{i<j} || h_l^mix^(i) - h_l^mix^(j) ||^2 / (mean_i ||h_l^mix^(i)||^2 + eps)
3b. Calibration diagnostic (logged, not used in radius):
     delta_l = || P_l - J/K ||_F / || J/K ||_F    # Frobenius distance of P_l to barycenter
     lambda_min_l = min eigenvalue of P_l          # directly computable mixing-depth measure
     # Verify empirically that d_l is monotone in delta_l and in (1 - lambda_min_l)
     # across depth; if the monotonicity fails at a layer, that layer is flagged
     # in the diagnostic log and its rho_l is reported alongside delta_l.
4. Measure gradient-difference accumulation (backward pass):
     q_l = || grad_h_l^(i) - grad_h_l^(j) ||^2 / (||grad_h_l||^2 + eps)
     averaged over stream pairs (i,j)
5. EMA update: s_l <- beta*s_l + (1-beta)*d_l ;  g_l <- beta*g_l + (1-beta)*q_l
6. Closed-form radius: rho_l = clamp( 1 - eta * (s_l + g_l), r_min, r_max )
7. Contract: P_l^c = (1 - rho_l) * P_l + rho_l * (J / K)   # J all-ones, K streams
8. Use P_l^c as the stream-mixing residual matrix in the forward path
9. Log rho_l, d_l, q_l, delta_l, lambda_min_l per layer for diagnostics
```

**Why this design.** We chose a closed-form radius `rho_l = 1 - eta*(s_l+g_l)` over a learned gating network because the quantity that must be controlled (distance of the mixing matrix from polytope vertices) is directly computable from the stream statistics, and a learned gate would reintroduce the very stationarity assumption we are removing by fitting a fixed mapping, accepting the cost that the coupling gain eta must be tuned rather than learned. We chose EMA running estimates over per-batch instantaneous statistics because per-batch estimates are noisy under small batch sizes and would make the radius oscillate, accepting the cost of a warm-up period during which the radius is uninformative. We chose Sinkhorn projection followed by explicit contraction over reparameterizing W_l to be doubly stochastic by construction because it keeps the contraction step auditable and separable from the learnable parameters, accepting the cost of 20 Sinkhorn iterations per layer. We chose to couple both stream-difference and gradient-difference rather than stream-difference alone because cmHC's own bounds (E9) treat mixing and gradient accumulation jointly, and a stream-only signal would let the radius drift when gradients are the binding constraint; a domain expert would not describe this as cmHC with a tuned schedule because the radius is a function of measured layer statistics, not a hyperparameter, and would not describe it as a learned router because there is no learned controller at all. We measure d_l on the pre-MLP mixing output rather than the post-MLP stream because the post-MLP stream is inflated by stream-specific features injected by the per-stream MLPs (as shown by orthogonal residual updates and LAuReL), which would make d_l reflect representational specialization rather than mixing deficiency; the pre-MLP measurement removes this confound at the cost that d_l no longer reflects the full residual stream seen by the next layer.

**Why this measures what we claim.** The computational quantity `d_l` (normalized pairwise stream difference measured on the pre-MLP mixing output P_l h_{l-1}) measures the layer's actual under-mixing because under-mixing is definitionally the regime where streams remain distinguishable after mixing, so a small `d_l` is equivalent to well-mixed streams under the assumption that the pre-MLP mixing output is dominated by the mixing matrix rather than by independent per-stream nonlinearities; this assumption fails when the per-stream MLPs inject stream-specific features that persist into the next layer's input, in which case `d_l` reflects representational specialization rather than mixing deficiency. The quantity `q_l` (normalized gradient difference) measures gradient-difference accumulation because it is the empirical analog of cmHC's gradient-difference bound term under the assumption that within-layer gradient differences track the cross-layer accumulation the bound describes; this assumption fails in the first few layers where gradients have not yet accumulated, in which case `q_l` reflects local curvature rather than accumulation. The contracted matrix `P_l^c` measures the achieved mixing depth because contraction toward the uniform barycenter is equivalent to increasing the minimum eigenvalue of the mixing operator under the doubly-stochastic constraint, which holds exactly for the Sinkhorn-projected matrices we contract; the diagnostic quantities `delta_l` (Frobenius distance of P_l to the barycenter) and `lambda_min_l` (minimum eigenvalue of P_l) are logged per layer to empirically verify that `d_l` is monotone in these directly computable mixing-depth measures, so the proxy-to-concept mapping is validated rather than assumed.

## Contribution

(1) A statistics-coupled contraction controller that computes the Birkhoff-polytope contraction radius in closed form from layer-wise EMA estimates of stream-difference and gradient-difference, replacing cmHC's fixed geometric schedule with a per-layer, distribution-adaptive radius. (2) A design principle showing that contraction strength should be a decreasing function of observed mixing, with the coupling gain and radius clamp as the only free hyperparameters, yielding an auditable per-layer diagnostic of mixing depth. (3) A per-layer radius trace that exposes where residual-stream statistics shift across depth, usable as a debugging signal for multi-stream residual architectures.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Deep multi-stream Transformer LM pre-training corpus (100M–350M params, depth sweep 12/24/36/48 layers) | Exposes cross-layer stream-mixing and gradient accumulation |
| Dataset | Deep ResNetV2 (CIFAR-100, TinyImageNet) | Tests generalization beyond Transformers |
| Dataset | Residual GNN on oversmoothing benchmark (Cora, Citeseer, PubMed) | Connects to E5 oversmoothing setting |
| Primary metric | Validation loss vs. depth curve | Detects under-mixing degradation and depth-scaling gap |
| Diagnostic metric | Per-layer d_l, q_l, delta_l, lambda_min_l logged; Spearman correlation between d_l and delta_l across depth | Validates proxy-to-concept mapping independently of controller |
| Baseline | cmHC (fixed contraction radius) | Isolates value of statistics-coupled adaptive radius |
| Baseline | mHC (Birkhoff-constrained hyper-connections) | Tests whether any contraction is needed at all |
| Baseline | Standard residual connections | Anchors against the canonical single-stream reference |
| Ablation-of-ours | SCC with stream-difference only (drop q_l) | Tests whether gradient-difference coupling is load-bearing |
| Ablation-of-ours | SCC with post-MLP d_l (original proxy) | Tests whether pre-MLP measurement matters |

### Why this setup validates the claim

The central claim is that replacing cmHC's fixed contraction radius with a closed-form radius computed from measured stream-difference and gradient-difference statistics yields layer-specific mixing that avoids both under-mixing (streams too independent) and over-mixing (streams indistinguishable) across depth. A validation-loss-versus-depth curve on a deep multi-stream Transformer is the right instrument because the predicted effect is depth-dependent: under-mixing compounds with depth, so a fixed radius should show a characteristic divergence or plateau at greater depths while an adaptive radius should track closer to the well-mixed optimum. cmHC is the critical baseline because it shares our Birkhoff-polytope machinery and differs only in radius policy, so any gap isolates the value of statistics coupling rather than of contraction per se. mHC tests whether contraction matters at all, and standard residuals anchor the multi-stream design against the canonical single-stream reference. The stream-only ablation is the load-bearing control: if dropping gradient-difference coupling does not hurt, then our joint-coupling rationale is wrong and the simpler controller suffices. The diagnostic logging of delta_l and lambda_min_l, together with the Spearman correlation between d_l and delta_l across depth, directly tests the load-bearing assumption that d_l is a monotone proxy for distance-to-barycenter; if the correlation is weak or non-monotone, the proxy is invalid and the controller's radius is not controlling the geometry it claims to control. The ResNetV2 and GNN settings test whether the statistics-coupled radius generalizes beyond multi-stream Transformers, and the post-MLP ablation tests whether the pre-MLP measurement choice is necessary.

### Expected outcome and causal chain

**vs. cmHC (fixed contraction radius)** — On a deep stack where later layers accumulate large gradient differences, cmHC's fixed radius keeps the mixing matrix near the polytope interior uniformly, so early layers are over-contracted (streams collapse toward the barycenter, losing specialization) while late layers are under-contracted relative to their gradient burden, producing a validation-loss plateau that worsens with depth. Our method instead raises rho_l where measured s_l + g_l is small and lowers it where accumulation is large, because the closed-form radius tracks the binding constraint per layer. We expect a modest gap at shallow depths but a clearly widening gap in the deepest third of layers, with early-layer stream-difference staying higher than cmHC's. The diagnostic log should show d_l tracking delta_l with Spearman rho > 0.7 across depth; if this correlation is weak, the gap over cmHC is not attributable to the claimed mechanism.

**vs. mHC (Birkhoff-constrained hyper-connections)** — On a case where residual matrices drift toward polytope vertices, mHC degenerates into nearly independent streams, so the residual path stops acting as a shared backbone and gradient flow fragments across streams, yielding slower convergence and higher final loss. Our method contracts toward the barycenter precisely when s_l signals under-mixing, because rho_l is a decreasing function of measured stream difference. We expect our loss curve to descend faster in the first portion of training and to end lower, with the advantage concentrated in layers whose logged rho_l sits above the clamp floor.

**vs. Standard residual connections** — On a deep multi-stream model, standard residuals offer a single fixed path, so they cannot adapt mixing to per-layer statistics and either under-mix (behaving like independent branches) or over-mix (losing multi-stream capacity) depending on initialization, showing a depth-scaling gap relative to the best fixed radius. Our method instead selects a per-layer radius from measured statistics, because the controller reads s_l and g_l directly rather than relying on a global hyperparameter. We expect parity at shallow depths and a growing advantage at greater depths, matching the predicted depth-dependent failure mode.

**vs. SCC with post-MLP d_l (original proxy)** — If the pre-MLP measurement is necessary, the post-MLP variant should show inflated d_l in layers where per-stream MLPs inject stream-specific features, causing the controller to contract when it should not and producing a higher validation loss at depth. We expect the post-MLP variant to underperform full SCC, with the gap concentrated in layers whose diagnostic log shows high per-stream module-output norm.

### What would falsify this idea

If the gain over cmHC is uniform across all depths rather than concentrated in the deep layers where s_l + g_l is largest, or if the stream-only ablation matches full SCC, or if the diagnostic Spearman correlation between d_l and delta_l is weak (rho < 0.3) across depth, then the claim that gradient-difference coupling and depth-dependent radius adaptation are the causal mechanisms is wrong.

## References

1. cmHC: Balancing Stream Mixing in Hyper-Connections via Geometric Contraction
