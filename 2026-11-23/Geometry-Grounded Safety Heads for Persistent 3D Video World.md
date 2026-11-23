# Geometry-Grounded Safety Heads for Persistent 3D Video World Models

## Motivation

Across the persistent-3D-memory and long-horizon video world model branches, safety/penetration prediction is decoupled from the explicit geometry that the world model itself uses for simulation: NoPeWorld trains its prediction head purely on frozen DiT hidden states with no tie to any explicit obstacle representation, while persistent embodied world models (Learning 3D Persistent Embodied World Models) and long-term spatial memory models build explicit 3D maps but never use them to constrain action-safety decisions. The root cause is structural: the safety head's decision boundary lives in an opaque latent space that is never anchored to the metric geometry (occupancy, signed distance) the simulator maintains, so when the map updates during revisits the safety verdict can drift arbitrarily relative to the true obstacle boundary. This leaves metric consistency between safety prediction and persistent 3D memory unestablished, a gap shared across both branches.

## Key Insight

Anchoring the safety head's input to map-derived signed-distance/occupancy features and tying its hidden-state penetration score to the map's geometric boundary via a differentiable consistency loss makes the safety decision boundary a function of the same metric geometry the simulator uses, so obstacle avoidance remains consistent under map updates rather than drifting in an unanchored latent space.

## Method

**What it is.** GeoSafe is a safety head for interactive video world models whose penetration score is computed from a fusion of frozen-DiT hidden states and explicit persistent-3D-map geometry (occupancy and signed-distance features), trained with a differentiable consistency loss that forces the hidden-state penetration score to agree with the map's geometric obstacle boundary. Inputs: candidate action $a_t$, frozen DiT hidden state $h_t$ (dimension $d_h=1024$), persistent 3D map $M$ (voxel occupancy $O(x)\in[0,1]$ + truncated signed-distance field $d(x)$ in meters, truncation $\tau_{sdf}=0.2$ m, voxel size 2 cm, plus per-voxel observation count $c(x)$). Output: a safe/unsafe verdict and, if unsafe, a replacement action.

**Load-bearing assumption (stated explicitly).** The method assumes (A) that the persistent map's truncated signed-distance field is a sufficiently accurate and current proxy for true obstacle geometry at query time, so that a binary penetration label derived from it ($p_{geo} = 1[\min_q sdf\_feat(q) < 0]$) can serve as a target for the consistency loss. This assumption fails (F) when the map is stale or incomplete — unobserved regions, recently-updated voxels, or moving obstacles not yet integrated — in which case $p_{geo}$ reflects map error rather than true geometry. To avoid inheriting map error as ground truth, the consistency loss is gated by per-voxel map confidence and the supervised label $L_{sup}$ remains the anchor; the consistency term acts as a regularizer, not a calibration source. The circularity is quantified on a held-out ground-truth-geometry validation subset where $p_{geo}$ is checked against simulator collision ground truth (see experiment).

**How it works.**

```pseudocode
# Phase 1: map-derived geometric features for the candidate action
M = persistent_3D_map            # voxel occupancy O(x), TSDF d(x), observation count c(x)
traj = rollout(a_t, M)           # candidate future trajectory in map frame, 16 query points
for each query point q in traj:
    occ_feat(q)  = trilinear(O, q)          # occupancy in {0,1} (soft)
    sdf_feat(q)  = trilinear(d, q)          # signed distance, meters
    conf(q)      = min(1, trilinear(c, q) / c_sat)   # c_sat = 5 observations
geo_t = [min_q sdf_feat, mean_q occ_feat, sdf_feat at closest approach, mean_q conf]

# Phase 2: fused safety score
h_t  = frozen_DiT_hidden_state(a_t, context)   # d_h = 1024
s_t  = MLP_head([h_t ; geo_t])              # 2-layer MLP, hidden=256, GeLU, scalar in [0,1]

# Phase 3: differentiable geometric consistency loss (training)
# geometric ground-truth penetration label from the MAP, gated by confidence
p_geo = 1[min_q sdf_feat(q) < 0]            # map says penetration
w_geo = mean_q conf(q)                       # confidence weight in [0,1]
L_cons = w_geo * BCE(sigmoid(s_t), p_geo)   # tie score to map boundary, down-weighted where map uncertain
L_sup  = BCE(sigmoid(s_t), y_penetration)   # supervised penetration label (anchor)
L = L_sup + lambda * L_cons                 # lambda = 0.5

# Phase 4: inference action replacement
if s_t > tau:                               # tau = 0.5
    a_safe = argmin_{a' in A_safe} ||a' - a_t||   # nearest safe action, A_safe = {a': s(a')<tau}
    execute(a_safe)
else:
    execute(a_t)
```

**Why this design.** We fuse map geometry into the head input rather than keeping the head purely on DiT hidden states (as NoPeWorld does) because the map is the only quantity that carries metric obstacle information and is updated on revisits; the cost is that the head now depends on map quality and must be retrained if the map representation changes. We add a differentiable consistency loss between the hidden-state score and the map's geometric boundary rather than relying on supervised labels alone, because labels are sparse and can be inconsistent with the map after updates; the cost is that the loss injects map noise into training, mitigated by the truncated SDF, soft occupancy, and the per-voxel confidence gate $w_{geo}$ that down-weights unobserved or recently-updated regions. We choose nearest-safe-action replacement over re-planning because it preserves the original intent with minimal deviation, accepting that it may be myopic in cluttered scenes. A domain expert would not describe this as 'NoPeWorld + a 3D map' because the consistency loss makes the hidden-state score a calibrated function of the map's signed-distance boundary — a coupling NoPeWorld's head does not have.

**Why it measures what we claim.** The quantity `min_q sdf_feat(q)` measures geometric obstacle proximity because the truncated signed-distance field is a metric distance to the nearest occupied voxel, so a negative value is by definition penetration; this assumption fails when the map is stale or incomplete (unobserved regions), in which case the SDF reflects map uncertainty rather than true geometry. The consistency loss `w_geo * BCE(sigmoid(s_t), p_geo)` measures metric consistency between the safety score and the map boundary because it forces the score to match the map's binary penetration label, weighted by map confidence; this assumption fails when the map label itself is wrong (e.g., moving obstacles not yet integrated), in which case the loss enforces agreement with an outdated boundary, which is why $L_{sup}$ remains the anchor and $w_{geo}$ gates the term. The fused feature `geo_t` measures the map-grounded safety context because occupancy and SDF are the same geometry the simulator conditions on; this fails when the candidate trajectory rollout diverges from the true dynamics, in which case geo_t reflects a hypothetical rather than realized path. The threshold `tau` measures the safe/unsafe decision boundary because it is trained to separate map-labeled safe from unsafe actions; this fails under distribution shift to new obstacle types, in which case it reflects the training distribution's boundary.

## Contribution

(1) A geometry-grounded safety head (GeoSafe) that fuses frozen-DiT hidden states with persistent-3D-map occupancy and signed-distance features, with a differentiable consistency loss tying the hidden-state penetration score to the map's geometric obstacle boundary. (2) A design principle: safety decisions for video world models should be anchored to the same explicit metric geometry used for long-horizon simulation, making the safety boundary a function of map state rather than an unanchored latent. (3) An evaluation protocol measuring safety-score stability under map updates (revisits) against map-derived geometric ground truth.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | NoPeBench + long-horizon revisit scenes | Obstacle-response benchmark plus memory-stress cases |
| Primary metric | Physical violation rate (penetration %) | Directly quantifies the claimed safety improvement |
| Baseline | NoPeWorld (hidden-state head only) | Isolates value of map-geometry fusion |
| Baseline | VideoPhy-style physical-commonsense evaluator | Tests whether gains are physics-general, not map-specific |
| Baseline | Autoregressive world model w/o safety head | Establishes the unguarded failure ceiling |
| Ablation-of-ours | GeoSafe minus consistency loss (supervised only) | Tests whether map-coupling loss drives calibration |
| Ablation-of-ours | GeoSafe with concat-only features, no L_cons | Isolates loss vs. geometry features |
| Validation subset | Ground-truth-geometry subset (simulator collision labels) | Quantifies circularity of p_geo vs. true penetration |

### Why this setup validates the claim

The central claim is that fusing persistent-3D-map geometry into the safety head, and tying the hidden-state score to the map's signed-distance boundary via a consistency loss, yields penetration scores that are better calibrated to true obstacle geometry than a hidden-state-only head — especially where the map carries metric information the DiT states lack. NoPeWorld is the decisive baseline: it shares our frozen-DiT substrate and action-replacement mechanism, so any gap isolates the map-fusion contribution rather than confounding it with architecture. The unguarded autoregressive model sets the untreated ceiling, confirming the task is non-trivial. The VideoPhy-style evaluator checks whether improvements transfer to physical commonsense beyond map-labeled penetration, guarding against overfitting to one benchmark. The supervised-only ablation is the load-bearing test: if the consistency loss is what calibrates the score to the map boundary, removing it should degrade calibration concentrated on revisit and stale-map subsets. The concat-only ablation isolates whether the loss — not the geometry features — is the novel ingredient. The ground-truth-geometry validation subset directly tests the load-bearing assumption (A) by checking $p_{geo}$ against simulator collision ground truth, so the circularity is measured rather than assumed. Violation rate is the right metric because it is exactly the behavior the method claims to fix, and it is measurable without trusting our own score.

### Expected outcome and causal chain

**vs. NoPeWorld (hidden-state head only)** — On a case where the agent approaches an obstacle first seen several frames earlier and the DiT's limited temporal context has dropped that obstacle from its hidden state, NoPeWorld's head scores the action as safe because its only input is a context-degraded hidden state. Our method instead reads occupancy and signed distance from the persistent map, which retains the obstacle across revisits, so the fused score flags penetration and replaces the action. We expect a noticeable violation-rate gap concentrated on revisit/long-horizon subsets, with near-parity on short single-view scenes where hidden states still carry the obstacle.

**vs. VideoPhy-style physical-commonsense evaluator** — On a case where a generated clip looks plausible but an object passes through a wall, the commonsense evaluator judges surface realism and misses the metric penetration because it has no explicit geometry. Our method instead grounds the verdict in a signed-distance field where negative distance is penetration by definition, so it catches violations the appearance-based evaluator accepts. We expect our method to show a clear advantage on metric-penetration instances while the two roughly agree on gross physical failures, confirming the gain is geometric rather than cosmetic.

**vs. Autoregressive world model without safety head** — On a case where compounding error drifts the predicted trajectory into an obstacle over a long rollout, the unguarded model executes the penetrating action because nothing checks the candidate against geometry. Our method intercepts the candidate, computes a fused score from map geometry plus hidden state, and substitutes the nearest safe action, so the same drift is caught before execution. We expect a large reduction from the untreated violation ceiling, largest on long-rollout and cluttered subsets, with the residual concentrated where the map itself is stale.

**vs. GeoSafe minus consistency loss (supervised only)** — On a case where the map is updated after training and the supervised labels no longer match the new boundary, the supervised-only head keeps its old score because nothing ties it to the current signed-distance field. Our consistency loss forces the score to agree with the map's geometric boundary, so it re-calibrates to the updated geometry. We expect the ablation to match us on static well-labeled scenes but to show worse calibration and higher violation rates specifically on post-update and revisited-map subsets.

**vs. GeoSafe with concat-only features, no L_cons** — On a case where the map is updated and the geometry features shift, the concat-only head has no term tying its score to the new boundary, so its decision boundary can drift relative to the updated SDF. Our consistency loss ties the score to the map boundary, so the score remains a monotone function of the SDF. We expect the concat-only ablation to match us on static scenes but to show worse calibration and higher violation rates on post-update and revisited-map subsets, isolating the loss as the novel ingredient.

**vs. Ground-truth-geometry validation subset** — On a held-out subset where simulator collision labels are available, we check $p_{geo}$ against true penetration. We expect $p_{geo}$ to agree with ground truth on well-observed voxels (confidence $c(x) \ge 5$) and to disagree on unobserved or recently-updated voxels, quantifying the circularity. The confidence gate $w_{geo}$ should down-weight the disagreeing regions, so the consistency loss does not inherit map error as ground truth.

### What would falsify this idea

If GeoSafe's gain over NoPeWorld is uniform across all subsets rather than concentrated on revisit and stale-map cases where the map carries information the hidden state lacks, then map fusion is not doing the causal work claimed and the central mechanism is wrong. Likewise, if removing the consistency loss leaves calibration on post-update maps unchanged, the geometric-coupling claim is falsified. If $p_{geo}$ disagrees with simulator collision ground truth even on well-observed voxels, the load-bearing assumption (A) is false and the consistency loss enforces agreement with map error rather than true geometry.

## References

1. Non-penetrating World Models
