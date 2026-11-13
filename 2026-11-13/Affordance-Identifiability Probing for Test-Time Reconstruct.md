# Affordance-Identifiability Probing for Test-Time Reconstruction Adaptation in Robot Learning from Demonstration

## Motivation

Test-time training methods for 3D reconstruction, such as TTT3R (2025), adapt the scene representation using only photometric and geometric consistency objectives. This leaves the identifiability of manipulation affordances—contact surfaces and grasp points—unverified, so adapted reconstructions may be photometrically plausible yet physically misleading for demonstration transfer. The structural root cause is that the adaptation objective has no term that forces affordance-relevant scene structure to be exposed and consistent with demonstrated contact statistics, a gap that recurs across reconstruction-for-manipulation pipelines and becomes binding when demonstrations are retargeted through reconstructed scenes.

## Key Insight

Affordance identifiability is a property of the reconstruction's local surface geometry relative to demonstrated contact distributions, and it can be made a differentiable test-time training signal by probing whether a small set of demonstrated contact points remains locally reconstructible under the adapted model.

## Method

**What it is.** We introduce AIRA (Affordance-Identifiability Reconstruction Adaptation), a test-time training procedure that augments a pretrained 3D reconstruction model with an affordance-identifiability probe and loss. Inputs are a short RGB-D sequence of a scene plus a set of demonstrated contact points (end-effector positions and surface normals from a demonstration). Outputs are an adapted reconstruction and a scalar identifiability score that gates whether the reconstruction is used for demonstration transfer.

**Load-bearing assumption (stated explicitly).** The method assumes that gradients from the affordance-identifiability loss on local patches around demonstrated contact points will reshape the adapted reconstruction's geometry toward physically correct contact surfaces, rather than the probe simply learning to fit whatever geometry the reconstruction happens to produce. To prevent the probe from absorbing the loss through co-adaptation (confirmation bias), we freeze the probe during test-time adaptation: it is pretrained once on external demonstration data (a disjoint set of scans and demonstrations from the same sensor suite) and its parameters are not updated in Phase 2. The probe therefore cannot adapt its decision boundary to the current geometry; the only way to reduce L_aff is to move the reconstruction. Negatives are defined in a geometry-independent frame: for each demonstrated contact point c_i with end-effector frame (R_i, t_i), we sample a negative point c_i' = c_i + R_i · u, where u is drawn uniformly from the set of unit vectors with angle ≥ 60° from the demonstrated approach direction (the end-effector z-axis), at a fixed distance ε=0.05 m. This places negatives in the demonstrated end-effector frame rather than by perturbing the current reconstructed surface, so the contrastive signal does not shift with the geometry it is meant to correct. Where ground-truth depth is available (RGB-D input), negatives are additionally filtered to points whose measured depth differs from the reconstructed surface by more than 0.02 m, ensuring they are off-surface in the sensor frame.

**How it works.**
```pseudocode
# Phase 1: Probe construction (frozen during Phase 2)
contacts = extract_contact_points(demonstration)          # N points, each with position p_i, normal n_i, end-effector frame (R_i, t_i)
probe = AffordanceProbe(contacts)                        # 2-layer MLP, hidden width 64, ReLU activation, input = 64-dim patch descriptor + 3-dim normal
# Pretrain probe on external demonstration data (disjoint scans, same sensor suite):
#   5000 positive patches (demonstrated contacts) + 5000 negatives (end-effector-frame perturbations, see below)
#   Adam lr=1e-3, batch=64, 2000 steps, early stopping on held-out 20% of external data
#   Freeze probe parameters after pretraining; do NOT update in Phase 2

# Phase 2: Test-time adaptation loop (K=200 steps, Adam lr=1e-4, probe frozen)
for step in 1..K:
    # standard reconstruction losses (kept from TTT3R)
    L_photo = photometric_loss(render(adapted_model), rgb)
    L_geo   = geometric_consistency_loss(adapted_model, depth)

    # affordance-identifiability loss (probe frozen; gradients flow only to adapted_model)
    for each contact point c_i in contacts:
        # local patch around c_i on the reconstructed surface
        patch_i = sample_local_patch(adapted_model, c_i, radius=0.02m)   # 64-dim descriptor: 4x4x4 occupancy grid + mean normal
        # frozen probe predicts whether c_i is a valid contact surface
        logit_i = probe(patch_i, c_i.normal)
    L_aff = binary_cross_entropy(logit_i, label=1)       # all demonstrated contacts are positive
    # contrastive negative in demonstrated end-effector frame (geometry-independent)
    for each c_i:
        u = sample_unit_vector_with_angle_ge(60deg, from=R_i[:,2])   # approach direction = end-effector z-axis
        c_i' = c_i + R_i @ (u * 0.05m)                                # ε=0.05 m, fixed in end-effector frame
        if depth_available and |depth(c_i') - reconstructed_depth(c_i')| < 0.02m:
            resample u   # ensure negative is off-surface in sensor frame
        L_aff += binary_cross_entropy(probe(patch(c_i')), label=0)

    total_loss = L_photo + L_geo + lambda_aff * L_aff     # lambda_aff=0.5
    update adapted_model only   # probe frozen

# Phase 3: Identifiability validation
score = mean(logit_i for held-out contacts)              # held-out contacts from same demonstration (20% split)
if score > tau:                                          # tau=0.7 calibrated on validation set (see below)
    accept adapted reconstruction for demonstration transfer
else:
    reject and fall back to unadapted reconstruction
```

**Calibration of τ.** τ=0.7 is fixed by a pilot on 5 scans (disjoint from the evaluation set): for each scan, we run AIRA and compute the held-out identifiability score and the downstream contact-placement success; τ is chosen as the score threshold that maximizes the Youden index (sensitivity + specificity − 1) for predicting success above 0.8. The pilot also fixes ε=0.05 m, λ_aff=0.5, and the probe pretraining hyperparameters listed above; no further hyperparameter search is performed on the evaluation set.

**Why this design.** We chose a learned probe over a hand-crafted geometric heuristic (e.g., local curvature thresholds) because affordance identifiability depends on the joint distribution of surface geometry and demonstrated contact placement, which a probe can capture but a fixed threshold cannot; the cost is that the probe itself must be trained, which we handle by pretraining it on external demonstration data and freezing it during adaptation, so its gradients cannot co-adapt with the reconstruction. We chose a contrastive negative sampling scheme over a purely positive classification loss because without negatives the probe can trivially achieve zero loss by predicting positive everywhere; the cost is that negative sampling introduces a hyperparameter (perturbation magnitude ε), which we set conservatively to 0.05 m in the demonstrated end-effector frame to avoid overlapping with true contact regions and to keep negatives geometry-independent. We chose to keep the original photometric and geometric losses active during adaptation rather than replacing them with the affordance loss, because the affordance probe relies on a geometrically valid surface to sample patches from; the cost is that the affordance signal may be diluted by the reconstruction losses, which we mitigate by weighting λ_aff=0.5. This design differs from the obvious composition of "TTT3R + affordance classifier" because the probe is not a post-hoc classifier: it is trained jointly with the reconstruction (via frozen-probe gradients flowing back into the reconstruction model), forcing the adapted geometry to expose contact surfaces that the probe can identify. A domain expert would not describe this as a variant of TTT3R with an added head because the probe's output is used to gate the reconstruction itself, creating a closed loop between affordance identifiability and geometric adaptation.

**Why it measures what we claim.** The probe's logit for a demonstrated contact point measures affordance identifiability because we assume that if the adapted reconstruction correctly represents the local surface geometry at that contact, then a probe trained on that patch will assign high probability to the contact label; this assumption fails when the probe overfits to the specific contact points seen during adaptation, in which case the logit reflects memorization rather than generalizable identifiability. The contrastive negative loss measures the specificity of the probe because we assume that perturbed points off the surface are not valid contacts; this assumption fails when the perturbation lands on another valid contact surface (e.g., in cluttered scenes), in which case the negative label is incorrect and the probe is pushed to suppress a true affordance. The held-out validation score measures the generalization of identifiability because we assume that held-out contacts from the same demonstration are drawn from the same contact distribution; this assumption fails when the demonstration is multimodal or the held-out contacts are from a different phase of the task, in which case the score reflects distribution shift rather than reconstruction quality. The gating threshold τ measures the reliability of the adapted reconstruction for demonstration transfer because we assume that high probe confidence on held-out contacts implies that the reconstruction will support correct contact placement during transfer; this assumption fails when the transfer task requires contact points not represented in the demonstration, in which case the gate may accept a reconstruction that is locally valid but globally insufficient.

## Contribution

['A test-time training procedure (AIRA) that augments reconstruction adaptation with a differentiable affordance-identifiability probe and contrastive loss, coupling geometric adaptation to demonstrated contact statistics.', 'A gating mechanism that uses held-out contact identifiability scores to decide whether an adapted reconstruction is safe for demonstration transfer, providing a concrete criterion for when test-time adaptation has succeeded.', 'An empirical design principle: affordance identifiability can be operationalized as local patch classification with contrastive negatives, and this signal can be trained jointly with reconstruction without degrading photometric or geometric quality.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Ropedia Xperience 2 contact-rich scans | Metric-scale 3D plus demonstrated contacts, real clutter |
| Primary metric | Contact-placement success on held-out demonstrations | Directly tests whether gated reconstruction supports transfer |
| Baseline | Frozen TTT3R reconstruction | Tests whether joint probe gradients matter |
| Baseline | TTT3R + post-hoc affordance classifier | Tests closed-loop gating versus added head |
| Baseline | Hand-crafted curvature-threshold contact detector | Tests learned probe versus geometric heuristic |
| Ablation-of-ours | AIRA without contrastive negatives | Tests whether negative sampling is load-bearing |
| Ablation-of-ours | AIRA with probe unfrozen during adaptation | Tests whether freezing the probe prevents co-adaptation |

### Why this setup validates the claim

The central claim is that jointly adapting reconstruction with an affordance-identifiability probe produces geometry that exposes transferable contact surfaces, and that the probe's held-out score reliably gates this. Frozen TTT3R tests the sub-claim that adaptation is necessary: if it already transfers contacts, the probe's gradients contribute nothing. The post-hoc classifier tests whether the closed loop matters: if a classifier bolted onto a fixed reconstruction matches AIRA, the probe's gradient flow into geometry is decorative. The curvature heuristic tests whether a learned probe is needed at all. The ablation without contrastive negatives isolates the negative sampling, which the design argument says prevents trivial positive-everywhere collapse. The ablation with an unfrozen probe directly tests the load-bearing assumption: if the probe can co-adapt with the reconstruction, it may absorb the loss without reshaping geometry, so we expect this ablation to show inflated identifiability scores that do not track transfer success. Contact-placement success on held-out demonstrations is the right metric because it is the downstream quantity the identifiability score claims to predict; a probe that scores well but does not raise transfer success falsifies the gating rationale directly.

### Expected outcome and causal chain

**vs. Frozen TTT3R reconstruction** — On a case where the demonstrated contact lies on a thin, low-texture handle that photometric and geometric losses leave slightly flattened, frozen TTT3R reconstructs a surface offset from the true contact patch, so transferred end-effector poses miss by a margin that grows with surface error. Our method instead pulls geometry toward the demonstrated contact because the frozen probe's gradient rewards surfaces where contact patches are identifiable, so we expect a clear success gap concentrated on low-texture, thin-geometry objects and rough parity on textured, bulky ones.

**vs. TTT3R + post-hoc affordance classifier** — On a case where the classifier is trained on the adapted surface but never reshapes it, the reconstruction retains a locally ambiguous contact region and the classifier's confidence is high yet misplaced, so the gate accepts a reconstruction that misplaces contacts. Our method instead forces the geometry itself to expose the contact surface during adaptation, so we expect the post-hoc variant to show high probe confidence but low transfer success, while AIRA shows both high, with the divergence largest where reconstruction error is contact-relevant.

**vs. Hand-crafted curvature-threshold detector** — On a case where the valid contact is a flat or gently curved region whose curvature is indistinguishable from nearby non-contact surface, the threshold detector flags the wrong patch because fixed geometry statistics cannot encode where demonstrations actually placed contact. Our method instead learns the joint geometry-placement distribution from the demonstration, so we expect the heuristic to fail specifically on low-curvature contact regions and AIRA to hold, producing a gap on flat-contact subsets and parity on high-curvature ones.

**vs. AIRA without contrastive negatives** — On a case where the probe sees only positive labels, it converges to predicting contact everywhere, so the identifiability score saturates and the gate accepts every reconstruction including poor ones. Our method instead uses perturbed off-surface negatives in the demonstrated end-effector frame to force specificity, so we expect the ablation to show inflated scores with no correlation to transfer success, while AIRA's score tracks success and its gate rejects the degraded reconstructions the ablation accepts.

**vs. AIRA with probe unfrozen during adaptation** — On a case where the probe is updated jointly with the reconstruction, it can minimize L_aff by shifting its decision boundary to match the current geometry rather than forcing the geometry to move, so the identifiability score rises without a corresponding gain in transfer success. Our method instead freezes the probe after pretraining on external data, so the only way to reduce L_aff is to reshape the reconstruction, and we expect the unfrozen ablation to show high scores with low transfer success (a co-adaptation signature) while AIRA's score and success move together.

### What would falsify this idea

If AIRA's transfer-success gain over frozen TTT3R is uniform across all object subsets rather than concentrated on low-texture, thin, and low-curvature contact regions where the predicted reconstruction failure occurs, the causal story that the probe reshapes contact-relevant geometry is wrong. Likewise, if the held-out identifiability score fails to separate accepted from rejected reconstructions on transfer success, or if the unfrozen-probe ablation matches AIRA's transfer success despite higher scores, the gating and freezing claims collapse.

## References

1. Ropedia Xperience 2: A Gold Standard for Scaling Physical AI Environments and Behaviors
