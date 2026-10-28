# Drift-Aware Adaptive Memory Capacity for Sparse Video Matting

## Motivation

Sparse memory propagation methods such as MatAnyone-E replace global attention with a cheap low-resolution memory search, but they inherit a fixed-size FIFO-like memory bank that is temporally bounded and capacity-invariant. The structural assumption that a constant number of memory slots suffices is violated under long-range temporal drift: when an object re-appears after prolonged occlusion or gradual appearance change, the informative frames have already been evicted, and the low-resolution selector cannot recover because the memory it searches no longer contains the evidence. The failure is therefore not in the selection rule but in the fixed capacity that selection operates over, a bottleneck shared across memory-based VOS and matting trees (e.g., Cutie's object-query memory and RAVOS's motion-path memory both assume a fixed retention budget).

## Key Insight

Retention capacity should be a function of estimated drift rather than a constant, because the number of past frames that remain informative about the current object is itself a drifting quantity that can be estimated from low-resolution feature statistics before high-resolution matching is attempted.

## Method

**What it is.** DriftCap is a memory controller that wraps the sparse memory bank of a propagation-based matting network. It takes low-resolution propagation features, the current low-resolution matte, and the existing memory bank as input, and outputs a resized and reweighted memory bank consumed by the unchanged high-resolution matting stage.

**How it works.**

```pseudocode
# per frame t, low-res features f_t (H/8 x W/8 x C), low-res matte a_t
# bank entries m_i = (key_i, value_i, weight_i, age_i)
# key_i is the H/8 x W/8 x C low-res feature map stored at frame i

1. DRIFT ESTIMATION (low-res only)
   d_t = cosine_dist( GAP(f_t * a_t), GAP(f_{t-1} * a_{t-1}) )   # masked global descriptor
   r_t = cosine_dist( GAP(f_t * a_t), GAP(f_{t-k} * a_{t-k}) )  # k=8 anchor distance
   drift_t = 0.5 * d_t + 0.5 * r_t
   # smooth over window w=5 with EMA: drift_hat = 0.7*drift_hat + 0.3*drift_t

2. CAPACITY MODULATION
   N_t = clip( round( N_base * (1 + beta * drift_hat_t) ), N_min, N_max )
   # N_base=8, beta=2.0, N_min=4, N_max=32

3. BANK RESIZE
   if N_t > |bank|:
       # re-admit evicted entries from a cold store (last 64 frames) whose
       # drift-estimator distance to current descriptor is below tau=0.3
       admit top (N_t - |bank|) cold entries by descriptor similarity
   else:
       evict lowest-weight entries, weight_i = exp(-gamma * age_i) * conf_i
       # gamma=0.1, conf_i = mean low-res matte confidence at frame i

4. REWEIGHTING
   for each retained entry i:
       w_i = softmax_i( -dist(descriptor_i, descriptor_t) / T )   # T=0.1
   bank = {(key_i, value_i, w_i, age_i)}

5. HANDOFF
   pass bank to the unchanged sparse high-res matching stage
```

The drift estimator runs entirely at H/8 resolution on masked global descriptors, so it adds negligible cost relative to the high-resolution matcher it governs. The cold store is a bounded ring buffer of the last 64 evicted entries, which is what makes re-admission possible without re-running propagation on past frames.

**Why this design.** We chose a descriptor-distance drift estimator over a learned drift classifier because the drift we care about is defined relative to the current frame's own appearance statistics, and a fixed cosine-distance on masked global descriptors is invariant to absolute feature scale, accepting the cost that it cannot distinguish genuine appearance change from camera motion unless the matte mask is applied (which we do). We chose multiplicative capacity modulation `N_t = N_base(1 + beta*drift)` over a binary expand/contract switch because capacity should scale continuously with the estimated number of informative past frames, accepting the cost of one extra hyperparameter beta that must be tuned per resolution. We chose a bounded cold store with similarity-gated re-admission over simply growing the bank because unbounded growth reintroduces the cost that sparse propagation (MatAnyone-E) was designed to avoid, accepting the cost that entries evicted more than 64 frames ago are unrecoverable. A domain expert would not describe this as a variant of Cutie or RAVOS: those methods fix the retention budget and change the *selector* (object queries, motion paths), whereas DriftCap changes the *budget itself* as a function of an estimated drift signal, leaving the selector untouched.

**Why it measures what we claim.** The smoothed descriptor distance `drift_hat_t` measures *long-range temporal drift* because it assumes that the masked global descriptor of the target is a sufficient statistic for the object's appearance state; this assumption fails when the object undergoes non-rigid deformation that changes the descriptor without changing identity, in which case `drift_hat_t` reflects pose change rather than appearance drift and capacity is inflated unnecessarily. The capacity `N_t` measures *the number of past frames still informative about the current object* because it assumes informativeness decays monotonically with drift; this assumption fails under abrupt re-appearance after occlusion, where recent frames are uninformative but old cold-store frames are highly informative, which is exactly the case the cold-store re-admission path is designed to catch. The reweighting `w_i` measures *relevance of each retained memory entry* because it assumes descriptor proximity implies matching utility; this assumption fails when two frames share a descriptor but differ in occlusion boundary, in which case the weight reflects appearance similarity rather than matte-boundary usefulness.

**(D) Load-bearing assumption and its operationalization.** The load-bearing assumption is that the masked global-average-pooled low-resolution descriptor distance is a calibrated, monotone proxy for the number of past frames still informative about the current object, so that a single scalar drift signal can drive capacity modulation and cold-store re-admission correctly. We make this explicit and test it as follows: (i) we compute `drift_hat_t` and, on a held-out clip split, plot it against a ground-truth re-identification distance defined as the cosine distance between the masked global descriptors of frame t and the frame at which the same object last had a verified matte (obtained from ground-truth masks); a monotone relationship is the assumption. (ii) We add a boundary-aware distance alternative to the reweighting step: `dist_boundary(i,t) = 0.5 * cosine_dist(descriptor_i, descriptor_t) + 0.5 * (1 - IoU( boundary_i, boundary_t ))`, where `boundary_i` is the set of pixels on the low-res matte boundary at frame i; this tests whether descriptor proximity alone is sufficient for matching utility. (iii) We report the diagnostic plot and the ablation with and without the boundary term, so the assumption is verified rather than asserted.

## Contribution

(1) A drift-aware adaptive memory controller that modulates sparse memory bank capacity and per-entry weights as a continuous function of an estimated low-resolution drift signal, with a bounded cold store enabling re-admission of evicted entries. (2) The design principle that retention capacity, not just selection, must be drift-adaptive, demonstrated by showing that fixed-capacity sparse propagation degrades specifically on re-appearance-after-occlusion segments while adaptive capacity recovers them. (3) A low-cost drift estimator operating at H/8 resolution that adds negligible overhead to the high-resolution matting stage it governs.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | MOSE + high-res video matting set | Contains occlusion/re-appearance and long sequences |
| Primary metric | Matte quality (MSE/MAD) on drift-heavy subsets | Directly reflects matte fidelity under appearance drift |
| Baseline | MatAnyone-E (fixed sparse bank) | Tests whether adaptive budget beats fixed budget |
| Baseline | Cutie (fixed-budget selector) | Tests budget-vs-selector distinction |
| Baseline | XMem | Tests against long-term memory retention |
| Baseline | Learned-budget MLP (predicts N_t from the same descriptors) | Tests whether the heuristic controller is necessary vs. a learned schedule |
| Ablation-of-ours | DriftCap w/o cold-store re-admission | Isolates the re-admission contribution |
| Ablation-of-ours | DriftCap with boundary-aware reweighting distance | Tests the descriptor-proximity-implies-utility assumption |
| Diagnostic | Plot of drift_hat_t vs. ground-truth re-identification distance on held-out clips | Tests the monotone-proxy assumption |
| Hyperparameter protocol | Grid over beta in {1.0, 1.5, 2.0, 2.5, 3.0} and tau in {0.2, 0.3, 0.4} on a held-out clip split (10% of MOSE), report sensitivity curves | Bounds per-resolution tuning cost and makes it reproducible |

### Why this setup validates the claim

The central claim is that modulating the retention budget as a function of an estimated drift signal recovers informative past frames that fixed-budget propagation methods discard, without paying the cost of unbounded memory. MatAnyone-E tests the sub-claim that a *fixed* sparse budget is the bottleneck: if DriftCap beats it, the gain must come from budget adaptation rather than from a better selector, since the selector is unchanged. Cutie and XMem test the complementary sub-claim that changing the budget is orthogonal to changing the selector: on drift-heavy subsets where their fixed budgets evict still-useful frames, DriftCap should close part of the gap, while on stable subsets it should match them. The learned-budget MLP baseline tests whether the heuristic controller is necessary: if a small MLP predicting N_t from the same descriptors matches DriftCap, the novelty reduces to a hand-tuned schedule. The cold-store ablation is the sharpest test: it isolates whether re-admission of old, drifted-but-informative entries is doing real work, so a gain that survives without it would mean the drift signal is not actually selecting the right frames. The boundary-aware reweighting ablation tests the assumption that descriptor proximity implies matching utility. The drift_hat-vs-reidentification diagnostic directly tests the monotone-proxy assumption. Matte error on drift-heavy subsets is the right metric because the predicted effect is a matte-fidelity effect localized to frames where appearance has drifted; a metric aggregated over all frames would dilute exactly the signal the causal story predicts. The beta/tau sensitivity curves bound the per-resolution tuning cost and make the hyperparameter protocol reproducible.

### Expected outcome and causal chain

**vs. MatAnyone-E** — On a case where the target reappears after a long occlusion with drifted appearance, MatAnyone-E produces blurred or bleeding mattes because its fixed sparse bank has evicted the pre-occlusion frames that still describe the object's identity. Our method instead inflates capacity via `drift_hat_t` and re-admits cold-store entries whose descriptors match the current frame, so we expect a noticeable matte-error gap concentrated on occlusion-reappearance frames and near-parity on stable intervals.

**vs. Cutie** — On a case where the object undergoes gradual non-rigid deformation across many frames, Cutie produces boundary drift because its fixed retention budget keeps only recent, selector-chosen entries and discards older frames that still carry the object's stable appearance. Our method instead scales `N_t` with the smoothed drift signal and reweights retained entries by descriptor proximity, so we expect a moderate matte-error advantage on long deformation sequences and parity on short, stable clips.

**vs. XMem** — On a case where the object is briefly occluded and then returns with a changed descriptor, XMem produces either stale memory (over-trusting old frames) or lost memory (evicting them) because its retention policy does not adapt to the current drift estimate. Our method instead modulates capacity continuously and reweights by current-frame descriptor distance, so we expect a gap that grows with occlusion length and shrinks on sequences without long occlusions.

**vs. Learned-budget MLP** — On drift-heavy subsets, if the MLP matches DriftCap, the heuristic controller is not necessary; we expect DriftCap to match or slightly exceed the MLP on occlusion-reappearance frames because the cold-store re-admission path is not replicated by a pure N_t predictor, and to be comparable elsewhere.

**vs. DriftCap w/o cold-store re-admission** — On occlusion-reappearance frames, removing re-admission should increase matte error because old informative entries cannot be recovered; on stable intervals, performance should be unchanged.

**vs. DriftCap with boundary-aware reweighting** — If descriptor proximity alone is sufficient, the boundary-aware distance should not improve matte error; if it does, the assumption that descriptor proximity implies matching utility is violated on frames with equal descriptors but different occlusion boundaries.

### What would falsify this idea

If the gain over MatAnyone-E is uniform across drift-heavy and stable subsets rather than concentrated where the drift estimator fires, the central claim that budget adaptation tracks informativeness is wrong. Likewise, if removing cold-store re-admission leaves performance unchanged on occlusion-reappearance frames, the re-admission mechanism is not doing the work it claims; and if the drift_hat-vs-reidentification diagnostic shows no monotone relationship, the load-bearing assumption is false.

## References

1. MatAnyone-E: Sparse Memory Propagation for Efficient High-Resolution Video Matting
2. Putting the Object Back into Video Object Segmentation
3. Global Spectral Filter Memory Network for Video Object Segmentation
4. Region Aware Video Object Segmentation With Deep Motion Modeling
