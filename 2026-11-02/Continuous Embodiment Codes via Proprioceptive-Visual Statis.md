# Continuous Embodiment Codes via Proprioceptive-Visual Statistics for Soft-Prompt Interpolation in Hierarchical Foresight VLA

## Motivation

HiFOR inherits X-VLA's discrete data-source-to-soft-prompt mapping, so its hierarchical foresight and retrieval modules are conditioned on a categorical embodiment index rather than a continuous representation of the robot's physical configuration. This means an unseen or mixed embodiment must either be forced into the nearest known category or fall back to a default prompt, and the structural reason is that a discrete lookup has no notion of distance or interpolation between embodiments. The same failure recurs across the branch: Mantis and DUST likewise assume a single backbone configuration and do not expose embodiment as a continuous conditioning variable, so cross-embodiment latent identifiability is never established — multiple embodiments share one latent without any invariance mechanism.

## Key Insight

Embodiment identity is recoverable as a continuous code from low-dimensional proprioceptive and visual statistics, and because the soft-prompt manifold is approximately convex in this code, linear interpolation between learned embodiment anchors yields a valid conditioning prompt for unseen and mixed embodiments without retraining the backbone.

## Method

**What it is.** We introduce CEBA (Continuous Embodiment-Anchor Adapter), a drop-in replacement for HiFOR's discrete data-source soft-prompt lookup. CEBA takes as input the current proprioceptive vector (joint positions, velocities, gripper state; 14-d for the 7-DoF Franka variants), a small set of visual statistics (end-effector image-space centroid (2-d), apparent scale (1-d), and workspace occupancy on a 4×4 grid (16-d)), and a bank of K=4 learned embodiment anchor prompts; it outputs a single soft-prompt embedding that conditions HiFOR's hierarchical foresight prediction and action-guided retrieval.

**Load-bearing assumption (stated explicitly).** CEBA assumes that the embodiment code `c` is a metric-space representation in which Euclidean distance corresponds to behavioral/embodiment similarity, so that linear interpolation between learned anchors yields a valid conditioning prompt for unseen and mixed embodiments. The alignment loss alone does not guarantee this: it enforces intra-embodiment compactness and inter-anchor separation for the K labeled embodiments but places no constraint on the geometry between anchors or outside the anchor convex hull. We therefore (i) add a dynamics-prediction auxiliary loss and an ordinal triplet loss over known embodiment-similarity pairs to supervise distance in `c` to track behavioral difference, and (ii) calibrate the anchor geometry on a held-out-embodiment validation loop, reporting the residual MLP as the primary carrier of unseen-embodiment information when calibration shows the interpolation weights are not metrically aligned.

**How it works.**

```pseudocode
# Phase 1: embodiment code extraction (per timestep)
z_proprio = LayerNorm(MLP_proprio(proprio))          # 64-d, 2-layer MLP, hidden=128, GeLU
z_visual  = LayerNorm(MLP_visual(visual_stats))      # 64-d, 2-layer MLP, hidden=128, GeLU
c = LayerNorm(W_c [z_proprio ; z_visual])            # 32-d embodiment code

# Phase 2: anchor bank and interpolation weights
tau = 0.1                                            # temperature
logits_k = -||c - a_k||^2 / tau   for k in 1..K       # a_k: learned 32-d anchors, K=4
tau_cal = 0.1                                        # calibrated temperature (see Stage A calibration)
logits_k_cal = -||c - a_k||^2 / tau_cal
w = softmax(logits_k_cal)                            # simplex weights over K anchors

# Phase 3: soft-prompt synthesis
P = sum_k w_k * E_k                                  # E_k: learned prompt embedding, dim d_prompt=256
P = P + MLP_res(c)                                   # residual adapter, 2-layer MLP, hidden=128, GeLU

# Phase 4: conditioning HiFOR (frozen backbone)
foresight_tokens = HDM(P, vision_tokens)             # hierarchical foresight prediction
retrieved = ActionGuidedRetrieval(P, foresight_tokens, action_query)
action = Policy(P, retrieved, vision_tokens)

# Training (two-stage)
# Stage A: train MLP_proprio, MLP_visual, W_c, anchors {a_k}, prompts {E_k}, MLP_res
#   with L_task + lambda_align * L_align + lambda_dyn * L_dyn + lambda_tri * L_tri
#   lambda_align = 0.5, lambda_dyn = 0.2, lambda_tri = 0.2
#   L_align = -log exp(-||c - a_{k*}||^2/tau) / sum_k exp(-||c-a_k||^2/tau)
#   where k* is the ground-truth embodiment label when available (self-supervised
#   clustering loss otherwise, using a momentum encoder for pseudo-labels).
#   L_dyn = || g_dyn(c, proprio_t) - proprio_{t+1} ||^2   (g_dyn: 2-layer MLP, hidden=128)
#   L_tri = max(0, ||c_i - c_j||^2 - ||c_i - c_k||^2 + margin), margin=0.1,
#     over triplets (i,j,k) where embodiment(i) and embodiment(j) are labeled
#     behaviorally similar (same kinematic family, payload difference < 0.5 kg)
#     and embodiment(k) is dissimilar (different kinematic family or payload
#     difference > 1.0 kg).
#   Calibration: after Stage A, hold out one embodiment, sweep tau_cal in
#     {0.05, 0.1, 0.2, 0.5} on a 512-episode calibration set from the held-out
#     embodiment, select tau_cal maximizing success rate; report the selected
#     value and the residual MLP's contribution separately.
# Stage B: freeze code extractor and anchors; fine-tune only MLP_res and
#   HiFOR's foresight/retrieval heads with L_task.
```

**Why this design.** We chose a continuous code with a learned anchor bank over HiFOR's discrete data-source lookup because a discrete index cannot represent an unseen embodiment or a mixed embodiment (e.g., a robot with a swapped gripper), whereas a continuous code has a well-defined neighborhood — accepting the cost that the anchor bank must be learned and can collapse without the alignment loss. We chose squared-distance softmax interpolation over a learned mixture-of-experts gate because the anchor geometry gives an explicit metric on embodiment space that the softmax weights inherit, so interpolation between anchors is meaningful by construction; a learned gate would have no such guarantee and would reduce to another discrete router. We chose a residual MLP adapter on top of the interpolated prompt over using only the interpolation because interpolation alone loses the fine-grained information in the code (two codes equidistant from the same anchors get identical prompts); the residual preserves it, at the cost of a small extra parameter count. We chose two-stage training (frozen backbone in stage B) over end-to-end training because end-to-end gradients would let the backbone reshape the embodiment code to fit the task, destroying the code's role as an embodiment descriptor — accepting the cost that the code is not optimized jointly with the policy. This is not a variant of X-VLA's soft-prompt mapping: X-VLA uses a one-hot data-source index to select a prompt, whereas CEBA computes a metric-space code and interpolates, so the conditioning surface is continuous and defined for unseen embodiments.

**Why it measures what we claim.** The embodiment code `c` measures embodiment identity because the alignment loss forces `c` to be close to its embodiment's anchor under the assumption that proprioceptive and visual statistics are sufficient to distinguish embodiments; this assumption fails when two embodiments share nearly identical statistics (e.g., two arms with the same kinematics but different payloads), in which case `c` reflects kinematic similarity rather than embodiment identity. The interpolation weights `w` measure the degree to which the current embodiment resembles each anchor because the softmax over negative squared distances is a kernel density estimate under the assumption that embodiment space is locally Euclidean around each anchor; this assumption fails for embodiments far outside the anchor convex hull, in which case `w` reflects extrapolation distance rather than resemblance. The residual `MLP_res(c)` measures the embodiment-specific information not captured by the anchor geometry because it is trained only in stage B with the backbone frozen, under the assumption that the frozen backbone's foresight/retrieval heads are embodiment-agnostic; this assumption fails when the backbone has already overfit to the training embodiments, in which case the residual absorbs embodiment information the anchors should have carried. The dynamics-prediction loss `L_dyn` measures behavioral similarity because it forces `c` to predict the next proprioceptive state under the embodiment, under the assumption that behavioral difference is reflected in transition dynamics; this assumption fails when two embodiments have identical dynamics but different visual appearance, in which case `c` reflects dynamics rather than full embodiment identity. The triplet loss `L_tri` measures ordinal embodiment similarity because it enforces that behaviorally similar embodiments are closer in `c` than dissimilar ones, under the assumption that the labeled similarity pairs are correct; this assumption fails when the similarity labels are noisy, in which case `c` reflects label noise rather than behavioral distance.

## Contribution

(1) CEBA, a continuous embodiment-embedding adapter that replaces HiFOR's discrete data-source soft-prompt mapping with a metric-space code and anchor-bank interpolation, providing a well-defined conditioning prompt for unseen and mixed embodiments. (2) A two-stage training procedure that decouples embodiment-code learning from policy learning, with an alignment loss that grounds the code in embodiment identity rather than task performance. (3) An empirical characterization of the embodiment-code manifold showing that interpolation weights correlate with physical embodiment similarity and that mixed-embodiment prompts can be synthesized by convex combination of anchors.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | Mixed-embodiment manipulation suite: RoboCasa + 4 Franka variants (base, swapped gripper, altered kinematics, payload +1 kg) with programmatic gripper/payload swaps in simulation | Contains multiple embodiments and held-out gripper swaps; 3-4 variants keep the benchmark tractable |
| Primary metric | Success rate on held-out / mixed-embodiment tasks | Directly tests generalization to unseen embodiment conditioning |
| Secondary metric | Prompt-trajectory smoothness (mean absolute difference between consecutive soft-prompt embeddings over a gripper-swap interpolation sweep) and behavioral monotonicity (success rate vs. graded gripper-mass sweep) | Tests whether prompt-space interpolation maps to behavioral interpolation |
| Baseline | HiFOR (discrete data-source soft-prompt lookup) | Tests whether continuity beats discrete lookup |
| Baseline | X-VLA-style one-hot prompt selector | Tests whether metric-space interpolation beats one-hot routing |
| Baseline | CEBA without residual adapter (interpolation only) | Isolates whether residual preserves fine-grained code |
| Baseline | Learned MoE gate (softmax over K learned gates from `c`, no anchor geometry) | Tests whether anchor geometry yields an interpolation advantage over a learned router |
| Baseline | Plain continuous MLP-to-prompt (no anchors; `P = MLP(c)`) | Tests whether anchors are needed at all |
| Ablation-of-ours | CEBA with alignment loss removed (anchor collapse) | Tests whether anchor geometry is what carries the gain |
| Ablation-of-ours | CEBA with `L_dyn` and `L_tri` removed (classification-only alignment) | Tests whether the metric/continuity objective is what calibrates the code |
| Ablation-of-ours | CEBA with `tau_cal` fixed at 0.1 (no calibration) | Tests whether the held-out calibration loop matters |

### Why this setup validates the claim

The central claim is that a continuous, metric-space embodiment code plus anchor interpolation conditions HiFOR better than a discrete data-source lookup, specifically on unseen and mixed embodiments. The dataset must therefore contain held-out embodiments (e.g., a swapped gripper, an arm with altered kinematics) and mixed-embodiment episodes; a single-embodiment benchmark would make the claim untestable because every method could route correctly. HiFOR with its discrete lookup tests the sub-claim that continuity matters: it should succeed on training embodiments but degrade on held-out ones. The X-VLA-style one-hot selector tests the stronger sub-claim that metric interpolation, not merely prompt conditioning, is what generalizes. The learned MoE gate and plain MLP-to-prompt baselines test whether the anchor geometry itself yields a measurable interpolation advantage (smooth prompt trajectories, monotonic behavioral sweeps) that a learned router or anchor-free MLP cannot produce. The interpolation-only variant isolates whether the residual adapter is needed to retain fine-grained code information, and the no-alignment and no-dynamics/triplet ablations test whether the anchor geometry and the metric/continuity objective are doing the work or whether any learned prompt bank suffices. Success rate on held-out/mixed subsets is the right metric because the predicted effect is a subset-specific gap, not a uniform improvement; a uniform gain would be consistent with generic capacity increases rather than the embodiment-conditioning mechanism. The prompt-trajectory smoothness and behavioral monotonicity metrics directly test the load-bearing assumption that Euclidean distance in `c` corresponds to behavioral similarity and that interpolation between anchors yields valid conditioning. This design thus forms a falsifiable test: the claim predicts a gap concentrated on held-out and mixed embodiments, with parity on training embodiments, and smooth monotonic behavior under interpolation sweeps.

### Expected outcome and causal chain

**vs. HiFOR (discrete data-source lookup)** — On a held-out embodiment such as a Franka with a swapped gripper, HiFOR's discrete lookup must map the episode to one of its training data-source indices, so it produces a prompt belonging to the nearest seen embodiment and mis-scales its foresight and retrieval to the wrong gripper dynamics. Our method instead computes a continuous code from proprioception and visual statistics and interpolates between anchors, so the prompt lands in the neighborhood of the swapped-gripper embodiment. We expect a noticeable success-rate gap on the held-out-embodiment subset but near-parity on training embodiments.

**vs. X-VLA-style one-hot prompt selector** — On a mixed-embodiment episode where the gripper changes mid-trajectory, the one-hot selector produces a single fixed prompt for the whole episode because its conditioning surface has no notion of degree, so the foresight tokens become inconsistent with the post-swap proprioception. Our method instead re-derives the code each timestep and the softmax weights shift continuously, so the prompt tracks the transition. We expect a gap concentrated on mixed/transition episodes, with parity on single-embodiment episodes.

**vs. CEBA without residual adapter (interpolation only)** — On two embodiments whose codes are nearly equidistant from the same two anchors, interpolation-only produces nearly identical prompts because the anchor geometry collapses fine-grained distinctions, so the policy applies the wrong fine manipulation scale. Our method instead adds the residual MLP(c), preserving the code's fine-grained information. We expect a modest but consistent gap on fine-grained precision tasks, smaller than the held-out-embodiment gap but present, confirming the residual's role.

**vs. Learned MoE gate** — The MoE gate learns a softmax over K gates from `c` without anchor geometry, so its prompt trajectories need not be smooth under a gripper-swap interpolation sweep and its behavioral sweeps need not be monotonic. Our method's anchor geometry gives an explicit metric, so we expect smoother prompt trajectories (lower mean absolute difference between consecutive prompts) and more monotonic success-vs-mass curves, with the success-rate gap concentrated on held-out and mixed subsets.

**vs. Plain continuous MLP-to-prompt (no anchors)** — The anchor-free MLP maps `c` directly to a prompt, so it has no explicit interpolation structure and may overfit to training embodiments. Our method's anchor bank provides a structured prior, so we expect a gap on held-out embodiments and smoother prompt trajectories, with parity on training embodiments.

**vs. CEBA with `L_dyn` and `L_tri` removed (classification-only alignment)** — Without the dynamics-prediction and triplet losses, `c` is only discriminative for the K labeled embodiments and its geometry between anchors is unconstrained, so interpolation for unseen embodiments is not metrically calibrated. Our method supervises distance in `c` to track behavioral difference, so we expect a gap on held-out and mixed embodiments and more monotonic behavioral sweeps, with parity on training embodiments.

**vs. CEBA with `tau_cal` fixed at 0.1 (no calibration)** — Without the held-out calibration loop, the interpolation temperature is not tuned to the anchor geometry, so the weights may be too sharp or too flat for unseen embodiments. Our method selects `tau_cal` on a 512-episode calibration set, so we expect a modest gap on held-out embodiments, confirming the calibration's role.

### What would falsify this idea

If CEBA's gain over HiFOR and the one-hot selector is uniform across training, held-out, and mixed embodiments rather than concentrated on held-out and mixed subsets, the central claim that continuity in embodiment space drives generalization would be wrong. Likewise, if removing the alignment loss or the dynamics/triplet losses leaves performance unchanged, or if prompt-trajectory smoothness and behavioral monotonicity do not differ from the MoE and MLP-to-prompt baselines, the anchor geometry and metric calibration are not what carry the effect and the mechanism story fails.

## References

1. HiFOR: Hierarchical Foresight with Action-Guided World Grounding for Vision-Language-Action Model
2. Mantis: A Versatile Vision-Language-Action Model with Disentangled Visual Foresight
3. Dual-Stream Diffusion for World-Model Augmented Vision-Language-Action Model
