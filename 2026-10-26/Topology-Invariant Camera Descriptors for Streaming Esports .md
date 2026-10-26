# Topology-Invariant Camera Descriptors for Streaming Esports Directing

## Motivation

DirCast learns a directing policy over a fixed per-hero camera rig by binding each camera to a learned identity embedding, so the hold-or-switch decision is entangled with the specific rig topology seen in training; the model has no representation of *where a camera is relative to the action*, only *which camera slot* it is. This is the same structural weakness that recurs across multi-view streaming video-language work (Ego-Exo4D's fixed exo rigs, DirCast's per-hero cameras): directing and commentary policies are evaluated only under the camera layout they were trained on, and the field's trajectory toward deployable live directing requires the decision to transfer to unseen rigs. Because the identity embedding carries no geometric meaning, retraining or re-indexing is required for every new topology, and no existing work tests whether the hold-or-switch decision is invariant to rig change.

## Key Insight

If each camera is described by a topology-agnostic geometric and scene-graph descriptor rather than an identity slot, then the hold-or-switch decision becomes a function of the *observable relation between cameras and the game state*, which is the same function across rigs — so invariance under topology shift is a property of the input representation, not something that must be learned from data covering every rig.

## Method

**What it is.** TIGER-D (Topology-Invariant Geometric Esports Director) is a streaming directing policy that replaces DirCast's per-hero identity embedding with a camera descriptor computed from observable geometry and scene-graph context, and is trained under synthetic-topology randomization so the hold-or-switch head sees the same game state through many rigs. The load-bearing assumption, stated explicitly: the focus-relative geometric descriptor (relative pose, FOV, occlusion ratio, scene-graph features) is a sufficient statistic for a camera's informativeness, such that the hold-or-switch decision computed from it is invariant across rig topologies and transfers to unseen rigs without any topology-specific learning. To bound this assumption rather than assert it, the descriptor is split into an invariant geometric part and a view-dependent informativeness part; the invariance penalty is applied only to the former, and a calibration/verification head predicts descriptor reliability from pose/FOV estimation confidence so the director head can down-weight unreliable geometry under novel optics.

**How it works.**

```pseudocode
# Per second t, for each candidate camera c in current rig R:
for c in R:
    # (1) Geometry descriptor from observable state only
    pose_c, pose_conf_c = estimate_pose(c)      # position + orientation, world frame; conf in [0,1]
    fov_c,  fov_conf_c  = estimate_fov(c)       # vertical FOV in degrees; conf in [0,1]
    focus_c      = nearest_hero_world_point(c)  # projected game-state anchor
    rel_pose_c   = world_to_relative(pose_c, focus_c)  # translation+rotation, focus-relative
    geom_inv_c   = [rel_pose_c, fov_c]          # invariant geometric part
    geom_view_c  = [occlusion_ratio(c, focus_c),
                    apparent_hero_scale(c, focus_c),
                    projected_screen_area_fraction(c, focus_c),
                    occlusion_gradient(c, focus_c)]  # view-dependent informativeness part

    # (2) Scene-graph context descriptor
    sg_c         = scene_graph_features(c)      # which heroes/objectives visible, screen-area fraction

    # (3) Fuse into topology-agnostic descriptor
    d_c = MLP_fuse([geom_inv_c, geom_view_c, sg_c, lowres_frame_embed(c)])
    # MLP_fuse: 3-layer MLP, hidden=256, GeLU, LayerNorm after each linear

    # (3b) Calibration / verification head: predict descriptor reliability
    rel_c = CalibHead([pose_conf_c, fov_conf_c, occlusion_ratio(c, focus_c)])
    # CalibHead: 2-layer MLP, hidden=128, GeLU, sigmoid output in [0,1]
    d_c_weighted = rel_c * d_c

# (4) Hold-or-switch head over descriptors (no camera identity)
current = d_{c_onair}_weighted
switch_logits = DirectorHead([current, {d_c_weighted for c != c_onair}], streaming_context)
# DirectorHead: 2-layer MLP, hidden=512, GeLU, dropout=0.1
action = argmax over {hold, switch-to-c}

# (5) Domain randomization over synthetic rigs during training
for each training match M in Dota2-27 and each epoch:
    R_syn = sample_synthetic_rig(M, K~Uniform{3..12}, pose~pose_prior, fov~U(40,110))
    render_views(M, R_syn)                 # cheap virtual cameras over the match replay
    loss += directing_CE(action, human_cut_reprojected_to_R_syn)
    loss += lambda_inv * invariance_penalty(geom_inv_c(R_syn), geom_inv_c(R_ref))  # same world point -> same invariant geometry
    # lambda_inv = 0.1; penalty = MSE on rel_pose_c and fov_c only; geom_view_c and sg_c excluded
    loss += lambda_cal * calibration_loss(rel_c, reliability_target)  # lambda_cal = 0.05; BCE on pose/fov confidence
```

**Why this design.** We chose a focus-relative geometric descriptor over a learned per-camera identity embedding because identity slots cannot generalize to unseen rigs by construction, accepting the cost that pose/FOV estimation noise now propagates into the directing decision. We chose synthetic-topology randomization over collecting real multi-rig data because real rigs are scarce and expensive, accepting the sim-to-real gap that rendered virtual cameras may not match broadcast optics. We chose an explicit invariance penalty tying descriptors of the same world anchor across rigs over relying on randomization alone, accepting the extra hyperparameter (penalty weight) in exchange for a direct training signal on the property we care about. We split the descriptor into an invariant geometric part and a view-dependent informativeness part, and apply the invariance penalty only to the former, because the penalty would otherwise erase the view-dependent cues (occlusion patterns, apparent hero scale, screen-area fraction under different FOVs) that distinguish a good from a bad shot. We added a calibration/verification head that predicts descriptor reliability from pose/FOV estimation confidence so the director head can down-weight unreliable geometry under novel optics, accepting the extra calibration loss term. A domain expert would NOT describe this as a variant of DirCast's identity-embedding director nor of standard multi-view retrieval: the descriptor is a *geometric sufficient statistic of the camera-action relation*, and the invariance penalty is what makes it topology-agnostic — neither of which appears in DirCast or in retrieval-style camera selection.

**Why this measures what we claim.** The focus-relative descriptor `d_c` measures *topology-agnostic camera relevance* because we assume the relation between camera pose/FOV and the game-state anchor fully determines how informative a view is; this assumption fails when a rig's camera has unusual optics (e.g., extreme fisheye) where geometry alone under-describes the view, in which case `d_c` reflects geometric proximity rather than true informativeness. The invariance penalty measures *topology invariance of the directing decision* because we assume two descriptors of the same world anchor under different rigs should coincide; this assumption fails when the synthetic renderer introduces view-dependent artifacts, in which case the penalty reflects renderer consistency rather than decision invariance. The cross-topology transfer evaluation measures *generalization of hold-or-switch* because we assume re-mapped Dota 2 matches to novel rigs are representative of deployment rigs; this assumption fails when novel rigs have topologies outside the randomization prior, in which case the metric reflects in-distribution transfer only. The calibration head `rel_c` measures *descriptor reliability* because we assume pose/FOV estimation confidence predicts descriptor accuracy; this assumption fails when estimation confidence is miscalibrated under novel optics, in which case `rel_c` reflects training-set calibration rather than true reliability.

## Contribution

(1) A topology-agnostic camera descriptor combining focus-relative pose, FOV, occlusion, and scene-graph features that replaces per-camera identity embeddings in a streaming video-language director. (2) A synthetic-topology domain-randomization training procedure with an explicit cross-rig descriptor invariance penalty, yielding a hold-or-switch policy whose decision is a function of observable camera-action geometry rather than rig slot. (3) A cross-topology transfer benchmark that re-maps Dota 2 matches onto novel virtual camera rigs to test directing invariance beyond the training rig.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|------------------------|
| Dataset | Dota 2 multi-rig replay benchmark (27 matches, synthetic rigs) | Real human cuts, controllable topologies for transfer. |
| Primary metric | Hold-or-switch accuracy on unseen rigs | Directly measures topology-agnostic directing decisions. |
| Baseline | DirCast (identity-embedding director) | Tests whether identity slots block cross-rig generalization. |
| Baseline | Zero-shot VLM director (GPT-4o) | Tests whether generic reasoning handles novel camera geometry. |
| Baseline | Rule-based geometric director | Tests whether hand-coded geometry suffices without learned invariance. |
| Baseline | Nearest-pose heuristic (same geometric descriptor, no learned fusion) | Tests whether learned fusion plus invariance penalty beats pure geometry. |
| Ablation-of-ours | TIGER-D w/o invariance penalty | Isolates the contribution of the descriptor-invariance signal. |
| Ablation-of-ours | TIGER-D w/o calibration head | Isolates the contribution of reliability weighting. |
| Ablation-of-ours | TIGER-D w/ invariance penalty on full descriptor | Tests whether collapsing view-dependent cues hurts. |

### Why this setup validates the claim

The central claim is that a focus-relative geometric descriptor plus synthetic-topology randomization yields a directing policy that transfers to unseen rigs, whereas identity-embedding directors cannot. The dataset supplies real human cut decisions re-projected onto synthetic rigs, so the metric—hold-or-switch accuracy on rigs held out from randomization—directly tests generalization rather than in-distribution fit. DirCast tests the sub-claim that per-camera identity slots fail by construction on novel topologies; the zero-shot VLM tests whether generic visual reasoning can substitute for an explicit geometric statistic; the rule-based director tests whether geometry alone, without learned fusion and invariance, is sufficient; the nearest-pose heuristic tests whether the learned fusion and invariance penalty add anything beyond a pure geometric ranking. The ablation removes the invariance penalty while keeping randomization, so any drop isolates the penalty's causal role rather than confounding it with data augmentation; the calibration-head ablation isolates the reliability weighting; the full-descriptor-penalty ablation tests whether the split between invariant and view-dependent parts is necessary. If the method's gain is real, it must appear specifically on held-out topologies and shrink under the ablations; a uniform gain across all subsets would instead suggest a generic capacity or data effect, not topology invariance.

### Expected outcome and causal chain

**vs. DirCast** — On a match re-rendered with a novel 7-camera rig whose poses and FOVs lie outside the original broadcast layout, DirCast produces erratic hold-or-switch decisions because its per-hero identity embedding has no slot for the new cameras and must extrapolate from unrelated slots. Our method instead computes a focus-relative geometric descriptor for each new camera, which is defined for any pose/FOV, so the director head sees familiar geometry. We expect a large accuracy gap on held-out-topology matches and near-parity on rigs resembling the original broadcast layout.

**vs. Zero-shot VLM director (GPT-4o)** — On a fast teamfight where the informative camera is the one with the closest focus-relative pose and least occlusion, the zero-shot VLM tends to hold on a visually salient but geometrically poor camera because it reasons over pixels without an explicit camera-action relation. Our method instead fuses the geometric descriptor and scene-graph context, so the switch logits are driven by the same quantity that determines informativeness. We expect our method to win disproportionately on high-motion, occlusion-heavy segments and to be roughly tied on slow, wide-shot segments.

**vs. Rule-based geometric director** — On a rig with an unusual but valid FOV (e.g., 100° vertical), the rule-based director mis-ranks cameras because its hand-tuned thresholds assume broadcast optics and cannot recalibrate. Our method instead learns the hold-or-switch mapping under randomized FOVs and is regularized by the invariance penalty, so it adapts to the new optics. We expect a moderate but consistent gap on extreme-FOV rigs and parity on rigs within the rule-based director's design range.

**vs. Nearest-pose heuristic** — On a held-out rig where the nearest camera to the action is heavily occluded, the nearest-pose heuristic switches to it because it ignores occlusion and view-dependent cues, whereas our method's learned fusion and view-dependent descriptor part down-weight it. We expect our method to outperform the heuristic on occlusion-heavy and extreme-FOV subsets, with parity on unoccluded, standard-FOV subsets.

**vs. TIGER-D w/o invariance penalty** — On a rig where two cameras see the same world anchor from different poses, the ablated model's descriptors for those cameras drift apart because randomization alone provides only indirect pressure toward agreement. Our method's invariance penalty ties them together, so the hold-or-switch decision is stable across the two views. We expect the ablation to show a concentrated drop on cross-rig consistency subsets, with little difference on single-rig accuracy.

**vs. TIGER-D w/o calibration head** — On a novel rig with high pose/FOV estimation noise, the ablated model treats all descriptors as equally reliable and is misled by noisy geometry, whereas our calibration head down-weights unreliable cameras. We expect the ablation to show a drop on high-noise subsets and parity on low-noise subsets.

**vs. TIGER-D w/ invariance penalty on full descriptor** — On a rig with varying FOVs, the full-descriptor penalty collapses view-dependent cues (apparent hero scale, screen-area fraction), making cameras with different optics indistinguishable, whereas our split penalty preserves them. We expect the ablation to show a drop on extreme-FOV and occlusion-heavy subsets.

### What would falsify this idea

If TIGER-D's gain over DirCast is uniform across held-out and in-distribution rigs rather than concentrated on novel topologies, or if removing the invariance penalty leaves cross-rig consistency unchanged, or if the calibration head provides no benefit under injected pose/FOV noise, then the topology-invariance claim is wrong and the descriptor is merely a generic capacity boost.

## References

1. DirCast: Live Esports Directing and Commentary with a Single Streaming Video-Language Model
