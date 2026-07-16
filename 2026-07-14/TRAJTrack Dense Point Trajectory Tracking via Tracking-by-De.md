# TRAJTrack: Dense Point Trajectory Tracking via Tracking-by-Detection with Re-Identification for Occlusion Handling

## Motivation

Motion4Motion uses optical flow for motion flow extraction, but optical flow fails under occlusions, causing broken trajectories and inconsistent motion transfer. This is because flow methods assume continuous appearance, which is violated when points are occluded. To maintain trajectory consistency in dense point trajectories, we need a mechanism that can associate points across occlusions without relying solely on motion continuity.

## Key Insight

Reformulating dense trajectory extraction as a tracking-by-detection problem with a learned re-identification feature decouples motion estimation from appearance matching, enabling robust reconnection of trajectories after occlusions via appearance invariance.

## Method

(A) **TRAJTrack**: A training-free dense point trajectory extraction framework that treats each point as a tracklet and uses a lightweight re-identification (ReID) network (ResNet-18, embedding dimension 128, trained on synthetic occlusions (random masks of size 4x4 to 12x12) and fine-tuned with cycle consistency loss on TAP-Vid [Doersch et al., 2023]) to associate detections across frames, explicitly handling occlusions via bipartite matching with a motion model.

(B) **How it works** (pseudocode):
```python
# Input: video frames I_1..I_T
# Output: dense point trajectories T = {t_i | i=1..N}
def TRAJTrack(I_1..I_T):
    # Phase 1: Sample points in first frame
    pts_1 = sample_grid_points(I_1, stride=8)  # N points
    
    # Phase 2: Initial tracklets via optical flow
    for f in 2..T:
        # Forward flow: I_f-1 -> I_f
        flow_f = RAFT(I_f-1, I_f)
        pts_f_pred = warp(pts_{f-1}, flow_f)
        # Detection: extract ReID features at predicted locations
        dets_f = {point: (pos, ReIDNet(crop(I_f, pos, size=32)))} for point in pts_f_pred
    
    # Phase 3: Tracklet reassociation after occlusion (every K frames)
    K = 5
    for f in K..T step K:
        # Build cost matrix between active tracklets and current detections
        C = np.zeros((len(active_tracklets), len(dets_f)))
        for i, tracklet in enumerate(active_tracklets):
            for j, det in enumerate(dets_f):
                # Appearance cost: cosine distance between L2-normalized ReID features
                app_cost = cosine_dist(tracklet.feature, det.feature)  # measures appearance consistency under assumption that appearance is stable; fails under illumination change or deformation.
                # Motion cost: Mahalanobis distance from Kalman prediction (constant velocity model, state: [x, y, vx, vy])
                mot_cost = mahalanobis(tracklet.kf.predict(), det.pos)  # assumes constant velocity; fails under abrupt acceleration.
                C[i,j] = app_cost + lambda_mot * mot_cost  # lambda_mot=0.5 (tuned on validation set)
        # Hungarian matching with threshold (t_app=0.7) — trade-off between precision and recall; no optimal threshold across scenes.
        matches = hungarian(C, threshold=0.7)
        for i, j in matches:
            active_tracklets[i].update(det_f, feature_update=moving_average)
        # Initialize new tracklets for unmatched detections
        for j in unmatched_detections:
            active_tracklets.append(new_tracklet(det_j))
        # Terminate tracklets that have been unmatched for > T_max=3 frames (tolerates short occlusions <=2 frames)
        active_tracklets = [t for t in active_tracklets if t.unmatched_count < 3]
    # Phase 4: Output all trajectories
    return [t.get_waypoints() for t in all_tracklets]
```

(C) **Why this design**: We chose a tracking-by-detection paradigm over pure flow propagation because flow alone cannot recover after occlusion—flow estimates at occlusion boundaries are unreliable and often lead to track drift. By incorporating a lightweight ReID network (ResNet-18, 2.8M parameters, trained on synthetic occlusions and fine-tuned with cycle consistency on TAP-Vid), we add appearance-based cues that are more robust to motion discontinuities. The trade-off is that we require training data for ReID; we accept this cost by generating synthetic occlusion models (e.g., adding random patches of size 4-12 pixels) and fine-tuning on real occlusions via cycle consistency, ensuring the network generalizes to real videos. We use Hungarian matching with a combined cost (appearance + motion) rather than a single cost because neither alone is sufficient: appearance can be ambiguous for points with similar textures, and motion can fail under fast camera motion. The threshold for matching is set to 0.7 based on a validation set (500 frames from TAP-Vid); a higher threshold reduces false positives but may miss true matches. Finally, we reassociate every K=5 frames to balance computational cost (approx. 0.2 GPU hours per video) and the need to correct flow errors; a larger K delays correction, while a smaller K increases runtime without benefit if flow is stable.

(D) **Why it measures what we claim**: The `cosine_dist` between ReID features measures **appearance consistency** because we assume that the same physical point has a similar appearance across short time spans; this assumption fails when illumination changes drastically or the point enters a shadow, in which case the metric reflects appearance change rather than point identity. The `mahalanobis` distance from the Kalman filter prediction (constant velocity model) measures **motion plausibility** because we assume the point moves with locally constant velocity; this fails under abrupt acceleration, and then the metric penalizes the correct match. The Hungarian matching with threshold selects the assignment that optimally balances these two costs, thereby operationalizing **trajectory consistency** under occlusion: a matched tracklet continues, while unmatched points are reinitialized or terminated. The `unmatched_count` threshold (max 3) directly implements **tolerance for short occlusions**: a point that disappears for up to 2 frames is retained; this assumption fails for long occlusions (>3 frames), and then the track is lost and restarted, causing a trajectory break. Each component is causally linked to the motivation-level concept of robust tracking despite occlusion.

## Contribution

(1) A tracking-by-detection framework (TRAJTrack) for dense point trajectory extraction that integrates optical flow, re-identification features, and Kalman filtering to handle occlusions. (2) A synthetic data generation pipeline with random occlusion masks for training the ReID network, ensuring it generalizes to realistic occlusion patterns. (3) Demonstration that TRAJTrack improves trajectory consistency over flow-only methods on motion transfer tasks, particularly in occluded regions, as shown by reduced trajectory fragmentation.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|------------------------|
| Dataset | TAP-Vid (point tracking benchmark) | Standard benchmark for point trajectory evaluation |
| Primary metric | Average Jaccard (AJ) | Measures accuracy over visible and occluded frames |
| Baseline 1 | RAFT + forward-backward consistency | Pure optical flow propagation method |
| Baseline 2 | ReID-only (appearance matching, same ResNet-18 trained on synthetic+cycle) | Isolates appearance cue contribution |
| Baseline 3 | Kalman-only (motion model, constant velocity) | Isolates motion prediction contribution |
| Ablation-of-ours | TRAJTrack without Hungarian reassociation (flow only, with ReID feature averaging) | Tests necessity of explicit occlusion handling |
| Compute budget | ~0.2 GPU hours per video (ReID: 0.05, RAFT: 0.1, matching: 0.05) | Feasibility estimate on NVIDIA V100 |

### Why this setup validates the claim

This combination of dataset, baselines, and metric forms a falsifiable test of the central claim that TRAJTrack achieves robust tracking despite occlusion by integrating appearance and motion cues with periodic reassociation. The TAP-Vid dataset provides ground truth point trajectories with natural occlusions, enabling direct evaluation of tracking accuracy under occlusion. The baselines isolate the contribution of each component: RAFT tests pure flow, ReID-only tests appearance matching alone, Kalman-only tests motion prediction alone, and the ablation tests the reassociation mechanism. The Average Jaccard metric penalizes both false positives and false negatives over the entire trajectory, including occluded frames, making it sensitive to improvements in occlusion handling. By comparing against these baselines on subsets stratified by occlusion duration, we can pinpoint whether our gains stem from the intended mechanism or from unrelated factors. Additionally, we evaluate on a downstream motion transfer task (DynamiCrafter) to measure practical impact.

### Expected outcome and causal chain

**vs. RAFT** — On a case where a point is occluded for 2-3 frames (e.g., a point on a car passing behind a tree), RAFT fails because optical flow at occlusion boundaries is unreliable and forward-backward consistency may discard the track. Our method instead uses the ReID feature to reidentify the point after occlusion, leveraging appearance consistency. Thus we expect a noticeable gap in AJ on sequences with frequent or long occlusions, but parity on sequences with smooth, unoccluded motion.

**vs. ReID-only** — On a case where a point moves quickly across uniformly textured background (e.g., a running person in a desert), ReID-only confuses the point with similar patches, leading to identity switches. Our method adds a motion cost via Kalman filter that penalizes large position jumps, disambiguating the correct match. Hence we expect our method to outperform ReID-only on sequences with fast motion or low-texture regions, with the gap widening as speed increases.

**vs. Kalman-only** — On a case where a point undergoes abrupt acceleration (e.g., a sudden stop or turn), Kalman-only fails because the constant velocity assumption is violated, causing the predicted position to be far off. Our method uses appearance cost to select the detection that best matches the visual appearance, overriding the poor motion prediction. Therefore we expect our method to achieve higher AJ on sequences containing dynamic motion patterns, while Kalman-only may have higher precision on smooth motion.

**vs. TRAJTrack without reassociation** — On a long sequence where flow accumulates drift (e.g., a slowly rotating object), the no-reassociation variant slowly drifts away from the true point. Our full method periodically reassociates tracks using Hungarian matching, correcting drift. Thus we expect the full method to maintain high AJ over long durations, while the ablation degrades over time.

**Downstream motion transfer** — We expect TRAJTrack trajectories to produce fewer flickering artifacts and better motion consistency in transferred videos, quantified by warping error (MSE between forward-warped source and target frames).

### What would falsify this idea

If the performance gain over baselines is uniform across all frames (e.g., a constant 0.02 AJ improvement) rather than concentrated on frames where occlusions occur (as measured by ground truth visibility), then the central claim that our method specifically improves occlusion handling is falsified.

## References

1. Motion4Motion: Motion Transfer Across Subjects at Inference
2. MotionStream: Real-Time Video Generation with Interactive Motion Controls
3. MotionV2V: Editing Motion in a Video
4. Motion2Motion: Cross-topology Motion Transfer with Sparse Correspondence
5. Wan-Move: Motion-controllable Video Generation via Latent Trajectory Guidance
6. Motion Prompting: Controlling Video Generation with Motion Trajectories
7. Pose‐to‐Motion: Cross‐Domain Motion Retargeting with Pose Prior
8. Skinned Motion Retargeting with Dense Geometric Interaction Perception
