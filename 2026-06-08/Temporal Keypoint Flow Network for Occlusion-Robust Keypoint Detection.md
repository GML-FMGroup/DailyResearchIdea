# Temporal Keypoint Flow Network for Occlusion-Robust Keypoint Detection

## Motivation

Existing keypoint detection methods, such as SKIL, treat each frame independently and thus lose keypoints under occlusion because the detector lacks temporal context. While Motion Tracks demonstrates that short-horizon 2D trajectories provide dense correspondences robust to occlusions, it applies trajectory prediction as a separate pipeline after detection. We argue that integrating trajectory prediction into the detection process—learning both tasks jointly—enforces temporal consistency as an inductive bias, enabling the detector to rely on motion cues when appearance is unreliable.

## Key Insight

Temporal consistency of keypoint motion provides an inductive bias that compensates for missing appearance cues under occlusion because the physical world imposes smooth, locally linear motion over short horizons, and jointly learning detection and flow forces the detector to exploit this structure.

## Method

### Temporal Keypoint Flow Network for Occlusion-Robust Keypoint Detection

(A) **What it is:** TKF-Net is a convolutional neural network that, given a video stream, simultaneously outputs per-frame keypoint heatmaps and dense optical flow between consecutive frames. The keypoints are then propagated forward using the predicted flow, and a consistency loss ensures that propagated keypoints align with detections in the next frame.

(B) **How it works:**
```
Input: Video frames I_t (t=1..T)
1. Encoder (shared CNN): f_t = ResNet18(I_t)  # feature maps for each frame
2. Detection head (deconv + softmax): h_t = argmax over heatmap H_t = Softmax(Conv1x1(Deconv(f_t)))
   Deconv: 2 transposed conv layers, kernel=4, stride=2, ReLU; output channels: 64 -> 64 -> 1
3. Flow head (conv): grid_t = FlowNetS(f_t, f_{t+1})  # FlowNetS with 6 refinement steps, output 2-channel flow
4. Warp: hat(H_{t+1}) = Warp(H_t, grid_t)  # bilinear sampling, grid normalized to [-1,1]
5. Consistency loss: L_cons = MSE( hat(H_{t+1}), H_{t+1} )  # encourage flow to predict next heatmap
6. Optional geometric consistency: L_cyc = MSE( Warp( Warp(H_t, grid_t), -grid_t ), H_t )  # cycle consistency
7. Total loss: L = L_det + lambda1 * L_cons + lambda2 * L_cyc  # L_det is heatmap L2 loss with Gaussian ground truth (sigma=3 px)
   Hyperparameters: lambda1=1.0, lambda2=0.5, learning rate 1e-4, Adam optimizer.
```
**Load-bearing assumption:** This design relies on the assumption that optical flow between consecutive frames is locally linear and invertible, so that warping a heatmap by flow preserves keypoint identity. To verify this, during validation we compute the endpoint error (EPE) of predicted flow on occluded regions (pixels where keypoint is absent). If EPE exceeds 5 pixels, we flag the warp as unreliable and fall back to the detection heatmap alone. In training, the consistency loss is masked for pixels with EPE > 5 to avoid penalizing invalid warps.

(C) **Why this design:** We chose a shared encoder rather than separate networks for detection and flow because joint features allow the two tasks to mutually reinforce each other—flow features guide detection under occlusion, and detection features refine flow boundaries—accepting that the encoder may need to trade off between appearance and motion sensitivity. We used a consistency loss (L_cons) instead of a separate tracking stage because it directly enforces that the detection head’s output is predictable from motion, which is the key inductive bias. We added a cycle consistency loss (L_cyc) to prevent flow from collapsing to a trivial solution (e.g., zero flow) while still allowing occlusions to be handled; without cycle consistency, the flow could simply predict zero and minimize L_cons, which would not propagate keypoints. This design contrasts with Motion Tracks, which uses a single trajectory prediction head and does not combine detection and flow in a coupled loss; a domain expert would not describe our method as a variant of Motion Tracks because our detection head is explicitly trained to be temporally consistent via the flow, not just predicting trajectories as an output.

(D) **Why it measures what we claim:** The consistency loss L_cons measures temporal consistency of keypoint detections because it assumes that the predicted flow faithfully captures the motion of keypoints; this assumption holds for smooth, short-horizon motion where the flow is locally linear. When occlusion occurs and the flow prediction is uncertain, L_cons forces the detection head to either produce a consistent detection or indicate uncertainty (low heatmap peak). The cycle consistency L_cyc measures geometric consistency because it assumes forward-backward flow is invertible; this assumption fails at motion boundaries where flow discontinuities exist, in which case L_cyc penalizes areas where the flow is not locally invertible, biasing the model to prefer simpler motions. The learned flow itself operationalizes the 'temporal keypoint identity' concept because it tracks how pixels move between frames; when a keypoint is occluded, the flow predicts where it should reappear, and the detection head is trained to output a high heatmap at that location, effectively measuring the model's ability to maintain identity across occlusions.

## Contribution

(1) A novel neural architecture that jointly learns keypoint detection and short-horizon flow prediction with a consistency loss that enforces temporal coherence as an inductive bias for detection.
(2) Empirical finding that integrating trajectory prediction into detection substantially improves keypoint recovery under occlusion compared to independent detection or post-hoc tracking.
(3) A new evaluation protocol for occlusion-robust keypoint detection using synthetic occlusions on the PartNet-Mobility dataset.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Robot manipulation with occlusions | Tests keypoint tracking under occlusion |
| Primary metric | Keypoint tracking accuracy (PCK) | Measures temporal consistency directly |
| Additional metric | Flow endpoint error (EPE) on occluded frames | Validates load-bearing assumption |
| Baseline 1 | Keypoint detector (no flow) | Isolates benefit of flow consistency |
| Baseline 2 | Motion Tracks (trajectory head) | Compares trajectory vs flow-based tracking |
| Baseline 3 | FlowRetrieval (retrieval-based) | Tests retrieval vs learned flow consistency |
| Baseline 4 | Keypoint detector + FlowNet post-processing | Isolates benefit of joint training |
| Ablation | TKF-Net w/o cycle consistency | Measures contribution of cycle loss |

### Why this setup validates the claim
This combination of dataset, baselines, and metrics forms a falsifiable test of the central claim that jointly learning keypoint detection and optical flow with consistency losses improves temporal keypoint identity across occlusions. The dataset, featuring occlusions in robot manipulation, creates scenarios where the consistency loss should force the model to maintain keypoint identity despite temporary disappearance. The ablation (w/o cycle consistency) isolates the effect of geometric consistency. Comparing against a pure keypoint detector tests whether flow helps; against Motion Tracks tests whether explicit flow is better than trajectory prediction; against FlowRetrieval tests whether learned consistency outperforms retrieval-based matching; against separate FlowNet tests whether joint training is beneficial. The primary metric (PCK) directly measures alignment of predicted keypoints with ground truth, reflecting the method's ability to track identities. The additional metric (EPE on occluded frames) directly validates the load-bearing assumption that flow is accurate under occlusion. If our method outperforms all baselines especially under occlusion, and EPE remains low, the claim gains support; if the gain is uniform or EPE is high, the claim is weaker.

### Expected outcome and causal chain

**vs. Keypoint detector (no flow)** — On a frame where a keypoint is occluded by another object, the pure detector outputs a false positive (e.g., a background blob) because it has no motion context to infer the correct location. Our method instead uses flow to propagate the keypoint from previous frames into the occlusion region, maintaining a high heatmap at the true location; therefore, we expect a noticeable gap on occluded frames (e.g., 20% higher PCK) but parity on non-occluded frames.

**vs. Motion Tracks** — On a case where a keypoint briefly disappears due to fast motion, Motion Tracks' single trajectory head may drift or interpolate incorrectly because it lacks local flow cues. Our method uses dense flow to warp the heatmap, capturing the exact motion of each pixel, so the keypoint reappears accurately after occlusion; we expect a 15% advantage on frames with fast motion or occlusion, with similar performance on slow, smooth sequences.

**vs. FlowRetrieval** — On a scenario where the occlusion pattern is atypical (not in the retrieval database), FlowRetrieval fails to find a matching flow field and predicts erroneous keypoints. Our method learns flow directly from the video, adapting to novel patterns without retrieval; we expect a 10-25% higher PCK on out-of-distribution occlusion instances, while on typical occlusions both perform similarly.

**vs. Keypoint detector + FlowNet post-processing** — When a keypoint is occluded for several frames, the separate pipeline first detects keypoints independently (which may fail) and then applies FlowNet to propagate; the detection errors accumulate. Our method’s joint training allows the detection head to anticipate occlusion and output a weak heatmap, which the flow then refines. We expect our method to achieve 10% higher PCK on occluded sequences, especially for occlusions lasting more than 3 frames.

**Flow accuracy (EPE)** — On occluded frames, we expect our method to maintain an average EPE < 3 pixels, significantly lower than the separarse FlowNet baseline (which may have no detection guidance). This validates the assumption that flow is accurate under occlusion due to joint training.

**vs. TKF-Net w/o cycle consistency** — On a sequence with repetitive motion (e.g., back-and-forth), omitting cycle consistency leads to flow collapsing to a trivial solution (zero flow) because the forward loss can be minimized by predicting no motion. Adding cycle loss forces invertibility, so flow remains accurate; we expect the full model to maintain high PCK (>85%) while the ablation drops to ~50% on such sequences.

### What would falsify this idea
If our method's gain over baselines is uniform across all frames (not concentrated on occluded or fast-motion subsets), or if the ablation without cycle consistency performs equally well on repetitive motions, then the central claim that consistency losses specifically improve temporal identity under occlusion is falsified. Additionally, if the flow EPE on occluded frames is not significantly lower than the separate FlowNet baseline, the load-bearing assumption is violated.

## References

1. Motion Tracks: A Unified Representation for Human-Robot Transfer in Few-Shot Imitation Learning
2. FlowRetrieval: Flow-Guided Data Retrieval for Few-Shot Imitation Learning
3. Flow as the Cross-Domain Manipulation Interface
4. Distilling and Retrieving Generalizable Knowledge for Robot Manipulation via Language Corrections
5. Deep Imitation Learning for Humanoid Loco-manipulation Through Human Teleoperation
6. XSkill: Cross Embodiment Skill Discovery
7. FlowBot++: Learning Generalized Articulated Objects Manipulation via Articulation Projection
8. VideoDex: Learning Dexterity from Internet Videos
