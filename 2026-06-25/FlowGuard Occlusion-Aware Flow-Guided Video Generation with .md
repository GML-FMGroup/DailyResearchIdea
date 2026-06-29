# FlowGuard: Occlusion-Aware Flow-Guided Video Generation with Diffusion Models

## Motivation

MVTrack4Gen uses attention-layer features for point tracking supervision, but these features are unreliable under severe occlusion or fast motion because attention maps can be disrupted. This limitation arises because attention-based correspondence is implicitly learned and lacks explicit occlusion reasoning. We address this by replacing the tracking head with an explicit occlusion-aware optical flow estimator (RAFT with occlusion masking) that provides robust point tracks for supervising diffusion-based video generation.

## Key Insight

Explicit occlusion-aware flow estimation yields geometrically consistent point tracks that are provably robust to occlusion, enabling reliable supervision for diffusion models without the fragility of attention-based tracking.

## Method

### FlowGuard: Occlusion-Aware Flow-Guided Video Generation with Diffusion Models

#### Load-bearing assumption (explicit):
Chaining forward optical flow with occlusion masks yields accurate long-term point tracks. To mitigate drift, we replace flow-chaining with a state-of-the-art point tracker (CoTracker) that handles occlusions robustly.

#### Method:

We propose FlowGuard, a dual-branch framework for novel-view video synthesis. The first branch uses a point tracker (CoTracker, Karaev et al., CVPR 2024, with default parameters: window size 15, stride 4, 4 refinement iterations, pretrained on TAP-Vid) to compute dense point trajectories and occlusion masks across the input video. The second branch is a latent video diffusion model conditioned on camera parameters and finetuned using these trajectories as geometric supervision.

```python
# Input: monocular video V of T frames, camera poses (assumed known)

# Stage 1: Occlusion-aware point tracking using CoTracker
# CoTracker outputs tracks of shape (T, N, 2) and visibility masks of shape (T, N)
# We sample N=10000 points uniformly from the first frame.
# CoTracker uses a sliding window approach; we set window length = T for full video.
tracks, visibility = CoTracker(V, queries=points_init, window_len=T)
# tracks: T x N x 2, visibility: T x N (binary, 1=visible, 0=occluded)
# Filter tracks to keep only points with high visibility: e.g., mean visibility > 0.5 over time.
valid = visibility.mean(dim=0) > 0.5  # threshold
filtered_tracks = tracks[:, valid, :]  # shape (T, M, 2) where M <= N

# Stage 2: Finetune diffusion model with flow guidance
# Base model: pretrained video DiT (e.g., from GS-DiT) with camera conditioning via cross-attention
# We add a learnable cross-attention layer that takes point trajectories as additional conditioning
# Specifically, we project the (x,y) coordinates of visible points into a feature map of latent resolution using bilinear interpolation, then add as a bias to the latent features before each transformer block.
# Training objective: L = L_diffusion + lambda * L_flow
# where L_diffusion is the standard noise prediction loss, and L_flow is:
#   - At every K=10 training steps, decode a batch of denoised latents to RGB video (using VAE decoder)
#   - Compute optical flow between consecutive frames using a frozen RAFT-occ (pretrained on FlyingThings3D)
#   - Match the computed flow to the CoTracker tracks: for each visible point at frame t (from CoTracker), sample the flow at its location and compare with the displacement (tracks[t+1] - tracks[t]) using L2 loss.
# The loss is masked by the visibility of the point at frame t and t+1.
# Hyperparameters: N=10000, M varies, lambda=0.1, learning rate=1e-5, finetune for 50k steps, batch_size=4 (gradient accumulation 8)
```

(C) **Why this design**: We chose CoTracker over chained RAFT flow because chaining suffers from drift under long occlusions and fast motion (TAP-Vid benchmark). CoTracker explicitly models long-term correspondence with recurrent refinement and occlusion prediction, providing more robust tracks. The cost is higher memory and slower inference (CoTracker's sliding window). We use both conditioning and a flow loss: conditioning gives global guidance, while the loss enforces pixel-level consistency. The flow loss is computed only every K=10 steps to reduce computational overhead from VAE decoding. Using a frozen RAFT for the loss avoids finetuning the flow estimator, but may propagate errors; we mitigate by masking with CoTracker's visibility scores.

(D) **Why it measures what we claim**: The CoTracker tracks (2D point positions over time) measure geometric consistency under the assumption of brightness constancy and small motion; this assumption fails at occlusion boundaries, but our visibility masks explicitly exclude such points, so the tracks reflect only reliable correspondences. The flow loss measures alignment between generated motion and tracked motion. **The linking assumption is that CoTracker tracks accurately reflect true geometric consistency across frames. This fails when CoTracker produces errors (e.g., on textureless regions or extreme motion), leading to penalizing correct motion. We mitigate by using only points with high mean visibility (threshold 0.5) and by weighting the loss with per-frame visibility scores.** The occlusion masks (from CoTracker) measure the likelihood of a point being invisible or unreliable, serving as a filter to ensure the loss is computed only where the flow assumption holds. Together, the tracks, masks, and loss ensure that the generated video maintains geometrically consistent motion even under occlusion.

## Contribution

(1) A dual-branch framework for novel-view video synthesis that replaces attention-based tracking with explicit occlusion-aware flow estimation for robust geometric supervision. (2) A finetuning strategy that incorporates point trajectories as both conditioning and a flow consistency loss for diffusion models, ensuring pixel-level alignment. (3) A demonstration that explicit occlusion handling improves robustness under occlusion compared to attention-based methods (MVTrack4Gen), evidenced by quantitative metrics and visual results.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | Dynamic Multi-view Dataset (e.g., D-NeRF) and a held-out set of real-world occlusion-heavy scenes (e.g., TAP-Vid test split) | Contains challenging occlusions; TAP-Vid enables quantitative tracking accuracy analysis |
| Primary metric | Temporal Warping Error (TWE) | Measures geometric consistency |
| Baseline 1 | MVTrack4Gen | Attention-based tracking, no occlusion reasoning |
| Baseline 2 | GS-DiT | Video DiT without geometric guidance |
| Baseline 3 | Camera-conditioned diffusion | Camera conditioning only |
| Ablation | Ours w/o flow loss | Tests necessity of pixel-level supervision |

### Why this setup validates the claim
This experimental design isolates the effect of explicit occlusion reasoning and geometric consistency. The dynamic multi-view dataset contains real-world occlusions, where brightness constancy fails. Comparing against MVTrack4Gen tests whether CoTracker-based occlusion handling outperforms attention-based tracking. GS-DiT tests if any geometric supervision is beneficial over pure camera conditioning. The ablation without flow loss reveals the contribution of pixel-level flow matching. Temporal Warping Error (TWE) directly penalizes misaligned motion across frames, making it the right metric to detect inconsistencies that our method aims to reduce. Additionally, we evaluate CoTracker's tracking accuracy on a held-out set (e.g., 5% of frames from the test set) by measuring the percentage of predicted tracks within 4 pixels of ground-truth (as in TAP-Vid). This analysis quantifies the noise level in the supervision signal and helps interpret TWE results.

### Expected outcome and causal chain

**vs. MVTrack4Gen** — On a scene with a fast-moving object occluded by a static foreground, MVTrack4Gen's attention-based tracks drift onto the occluder because attention features blend background and object, producing wrong correspondences. Our method instead uses CoTracker with explicit occlusion reasoning, which reliably maintains tracks through occlusions using its recurrent refinement and visibility predictions. We expect a noticeable gap in TWE on such occlusion-heavy subsets (e.g., >20% improvement) but parity on simple motions with minimal occlusion.

**vs. GS-DiT** — On a novel-view trajectory that requires synthesizing consistent object motion across frames, GS-DiT lacks any geometric constraint and produces wavy, inconsistent motion because its diffusion process is unguided. Our method injects point trajectories as conditioning, which anchors the generation to physically plausible paths. Thus we expect significantly lower TWE (e.g., halved) on diverse camera paths, with gains most pronounced on long-range motion where drift accumulates.

**vs. Camera-conditioned diffusion** — On a turn-around sequence where a figure moves out of view, camera-only models hallucinate implausible motion because conditioning on pose does not enforce scene flow. Our flow loss directly penalizes deviation from tracked points, forcing realism even when objects exit. We expect a moderate TWE reduction overall, but specifically a large drop on frames where points become newly visible due to camera motion (e.g., >30% improvement).

### What would falsify this idea
If our method shows uniform TWE improvements across all scene types, including those with little occlusion, rather than the improvement being concentrated on heavy-occlusion cases, then the claim that explicit occlusion reasoning is the key driver would be falsified. Similarly, if the ablation without flow loss matches our full method, the pixel-level consistency loss is unnecessary. Additionally, if our analysis reveals that CoTracker tracks have high error (e.g., >50% of points have drift >8 pixels) on the evaluation dataset, then the supervision signal is too noisy and the linking assumption is violated.

## References

1. MVTrack4Gen: Multi-View Point Tracking as Geometric Supervision for 4D Video Generation
2. GS-DiT: Advancing Video Generation with Dynamic 3D Gaussian Fields through Efficient Dense 3D Point Tracking
