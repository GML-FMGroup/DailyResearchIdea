# SAC-VG: Scale-Action Consistent Video Generation for Action-Critical Detail Preservation

## Motivation

Existing video generators optimized for pixel-level quality, such as DreamGen, fail to preserve fine visual details critical for accurate action recovery because the diffusion training loss does not emphasize action-relevant regions. This leads to unreliable pseudo-actions when videos are used for downstream policy learning. We identify that the generative model's loss treats all pixels equally, causing high-frequency action-discriminative features to be sacrificed for global perceptual quality.

## Key Insight

The consistency of action predictions across spatial scales provides a structural surrogate that forces the generator to encode motion-critical details in a scale-invariant manner, because any detail that disappears under downsampling will cause misalignment in the action space.

## Method

SAC-VG (Scale-Action Consistent Video Generation) extends a video diffusion generator with a consistency loss that aligns action predictions from full-resolution and downsampled versions of the generated video. The generator takes a text prompt and outputs a video; a frozen pretrained inverse dynamics model (IDM) predicts actions from both resolutions; the consistency loss penalizes differences.

**Assumption:** The frozen IDM (ResNet-18, trained on real robot trajectories) generalizes to the generated video distribution; i.e., its action predictions reliably reflect differences in action-relevant visual details across scales. To verify this, we compute the IDM's action prediction accuracy on a calibration set of 100 generated videos (with ground-truth actions from a held-out replay buffer) before training; we require top-1 accuracy > 80% on this set, otherwise we stop and adjust generation quality or replace IDM. This calibration is done once at the start.

```pseudocode
# Input: text prompt T, pretrained video generator G (diffusion U-Net), frozen IDM (ResNet-18, trained on real robot trajectories), downscale factor s=0.5, consistency weight λ=0.1, diffusion loss weight α=1.0, calibration set C of 100 generated videos with ground-truth actions

# Calibration step (before training):
for (V_cal, a_gt) in C:
    a_pred = IDM(resize(V_cal, 224))
    compute accuracy; stop if < 80%

for each training step:
    noise ~ N(0, I)
    V_gen = G(T, noise)                    # generate video frames (shape T x H x W x C)
    V_low = bicubic_downsample(V_gen, s)  # shape T x sH x sW x C
    # Resize both to IDM input resolution (224x224) for action prediction
    V_full_resized = resize(V_gen, 224)
    V_low_resized = resize(V_low, 224)
    a_full = IDM(V_full_resized)           # action sequence (shape T x D)
    a_low = IDM(V_low_resized)
    L_cons = MSE(a_full, a_low)            # scale-consistency loss
    L_diff = diffusion_score_matching(V_gen, T)  # standard denoising loss
    L_total = α * L_diff + λ * L_cons
    update G
```
Hyperparameters: downscale factor s=0.5 (bicubic), consistency weight λ=0.1, IDM input resolution 224x224. Training uses 4 NVIDIA A100 GPUs for 48 hours.

**(C) Why this design**
We chose a frozen pretrained IDM over a jointly learned one because joint training could collapse to trivial solutions where the IDM ignores both scales (e.g., always predicting zeros), rendering the consistency loss meaningless. Accepting the cost that the frozen IDM may not generalize perfectly to the generated video distribution, it provides a stable action reference anchored to real-world dynamics. We verify this generalization via a calibration step to ensure reliability. We used MSE for action consistency over cosine similarity because actions are continuous-valued; MSE directly penalizes magnitude differences, which are more interpretable for downstream policy loss. We adopted a deterministic downsampling (bicubic) rather than random augmentations to ensure that the scale-consistency signal is consistent across training steps, avoiding noisy gradients from stochastic downsampling. The downscale factor of 0.5 was chosen as a moderate reduction that removes high-frequency details while preserving overall structure; a more aggressive factor would cause excessive blurring and overly penalize the generator. The consistency weight λ=0.1 balances detail preservation with generative quality; higher λ suppresses diversity, lower λ fails to enforce the constraint.

**(D) Why it measures what we claim**
The consistency loss L_cons = MSE(a_full, a_low) measures the extent to which action-relevant features are preserved across spatial scales. This operationalizes the concept of "preservation of fine details necessary for action recovery" under the assumption that the frozen IDM is a reliable estimator of actions from video. Specifically, if a high-frequency detail critical for action is lost in the downsampled version, a_low will differ from a_full, increasing L_cons. The assumption that the IDM is reliable fails when the generated video distribution drifts far from the IDM's training data (e.g., novel objects or dynamics); in that case, L_cons may reflect the IDM's inability to handle the new distribution rather than genuine detail loss. Additionally, the assumption that consistent actions imply detail preservation relies on the notion that discrepancies arise solely from missing details; however, it is possible that both resolutions lose details in a correlated way (e.g., both versions blur the same region), leading to a false positive where L_cons is low even though critical details are absent. This failure mode is mitigated by the IDM's sensitivity to motion cues: if both resolutions lack the same motion-relevant feature, the IDM's outputs are likely similar, but the generator still fails to produce the needed detail. Despite these caveats, the scale-consistency loss provides a stronger inductive bias than pixel-level losses alone, as it directly targets action-critical regions. To test this failure mode, we introduce an auxiliary evaluation: we manually apply a global Gaussian blur (σ=2) to generated videos and check if L_cons is artificially low (i.e., < threshold) despite detail loss. We report the proportion of such cases.

## Contribution

(1) A novel scale-action consistency loss for video generation that enforces preservation of action-critical details across spatial resolutions. (2) An end-to-end training framework that integrates a frozen inverse dynamics model with a diffusion-based video generator, demonstrating improved action recovery accuracy. (3) A design principle that generative models should optimize for signal preservation in downstream task spaces, not just pixel quality.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|----------------------|
| Dataset | Robot manipulation videos (e.g., BridgeData) | Robot actions require consistent scale. |
| Primary metric | Action MSE on held-out prompts | Directly measures action consistency. |
| Baseline 1 | DreamGen | Video world model for policy learning. |
| Baseline 2 | Prediction with Action | Joint image-action denoising. |
| Ablation of ours | SAC-VG w/o consistency loss | Isolates effect of scale-consistency. |
| Calibration test (ours) | Accuracy of frozen IDM on 100 generated videos | Verifies IDM reliability on generated distribution. |

### Why this setup validates the claim

The dataset of robot manipulation videos is ideal because the frozen IDM is pretrained on similar real-world trajectories, ensuring stable action references. The primary metric, action MSE on generated videos, directly quantifies how well the generated video preserves action-relevant details: lower MSE indicates better consistency with real-world dynamics. Comparing against DreamGen tests whether our scale-consistency loss provides an advantage over a video generation baseline that lacks such an inductive bias. Comparing against Prediction with Action tests whether decoupling generation and action inference is superior to joint denoising. Finally, the ablation (without consistency loss) isolates the contribution of our core mechanism; if our full method outperforms it, we confirm that the consistency loss is responsible for improvements. The calibration test ensures our load-bearing assumption holds; we report IDM accuracy on a held-out set of 100 generated videos (with ground-truth actions) before training. If accuracy is below 80%, we adjust generation quality (e.g., increase diffusion steps) or replace the IDM. This combination allows a falsifiable test: if the full method fails to beat the ablation on tasks where downsampling removes critical features, the central claim is unsupported.

### Expected outcome and causal chain

**vs. DreamGen** — On a case where the text prompt specifies a fine-grained action (e.g., "grasp a thin screwdriver"), DreamGen may generate a video where the grasping motion is realistic but the screwdriver's orientation or the precise grip is slightly off, because its video generation objective only enforces pixel-level realism. Our method, by penalizing differences in action predictions from full and downsampled videos, forces the generator to preserve motion details critical for action recovery. Consequently, we expect our method to yield lower action MSE on such prompts, especially for tasks with high-precision requirements, while performing comparably on coarse actions (e.g., "push a box").

**vs. Prediction with Action** — On a case where the video background is cluttered (e.g., a table with many objects), Prediction with Action's joint denoising may produce artifacts that confuse the action prediction because it learns action and image distributions simultaneously without a scale-focused signal. Our method, by anchoring action consistency to a frozen, real-world-trained IDM, maintains a clear separation: the generator focuses on video quality while the IDM provides a stable action target. We expect our method to achieve lower action MSE in cluttered scenes, but may see higher variance due to the frozen IDM's mismatch on out-of-distribution backgrounds.

### What would falsify this idea

If our full method shows uniform improvement over the ablation across all subsets (including those where downsampling does not remove critical details), rather than concentrated gains on high-precision tasks, then the scale-consistency loss is not targeting the intended failure mode and the central claim is wrong. Additionally, if the IDM's accuracy on the calibration set is below 80% and we cannot improve generation quality, the assumption fails and the method's rationale is undermined.

## References

1. DreamGen: Unlocking Generalization in Robot Learning through Video World Models
2. Prediction with Action: Visual Policy Learning via Joint Denoising Process
3. HunyuanVideo: A Systematic Framework For Large Video Generative Models
4. How Far is Video Generation from World Model: A Physical Law Perspective
5. Animate Anyone: Consistent and Controllable Image-to-Video Synthesis for Character Animation
6. MagicAnimate: Temporally Consistent Human Image Animation using Diffusion Model
7. Stable Video Diffusion: Scaling Latent Video Diffusion Models to Large Datasets
8. RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control
