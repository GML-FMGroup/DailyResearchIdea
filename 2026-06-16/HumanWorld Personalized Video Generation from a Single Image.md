# HumanWorld: Personalized Video Generation from a Single Image via World Model Fine-Tuning

## Motivation

Current personalized video generation methods (e.g., DreamVideo, Still-Moving) rely on separate subject personalization from text-to-image models and motion adaptation, leading to inconsistency and dependence on pre-trained text-to-image models that may not generalize to new subjects. DreamGen's pipeline for robot videos shows that fine-tuning a video world model with frame replacement and task conditioning can directly adapt to a new embodiment from a single start image, bypassing separate personalization steps. However, this has not been applied to human action videos due to domain shift and need for precise motion control. We identify that the structural property enabling DreamGen—replacing the first frame and conditioning on action while exploiting the world model's temporal consistency—transfers to human videos if the fine-tuning data includes pose-conditioned human actions.

## Key Insight

Fine-tuning a video world model on human action video pairs with frame replacement and pose conditioning implicitly learns a mapping from a single subject image and a pose sequence to a temporally consistent video, because the world model's temporal prior, when forced to adapt to new appearance via frame replacement, becomes a generative model that can autoregressively extend the appearance under the given motion constraints.

## Method

### (A) What it is
HumanWorld is a fine-tuning framework that adapts a pre-trained video world model (e.g., Stable Video Diffusion) to generate personalized videos of a new human subject from a single input image and a desired pose sequence, without requiring any pre-trained text-to-image model or subject-specific fine-tuning. Input: a single subject image (start frame) and a sequence of 2D body keypoints (17 joints). Output: a video of that subject performing the motion specified by the pose sequence.

### (B) How it works
The method consists of three phases: Data Preparation, Fine-Tuning, and Inference. Below is the pseudocode describing the core fine-tuning and inference procedures.

```pseudocode
# Hyperparameters:
# lr = 1e-5, batch_size = 16, num_frames = 16, num_steps = 100000
# pretrained_model = StableVideoDiffusion (SVD) with temporal layers
# LoRA rank = 16, target_modules = ['to_q', 'to_k', 'to_v', 'to_out']
# prior_preservation_weight = 0.1

# Phase 1: Data Preparation
# Collect dataset D of human action videos (e.g., Human3.6M or custom) with pose annotations.
# For each video clip V = {I_1, I_2, ..., I_T} and corresponding pose sequence P = {p_1,...,p_T}:
#   - Extract the first frame I_1 as the original start.
#   - Randomly select a subject image S from a pool of static human images (foreground on white background).
#   - Apply appearance augmentation to S: random hue shift in [-0.05, 0.05], saturation shift in [-0.1, 0.1], lighting intensity shift in [-0.1, 0.1].
#   - Create modified clip V' = {S, I_2, ..., I_T} (replace first frame).
#   - Store tuple (V', P, I_1) as training sample.

# Phase 2: Fine-Tuning
# Initialize model M from pretrained SVD weights.
# For each training step:
#   Sample batch of (V', P, I_1) from D.
#   Add noise to V' according to diffusion schedule (e.g., linear, 1000 timesteps).
#   Condition M on pose sequence P via cross-attention (pose tokens projected to query dimension).
#   Predict the noise added to V' (i.e., denoise all frames except first which is fixed as S).
#   Compute MSE loss between predicted noise and actual noise: L_rec = MSE(noise_pred, noise_actual).
#   Compute prior preservation loss: L_prior = MSE(M_original(V) - M(V)), where M_original is frozen pretrained model and V is original clip (no frame replacement).
#   Total loss: L = L_rec + prior_preservation_weight * L_prior.
#   Update M's LoRA parameters via Adam optimizer with lr = 1e-5.
#   (Optional: Exclude ambiguous poses where keypoint confidence < 0.5 from MPJPE evaluation.)

# Phase 3: Inference
# Given start image S_new and target pose sequence P_target = {p_1,...,p_T}:
#   Set first frame = S_new.
#   For t = 2 to T:
#       Denoise frame I_t from random noise conditioned on [S_new, P_target] via M (autoregressive: use generated frames as context).
#   Return video {S_new, I_2, ..., I_T}.
```

**Load-bearing assumption**: The fine-tuning dataset (e.g., Human3.6M) contains sufficient appearance diversity that the model, via frame replacement, learns to generalize to arbitrary unseen subject images from a single start frame. To mitigate the risk of limited diversity, we apply appearance augmentation (as above) and incorporate a prior preservation loss to prevent overfitting to training appearances. We will verify this assumption by measuring LPIPS on test subjects with significantly different appearance (e.g., different clothing, backgrounds).

**Computational budget**: Training requires 4 V100 GPUs for approximately 1 week (100,000 steps). LoRA reduces memory to ~12 GB per GPU.

### (C) Why this design
We chose to use a pretrained video diffusion model (SVD) over training from scratch because it already encodes a strong temporal prior from large-scale video data, reducing the amount of human action data needed. We chose frame replacement over subject-specific fine-tuning (as in DreamVideo) because it forces the model to handle arbitrary new appearances without per-subject optimization, enabling zero-shot personalization. We chose explicit 2D keypoint pose sequences over abstract task descriptions (as in DreamGen) because keypoints provide precise motion control and are readily extractable from existing datasets, avoiding the need for inverse dynamics models. The trade-off is that pose conditioning requires pose annotations during training, which may be costly, but existing human motion datasets like Human3.6M provide them. The cost of frame replacement is that the model must learn to separate appearance from motion from limited data, but the world model's temporal layers already capture motion dynamics, so the fine-tuning mainly adapts appearance injection. We opted for autoregressive generation over joint full-frame generation to ensure consistency with the initial image, accepting a slight increase in inference time.

### (D) Why it measures what we claim
The computational quantity `MSE loss between predicted noise and actual noise` operationalizes `temporal consistency and appearance adherence` because the model is trained to reconstruct the original video frames (except the replaced first frame) given the new appearance S and the pose sequence P. This loss measures the model's ability to generate frames that are both visually consistent with S (by learning to propagate S's appearance through time) and motionally faithful to P (by conditioning on pose). The assumption that `pose sequence fully determines motion` implies that the model's error is solely due to failure in appearance propagation or motion coherence; this assumption fails when the pose sequence is noisy or incomplete (e.g., missing hand gestures), in which case the model may produce plausible but incorrect motions that still minimize the loss due to the prior of the training data. The replacement of the first frame ensures that the model cannot simply memorize the original appearance but must adapt to S; the loss on subsequent frames thus measures how well the model transfers S's properties under the pose constraint. If the training data contains diverse appearances, the model learns to generalize to novel S, making the loss a valid proxy for zero-shot personalization quality. To mitigate the failure mode, we compute pose accuracy (MPJPE) separately and exclude ambiguous poses (keypoint confidence < 0.5) from training and evaluation.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|----------------------|
| Dataset | Human3.6M with pose annotations | Standard benchmark, provides diverse poses and appearances. |
| Primary metric | Average of LPIPS and MPJPE (MPJPE on joints with conf. >0.5) | Combines appearance fidelity and pose accuracy. |
| Baseline 1 | DreamVideo (per-subject fine-tuning) | Requires multiple images, tests zero-shot vs. few-shot. |
| Baseline 2 | Still-Moving (adapter-based) | Tests generalization to unseen poses from static images. |
| Baseline 3 | Vanilla SVD (no pose conditioning) | Isolates benefit of pose signal. |
| Ablation | HumanWorld without pose conditioning | Measures contribution of pose conditioning. |

### Why this setup validates the claim

The chosen dataset (Human3.6M) contains a wide variety of human subjects and actions with accurate pose annotations, allowing us to test both appearance generalization and motion control. The primary metric (LPIPS + MPJPE) directly operationalizes the two core requirements: visual consistency with the given subject and kinematic fidelity to the target pose sequence. DreamVideo requires per-subject fine-tuning with multiple images; a zero-shot alternative that matches its quality on a single image would confirm strong appearance transfer. Still-Moving adapts using frozen video clips; outperforming it on novel pose sequences would prove that explicit pose conditioning is more effective than interpolation. Vanilla SVD lacks any pose input; beating it on pose accuracy validates that our method indeed controls motion. The ablation (removing pose conditioning) isolates whether any gains come from the injected pose signal or from fine-tuning alone. This combination creates a clear falsifiable test: if HumanWorld does not significantly outperform Still-Moving and Vanilla SVD on pose accuracy, or if its LPIPS/MPJPE gap relative to DreamVideo is equivalent to the ablation, then the central claim of zero-shot personalization with precise motion control is unsupported. To address the load-bearing assumption about appearance diversity, we will additionally evaluate on a held-out set of subjects with significantly different appearance (e.g., from DeepFashion or in-the-wild images) to verify generalization.

### Expected outcome and causal chain

**vs. DreamVideo** — On a novel subject seen only once (single image), DreamVideo must per-subject fine-tune on that image, which often leads to overfitting and unnatural motion due to insufficient data. HumanWorld instead leverages the pretrained world model's temporal prior and replaces only the first frame, so it retains natural motion dynamics from the pretrained model while adapting appearance via the fine-tuning on diverse subjects. Thus we expect HumanWorld to achieve comparable LPIPS (visual similarity) to DreamVideo fine-tuned on the same single image, but with better MPJPE (pose accuracy) because DreamVideo's fine-tuning may distort motion patterns. The observable signal: a smaller LPIPS gap but a noticeable MPJPE advantage (e.g., >20% reduction) for HumanWorld on unseen subjects.

**vs. Still-Moving** — Still-Moving adapts by injecting appearance into frozen videos; on a pose sequence not present in its frozen set, it must interpolate between videos, often producing motion artifacts or static blobs. HumanWorld explicitly conditions on the full pose sequence via cross-attention, so each frame's motion is directly controlled by the target pose. On a novel pose (e.g., a complex dance), Still-Moving fails to generate coherent motion, while HumanWorld produces accurate limb positions. We expect a large gap in MPJPE (e.g., >30% reduction) on the most novel poses, with comparable LPIPS on simple poses but better on complex ones.

**vs. Vanilla SVD** — Vanilla SVD with no pose conditioning generates videos with random motion, leading to high MPJPE (near chance level ~15-20 pixels). HumanWorld explicitly guides motion via pose, so MPJPE should be low (<5 pixels) on seen pose sequences and still low on novel poses due to generalization of temporal layers. The gap in MPJPE will be dramatic, while LPIPS may be similar if the appearance injection works. This baseline confirms that pose conditioning is necessary for motion control.

### What would falsify this idea

If HumanWorld's pose accuracy (MPJPE) on novel poses is not significantly lower than that of Vanilla SVD (i.e., the pose signal provides no benefit), or if its LPIPS on unseen subjects is no better than Still-Moving while MPJPE is similar, then the central claim of zero-shot personalization with pose control fails.

## References

1. DreamGen: Unlocking Generalization in Robot Learning through Video World Models
2. VideoJAM: Joint Appearance-Motion Representations for Enhanced Motion Generation in Video Models
3. Still-Moving: Customized Video Generation without Customized Video Data
4. MagicEdit: High-Fidelity and Temporally Coherent Video Editing
5. Dream Video: Composing Your Dream Videos with Customized Subject and Motion
6. IP-Adapter: Text Compatible Image Prompt Adapter for Text-to-Image Diffusion Models
7. VideoBooth: Diffusion-based Video Generation with Image Prompts
8. Emu Video: Factorizing Text-to-Video Generation by Explicit Image Conditioning
