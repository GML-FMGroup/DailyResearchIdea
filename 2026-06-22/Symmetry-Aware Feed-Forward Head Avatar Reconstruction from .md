# Symmetry-Aware Feed-Forward Head Avatar Reconstruction from a Single Image

## Motivation

Current feed-forward head avatar reconstruction methods (e.g., SpatialAvatar) rely on training data (VFHQ, synthetic) biased towards frontal views and neutral expressions, causing failure on extreme poses. This bias is a structural issue: the training distribution lacks coverage of extreme poses, yet these methods lack a mechanism to enforce consistency for unseen poses. We propose to leverage facial symmetry—a universal property of human faces—as a dense self-supervised constraint that can be applied at test time without paired data, enabling generalization beyond the training distribution.

## Key Insight

Facial symmetry provides a pixel-wise correspondence between left and right face halves that holds for any pose and expression (under weak perspective), allowing direct photometric supervision without requiring ground-truth 3D for out-of-distribution samples.

## Method

(A) **What it is**: SymAvatar, a feed-forward network that reconstructs a 3D Gaussian head avatar from a single image, augmented with a symmetry-guided refinement that enforces consistency between the rendered avatar and its mirror image. Input: single RGB image. Output: FLAME-bound 3D Gaussians and rendered novel views.

(B) **How it works** (pseudocode):
```python
# SymAvatar: Feed-forward + Symmetry Refinement
def sym_avatar(input_img):
    # Feed-forward reconstruction (similar to SpatialAvatar)
    params = feed_forward_UNet(input_img)  # predicts FLAME params and Gaussian offsets
    avatar = construct_avatar_from_params(params)  # FLAME-bound 3D Gaussians

    # Symmetry constraint as self-supervised refinement
    for step in range(num_refine_steps=100):
        # Render the avatar from the estimated frontal view (canonical pose)
        render_frontal = render(avatar, camera_pose=frontal)
        # Create mirror image by horizontal flip
        render_mirror = torch.flip(render_frontal, dims=[-1])
        # Split render into left and right halves (assuming a midline)
        left_half = render_frontal[:, :, :, :height//2]
        right_half = render_frontal[:, :, :, height//2:]
        left_mirror_half = render_mirror[:, :, :, :height//2]
        # Symmetry loss: left_half should match mirror's left_half (which is original right)
        loss_sym = L1(left_half, left_mirror_half) + perceptual_loss(left_half, left_mirror_half)
        # Backprop to update Gaussian parameters
        loss_sym.backward()
        optimizer.step(learning_rate=1e-3)
        # Anti-spike regularization (as in SpatialAvatar) to maintain Gaussian layout
        reg_loss = anti_spike_regularization(avatar.gaussians)
        reg_loss.backward()
    return avatar
```
Hyperparameters: refinement steps=100, learning rate=1e-3, perceptual loss weight=0.1.

(C) **Why this design**: We chose symmetry-based self-supervision over alternative test-time adaptation methods (e.g., optimization on multi-view consistency from random poses) because symmetry provides dense pixel-level correspondence without requiring multiple views or depth estimation, accepting the cost that symmetry assumption fails for asymmetric facial features (e.g., side-parted hair, asymmetric lighting) leading to potential artifacts. We chose to apply symmetry loss in the frontal view (canonical pose) rather than in arbitrary views because the frontal view minimizes perspective distortion and makes the left-right correspondence straightforward; the trade-off is that we must estimate a canonical pose robustly, which may be inaccurate for extreme input poses. We used only 100 refinement iterations (compared to 10k in SpatialAvatar) because symmetry provides strong gradients per iteration, but at the risk of not fully converging for very large expression differences. We opted for a combined L1 + perceptual loss rather than a GAN or adversarial loss to avoid destabilizing the Gaussian representation; the trade-off is slightly less sharp fine details.

(D) **Why it measures what we claim**: The symmetry loss L1(render_left, flip(render_right)) measures identity consistency between mirrored face halves under the assumption that the face is bilaterally symmetric; this assumption fails when the subject has natural asymmetries (e.g., scar, asymmetric expression), in which case the loss measures model-fit deviation from symmetry rather than reconstruction accuracy. We use perceptual loss on half-images to measure structural similarity beyond pixel intensity, assuming that face texture and geometry are symmetric in the same regions; this assumption fails for asymmetric lighting or shadows, causing the loss to penalize photorealism differences rather than symmetry errors. The refinement updates Gaussian parameters to minimize this loss, operationalizing the principle that minimizing symmetry violation forces the avatar to capture pose- and expression-invariant facial structure; when the input pose is extreme, the symmetry loss derived from the avatar's hypothetical frontal view provides a signal absent from the feed-forward model's training distribution, bridging the gap to out-of-distribution poses. The anti-spike regularization prevents Gaussians from collapsing to thin needles that could artificially satisfy symmetry via degenerate geometry; without it, the symmetry loss could be minimized by zero-opacity Gaussians on one side, which would not correspond to valid face geometry. Thus, the combination enforces that the avatar's 3D structure remains plausible while being symmetric.

## Contribution

(1) A self-supervised symmetry constraint that provides dense pixel-level supervision for test-time refinement, enabling feed-forward head avatar reconstruction to generalize to out-of-distribution poses and expressions without requiring paired multi-view data.
(2) A demonstration that applying this constraint in a canonical frontal view, combined with a lightweight refinement (100 steps), significantly improves reconstruction fidelity for extreme poses compared to training-only feed-forward methods.
(3) An analysis of the limitations and failure modes of symmetry-based supervision, including robustness to natural facial asymmetries and lighting variations.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | VFHQ head avatar | Diverse identities and poses; standard benchmark |
| Primary metric | PSNR | Measures pixel accuracy; sensitive to symmetry |
| Baseline1 | Feed-forward (no refinement) | Tests refinement contribution |
| Baseline2 | Test-time optimization (SpatialAvatar full) | Tests efficiency vs quality trade-off |
| Ablation-of-ours | Ours with random view consistency | Tests symmetry-specific benefit |

### Why this setup validates the claim

The VFHQ dataset contains single images with varied poses, enabling evaluation of generalization. Comparing to the feed-forward baseline isolates the effect of the symmetry-guided refinement step; any improvement must stem from the added constraint. Comparing to the full test-time optimization baseline (SpatialAvatar, 10k iterations) tests whether our 100-step refinement achieves similar quality with far fewer updates, validating the efficiency claim. The ablation replaces symmetry with random-view photometric consistency; if symmetry yields higher PSNR, it confirms that dense mirror correspondence (not just any self-supervision) drives the gain. PSNR is chosen because symmetry directly enforces pixel-level consistency between mirrored halves, which should translate into higher reconstruction accuracy for symmetric regions, a measurable signal.

### Expected outcome and causal chain

**vs. Feed-forward (no refinement)** — On an input with extreme yaw (e.g., 60° right profile), the feed-forward baseline produces a blurry left side because it never sees the left cheek and relies on weak training prior. Our method instead infers the left side by mirroring the visible right side via symmetry loss, so we expect a noticeable PSNR gap of ~2–3 dB on such extreme poses, but parity on near-frontal views.

**vs. Test-time optimization (SpatialAvatar full)** — On a case with natural facial asymmetry (e.g., one raised eyebrow), full optimization can overfit to the asymmetry, distorting the other side to match the input, while our symmetry constraint resists such overfitting, preserving a more natural symmetric baseline. Thus we expect comparable PSNR (~0.5 dB difference) but our method uses 100 iterations vs 10k, demonstrating a 100× speedup with negligible quality loss on symmetric metrics.

### What would falsify this idea

If our method shows no PSNR improvement over the feed-forward baseline on a held-out set of symmetric faces (e.g., near-frontal neutral expressions), it would falsify the claim that symmetry refinement adds value; conversely, if the improvement is uniform across all poses (including asymmetric ones) rather than concentrated on extreme-pose cases, the symmetry assumption may be causing a different beneficial regularization, not the hypothesized mechanism.

## References

1. SpatialAvatar-0: High-Quality 4D Head Avatar with Multi-Stage Reconstruction
2. GPAvatar: Generalizable and Precise Head Avatar from Image(s)
3. Generalizable and Animatable Gaussian Head Avatar
4. Portrait4D-v2: Pseudo Multi-View Data Creates Better 4D Head Synthesizer
5. RigNeRF: Fully Controllable Neural 3D Portraits
6. GaussianAvatar: Towards Realistic Human Avatar Modeling from a Single Video via Animatable 3D Gaussians
7. OTAvatar: One-Shot Talking Face Avatar with Controllable Tri-Plane Rendering
8. VOODOO 3D: Volumetric Portrait Disentanglement for One-Shot 3D Head Reenactment
