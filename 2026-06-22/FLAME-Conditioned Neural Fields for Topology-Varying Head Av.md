# FLAME-Conditioned Neural Fields for Topology-Varying Head Avatars

## Motivation

Existing feed-forward+refinement head avatars such as SpatialAvatar-0 assume a fixed-topology Gaussian representation bound to the FLAME mesh, preventing adaptation to non-standard head shapes and fine details during refinement. This structural constraint arises because the refinement stage freezes Gaussian count and layout, relying on the FLAME mesh as a rigid backbone. Consequently, avatars of subjects with diverse hairstyles, accessories, or extreme expressions cannot be accurately captured without manual per-subject tuning.

## Key Insight

By making the existence and parameters of each Gaussian a direct function of the FLAME expression code through a shared neural field, any Gaussian added or removed remains controllable by the same global code, preserving animatability despite topology changes.

## Method

(A) **What it is:** FANA (FLAME-Adaptive Neural Avatars) is a refinement framework that converts a fixed-topology set of FLAME-bound Gaussians into a variable-topology set by optimizing a per-subject neural field. The neural field takes a canonical position and the FLAME code (shape, expression, pose) as input and outputs deformation offset, scale, opacity, color, and an existence probability. The output is a set of 3D Gaussians whose count adapts per subject while remaining animatable via the FLAME code.

(B) **How it works:**
```python
# Input: initial Gaussians from feed-forward (e.g., SpatialAvatar) anchored at FLAME vertices
# Output: refined variable-topology Gaussians

# Step 1: Initialize neural field parameters (MLP with 4 hidden layers, 256 units)
# Step 2: For each subject, sample N=10,000 canonical positions (from a 3D grid within the head bounding box)
# Step 3: For each training frame with FLAME code c_t, compute:
#   For each canonical position phi_i:
#       (mu_i_t, s_i_t, o_i_t, col_i_t, e_i_t) = MLP([phi_i; c_t])
#       # e_i_t is logit, convert to probability via sigmoid
#   # Gaussian parameters: position = phi_i + mu_i_t, scale = exp(s_i_t), opacity = sigmoid(o_i_t), color = col_i_t
#   # Render these Gaussians (with differentiable rasterization) to get image
# Step 4: Loss = L_photometric (L1 + LPIPS) + lambda_spike * (|s_i|_2 + |o_i|_2) + lambda_exist * mean(|e_i_t - 0.5|) 
#   # Hyperparameters: lambda_spike=0.01, lambda_exist=0.1, learning_rate=0.001, optimizer=Adam
# Step 5: Backpropagate through all steps, including thresholding: during forward use Gumbel-softmax for existence to keep gradient
# Step 6: After training, prune Gaussians with sigmoid(e_i) < 0.5 at neutral expression, resulting in k (typically 3000-8000) Gaussians.
# Step 7: Animation: for a new expression code c_new, compute Gaussians from the same neural field (pruning mask fixed after training).
```
(C) **Why this design:** We chose a shared MLP conditioned on canonical position and FLAME code (over a separate per-Gaussian latent) because it guarantees that the same expression code drives all Gaussians consistently, avoiding the need to learn per-Gaussian animation weights that could break invariance under expression changes. We opted for a 3D grid of canonical positions (instead of starting from FLAME vertices) to allow the neural field to spawn Gaussians in regions not covered by the mesh, such as hair or loose clothing, at the cost of increased memory during training. The use of Gumbel-softmax for existence enables differentiable pruning, which is crucial for gradient-based optimization of topology; the trade-off is that the temperature parameter must be carefully tuned (we used 0.5) to balance exploration and deterministic pruning. We added an L1 regularization on existence logits to encourage sparsity, preventing the model from keeping unnecessary Gaussians that would increase rendering cost without quality gain. Compared to SpatialAvatar's anti-spike regularization (which only penalizes extreme scales/opacities), our existence regularization explicitly controls count, a necessary addition for adaptive topology.

(D) **Why it measures what we claim:** The existence probability output e_i, after sigmoid, measures the **necessity** of a Gaussian at position phi_i for the subject's appearance under any expression, because it is trained to be high only if the Gaussian contributes to photometric accuracy across all training frames; this assumes that the training set covers all relevant expressions and poses, which fails if the subject's appearance changes drastically under unseen expressions (e.g., a wide-open mouth revealing teeth not present in training), in which case e_i may underestimate the need for additional Gaussians. The deformation offset mu_i_t measures **expression-driven motion** relative to the canonical position, grounded on the assumption that the canonical space is a mean shape; this fails if the subject has non-rigid deformations that are not smooth functions of the FLAME code (e.g., dynamic hair physics), in which case mu_i_t captures only the FLAME-decomposable part. The opacity o_i measures **visibility** under the assumption that the neural field can model view-dependent opacity consistently; this fails if the training views are sparse, leading to opacity artifacts. The combined set of Gaussians after pruning measures **topological adaptability** because the count and layout emerge from the neural field's learning of where details are needed; this assumes that the canonical positions are dense enough to cover all detail regions, which fails for extremely fine structures (e.g., individual hair strands) that require smaller scales than the grid resolution, in which case the model would still need denser sampling or hierarchical refinement.

## Contribution

(1) A neural field architecture that maps FLAME codes to variable-topology 3D Gaussians via differentiable existence probabilities, enabling adaptive topology while preserving animatability. (2) The insight that coupling Gaussian existence and parameters to the same FLAME conditioning code ensures expression control is invariant to topology changes, a design principle validated by the monotonic relationship between FLAME code and Gaussian outputs. (3) A training recipe combining Gumbel-softmax pruning with topology-aware regularization (sparsity on existence logits) that allows end-to-end optimization of Gaussian count during refinement.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|-----------------------|
| Dataset | VFHQ | Wide head poses and expressions |
| Primary metric | PSNR | Standard for image quality |
| Baseline 1 | SpatialAvatar | Fixed-topology 3DGS baseline |
| Baseline 2 | GPAvatar | State-of-the-art animatable avatar |
| Baseline 3 | Per-subject 3DGS | Dense optimization baseline |
| Ablation-of-ours | No existence loss | Fixed number of Gaussians |

### Why this setup validates the claim
This combination of dataset, baselines, and metric forms a falsifiable test of the central claim that adaptive topology improves avatar quality while preserving animatability. VFHQ challenges the model with varied expressions and poses, testing generalization. SpatialAvatar and GPAvatar represent fixed-topology and non-Gaussian approaches respectively, isolating the benefit of variable topology. The per-subject 3DGS baseline shows whether adaptive topology adds value over dense but fixed-count optimization. PSNR directly measures image fidelity, the ultimate goal. The ablation (no existence loss) tests whether the adaptive topology mechanism itself, not just the neural field, drives improvement. If our method outperforms all baselines on PSNR, especially on novel expressions, the claim is supported.

### Expected outcome and causal chain

**vs. SpatialAvatar** — On a case where the subject has detailed hair not covered by the FLAME mesh, SpatialAvatar's fixed Gaussians anchored at vertices miss those regions, producing blurry hair because no mechanism exists to add Gaussians. Our method instead spawns new Gaussians from the neural field where needed, because the neural field can assign high existence probability to canonical positions in hair regions, so we expect a noticeable PSNR gap on hairy subjects (e.g., +2 dB) but parity on tight-scalp subjects.

**vs. GPAvatar** — On a case with extreme expression (e.g., wide-open mouth), GPAvatar's NeRF-based rendering may struggle with high-frequency details due to MLP capacity, producing oversmoothed teeth and inner mouth. Our method uses variable-topology Gaussians that can cluster near the mouth interior, because the per-subject neural field can allocate more Gaussians to high-detail regions by adjusting existence probabilities, so we expect sharper rendering (LPIPS improvement) on frames with open mouth while matching on neutral expressions.

**vs. Per-subject 3DGS** — On a case where the subject has both high-detail and low-detail regions (e.g., eyes vs. cheek), the fixed-count 3DGS must allocate Gaussians uniformly, wasting capacity on flat areas and starving detail regions, causing oversmoothing. Our method adapts count distribution by pruning unnecessary Gaussians via existence loss, because the regularization penalizes Gaussians that do not contribute consistently, so we expect higher PSNR on high-detail subjects (e.g., +1.5 dB) while maintaining similar performance on low-detail subjects.

### What would falsify this idea
If our method's improvement over baselines is uniform across all subjects and expressions rather than concentrated on cases with missing topology (e.g., hair, teeth, or extreme expressions), then the adaptive topology is not the cause—suggesting the neural field itself or other design choices drive the gain.

## References

1. SpatialAvatar-0: High-Quality 4D Head Avatar with Multi-Stage Reconstruction
2. GPAvatar: Generalizable and Precise Head Avatar from Image(s)
3. Generalizable and Animatable Gaussian Head Avatar
4. Portrait4D-v2: Pseudo Multi-View Data Creates Better 4D Head Synthesizer
5. RigNeRF: Fully Controllable Neural 3D Portraits
6. GaussianAvatar: Towards Realistic Human Avatar Modeling from a Single Video via Animatable 3D Gaussians
7. OTAvatar: One-Shot Talking Face Avatar with Controllable Tri-Plane Rendering
8. VOODOO 3D: Volumetric Portrait Disentanglement for One-Shot 3D Head Reenactment
