# ImplicitGaussianHead: Topology-Free 3D Head Avatar Reconstruction via Joint Optimization of Gaussians and Implicit Surface

## Motivation

Existing FLAME-mesh-bound Gaussian representations, such as SpatialAvatar, are constrained by the fixed topology of the parametric head model, preventing adaptation to non-standard head shapes or hairstyles outside FLAME's morphable space. This structural limitation arises because the Gaussian positions are bound to mesh vertices, which cannot represent arbitrary topologies like long hair or unconventional head shapes. We address this by introducing a topology-free geometric prior that jointly optimizes Gaussians with an implicit surface field.

## Key Insight

Constraining Gaussians to lie on the zero-level set of a jointly learned implicit surface field decouples geometry from appearance, enabling the implicit field to represent arbitrary topologies while the Gaussians handle high-frequency details.

## Method

### (A) What it is
We propose ImplicitGaussianHead, a method that jointly optimizes a set of 3D Gaussians and a learnable implicit surface field (SDF) for head avatar reconstruction. Input: multi-view images (or monocular video). Output: a 3D representation enabling novel view synthesis and animation.

**Load-bearing assumption:** This method relies on the assumption that joint optimization of photometric loss with surface regularizations (L_surface, L_normal) can recover accurate head geometry and appearance without any shape prior or explicit geometry supervision. We verify this with diagnostic metrics during training (see Section D).

### (B) How it works
```python
# Pseudocode for ImplicitGaussianHead training loop
# Initialize: N Gaussians (position p_i, scale s_i, rotation r_i, opacity alpha_i, color c_i)
# Initialize MLP for implicit surface: f(x; theta) -> signed distance
# MLP: 2-layer MLP with hidden=256, GeLU activation, positional encoding (6 frequencies)
# Hyperparameters: lambda_surface=0.1, lambda_splitting=0.01, lambda_perceptual=0.1, max_Gaussians=100000
# Optimizer: Adam for Gaussians (lr=0.001) and MLP (lr=0.0001)

for iteration in range(max_iterations):
    # Render from Gaussians using differentiable rasterizer: I_rendered = render(Gaussians, camera)
    I_rendered = render_gaussians(Gaussians, camera)
    
    # Compute photometric loss
    L_photo = L1(I_rendered, I_target) + lambda_perceptual * LPIPS(I_rendered, I_target)
    
    # Compute surface constraint: Gaussians should be on zero-level set
    sdf_values = f(p_i, theta)
    L_surface = mean(sdf_values**2)
    
    # Regularization: normal consistency
    normals = gradient(f, p_i)  # using finite differences or autograd
    L_normal = mean(1 - abs(dot(rotation_z_of_Gaussian, normals)))
    
    # Adaptive control: split/merge every K=2000 steps
    if iteration % K == 0 and num_Gaussians < max_Gaussians:
        grad_mag = norm(gradient(f, p_i))
        split_mask = grad_mag > threshold_split  # threshold_split=0.1
        new_Gaussians = split(Gaussians[split_mask])  # duplicate and perturb along normal by 0.01
        Gaussians = merge(Gaussians, new_Gaussians)
        merge_mask = find_close_pairs(Gaussians, distance_threshold=0.01)
        Gaussians = merge_pairs(Gaussians, merge_mask)
    
    # Total loss
    L_total = L_photo + lambda_surface * (L_surface + L_normal) + lambda_splitting * split_penalty
    
    # Update Gaussians and MLP parameters via gradient descent
    update(Gaussians, theta)
```

### (C) Why this design
We chose a joint optimization of Gaussians and an implicit field rather than a two-stage approach (e.g., first reconstruct surface, then attach Gaussians) because it allows the Gaussians to guide the implicit field refinement via photometric gradients, avoiding the need for a separate surface reconstruction step that may miss high-frequency details. We enforce Gaussians to lie on the zero-level set via L_surface, accepting the risk that Gaussians may drift slightly off-surface if the implicit field is inaccurate early in training; the normal alignment loss L_normal mitigates this by penalizing orientation mismatch, ensuring the Gaussians remain tangent to the surface. We use adaptive splitting/merging based on SDF gradient magnitude rather than a uniform grid or fixed count, trading off a slight increase in computational cost for the ability to automatically allocate more Gaussians to geometrically complex regions like hair strands. Compared to SpatialAvatar, which relies on a fixed FLAME mesh topology, our implicit field can represent any shape, at the cost of requiring an additional MLP forward pass and more careful initialization to avoid local minima.

### (D) Why it measures what we claim
The surface constraint loss L_surface = mean(sdf_values^2) measures **topology-free geometric fitting** because it assumes that the true head surface is exactly the zero-level set of the implicit field; this assumption fails when the SDF is inaccurate (e.g., early in training), in which case L_surface instead measures deviation from the current (potentially wrong) surface estimate, but the joint optimization with photometric loss gradually corrects this. **Linking assumption:** Photometric gradients correct SDF errors over time; failure mode occurs when photometric loss dominates and SDF diverges (e.g., textureless regions). **Diagnostic:** We monitor the correlation between SDF gradient magnitude and photometric loss during training. A high positive correlation (>0.5) indicates that photometric updates are effectively driving SDF refinement, validating the assumption. Low or negative correlation suggests the SDF is uncoupled, potentially leading to degenerate solutions. The normal alignment loss L_normal measures **local geometric consistency** because it assumes that Gaussian orientations should align with the surface normal for plausible rendering; this assumption fails in concave regions where the normal direction is ambiguous, in which case L_normal may encourage suboptimal orientations but still retains a preference for smoothness. The adaptive splitting based on SDF gradient magnitude measures **geometric complexity** of the surface, as higher gradients indicate faster change in signed distance, corresponding to finer details; this assumption fails in regions with near-zero gradients (flat areas) where splitting is unnecessary, but the threshold prevents oversplitting. Together, these components operationalize the concept of topology-free adaptation by replacing the fixed-mesh prior with a learnable implicit field that adapts to arbitrary shapes, as evidenced by the method's ability to represent non-standard heads and hairstyles.

## Contribution

(1) A joint optimization framework that learns 3D Gaussians and an implicit surface field with a zero-level set constraint, enabling topology-free head avatar reconstruction without relying on a parametric head model. (2) An adaptive Gaussian splitting/merging strategy guided by implicit surface gradients, which automatically allocates representation capacity to geometrically complex regions. (3) Empirical demonstration that the method reconstructs plausible avatars for heads with non-standard shapes, long hair, and accessories beyond the FLAME morphable space, outperforming mesh-bound approaches in geometric flexibility.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | VFHQ | Standard benchmark for head avatars |
| Primary metric | PSNR | Measures pixel-level reconstruction fidelity |
| Baseline 1 | SpatialAvatar | Tests fixed-topology vs. ours |
| Baseline 2 | PointAvatar | Tests point-based vs. ours |
| Baseline 3 | 3DGS (plain) | Tests pure Gaussian vs. ours |
| Ablation | Ours w/o surface constraints | Isolates effect of surface regularization |

### Why this setup validates the claim
This combination creates a falsifiable test of our central claim: joint optimization of Gaussians and an implicit field enables topology-free adaptation. VFHQ provides multi-view data with diverse head shapes and hairstyles, challenging fixed-topology methods. SpatialAvatar relies on FLAME mesh topology, so it will fail on shapes that deviate from the parametric model; comparing to it tests whether our method truly escapes topological constraints. PointAvatar uses explicit points but no adaptive densification guided by surface complexity; contrasting with it isolates the benefit of our SDF-guided splitting. Plain 3DGS matches our Gaussian backbone but lacks the implicit surface guidance; its performance gap quantifies the value of the surface regularizations. Our ablation removes both surface constraints, revealing whether they are crucial for maintaining Gaussians on the surface and enabling adaptive control. PSNR is chosen because it directly reflects rendering accuracy, which is the ultimate goal; if our method achieves higher PSNR on non-standard shapes, it confirms the claim.

### Expected outcome and causal chain

**vs. SpatialAvatar** — On a case with extreme hairstyle (e.g., spiky hair), SpatialAvatar produces blurry renderings because its fixed FLAME mesh cannot represent sharp geometry changes, forcing the texture to compensate. Our method instead allocates more Gaussians to hair tips via SDF-gradient-based splitting, capturing fine details, so we expect a noticeable gap (e.g., 1-2 dB PSNR) on challenging subjects but parity on near-neutrical faces.

**vs. PointAvatar** — On a case with complex concavities (e.g., deep eye sockets), PointAvatar fails because its points lack orientation regularity, causing view-dependent artifacts from misaligned normals. Our method enforces normal alignment loss, keeping Gaussian rotations tangent to the surface, so we expect fewer rendering artifacts (higher LPIPS) in such regions, though overall PSNR may be similar for smooth areas.

**vs. 3DGS (plain)** — On a case with large flat regions (e.g., bald head), plain 3DGS wastes Gaussians on low-detail areas due to uniform densification, causing overfitting noise. Our method uses SDF gradient to guide splitting only in complex areas, leading to cleaner background and better generalization across views; we expect higher PSNR on hold-out views (e.g., +0.5 dB) but similar training-view performance.

### What would falsify this idea
If our method fails to outperform SpatialAvatar specifically on shapes that deviate significantly from the FLAME template (e.g., non-standard facial features), then the claim of topology-free adaptation is false. Alternatively, if the ablation (without surface constraints) matches or outscores our full method, it indicates the implicit field is not beneficial and the complexity adds no value.

## References

1. SpatialAvatar-0: High-Quality 4D Head Avatar with Multi-Stage Reconstruction
2. GPAvatar: Generalizable and Precise Head Avatar from Image(s)
3. Generalizable and Animatable Gaussian Head Avatar
4. Portrait4D-v2: Pseudo Multi-View Data Creates Better 4D Head Synthesizer
5. RigNeRF: Fully Controllable Neural 3D Portraits
6. GaussianAvatar: Towards Realistic Human Avatar Modeling from a Single Video via Animatable 3D Gaussians
7. OTAvatar: One-Shot Talking Face Avatar with Controllable Tri-Plane Rendering
8. VOODOO 3D: Volumetric Portrait Disentanglement for One-Shot 3D Head Reenactment
