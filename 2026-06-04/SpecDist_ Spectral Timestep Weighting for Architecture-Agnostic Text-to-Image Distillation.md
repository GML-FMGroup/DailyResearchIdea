# SpecDist: Spectral Timestep Weighting for Architecture-Agnostic Text-to-Image Distillation

## Motivation

Current distillation methods like SenseFlow rely on manually tuned hyperparameters such as timestep schedules and loss weights that vary with backbone architecture (e.g., UNet, DiT, flow-matching). For instance, SenseFlow's Intra-Segment Guidance (ISG) requires adjusting denoising importance weights per architecture, yet its heuristic design lacks a principled connection to the teacher model's information content. This architectural sensitivity stems from ignoring the spectral structure of the teacher's denoising operator, which inherently captures the timestep-varying informativeness for student learning.

## Key Insight

The singular value spectrum of the teacher's denoising operator at each timestep provides a natural, architecture-independent measure of the information content, enabling automatic determination of optimal timestep weighting without manual tuning.

## Method

### (A) What it is
SpecDist automatically computes timestep weights for distillation loss via spectral analysis of the teacher model's denoising operator, using a reference prompt set. Input: teacher model, reference prompts; Output: per-timestep weights for distillation loss.

### (B) How it works
```python
# Phase 1: Compute spectral weights offline
reference_prompts = load_reference_prompts(num=1000)  # 1000 diverse prompts
timesteps = [1,2,...,T]  # discrete timesteps (e.g., 1000 for DDPM)
weights = []
for t in timesteps:
    # Collect teacher denoising outputs
    outputs = []
    for prompt in reference_prompts:
        z_T = sample_noise(prompt)  # same latent size
        z_t = add_noise(z_T, t)
        # Teacher's denoising operator: 
        # from noisy sample to predicted clean or noise
        pred = teacher.denoise(z_t, t, prompt_embedding)
        outputs.append(pred.flatten())
    outputs = np.array(outputs)  # shape: (num_samples, dim=latent_dim)
    # Compute SVD on outputs (approximate via randomized SVD)
    # Use randomized_svd with n_components=50, n_iter=2, random_state=42
    U, S, Vt = randomized_svd(outputs, n_components=50, n_iter=2, random_state=42)
    # Spectral importance: sum of all singular values (trace norm)
    weight = np.sum(S)
    weights.append(weight)
weights = np.array(weights)
weights = weights / np.sum(weights)  # normalize to sum=1

# Phase 2: Distillation training with weighted loss
student = StudentModel()  # architecture matches teacher but smaller capacity
for each step:
    prompt, x_0 = sample_batch()
    t = sample_timestep()  # uniformly from [1,T]
    z_t = add_noise(x_0, t)
    student_pred = student(z_t, t, prompt)
    teacher_pred = teacher(z_t, t, prompt) # frozen
    loss = weights[t] * MSELoss(student_pred, teacher_pred.detach())
    update(student, loss)
```

### (C) Why this design
We chose to compute spectral weights offline from a fixed reference set rather than online during training to avoid recomputation overhead (SVD every iteration would be prohibitive, see compute estimate below), accepting that the weights may be suboptimal for out-of-distribution prompts. We chose to use the sum of all singular values (trace norm) instead of the largest singular value because it captures total variance across all directions and is more stable across timesteps, but this can dilute the effect of a few dominant directions if many low-variance components exist. We chose to normalize weights to sum to 1 to keep the overall loss magnitude consistent, which prevents training instability but may underweight a timestep if others have high variance, potentially reducing learning signal. We use a fixed set of 1000 prompts as a practical compromise between accuracy and compute; too few prompts lead to noisy weight estimates, while too many increase computation. Compute estimate: For 1000 prompts, 1000 timesteps, latent dimension ~4×64×64≈16384, randomized SVD (50 comps) takes ~0.1s per timestep on one GPU, total ~100 GPU-seconds for weight computation. Baseline distillation training typically requires 1000+ GPU-hours, so the overhead is negligible (<0.1%).

### (D) Why it measures what we claim
**Assumption (explicit):** The sum of singular values of the teacher's denoising outputs at timestep t measures informativenss because high-variance timesteps indicate where the teacher makes diverse corrections that the student must learn. Specifically, we assume teacher output variance at timestep t correlates with learning significance. This assumption fails when the variance is dominated by noise directions (e.g., high-frequency detail that is not perceptually important), in which case the weight may overemphasize timesteps that are noisy but not semantically important. The spectral decomposition captures all directions equally, but our method assumes that variance correlates with learning significance; if the teacher's denoising operator has many singular values driven by nuisance factors, the weight may misallocate emphasis. To mitigate this, we also consider using only the top-k singular values (e.g., explaining 90% variance) as a calibration step (explored in ablation).

## Contribution

(1) A novel distillation framework (SpecDist) that automatically determines timestep weights via spectral analysis of the teacher's denoising operator, eliminating per-architecture hyperparameter tuning. (2) Empirical demonstration that spectral weights derived from the teacher's denoising outputs generalize across UNet, DiT, and flow-matching backbones without modification. (3) Open-source implementation of efficient randomized SVD-based weight computation for distilling large text-to-image models.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | COCO2017 | Diverse prompts, standard benchmark for text-to-image generation.
| Primary metric | FID-30k | Measures distribution fidelity to real images.
| Baseline 1 | Vanilla DMD | No spectral weighting; uniform timestep loss.
| Baseline 2 | Progressive Distillation | Step-by-step distillation with fixed schedule (e.g., halving steps every 100K iterations).
| Baseline 3 | Adversarial Distillation | Uses discriminator loss alongside MSE distillation.
| Baseline 4 (new) | Variance-based weighting | Uses per-timestep variance of teacher outputs (sum of variances across dimensions) instead of SVD, normalized to sum=1. Isolates benefit of spectral decomposition.
| Ablation-of-ours | SpecDist (uniform weights) | Isolates spectral weighting effect by using uniform weights across timesteps.
| Ablation-of-ours (new) | Online weight recomputation | Recomputes spectral weights every 5000 training steps on a subset of 100 reference prompts to assess robustness to distribution shift.

### Why this setup validates the claim

The combination of COCO2017, FID, and selected baselines forms a falsifiable test of the central claim that SpecDist's spectral timestep weighting improves distillation quality. COCO2017 offers diverse prompts covering various visual concepts, ensuring the evaluation generalizes. FID is the standard metric for measuring the fidelity of generated images to real data, directly capturing the effect of weighting on output distribution. Vanilla DMD and Progressive Distillation represent standard distillation methods without our weighting, so a comparison reveals the improvement due to spectral analysis. Adversarial distillation tests whether our method complements or competes with adversarial objectives. The new variance-based baseline isolates the benefit of spectral decomposition over simple variance weighting. The online recomputation ablation tests whether offline weights are sufficient or if adaptation is needed. If SpecDist outperforms all baselines and the ablation, it confirms the importance of spectral weight assignment. Conversely, if gains are absent or uniform, the claim is falsified.

### Expected outcome and causal chain

**vs. Vanilla DMD** — On a case where the teacher produces high-variance denoising at early timesteps (e.g., large structural changes), vanilla DMD applies equal weight across all timesteps, causing the student to underlearn critical steps because the loss is diluted. Our method assigns higher weight to those high-variance timesteps via spectral analysis, amplifying the learning signal for structural components. As a result, we expect a noticeable FID improvement (e.g., 5-10% relative reduction) while maintaining similar performance on low-variance steps.

**vs. Progressive Distillation** — On a case where the teacher's denoising operator has a sharp spectral drop-off at a specific timestep (e.g., fine details emerge), progressive distillation's fixed schedule (e.g., halving steps) may skip or overemphasize that timestep, leading to blurry outputs. Our method dynamically weights that timestep using its singular value sum, ensuring the student captures the critical details. We expect SpecDist to produce sharper images on fine-grained subsets (e.g., animal fur, text) with a FID gap of at least 3-5% over progressive distillation.

**vs. Adversarial Distillation** — On a case where the teacher's predictions contain high-frequency noise that dominates variance (e.g., textured background), adversarial distillation's discriminator may penalize the student for mimicking such noise, causing mode dropping. Our method's spectral weighting sums all singular values, which can overemphasize noisy timesteps if variance is high but uninformative. However, we expect that on most natural images, the dominant singular directions correspond to semantic content, so SpecDist will not amplify noise. The expected outcome is competitive FID with adversarial distillation, but with better recall (fewer missing modes) because we avoid adversarial collapse.

**vs. Variance-based weighting** — On a case where the teacher's output has correlated structure across dimensions (e.g., coherent object shapes), variance-based weighting (sum of per-dimension variances) ignores correlations and may overcount trivial directions. Spectral decomposition captures the covariance structure, so SpecDist should yield more informative weights. We expect SpecDist to achieve 2-3% lower FID compared to variance-based weighting.

### What would falsify this idea

If SpecDist performs uniformly better across all timesteps and prompt subsets, or if its improvement over uniform weighting is negligible (e.g., <1% FID), then the central claim that spectral weighting identifies important timesteps is false. Specifically, observing that the gain is concentrated on high-variance timesteps but FID does not improve would also falsify the variance-informativeness assumption. Additionally, if variance-based weighting matches SpecDist's performance, then the spectral decomposition is unnecessary.

## References

1. SenseFlow: Scaling Distribution Matching for Flow-based Text-to-Image Distillation
2. Flash Diffusion: Accelerating Any Conditional Diffusion Model for Few Steps Image Generation
3. Improved Distribution Matching Distillation for Fast Image Synthesis
4. Score Jacobian Chaining: Lifting Pretrained 2D Diffusion Models for 3D Generation
5. On Distillation of Guided Diffusion Models
6. DreamFusion: Text-to-3D using 2D Diffusion
7. Few-Step Distillation for Text-to-Image Generation: A Practical Guide
8. Latent Consistency Models: Synthesizing High-Resolution Images with Few-Step Inference
