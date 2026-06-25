# PathStraightness: Automatic Guidance Scale Selection via Diffusion Path Geometry in Modular DiT Frameworks

## Motivation

Modular frameworks like NanoGen (DiffusionBench) enable flexible evaluation of DiT variants but still require manual tuning of the classifier-free guidance scale (gamma) per sample. This manual process is labor-intensive and suboptimal because the optimal gamma varies across samples due to differences in conditional information quality and distribution mismatch. Prior classifier-free guidance (Classifier-Free Diffusion Guidance) uses a fixed gamma, leading to artifacts or over-smoothing for samples where the unconditional and conditional scores diverge non-uniformly.

## Key Insight

The straightness of the diffusion path—quantified by the cosine similarity between consecutive denoising velocity vectors—is inversely correlated with the optimal guidance scale because straight paths indicate the model is already well-aligned with the conditional distribution, requiring less steering.

## Method

**(A) What it is:** PathStraightness is a sampling-time method that dynamically adjusts the classifier-free guidance scale gamma for each sample based on the straightness of its diffusion trajectory in latent space. The input is a sequence of latent states {z_t} during reverse diffusion, and the output is a per-sample gamma(t) (optionally time-varying).

**(B) How it works:**
```pseudocode
# Given: conditional score s_c(z_t, t), unconditional score s_u(z_t, t), current gamma(t)
# Initial gamma(0) = 1.5 (or other default)
# For each sampling step t from T-1 down to 0:
    # Compute straightness at step t:
    v_t = s_c(z_t, t) - s_u(z_t, t)  # guidance direction
    # If t < T-1, compute cosine similarity between consecutive guidance directions:
    if t < T-1:
        sim = cosine_similarity(v_t, v_{t+1})  # range [-1,1]
        straightness = (sim + 1) / 2  # normalize to [0,1]
    else:
        straightness = 1  # for first step
    # Map straightness to gamma adjustment:
    gamma(t) = gamma_init * (1 - alpha * (1 - straightness))
    # where alpha is a fixed hyperparameter (e.g., 0.5)
    # Clamp gamma(t) to [0.5, 3.0]
    gamma(t) = clamp(gamma(t), 0.5, 3.0)
    # Combined score:
    s = s_u(z_t, t) + gamma(t) * (s_c(z_t, t) - s_u(z_t, t))
```
Hyperparameters: gamma_init=1.5, alpha=0.5, clamp bounds [0.5,3.0].

**(C) Why this design:** We chose a geometric straightness metric derived from consecutive guidance directions over alternative approaches like a learned regressor because our metric is domain-agnostic and requires no additional training data or labels, accepting that it may not capture complex nonlinear dependencies between path curvature and optimal gamma (e.g., when the path is straight but the model is poorly conditioned). We selected cosine similarity over angular distance or norm-based metrics because it is scale-invariant and robust to varying gradient magnitudes across steps, at the cost of losing magnitude information that might indicate confidence. We opted for a linear mapping from straightness to gamma (with clamping) rather than a more complex function (e.g., exponential or neural network) because it is interpretable and guarantees monotonicity—lower straightness yields higher gamma—at the risk of being too simplistic for cases where the relationship is non-monotonic. We chose to clamp gamma to [0.5,3.0] based on typical ranges from prior work (Classifier-Free Diffusion Guidance) to avoid extreme values that could cause instability, but this may limit adaptation for samples that require very high or low guidance. We also decided not to vary gamma within a single sample as a function of time (though possible) because empirical observations in Scaling Rectified Flow show that early steps benefit more from guidance; our step-wise computation can naturally adapt if we use time-varying alpha, but we keep it constant for simplicity.

**(D) Why it measures what we claim:** The cosine similarity of consecutive guidance directions (v_t and v_{t+1}) measures the straightness of the diffusion path because, under the assumption that a well-conditioned sample follows a near-linear trajectory in score space (as in rectified flow), high similarity indicates the model is consistently steering toward the conditional mode. This assumption fails when the conditional distribution is multi-modal and the model switches between modes at different steps, in which case low straightness reflects mode ambiguity rather than poor conditioning, and the metric would overestimate the need for guidance (gamma too high). The computed straightness value is then used to modulate gamma via the formula gamma(t) = gamma_init * (1 - alpha*(1 - straightness)), which claims to reduce gamma for straight paths and increase for curved. The assumption here is that the optimal gamma is proportional to the inverse of straightness; this assumption fails when the path is straight but the unconditional score is far from the data manifold (e.g., due to truncation), in which case a higher gamma would still be beneficial but our method reduces it, potentially causing mode collapse. The clamping operation ensures gamma stays within a reasonable range, measuring a practical bound on guidance; this assumption fails if the optimal gamma lies outside [0.5,3.0] for some samples, in which case the method cannot fully adapt.

## Contribution

(1) A novel sampling-time algorithm, PathStraightness, that automatically selects the classifier-free guidance scale per sample by measuring the straightness of the diffusion path in latent space, eliminating manual tuning in modular DiT frameworks. (2) The insight that path straightness (cosine similarity of consecutive guidance directions) is inversely correlated with the optimal guidance scale, providing a domain-agnostic geometric signal for adaptive guidance. (3) A demonstration that this approach requires no additional training or external supervision, making it a lightweight add-on to existing DiT samplers.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|-----------------------|
| Dataset | ImageNet 256x256 | Standard class-conditional benchmark for DiT. |
| Primary metric | FID | Measures distribution fidelity, sensitive to guidance. |
| Baseline 1 | Fixed gamma=1.5 | Default guidance, baseline for comparison. |
| Baseline 2 | Fixed gamma=3.0 | High guidance, tests oversaturation. |
| Baseline 3 | Time-decay gamma | Linear decay 3→0.5, common heuristic. |
| Ablation | Ours w/ L2 straightness | Replaces cosine with L2 norm. |

### Why this setup validates the claim
ImageNet 256x256 is the de facto benchmark for class-conditional image generation with diffusion transformers, offering a wide range of classes and image complexities that lead to varied trajectory straightness. FID is the standard metric for measuring distributional fidelity and is known to be sensitive to guidance scale adjustments—it penalizes both blurriness (low gamma) and oversaturation (high gamma). The baselines isolate the effect of our dynamic adaptation: fixed gamma=1.5 is the typical starting point, fixed gamma=3.0 tests oversaturation, and time-decay gamma represents a simple heuristic alternative. The ablation (L2 norm instead of cosine similarity) directly tests the claim that scale-invariant straightness is key. Together, this setup creates a falsifiable test: if our method improves FID over all baselines, the central claim that adjusting gamma per-sample based on straightness is beneficial is supported; if not, the claim is invalid.

### Expected outcome and causal chain

**vs. Fixed gamma=1.5** — On a case where the diffusion path is highly curved (e.g., generating a rare bird species with intricate texture), the fixed gamma=1.5 baseline produces a blurry, less faithful image because it applies uniform guidance, insufficient for complex details. Our method detects low straightness and raises gamma (e.g., to 2.5), giving more conditional emphasis, thus the image is sharper and more accurate. We expect a noticeable FID improvement on the most semantically diverse classes (e.g., those with high intra-class variance) but parity on simple classes (e.g., plain backgrounds).

**vs. Fixed gamma=3.0** — On a case where the path is very straight (e.g., generating a simple geometric pattern), the fixed gamma=3.0 baseline produces oversaturated, unnatural colors because excessive guidance forces the sample too strongly toward the conditional mode. Our method detects high straightness and lowers gamma (e.g., to 0.8), preserving natural appearance. We expect our method to have significantly better FID on samples with low curvature (e.g., man-made objects with clean edges) while matching performance on high-curvature ones.

**vs. Time-decay gamma** — On a case where optimal guidance varies non-monotonically with time (e.g., a scene needing strong guidance early for layout but mild later for texture), the time-decay baseline applies a predetermined schedule that may over-guide at intermediate steps. Our method adapts step by step using straightness, which can increase gamma in later steps if necessary (e.g., when the trajectory curves again). We expect our method to achieve lower FID overall as it handles such non-monotonic cases better, but the gap may be small unless the dataset contains many such samples.

### What would falsify this idea
If our method’s FID improvement is uniformly distributed across all samples rather than concentrated in exactly those where fixed guidance fails (high curvature for low gamma, low curvature for high gamma), then the straightness metric is not capturing the correct adaptation signal and the central claim is false. Similarly, if the L2-based ablation performs equally well, the scale-invariance claim is unsupported.

## References

1. DiffusionBench: On Holistic Evaluation of Diffusion Transformers
2. Scaling Rectified Flow Transformers for High-Resolution Image Synthesis
3. Classifier-Free Diffusion Guidance
4. Mean Flows for One-step Generative Modeling
5. Emu Edit: Precise Image Editing via Recognition and Generation Tasks
6. CogVLM: Visual Expert for Pretrained Language Models
7. LAION-5B: An open large-scale dataset for training next generation image-text models
8. SmartBrush: Text and Shape Guided Object Inpainting with Diffusion Model
