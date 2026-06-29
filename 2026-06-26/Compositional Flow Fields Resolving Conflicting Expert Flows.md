# Compositional Flow Fields: Resolving Conflicting Expert Flows via Mixed Velocity Learning

## Motivation

DanceOPD (DanceOPD: On-Policy Generative Field Distillation) treats each capability field independently, with no mechanism for mutual adjustment when fields conflict. This leads to compositional closure failure under conflicting expert flows. The root cause is the regression framework inherited from Stochastic Interpolants, which optimizes each field separately without modeling interaction terms between velocity fields.

## Key Insight

By modeling the mixed field as a learned combination that minimizes a joint objective enforcing simultaneous satisfaction of all constraints, the resulting flow can resolve conflicts because the mixture coefficients are themselves learned to balance competing gradients.

## Method

**(A) What it is:** Compositional Flow Fields (CFF) is a method that learns a mixed velocity field from multiple expert velocity fields by modeling pairwise interaction terms and optimizing a joint consistency objective. Inputs: a set of expert velocity fields {v_i}, a base density (e.g., Gaussian). Outputs: a single mixed velocity field v_mixed that satisfies all expert constraints simultaneously.

**(B) How it works — Pseudocode:**
```python
# CFF Training Procedure
# Hyperparameters: λ_pair=1.0, λ_cons=0.5, K=3 (number of interaction expansion terms)

for each training step:
    sample x from student rollout (on-policy)
    for each expert i:
        compute v_i(x)
    # Initialize mixed field as weighted average
    w = small_MLP(x)  # learned weights, output dim = number of experts
    v_mixed_init = sum_i w_i * v_i
    # Add interaction terms via pairwise products
    interactions = 0
    for k in range(K):
        for i,j pairwise:
            interactions += phi_ijk(x) * (v_i(x) ⊙ v_j(x))  # element-wise product
    v_mixed = v_mixed_init + interactions
    # Compute loss
    L = lambda_cons * MSE(v_mixed, target_from_random_expert) + \
        lambda_pair * sum_{i,j} ||v_i(x) - v_j(x)||^2  # conflict penalty
    update all parameters (MLP weights, phi_ijk parameters)
```

**(C) Why this design:** We chose a learned weighted average over a fixed hand-crafted combination because the optimal mixing depends on the state and the specific configuration of conflicts; the weights can adapt per-sample. Adding interaction terms via pairwise products (with learned per-pair projection phi_ijk) captures higher-order couplings that a linear combination misses, accepting increased parameter count. The conflict penalty term (MSE between fields) is included to explicitly discourage large disagreements, guiding the mixed field toward a consensus while the consistency loss ensures fidelity to at least one expert per sample. We chose on-policy sampling (student rollout) over off-policy because conflicts are most severe on states the student actually visits, following the rationale in DanceOPD; this incurs computational cost for generation but improves targeted resolution.

**(D) Why it measures what we claim:** The learned weights w_i measure per-state expert relevance because the MLP is trained to output values that minimize the overall objective; this equivalence assumes that the objective's minima correspond to a Pareto-optimal compromise among experts—an assumption that fails when the objective landscape has spurious minima (e.g., if all experts are equally wrong near a rare state), in which case weights reflect gradient magnitude rather than true relevance. The pairwise interaction terms phi_ijk(x) measure cross-expert coupling because they are structured as bilinear forms over velocity fields; this assumes that conflicts manifest as multiplicative interactions (e.g., one expert pulling in a direction that another cancels)—a failure mode arises when conflicts are additive (non-multiplicative), in which case the interaction terms may not capture them and instead introduce noise. The conflict penalty term sum_{i,j} ||v_i - v_j||^2 measures overall disagreement because it directly quantifies differences; this assumes that the Euclidean metric is appropriate for comparing velocity fields, which fails when fields are in different coordinate systems (e.g., one field uses a different noise schedule), in which case the penalty may incorrectly signal conflict.

## Contribution

(1) A compositional flow formulation that explicitly models interaction terms between velocity fields via a learned mixing mechanism with pairwise corrections. (2) A joint training objective that combines a consistency loss with a conflict-penalty term, enabling on-policy distillation to resolve conflicting expert flows. (3) Introduction of a small MLP-based mixing network and per-pair interaction functions that can be trained alongside the student flow.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | MS-COCO 30K validation | Diverse prompts with conflicting expert demands |
| Primary metric | FID | Measures global distribution fidelity |
| Baseline 1 | Naive averaging (fixed weights) | Tests need for state-dependent weighting |
| Baseline 2 | MultiDiffusion (pixel-space fusion) | Tests superiority of velocity-space fusion |
| Ablation of ours | CFF without interaction terms | Isolates contribution of pairwise coupling |

### Why this setup validates the claim
This combination forms a falsifiable test because MS-COCO includes prompts where expert velocity fields naturally conflict (e.g., realism vs. style). FID captures both perceptual quality and mode coverage, directly reflecting the coherence of generated images. The naive averaging baseline isolates the effect of adaptive weighting; if CFF only benefits from higher capacity, it would not outperform on conflict-heavy subsets. MultiDiffusion represents a strong alternative fusion strategy operating in pixel space; if CFF improves similarly on all tasks, the benefit is not due to its velocity-space mechanism. The ablation directly tests whether pairwise interaction terms are necessary for handling multiplicative conflicts. Thus, observing that gains concentrate on conflict-heavy prompts (as measured by FID) would confirm that CFF resolves conflicts via its designed components, while uniform gains would falsify the core claim.

### Expected outcome and causal chain

**vs. Naive Averaging** — On prompts with conflicting expert requirements (e.g., "photorealistic cat in Van Gogh style"), naive averaging produces a blurred velocity field, leading to artifacts like inconsistent textures. Our method uses learned per-state weights to prioritize the realism expert in detailed regions and the style expert elsewhere, generating coherent images. We expect a noticeable FID improvement (e.g., >2 points) on conflict-heavy subsets and parity on non-conflicting prompts.

**vs. MultiDiffusion** — On global consistency tasks (e.g., long panorama), MultiDiffusion denoises overlapping windows independently and averages in pixel space, causing visible seams. Our method fuses velocities before integration, preserving global structure. We expect significantly lower FID on multi-region tasks (e.g., >3 points) but comparable FID on single-region tasks.

### What would falsify this idea
If CFF's FID gains over naive averaging are uniform across all prompt types rather than concentrated on conflict-heavy prompts, it would indicate that the method's success is not due to conflict resolution but to other factors (e.g., increased capacity), undermining the central claim.

## References

1. DanceOPD: On-Policy Generative Field Distillation
2. On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes
3. MultiDiffusion: Fusing Diffusion Paths for Controlled Image Generation
4. Stochastic Interpolants: A Unifying Framework for Flows and Diffusions
5. Large Language Models Are Reasoning Teachers
6. SpaText: Spatio-Textual Representation for Controllable Image Generation
7. Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow
8. Building Normalizing Flows with Stochastic Interpolants
