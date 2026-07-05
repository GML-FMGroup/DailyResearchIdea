# Scale-Adaptive Implicit Perception for Fine-Grained Visual Reasoning

## Motivation

The P2R framework decouples perception and reasoning but relies on fixed-size cropping, which cannot capture evidence at varying scales required for fine-grained tasks like reading small text or recognizing distant objects. This structural mismatch forces a discrete bottleneck: reasoning queries needing fine detail are limited by the crop resolution, while wide-context queries waste computation on irrelevant high-resolution regions. The field must move beyond fixed-resolution sampling to a continuous, query-adaptive perception stage.

## Key Insight

Because implicit neural fields provide a differentiable continuous representation that can be sampled at any resolution, they enable the perception module to dynamically adapt its effective resolution to match the scale of evidence required by each reasoning query, eliminating the discrete cropping bottleneck.

## Method

(A) **What it is** — We propose SCAIN (Scale-Controllable Adaptive Implicit Network) which replaces the Perceiver's fixed-size crop with an implicit neural field representation. Input: a high-resolution image and a reasoning query (text). Output: a set of task-relevant visual features sampled at dynamically determined locations and scales.

(B) **How it works** — Pseudocode:
```python
# Hyperparameters: K = 64 sample points, scale range [0.1, 10.0], L_max = 10 frequency bands.
# Implicit field: 4-layer MLP, hidden=256, GeLU, output dimension d=128.
# Positional encoding: Gaussian Fourier features with frequencies [1, 2, ..., s*L_max] linearly spaced.
# Scale prediction MLP: 2-layer MLP, hidden=256, output (1 + 2K) = (1 + 128) = 129 dims, tanh on s, scaled to [0.1,10.0] via sigmoid then rescale.
# Training: Adam optimizer, lr=1e-4, batch size 16, 50 epochs, single GPU (24GB memory, ~100 hours).

# (1) Encode image into implicit field: coordinate MLP phi: (x,y) -> feature vector f.
#     Train with L2 reconstruction on RGB and a contrastive consistency loss (InfoNCE, temperature τ=0.07).
#     Verification: After training, evaluate reconstruction PSNR on a held-out set of 500 4K images; if PSNR < 30dB, increase hidden size to 512 or L_max to 15.
# (2) Encode query Q into query embedding q using a text encoder (e.g., CLIP text encoder, output 512-d vector).
# (3) Scale prediction: small MLP psi(q) outputs scale s and K point offsets (dx_i, dy_i) from image center.
#     Offsets are passed through tanh to lie in [-1,1] then scaled by image dimensions.
# (4) Sample features: For each point (x+dx_i, y+dy_i), query phi with positional encoding using frequencies up to s*L_max.
# (5) Aggregate sampled features (mean pooling), feed to Reasoner (e.g., 2-layer transformer with 4 heads, hidden 256) with query q for answer.
# (6) Training: Jointly optimize phi, psi, Reasoner via REINFORCE with self-critical baseline (baseline is running average of past rewards, decay 0.9) + contrastive loss (InfoNCE) forcing split ensembles to agree.
```

(C) **Why this design** — We chose an implicit neural field over a multi-scale feature pyramid because the continuous representation allows arbitrary scale queries without discretizing into levels; the trade-off is computational cost of MLP forward passes, which we mitigate by limiting K=64 and using a 4-layer MLP. We predict query points instead of full-grid sampling because only few locations are task-relevant; this avoids exorbitant computation but risks missing regions, handled by learning offsets end-to-end. We scale the frequency bandwidth of positional encoding by s instead of modifying network weights, enabling continuous scale adaptation at the cost of aliasing at extreme scales, mitigated by clamping s and using smooth frequency cutoff. Unlike HiDe's attention-based region selection that still produces fixed-size crops, our method yields continuous features and is differentiable end-to-end.

(D) **Why it measures what we claim** — The predicted scale s measures the required resolution for each query because we assume monotonicity between evidence granularity and optimal scale; this fails when a query is ambiguous (e.g., "count people" where some people are far, some near), causing s to reflect the dominant training scale or a compromise. Sampled points (x_i,y_i) measure spatial relevance because the offset MLP is trained to attend to query-tied regions; this assumes query embedding encodes spatial locality, failing for location-agnostic queries (e.g., "what color?") where points become random but still average correct color due to image-wide distribution. The contrastive consistency loss measures task-relevant feature invariance across scales because we assume agreement between differently-scaled views correlates with correctness; this fails if the model collapses to trivial constant outputs, in which case the loss signals invariance but not accuracy—we detect collapse by monitoring loss magnitude; if below 0.1, we early-stop and retrain with higher τ.

**Load-bearing assumption:** The implicit neural field phi trained with L2 reconstruction and contrastive loss can provide accurate high-frequency features when queried at high scales (s up to 10), enabling fine-grained visual reasoning from sparse point samples. This assumption is verified by the reconstruction PSNR check in step (1); if verification fails, model capacity is increased until PSNR meets threshold.

## Contribution

(1) A novel perceptual architecture that replaces fixed-size crops with a continuous implicit neural field for scale-adaptive visual reasoning. (2) A training framework combining reinforcement learning and contrastive consistency that jointly optimizes the implicit field, scale predictor, and reasoner without explicit perception supervision. (3) A new paradigm for high-resolution visual reasoning that removes the discrete cropping bottleneck.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|-----------------------|
| Dataset | HR-Bench (4K & 8K) | High-res images with fine-grained queries. |
| Primary metric | Answer accuracy | Directly measures reasoning performance. |
| Baseline 1 | Standard MLLM (e.g., LLaVA-NeXT) | No adaptive scale or point sampling. |
| Baseline 2 | HiDe (hierarchical zoom-in) | Fixed discrete cropping, no continuous scale. |
| Ablation-of-ours | SCAIN w/o scale prediction (fixed scale s=1.0) | Fixed scale to isolate adaptation benefit. |

### Why this setup validates the claim
Combining HR-Bench (which contains images with objects at varying scales and requires fine-grained localization) with a standard MLLM baseline tests the necessity of scale adaptation: if the model cannot adjust resolution per query, accuracy should drop on instances where the relevant evidence is small or large. HiDe, which uses hierarchical discrete crops, tests whether continuous scale adaptation yields further gains. Our ablation (fixed scale) isolates the contribution of the learned scale prediction mechanism. Accuracy as the metric directly reflects the ultimate reasoning quality, ensuring that any improvement is task-relevant. This design creates a falsifiable test: if SCAIN outperforms baselines specifically on queries where object scale varies significantly, the central claim (continuous scale adaptation helps) is supported; otherwise, the claim is weakened.

### Expected outcome and causal chain

**vs. Standard MLLM** — On a case where a query asks about a small object (e.g., "What color is the license plate?") in a 4K image, the standard MLLM downsamples the image uniformly, making the license plate too blurry to read. It thus guesses wrong or hallucinates. Our method predicts a large scale s (zooming in) and samples points around the plate, obtaining clear features from the implicit field. We expect a noticeable accuracy gap on HR-Bench subsets with small objects vs. near parity on queries about large, salient objects.

**vs. HiDe** — On a query that requires an intermediate zoom level not aligned with HiDe's discrete scales (e.g., "Count the birds in the tree" where the tree is medium-sized), HiDe picks the nearest crop level (either too zoomed or too wide), losing or aggregating irrelevant details. Our continuous scale prediction selects the exact s needed, and the sampled points cover the tree precisely. We expect SCAIN to outperform HiDe particularly on queries where the optimal scale falls between HiDe's predefined levels, observed as a consistent advantage on those subsets.

### What would falsify this idea
If SCAIN's accuracy improvement over baselines is uniform across all query types (e.g., both small and large objects) rather than concentrated on cases requiring non-default scales, then the predicted scale is not actually adapting meaningfully, and the claim fails.

## References

1. Perceive-to-Reason: Decoupling Perception and Reasoning for Fine-Grained Visual Reasoning
2. HiDe: Rethinking The Zoom-IN method in High Resolution MLLMs via Hierarchical Decoupling
3. Divide, Conquer and Combine: A Training-Free Framework for High-Resolution Image Perception in Multimodal Large Language Models
4. ZoomEye: Enhancing Multimodal LLMs with Human-Like Zooming Capabilities through Tree-Based Image Exploration
5. Alphazero-like Tree-Search can Guide Large Language Model Decoding and Training
