# FedStateMark: Model-State-Dependent Watermarking for Federated Generative Models

## Motivation

Existing watermarking methods for federated learning, such as FedOT, embed static watermark patterns into the global model that remain constant across communication rounds. This static nature makes them vulnerable to a simple collusion attack: an adversary who collects the global model from multiple rounds can average or interpolate those models to erase the fixed watermark signal. The root cause is that the watermark is independent of the model's current state, so averaging preserves the common pattern. Our method breaks this by making the watermark a deterministic function of the global model weights, which vary per round, ensuring that cross-round averaging destroys the pattern.

## Key Insight

A watermark derived via a keyed hash of the global model weights is uniquely tied to that specific weight instance, so any averaging across rounds mixes uncorrelated patterns and collapses the watermark signal.

## Method

```
# FedStateMark: Dynamic Watermarking for Federated Generative Models
# Notation:
#   T: number of communication rounds (500)
#   C: set of clients
#   W_t: global model weights at round t
#   trigger: fixed set of latent codes/prompts for watermarking (e.g., 100 fixed seeds)
#   target: desired watermark pattern (e.g., binary signature of 128 bits)
#   Key: server-held secret key for random projection (matrix P of size 128 x |W_t|, with rows orthonormal)
#   λ: watermark loss weight (0.1)
#   η: fine-tuning learning rate (1e-5)
#   K: number of fine-tuning steps per round (10)
#   Total compute: ~1000 GPU hours on A100 (for T=500, 10 clients per round)

# Load-bearing assumption: The mapping from weights to watermark target via random projection is highly nonlinear
# such that averaging two weight sets produces a target that is uncorrelated with either original,
# because the projection is orthogonal across different keys (i.e., E[P_i · P_j^T] = 0 for i≠j).

1. For each round t = 1 to T:
   a. Sample clients, aggregate local updates to get W_t.
   b. Compute p = sign(P * flatten(W_t))   # random projection: P is fixed 128 x d matrix with rows orthonormal; sign gives binary bits
   c. Derive watermark pattern p = first-128-bits(h)  # (same as before, but now p from projection)
   d. Fine-tune W_t for K steps:
      For each step:
          - Generate images I_i from trigger set using W_t (e.g., Stable Diffusion)
          - For each I_i, extract bits e_i via fixed watermark extractor E (e.g., DCT-based decoder)
          - L_water = BinaryCrossEntropy(e_i, p) over all trigger samples
          - L_quality = LPIPS(I_i, reference I_i^0)  # consistency with pre-watermark output
          - Total loss = L_water + λ * L_quality
          - Update W_t (only decoder/unet part) with SGD
   e. Broadcast W_t to clients.

2. Verification at inference:
   a. Obtain suspect model W_s.
   b. Compute p_s = sign(P * flatten(W_s))
   c. Generate images from trigger set using W_s.
   d. Extract bits from images, compute accuracy against p_s.
   e. If accuracy > threshold (e.g., 0.9) then watermark detected.
```

**(C) Why this design**
We chose a random projection of the full model weights rather than a simpler hash because the projection can be designed to be orthogonal across different keys, ensuring that averaging two distinct weight snapshots yields a target that is uncorrelated with either original. This breaks the linear interpolation effect observed in model soups. Using a fixed projection matrix P with orthonormal rows guarantees that the expected inner product between projections of different keys is zero, assuming the keys are independent. The cost is that the projection matrix is large (128 x d, where d is the number of parameters), but it can be stored efficiently using a seed and generated on the fly. We fine-tune only the decoder to preserve semantic knowledge and reduce computational cost.

**(D) Why it measures what we claim**
The computational quantity `sign(P * flatten(W_t))` measures the **state specificity** of the watermark because it produces a unique output for nearly every distinct weight tensor, assuming the projection matrix is fixed and full-rank; this assumption fails if the projection matrix is not orthogonal (e.g., using a pseudorandom matrix with correlated rows), in which case averaging might not destroy the watermark—we use rows that are orthonormal. The `BinaryCrossEntropy` between extracted bits and the target pattern `p` measures **watermark presence** because it quantifies how closely the generated images match the expected pattern; this assumes the extractor E is robust to image transformations (assumption A). Failure mode F: compression or augmentation alters extracted bits, breaking equivalence—we mitigate by training with data augmentation and using a DCT-based extractor that is resilient to JPEG compression. The `LPIPS` loss measures **generation quality preservation** because it penalizes perceptual deviation from the original output; this assumes the pre-watermark output is a valid reference, but if the model has high variance across seeds, LPIPS may penalize acceptable variations—we mitigate by using a large trigger set and averaging. Together, these quantities ensure that a detected watermark is state-specific and not trivially removable by averaging.

## Contribution

(1) A novel federated watermarking framework, FedStateMark, that generates watermark patterns as a deterministic function of the global model weights via a keyed hash, making them round-specific and collusion-resistant. (2) A design principle that for federated learning, watermark patterns must be coupled with the model state to prevent cross-round averaging attacks; this generalizes beyond generative models to any federated instance. (3) A practical verification protocol that detects the watermark without storing per-round patterns, relying only on the secret key and trigger set.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | LAION-5B subset (10M images) | Large-scale, realistic for generative models. |
| Primary metric | Detection accuracy at 1% FPR | Measures watermark presence fairly. |
| Baseline 1 | FedOT (Yang et al.) | Static watermark not tied to weights. |
| Baseline 2 | WMAdapter (Liu et al.) | Separate encoder, no weight binding. |
| Baseline 3 | Stable Signature (Fernandez et al.) | Decoder fine-tuning but static target. |
| Ablation of ours | Static key instead of random projection | Tests necessity of weight-dependent target. |

### Why this setup validates the claim
This setup tests the central claim that watermark state-specificity (derived from a random projection of weights) improves robustness against model theft and averaging attacks. The LAION-5B subset ensures a realistic and challenging distribution where generative models are commonly trained. Detection accuracy at 1% FPR is chosen because it reflects the watermark's reliability while controlling for false alarms; a high accuracy under this metric would confirm that the watermark is present and uniquely bound to the model. The baselines represent prior art: FedOT uses a fixed pattern embedded into the model during training, WMAdapter adds a separate watermark encoder that is not weight-dependent, and Stable Signature fine-tunes the decoder with a static target. Our ablation replaces the dynamic random projection with a static key, isolating the contribution of weight-dependent target derivation. If our method outperforms all baselines on detection accuracy while maintaining parity on generation quality (measured separately but not as primary metric), the claim holds. The key falsification scenario is if our method's advantage is uniform across all attacks rather than concentrated against averaging — this would suggest that the random projection is not the source of improvement. Additionally, to ensure the extractor robustness assumption, we use a DCT-based extractor and train with random image augmentations (random crop, JPEG compression at quality 70).

### Expected outcome and causal chain
**vs. FedOT** — On a case where an attacker averages the model weights from multiple rounds (e.g., stealing via FedAvg), FedOT’s static watermark remains unchanged despite weight changes, so detection fails because the embedded watermarks are not tied to the current weights. Our method recomputes the target from the current weights via random projection; averaging preserves the watermark only if the projection is linear, but our orthogonal projection ensures that the mixed weights yield a target uncorrelated with either original; we expect detection accuracy >90% under averaging for FedStateMark vs <50% for FedOT.

**vs. WMAdapter** — On a case where an attacker fine-tunes the model on a new dataset, WMAdapter’s separate encoder may produce watermark patterns that are not bound to the diffusion decoder, so the extracted bits become random after fine-tuning. Our method fine-tunes the decoder itself with the weight-dependent target, so the watermark remains embedded even after adaptation; we expect detection accuracy >85% for FedStateMark vs <20% for WMAdapter after 1000 fine-tuning steps.

**vs. Stable Signature** — On a case where an attacker applies JPEG compression to generated images, Stable Signature’s static target (e.g., a fixed string) can be removed by training a surrogate decoder that suppresses the pattern. Our method’s target changes with each weight snapshot, making it harder to remove without access to the key; we expect detection accuracy >80% under JPEG compression (quality 50) for FedStateMark vs <40% for Stable Signature.

**vs. Ablation (static key)** — On a case where the attacker averages models from two different rounds (identical except for weight perturbation), the static key version uses the same target for both, so the averaged model's watermark is ambiguous. Our dynamic key gives each round a unique target, so averaging still yields the correct target from the mixed weights; we expect a gap of ≈30% in detection accuracy on weight-averaged models.

### What would falsify this idea
If our method's detection accuracy is not significantly higher than the best baseline on the most critical attack (e.g., model averaging or fine-tuning), or if the ablation with static key performs equally well, then the claim that random-projection state-specificity is crucial would be falsified. Additionally, if the random projection does not guarantee orthogonality in practice (e.g., due to numerical issues), averaging might still preserve watermark, falsifying the assumption.

## References

1. FedOT: Ownership Verification and Leakage Tracing via Watermarks for Federated LDMs
2. WMAdapter: Adding WaterMark Control to Latent Diffusion Models
3. The Stable Signature: Rooting Watermarks in Latent Diffusion Models
4. Flexible and Secure Watermarking for Latent Diffusion Model
5. IP-Adapter: Text Compatible Image Prompt Adapter for Text-to-Image Diffusion Models
6. CycleGANWM: A CycleGAN watermarking method for ownership verification
7. Versatile Diffusion: Text, Images and Variations All in One Diffusion Model
