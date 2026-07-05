# RealFilter: Closing the Real-World Fidelity Gap in Synthetic Data Generation for Text-Rich Images

## Motivation

Existing synthetic data generation methods for text-rich images, such as DataEvolver, rely on a synthetic verifier to assess data quality. However, this verifier is trained exclusively on synthetic data, causing it to be blind to artifacts that are absent from the synthetic distribution but present in real-world images. Consequently, generated data may pass the verifier yet still lack real-world fidelity, limiting downstream model performance.

## Key Insight

The synthetic verifier's decision boundary and a real discriminator's decision boundary are complementary—one operates on synthetic-relevant features, the other on real-world texture and lighting statistics—so their intersection yields a faithful subset.

## Method

### (A) What it is
RealFilter is a dual-verification pipeline that augments a synthetic verifier (from DataEvolver) with an **ensemble** of real-world discriminators trained on a small set of authentic text-rich images. Its input is a set of candidate synthetic images produced by the Generator agent; its output is a filtered subset that satisfies both critics.

### (B) How it works (pseudocode)
```python
# Phase 1: Synthetic Verification (as in DataEvolver)
candidates = Generator.synthesize(batch_size=64)
scores, reasons = Verifier.evaluate(candidates)  # returns quality score 0-1 and failure reasons
passed_synthetic = [c for c, s in zip(candidates, scores) if s > threshold_syn]  # threshold_syn=0.7

# Phase 2: Ensemble of Real Discriminators
# Train 5 ViT-B/16 discriminators on bootstrapped subsets of N=500 real text-rich images (e.g., from Paper2Fig100k or AnyText dataset)
# Each bootstrap sample is drawn with replacement from the 500 real images, size = 500.
# For each, fine-tune ViT with binary classifier head (real=1, synthetic negatives=0) for 10 epochs, lr=1e-5.
# Synthetic negatives are 5000 random samples from DataEvolver's generator before filtering.
ensemble_discs = []
for i in range(5):
    boot_real = bootstrap_sample(real_images, size=500)
    disc = load_pretrained_dino_vit()
    fine_tune(disc, boot_real + synthetic_negatives)
    ensemble_discs.append(disc)
# Apply to passed_synthetic: average logits from ensemble
ensemble_scores = []
for c in passed_synthetic:
    logits = [disc(c) for disc in ensemble_discs]
    avg_logit = torch.mean(torch.stack(logits), dim=0)
    prob_real = torch.sigmoid(avg_logit)
    ensemble_scores.append(prob_real)
passed_both = [c for c, d in zip(passed_synthetic, ensemble_scores) if d > threshold_real]  # threshold_real=0.8
```
**Hyperparameters**: batch_size=64, threshold_syn=0.7 (tuned on a small validation set of 50 real images), threshold_real=0.8 (chosen to allow at least 50% pass rate on synthetic data that also passed verifier, adjusted based on held-out real images to maintain 95% recall of real images). Ensemble size = 5; each ViT-B/16 fine-tuned for 10 epochs with lr=1e-5; synthetic negatives pool = 5000. The real discriminator ensemble is frozen after fine-tuning.

### (C) Why this design
We chose an ensemble of pre-trained ViTs fine-tuned on bootstrapped subsets of a small real set rather than a single model because the small real set (500 images) is insufficient for learning high-level features and prone to overfitting; the ensemble averages out variance and improves generalization. The ViT's pre-training on ImageNet provides a strong prior that generalizes to text-rich images, accepting the cost that some domain-specific artifacts (e.g., OCR-specific) might not be captured. We set a high threshold_real=0.8 to prioritize precision over recall, ensuring that kept samples are very likely to be real-looking, while the synthetic verifier already ensures text correctness; the trade-off is discarding some plausible synthetic samples that are borderline real. We treat the synthetic verifier and real discriminator as independent cascades rather than a joint model because independence allows modular replacement (e.g., switch verifier) and avoids training a large combined model with limited real data; the cost is that we cannot leverage correlations between verifier and discriminator scores.

### (D) Why it measures what we claim
The synthetic verifier score s measures adherence to synthetic data distribution (i.e., text accuracy, layout consistency) because the Verifier was trained on synthetic samples with known failures; this assumption fails when synthetic distribution differs from real, in which case s reflects fitness to synthetic norms only. The real discriminator ensemble score d measures realism relative to a small real sample because it is trained as a binary classifier between bootstrapped real subsets and synthetic negatives; this assumption fails when the real set is not representative (e.g., only one font type), in which case d reflects only that specific real distribution. The intersection filter (s > threshold_syn and d > threshold_real) measures real-world fidelity closure because it requires approval from both complementary critics; the **load-bearing assumption** is that any artifact that escapes the synthetic verifier is captured by the real discriminator (since the real discriminator sees no synthetic artifacts of that type), and vice versa. This assumption fails when an artifact is invisible to both (e.g., a realistic-looking but semantically wrong text recognized by neither verifier nor discriminator), in which case the filtered data may still contain such hidden flaws. To mitigate overfitting and improve generalization of the real discriminator, we use an ensemble fine-tuned on bootstrapped subsets; we verify calibration by ensuring that the ensemble's average score on a held-out set of 100 real images exceeds 0.9.

## Contribution

(1) RealFilter, a dual-verification pipeline that combines a synthetic verifier with a real-world discriminator to close the fidelity gap in synthetic data generation. (2) An empirical finding that a small real dataset (500 images) suffices to train a discriminator that effectively filters out synthetic artifacts missed by the verifier, demonstrated through downstream generation quality on text-rich images.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Paper2Fig100k test set | Real text-rich images for evaluation |
| Primary metric | FID (lower is better) | Measures visual realism globally |
| Baseline 1 | DataEvolver (verifier only) | Synthetic verifier without real discriminator |
| Baseline 2 | RealDiscrimOnly (discriminator only) | Single fine-tuned ViT (no ensemble) without synthetic verifier |
| Baseline 3 | StaticGeneration (no filter) | Unfiltered synthetic images |
| Ablation-of-ours | RealFilter (no fine-tune) | ViT frozen, no fine-tuning on real images |

### Why this setup validates the claim
This combination tests the central claim that dual verification (synthetic + real) improves real-world fidelity. Paper2Fig100k provides diverse real text-rich images for FID evaluation. DataEvolver isolates the added benefit of the real discriminator; RealDiscrimOnly isolates the benefit of the synthetic verifier; StaticGeneration measures the baseline without any filtering. The ablation (no fine-tune) tests whether fine-tuning the discriminator on a small real set is crucial. FID captures overall realism; if our method reduces FID more than baselines, it supports the claim that the dual filter removes unrealistic artifacts missed by individual critics. The ablation shows the contribution of the fine-tuning step. A falsifiable pattern is that gains should be largest on images where one critic is weak (e.g., synthetic verifier passes unrealistic but textually correct images). Additionally, we quantify the fraction of artifacts captured by at most one critic by manually labeling a random subset of 200 filtered images; if this fraction is >50%, the complementary assumption is supported.

### Expected outcome and causal chain

**vs. DataEvolver** — On an image where the synthetic verifier passes a candidate with correct text but unnatural background (e.g., a blurry font on a repetitive pattern), DataEvolver accepts it because text is correct, but the real discriminator ensemble detects the unrealistic background (low d score). Our method rejects it, lowering FID by excluding such samples. We expect a noticeable FID gap (e.g., 10% lower) on subsets with complex backgrounds, but parity on simple backgrounds where both critics agree.

**vs. RealDiscrimOnly** — On an image where the real discriminator passes a candidate with realistic texture but garbled text (e.g., characters missing strokes but overall appearance is realistic), RealDiscrimOnly accepts it because the discriminator focuses on global realism, missing text errors. Our synthetic verifier flags it (low s score) and rejects it, thereby removing artifacts that increase FID due to text inconsistencies. We expect a larger FID advantage for our method on images with text-heavy content, especially those with unusual fonts or layouts that the discriminator cannot penalize. Since RealDiscrimOnly uses a single ViT (not ensemble), our ensemble also reduces overfitting, leading to additional gains on diverse backgrounds.

**vs. StaticGeneration** — On any typical candidate, StaticGeneration accepts all images, including those with obvious artifacts like distorted text or implausible colors. Our method filters out such low-quality samples via both critics. The FID improvement over StaticGeneration is expected to be large (e.g., 30% lower) and consistent across all test image categories since no filtering is applied.

### What would falsify this idea
If our method's FID improvement over DataEvolver is uniform across all image categories rather than concentrated on complex-background or text-heavy subsets where the synthetic verifier alone is predicted to fail, then the dual verification is not capturing complementary failure modes. Additionally, if the fraction of artifacts captured by at most one critic (measured by human annotation on 200 filtered images) is less than 20%, the complementary blind spots assumption is unsupported.

## References

1. DataEvolver: Self-Evolving Multi-Agent Data Construction for Text-Rich Image Generation
2. AnyText: Multilingual Visual Text Generation And Editing
3. AgentInstruct: Toward Generative Teaching with Agentic Flows
4. AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation
5. MetaMath: Bootstrap Your Own Mathematical Questions for Large Language Models
6. OCR-VQGAN: Taming Text-within-Image Generation
7. Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding
8. Character-Aware Models Improve Visual Text Rendering
