# RefineText: Iterative Fine-Tuning of Text-Rich Image Diffusion Models via Verifier-Guided Self-Improvement

## Motivation

Existing text-rich image generation methods like AnyText produce rendering errors such as missing strokes or illegible characters, yet the error feedback loop is incomplete because verifiers (e.g., OCR-based scorers) only filter low-quality samples without improving the generator itself. DataEvolver constructs synthetic data using a multi-agent pipeline but does not fine-tune the generator on verified samples, so fundamental errors persist across rounds. A closed-loop mechanism that uses verifier scores to select high-quality synthetic outputs for retraining could directly correct recurring rendering mistakes.

## Key Insight

Fine-tuning the diffusion model on its own verifier-approved outputs creates a positive feedback loop where the model learns to avoid its typical errors because the verifier's scoring function identifies those errors and the training distribution shifts toward corrected samples.

## Method

## (A) What it is
**RefineText** is an iterative fine-tuning framework that takes a base text-rich image diffusion model (e.g., AnyText), generates candidate images, scores them with a verifier, collects top-scoring samples, and fine-tunes the model on those samples using a composite loss. It outputs an improved generator after several rounds.

## (B) How it works
```python
# Hyperparameters
rounds = 3
k = 100  # number of top samples per round
buffer_capacity = 300
images_per_round = 1000  # generate 1000 images from 200 prompts, 5 each

# Initialize base model: AnyText (pretrained)
model = AnyText()
# Initialize verifier: CRNN model from OCR-VQGAN (pretrained on ICDAR and SynthText)
verifier = CRNN()  # outputs CER ∈ [0,1], lower is better
# Initialize replay buffer
replay_buffer = []

for round in range(rounds):
    # 1. Generate batch of images
    prompts = [...]  # 200 prompts sampled from Paper2Fig100k (diverse fonts and styles)
    generated = model.generate(prompts, num_images_per_prompt=5)  # outputs 1000 images
    
    # 2. Score each image via verifier
    scores = [verifier.compute_cer(img, prompt) for img, prompt in zip(generated, prompts)]
    # Low CER = high quality
    
    # 3. Select top-k samples
    indices = np.argsort(scores)[:k]
    top_images = [generated[i] for i in indices]
    top_prompts = [prompts[i] for i in indices]
    
    # 4. Update replay buffer
    replay_buffer.extend(list(zip(top_images, top_prompts)))
    if len(replay_buffer) > buffer_capacity:
        replay_buffer = replay_buffer[-buffer_capacity:]
    
    # 5. Fine-tune model for one epoch on replay buffer
    loss_fn = lambda img, prompt: L1_diffusion(img, prompt) + 0.5 * text_perceptual_loss(img, prompt)
    # L1_diffusion: standard denoising loss from AnyText (text-control diffusion loss)
    # text_perceptual_loss: L2 distance between OCR features (from verifier's layer4) of generated and target images
    model.fine_tune(replay_buffer, loss_fn, lr=1e-5, batch_size=16, steps=200)

# Output: improved model
```

## (C) Why this design
We chose iterative fine-tuning over single-step because the error distribution shifts as the model improves; a single round may correct only surface errors while deeper issues persist. The composite loss combines the original diffusion loss with a text perceptual loss (from OCR-VQGAN) – the former preserves overall image structure, the latter explicitly forces high text fidelity. The trade-off is that the perceptual loss adds computational cost and may overemphasize OCR-feature alignments that do not always match human perception. We use a replay buffer of past high-quality samples to prevent catastrophic forgetting of previously learned capabilities, accepting increased memory usage. We selected top-k (k=100) rather than a fixed threshold because the score distribution changes across rounds; a fixed threshold could become too strict or too loose. The verifier outputs CER, which is a direct measure of text accuracy, but its reliability depends on OCR model quality – we assume the OCR model is trained on diverse fonts and languages; if it fails on rare scripts, the selection becomes noisy. **Load-bearing assumption**: The verifier's CER is monotonic with human-judged text accuracy across all styles (including stylized fonts and rare characters). If violated, self-training may amplify verifier errors. To monitor this, we use a held-out human-annotated validation set (512 examples) to measure CER–human correlation per font category, and we discard fine-tuning rounds where correlation drops below 0.7.

## (D) Why it measures what we claim
The verifier's CER score measures text rendering accuracy because it directly compares the generated text to the ground-truth string, assuming the OCR decoder maps pixels to characters correctly; this assumption fails when the OCR misreads stylized fonts (e.g., gothic or artistic text), in which case CER reflects OCR error rather than true rendering accuracy. The human-annotated validation set (512 examples with 4 font categories: normal, stylized, rare characters, artistic) provides a calibration: if CER–human correlation is high for a category, it is reliable; otherwise, we exclude such categories from fine-tuning. The text perceptual loss (L2 distance between OCR features of generated and target image) measures structural text fidelity because it penalizes deviations in feature space that correspond to character-level shape, assuming these features are discriminant and invariant to small non-text variations; this assumption fails when the OCR model's feature space does not align with human perceptual boundaries (e.g., it may be over-sensitive to font family while humans tolerate minor style changes). The fine-tuning loss (L1_diffusion + text perceptual) operationalizes the concept of error correction by rewarding samples that the verifier deems correct, thereby shifting the model's output distribution toward the verifier-accepted region; the key assumption is that the verifier's scoring is monotonic with true text accuracy, which holds for standard OCR in common settings but breaks for out-of-distribution text styles.

## Contribution

(1) RefineText, a closed-loop fine-tuning framework that iteratively improves a text-rich image diffusion model by using its own verifier-approved synthetic samples. (2) Empirical finding that combining the original denoising loss with an OCR perceptual loss during fine-tuning yields better text accuracy than either loss alone, as the perceptual loss targets text-specific features while the denoising loss preserves overall image quality. (3) Demonstration that iterative self-improvement with verifier feedback can correct systematic rendering errors in base models like AnyText, reducing character error rates on benchmarks.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
|------|--------|-----------|
| Dataset | Paper2Fig100k (2000 prompts, 5 images each) | Standard text-rich image benchmark; we use 2000 prompts sampled to cover diverse fonts, languages, and complexities. |
| Primary metric | CER (Character Error Rate) | Direct text accuracy measure; computed using the verifier CRNN (same as used in selection). |
| Baseline 1 | AnyText (base) | Base model without iterative fine-tuning. |
| Baseline 2 | DataEvolver | Self-evolving data construction baseline (reimplemented with same prompts). |
| Baseline 3 | Static data pipeline | Traditional static data curation (no iterative selection). |
| Ablation-of-ours | Ours w/o text perceptual loss | Isolates contribution of perceptual loss (loss = L1_diffusion only). |
| Ablation-of-ours | Ours w/o verifier selection (random selection) | Isolates contribution of verifier-guided selection (selects k=100 random samples each round). |

**Compute estimate**: Each round: generate 1000 images (~50 A100 GPU hours), fine-tune for 200 steps (~10 GPU hours). Total: ~180 GPU hours for 3 rounds (including verifier scoring).

### Why this setup validates the claim

This experimental design tests the central claim that iterative fine-tuning guided by a verifier and composite loss improves text rendering over static training. By comparing to the base model AnyText, we isolate the effect of iterative refinement. DataEvolver represents a prior self-evolving approach, testing whether our verifier-based selection yields better error correction. The static pipeline baseline demonstrates the necessity of dynamic data curation. The ablation without text perceptual loss reveals the specific contribution of the auxiliary loss. The ablation without verifier selection (random selection) specifically tests the necessity of the verifier's scoring: if random selection performs similarly, then the improvement is not due to verifier guidance but merely iterative fine-tuning. The primary metric, CER, directly quantifies text rendering accuracy, enabling a falsifiable test: if our method fails to significantly reduce CER on challenging subsets (e.g., stylized fonts, rare characters) where baselines struggle, the claim is invalidated. The combination ensures that observed improvements can be attributed to our proposed mechanisms rather than confounds. Additionally, we evaluate per font category using human annotations from a validation set (512 examples) to ensure that CER improvements are not due to verifier bias.

### Expected outcome and causal chain

**vs. AnyText (base)** — On a case with highly stylized font (e.g., gothic), the base model produces misaligned or missing characters because its single-stage training cannot correct specific rendering errors. Our method iteratively fine-tunes on top-CER samples, directly targeting these failures; the composite loss with text perceptual loss enforces fidelity to OCR features. We expect a noticeable gap (e.g., 5% vs 20% CER) on stylized fonts, but near-parity on simple fonts where base model already performs well. However, if the verifier's CER is unreliable on gothic fonts (low correlation with human judgment), our method may overfit to verifier-preferred styles; we monitor this via the human validation set and would exclude such rounds if correlation <0.7.

**vs. DataEvolver** — On a rare character (e.g., Chinese character with many strokes), DataEvolver's static crawl-filter pipeline may fail to generate enough high-quality examples, leading to poor rendering. Our verifier selects top-CER samples regardless of frequency, including rare characters that pass verification; iterative fine-tuning on these samples progressively improves rendering. We expect lower CER on rare characters (e.g., halving error rate) while maintaining performance on common ones. The human validation set ensures that verifier's decisions for rare characters align with human correctness.

**vs. Static data pipeline** — On out-of-distribution text (e.g., artistic calligraphy), static pipeline cannot adapt, resulting in high CER because training data lacks diversity. Our method dynamically selects generated images with low CER, effectively adapting to the target distribution via fine-tuning. We expect our method to degrade gracefully (e.g., 10% CER) whereas static pipeline fails drastically (e.g., 40% CER) on such cases.

**vs. Ours w/o verifier selection (random selection)** — On any challenging subset, random selection picks many low-quality samples, leading to slower convergence and no targeted error correction. We expect verifier-guided selection to outperform random by a clear margin (e.g., 5% vs 15% CER on stylized fonts) because verifier directly identifies the model's current weaknesses.

### What would falsify this idea

If the CER improvement over baselines is uniform across all text categories (rather than concentrated on challenging subsets like stylized fonts or rare characters), or if the ablation without perceptual loss or without verifier selection performs equally, then the central claim that iterative selection and text perceptual loss cause the improvement is false. Moreover, if the human validation set shows that CER improvements on stylized fonts come with high verifier-human disagreement (e.g., correlation <0.5), then the verifier is unreliable and the method may be amplifying noise.

## References

1. DataEvolver: Self-Evolving Multi-Agent Data Construction for Text-Rich Image Generation
2. AnyText: Multilingual Visual Text Generation And Editing
3. AgentInstruct: Toward Generative Teaching with Agentic Flows
4. AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation
5. MetaMath: Bootstrap Your Own Mathematical Questions for Large Language Models
6. OCR-VQGAN: Taming Text-within-Image Generation
7. Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding
8. Character-Aware Models Improve Visual Text Rendering
