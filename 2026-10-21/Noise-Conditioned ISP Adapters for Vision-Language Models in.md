# Noise-Conditioned ISP Adapters for Vision-Language Models in Low-Light Scenes

## Motivation

Existing benchmarks such as DarkQA evaluate a fixed pretrained vision-language model (VLM) as a black box under low-light and sensor noise, never modifying architecture or training, so noise-aware representational adaptation remains unestablished. The root cause is structural: current VLMs are trained on clean, ISP-processed images and thus lack any mechanism to condition their visual representations on the noise statistics that corrupt low-light inputs. Consequently, even state-of-the-art VLMs like LLaVA (Improved Baselines with Visual Instruction Tuning) degrade consistently on DarkQA's primitive QA tasks, and no prior work has tested whether conditioning on noise parameters can recover accuracy.

## Key Insight

Because sensor noise in low-light images is physically parameterized by ISO and exposure time, a VLM's visual encoder can be made noise-aware by conditioning its intermediate features on these parameters through lightweight adapters, turning a fixed black-box model into one whose representation is explicitly matched to the noise level—without retraining the entire model.

## Method

**What it is**: We introduce Noise-Conditioned ISP Adapters (NCIA), a lightweight module inserted into a pretrained VLM's vision encoder that modulates visual features using the camera's ISO and exposure time. Inputs: a low-light image and its associated noise parameters (ISO, exposure). Output: adapted visual tokens fed to the VLM's language model for question answering.

**How it works**:
```python
# Given: pretrained VLM with vision encoder E (e.g., CLIP-ViT), LLM decoder D
# Hyperparameters: adapter bottleneck dim r=64, FiLM scaling factor alpha=0.1, noise embedding dim d=32

# 1. Noise parameter encoding
z_noise = MLP([ISO, exposure])  # shape: (d,)

# 2. For each transformer block l in E (only last 4 blocks are adapted):
#    Insert a FiLM adapter after the MLP sublayer
h = E_l(h)  # original block output
# FiLM: generate scale and shift from z_noise
# gamma, beta are per-token (spatially varying): shape (num_tokens, hidden_dim)
# z_noise is broadcast to each token and concatenated with a per-token local noise proxy
# local noise proxy: for each token, compute the variance of pixel intensities in its corresponding image patch (patch size 16x16), then map to a scalar via log(1+variance)
# token_noise = [z_noise; log(1+variance_patch)]  # shape: (num_tokens, d+1)
gamma, beta = Linear(token_noise)  # each shape: (num_tokens, hidden_dim)
h_adapted = h * (1 + alpha * tanh(gamma)) + alpha * beta

# 3. Visual tokens from adapted encoder are projected and fed to D as usual
# 4. Fine-tune only adapters (and LayerNorm) on DarkQA training split
#    Loss: standard next-token cross-entropy on QA pairs
```

**Why this design**: We chose FiLM conditioning over concatenating noise parameters to the input because FiLM modulates intermediate features multiplicatively, which is more parameter-efficient and preserves the pretrained encoder's spatial structure, accepting the cost that the modulation is global per block rather than spatially varying. We adapt only the last 4 transformer blocks rather than all blocks because low-level noise primarily affects high-level semantic features, and adapting all blocks would overfit to the training noise distribution, accepting the risk that some low-level noise artifacts remain unaddressed. We fine-tune only adapters and LayerNorm while freezing the rest of the VLM, because full fine-tuning on DarkQA's 9.4K QA pairs would destroy the model's general visual-linguistic knowledge, accepting the limitation that adaptation capacity is bounded. We chose to condition on ISO and exposure (rather than a learned noise embedding) because these parameters are physically interpretable and directly available from camera metadata, accepting that the method requires such metadata at test time. A domain expert would not describe this as a variant of LLaVA's MLP projection or standard adapter tuning (Houlsby et al.) because the conditioning signal is the camera's noise parameters, which are absent from those methods; nor is it a simple concatenation of noise features, as the FiLM modulation acts multiplicatively on intermediate representations.

**Why it measures what we claim**: The FiLM-generated scale and shift (gamma, beta) measure the model's sensitivity to noise level because they are a deterministic function of ISO and exposure, under the assumption that these parameters fully capture the noise statistics of the image formation process; this assumption fails when noise is spatially heterogeneous (e.g., due to denoising in the ISP), in which case the modulation reflects only global noise level and ignores local variations. The adapter's output h_adapted measures noise-conditioned visual representation because the multiplicative modulation is applied after the pretrained block, under the assumption that the pretrained features are otherwise noise-agnostic; this assumption fails if the pretrained encoder already encodes some noise cues, in which case the adapter's effect is partially redundant. The final QA accuracy measures the recovery of primitive QA under sensor noise because the only difference between training and evaluation is the noise level, under the assumption that DarkQA's synthetic noise matches real sensor noise; this assumption fails if the synthesis deviates from real camera pipelines, in which case the accuracy gain may not transfer to real low-light images.

## Contribution

["A novel noise-conditioned adapter architecture (NCIA) that injects camera noise parameters (ISO, exposure) into a pretrained VLM's vision encoder via FiLM modulation, enabling noise-aware representational adaptation without full fine-tuning.", "A demonstration that noise-conditioned training on DarkQA's low-light scenes recovers primitive QA accuracy under sensor noise, establishing that parameter-conditioned adaptation is a viable bridge between black-box evaluation and noise-robust VLMs.", 'An analysis of which transformer blocks benefit most from noise conditioning, providing a design principle for efficient adaptation of VLMs to low-light conditions.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | DarkQA low-light QA pairs | Controlled multi-level illumination with sensor noise |
| Primary metric | QA accuracy on noise-stratified subsets | Detects gains where noise conditioning should matter |
| Baseline | Pretrained VLM, no adaptation | Isolates whether any adaptation helps |
| Baseline | LLIE-preprocessed VLM | Tests whether explicit denoising beats noise conditioning |
| Baseline | Standard Houlsby adapter (no noise conditioning) | Tests whether the ISO/exposure signal is the active ingredient |
| Ablation-of-ours | NCIA with noise input zeroed at test time | Confirms modulation depends on actual noise parameters |

### Why this setup validates the claim

The central claim is that conditioning visual features on camera noise parameters (ISO, exposure) via FiLM in the last encoder blocks recovers QA accuracy that degrades under sensor noise, and that this gain is causally attributable to the noise signal rather than to extra adapter capacity. The no-adaptation baseline tests whether the problem exists at all (accuracy should drop monotonically with noise level). The LLIE-preprocessed baseline tests the competing hypothesis that explicit image restoration is the right fix — if our method matches or beats it without restoring the image, the noise-conditioning route is validated as an alternative. The standard Houlsby adapter is the critical control: it has identical parameter budget and insertion points but no noise input, so any gap between it and NCIA isolates the contribution of the ISO/exposure conditioning. The zeroed-noise ablation directly probes the causal mechanism: if performance collapses to the unconditioned-adapter level when the noise signal is removed, the modulation is doing the work. QA accuracy stratified by noise level is the right metric because the predicted effect is a noise-dependent recovery, which a single aggregate accuracy would wash out.

### Expected outcome and causal chain

**vs. Pretrained VLM, no adaptation** — On a case where a dim indoor scene at high ISO contains a small object whose edges are smeared by noise, the frozen VLM produces a wrong primitive answer (e.g., misidentifies material or count) because its CLIP features were never exposed to this noise regime and the semantic tokens are corrupted. Our method instead rescales and shifts those block outputs by a function of the actual ISO and exposure, because the FiLM parameters learned during fine-tuning push noisy feature distributions back toward the clean manifold. We expect a substantial accuracy gap at the highest noise levels and near-parity at the lowest, i.e., a gap that grows with ISO.

**vs. LLIE-preprocessed VLM** — On a case where the low-light image has heterogeneous noise (bright window region plus dark corner), the LLIE baseline over-smooths or introduces artifacts in one region because its restoration is tuned for global illumination and can destroy fine detail. Our method instead leaves pixels untouched and modulates only high-level features, because the noise parameters provide a global conditioning signal that steers semantics without committing to a pixel-level restoration. We expect NCIA to be competitive on uniformly dark scenes but noticeably better on scenes with mixed illumination, where restoration artifacts hurt the LLIE pipeline.

**vs. Standard Houlsby adapter (no noise conditioning)** — On a case where two images share identical content but differ only in ISO, the unconditioned adapter produces the same adapted features and thus the same (possibly wrong) answer for the noisier one, because it has no input that distinguishes noise levels. Our method instead produces different gamma/beta for the two images, because the noise embedding is a deterministic function of ISO and exposure, so the noisier image receives stronger corrective modulation. We expect a consistent gap concentrated in the high-noise subset, with the two methods converging at low ISO — the signature that the noise signal, not adapter capacity, drives the gain.

### What would falsify this idea

If NCIA's advantage over the unconditioned Houlsby adapter is uniform across all noise levels rather than concentrated at high ISO, or if zeroing the noise input at test time leaves accuracy unchanged, then the conditioning signal is not the causal ingredient and the central claim is wrong.

## References

1. DarkQA: Benchmarking Vision-Language Models on Visual-Primitive Question Answering in Low-Light Indoor Scenes
2. Improved Baselines with Visual Instruction Tuning
3. Thinking in Space: How Multimodal Large Language Models See, Remember, and Recall Spaces
