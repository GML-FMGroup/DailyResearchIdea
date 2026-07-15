# CrossModalCalibrator: Calibrating MLLM Likelihood Rewards via Cross-Modal Consistency for Text-to-Image Generation

## Motivation

Existing MLLM-based reward models like SpectraReward (Read It Back) rely solely on the likelihood of the prompt given the image, which is uncalibrated: images that are visually plausible but semantically misaligned can achieve high likelihood because the MLLM can hallucinate missing details. The root cause is the absence of a cross-modal validation mechanism that verifies the image supports the prompt bidirectionally. Without such calibration, rewards may favor MLLM biases over true human preference.

## Key Insight

Cross-modal consistency, where the image both supports generating the prompt and the prompt supports generating the image's caption, provides a self-validating signal that penalizes mismatched pairs without requiring external human judgments.

## Method

### (A) What it is
**CrossModalCalibrator (CMC)** computes a reward for a generated image as an adaptively weighted combination of two log-likelihoods: (1) the log-likelihood of the original prompt given the image, and (2) the log-likelihood of an automatic caption of the image given both the prompt and the image (cross-modal condition). Both likelihoods are computed by the same frozen MLLM (e.g., LLaVA-1.5-7B or BLIP-2 with FlanT5-XL). The weighting is controlled by the semantic similarity between the prompt and the caption, penalizing cases where the caption diverges from the prompt. 

**Load-bearing assumption**: The log-likelihood of the caption given both the prompt and the image (ll_caption_given_prompt_and_image) measures cross-modal consistency because the image provides grounding for the caption. If this assumption fails (e.g., MLLM ignores the image when conditioned on prompt), then the metric becomes text-only. We mitigate this by using an MLLM that attends to images, but we do not guarantee perfect grounding.

### (B) How it works
```
Input: prompt p, image I, MLLM M, text encoder E (e.g., all-mpnet-base-v2 sentence-BERT)
Hyperparameters: k=5, threshold=0.5 (logistic function steepness and midpoint)

1. ll_prompt_given_image = M.log_likelihood( p | I )   # teacher-forced, average over tokens
2. caption = M.generate( I, max_tokens=30, temperature=1.0 )  # greedy decoding
3. ll_caption_given_prompt_and_image = M.log_likelihood( caption | p, I )  # cross-modal condition
4. sim = cosine_similarity( E(p), E(caption) )
5. weight = 1 / (1 + exp( -k * (sim - threshold) ))
6. reward = weight * ll_prompt_given_image + (1 - weight) * ll_caption_given_prompt_and_image
7. return reward
```
All likelihoods are averaged over token positions to normalize by length.

### (C) Why this design
We chose to combine two likelihoods weighted by semantic similarity rather than using a single likelihood or a fixed arithmetic mean. First, using only the prompt-given-image likelihood (as in SpectraReward) is insufficient because the MLLM can attribute high probability to fallacious details; adding the reverse direction introduces a consistency constraint. Second, we use a logistic weighting based on semantic similarity instead of a fixed balance because when the caption is semantically close to the prompt, the caption likelihood is a reliable signal, but when they diverge (e.g., due to noisy caption generation), we downweight that component to avoid penalizing good images due to poor captions. Third, we use a pretrained text encoder (sentence-BERT all-mpnet-base-v2) for similarity rather than the MLLM's own embeddings to maintain a lightweight, off-the-shelf similarity metric that is independent of the MLLM's biases. The trade-off is that the logistic function adds a hyperparameter that requires tuning, but we accept this cost for adaptive calibration. We avoid using a learned weighting because that would require preference data which contradicts the zero-shot goal.

### (D) Why it measures what we claim
- `ll_prompt_given_image` measures the **semantic fidelity** of the image to the prompt, under the assumption that the MLLM perfectly captures all prompt constraints when reconstructing the prompt from the image. This assumption fails when the prompt is generic or ambiguous, in which case the metric reflects the MLLM's prior over plausible captions rather than true alignment.
- `ll_caption_given_prompt_and_image` measures **cross-modal consistency**, under the assumption that the MLLM uses the image to ground the caption when conditioned on both prompt and image. This assumption fails when the MLLM ignores the image (due to strong language prior) or when the caption is hallucinated, in which case the metric becomes text-only coherence or generation artifacts.
- `weight` (based on semantic similarity between p and caption) measures the **reliability of the caption likelihood**: when similarity is high, the caption is assumed to be a faithful summary of the image w.r.t. the prompt, so its likelihood is trusted; when similarity is low, the caption is likely off-target and its likelihood is downweighted. This assumption fails when semantic similarity is insensitive to fine-grained misalignment (e.g., the prompt and caption are topically similar but miss key attributes), in which case the weight may overtrust the caption likelihood.

## Contribution

(1) A zero-shot reward calibration framework, CrossModalCalibrator, that adaptively combines prompt-to-image and caption-to-prompt likelihoods using semantic similarity, addressing the lack of cross-modal validation in existing MLLM reward models. (2) A structural design principle: cross-modal consistency provides a self-validating signal that inherently penalizes images with high unilateral likelihood but poor alignment, without requiring any training or human annotation. (3) An operationalization of adaptive weighting via logistic function on semantic similarity, demonstrating how to balance two imperfect signals in a reliability-aware manner.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|----------------------|
| Dataset | DrawBench + TIFA v1.0 (500 prompts) | DrawBench for diverse composition; TIFA for fine-grained alignment |
| Primary metric | Human Preference (win rate) | Direct measure of perceived alignment |
| Baseline 1 | CLIP Score (ViT-L/14) | Standard zero-shot reward baseline |
| Baseline 2 | ImageReward (v1.0) | Supervised reward model baseline |
| Baseline 3 | SpectraReward | MLLM-only prompt-given-image baseline |
| Ablation-of-ours | CMC (fixed average, weight=0.5) | Removes adaptive weighting, tests its necessity |

### Why this setup validates the claim
DrawBench contains 200+ prompts covering various linguistic phenomena (e.g., spatial relations, counting, colors), making it ideal to test alignment beyond simple object detection. TIFA v1.0 provides 500 prompts with fine-grained semantic attributes (e.g., object count, relative size) that directly test whether the reward captures subtle misalignment. Human preference (win rate) is the gold standard for perceived image-text alignment, directly capturing whether our reward better matches human judgment than baselines. Comparing against CLIP Score tests the necessity of going beyond visual-semantic cosine similarity; ImageReward tests whether zero-shot MLLM-based reward can surpass supervised models; SpectraReward isolates the effect of adding the reverse likelihood direction with cross-modal conditioning. The ablation (fixed average) tests whether the adaptive weighting mechanism is crucial. If CMC consistently outperforms all baselines, especially SpectraReward and the ablation, it confirms that combining both likelihoods with adaptive weighting and cross-modal conditioning provides a better reward signal. Conversely, failure on any baseline would pinpoint the flaw in our central claim.

### Expected outcome and causal chain

**vs. CLIP Score** — On a compositional prompt like "a red cube on top of a blue sphere", CLIP may assign high similarity to an image with red and blue objects but wrong spatial layout because it treats the image as a bag of visual features. Our method uses the MLLM to verify prompt recoverability (ll_prompt_given_image) and cross-modal caption consistency (ll_caption_given_prompt_and_image), so it penalizes spatial errors. We expect a noticeable gap in win rate on compositional prompts (e.g., +10–15%) but near-parity on simple color-object prompts where CLIP already works well. Toy example: For prompt "a cat and a dog", an image of only a cat yields ll_prompt_given_image moderate (MLLM can reconstruct "a cat"), but caption given prompt+image yields low likelihood because caption will mention only cat, but prompt expects both; weight remains high (semantic similarity high since cat present), but low caption likelihood reduces reward. Fixed average would also penalize, but adaptive weighting amplifies the penalty when caption misses content.

**vs. ImageReward** — On a stylized prompt like "a photorealistic lion in a fantasy forest", ImageReward may favor aesthetically pleasing but less faithful images due to its training on human preferences that mix alignment and aesthetics. Our method is zero-shot and focuses solely on semantic consistency between prompt, image, and caption, avoiding aesthetic bias. We expect a moderate gap (e.g., +5–8%) on stylized or abstract prompts where aesthetic quality diverges from alignment, but similar performance on straightforward photorealistic prompts.

**vs. SpectraReward** — On a generic prompt like "a cat", SpectraReward assigns high likelihood to any cat image because the MLLM can reconstruct "a cat" from most cat pictures, ignoring extra details. Our method computes caption likelihood given prompt+image; a generic prompt with a detailed image yields low caption likelihood (caption is detailed, but prompt is generic), reducing reward. Thus, we expect a gap on generic prompts (e.g., +8–12%) where SpectraReward over-rewards, but near-identical performance on specific prompts where both directions are high.

**vs. CMC (fixed average)** — On prompts where the caption semantically diverges from the prompt (low sim), the fixed average downweights the reverse direction incorrectly, while adaptive weighting reduces its influence. We expect CMC (adaptive) to outperform fixed average on such prompts (e.g., +3–5%), but perform similarly when sim is high.

### What would falsify this idea
If CMC fails to outperform the ablation (fixed average) on prompts where the caption diverges from the prompt (low semantic similarity), then the adaptive weighting is unnecessary. Additionally, if CMC underperforms CLIP Score across all prompt types, the MLLM-based approach is fundamentally weaker than simple similarity. If the improvement over SpectraReward is negligible, the cross-modal conditioning added no value.

### Feasibility
Each image requires two forward passes (for ll_prompt_given_image and ll_caption_given_prompt_and_image) and one caption generation. With LLaVA-1.5-7B on a single A100 GPU, estimated runtime is ~0.3 seconds per image (batch size 1). Memory usage ~16 GB for model weights plus ~2 GB for intermediate activations. For 10,000 images, total evaluation time ~50 minutes, feasible for research-scale experiments.

## References

1. Read It Back: Pretrained MLLMs Are Zero-Shot Reward Models for Text-to-Image Generation
2. SRUM: Fine-Grained Self-Rewarding for Unified Multimodal Models
3. Uniworld-V2: Reinforce Image Editing with Diffusion Negative-aware Finetuning and MLLM Implicit Feedback
4. RewardDance: Reward Scaling in Visual Generation
5. Emu3.5: Native Multimodal Models are World Learners
6. VisionReward: Fine-Grained Multi-Dimensional Human Preference Learning for Image and Video Generation
7. Mixture-of-Transformers: A Sparse and Scalable Architecture for Multi-Modal Foundation Models
8. LMFusion: Adapting Pretrained Language Models for Multimodal Generation
