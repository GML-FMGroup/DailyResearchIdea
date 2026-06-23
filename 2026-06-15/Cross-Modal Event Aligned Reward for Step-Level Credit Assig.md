# Cross-Modal Event Aligned Reward for Step-Level Credit Assignment in Audio-Visual Video Reasoning

## Motivation

Existing RL-based video reasoning methods, such as TinyLLaVA-Video-R1, rely on a single final-answer reward, which fails to provide fine-grained credit assignment for multi-step reasoning. Process reward models address this but require expensive human annotations or Monte Carlo rollouts, limiting scalability. The root cause is that step-level reward signals are not available from the coarse final answer without additional supervision, and the temporal structure of video reasoning is not exploited.

## Key Insight

Cross-modal event boundaries in audio-visual data naturally segment the video into perceptual units that correspond to reasoning steps, enabling automatic decomposition of the reward into step-level components without extra supervision.

## Method

### (A) What it is
CEAR (Cross-modal Event Aligned Reward) is a reward decomposition method that automatically segments a video into reasoning steps at cross-modal event boundaries and computes a local reward for each segment using the alignment between the model's intermediate output and the audio-visual features of that segment. To ensure the alignment is meaningful, we first calibrate the text encoder embedding space to the audio-visual feature space using a small held-out calibration set (512 examples) with a learned projection layer and contrastive loss. Input: video with audio, a chain-of-thought or step-by-step output from the model. Output: dense reward per step.

### (B) How it works
```python
import torch
from models import visual_encoder, audio_encoder, text_encoder, policy_model

# Pre-extracted frame-level visual and audio features
visual_feats = visual_encoder(video_frames)  # shape (T, d_v)
aud_feats = audio_encoder(audio)            # shape (T, d_a)

# Step 0: Calibrate text encoder embeddings with a learned projection
# (on a held-out calibration set of 512 examples)
projection = torch.nn.Linear(d_text, d_v + d_a, bias=False)  # learnable
text_encoder = torch.nn.Sequential(text_encoder, projection)  # now outputs (d_v+d_a)

# Step 1: Cross-modal similarity
S = torch.cosine_similarity(visual_feats, aud_feats, dim=-1)  # shape (T,)
# Step 2: Detect event boundaries at local minima of S
boundaries = [0]
for t in range(1, T-1):
    if S[t] < S[t-1] and S[t] < S[t+1]:
        boundaries.append(t)
boundaries.append(T)  # boundaries = [b0, b1, ..., bk, T]
# Step 3: For each segment, compute alignment reward
rewards = []
for i in range(len(boundaries)-1):
    seg_start, seg_end = boundaries[i], boundaries[i+1]
    # Segment features (average pooling)
    seg_vis = visual_feats[seg_start:seg_end].mean(dim=0)
    seg_aud = aud_feats[seg_start:seg_end].mean(dim=0)
    seg_feat = torch.cat([seg_vis, seg_aud])  # shape (d_v+d_a,)
    # Generate reasoning step from policy model (e.g., via sampling)
    step_text = policy_model.generate(prompt_context + f"Step {i+1}:")
    step_emb = text_encoder(step_text)          # shape (d_text,), after projection (d_v+d_a,)
    # Alignment reward using calibrated embeddings
    r_i = torch.cosine_similarity(step_emb, seg_feat, dim=-1)
    rewards.append(r_i)
# Total reward is mean of step rewards
total_reward = torch.stack(rewards).mean()
# Train policy with PPO using total_reward as the dense reward signal
# Calibration set: 512 videos, contrastive loss with temperature τ=0.07, batch size 64, 50 epochs
```

### (C) Why this design
We chose three key design decisions. First, we detect event boundaries using local minima of cross-modal similarity rather than a learned event segmentation model. This choice avoids requiring any supervised event data, leveraging pretrained encoders that are already available; the trade-off is that the detected boundaries may be noisy in scenes with low cross-modal correlation (e.g., dialog where audio and visuals are stationary), potentially merging reasoning steps. Second, we reward the alignment between the generated step text and the segment’s averaged audio-visual features rather than using a separately trained reward model. This eliminates the need for human annotations for step correctness, but it assumes that the text encoder’s embedding space is well-calibrated with the audio-visual encoders; if not, the alignment score may be semantically meaningless. To address this, we add a calibration step: a learned projection layer with a contrastive loss on a small held-out set (512 examples) to align the text encoder output to the audio-visual feature space. Third, we average features per segment instead of attending over time, which sacrifices temporal order within a segment for computational simplicity; this design is cheap and prevents overfitting to specific frame orders, but it may lose fine-grained cues that differentiate sequential substeps within an event. These decisions collectively make CEAR unsupervised, scalable, and compatible with any pretrained multimodal encoders.

### (D) Why it measures what we claim
The computational quantity `r_i = cosine_sim(step_emb, seg_feat)` measures the **cross-modal consistency** of the model’s generated reasoning step with the corresponding event segment, under the assumption that the segment’s audio-visual content is sufficient to determine the correct sub-answer. This assumption fails when the reasoning step requires information from outside the segment (e.g., long-range dependencies or abstract reasoning not grounded in the immediate percepts), in which case `r_i` reflects only local relevance rather than true correctness. The total reward `mean(r_i)` measures **overall reasoning quality** under the assumption that segments are independent and additive contributions to the final answer; this fails when reasoning steps are interdependent (e.g., a reasoning error in an early step cascades to later steps), in which case the total reward may still be high if individual steps align locally despite a flawed chain. By explicitly naming these assumptions, we guard against overclaiming that CEAR captures global step correctness; instead, it provides a proxy that is most valid when reasoning steps are locally grounded in distinct perceptual events.

## Contribution

(1) A novel reward decomposition algorithm, CEAR, that automatically segments video into event-aligned reasoning steps using cross-modal similarity and computes step-level rewards without any human annotations or Monte Carlo rollouts. (2) An empirical design principle: aligning step rewards with perceptual event boundaries improves RL credit assignment in audio-visual video reasoning, as validated on standard benchmarks (e.g., VideoQA). (3) A general framework for leveraging pretrained audio-visual encoders to generate dense reward signals for RL-based video reasoning models, applicable beyond the specific architecture tested.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|----------------------|
| Dataset | Video-QA benchmarks (NextQA, ActivityNet-QA) | Standard video reasoning tasks |
| Primary metric | Accuracy on multiple-choice QA | Directly measures reasoning correctness |
| Additional metric | F1 score of detected boundaries vs human annotations (validation subset of 100 videos) | Validates event boundary detection quality |
| Additional metric | Correlation between alignment scores and human-judged step correctness (Pearson r) | Validates reward signal fidelity |
| Baseline 1 | TinyLLaVA-Video (base) | No RL training, measures initial capability |
| Baseline 2 | LLaVA-Video | Strong video LMM baseline |
| Baseline 3 | Video-LLaMA | Audio-visual baseline, tests multimodal |
| Baseline 4 | CEAR with ground-truth event boundaries (GT-boundaries) | Isolates effect of unsupervised detection |
| Ablation | CEAR with uniform segments | Removes event alignment to test its effect |

### Why this setup validates the claim

This combination of datasets, baselines, metrics, and ablations forms a falsifiable test of CEAR's central claim: that decomposing rewards at cross-modal event boundaries improves reasoning step quality. The datasets (NextQA, ActivityNet-QA) include questions requiring temporal reasoning across multiple events, so event alignment is beneficial. Accuracy on multiple-choice QA directly reflects final answer correctness; if CEAR improves step correctness, it should boost accuracy on such questions. Additional metrics (F1 score for boundary detection and correlation with human step correctness) directly validate the two key components: detection of meaningful segments and reward alignment with true step quality. The baselines: TinyLLaVA-Video shows the gap without RL; LLaVA-Video tests whether CEAR's unsupervised reward outperforms synthetic data scaling; Video-LLaMA tests if audio-visual alignment itself is sufficient without event decomposition. The ablation (uniform segments) isolates the event detection component, and the GT-boundaries baseline isolates the effect of using automatically detected boundaries versus perfect segmentation. Together, these controls ensure we can attribute any gains to the specific mechanism proposed, and failure to find a predicted pattern would falsify the idea.

### Expected outcome and causal chain

**vs. TinyLLaVA-Video** — On a case where a question requires integrating information from two distinct events (e.g., first person picks up object, then later places it), TinyLLaVA-Video receives only the final answer reward, so its policy may learn to guess based on superficial cues, often missing the intermediate reasoning. Our method segments the video at the event boundaries (e.g., when cross-modal similarity dips between pickup and placement) and rewards alignment of the generated steps with each segment's audio-visual content. Thus, we expect our method to produce a multistep chain that correctly identifies both events, leading to a noticeable accuracy gap (e.g., 10-15% relative improvement) on such multi-event questions, with parity on single-event queries.

**vs. LLaVA-Video** — On a case where the video contains an ambiguous action (e.g., pouring liquid that could be water or juice), LLaVA-Video trained on synthetic data may rely on spurious correlations (e.g., label bias) and misidentify the substance. Our method uses the cross-modal alignment reward to force the reasoning step to be grounded in both visual (liquid color) and audio (pouring sound) cues from that segment. If the audio or visual contains a distinctive signal, our method will produce a more accurate step. We expect our method to outperform LLaVA-Video on questions requiring fine-grained audio-visual disambiguation, with a gap (e.g., 5-8% accuracy) concentrated on such subsets.

**vs. Video-LLaMA** — On a case where the audio and visual streams are weakly correlated (e.g., a conversation scene with little visual change), Video-LLaMA may still attend cross-modally but fail to break the reasoning into steps because it lacks event segmentation. Our method detects boundaries at cross-modal similarity minima; in such scenes, the similarity is flat, so boundaries are sparse, effectively treating the whole video as one segment. Then the dense reward collapses to a single global alignment score, similar to a final reward. Thus, we expect our method to perform similarly to Video-LLaMA on such weakly correlated scenes (small or no gap), but to significantly outperform on highly correlated dynamic scenes where steps align with events.

**vs. CEAR with GT-boundaries** — On a case where cross-modal similarity minima are noisy (e.g., rapid scene changes causing many false positives), the full CEAR may oversegment and degrade step quality, while GT-boundaries use perfect segmentation. We expect full CEAR to approach GT-boundaries when boundary detection F1 score is high, but to have a small gap (e.g., 2-3% accuracy) on videos with low cross-modal correlation. This comparison directly quantifies the cost of unsupervised detection.

### What would falsify this idea

If our method fails to show a concentrated improvement on multi-event questions compared to single-event ones, or if the ablation (uniform segments) performs equally to full CEAR, then the claim that event alignment drives the gains would be falsified. Additionally, if the F1 score for boundary detection is below 0.4 or the correlation with human step correctness is below 0.3, the underlying assumptions are weak and the method requires redesign.

## References

1. TinyLLaVA-Video-R1: Towards Smaller LMMs for Video Reasoning
2. LLaVA-Video: Video Instruction Tuning With Synthetic Data
3. Video-LLaMA: An Instruction-tuned Audio-Visual Language Model for Video Understanding
4. Video-ChatGPT: Towards Detailed Video Understanding via Large Vision and Language Models
5. EVA: Exploring the Limits of Masked Visual Representation Learning at Scale
6. Scaling Instruction-Finetuned Language Models
7. Expanding Language-Image Pretrained Models for General Video Recognition
