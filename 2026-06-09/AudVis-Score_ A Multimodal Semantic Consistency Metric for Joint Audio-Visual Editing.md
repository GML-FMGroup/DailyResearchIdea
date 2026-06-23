# AudVis-Score: A Multimodal Semantic Consistency Metric for Joint Audio-Visual Editing

## Motivation

Existing evaluation metrics for joint audio-visual editing, such as those in JAVEditBench (JAVEDIT), rely on predefined rubric decompositions that cannot capture the semantic fidelity of open-ended editing instructions (e.g., "change the background music to a sad piano"). This limitation stems from the lack of a cross-modal semantic bridge that can compare the edited video's audio and visual content against the instruction without manual decomposition. Prior work like JAVEDIT and AV-Edit use task-specific metrics that do not generalize to novel instructions, structurally failing because they treat evaluation as a classification over atomic operations rather than a semantic consistency check.

## Key Insight

By jointly leveraging pretrained audio and video captioners to convert both modalities into semantic text, and using an LLM to compare the resulting caption deltas against the instruction, we obtain a metric that inherently captures multimodal instruction-following without requiring rubric design.

## Method

### (A) What It Is
AudVis-Score is a metric that takes an original video, an edited video, and a natural-language editing instruction, and returns a score in [0,1] indicating how faithfully the edit follows the instruction. It uses modality-specific captioners and an LLM judge.

### (B) How It Works
```
Input: original_video, edited_video, instruction

Step 1: Extract audio tracks from both videos.
Step 2: Generate audio captions:
   audio_caption_orig = AudioCaptioner(audio_orig)
   audio_caption_edit = AudioCaptioner(audio_edit)
Step 3: Generate video captions (visual content only, no audio):
   video_caption_orig = VideoCaptioner(original_video)
   video_caption_edit = VideoCaptioner(edited_video)
Step 4: For each modality m in {audio, video}:
   prompt = f"Original {m} caption: {caption_orig_m}\nEdited {m} caption: {caption_edit_m}\nInstruction: {instruction}\nDescribe the change in {m} that occurred, then rate how well the change matches the instruction (0: no match, 1: perfect)."
   response = LLM(prompt)
   score_m = extract_score(response)
Step 5: AudVis-Score = (score_audio + score_video) / 2
```
Hyperparameters: AudioCaptioner: CLAP-based (e.g., LAION-CLAP); VideoCaptioner: Video-LLaMA; LLM: GPT-4 with temperature=0.

### (C) Why This Design
We chose to separate audio and video captioning rather than using a unified multimodal captioner because audio and visual changes can be independent (e.g., replacing background music does not alter visual content); independent captioners allow the LLM to assess each modality's fidelity without interference. We selected an LLM-based scorer over a trained regression model because the LLM can handle the infinite space of open-ended instructions without requiring task-specific training data, accepting the trade-off of higher computational cost (approx. $0.01 per evaluation). We opted to average the modality scores rather than multiply them to avoid penalizing edits that intentionally change only one modality (e.g., a purely audio instruction), which would be incorrectly scored near zero under a product. This design choice prioritizes fairness across instruction types, albeit with the consequence that cross-modal interactions (e.g., 'make the dog bark when it appears') are evaluated independently per modality and may miss fine-grained alignment.

### (D) Why It Measures What We Claim
The audio captioning step converts low-level audio signals into semantic text, enabling the LLM to measure alignment with the instruction; this assumes that the captioner captures all semantically relevant audio changes (e.g., timbre, tempo, event presence). The video captioning step does the same for visual changes, assuming the captioner describes key visual elements. The LLM's judgment compares the semantic deltas (difference between original and edited captions) to the instruction; this assumes that the LLM can accurately infer whether the described changes match the instruction. The assumption fails when captions omit subtle yet perceptually important changes (e.g., a slight pitch shift), in which case the score reflects only the coarse semantic shift the captioner captured. The final average score combines modality-specific alignment; this assumes that the instruction modifies each modality independently. The assumption fails for instructions requiring cross-modal coordination (e.g., 'synchronize the explosion sound with the fireball'), where independent scores may not detect misalignment. In such cases, the metric reflects aggregate per-modality fidelity rather than true cross-modal synchronization.

## Contribution

(1) AudVis-Score, a novel multimodal semantic consistency metric for joint audio-visual editing that uses pretrained captioners and an LLM judge to assess instruction-following without manual rubric design. (2) A design principle that separating modality-specific captioning and using LLM-based comparison yields a metric robust to open-ended instructions and modality-specific edits. (3) An open-source evaluation toolkit implementing AudVis-Score for reproducible benchmarking.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|-----------------------|
| Dataset | JAVEditBench | Diverse audio-video edits with human ratings |
| Primary metric | Spearman correlation with human | Measures rank alignment |
| Baseline 1 | Video-only (CLIP score) | Tests need for audio channel |
| Baseline 2 | Audio-only (CLAP score) | Tests need for video channel |
| Baseline 3 | Direct caption cosine similarity | Tests LLM judge importance |
| Ablation-of-ours | Remove LLM judge, use embedding sim | Tests LLM reasoning value |

### Why this setup validates the claim
This experimental design tests the central claim that modality-specific captioning followed by LLM reasoning yields a better measure of instruction fidelity than simpler alternatives. JAVEditBench provides ground-truth human ratings across diverse edits. Comparing against unimodal baselines (video-only, audio-only) isolates the contribution of each modality; if our metric outperforms both, it confirms both audio and video are necessary. The direct caption similarity ablation tests whether the LLM judge adds value beyond naive embedding comparison, isolating the reasoning component. The Spearman correlation metric directly tests whether our scores rank edits in the same order as humans, which is the ultimate validation of a perceptual metric. If our method fails to outperform these baselines on the full dataset or on subsets predicted to favor our design, the central claim is falsified.

### Expected outcome and causal chain

**vs. Video-only (CLIP score)** — On a case where the edit only changes audio (e.g., "replace background music with rain"), the video-only baseline gives a high score because visual content is unchanged, falsely indicating perfect fidelity. Our method separately captions the audio; the LLM sees that audio changed from "upbeat music" to "rain sounds" and matches the instruction, yielding a correct low-to-moderate score. We expect a noticeable gap on audio-only edits (our method correlates 0.7 vs. baseline 0.2) but parity on video-only edits.

**vs. Audio-only (CLAP score)** — On a case where the edit only changes video (e.g., "add a fireball at second 3"), the audio-only baseline remains unaffected, giving a perfect score despite visual mismatch. Our method captions the video change and the LLM detects the missed match, assigning a low score. We expect a similar asymmetric performance: our method outperforms on video-only edits (correlation ~0.7 vs. ~0.2), while matched on audio-only edits.

**vs. Direct caption cosine similarity** — On a case where the instruction is "make the balloon pop loudly" and the edit changes the audio to a loud pop but also slightly changes the visual (e.g., balloon disappears), the direct similarity between original and edited audio captions may be high (both describe "pop") and video captions may differ, but the embedding similarity might average to a moderate score. However, the instruction demands a loud pop; if the audio is actually a quiet pop, the LLM can reason that "loud" is missing and assign a lower score. We expect our method to achieve higher correlation on edits where instruction requires fine-grained semantic alignment (e.g., adjectives, direction of change), with a gap of ~0.1 on the full dataset but larger on subsets of such instructions.

### What would falsify this idea
If our metric's Spearman correlation is not significantly higher than the best baseline (especially the direct caption similarity ablation) on the full JAVEditBench, or if the gap is uniform across all edit types rather than concentrated on cross-modal or fine-grained instructions, then the claim that modality separation and LLM reasoning are beneficial is falsified.

## References

1. JAVEDIT: Joint Audio-Visual Instruction-Guided Video Editing with Agentic Data Curation
2. MMAE: A Massive Multitask Audio Editing Benchmark
3. AV-Edit: Multimodal Generative Sound Effect Editing via Audio-Visual Semantic Joint Control
4. OpenVE-3M: A Large-Scale High-Quality Dataset for Instruction-Guided Video Editing
5. Audio-sync Video Instance Editing with Granularity-Aware Mask Refiner
6. Scaling Instruction-Based Video Editing with a High-Quality Synthetic Dataset
7. OminiControl: Minimal and Universal Control for Diffusion Transformer
8. CogVideoX: Text-to-Video Diffusion Models with An Expert Transformer
