# Prosody-Aware Inference-Time Knowledge Distillation via Jointly Learned Multi-Modal Tokenization for Speech

## Motivation

Current inference-time knowledge distillation methods for speech, such as ConvFill, rely on text-only tokenization, which discards prosodic, emotional, and stylistic information. This structural gap prevents the talker model from generating speech with natural expressiveness, as it inherits a representation that ignores multi-modal aspects of conversation. The root cause is that the tokenizer and knowledge transfer framework are designed for text, failing to capture paralinguistic features essential for engaging voice agents.

## Key Insight

Jointly learning a multi-modal tokenizer that maps speech into discrete tokens representing both linguistic and paralinguistic dimensions ensures that the talker's representation space is inherently aligned with the reasoner's knowledge, enabling direct infilling of prosodic cues without separate alignment modules.

## Method

### (A) What it is
**Multi-Modal Conversational Infill (MM-ConvFill)** is an inference-time knowledge distillation framework that uses a jointly learned multi-modal tokenizer to represent speech as discrete tokens encoding phonetic content, prosody, emotion, and speaking style. It takes audio history and user text query as input and generates a speech response with appropriate expressiveness.

### (B) How it works
```python
# MM-ConvFill Inference Procedure
# Input: audio history H (raw waveform), user query Q (text)
# Output: synthesized speech R (waveform)

# Step 1: Multi-modal tokenization of audio history
# Use a VQ-VAE with multiple codebooks: one for phonetic (HuBERT features), one for prosody (pitch, energy, duration), one for emotion (7-class 'neutral', 'happy', 'sad', 'angry', 'surprised', 'fearful', 'disgusted'), one for style (formal/casual)
# Jointly trained to reconstruct audio with a multi-head quantization loss + disentanglement loss (mutual information minimization via CLUB estimator, weight λ_dis=0.1) to encourage orthogonality
# Codebook sizes: 512 each (default), also evaluated at 256 and 1024 in ablation
history_tokens = multi_modal_tokenizer.encode(H)  # shape (seq_len, 4) each token is a tuple of 4 discrete indices

# Step 2: Tokenize user query with standard BPE (tokenizer from GPT-2, vocab size 50257)
query_tokens = bpe_tokenizer.encode(Q)

# Step 3: Talker model initialization
# Talker is a small transformer (1.5B parameters, 24 layers, hidden size 2048, 16 attention heads) with two cross-attention modules: one for self-generated tokens, one for reasoner knowledge
# Context = query_tokens + [SEP] + history_tokens
# Begin generating response token sequence autoregressively
response_tokens = []

# Step 4: Parallel reasoner processing
# Reasoner (GPT-4, accessed via API) processes Q and text-transcribed history (from ASR) + prosodic annotations (e.g., predicted emotion from query sentiment using a pre-trained emotion classifier)
# Reasoner outputs knowledge tokens: up to 16 tokens, each is a tuple (text, prosody_suggestion)
# Example: ("I agree", "excited")
knowledge_tokens = reasoner.generate(Q, history_text, emotion_context)  # shape (16, 2)

# Step 5: Inference-time knowledge integration
for t in range(max_response_length):  # default max length = 128 tokens
    # Talker forward pass: attend to previous self-tokens and to knowledge_tokens
    # Two attention paths with learnable gating weight alpha (default alpha=0.7 fixed per timestep, also evaluated with per-timestep learned alpha)
    logits = talker(response_tokens, context, knowledge_tokens, alpha=0.7)
    # Sample next token: a 4-tuple (phonetic, prosody, emotion, style)
    next_token = sample(logits)  # using temperature sampling, temperature=0.9
    response_tokens.append(next_token)

# Step 6: Decode response_tokens to waveform using tokenizer decoder (VQ-VAE decoder, 12-layer transformer)
R = multi_modal_tokenizer.decode(response_tokens)

# Training procedure:
# 1. Pre-train VQ-VAE on large speech corpus (e.g., LibriTTS + Emotional Speech Dataset) with reconstruction loss + codebook commitment loss + λ_dis * CLUB mutual information loss between codebook assignments. Train for 100k steps, batch size 64, learning rate 1e-4, Adam optimizer.
# 2. Freeze VQ-VAE. Train talker on conversational speech dataset (Expressiveness subset) using teacher forcing with knowledge tokens from a fixed pre-trained reasoner. Train for 50k steps, batch size 32, learning rate 5e-5, Adam optimizer. Talker parameters: 350M (due to resource constraints, scaled down from 1.5B; original method uses 1.5B but we adopt 350M for feasibility).

# Hyperparameters: VQ-VAE latent dim=512, codebook size=512 each (ablate 256, 1024), talker layers=12 (scaled down), alpha=0.7 (fixed, ablate with per-timestep learned alpha), max_knowledge_tokens=16, temperature=0.9.
```

### (C) Why this design
We chose a jointly learned multi-modal VQ-VAE over separate modules for prosody and emotion because joint training captures correlations (e.g., sad speech tends to have low pitch), enabling more compact token representations; the cost is increased training complexity and need for large multi-modal datasets. We chose discrete tokens over continuous representations to align with the autoregressive transformer talker, allowing direct use of attention mechanisms without additional projection layers. The trade-off is quantization loss, but it facilitates efficient generation. We designed the talker with two separate cross-attention paths for self-generated and reasoner tokens, rather than concatenating them, to allow a learnable gating weight that controls the influence of knowledge; this introduces an extra hyperparameter but provides flexibility. The reasoner generates both text and prosody suggestions because the reasoner's broader context can infer appropriate emotional tone that the talker cannot derive locally; the cost is increased latency and API calls. These design choices prioritize expressiveness and controllability over simplicity.

### (D) Why it measures what we claim
The multi-modal tokenizer's four codebooks (phonetic, prosody, emotion, style) operationalize the concept of "prosodic and emotional awareness" because each codebook isolates a specific paralinguistic dimension. The assumption that these dimensions are orthogonal is a simplification; in practice, they interact (e.g., emotion correlates with prosody), so the tokenizer may conflate them if not explicitly separated. To partially address this, we add a disentanglement loss (mutual information minimization) that encourages codebook independence. When this assumption fails, the generated emotion may be coupled with incorrect prosody. The talker's cross-attention to reasoner knowledge tokens operationalizes "inference-time knowledge transfer" because it allows reasoner-provided cues to infill the generation. The assumption is that reasoner's prosody suggestions are compatible with the tokenizer's codebook; this fails if the reasoner suggests an emotion not covered by the codebook (e.g., "sarcastic" not in the 7 emotions), leading to a default mapping (we map missing emotions to 'neutral'). The gating alpha measures the "knowledge influence" directly, but the assumption that alpha=0.7 is optimal across all timesteps is a simplification; we ablate with per-timestep learned alpha to test this. The framework's output speech naturalness is the ultimate measure, but the above components are designed to produce it.

## Contribution

(1) A jointly learned multi-modal speech tokenizer that encodes phonetic, prosodic, emotional, and stylistic information into discrete tokens via a VQ-VAE with multiple codebooks. (2) The MM-ConvFill framework that integrates this tokenizer with inference-time knowledge distillation, enabling a small talker model to generate prosody-aware speech responses conditioned on a frontier reasoner's paralinguistic suggestions. (3) A synthetic dataset of multi-modal conversation examples annotated with emotion and speaking style for training and evaluation.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | Expressiveness subset of LibriTTS (2000 utterances with synthetic prosody labels from an emotion classifier, covering 7 emotions) | Covers diverse expressive scenarios while feasible size; synthetic labels enable controlled experiments |
| Primary metric | Naturalness Mean Opinion Score (MOS) with 5-point scale, 100 raters per sample | Direct measure of speech quality |
| Secondary metrics | Codebook mutual information (MI) between codebook indices; MOS drop when removing one emotion codebook index | Test orthogonality assumption and robustness to missing prosody |
| Baseline 1 | Talker only (no reasoner knowledge) | Tests impact of knowledge injection |
| Baseline 2 | Reasoner + standard TTS (Tacotron2 + WaveGlow) | Tests need for prosodic control |
| Ablation 1 | MM-ConvFill without gating (concat attention) | Tests importance of adaptive knowledge weighting |
| Ablation 2 | MM-ConvFill with codebook size=256 | Tests effect of quantization granularity |
| Ablation 3 | MM-ConvFill with codebook size=1024 | Tests effect of quantization granularity |
| Ablation 4 | MM-ConvFill with per-timestep learned alpha (instead of fixed 0.7) | Tests need for dynamic weighting |
| Failure mode test | Artificially remove one emotion codebook index (e.g., 'happy') and measure MOS on utterances requiring that emotion | Tests robustness to missing prosodic cues |

### Why this setup validates the claim
The chosen dataset provides a wide range of expressive contexts (e.g., sarcasm, excitement) necessary to evaluate prosodic and emotional awareness. The primary metric, MOS, captures the overall naturalness that the method aims to improve. Baseline 1 (talker only) isolates the effect of inference-time knowledge; if our method outperforms it, knowledge transfer is beneficial. Baseline 2 (reasoner + standard TTS) tests whether the multi-modal tokenizer and discrete knowledge tokens are essential for controllability. Ablation 1 (no gating) tests whether the learnable gating mechanism provides additional flexibility. Ablations 2-4 test sensitivity to key hyperparameters (codebook size, alpha). The failure mode test directly probes the assumption that the codebook covers all needed prosodic variations; if MOS drops significantly, the method's reliance on fixed discrete categories is a weakness. Together, these conditions form a falsifiable test: if our method fails to show a significant MOS gain over baselines on expressive utterances, or if the failure mode test shows no drop (suggesting codebook is not used), the central claim is unsupported.

### Expected outcome and causal chain

**vs. Talker only** — On a case where the user query is sarcastic (e.g., "Oh, great, another meeting"), the talker-only model produces flat, neutral prosody because it lacks higher-level context to infer the intended emotion. Our method uses the reasoner to recognize sarcasm from the query and history, generating a prosody suggestion (e.g., "pitch rise then fall") that the talker infills via cross-attention. We expect a noticeable MOS gap on sarcastic turns (e.g., 0.5 points higher) but parity on simple factual queries where prosody is default.

**vs. Reasoner + standard TTS** — On a case requiring subtle emotional mixing (e.g., "I'm happy, but I'm also nervous"), the reasoner alone generates text, but the standard TTS cannot produce the appropriate blend of excitement and tremor. Our method encodes the reasoner's joint text-prosody suggestion directly into discrete tokens, allowing the talker to generate a mixed emotional output. We expect a larger MOS improvement (e.g., 0.7 points) on complex emotional utterances, but a smaller gap on straightforward expressions.

**vs. Ablation (no gating)** — On a case where the reasoner's suggestion is partially irrelevant (e.g., the history indicates calm, but reasoner mispredicts anger), the no-gating model overrides the talker's own style, producing unnatural speech. Our gated model self-adapts via alpha=0.7, blending knowledge with self-generated tokens for a more natural outcome. We expect the gated version to have higher MOS on cases with conflicting cues, but similar performance when knowledge is perfectly aligned.

**Failure mode test** — When we remove a codebook index (e.g., 'happy'), utterances that require happy prosody should see a significant MOS drop (e.g., 0.8 points) because the talker cannot generate the appropriate emotion. If no drop occurs, the codebook is not being effectively used, falsifying the claim.

### What would falsify this idea
If our method shows uniform MOS gains across all utterance types (not concentrated on expressive ones), or if the no-gating ablation performs equally well, or if the failure mode test shows no MOS drop when an emotion codebook index is removed, the claimed benefits of inference-time knowledge gating and multi-modal tokenization are not supported.

## References

1. Thinking While Speaking: Inference-Time Knowledge Transfer for Responsive and Intelligent Conversational Voice Agents
2. Agents Thinking Fast and Slow: A Talker-Reasoner Architecture
3. Reasoning with Language Model is Planning with World Model
4. DayDreamer: World Models for Physical Robot Learning
5. Inner Monologue: Embodied Reasoning through Planning with Language Models
6. ProgPrompt: Generating Situated Robot Task Plans using Large Language Models
7. Language Models Are Greedy Reasoners: A Systematic Formal Analysis of Chain-of-Thought
