# Compositional Task Synthesis for Evaluation-Driven Fine-Tuning of Multi-Reference Video Generation Models

## Motivation

Existing benchmarks like MultiRef-Compass provide standardized evaluation with only 350 curated samples, but models fine-tuned solely on these samples overfit because the benchmark lacks compositional coverage across reference types (audio, video, text). This structural limitation prevents systematic improvement on the benchmark itself, as the fixed samples do not cover the combinatorial space of multi-reference inputs that the evaluation protocol defines.

## Key Insight

The evaluation suite's dimensional structure (audio type, video type, text caption, etc.) defines a combinatorial space of valid tasks; by sampling from this space according to a coverage distribution, we can generate infinite diverse training tasks that match the evaluation protocol without additional human annotation.

## Method

### (A) What it is
**CompTaskGen** is a training data synthesis and fine-tuning framework for multi-reference video generation. It takes as input the benchmark's reference-type taxonomy (e.g., audio: speech/music/ambient; video: human/object/scene; text: instruction type) and outputs a set of diverse multi-reference video generation tasks. The framework then fine-tunes a base model (e.g., a diffusion transformer) on these synthesized tasks using a reward from the evaluation metrics.

### (B) How it works
```python
# Pseudocode: CompTaskGen Training Pipeline

# Step 1: Define probabilistic context-free grammar (PCFG) from taxonomy
# Nonterminals: {Task, AudioRef, VideoRef, TextCap, VideoOutput}
# Productions:
# Task -> AudioRef VideoRef TextCap VideoOutput (prob=1.0)
# AudioRef -> 'speech' | 'music' | 'ambient'   (weights from taxonomy frequency)
# VideoRef -> 'human' | 'object' | 'scene'
# TextCap -> 'action' | 'style' | 'relation'
# VideoOutput -> 'generated video'
# Each production has a probability; we set them using empirical co-occurrence 
# counts computed from the asset pool (VidProM+OpenS2V-5M metadata). For example,
# P(AudioRef='speech' | VideoRef='human') = count(speech & human) / count(human).
# This yields a conditional grammar that reflects real-world joint distributions.

# Step 2: Synthesize tasks by sampling derivations from conditional PCFG
for i in range(num_tasks):
    task = sample_derivation(pcfg, max_depth=4)  # yields atomic types
    # e.g., (audio='speech', video='human', cap='action')
    
    # Step 3: Retrieve concrete assets from a large pool (e.g., VidProM, OpenS2V-5M)
    # We use a stratified sample of 100k videos from VidProM per type to reduce pool size.
    audio_clip = retrieve_asset(pool, type=task['AudioRef'], condition=task['TextCap'])
    video_clip = retrieve_asset(pool, type=task['VideoRef'], condition=task['TextCap'])
    text_caption = generate_caption(task)  # e.g., "a person running with speech"
    
    # Step 4: Use ground-truth generation (or a teacher model) to produce target video
    # We use VideoCrafter2 (public checkpoint) as teacher model.
    target_video = teacher_generator(audio=audio_clip, video=video_clip, text=text_caption)
    
    # Store as training example: (audio, video, text, target_video)
    training_set.append((audio_clip, video_clip, text_caption, target_video))

# Step 5: Fine-tune base model (e.g., VideoCrafter2) using DPO-like reward from evaluation metrics
# Use MultiRef-Compass evaluation protocol to compute rewards
# Reward = weighted sum of Basic Quality, Reference Consistency, AV Consistency, Instruction Following
model = load_pretrained()
for epoch in epochs:
    for batch in training_set:
        # Generate candidate video from current model
        candidate = model.generate(audio=batch.audio, video=batch.video, text=batch.text)
        # Compute reward via MultiRef-Compass metrics (precomputed oracle)
        reward = evaluate_on_benchmark(candidate, batch.target_video, multi_ref_set)
        # Update model with DPO-style loss: maximize log-prob of high-reward samples
        loss = -reward * log_prob(candidate)
        loss.backward()
```

### (C) Why this design
We chose a PCFG over simple template-based generation because templates are rigid and cannot capture the full combinatorial space; PCFG allows controlled probabilistic sampling with guarantee of syntactic validity. We used a large asset pool (from VidProM and OpenS2V-5M) rather than generative model for reference creation because assets from real datasets preserve natural distribution, accepting that pool may have biases in category frequency. Note that we use conditional probabilities from empirical co-occurrence counts instead of uniform probabilities; this ensures that sampled tasks reflect real-world joint distributions, mitigating the risk of unrealistic combinations that degrade performance. We adopted DPO-style reward optimization over supervised fine-tuning on ground-truth because the evaluation metrics directly target the benchmark dimensions we aim to improve; the trade-off is that reward hacking becomes possible, which we mitigate by ensuring diversity via PCFG sampling. We chose conditional probabilities over uniform ones to match real distribution, at the cost of reduced coverage of rare combinations, but this aligns with our goal of preventing overfitting to unrealistic tasks.

### (D) Why it measures what we claim
**Assumption 1: Conditional PCFG sampling covers the true task distribution.** The PCFG-generated tasks' reference-type diversity measures the compositional coverage of the training set because the grammar enumerates legal combinations of atomic types with probabilities matching real co-occurrence; this assumption fails when the asset pool itself has selection bias (e.g., missing certain rare combinations entirely), in which case coverage under the grammar may still be incomplete, and the metric reflects only the pool's coverage. **Empirical test:** Compute KL divergence between the type co-occurrence distribution of sampled tasks and that of held-out real tasks from MultiRef-Compass. If divergence exceeds a threshold (e.g., 0.1 nats), augment pool or reweight.

**Assumption 2: Reward from evaluation metrics measures alignment with benchmark quality dimensions.** The reward from evaluation metrics measures alignment with benchmark quality dimensions because the metrics are designed to correlate with human judgment; this assumption fails when metrics are imperfect proxies (e.g., VideoScore may miss temporal artifacts), in which case the reward reflects only the metric's biases. **Empirical test:** Compute correlation between reward and human ratings on a held-out set; if correlation < 0.7, switch to a different metric set.

**Assumption 3: Retrieval relevance is sufficient for task validity.** The sampling step from the asset pool measures retrieval relevance because we condition on type and caption; this assumption fails when the pool lacks appropriate assets for a given type (e.g., rare 'ambient' audio), in which case the task becomes invalid and the training degrades. **Empirical test:** For each type combination, verify that at least 5 distinct assets exist; if not, drop that combination from the grammar.

## Contribution

(1) A compositional task synthesis method that leverages evaluation benchmark taxonomy to generate diverse training tasks without human annotation, using a probabilistic context-free grammar to ensure coverage. (2) A closed-loop fine-tuning framework that iteratively improves model performance on the benchmark by expanding coverage, using evaluation metrics as rewards. (3) Empirical demonstration that models fine-tuned with CompTaskGen outperform those fine-tuned on fixed samples on MultiRef-Compass, reducing overfitting.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|------------------------|
| Dataset | MultiRef-Compass | Target benchmark with multi-reference tasks. |
| Primary metric | Reference Consistency | Directly measures multi-reference fidelity. |
| Secondary metric | Effective coverage (fraction of metric variance explained by grammar entropy) | Measures how well PCFG diversity translates to performance. |
| Baseline 1 | VideoCrafter (text-to-video) | Tests text-only conditioning baseline. |
| Baseline 2 | AnimateDiff (image-to-video) | Tests image-only conditioning baseline. |
| Baseline 3 | T2AV baseline (audio-visual) | Tests audio-visual generation baseline. |
| Ablation-of-ours | CompTaskGen w/ uniform PCFG | Isolates effect of conditional vs. uniform PCFG. |
| Ablation 2 | Template-based task synthesis | Tests simple template vs. PCFG diversity. |

### Why this setup validates the claim

The experimental design forms a falsifiable test by pitting CompTaskGen against baselines that each lack one or more reference modalities. The dataset (MultiRef-Compass) provides tasks that require joint conditioning on audio, video, and text, so success demands multi-reference reasoning. The primary metric (Reference Consistency) directly quantifies alignment with all references—if our method truly improves compositionality, it must excel here. The baselines isolate the contribution of each modality: VideoCrafter and AnimateDiff handle only text or image, while the T2AV baseline ignores text; all fail to jointly integrate references. The ablation with uniform PCFG tests whether conditional probabilities (which reflect real co-occurrence) are crucial: if uniform works better, our repair is unnecessary. The template-based ablation tests whether PCFG's combinatorial coverage adds value beyond simple templates. The secondary metric (effective coverage) quantifies how much of the metric variance is explained by grammar entropy, directly testing the claim that PCFG diversity drives improvement. A clear outcome pattern—superior performance on multi-reference tasks and high effective coverage—would support the claim, while uniform performance across subsets or low effective coverage would refute it.

### Expected outcome and causal chain

**vs. VideoCrafter** — On a task with audio (music), video (object), and text (style), VideoCrafter ignores audio and video, generating a generic video that mismatches the music or object. Our method synthesizes a task with all references via conditional PCFG sampling, retrieves matching assets, and fine-tunes with reward from Reference Consistency metric, so it learns to align with all inputs. We expect a noticeable gap on tasks combining audio, video, and text, but parity on text-only tasks.

**vs. AnimateDiff** — On a task with two video references (human and scene), AnimateDiff can only condition on one image, causing poor composition (e.g., wrong background). Our method handles multiple video references by retrieving both assets and training with multi-consistency reward. We expect a substantial gain on tasks with multiple video references, but smaller improvement on single-video tasks.

**vs. T2AV baseline** — On a task with ambiguous audio and video but clear text instruction (e.g., "slow motion"), the T2AV baseline may ignore text, failing to follow instruction. Our method uses DPO reward on Instruction Following dimension, so it prioritizes text alignment. We expect a clear improvement on tasks where text disambiguates.

**vs. CompTaskGen w/ uniform PCFG** — On tasks with realistic co-occurrences (e.g., speech + human), conditional grammar will produce more valid tasks, leading to higher Reference Consistency. On rare combinations (e.g., ambient + object), uniform PCFG may sample tasks that are invalid due to missing assets, degrading performance. We expect conditional to outperform overall, but uniform may cover more rare combinations if assets exist.

**vs. Template-based task synthesis** — Template-based synthesis uses fixed combinations (e.g., always speech+human+action) and thus lacks coverage of the full combinatorial space. Our PCFG generates diverse combinations, leading to better generalization to unseen task types. We expect a clear gap on tasks that deviate from the template templates.

### What would falsify this idea

If CompTaskGen's Reference Consistency scores are not significantly higher than baselines on tasks that require all three reference types (audio+video+text), or if the uniform PCFG ablation matches or outperforms the conditional version, or if effective coverage (fraction of metric variance explained by grammar entropy) is low (<0.3), then the central claim that conditional PCFG-based multi-reference synthesis drives improvement would be falsified.

## References

1. MultiRef-Compass: Towards Comprehensive Evaluation of Multi-Reference-to-Audio-Video Generation
2. OpenS2V-Nexus: A Detailed Benchmark and Million-Scale Dataset for Subject-to-Video Generation
3. T2AV-Compass: Towards Unified Evaluation for Text-to-Audio-Video Generation
4. VidProM: A Million-scale Real Prompt-Gallery Dataset for Text-to-Video Diffusion Models
5. VideoScore: Building Automatic Metrics to Simulate Fine-grained Human Feedback for Video Generation
6. VBench: Comprehensive Benchmark Suite for Video Generative Models
7. VIEScore: Towards Explainable Metrics for Conditional Image Synthesis Evaluation
8. Diffusion Model Alignment Using Direct Preference Optimization
