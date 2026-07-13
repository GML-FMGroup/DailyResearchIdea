# AutoBench: Automated Generation of Long-Video Temporal Reasoning Benchmarks via Template-Guided Segmentation

## Motivation

Existing video understanding benchmarks, as audited by Video-Oasis, contain non-video-native shortcuts that allow models to perform well without true temporal reasoning, yet they remain the standard for evaluation. Moreover, these benchmarks are manually curated, limiting their coverage of long-video temporal dependencies across diverse domains and making them difficult to scale. The root cause is the lack of a systematic method to generate tasks that inherently require understanding of causally connected events over extended time spans, leaving temporal reasoning evaluation incomplete.

## Key Insight

By decomposing long videos into meaningful event segments and composing them via causal templates, an exponential number of unique temporal reasoning tasks can be generated that require true temporal ordering, causation, and continuity, ensuring task validity by construction.

## Method

**AutoBench: Automated Benchmark Generation**

**(A) What it is:** AutoBench is a fully automated pipeline that takes a collection of long videos and outputs a benchmark of temporal reasoning multiple-choice questions (MCQs). Input: raw videos; output: a set of questions with answer options, where correctness requires understanding temporal dependencies across at least two distinct events.

**(B) How it works (Pseudocode):**
```python
# Phase 1: Segmentation
def segment_video(video):
    # Use a pretrained event boundary detection model (TransNetV2, pretrained on Kinetics-400)
    # Assumption: TransNetV2 identifies semantically meaningful atomic events.
    # This assumption fails for continuous activities with gradual transitions, leading to over- or under-segmentation.
    # We calibrate the segmentation by computing average segment count on a development set of 100 videos
    # and adjusting threshold if more than 20% of videos have <3 segments.
    boundaries = detect_boundaries(video, threshold=0.5)  # hyperparameter: threshold for transition probability
    segments = clip_video(video, boundaries)
    # Filter out segments shorter than 2 seconds (min_event_duration)
    valid_segments = [s for s in segments if len(s) >= 2]
    return valid_segments

# Phase 2: Segment Annotation
def annotate_segments(segments):
    # Use Video-LLaVA (7B parameters) to generate natural language description for each segment.
    # Captioning prompt: "Describe the main event in this video clip in one sentence."
    captions = [generate_caption(seg) for seg in segments]
    return captions

# Phase 3: Template Selection and Question Generation
def generate_questions(segments, captions):
    questions = []
    for i in range(len(segments)):
        for j in range(i+1, len(segments)):
            # Temporal templates with slots for segment indices
            template_pool = [
                f"After {captions[i]}, what happens next?",
                f"What caused {captions[j]}?",
                f"Which event occurs earlier: {captions[i]} or {captions[j]}?",
                f"Order the following events: (1) {captions[i]}, (2) {captions[j]}."
            ]
            for template in template_pool:
                q = fill_template(template, i, j, captions)
                correct_answer = compute_answer(i, j, template)  # based on temporal order
                # Generate distractors from other segment captions within the same video
                distractor_pool = [captions[k] for k in range(len(segments)) if k != i and k != j]
                distractors = random_sample(distractor_pool, num=3)  # hyperparameter: number of distractors (3)
                questions.append((q, correct_answer, distractors))
    return questions

# Phase 4: Verification (filter non-video-native)
def verify_question(q, correct, distractors, video, segments):
    # Frame-shuffle test: shuffle all frames in the video and test if the question is still answerable.
    # If model accuracy on shuffled video > 30% (chance=25%), discard the question.
    shuffled_video = shuffle_frames(video)
    if compute_accuracy(shuffled_video, q) > 0.3:
        return False
    # Single-segment solvability test: check if the question can be answered by viewing only the relevant segment(s).
    # Assumption: a question solvable from one segment does not test temporal reasoning.
    # Failure mode: a single segment may contain implicit temporal cues (e.g., clock, elapsed time indicator).
    # We further filter using a temporal attribute detector (pretrained CLIP-based classifier) that identifies segments with explicit time cues.
    # If the segment contains a clock, timer, or other temporal indicators, the question is also removed.
    for segment_idx in relevant_segments(q):
        if answerable_from_single_segment(segment_idx, q):  # using same model on just that segment
            return False
        if has_temporal_indicator(segment_idx, segments, threshold=0.8):  # probability > 0.8
            return False
    return True
```

**(C) Why this design:**
We chose a modular pipeline over end-to-end generation because each stage can be independently validated and improved. The segmentation threshold (0.5) was set to balance between over-segmentation (breaking meaningful events) and under-segmentation (missing boundaries); we accept the cost that some event boundaries may be missed, leading to questions that span multiple unrelated events, which can be filtered later. Using a pretrained captioner (Video-LLaVA) instead of a simpler keyword extractor ensures rich semantic descriptions but introduces potential captioning errors; we mitigate this by using multiple templates that are robust to synonyms (e.g., different phrasing for the same event). The verification step explicitly rejects questions solvable from a single segment, directly addressing the shortcut problem identified in Video-Oasis. We chose rule-based template filling over learned question generation to guarantee that every question requires temporal reasoning between two distinct segments; the trade-off is limited linguistic diversity, but diversity can be increased by paraphrasing templates. The number of distractors (3) is a standard choice to reduce guess probability while keeping the cognitive load manageable.

**(D) Why it measures what we claim:**
The computational quantities in (B) measure temporal reasoning coverage and task validity. The segmentation operation extracts temporal intervals that correspond to distinct events; the assumption that event boundaries are detectable by the neural model is that local changes in visual features indicate event transitions; this assumption fails for continuous activities (e.g., a person walking) where boundaries are ambiguous, in which case segments may not represent atomic events and questions may become arbitrary. The template-driven question generation ensures that every question explicitly references a temporal relation (e.g., "after", "cause", "order"); the assumption is that answering these questions requires understanding the causal or chronological relation between the two events; this fails when the relation can be inferred from frame-level cues (e.g., visual similarity) rather than sequence-level reasoning, but the verification filter reduces such cases. The verification step checks answerability from a single segment; the assumption is that if a question can be answered from one segment, it does not require temporal reasoning; this fails when the single segment contains implicit temporal cues (e.g., a time-lapsed clock), in which case the question might still test temporal reasoning within a single segment, and we accept this limited false positive. The additional temporal indicator filter and frame-shuffle test further mitigate shortcuts. Overall, the pipeline ensures that generated questions are video-native (require both visual and temporal processing) by construction, because they involve multiple segments and are verified for non-solvability from isolated views and shuffled frames.

**Resource estimate:** Running the full pipeline on a single NVIDIA A100 GPU takes approximately 48 hours for segmenting and captioning 100 hours of video (∼200 videos of 30 min each). Segmentation runs at ∼2 fps, captioning at ∼0.5 fps per segment.

## Contribution

(1) A fully automated pipeline for generating long-video temporal reasoning benchmarks from raw videos, combining event segmentation, captioning, and template-based question generation with a shortcut verification step. (2) An analysis demonstrating that generated benchmarks avoid non-video-native shortcuts and achieve higher temporal dependency coverage than existing datasets, as measured by the proportion of questions requiring multi-segment reasoning. (3) A publicly released benchmark of 10,000 long-video temporal reasoning questions across 5 domains (surveillance, cooking, sports, news, home activities).

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
|------|--------|-----------|
| Dataset | Long videos from ActivityNet (version 1.3, test set, 100 videos) | Diverse events with temporal structure; each video 5-10 min. |
| Primary metric | Accuracy on generated questions | Direct measure of question solvability. |
| Baseline 1 | Random guessing | Chance-level performance lower bound (25% for 4-option MCQ). |
| Baseline 2 | Video-LLaVA (zero-shot) | State-of-the-art video QA model (7B parameters). |
| Baseline 3 | Text-only (captions, no video) | Tests necessity of visual input; uses same captions as AutoBench. |
| Ablation of ours | AutoBench w/o verification | Isolates verification filter contribution. |
| Human evaluation | 5 annotators on 200-question subset | Validates temporal reasoning requirement; inter-annotator agreement (Fleiss' kappa) reported. |

### Why this setup validates the claim

The ActivityNet videos provide a rich source of untrimmed, temporally structured events, allowing AutoBench to generate diverse temporal reasoning questions. Accuracy on the generated benchmark directly reflects question solvability: random guessing establishes a floor; Video-LLaVA tests whether current models can answer without shortcuts; text-only (captions from the same segments) checks if visual information is essential. The ablation (without verification) isolates the contribution of the verification step, which ensures that questions require multiple segments. The human evaluation directly assesses whether questions indeed require multi-event temporal reasoning, with inter-annotator agreement measuring consistency. Together, these components form a falsifiable test: if the generated questions truly require temporal reasoning, then (1) Video-LLaVA should significantly outperform random guessing but not reach perfect accuracy, (2) text-only should be near chance, (3) the full method should yield more challenging (lower accuracy) but more valid questions than the ablation, and (4) human annotators should agree that questions require at least two events.

### Expected outcome and causal chain

**vs. Random guessing** — On a question asking "After event A, what happens next?", random guessing picks one of four options uniformly, yielding 25% accuracy because it ignores temporal structure. Our method requires reasoning over the video's sequence, so we expect accuracy for capable models like Video-LLaVA to be well above chance (e.g., 60-70%) on such questions, clearly distinguishing from random.

**vs. Video-LLaVA** — On a question about causal dependency (e.g., "What caused the explosion?"), Video-LLaVA may exploit visual shortcuts like object co-occurrence in a single frame, leading to plausible but wrong answers. Our verification step filters questions solvable from one segment, forcing the model to integrate information across multiple events. Thus we expect a notable performance gap (≈15-20 points) on questions involving non-local causality compared to subsets where shortcuts exist.

**vs. Text-only** — On a question requiring visual details (e.g., "Which event occurred first: the person slipping or the glass breaking?"), text-only receives only coarse captions (e.g., "a person slips", "glass breaks") but lacks the visual evidence of the exact moment. Without seeing the frames, it cannot determine order reliably and drops to near-chance accuracy (≈25%). Our full method uses video, so we expect much higher accuracy (e.g., 60-70%) on such questions.

**vs. Ablation (w/o verification)** — On a question that can be answered from a single segment (e.g., "What happens after the person enters the room?" if the room's state is visible in the first segment), the ablation incorrectly includes it as a temporal reasoning question, yielding high accuracy from simple visual cues. Our verification rejects such questions, ensuring the benchmark targets genuine multi-event reasoning. Consequently, the full benchmark shows lower average accuracy but higher correlation with temporal reasoning ability.

### What would falsify this idea
If Video-LLaVA achieves near-perfect accuracy (e.g., >90%) on the generated benchmark, or if the text-only model performs similarly to Video-LLaVA, then the questions likely do not require both visual and temporal reasoning, invalidating the central claim. Additionally, if human annotators show low agreement (Fleiss' kappa <0.4) on whether questions require multi-event reasoning, the benchmark's validity is questionable.

## References

1. Video-Oasis: Rethinking Evaluation of Video Understanding
