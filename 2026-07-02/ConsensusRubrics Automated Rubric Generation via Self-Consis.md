# ConsensusRubrics: Automated Rubric Generation via Self-Consistency in Vision-Language Models

## Motivation

Existing rubric-based evaluation frameworks such as PerceptionRubrics achieve high human alignment but require labor-intensive manual construction of instance-specific golden captions and essential fact lists. This cost prevents scaling to diverse domains and tasks. The root cause is the assumption that human annotation is necessary to define correctness criteria for each instance. We exploit a structural property of vision-language models (VLMs): factual assertions that are essential for correctness tend to be consistently generated across multiple chain-of-thought reasoning samples, providing a weak but automatable supervisory signal.

## Key Insight

Self-consistency of factual assertions across diverse reasoning paths approximates essentialness because essential facts are necessary for correct reasoning and thus appear invariantly regardless of the particular reasoning trajectory, making consistency a reliable proxy for human-defined rubrics.

## Method

### ConsensusRubrics: Automated Rubric Generation via Self-Consistency in Vision-Language Models

(A) **What it is**: ConsensusRubrics is a two-stage pipeline that automatically generates instance-specific evaluation rubrics from VLMs. Input: an image and a small seed set of human-annotated rubrics (e.g., 50 images with Must-Right and Easy-Wrong labels). Output: a rubric for each new image consisting of candidate essential facts and a binary Must-Right flag per fact.

**Load-bearing assumption**: Essential facts for correctness are consistently and explicitly generated across multiple chain-of-thought reasoning samples from a vision-language model. Additionally, essential facts are assumed to be visually present in the image; we verify this via a visual grounding step.

(B) **How it works**:
```python
# Phase 1: Candidate assertion extraction via self-consistency and visual grounding
def extract_candidate_assertions(image, VLM, N=8, threshold=0.75, visual_grounding_model, sim_threshold=0.5):
    responses = [VLM.generate_chain_of_thought(image) for _ in range(N)]
    # Parse each response into factual assertions using an LLM (e.g., GPT-4)
    assertions_list = [parse_facts(r) for r in responses]  # each returns set of strings
    # Compute frequency of each assertion across all responses
    freq = count_occurrences(assertions_list)  # dict: assertion -> count
    # Select assertions that appear in >threshold proportion of responses
    candidate_facts = {a for a, c in freq.items() if c / N >= threshold}
    # Visual grounding verification: discard facts not supported by image content
    grounded_facts = set()
    for fact in candidate_facts:
        # Encode fact and image regions (using object detector + region captioning)
        fact_embed = encode_sentence(fact)  # Sentence-BERT
        region_embeds = [encode_region_caption(cap) for cap in get_region_captions(image, visual_grounding_model)]
        max_sim = max([cosine_sim(fact_embed, r_embed) for r_embed in region_embeds])
        if max_sim >= sim_threshold:
            grounded_facts.add(fact)
    return grounded_facts

# Phase 2: Calibrate with seed human rubrics
def calibrate_on_seeds(seed_images, seed_rubrics, candidate_generator, visual_grounding_model):
    # For each seed image, generate candidate facts using Phase 1 (including visual grounding)
    # Build feature vector: frequency, average VLM confidence (if available),
    #                      position in CoT, length, visual grounding similarity score, etc.
    # Train a logistic regression classifier (L2 regularization, C=1.0) to predict
    # whether a candidate fact is Must-Right (1) or not (0)
    # using the seed rubrics as ground truth.
    features, labels = [], []
    for img, rubric in zip(seed_images, seed_rubrics):
        cands = candidate_generator(img, visual_grounding_model=visual_grounding_model)
        for fact in cands:
            vec = [freq_normalized, conf_score, pos_in_CoT, len, visual_sim]  # added visual_sim
            features.append(vec)
            labels.append(1 if fact in rubric['Must_Right'] else 0)
    classifier = train_logistic_regression(features, labels, C=1.0)
    return classifier

# At inference:
def generate_rubric(image, VLM, classifier, visual_grounding_model, N=8, threshold=0.75, sim_threshold=0.5):
    candidates = extract_candidate_assertions(image, VLM, N, threshold, visual_grounding_model, sim_threshold)
    scored = []
    for fact in candidates:
        feature_vector = compute_features(fact, image, VLM, visual_grounding_model)
        must_right_prob = classifier.predict_proba(feature_vector)[1]
        if must_right_prob > 0.5:
            scored.append((fact, 'Must-Right'))
        else:
            scored.append((fact, 'Easy-Wrong'))
    return scored
```
Hyperparameters: N=8; threshold=0.75; classifier = logistic regression with L2 regularization (C=1.0); visual grounding model = DETR for object detection + VinVL for region captioning; similarity threshold = 0.5 using Sentence-BERT (all-mpnet-base-v2).

(C) **Why this design**: We chose to use self-consistency frequency as the primary signal over alternatives like log-probability or model confidence because log-prob is poorly calibrated for factual correctness, whereas frequency across independent samples is a more robust indicator of essentialness (anti-pattern 3). We accept the computational cost of generating N=8 CoT samples per image. We chose a simple logistic regression calibrator rather than a large neural network to avoid overfitting to the small seed set (only 50 images), accepting that feature engineering is required. We parse facts using an LLM (e.g., GPT-4) instead of a rule-based extractor to handle diverse expressions, trading off API cost for flexibility. The threshold of 0.75 was chosen empirically to balance recall and precision; a lower threshold would include more noise, a higher threshold would miss valid essential facts. The visual grounding step was added to address the failure mode where essential facts are implicitly present but not explicitly stated in CoT; by verifying visual presence, we reduce false positives from hallucinated facts. This design deliberately avoids a learned disentanglement head (anti-pattern 1) because consistency is a direct observable, not a latent variable. This pipeline is not a simple combination of existing methods (anti-pattern 5): it uniquely uses consistency as a proxy for essentialness, a property not exploited by PerceptionRubrics or RM-R1—those rely on human-generated or distillation-based rubrics.

(D) **Why it measures what we claim**: The computational quantity `frequency of an assertion across N CoT samples` measures `essentialness` because we assume that essential facts are required for correct reasoning and thus will be consistently mentioned; this assumption fails when an essential fact is implied but not explicitly stated (e.g., spatial relations), in which case frequency underestimates essentialness, or when a non-essential but commonly stated artifact appears (e.g., image labels) leading to false positives. The `visual grounding similarity score` measures `visual presence` because it computes alignment between fact text and image regions; this assumes essential facts correspond to visually detectable elements, failing for abstract or inferential facts (e.g., 'the scene is chaotic'), leading to false negatives. The `must_right_prob` output from the logistic regression measures `human-aligned essentialness` because the seed rubrics provide supervision to distinguish genuine essential facts from consistent but non-essential ones; this assumption fails when the seed set does not represent the distribution of future images (domain shift), causing calibration drift. The `binary must_right flag` operationalizes the `Gated Scoring` concept from PerceptionRubrics, where failure on any must-right fact penalizes the entire response; the assumption is that essential facts are disjunctive—all must be correct—which may be too strict for tasks where partial credit is appropriate.

(E) **Assumptions and Failure Modes**
| Assumption | Failure Mode | Impact on Performance |
|------------|--------------|----------------------|
| Essential facts are consistently and explicitly generated in CoT | Implicit or spatially ambiguous facts omitted | Undetection of essential facts, lower recall |
| Essential facts are visually present in the image | Abstract or inferential facts without direct visual evidence | False negatives due to visual grounding filtering |
| Seed rubrics represent the distribution of test images | Domain shift (e.g., new object categories) | Calibration drift, reduced precision |
| All essential facts are disjunctive (must all be correct) | Tasks allowing partial credit | Overly strict scoring, lower human alignment |

## Contribution

(1) A novel method for automatic rubric generation that uses self-consistency of VLM chain-of-thought reasoning as a proxy for essentialness, eliminating the need for per-instance human annotation. (2) A calibration framework that leverages a small set of seed human rubrics to distill self-consistent assertions into Must-Right and Easy-Wrong categories, demonstrating that minimal human input suffices to achieve high alignment with human perception. (3) An analysis of the conditions under which self-consistency approximates essentialness, identifying failure modes such as implied facts and spurious correlations.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
|------|--------|-----------|
| Dataset | PerceptionRubrics test set (200 images with human rubrics) | Standard benchmark for rubric-based evaluation |
| Primary metric | Spearman ρ of gated score vs human judgment | Directly measures rubric alignment with humans |
| Baseline 1 | PerceptionRubrics (human-written rubrics) | Upper bound for automatic rubric quality |
| Baseline 2 | RM-R1 (rubric via reasoning) | Alternative automatic rubric generation method |
| Ablation-of-ours | ConsensusRubrics w/o calibration (frequency only) | Isolates effect of calibration on rubric accuracy |

### Why this setup validates the claim

This experimental design directly tests the central claim: that ConsensusRubrics generates evaluation rubrics whose quality approaches that of human-written rubrics. By using PerceptionRubrics' human-annotated test set as the dataset, we compare against the gold standard (human rubrics) and a competing automatic method (RM-R1). The primary metric, Spearman correlation between gated scores and human judgments, quantifies how well the rubrics predict human-perceived correctness. The ablation (no calibration) isolates the contribution of the calibration phase. If our method achieves a correlation close to human rubrics and significantly higher than RM-R1 and the ablation, it validates that self-consistency plus calibration effectively approximates human rubric generation. Conversely, if the ablation performs as well as the full method, calibration is unnecessary; if our method trails RM-R1, the design is inferior.

### Expected outcome and causal chain

**vs. PerceptionRubrics (human rubrics)** — On a complex scene with implicit essential facts (e.g., a mirror causing spatial ambiguity), human rubrics include the fact "object is to the left" because they infer it; our method may miss it if the fact is not consistently stated across CoTs. The gap arises because frequency underestimates essentialness when facts are implied. We expect a noticeable gap on such ambiguous subsets (e.g., 0.10 lower Spearman ρ) but parity on straightforward scenes where facts are explicit.

**vs. RM-R1** — On an image of a common object (e.g., a stop sign), RM-R1 may hallucinate a non-essential fact like "the sign has a white border" (a frequent but irrelevant detail). Our method filters this using calibration on seed rubrics that rarely include such details. The difference is due to calibration learning to suppress high-frequency but non-essential assertions. We expect our method to show higher precision (e.g., 0.15 higher Spearman ρ) and better correlation on samples with many visual cues.

**vs. Ablation (no calibration)** — On an image with a commonly co-occurring object (e.g., "sky" in outdoor scenes), the ablation includes the fact "there is sky" because it appears in all CoTs, even though it's not essential for the task. The calibration step corrects this by learning from seed rubrics that rarely mark sky as must-right. We expect the ablation to have at least 0.05 lower Spearman ρ, especially on scenes where non-essential but frequent facts are prevalent.

### What would falsify this idea
If the full method's correlation is not significantly higher than the ablation (i.e., calibration adds no value) or if it is lower than RM-R1, the central claim that self-consistency combined with calibration improves upon existing automatic rubric methods would be wrong. A uniform gain across all subsets rather than a concentrated gain on the predicted failure modes (e.g., implicit facts or frequent non-essentials) would also contradict the proposed causal mechanism.

## References

1. PerceptionRubrics: Calibrating Multimodal Evaluation to Human Perception
2. RM-R1: Reward Modeling as Reasoning
3. ResearchRubrics: A Benchmark of Prompts and Rubrics For Evaluating Deep Research Agents
4. Limits to scalable evaluation at the frontier: LLM as Judge won't beat twice the data
5. Fact, Fetch, and Reason: A Unified Evaluation of Retrieval-Augmented Generation
6. Skywork-Reward: Bag of Tricks for Reward Modeling in LLMs
7. Self-Generated Critiques Boost Reward Modeling for Language Models
8. PPI++: Efficient Prediction-Powered Inference
