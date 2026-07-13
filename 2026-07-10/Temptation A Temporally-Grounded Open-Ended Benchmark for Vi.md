# Temptation: A Temporally-Grounded Open-Ended Benchmark for Video Understanding

## Motivation

Existing video understanding benchmarks predominantly rely on multiple-choice QA, which suffers from shortcut bias as demonstrated by Video-Oasis: a significant portion of benchmark samples are not video-native, and models perform only marginally above random after filtering. The root cause is that the fixed answer set allows models to exploit answer distribution biases and text-only cues, bypassing genuine temporal reasoning. To move beyond this limitation, we need evaluation tasks that force models to generate structured temporal narratives, where shortcut strategies are ineffective.

## Key Insight

Requiring models to generate event graphs that are temporally consistent with the video eliminates shortcut biases inherent to fixed-answer benchmarks by making the evaluation criterion depend on explicit temporal structure that cannot be inferred from static frames or text alone.

## Method

(A) **What it is**: We propose **Temptation** (Temporal Prompt Evaluation for Understanding), a benchmark framework that presents models with a video and open-ended prompts requiring structured answers (e.g., event sequences). Outputs are parsed into event graphs and compared to ground-truth graphs via a temporal consistency metric that jointly measures event detection and temporal ordering.

(B) **How it works** (pseudocode):
```
for each (video, prompt) in benchmark:
    ground_truth_graph = load_event_graph(video)  # nodes: (event, start_time, end_time, attributes); edges: temporal_relation
    model_output = model.generate(prompt, video)   # free-form text
    predicted_graph = parse_to_graph(model_output, prompt)  # parser uses few-shot examples, threshold=0.7 for event matching
    score = temporal_consistency(predicted_graph, ground_truth_graph)
    temporal_consistency:
        # edge recall: fraction of ground_truth edges (temporal_relation) matched in predicted_graph
        # edge precision: fraction of predicted edges matched in ground_truth
        # event recall/precision: based on attribute similarity (cosine on embeddings, threshold 0.7)
        # final = harmonic_mean(edge_f1, event_f1)
        hyperparameters: edge_match_tolerance=0.5, event_sim_threshold=0.7
```

(C) **Why this design**: We chose event graphs over linear text because graphs capture nonlinear temporal relations (e.g., overlap, concurrency) common in video, which checks deeper understanding than simple ordering. We chose parser-based extraction over LLM-as-judge to avoid metric leakage; the trade-off is parsing errors, which we mitigate by using few-shot examples and conservative matching thresholds. We chose a harmonic mean of edge and event F1 to balance completeness and precision of temporal knowledge; this penalizes models that identify events but miss relations, or vice versa. We set a high event similarity threshold (0.7) to ensure only confident alignments count, accepting that some correct but paraphrased events may be missed, because false positives would inflate scores.

(D) **Why it measures what we claim**: The computational quantity `edge recall` measures the proportion of ground-truth temporal relations the model captures, which operationalizes **temporal understanding** because a model that outputs correct relations must have processed the full sequence and inferred cause-effect; this assumption fails when a relation is inferable from commonsense (e.g., 'person picks up object' then 'object moves'), in which case recall overestimates understanding. To mitigate, we select videos where alternative orders are plausible but wrong (counterfactual clips). `event recall` measures detection of individual events, operationalizing **visual perception**; the assumption is that detecting an event requires integrating visual frames over its duration, failing when events are recognizable from a single frame (e.g., 'a car is present'), in which case event recall may reflect static understanding. We address this by focusing on events that require temporal context (e.g., 'a car passes a pedestrian' vs. 'a car stops'). `edge precision` ensures the model does not hallucinate relations, measuring **faithfulness**; the assumption is that a predicted relation matches ground truth only if the model observed the actual temporal contingency, failing when the relation is coincidentally plausible given the video (e.g., two independent events occurring in order), in which case precision may overestimate faithfulness. We counter this by including distractor videos with similar events but different orders.

## Contribution

(1) Temptation, a benchmark framework with 1,500 videos annotated with event graphs covering 50 temporal relations, designed to be video-native by construction (all answers require temporal reasoning beyond static clues). (2) A temporal consistency metric that jointly evaluates event detection and temporal ordering, with known failure modes explicitly quantified. (3) A diagnostic analysis showing that open-ended generation exposes model weaknesses that multiple-choice QA masks, providing a more faithful assessment of video understanding.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
|------|---------|-----------|
| Dataset | Video-native challenges + temporal counterfactuals | Tests necessity of temporal comprehension |
| Primary metric | Temporal consistency (harmonic mean of edge & event F1) | Jointly measures detection and ordering |
| Baseline 1 | Random guessing | Shows benchmark not trivially solvable |
| Baseline 2 | SOTA Video-LLM (Video-LLaVA) | Represents current best on video understanding |
| Baseline 3 | Text-level temporal ordering (linear) | Isolates benefit of graph structure |
| Ablation-of-ours | Ours without edge recall (only event F1) | Tests role of temporal relation component |

### Why this setup validates the claim
This combination forms a falsifiable test of the central claim that temporal consistency via graph parsing reveals deeper video understanding. The dataset includes counterfactual videos where temporal order is not obvious, ensured that random guessing yields near-zero performance (baseline 1). Comparing against SOTA Video-LLM (baseline 2) tests whether existing models truly capture temporal relations or rely on shortcuts. The text-level baseline (3) isolates the benefit of graph representation for concurrency and overlap. The ablation removes the edge component, testing whether edge recall is necessary for measuring temporal understanding. The metric is the harmonic mean of edge and event F1, which jointly penalizes failures in both detection and ordering; if a model gets events right but orders wrong, the metric drops. This design ensures that only methods that correctly parse both events and relations can achieve high scores, directly validating the claim.

### Expected outcome and causal chain

**vs. Random guessing** — On a video where two events occur in a non-obvious order (e.g., wind blows flag, then person walks past), random guessing produces random temporal relations and random event labels (if guessing from a list), yielding near-zero edge recall and event recall. Our method parses the model's output into a graph and matches events only if similarity >0.7, so even a weak model can get some events right; expected: random ~0.01, our method ~0.3+ on event F1, and edge F1 near zero for random but possibly nonzero for method. The observable signal is a large gap on all subsets, especially on the counterfactual subset where random cannot benefit from commonsense.

**vs. SOTA Video-LLM** — On a counterfactual video where event order is reversed from typical (e.g., person falls before being pushed), a SOTA model relying on script knowledge may output the typical order (push then fall) despite visual evidence, leading to high event F1 but low edge recall. Our method, by requiring high similarity threshold (0.7) for event matching, may miss some events but will only accept temporally consistent edges; hence edge recall on counterfactuals will be lower for SOTA than for our method if our method is finetuned or prompted correctly. Expected: similar event F1 (~0.6), but edge F1 gap of ~0.2 favoring our method on counterfactual subset, with parity on canonical order videos.

**vs. Text-level temporal ordering** — On a video with concurrent events (e.g., one person talks while another walks), a linear text baseline can only output strict sequential order, missing overlap edges. Our method's graph representation explicitly models concurrent relations via edges. Expected: on overlap subset, text-level has edge recall ~0.1 (if it predicts false order), whereas our method achieves ~0.5. On sequential subsets, both may perform similarly. Observable signal: a noticeable edge recall gap on overlap subset, with minimal difference on event F1.

### What would falsify this idea
If our method's gain over the text-level baseline is uniform across all subsets (including those without concurrency), or if SOTA achieves comparable edge recall on counterfactual videos, then the claim that graph-based temporal consistency captures deeper understanding is falsified. Additionally, if the ablation without edge recall performs equally well on temporal subsets, then the edge component is not necessary.

## References

1. Video-Oasis: Rethinking Evaluation of Video Understanding
