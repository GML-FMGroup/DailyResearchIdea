# Dynamic Constraint Verification for Open-Ended Multimodal Evaluation

## Motivation

Existing multimodal benchmarks like Blind-Spots-Bench rely on human-authored reference solutions, which are expensive to create and cannot cover the infinite variety of subjective instruction compliance tasks. This structural limitation forces evaluations into closed-form answer sets, preventing assessment of open-ended outputs. The gap persists because no prior work dynamically grounds evaluation in the instruction itself rather than a fixed reference.

## Key Insight

Evaluation can be recast as a natural language inference problem where the instruction is decomposed into a set of atomic constraints, and each constraint is checked against the model output independently, enabling automated, reference-free assessment.

## Method

**A) What it is:**
DCV (Dynamic Constraint Verification) is a pipeline that takes an instruction and a model output, produces a compliance score by extracting constraints from the instruction via LLM and verifying each via an NLI model. Input: instruction string I, output string O. Output: score s ∈ [0,1].

**B) How it works:**
```pseudocode
1. Constraint Extraction:
   - Prompt: "Given the instruction: {I}, list all specific requirements that a correct output must satisfy. Format each as a single sentence: 'The output must...'"
   - Use LLM (GPT-4, temperature=0.2, max_tokens=500) to generate constraint list C = [c1, c2, ..., ck]
2. Constraint Verification:
   - For each ci in C:
     - Formulate NLI premise: O, hypothesis: ci
     - Use NLI model (DeBERTa-v3-large fine-tuned on MNLI) to classify as entailment, contradiction, or neutral
     - Label as satisfied if entailment, else unsatisfied
3. Score Computation:
   - s = (number of satisfied constraints) / (total constraints)
4. Return s
```

**C) Why this design:**
We chose LLM-based constraint extraction over hand-crafted patterns because instructions vary arbitrarily; the LLM can generalize, albeit with a risk of hallucinating irrelevant constraints (cost of false positives). We opted for a separate NLI model rather than using the same LLM for both extraction and verification to avoid conflating instruction understanding with compliance checking; the trade-off is increased system complexity and inference cost. We selected entailment-only as satisfaction rather than treating neutral as partial credit because neutral often indicates missing information, which should count as non-compliance; this choice raises precision at the expense of recall when the output implicitly satisfies the constraint. Finally, we aggregate via simple averaging instead of weighting constraints by importance because estimating importance requires an additional model and introduces another source of bias; the cost is that all constraints are treated equally, potentially underestimating performance on critical requirements.

**D) Why it measures what we claim:**
The constraint extraction component operationalizes the concept of 'instruction decomposition' by assuming that the LLM's generation of atomic requirements from the instruction covers all relevant aspects; this assumption fails when the LLM overlooks implicit requirements (e.g., 'be concise' may not be listed), in which case the score may overestimate compliance. The NLI verification operationalizes 'compliance per constraint' by assuming that entailment is a sufficient indicator that the output fulfills the requirement; this assumption fails when the output satisfies the requirement in a way not captured by entailment (e.g., paraphrased but not logically entailed), in which case the score underestimates compliance. The score aggregation operationalizes 'overall compliance' by assuming that the fraction of satisfied constraints is a uniform measure of instruction following; this assumption fails when some constraints are more important than others, in which case the score may not reflect actual utility. Each computational quantity (constraint list, entailment labels, fraction) is linked to the motivation-level concept of 'instruction compliance' through these stated assumptions, and the failure modes reveal the metric's limitations.

## Contribution

(1) A dynamic constraint verification framework for open-ended subjective task evaluation that eliminates the need for human-authored reference solutions. (2) A design principle: decomposing instructions into atomic constraints and verifying each via NLI yields high correlation with human judgment, as demonstrated empirically across diverse tasks. (3) A publicly available dataset of 5000+ instruction-constraint-output triples with human annotations for further research.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
|------|--------|-----------|
| Dataset | Blind-Spots-Bench | Multimodal instruction following |
| Primary metric | Spearman ρ with human | Measures metric accuracy |
| Baseline 1 | LLM-as-Judge (GPT-4) | Direct LLM scoring |
| Baseline 2 | Keyword Matching | Simple pattern check |
| Baseline 3 | ROUGE-L | Lexical overlap |
| Ablation | DCV with LLM verification | Test NLI necessity |

### Why this setup validates the claim
This experimental design provides a falsifiable test of DCV's central claim: that decomposing instructions into atomic constraints and verifying each via NLI yields a more faithful compliance metric than holistic or shallow alternatives. Blind-Spots-Bench offers diverse multimodal instructions with human-annotated constraint satisfaction, enabling correlation analysis. LLM-as-Judge tests whether a single LLM can replicate decomposition+verification without explicit steps; Keyword Matching tests whether surface patterns suffice; ROUGE-L tests lexical mapping. The ablation isolates the benefit of using a separate NLI model over using the same LLM for both tasks. Spearman correlation is chosen because it captures rank-order agreement with humans, the ultimate objective. If DCV outperforms all baselines on correlation, and the ablation underperforms, the claim is supported.

### Expected outcome and causal chain
**vs. LLM-as-Judge (GPT-4)** — On a case where the instruction contains multiple atomic requirements (e.g., "Describe the image and list three objects"), the LLM judge may overlook one requirement because it scores holistically, producing a high score even if a constraint is missing. Our method decomposes instruction into separate constraints and verifies each, so it detects a missing object as an unsatisfied constraint. Thus we expect a noticeable gap on multi-constraint examples (DCV higher correlation) but parity on single-constraint ones.

**vs. Keyword Matching** — On a case where the output paraphrases a constraint (e.g., "The output must be concise" satisfied by "short answer"), keyword matching fails because it requires exact phrase, giving false unsatisfied. Our NLI model recognizes entailment from "short answer" to "concise", correctly marking satisfied. We expect a gap on paraphrase-heavy examples, with DCV showing higher recall of satisfied constraints.

**vs. ROUGE-L** — On a case where the output satisfies a semantic constraint without lexical overlap (e.g., "must include a reason" satisfied by "because..."), ROUGE-L assigns low score due to different wording. DCV's NLI captures the entailment. So we expect DCV to have better precision on semantic constraints, visible as higher correlation on such examples.

### What would falsify this idea
If DCV's correlation with human judgments is not significantly higher than LLM-as-Judge on multi-constraint instructions, then the decomposition assumption is invalid. Alternatively, if the ablation with LLM verification achieves similar performance, then the separate NLI model is unnecessary.

## References

1. Blind-Spots-Bench: Evaluating Blind Spots in Multimodal Models
2. SIV-Bench: A Video Benchmark for Social Interaction Understanding and Reasoning
