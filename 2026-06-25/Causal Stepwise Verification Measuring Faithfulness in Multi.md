# Causal Stepwise Verification: Measuring Faithfulness in Multimodal Chain-of-Thought Reasoning

## Motivation

Existing evaluations of multimodal chain-of-thought reasoning, such as those in Look Light, Think Heavy and PuzzleVQA, rely on final answer accuracy, which cannot distinguish whether correct answers arise from valid reasoning or spurious correlations. This limitation stems from the lack of per-step ground-truth verification, leaving intermediate reasoning as a black box.

## Key Insight

A step's faithfulness is directly quantified by the invariance of the final answer probability when that step is replaced by its ground-truth sub-answer, because any change in probability indicates that the model either ignored or contradicted the correct sub-answer.

## Method

(A) **What it is**: Causal Stepwise Verification (CSV) is a procedure that assesses the faithfulness of each chain-of-thought (CoT) step by measuring the causal effect of replacing that step with its ground-truth sub-answer on the final answer likelihood. Input: multimodal input (image, question), model M, generated CoT steps S = {s_1,...,s_n}, final answer a, and ground-truth sub-answers for each step G = {g_1,...,g_n}. Output: faithfulness scores F = {f_1,...,f_n} where each f_i is a real number between 0 and 1 (after calibration).

(B) **How it works** (pseudocode):
```python
def csv_faithfulness(model, input, steps, final_answer, ground_truth_subanswers):
    # Compute original final answer probability
    original_prob = model.p(input, steps)[final_answer]
    faithfulness = []
    for i in range(len(steps)):
        # Create intervened CoT: replace step i with its ground-truth sub-answer
        intervened_steps = steps[:i] + [ground_truth_subanswers[i]] + steps[i+1:]
        intervened_prob = model.p(input, intervened_steps)[final_answer]
        # Difference score: original - intervened (range approx -1 to 1)
        f_i = original_prob - intervened_prob
        faithfulness.append(f_i)
    # Calibration: apply affine transformation to map to [0,1] using a held-out calibration set
    # (details in experiment)
    return faithfulness
```
Hyperparameter: calibration scale and shift learned on a calibration set of 100 steps from a held-out split of ScienceQA.

(C) **Why this design**: We chose direct probability measurement over binary accuracy because it provides a continuous measure sensitive to small perturbations, accepting the cost of requiring access to model logits. We chose intervention at the step level rather than at the token level to match the granularity of reasoning steps, acknowledging that finer granularity would be more precise but requires more ground-truth annotations. We chose replacement with ground-truth sub-answers rather than masking or random tokens because it tests whether the model's reasoning conforms to the correct information, at the cost of not detecting cases where the step is faithful but the ground-truth sub-answer is formulated differently. Finally, we compute faithfulness as the difference score (original_prob - intervened_prob) rather than raw intervened_prob because it isolates the effect of the intervention and is less sensitive to low original probabilities; the difference is then calibrated to [0,1] via a linear mapping on a held-out set.

(D) **Why it measures what we claim**: The quantity `original_prob - intervened_prob` measures faithfulness to ground-truth reasoning because we assume that a faithful step, when replaced by the correct sub-answer, should not change the model's final answer probability; this assumption fails when the model relies on spurious correlations in the original step that happen to align with the ground truth, in which case intervened_prob may remain high even though the original step was not truly faithful—the difference score mitigates this by capturing the drop. Conversely, if intervened_prob drops, it indicates that the original step was necessary for the answer, but could still be incorrect (e.g., a wrong step that the model uses). To isolate correctness, we would need to also check whether replacing with a wrong sub-answer increases probability, which is a natural extension. The method thus measures a combination of necessity and consistency with ground truth, not pure correctness. 

**Assumption**: The final answer probability after replacing a step with its ground-truth sub-answer directly reflects the faithfulness of that step, under the premise that the model's reasoning is monotonic — i.e., incorporating correct sub-answers should not lower answer probability. This assumption may fail due to distributed representations (Geiger et al., 2021). To mitigate, we use the difference score instead of raw intervened_prob, and we perform a calibration experiment across multiple model architectures to ensure the measure correlates with human judgments. The calibration learns a linear mapping to align the difference scores with human faithfulness ratings on a small held-out set.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | ScienceQA reasoning split | Contains ground-truth sub-answers |
| Primary metric | Spearman r with human faithfulness | Direct test of claimed measure |
| Baseline 1 | Random scoring | Null baseline to show signal |
| Baseline 2 | Step likelihood | Tests confidence as faithfulness |
| Baseline 3 | Masking intervention | Tests intervention necessity |
| Baseline 4 | Token-level intervention | Tests granularity choice |
| Ablation-of-ours | Random token intervention | Tests ground-truth importance |

### Why this setup validates the claim

ScienceQA provides step-by-step reasoning annotations with human faithfulness ratings per step. By correlating CSV scores with these ratings, we directly test whether CSV captures the degree to which a step is faithful to correct reasoning. The random baseline sets a lower bound; any positive correlation indicates systematic signal. Step likelihood tests whether a simpler confidence metric suffices—if not, the intervention mechanism is necessary. Masking intervention tests whether replacement with any token (rather than ground-truth) yields similar scores; if CSV outperforms masking, the ground-truth content matters. Token-level intervention tests whether step-level granularity is superior to finer granularity; if CSV outperforms, the step-level design is validated. Finally, the ablation with random tokens isolates the role of using correct sub-answers: if it performs worse than CSV, then the specific choice of ground-truth is critical. Additionally, we calibrate the difference score using a held-out set of 100 steps from a separate split of ScienceQA, fitting a linear mapping to human faithfulness ratings, and report the Spearman correlation after calibration. This combination of baselines ensures each sub-claim is independently falsifiable, and the metric directly measures alignment with human intuition.

### Expected outcome and causal chain

**vs. Random scoring** — On a case where the model uses a spurious correlation in a step (e.g., a visual feature that correlates with the answer but is not causally related), random scoring assigns arbitrary values. Our method, by replacing the step with its ground-truth sub-answer, would likely produce a large positive difference (low faithfulness) because the model's final probability drops when the spurious step is corrected. Thus, we expect CSV to show a noticeable positive correlation with human ratings, whereas random scoring shows near-zero correlation.

**vs. Step likelihood** — On a case where a step is highly confident but factually wrong (e.g., the model confidently states “the cat is black” when the cat is white), step likelihood gives a high faithfulness score, contradicting human judgment. CSV, by intervening with the correct color, would yield a large positive difference (low faithfulness), correctly indicating unfaithfulness. Therefore, we expect CSV’s correlation to be substantially higher than that of step likelihood, especially on steps where confidence and correctness diverge.

**vs. Masking intervention** — On a case where a step contains irrelevant but true information (e.g., “the sky is blue” in a question about bird species), masking with [MASK] might artificially drop probability because the model expects a token there, while CSV using the ground-truth sub-answer (which is the correct next step) would maintain high probability (small difference). Thus, CSV should assign high faithfulness where masking assigns low faithfulness, aligning with human ratings. We expect CSV to show higher correlation than masking, particularly on steps that are faithful but not predictable from context.

**vs. Token-level intervention** — Token-level interventions are more precise but require per-token ground-truth annotations, which are not available in ScienceQA. We generate token-level ground truth by tokenizing the step's sub-answer and intervening on each token individually, averaging the differences. We expect step-level CSV to achieve higher correlation with human ratings because human faithfulness judgments are at the step level, and token-level interventions may be noisier due to token dependencies.

### What would falsify this idea

If CSV’s correlation with human faithfulness is not significantly higher than all baselines (especially the random token ablation and token-level intervention), or if the gain is uniform across all step types rather than concentrated on steps where confidence or masking diverge from faithfulness, then the central claim—that CSV measures faithfulness via ground-truth replacement—is not supported.

## References

1. Look Light, Think Heavy: What Multimodal Chain-of-Thought Reasoning Can and Cannot Do
2. PuzzleVQA: Diagnosing Multimodal Reasoning Challenges of Language Models with Abstract Visual Patterns
3. The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)
4. The ConceptARC Benchmark: Evaluating Understanding and Generalization in the ARC Domain
5. Sparks of Artificial General Intelligence: Early experiments with GPT-4
