# Consensus-Constrained Multi-Teacher Distillation (CC-MTD): Debiasing Teacher Outputs via Inlier Selection

## Motivation

Existing knowledge distillation methods assume teacher outputs are unbiased, but language model teachers exhibit calibration errors and prompt sensitivity that are passed to the student. For instance, MCC-KD uses multiple rationales but averages all teacher outputs without identifying which ones are biased, allowing systematic errors to propagate. The root cause is that no method explicitly models the fact that prompt-induced biases are inconsistent across diverse prompt templates, while true knowledge is stable. We need a mechanism to detect and remove such biases per sample.

## Key Insight

Systematic biases due to prompt sensitivity cause teacher outputs to deviate from the true distribution in an inconsistent manner across diverse prompts, whereas the true signal is consistent, enabling detection via pairwise divergence clustering.

## Method

### (A) What it is
**CC-MTD** (Consensus-Constrained Multi-Teacher Distillation) is a distillation method that uses a set of teacher models each with a different prompt template to produce distributions over tokens, then selects a consensus subset of teachers per sample whose outputs are mutually consistent, and uses their average as the debiased soft target for student training.

### (B) How it works
```pseudocode
Input: Set of N teacher models T = {t_1,...,t_N}, each with a distinct prompt template; student model S; temperature tau.
For each training sample x:
  1. For each teacher i, compute output distribution p_i = softmax(t_i(x) / tau).
  2. For each pair (i,j), compute KL_div(i||j) = sum_k p_i(k) * log(p_i(k)/p_j(k)).
  3. Build similarity graph G where edge (i,j) exists if KL_div(i||j) < epsilon AND KL_div(j||i) < epsilon (epsilon is a hyperparameter, e.g., 0.1).
  4. Find the largest clique (or connected component) in G. Let C be the set of teachers in that clique. (If multiple cliques of same size, pick one; if none, fallback to all teachers.)
  5. **Calibration verification (new step):** Compute the average probability assigned by the consensus set C to a held-out unbiased reference (e.g., human labels if available, or a robust teacher such as GPT-4 with a neutral prompt) on a small calibration set of 512 examples. If the average agreement is below 0.5 or |C| < N/2, fallback to using all teachers (i.e., set C = all teachers).
  6. Compute debiased target distribution q = (1/|C|) sum_{i in C} p_i.
  7. Train student on cross-entropy between its output and q.
```
Hyperparameters: N=5-10 teachers; tau=1; epsilon=0.1 (tuned on validation); calibration threshold=0.5, calibration set size=512.

### (C) Why this design
We chose threshold-based inlier selection (step 3) over soft weighting (e.g., inverse variance) because it cleanly separates biased and unbiased teachers per sample, avoiding dilution of the signal. We chose KL divergence as the pairwise metric because it is asymmetric and sensitive to distribution mismatch—directly capturing the notion of bias. A symmetric alternative (e.g., Jensen-Shannon) could be used, but KL is more sensitive to one-sided deviations (bias). The trade-off is that epsilon requires tuning; too high and biased teachers may be included, too low and we may discard informative but slightly noisy teachers. We also chose to use all prompt templates (not a fixed set) because prompt diversity is critical for detecting systematic (rather than random) biases; the cost is increased computational burden during teacher inference. Finally, we opted for clique or largest connected component (step 4) over simply averaging all teachers because it removes outliers; the trade-off is a potential loss of sample size when the consensus set is small, but the gain in target quality outweighs this. The calibration verification (step 5) ensures robustness when the assumption of unimodality is violated, by falling back to all teachers if the consensus set is too small or agrees poorly with a trusted reference.

### (D) Why it measures what we claim
The inlier selection step (the computational quantity) measures *teacher trustworthiness* by identifying teachers whose outputs are mutually consistent. This assumes that systematic biases due to prompt cause deviations that are not shared across diverse prompts, so inconsistency signals bias. **Assumption A:** The true distribution is unimodal and prompt-induced biases are independent across teachers. **Failure mode F:** When the true distribution is multimodal (e.g., multiple equally valid answers), consistency may select only one mode, causing the metric to mistake diversity for bias. In that case, the selected subset measures *mode selection* rather than trustworthiness. The fallback to all teachers (when no clique or calibration check fails) also assumes that when no consistency emerges, the teachers are all equally unreliable, which may be pessimistic but safer. The calibration verification step provides a failsafe against violation of Assumption A.

## Contribution

(1) A novel debiasing mechanism for knowledge distillation that selects a consensus set of teacher outputs per sample using pairwise KL divergence, without requiring any external supervision. (2) The empirical finding that prompt-induced biases are inconsistent across diverse prompt templates, enabling principled bias removal and improving student robustness.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|-----------------------|
| Dataset | HellaSwag | Hard multiple-choice with known label ambiguity |
| Primary metric | Accuracy | Measures correct concept understanding |
| Baseline 1 | Vanilla KD (single teacher) | Standard distillation without diversity |
| Baseline 2 | Uniform Multi-Teacher (average all) | No outlier removal |
| Baseline 3 | Best Single Prompt Teacher | Upper bound from best prompt |
| Baseline 4 | Ranking by Variance (select teachers with least variance) | Tests importance of graph structure vs. simpler heuristic |
| Baseline 5 | Trimmed Mean (remove top/bottom 20% of teachers) | Robust averaging alternative |
| Ablation | CC-MTD without graph (all teachers) | Tests necessity of consensus |

### Why this setup validates the claim
HellaSwag is chosen because it contains ambiguous commonsense reasoning instances where different prompt templates can elicit different correct answers from teachers, creating systematic bias. The baselines isolate distinct failure modes: Vanilla KD lacks teacher diversity, Uniform Multi-Teacher blindly averages all including biased ones, Best Single Prompt Teacher ignores per-instance variation, Ranking by Variance uses a simpler consistency measure, and Trimmed Mean provides a non-graph-based robust aggregation. Our ablation removes the graph consensus step to test its necessity. Accuracy directly measures whether the student internalizes correct knowledge from debiased targets; improvement on ambiguous samples (where teachers disagree) would confirm that consensus selection filters out prompt-induced bias, while performance on clear samples should remain comparable, forming a falsifiable pattern. The inclusion of a held-out calibration set (512 examples) in the method allows verification that the consensus set agrees with a trusted reference, providing a safety check against the assumption that consistency implies unbiasedness.

### Expected outcome and causal chain

**vs. Vanilla KD (single teacher)** — On a sample where the single teacher's prompt induces a wrong answer due to phrasing (e.g., negation), Vanilla KD trains the student to mimic that error because it has no alternative teacher. Our method instead excludes that teacher from the consensus set if its output diverges from others, yielding a target free of that bias. We expect our method to outperform Vanilla KD primarily on samples with high teacher variance (e.g., 5–10% accuracy gap), while performing similarly on low-variance samples.

**vs. Uniform Multi-Teacher (average all)** — On a sample where one teacher is strongly biased (e.g., due to an ambiguous pronoun), Uniform averaging dilutes the correct signal from other teachers, producing a noisy target. Our method detects that biased teacher via pairwise KL divergence and removes it from the consensus set, so the target remains clean. We expect our method to show a noticeable accuracy gain (e.g., 2–5%) on samples where outlier teachers exist, but close to parity when all teachers agree.

**vs. Best Single Prompt Teacher** — On a sample where the best prompt (selected on validation) triggers a rare bias (e.g., due to topic shift), that teacher fails. Our method adapts per sample by selecting the largest consistent clique of teachers, which may exclude the best teacher if it is inconsistent with others. Thus, on those rare biased samples, our method recovers correct targets. We expect our method to match or exceed the best teacher on most samples, and surpass it by a few percentage points on the subset where the best teacher is inconsistent.

**vs. Ranking by Variance** — On a sample where a teacher has low variance but is consistently wrong (e.g., due to a common prompt template flaw), Ranking by Variance would incorrectly retain that teacher because its output is stable. Our method's pairwise KL divergence would detect that this teacher's output differs from others even if its variance is low, and thus exclude it from the consensus set. We expect our method to outperform Ranking by Variance on samples where low-variance teachers are biased (e.g., 3–5% accuracy gain), while performing similarly on unbiased low-variance samples.

**vs. Trimmed Mean** — On a sample where biases are not extreme (e.g., only one teacher out of five is biased), Trimmed Mean would remove 20% of the teachers (1 out of 5) and potentially remove the biased one. However, if the bias is moderate (not extreme high/low), Trimmed Mean may fail to exclude it. Our method's graph-based approach can identify discrepant teachers even if their outputs are not extreme. We expect our method to show a modest improvement (1–3%) over Trimmed Mean on samples with moderate biases, and parity on samples with extreme outliers.

### What would falsify this idea
If our method shows no greater improvement on samples with high teacher variance compared to low variance, or if the ablation (using all teachers without graph) performs equally well on all samples, then the consensus mechanism is not actually debiasing targets. Additionally, if the calibration verification step (falling back to all teachers when the consensus set is small or disagrees with the reference) does not improve performance on multimodal samples, that would indicate the fallback is unnecessary or detrimental.

## References

1. Knowledge distillation and dataset distillation of large language models: emerging trends, challenges, and future directions
2. TAGCOS: Task-agnostic Gradient Clustered Coreset Selection for Instruction Tuning Data
3. A Comprehensive Survey of Small Language Models in the Era of Large Language Models: Techniques, Enhancements, Applications, Collaboration with LLMs, and Trustworthiness
4. From Quantity to Quality: Boosting LLM Performance with Self-Guided Data Selection for Instruction Tuning
5. The Internal State of an LLM Knows When its Lying
6. MCC-KD: Multi-CoT Consistent Knowledge Distillation
7. AWQ: Activation-aware Weight Quantization for On-Device LLM Compression and Acceleration
8. Self-Knowledge Guided Retrieval Augmentation for Large Language Models
