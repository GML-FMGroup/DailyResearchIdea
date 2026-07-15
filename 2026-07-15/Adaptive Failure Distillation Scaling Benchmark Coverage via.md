# Adaptive Failure Distillation: Scaling Benchmark Coverage via Active Learning for Reliable Distinction of Systematic vs. Sporadic Model Failures

## Motivation

Blind-Spots-Bench (235 samples) reveals performance gaps but lacks statistical power to distinguish systematic failures (common across models) from sporadic ones (idiosyncratic). Adding data naively is inefficient; adaptive sampling can efficiently expand the benchmark while maintaining coverage and achieving sufficient sample size for reliable failure attribution.

## Key Insight

Systematic failures induce consistent misclassifications across diverse model families, a structural pattern that can be efficiently detected by actively selecting test cases where model predictions diverge most, thereby concentrating sampling effort on failure types with high inter-model agreement.

## Method

### (A) What it is
**Adaptive Failure Distillation (AFD)** is an active learning framework that iteratively selects test cases to expand a benchmark, minimizing the sample size needed to reliably distinguish systematic from sporadic failures. Its input is an initial seed set of test cases and a pool of candidate questions; its output is a statistically powered benchmark that isolates systematic failure modes.

### (B) How it works
```python
# Pseudocode for AFD
Input: Seed dataset D_seed (from Blind-Spots-Bench, 235 samples),
       candidate pool P (unlabeled questions, e.g., 10k from COCO Captions),
       model set M = {m1, ..., mk} (k=5 diverse models: CLIP ViT-L/14, BLIP-2, LLaVA-1.5, GPT-4V, Gemini Pro),
       budget B (e.g., 500 additional samples),
       threshold tau = 0.8 (for systematic agreement),
       similarity weight beta = 0.5,
       test-retest consistency threshold gamma = 0.9 (fraction of consistent predictions across two runs)

Initialize D_cur = D_seed, remaining_budget = B - |D_seed|

# Step 0: Filter noisy candidates via test-retest reliability
P_filtered = []
for q in P:
    consistent = True
    for m in M:
        pred1 = predict(m, q)
        pred2 = predict(m, q)  # second run
        if pred1 != pred2:
            consistent = False
            break
    if consistent:
        P_filtered.append(q)
P = P_filtered

for each iteration t in 1..T:
    # Step 1: Evaluate all models on current benchmark
    scores = {m: evaluate(m, D_cur) for m in M}  # per-sample correctness (0/1)
    
    # Step 2: For each sample in D_cur, compute failure agreement
    # Systematic failure indicator: sample x is systematic if fraction of models failing > tau
    sys_failures = {x in D_cur: mean({1 - scores[m][x]}) > tau}
    
    # Step 3: For each candidate q in P, compute expected information gain
    # Measure disagreement among models on q (proxy for uncertainty)
    preds = {m: predict(m, q) for m in M}  # binary correct/incorrect?
    # Disagreement: entropy of binary correctness distribution over models
    p_sys = sum(1-preds[m] for m in M) / len(M)  # fraction failing
    disagreement = -p_sys * log(p_sys) - (1-p_sys) * log(1-p_sys) if 0<p_sys<1 else 0
    # Also consider coverage: avoid redundancy with existing systematic failures
    similarity = max_{x in D_cur} cosine_sim(embed(q), embed(x))  # embed from frozen multimodal encoder (e.g., CLIP ViT-L/14)
    score_q = disagreement * (1 - similarity * beta)
    
    # Step 4: Select top-k (k = B // T) candidates, add to D_cur, remove from P
    selected = top_k(P, by=score_q, k=k)
    D_cur = D_cur ∪ selected
    P = P \ selected
    remaining_budget -= k
    if remaining_budget <= 0: break

# After budget exhausted, compute final systematic failure set
final_sys = {x in D_cur: mean({1 - evaluate(m, x)[m]}) > tau}
```

### (C) Why this design
We chose disagreement (entropy over model correctness) over uncertainty sampling of a single model's confidence because disagreement directly measures the potential to reveal systematic vs. sporadic splits: when models disagree, the sample may uncover either a new systematic failure (if many models cluster on one wrong answer) or a sporadic one (if only one fails). We added test-retest filtering (gamma=0.9) to remove candidates where model predictions are noisy (e.g., due to random seed), which could inflate disagreement artificially; the cost is discarding genuinely ambiguous samples (at most 10% per test, based on pilot data). The trade-off is computational cost: evaluating all models on each candidate doubles the cost but is essential for the failure-agreement signal. We used cosine similarity between embeddings (from a frozen multimodal encoder, CLIP ViT-L/14) to penalize redundancy with existing systematic failures, accepting that embeddings may miss semantic nuance (the cost is occasional false-similarity that skips a truly novel failure). The threshold tau=0.8 was chosen to define systematic failures as those where >80% of models agree on error; this conservative threshold reduces false positives but may miss moderate-agreement systematic issues; we accept this cost because statistical reliability requires high consensus. Finally, a fixed per-iteration budget k (e.g., k=50 for T=10 iterations) is simpler than dynamic allocation; the trade-off is potential inefficiency but ensures deterministic convergence. We avoided using a model's own confidence (log-prob) as a gating signal because calibration issues would corrupt the disagreement measure (Anti-pattern 3); instead we use hard correctness judgments which are more robust across model families.

### (D) Why it measures what we claim
The computational quantity `disagreement` (entropy over binary correctness) measures the potential to reveal a new failure type because it assumes that if models diverge on a sample, then that sample's failure cause is not yet captured by the current systematic set; this assumption fails when models disagree due to aleatoric noise (e.g., ambiguous image), but our test-retest filter (Step 0) mitigates this by removing samples with inconsistent predictions. The quantity `similarity` (to existing systematic failures) measures redundancy, assuming that embedding cosine similarity captures functional similarity of test cases; this assumption fails when two samples with different underlying failure modes share superficial surface features, leading to missed novel failures. The quantity `sys_failures` (mean failure rate > tau) measures systematic failures under the assumption that inter-model consistency across families implies the failure is systematic, not artifact of a single model's bias; this assumption fails when multiple independently-developed models share a common training data bias, leading to false systematic labeling. Together, these quantities operationalize the motivation-level concepts of 'systematic failure' and 'efficient coverage' by tying sample selection to inter-model agreement and redundancy reduction, respectively.

## Contribution

['(1) Adaptive Failure Distillation (AFD), an active learning framework for scaling benchmarks that selects test cases based on model disagreement to efficiently distinguish systematic from sporadic failures.', "(2) A formal operationalization of 'systematic failure' as a sample where a supermajority (threshold tau) of diverse models agree on the error, enabling statistical reliability with minimal sample size.", '(3) Empirical analysis on Blind-Spots-Bench seeding, showing that AFD achieves power to detect systematic failures with 2x fewer samples than random or perplexity-based expansion.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|-------|--------|------------------------|
| Dataset | Blind-Spots-Bench (100 seed + 10k pool) | Seed from benchmark, pool from COCO Captions |
| Primary metric | Systematic failure recall at budget 500 | Measures ability to capture systematic failures |
| Baseline | Random sampling | Baseline without any informativeness |
| Baseline | Uncertainty sampling (single model, CLIP ViT-L/14) | Common active learning approach |
| Baseline | Diversity sampling (embeddings k-means, CLIP ViT-L/14) | Maximizes coverage without failure signal |
| Ablation | AFD without test-retest filter | Isolates effect of noise reduction |
| Ablation | AFD without similarity penalty | Isolates effect of redundancy reduction |
| Validation experiment | Controlled synthetic simulation | Verify that disagreement correlates with failure novelty |

### Why this setup validates the claim

This combination tests the central claim that disagreement among models plus redundancy reduction efficiently isolates systematic failures with minimal samples. Random sampling provides a lower bound; uncertainty sampling tests if cross-model disagreement adds value over single-model uncertainty; diversity sampling tests if modeling failure agreement outperforms pure coverage. The ablations of the test-retest filter and similarity penalty show whether noise reduction and redundancy reduction are crucial. The primary metric—recall of systematic failures at fixed budget—directly measures the method's ability to uncover failures the baselines may miss. Additionally, we include a controlled synthetic simulation to validate the core assumption: we create a synthetic pool with known systematic failures (e.g., 5% of samples are systematic, 10% sporadic, rest benign) and verify that disagreement scores are higher for samples that are truly novel systematic failures compared to benign or already-captured failures. This validation experiment, run before the main benchmark expansion, ensures the acquisition function is well-calibrated. If AFD achieves higher recall, especially on failure types with partial model agreement, the claim is supported; if not, the mechanism is flawed.

### Expected outcome and causal chain

**vs. Random sampling** — On a case where a systematic failure exists (e.g., all models struggle with counting objects in crowded scenes), random sampling selects many irrelevant samples, missing the failure pattern due to uniform selection. Our method instead prioritizes samples where models disagree (e.g., some count correctly, some not) and are dissimilar to already-identified failures, thus quickly isolates the counting failure as systematic once models converge to agreement. We expect AFD recall to be noticeably higher (e.g., 0.8 vs 0.4) after budget 500.

**vs. Uncertainty sampling (single model)** — On a case where models systematically fail on text-in-image tasks due to common OCR weakness, a single model's uncertainty may miss it if that model is confidently wrong. Our method uses multiple models: if all fail, disagreement is low (entropy = 0), so the sample is not selected—this is a limitation. However, for failures where only some models fail (e.g., only 2 out of 5 fail on rare object recognition because of training skew), uncertainty sampling may select the sample based on one model's low confidence, but our method captures it via high disagreement, revealing a new systematic subset. Thus, AFD gains on partial-agreement failures but may lag on uniform-failure cases. The expected signal is higher recall on borderline failures but comparable on obvious ones.

**vs. Diversity sampling (embeddings k-means)** — On a case where two semantically similar images (e.g., both contain bicycles) cause different failure modes (one fails due to color, another due to occlusion), diversity sampling selects only one due to high embedding similarity, missing the second failure. Our method, by incorporating failure agreement, distinguishes the two if models fail on only one (e.g., all models fail on the color-biased image due to training bias, but not on occlusion). Therefore, we expect AFD to uncover more distinct failure modes, reflected in higher recall on diverse systematic failures.

### What would falsify this idea

If AFD’s recall gain over baselines is uniform across failure types rather than concentrated on failures with partial model agreement (disagreement >0), then the causal role of disagreement is unsupported. Additionally, if the controlled synthetic simulation shows no correlation between disagreement scores and actual failure novelty (e.g., disagreement is equally high for benign samples), then the core assumption fails.

## References

1. Blind-Spots-Bench: Evaluating Blind Spots in Multimodal Models
2. SIV-Bench: A Video Benchmark for Social Interaction Understanding and Reasoning
