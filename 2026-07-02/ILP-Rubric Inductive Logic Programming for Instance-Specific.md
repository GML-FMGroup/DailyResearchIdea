# ILP-Rubric: Inductive Logic Programming for Instance-Specific Rubric Generation from Few Examples

## Motivation

Rubric-based evaluation (e.g., PerceptionRubrics) relies on hand-crafted rubrics per task, which do not scale to new domains. Automatic rubric generation is needed, but existing methods (e.g., neural rule learning) sacrifice interpretability or determinism. The core challenge is to produce a rubric—a conjunctive set of binary tests—from only a few correct/incorrect examples while preserving the deterministic gating behavior of the endpoint.

## Key Insight

By constraining inductive logic programming to produce a minimal set of Horn clauses that exactly cover the positive examples and are all violated by each negative example, we guarantee the rubric is both deterministic and interpretable while needing only a handful of labeled instances.

## Method

## (A) What it is
ILP-Rubric is an algorithm that, given a small set of correct/incorrect answers for a specific query, outputs a rubric consisting of a conjunctive set of binary test rules. Each rule is a Horn clause over atomic predicates representing answer properties (e.g., "contains_time", "has_correct_unit").

## (B) How it works
```pseudocode
Input: Positive examples P (correct answers), negative examples N (incorrect answers), background knowledge B (atomic predicates extracted from answer text)
Output: Rubric R = {clause_1, ..., clause_k} where each clause is a conjunction of literals

Procedure ILP-Rubric(P, N, B):
  R = empty set
  Covered = P  # set of positives not yet fully covered (i.e., not failing a clause? We want all positives to satisfy all clauses)
  # We iteratively add clauses that all positives must satisfy and all negatives must fail at least one.
  while N is not empty:  # stop when no negative satisfies all current rules
    # Find best clause that discriminates positives from negatives
    best_clause = None
    best_score = -inf
    for each initial literal l from B:
      candidate = refine_clause(l, P, N, max_literals=5)  # beam search with beam size=10
      # Use cross-validated MDL for selection: 3-fold CV on P∪N, compute average MDL on held-out fold
      score = cross_validated_MDL(candidate, P, N, folds=3)
      if score > best_score:
        best_clause = candidate
        best_score = score
    R = R ∪ {best_clause}
    # Remove negatives that fail this clause (i.e., do not satisfy the clause) from N
    N = {neg in N : neg satisfies best_clause}  # keep only negatives that still satisfy all clauses (we want to force them to fail eventually)
    # Remove positives that fail this clause? Actually all positives must satisfy; so if a positive fails, that's an error; we want to minimize such errors. So we keep P unchanged.
  return R

function refine_clause(l, P, N, max_literals):
  best = l
  for depth = 1 to max_literals-1:
    neighbors = [add a new literal from B to best]
    best = argmax over neighbors of cross_validated_MDL(., P, N, folds=3)
  return best

function cross_validated_MDL(clause, P, N, folds):
  # Partition P∪N into folds
  # For each fold as held-out, train MDL on remaining, compute errors on held-out
  # Return negative average errors over folds (since we maximize)
  total_errors = 0
  for each fold (P_held, N_held) from P,N:
    errors = count(positives in P_held not satisfying clause) + count(negatives in N_held satisfying clause)
    total_errors += errors
  avg_errors = total_errors / folds
  # Compute BIC-based alpha: alpha = 0.5 * log(|P|+|N|) / (2*log(2)) ≈ 0.5 * log2(|P|+|N|)
  alpha = 0.5 * log2(len(P)+len(N))
  return - (avg_errors + alpha * length(clause))  # negative because we maximize

Hyperparameters: folds=3 (cross-validation), beam_size=10, max_literals=5, alpha dynamically set via BIC.
```

## (C) Why this design
We chose ILP over neural methods (e.g., LENSR) because ILP produces purely symbolic rules that are transparent and deterministic, whereas neural rule extraction may yield probabilistic or opaque conditions. We adopted a covering strategy (iteratively adding clauses until no negative satisfies all) to enforce the conjunctive structure of rubrics: all tests must pass for a correct answer. Using cross-validated MDL with a BIC-derived penalty (alpha = 0.5 * log2(|P|+|N|)) trades off rule complexity for accuracy while mitigating overfitting via out-of-sample error estimation. We hard-bound the literal count per clause (max_literals=5) to keep rules interpretable, accepting the risk that some concepts may require longer expressions. The beam search in refinement limits the search space, which is necessary for efficiency but may miss optimal clauses. Cross-validation addresses the load-bearing assumption that MDL-optimal rules generalize; the BIC penalty adapts the complexity penalty to dataset size.

## (D) Why it measures what we claim
* `errors` (positives not satisfying clause + negatives satisfying clause) measures the consistency of the rubric with provided examples. This operationalizes 'essential correctness' under the assumption that the examples are representative and unbiased. Failure mode: if examples are biased (e.g., all negatives miss a common but irrelevant detail), the rubric may reflect spurious correlations, and errors may not align with human judgment. * Clause length penalty alpha*length operationalizes Occam's razor, assumed to correlate with generalizability; this fails when the simple rule is coincidentally correct but misses a latent essential condition. In that case, the rubric reflects simplicity rather than ground truth necessity. * The cross-validated MDL score with BIC penalty directly estimates out-of-sample error, providing a data-driven balance between fit and complexity.

## Contribution

(1) A novel inductive logic programming algorithm tailored to produce conjunctive sets of binary test rules (rubrics) from as few as 3-5 labeled examples per query. (2) The design principle that minimal MDL-optimal rule sets from few examples yield deterministic evaluation that matches human-provided rubrics on held-out instances. (3) An open-source implementation of the ILP-Rubric system, usable as a drop-in module for any rubric-based evaluation pipeline.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|----------------------|
| Dataset | PerceptionRubrics dataset | Multimodal QA with human judgments |
| Primary metric | Human-Agreement Rate | Directly measures rubric alignment with human perception |
| Baseline 1 | LENSR neural rule extraction | Tests symbolic vs. neural rule quality |
| Baseline 2 | Holistic similarity (BERTScore) | Tests conjunctive vs. single-score evaluation |
| Ablation-of-ours | ILP-Rubric (alpha=0, no CV) | Isolates effect of complexity penalty and CV |

### Why this setup validates the claim
This experimental design directly tests the central claim that ILP-Rubric's symbolic, conjunctive rubrics capture essential correctness conditions better than existing approaches. The PerceptionRubrics dataset provides human judgments as ground truth, enabling direct measurement of alignment. LENSR represents neural rule extraction—if ILP-Rubric outperforms it, the symbolic inductive bias is validated. Holistic similarity (e.g., BERTScore) tests whether a single, continuous score can match rubric-based evaluation; superiority of ILP-Rubric would demonstrate the necessity of multiple necessary conditions. The ablation (alpha=0, no CV) isolates the impact of the MDL complexity penalty and cross-validation, testing whether simplicity and generalization estimation are crucial for performance. Human-Agreement Rate is the appropriate metric because it directly quantifies how often the rubric's pass/fail classification matches human annotators, which is the ultimate desideratum for evaluation benchmarks.

### Expected outcome and causal chain

**vs. LENSR** — On a case where a correct answer requires simultaneously containing a specific time and unit, LENSR might learn a probabilistic rule that fires with high probability when only one condition is met, due to correlations in training data, leading to false positives. Our ILP-Rubric instead learns a conjunctive clause requiring both predicates, because the ILP covering algorithm forces all positives to satisfy all clauses and negatives to fail at least one. We expect ILP-Rubric to achieve noticeably higher human agreement on such conjunctive instances, with an estimated 10-20% gap over LENSR, while performing similarly on simpler single-condition cases.

**vs. Holistic similarity (BERTScore)** — On a case where an answer is semantically similar to a correct answer but omits a critical negation (e.g., "not applicable" vs. "applicable"), a holistic score may award a high similarity due to strong lexical overlap, leading to a false positive. Our method, by requiring all constituent clauses (e.g., "contains_negation"), correctly fails such answers because the negation predicate is false. We expect ILP-Rubric to show a substantial advantage on ambiguity-driven cases, with an estimated 15-25% higher agreement, and parity on straightforward cases.

**vs. ILP-Rubric (alpha=0, no CV)** — On a small training set (e.g., 5 positives, 5 negatives), removing the complexity penalty and cross-validation leads to learning overly specific clauses that overfit idiosyncratic details (e.g., "contains_word_‘the’"), which fail to generalize to held-out examples. The full model with BIC-based alpha and cross-validation produces shorter, more general clauses. We expect the full model to maintain 85-90% human agreement on held-out sets, while the ablation drops to 70-75% due to overfitting.

### What would falsify this idea
If ILP-Rubric’s human agreement is not significantly higher than holistic similarity on conjunctive-critical subsets, or if the ablation without complexity penalty and cross-validation matches or exceeds the full model’s performance on held-out data, then the central claim that symbolic conjunctive rules and simplicity bias are essential would be refuted.

## References

1. PerceptionRubrics: Calibrating Multimodal Evaluation to Human Perception
2. RM-R1: Reward Modeling as Reasoning
3. ResearchRubrics: A Benchmark of Prompts and Rubrics For Evaluating Deep Research Agents
4. Limits to scalable evaluation at the frontier: LLM as Judge won't beat twice the data
5. Fact, Fetch, and Reason: A Unified Evaluation of Retrieval-Augmented Generation
6. Skywork-Reward: Bag of Tricks for Reward Modeling in LLMs
7. Self-Generated Critiques Boost Reward Modeling for Language Models
8. PPI++: Efficient Prediction-Powered Inference
