# Counterfactual Step-Level Credit for Hierarchical Multi-Agent Distillation

## Motivation

Existing on-policy distillation methods such as MultiAgent-OPD and GKD assume that token-level teacher scores on student-generated sequences provide sufficient step-level credit for intermediate decisions, yet they never test whether those scores are identifiable: a token's raw teacher preference can reflect superficial continuation quality rather than the causal effect of the intermediate search or summary decision that produced it. This structural gap—shared across the teacher-distillation and multi-agent training literature—means that the frozen Teacher's token scores can reward locally fluent but causally irrelevant intermediate steps, breaking the alignment between external supervision and the student's reasoning state.

## Key Insight

Step-level credit becomes identifiable when it is defined as the contrast between Teacher scores on student continuations that differ only at the current decision, because this contrast isolates the causal effect of that decision from all downstream token-level confounds.

## Method

```pseudocode
Input: frozen Teacher T, student team (Leader, Searcher, Answerer) with shared parameters θ, on-policy trajectory τ = (s_1, a_1, ..., s_K, a_K) where each a_k is a search query or summary.
Output: step-level advantage A_k for each intermediate decision a_k, used in the team's RL update.

For each training question q:
  1. Student team generates on-policy trajectory τ, recording each intermediate decision a_k and its prefix context c_k.
  2. For each decision index k:
     a. Sample M=4 counterfactual continuations {a_k^(1), ..., a_k^(M)} from the student's current policy conditioned on c_k, using temperature 1.0 (these differ ONLY at position k; all earlier decisions are fixed to the on-policy prefix).
     b. For each continuation a_k^(m), form the full suffix by having the frozen Teacher T complete the remaining reasoning from c_k + a_k^(m) with greedy decoding, yielding a completed trajectory τ_k^(m).
     c. Teacher scores the decision token span of a_k^(m) within τ_k^(m): s_k^(m) = mean log-probability assigned by T to the tokens of a_k^(m) given c_k.
     d. Compute the contrastive step advantage: A_k = (1/M) Σ_m s_k^(m) − s_k^(0), where s_k^(0) is the Teacher score of the on-policy decision a_k. Normalize A_k by the empirical standard deviation of {s_k^(m)} across m to control scale.
  3. The team's RL update uses A_k as the advantage for the Searcher's intermediate decision at step k, while the Leader and Answerer continue to use their original answer-correctness rewards.
```

**Assumption block (D):** The contrastive step advantage A_k measures the causal necessity of the intermediate decision a_k because the assumption that continuations differing only at position k isolate the effect of that decision is equivalent to a no-unobserved-confounder condition within the fixed prefix; this assumption fails when the Teacher's score of a decision span is sensitive to the specific suffix completion (e.g., a search query that is good only under one particular follow-up), in which case A_k reflects the interaction between the decision and the Teacher's suffix rather than the decision's standalone causal effect. The empirical standard deviation normalization measures the reliability of the contrast because it assumes that the variance across counterfactuals is a proxy for the noise in the Teacher's score of that decision; this assumption fails when the counterfactuals are highly diverse for reasons unrelated to decision quality (e.g., lexical diversity), in which case the normalization down-weights genuinely important decisions. The use of the frozen Teacher's greedy completion to form the suffix measures the decision's effect under a stable reference policy because it assumes that the Teacher's completion is a sufficient statistic for the downstream context; this assumption fails when the Teacher's completion diverges from the student's distribution in a way that changes the interpretation of the decision, in which case the score reflects the Teacher's own reasoning rather than the student's.

**Why this design**
We chose counterfactual contrastive scoring over raw Teacher token scores (as in MultiAgent-OPD) because raw scores are confounded by the fluency of the entire continuation, whereas contrasting continuations that share a prefix and differ only at decision k cancels all prefix-level and suffix-level effects, leaving a signal that is structurally tied to the decision itself; the cost is that we must sample M=4 continuations per step, increasing Teacher calls by a factor of M. We chose to hold the prefix fixed to the on-policy trajectory rather than resampling earlier decisions because this preserves the student's actual reasoning state at step k, which is the state the credit must be assigned to; the trade-off is that we cannot estimate how errors in earlier steps propagate. We chose a fixed M=4 rather than an adaptive budget because the variance of the contrastive estimate is controlled by the empirical standard deviation normalization, and a fixed budget keeps the per-step cost predictable; the cost is that highly variable decisions may receive noisier advantages. We chose to use the frozen Teacher's greedy completion to form the suffix rather than the student's own continuation because the Teacher's completion provides a stable reference distribution that does not drift with the student's policy; the trade-off is that the Teacher's suffix may be off-policy relative to the student, but since we only score the decision span, the suffix serves only to provide context and its off-policy nature does not enter the contrast. A domain expert would not describe this as a variant of GKD or MultiAgent-OPD because those methods use raw token-level teacher scores as a proxy for step credit, whereas our method constructs a counterfactual intervention at each decision and uses the resulting contrast as the credit signal—a mechanism that neither GKD nor MultiAgent-OPD contains.

## Contribution

['A counterfactual token-scoring protocol that replaces raw Teacher token scores with contrastive step-level advantages computed from student-generated continuations that differ only at a single decision, providing an operational identifiability condition for step-level credit in hierarchical multi-agent distillation.', 'A concrete design principle: step-level credit in teacher-guided multi-agent RL should be defined as a controlled contrast rather than a raw preference, because raw preferences are confounded by continuation-level fluency, and the contrast cancels those confounds.', 'An end-to-end training procedure that integrates the counterfactual step advantage into the existing Leader-Searcher-Answerer on-policy distillation pipeline without modifying the Teacher or the team architecture.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Multi-hop QA (HotpotQA / 2WikiMultiHop) | Requires distinct intermediate search decisions to isolate |
| Primary metric | Macro avg@4 answer accuracy | Detects step-credit gains on multi-step reasoning |
| Baseline | MultiAgent-OPD (raw teacher token scores) | Tests whether contrast beats raw teacher scoring |
| Baseline | C-MAS + RL (answer-correctness only) | Tests whether step credit beats sparse outcome reward |
| Baseline | COMA-style counterfactual baseline | Tests whether our contrast beats standard counterfactual credit |
| Baseline | Contrastive distillation (GKD) | Tests whether our step-level contrast beats token-level contrastive distillation |
| Baseline | MultiAgent-OPD with 4x samples | Cost-matched control for extra teacher calls |
| Ablation-of-ours | Contrast without std normalization | Tests whether normalization guards noisy decisions |
| Ablation-of-ours | Teacher-sampled suffix (temperature 1.0) instead of greedy | Tests sensitivity to suffix distribution |
| Ablation-of-ours | Score under multiple Teacher suffixes (K=3) | Measures decision-suffix interaction |

### Why this setup validates the claim

The central claim is that a counterfactual contrast at each decision isolates the decision's causal contribution better than raw teacher token scores or sparse answer-correctness rewards, and that this improves credit assignment on multi-step search. Multi-hop QA is the right dataset because it forces genuinely distinct intermediate decisions (which entity to search, how to summarize) whose quality is separable from final-answer fluency — exactly the regime where raw teacher scores are confounded. MultiAgent-OPD tests the sub-claim that contrasting cancels prefix/suffix confounds that raw token scoring retains; C-MAS + RL tests the sub-claim that dense step credit beats delayed answer-only reward. The COMA-style baseline tests whether our contrastive step advantage is better than a standard counterfactual baseline that marginalizes over actions; the contrastive distillation (GKD) baseline tests whether our step-level contrast beats token-level contrastive distillation. The cost-matched MultiAgent-OPD with 4x samples controls for the extra teacher calls (M=4 per step) so that any gain is attributable to the contrast mechanism, not to more compute. The ablation without std normalization isolates the normalization mechanism: if normalization is doing real work, removing it should degrade performance specifically on high-variance decisions. The Teacher-sampled suffix ablation tests the assumption that the greedy suffix is a sufficient statistic; if A_k rankings change, the decision-suffix interaction is significant. The multi-suffix scoring ablation directly measures the interaction by computing the variance of s_k^(m) across K=3 different Teacher suffixes; high variance indicates that the decision's score depends on the suffix, violating the no-interaction assumption. Macro avg@4 is the right metric because it rewards consistent correctness across attempts rather than lucky single samples, matching the claim that better step credit yields more reliable multi-step trajectories rather than occasional wins.

### Expected outcome and causal chain

**vs. MultiAgent-OPD (raw teacher token scores)** — On a case where the student emits a fluent but unhelpful search query (e.g., a plausible-sounding entity that does not appear in the corpus), the baseline assigns it a high raw teacher score because the query is linguistically clean and the teacher's continuation is fluent, so the bad decision gets reinforced. Our method instead samples counterfactual queries at that step, has the teacher complete each, and finds the on-policy query scores no better than alternatives, yielding a low or negative contrast. We expect a noticeable accuracy gap concentrated on questions requiring entity disambiguation, with near-parity on single-hop questions where fluent queries are usually correct.

**vs. C-MAS + RL (answer-correctness only)** — On a case where the team reaches the correct answer through a wrong intermediate search (lucky retrieval), the baseline broadcasts positive advantage to every step because only final correctness is observed, reinforcing the wrong search decision. Our method instead computes A_k from the counterfactual contrast at that step, which is near zero or negative for the unhelpful search even though the answer was right, so the wrong step is not reinforced. We expect the gap to grow with trajectory length and to appear as fewer repeated-error patterns on multi-hop subsets, with parity on short single-decision questions.

**vs. COMA-style counterfactual baseline** — COMA computes a counterfactual baseline by marginalizing out an agent's action while holding others fixed, using a learned critic. In our setting, the 'agents' are sequential decisions, and COMA would require a centralized critic that estimates the value of each decision given the prefix. Our method instead uses the frozen Teacher's scores on counterfactual continuations, which provides a direct, model-based counterfactual without learning a critic. We expect our method to outperform COMA when the critic is poorly estimated (early training, high-dimensional action space), but COMA may catch up with sufficient critic training. The key difference is that our counterfactual is grounded in the Teacher's likelihood, while COMA's is grounded in the learned value function.

**vs. Contrastive distillation (GKD)** — GKD uses token-level contrastive loss between student and teacher distributions on student-generated sequences. Our method contrasts at the step level using counterfactual decisions. We expect our method to better isolate the effect of intermediate decisions because GKD's token-level contrast is still confounded by fluency and does not explicitly intervene on the decision. The gap should be largest on decisions with high lexical diversity (e.g., summaries) where token-level contrast is noisy.

**vs. MultiAgent-OPD with 4x samples** — This baseline uses the same number of Teacher forward passes as our method (4 per step) but computes raw teacher scores on 4 samples and averages them. If our gain is due to the contrast mechanism rather than extra compute, we expect our method to outperform this cost-matched baseline, especially on disambiguation-heavy questions. If the cost-matched baseline matches our method, the benefit is simply from more samples, not from contrasting.

**vs. Contrast without std normalization** — On a case where a decision has high lexical diversity among counterfactuals (e.g., open-ended summary phrasing), the unnormalized contrast produces a large-magnitude advantage driven by variance unrelated to decision quality, destabilizing the update. Our normalized version divides by the empirical std across counterfactuals, down-weighting that noisy step. We expect the unnormalized ablation to show higher variance across runs and worse performance specifically on summary-heavy steps, while matching our method on search-query steps with low counterfactual diversity.

**vs. Teacher-sampled suffix** — If the Teacher's greedy suffix is a sufficient statistic, then using a sampled suffix (temperature 1.0) should yield similar A_k rankings and similar performance. If performance degrades, the greedy suffix is not sufficient and the decision-suffix interaction matters. We expect a small degradation, indicating that the greedy suffix is a reasonable but imperfect approximation.

**vs. Multi-suffix scoring (K=3)** — This ablation measures the decision-suffix interaction directly. For each decision, we compute s_k^(m) under K=3 different Teacher suffixes (greedy and two sampled) and compute the variance across suffixes. If the variance is high, the decision's score is sensitive to the suffix, violating the no-interaction assumption. We expect the variance to be higher for summary decisions than for search queries, and that high-variance decisions receive noisier advantages. This ablation does not change the training procedure; it only diagnoses the assumption.

### What would falsify this idea

If our gain over MultiAgent-OPD is uniform across single-hop and multi-hop subsets rather than concentrated where raw teacher scores are confounded (disambiguation-heavy, long-trajectory questions), the claim that contrast cancels confounds is wrong. Likewise, if the std-normalization ablation matches our method everywhere, the normalization mechanism is not doing the work we attribute to it. If the multi-suffix scoring ablation shows that A_k rankings are highly sensitive to the Teacher suffix (high variance across suffixes), then the decision-suffix interaction is significant and the contrast does not isolate the decision's causal effect.

## References

1. MultiAgent-OPD: End-to-End Training of Hierarchical Multi-Agent Systems with On-Policy Distillation
2. On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes
3. Explanations from Large Language Models Make Small Reasoners Better
