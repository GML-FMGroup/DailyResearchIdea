# Rollback-Identifiable Value Learning for Multimodal Reasoning Chains

## Motivation

Sequence-level scalar rewards over monolithic autoregressive chains leave the token-to-outcome mapping unidentifiable: VL-Rethinker optimizes a GRPO final-answer reward and DistilDoc optimizes an accuracy-plus-token-efficiency reward, so neither can localize the first erroneous reasoning token and neither can define a causally validated rollback point. Any step-level value learned under such rewards is confounded by downstream tokens, because the counterfactual outcome of truncating at a step is never observed. This is the recurring structural gap: credit assignment to intermediate multimodal decisions is impossible without an explicit intervention that makes the counterfactual outcome observable.

## Key Insight

Truncating the chain at token t and re-decoding the suffix yields a counterfactual outcome whose difference from the factual outcome is a clean causal effect of tokens at and after t, so constraining the per-token advantage gradient to be zero except at the single token whose removal flips the answer forces the value function to concentrate credit at the earliest causally necessary token rather than spreading it across the chain.

## Method

**What it is.** We introduce RIVaL (Rollback-Identifiable Value Learning), which trains a token-level value head V_phi over a vision-language reasoning chain together with a counterfactual truncation operator and a contrastive identifiability loss. Inputs: a multimodal question (image, text) and a sampled reasoning chain of tokens t_1..T with a final answer. Outputs: a per-token advantage a_1..a_T that is localized to the earliest error token, and a rollback index r* at which the chain is truncated and revised.

**How it works.**

```pseudocode
Input: VLM policy pi_theta, value head V_phi (2-layer MLP, hidden=256, GeLU), image x, question q, chain c=(t_1..T), answer y
Hyperparams: truncation budget K=8, re-decodes per truncation n=5, sparsity weight lambda=0.1, temperature tau=0.5, identifiability margin m=0.3, flip-probability threshold p_thr=0.8, confidence level 0.95 (Wilson interval)

Phase 1 - Factual rollout:
  c, y = pi_theta(x, q)                      # standard autoregressive chain, temperature tau=0.5
  v_t = V_phi(x, q, t_1..t_t) for all t       # per-token value predictions

Phase 2 - Counterfactual truncation (intervention, variance-controlled):
  for each candidate index k in a sampled subset S of {1..T} (|S|=K):
      c_k = truncate(c, k)                    # keep prefix t_1..t_k
      for j in 1..n:
          y_k^{(j)} = pi_theta(x, q, c_k)     # re-decode suffix from prefix, temperature tau=0.5
      p_flip(k) = (1/n) * sum_j 1[y_k^{(j)} != y]   # empirical flip probability
      CI(k) = Wilson_interval(p_flip(k), n, confidence=0.95)
  # Delta_k is now a calibrated probability, not a single-sample indicator

Phase 3 - Causal credit target (statistical criterion):
  r* = min{ k in S : p_flip(k) >= p_thr AND lower_bound(CI(k)) > 0.5 }
  # soft target: target_t = p_flip(t) / sum_{k in S} p_flip(k) for t in S, else 0
  # (one-hot fallback if no k meets criterion: r* = argmax_k p_flip(k))

Phase 4 - Contrastive identifiability loss:
  L_value = sum_t BCE(v_t, target_t)          # fit value to soft causal target
  L_grad  = sum_{t != r*} || d v_t / d theta ||^2   # zero gradient off the rollback token
  L_contrast = max(0, m - (v_{r*} - max_{t != r*} v_t))  # margin: rollback token dominates
  L_total = L_value + lambda * L_grad + L_contrast

Phase 5 - Truncate-and-revise:
  r_hat = argmax_t v_t                        # predicted rollback point
  c_revised = pi_theta(x, q, truncate(c, r_hat))  # regenerate suffix from r_hat
  return c_revised, r_hat
```

Phase 2 is the only place a counterfactual outcome is observed: the truncation operator is an explicit intervention on the chain, not a heuristic guess. The n=5 re-decodes per truncation point turn the single-sample flip indicator into a calibrated flip probability p_flip(k) with a Wilson confidence interval, so that Delta_k reflects causal necessity rather than decode variance. Phase 4's gradient-sparsity term enforces that the value function's gradient is zero on all tokens except the one whose removal flips the answer, which is what makes the per-step advantage identifiable rather than a downstream-confounded scalar.

**Why this design.** We chose an explicit truncation-and-redecode operator over a learned step-level reward model (as in process reward models / VinePPO-style Monte-Carlo prefix rewards) because only an intervention makes the counterfactual outcome observable; a reward model trained on factual chains cannot distinguish a token that caused the error from a token that merely preceded it, accepting the cost that each training step requires K*n extra decodes (K=8, n=5). We chose a sparse credit target over a soft distribution because the identifiability claim requires the advantage to concentrate at a single token, accepting the cost that the method cannot represent multi-token joint errors; the soft target derived from p_flip(k) is a variance-reduction of that sparse target, not a departure from it. We chose a contrastive margin loss over plain regression because the rollback token must dominate all others by a margin for truncate-and-revise to be stable, accepting the cost of one extra hyperparameter m. A domain expert would not describe this as a variant of VinePPO or process reward models because those estimate per-step values from factual Monte-Carlo returns, whereas RIVaL's value is fit to a counterfactual flip probability produced by an intervention the prior methods never perform.

**Why it measures what we claim.** The computational quantity p_flip(k) (Monte-Carlo estimate of counterfactual flip probability) measures causal necessity of the prefix up to token k because the truncation operator is a valid intervention on the chain, assuming re-decoding from a truncated prefix is a faithful counterfactual and that n=5 samples suffice to estimate the flip probability with a Wilson interval of width < 0.2 at p=0.5; this assumption fails when the policy's re-decode is stochastic enough that a flip reflects sampling noise rather than causal dependence, in which case p_flip(k) reflects decode variance instead of token necessity. The quantity target_t (soft credit at r*) measures earliest-error localization because r* is defined as the minimum truncation index whose flip probability exceeds p_thr with a confidence interval excluding 0.5, assuming the flip set is downward-closed in k; this assumption fails when an intermediate truncation accidentally flips the answer back, in which case r* reflects the first accidental flip rather than the true error token. The quantity v_t (per-token value) measures per-step advantage because the gradient-sparsity constraint forces v_t to be zero off r*, assuming the BCE fit converges to the soft target; this assumption fails under residual label noise in p_flip(k), in which case v_t reflects smoothed credit rather than a sharp rollback point.

## Contribution

(1) A rollback-identifiable value learning framework that couples a counterfactual truncation operator with a contrastive identifiability loss to produce per-token advantages localized to the earliest error token. (2) A truncate-and-revise decoding procedure that uses the learned per-token value to select a causally validated rollback index rather than a heuristic or sequence-level reweighting. (3) A design principle showing that gradient sparsity on the value head is what makes per-step credit assignment identifiable under outcome-only rewards, with the truncation operator supplying the counterfactual supervision that outcome rewards lack.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | MathVista (multi-step visual reasoning) | Chains long enough to contain localizable errors |
| Primary metric | Rollback-hit rate (r_hat == r*) | Directly tests localization claim, not just accuracy |
| Secondary metric | Final-answer accuracy | End-to-end consequence of correct rollback |
| Secondary metric | Compute cost (K*n extra decodes per step, wall-clock GPU-hours) | Measures accuracy-per-FLOP gain |
| Baseline | VL-Rethinker (GRPO + SSR) | Tests whether intervention beats outcome-only RL |
| Baseline | VinePPO-style prefix Monte-Carlo | Tests whether causal intervention beats factual prefix returns |
| Baseline | Vanilla VLM self-correction (no value head) | Tests whether credit target matters at all |
| Baseline | Truncation + soft target (no one-hot) | Tests whether sparse-identifiability construction is the novel claim |
| Ablation-of-ours | RIVaL minus contrastive margin (L_contrast=0) | Isolates the margin's role in stable rollback |
| Ablation-of-ours | RIVaL with n=1 (single-sample Delta_k) | Quantifies decode-variance contamination |
| Diagnostic | Fraction of chains with non-monotone flip set | Empirically tests downward-closure assumption |

### Why this setup validates the claim

The central claim is that a value head fit to a counterfactual flip probability localizes the earliest error token better than methods that only observe factual chains. MathVista is the right testbed because its chains are long enough that a single early error propagates to a wrong answer, giving a well-defined r* for each chain. Rollback-hit rate is the metric of interest because it measures localization directly rather than confounding it with downstream answer accuracy; accuracy and compute cost are secondary consequences we expect to follow. VL-Rethinker tests the outcome-only sub-claim: if intervention adds nothing beyond outcome RL, its localization should match ours. The VinePPO-style prefix baseline tests the causal sub-claim: if factual Monte-Carlo prefix returns suffice, our flip-based target is redundant. The no-value-head baseline tests whether a learned credit target is necessary at all. The truncation + soft target baseline tests whether the sparse-identifiability construction (one-hot / margin) is the novel claim rather than the intervention itself. The n=1 ablation quantifies how much of our gain comes from variance control. The non-monotone diagnostic directly measures the downward-closure assumption on a held-out set of 512 chains.

### Expected outcome and causal chain

**vs. VL-Rethinker (GRPO + SSR)** — On a case where the chain makes a visual grounding slip at token 4 (misreads a chart axis) and then reasons correctly, VL-Rethinker's outcome-only reward assigns uniform advantage across the whole chain because it only sees the final answer is wrong; SSR rethinks the whole chain rather than the faulty step. Our method instead re-decodes from each truncation point (n=5 samples), observes that truncating at token 4 flips the answer with p_flip >= 0.8 while truncating at 5 does not, and concentrates credit at token 4. We expect a noticeable gap in rollback-hit rate on chains whose errors occur early (first third of tokens) but near-parity on chains with late errors, where outcome-only credit is already roughly correct. We also expect a positive accuracy-per-FLOP gain despite the K*n decode overhead.

**vs. VinePPO-style prefix Monte-Carlo** — On a case where token 6 is a harmless restatement and token 4 is the actual error, prefix Monte-Carlo assigns token 6 a high prefix value because factual continuations from token 6 still reach the right answer; it cannot distinguish "preceded the error" from "caused the error." Our method instead uses the truncation intervention, so token 6's truncation preserves the answer (p_flip near 0) and receives no credit, while token 4's truncation flips it with high probability. We expect our rollback index to land on the true error token far more often on chains containing redundant restatement steps, with the gap shrinking on chains with no such distractors.

**vs. Vanilla VLM self-correction (no value head)** — On a case where the chain fails at token 3 but the model's self-correction prompt regenerates from the start, the vanilla baseline produces a fresh chain that may repeat the same grounding error because it has no signal about which step was faulty. Our method instead truncates at the predicted r_hat and re-decodes only the suffix, preserving correct early reasoning and forcing revision of the faulty step. We expect a larger accuracy gain on chains where the early prefix is correct and the error is localized, and near-parity on chains that are wrong from the first token.

**vs. Truncation + soft target (no one-hot)** — On a case where two adjacent tokens both receive moderate p_flip due to residual decode variance, the soft-target baseline spreads credit across both, so argmax r_hat is less sharp and truncate-and-revise sometimes rolls back one token too late. Our method's one-hot / margin construction forces the rollback token to dominate, stabilizing argmax. We expect the soft-target baseline to show lower rollback-hit rate specifically on chains where the flip set is not cleanly downward-closed, confirming that the sparse-identifiability construction is the novel claim.

**vs. RIVaL minus contrastive margin** — On a case where two adjacent tokens both receive nonzero BCE signal due to decode variance in p_flip(k), the ablation's value head spreads credit across both, so argmax r_hat is unstable across samples and truncate-and-revise sometimes rolls back one token too late. Our method's margin forces the rollback token to dominate, stabilizing argmax. We expect the ablation to show higher variance in r_hat across repeated decodes and lower rollback-hit rate specifically on chains where the flip set is not cleanly downward-closed.

**vs. RIVaL with n=1 (single-sample Delta_k)** — On a case where a single re-decode at an early truncation point accidentally recovers the correct answer, the n=1 variant sets r* to that noisy flip and confidently trains the value head on it. Our method's n=5 Monte-Carlo estimate with Wilson interval suppresses such spurious flips. We expect the n=1 variant to show substantially lower rollback-hit rate and higher variance in r_hat, quantifying the decode-variance contamination.

### What would falsify this idea

If RIVaL's rollback-hit rate advantage over the VinePPO-style prefix baseline is uniform across all chain positions rather than concentrated on chains with early errors and redundant distractor steps, the causal-intervention mechanism is not doing the work and the gain is attributable to something else. If the no-value-head baseline matches RIVaL's accuracy after truncate-and-revise, the learned credit target is not necessary and the central identifiability claim is unsupported. If the non-monotone flip-set fraction is high (>30%) and the n=5 estimator does not recover r* on those chains, the downward-closure assumption is violated and the localization claim is weakened.

## References

1. VL-Rethinker: Incentivizing Self-Reflection of Vision-Language Models with Reinforcement Learning
2. Distildoc: Deep Reinforcement Learning for Token-Efficient Multimodal Document Question Answering
