# Calibrated Difficulty Intervals for Adaptive Dual-Process Reasoning

## Motivation

Existing dual-process reasoning methods, such as ACPO (Incentivizing Dual Process Thinking for Efficient Large Language Model Reasoning), rely on a point-estimate difficulty scalar to drive mode-switch and token-budget allocation. This scalar is uncalibrated and never revised during reasoning, so the mapping from query features to difficulty is unidentifiable and decisions cannot be audited or corrected mid-trajectory. Consequently, the assumed monotone benefit of System-2 mode fails when the difficulty estimate is wrong, leading to wasted computation or insufficient reasoning.

## Key Insight

A sequentially-revised conformal difficulty interval with finite-sample coverage guarantees makes the mode-switch and budget decisions statistically valid and correctable mid-trajectory, because the interval width directly quantifies uncertainty and its lower bound provides a calibrated signal for when to escalate.

## Method

```pseudocode
# Algorithm: Calibrated Difficulty Intervals for Dual-Process Reasoning (CDI-DPR)
# Hyperparameters: alpha=0.1 (miscoverage rate), tau (mode-switch threshold for lower bound; set to the 70th percentile of calibration partial-score lower bounds), beta=2.0 (budget expansion factor), window_size=8 (number of recent partial scores for exchangeability test), chunk_size c=16 tokens

Input: query x, base LLM M, initial token budget B0=256, calibration set D_cal of (query, difficulty score) pairs
Output: reasoning trace, final answer

1. Compute initial difficulty score s0 = f(x) using a lightweight scorer (perplexity of x under M).
2. Initialize conformal interval C0 = [L0, U0] from split-conformal calibration on D_cal with miscoverage alpha=0.1. **Note:** This interval is for f(x) scores only and is NOT used for escalation decisions; it serves as a reference for the initial mode (System-1) and budget (B0).
3. Set current mode = System-1 (fast), budget = B0.
4. Generate reasoning trace in chunks of size c=16 tokens. After each chunk, compute partial difficulty score s_partial = g(partial trace, x), where g is defined as the negative log-likelihood of the next token under M given the partial trace and x, normalized by trace length.
5. Maintain a sliding window of the last window_size=8 partial scores. Perform an online exchangeability test (Kolmogorov-Smirnov test) comparing the window to a held-out reference set of partial scores (see calibration below). If the test rejects exchangeability at level alpha=0.1, update the conformal interval using online conformal prediction (ACI) with the new partial scores, yielding C_t = [L_t, U_t]. **Calibration for partial scores:** Build a separate calibration set D_cal_partial by taking held-out reasoning traces (from the training split of each dataset) and computing g on truncated prefixes of length c, 2c, ..., up to the full trace, matching the chunking and window used at inference. Use these partial scores to compute the initial ACI quantile and the reference distribution for the KS test. The ACI update uses the residual of s_partial against the reference mean, with learning rate gamma=0.05.
6. If L_t > tau and mode == System-1, switch to System-2 (slow) mode, expand budget to beta * B0 = 512 tokens, and continue generation with System-2 prompting.
7. If mode == System-2 and L_t <= tau for a sustained period (last 3 intervals), revert to System-1 and shrink budget to B0.
8. Stop when budget exhausted or answer produced.
```

**Why this design**
We chose a sequentially-revised conformal interval over a point-estimate difficulty scalar because point estimates provide no uncertainty quantification, making mode-switch decisions unauditable; the cost is increased computational overhead from maintaining intervals and running exchangeability tests. We chose an online exchangeability test to trigger interval revision rather than fixed periodic updates because reasoning traces are non-stationary, and a test-based trigger ensures revisions occur only when the partial scores distribution shifts, accepting the risk of false positives that could cause unnecessary escalations. We chose to use the lower bound of the interval as the mode-switch signal rather than the point estimate because the lower bound provides a conservative guarantee: switching only when we are confident the difficulty is high, which reduces wasteful System-2 invocations at the cost of potentially missing some hard cases. We chose to expand budget multiplicatively (beta) rather than additively because difficulty scales superlinearly with reasoning steps in many tasks, and a multiplicative factor maintains proportional allocation. These decisions distinguish our method from ACPO's uncalibrated scalar and from standard conformal prediction applied statically; a domain expert would not describe our method as a variant of ACPO because we replace the scalar with a calibrated interval and introduce online revision via exchangeability testing, which ACPO lacks entirely.

**Why it measures what we claim**
The computational quantity `s_partial` (partial difficulty score) measures the motivation-level concept of `difficulty` because we assume that the difficulty score function g, computed on partial reasoning traces, is monotonically related to the true difficulty of completing the reasoning; this assumption fails when partial traces are misleading (e.g., early errors that later self-correct), in which case s_partial reflects transient local difficulty rather than global difficulty. The conformal interval width measures `uncertainty` because we assume exchangeability between calibration and test scores; this assumption fails under distribution shift (e.g., novel query types), in which case the interval may undercover and the width underestimates true uncertainty. The lower bound L_t measures `confidence that difficulty exceeds threshold` because we assume the conformal guarantee holds conditionally on the exchangeability test; this assumption fails when the test has low power, in which case L_t may be too high and trigger premature escalation. The online exchangeability test measures `distribution shift` because we assume the test statistic is sensitive to changes in the difficulty score distribution; this assumption fails when shifts are subtle or gradual, in which case revisions may be delayed. **Explicit load-bearing assumption:** The partial-trace difficulty score g(partial trace, x) is a monotone, exchangeability-preserving proxy for the true completion difficulty, so that a conformal interval built on calibration scores of f(x) remains valid when applied to online partial scores s_partial and its lower bound L_t is a calibrated signal for escalation. This assumption is addressed by recalibrating on the actual score being thresholded: we build the conformal calibration set from g(partial trace, x) evaluated on truncated prefixes of held-out reasoning traces (matched to the chunking and window used at inference), and we replace the KS-triggered ACI with a coverage-tracking update (ACI on the partial-score residual against a held-out reference) so the guarantee is stated for the partial-score sequence rather than inherited from f(x).

## Contribution

['We introduce CDI-DPR, a framework that replaces the point-estimate difficulty scalar in dual-process reasoning with a sequentially-revised conformal difficulty interval, where the interval width itself drives mode-switch and budget allocation decisions.', 'We design an online exchangeability test on partial reasoning traces that triggers statistically valid mid-trajectory revisions of the difficulty interval, enabling auditable and correctable mode-switch decisions.', 'We provide a concrete algorithm that integrates split-conformal calibration with online conformal prediction and a lower-bound thresholding rule, and we empirically validate its effectiveness in reducing redundant reasoning while maintaining accuracy on math and reasoning benchmarks.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|------------------------|
| Dataset | GSM8K, MATH500, GPQA, AIME | Difficulty spectrum exposes over/under-thinking |
| Primary metric | Accuracy at fixed average token budget (B0=256 for System-1, beta*B0=512 for System-2) | Directly measures efficiency-accuracy tradeoff |
| Secondary metric | Empirical miscoverage rate of L_t on shifted subsets (target ≤ alpha=0.1) | Verifies conditional coverage claim |
| Baseline | ACPO | Uncalibrated scalar mode-switch comparator |
| Baseline | Fixed System-2 (always slow) | Upper-bound accuracy, ignores efficiency |
| Baseline | Static conformal point-estimate switch | Tests interval vs point uncertainty |
| Baseline | MC-dropout uncertainty switch (5 samples) | Tests conformal vs ensemble-variance signal |
| Ablation-of-ours | CDI-DPR without online ACI revision | Isolates exchangeability-test contribution |
| Ablation-of-ours | CDI-DPR with no-op exchangeability test (always update) | Isolates test-trigger contribution |

### Why this setup validates the claim

The central claim is that calibrated difficulty intervals with online revision yield better accuracy-per-token than uncalibrated scalar switching or static conformal switching, especially under distribution shift. The dataset suite spans easy (GSM8K), medium (MATH500), and hard (GPQA, AIME) regimes, so we can check whether gains concentrate where the mechanism predicts: on queries whose partial-trace difficulty is high but uncertain, and on subsets where score distributions shift mid-trace. ACPO tests the sub-claim that calibration matters: if our interval lower bound reduces wasteful escalations, we should beat ACPO on token efficiency at matched accuracy. Fixed System-2 tests the sub-claim that adaptive budgeting is necessary: if our method merely matches always-slow accuracy at lower cost, the efficiency claim holds. Static conformal tests whether online revision is essential: if static intervals already capture the benefit, the exchangeability test is superfluous. The MC-dropout baseline tests whether conformal intervals provide a distinct, non-redundant signal over ensemble-variance uncertainty; if they do not, the contribution is an empirical calibration study rather than a novel mechanism. The ablation isolates the revision trigger. Accuracy at a fixed average token budget is the right metric because it forces both over- and under-thinking to show up as accuracy loss, making the tradeoff falsifiable rather than hidden behind separate efficiency and accuracy numbers. The empirical miscoverage rate of L_t on shifted subsets directly verifies the conditional-coverage claim: if the observed miscoverage exceeds alpha=0.1 significantly, the interval is not calibrated for the partial-score sequence, falsifying the load-bearing assumption.

### Expected outcome and causal chain

**vs. ACPO** — On a GSM8K problem with a short but superficially complex prompt (e.g., nested arithmetic), ACPO's uncalibrated scalar may exceed its switch threshold and invoke System-2, producing a long redundant trace at high token cost. Our method's conformal lower bound stays below tau because calibration shows such prompts rarely require slow mode, so we remain in System-1 and answer efficiently. We expect a noticeable token reduction on easy subsets with accuracy parity, and a smaller but positive accuracy gain on hard subsets where ACPO's scalar under-triggers.

**vs. Fixed System-2 (always slow)** — On an AIME problem requiring multi-step derivation, fixed System-2 achieves high accuracy but spends the full expanded budget on every query, including trivial ones. Our method escalates only when the lower bound exceeds tau, so it matches System-2 accuracy on hard instances while spending far fewer tokens on easy ones. We expect near-identical accuracy on AIME/GPQA but a large average-token gap driven by GSM8K/MATH500, where most queries stay in System-1.

**vs. Static conformal point-estimate switch** — On a MATH500 problem whose partial trace initially looks easy but becomes hard after a misleading simplification, the static conformal method never revises its interval and remains in System-1, producing an incorrect final answer. Our online exchangeability test detects the distribution shift in partial scores, ACI widens and raises the lower bound above tau, and we escalate to System-2 mid-trace. We expect a concentrated accuracy gain on the subset of problems with mid-trace difficulty shifts, with parity elsewhere.

**vs. MC-dropout uncertainty switch** — On a GPQA problem with high epistemic uncertainty, MC-dropout may produce a wide variance interval that triggers escalation even when the difficulty is moderate, because dropout variance conflates model uncertainty with difficulty. Our conformal interval, calibrated on partial scores, provides a tighter lower bound that stays below tau for moderate-difficulty queries, avoiding unnecessary System-2 invocations. We expect our method to achieve higher token efficiency at matched accuracy, and to show lower empirical miscoverage on shifted subsets, demonstrating that conformal intervals provide a distinct signal.

### What would falsify this idea

If our method's accuracy gain over ACPO and static conformal is uniform across easy and hard subsets rather than concentrated on queries with high partial-score uncertainty or detected distribution shift, then the claimed causal mechanism (calibrated intervals plus online revision) is not what drives performance. Similarly, if the ablation without ACI revision matches our full method, the exchangeability-test trigger is not doing the work we claim. Additionally, if the empirical miscoverage rate of L_t on shifted subsets exceeds alpha=0.1 significantly, the conditional-coverage claim is falsified.

## References

1. Incentivizing Dual Process Thinking for Efficient Large Language Model Reasoning
2. Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters
3. Do NOT Think That Much for 2+3=? On the Overthinking of o1-Like LLMs
4. Beyond Chinchilla-Optimal: Accounting for Inference in Language Model Scaling Laws
5. Alphazero-like Tree-Search can Guide Large Language Model Decoding and Training
6. Broken Neural Scaling Laws
7. Solving Math Word Problems via Cooperative Reasoning induced Language Models
8. Solving Quantitative Reasoning Problems with Language Models
