# Counterfactual Evidence Attribution Certificates for Token-Efficient Multimodal RL

## Motivation

Token-efficient multimodal RL such as DistilDoc optimizes a single scalar reward that mixes answer correctness with compression ratio, which makes the reward unidentifiable at the token level: two rollouts can receive identical scalar reward while one compressed away a load-bearing evidence token and the other compressed away a redundant one, so the policy receives no gradient distinguishing them. The field's trajectory toward ever-finer compression therefore hits a structural bottleneck — efficiency and reasoning quality become non-separable because no signal attributes load-bearing versus redundant compressed-away tokens, and VL-Rethinker-style self-reflection cannot repair this since reflection operates on the answer, not on the attribution of dropped evidence.

## Key Insight

A frozen answer head re-run on restored-token variants yields a counterfactual sufficiency label per compressed-away token that is independent of the compression ratio, so agreement between the policy's emitted attribution and this probe is an identifiable, ratio-invariant training signal that a scalar correctness-plus-efficiency reward structurally cannot provide.

## Method

**What it is.** CECA (Counterfactual Evidence Certificate Attribution) is an RL fine-tuning procedure for a compression policy on multimodal document QA. Inputs: a multimodal document (text chunks, table rows, image captions), a question, and a frozen answer head. Outputs: a compressed context plus, for every token the policy compresses away, a scalar counterfactual sufficiency score in [0,1]; the policy is trained on the agreement between these scores and a held-out counterfactual probe.

**How it works.**

```pseudocode
# Phase 1: rollout
context = document_elements            # list of (modality, tokens)
for step in range(T_max=len(context)):
    action = policy.sample(element_i, state)   # KEEP | COMPRESS
    if action == COMPRESS:
        s_hat[element_i] = policy.attribution_head(state, element_i)  # in [0,1]
compressed = [e for e in context if action[e]==KEEP]
answer = frozen_answer_head(compressed, question)
correct = EM(answer, gold)

# Phase 2: held-out counterfactual probe (cheap, no full rollout)
# Sample a probe subset S of compressed-away elements, |S| = min(8, |dropped|)
for e in S:
    variant = compressed + [e]                 # restore exactly one dropped element
    a_e = frozen_answer_head(variant, question)
    suff[e] = 1.0 if EM(a_e, gold) > EM(answer, gold) else 0.0
# suff[e] = 1 means e was load-bearing; 0 means redundant

# Phase 3: certificate reward
agreement = mean_{e in S} [ 1 - (s_hat[e] - suff[e])^2 ]   # Brier-style
r_ceca = correct * ( lambda_ans + lambda_cert * agreement )
# lambda_ans = 1.0, lambda_cert = 0.5, no token-ratio term
# (compression is enforced by a fixed budget B on KEEP count, not by reward shaping)

# Phase 4: policy gradient update (PPO)
loss = -E[ r_ceca * log pi(action|state) ] + beta * KL(pi || pi_ref)   # beta = 0.01
```

**Why this design.** We chose a held-out counterfactual probe over a learned independence head because the probe directly intervenes on the compressed context and observes the frozen answer head's response, whereas a learned head would assert separability the data does not contain (the exact failure anti-pattern 1 names); the cost is that probe calls are extra forward passes, bounded to |S| ≤ 8 per rollout. We chose to remove the token-ratio term from the reward and enforce compression via a hard budget B on KEEP count, because keeping ratio in the reward is what makes the scalar unidentifiable — two rollouts with the same ratio can have opposite attribution structures; the cost is that the policy cannot trade ratio against accuracy smoothly and must be run at several budgets. We chose a Brier-style squared-error agreement over a ranking loss because the probe emits binary sufficiency labels, and squared error is a proper scoring rule whose optimum is the true conditional sufficiency probability; the cost is that it is sensitive to the probe's label noise on ambiguous restorations. We chose to sample the probe subset S uniformly at random over dropped elements rather than over the policy's highest-attribution elements, because sampling by the policy's own scores would let it game the signal; the cost is higher variance in the per-rollout agreement estimate. A domain expert would not describe CECA as a variant of DistilDoc: DistilDoc's reward is a scalar over (correctness, ratio) with no per-token attribution surface, and CECA's reward is a function of a counterfactual certificate that DistilDoc never computes.

**Why this measures what we claim.** The quantity `suff[e]` measures load-bearing evidence status of a compressed-away element because restoring e and observing an exact-match improvement under the frozen answer head is equivalent to e being causally necessary for the answer under the assumption that the answer head is deterministic and the document elements are non-interacting in their effect on the answer; this assumption fails when two dropped elements are jointly sufficient but individually redundant (synergy), in which case suff[e]=0 reflects marginal rather than joint necessity. The quantity `agreement` measures the policy's attribution faithfulness because squared error between a probabilistic attribution and a binary counterfactual label is a proper scoring rule whose expectation is minimized iff s_hat equals the true conditional sufficiency probability, under the assumption that the probe subset S is drawn uniformly from the dropped set; this assumption fails when the dropped set is dominated by one modality (e.g., all images dropped), in which case agreement reflects calibration on that modality only. The quantity `r_ceca` measures the joint objective of answer correctness weighted by attribution faithfulness because the product structure makes attribution agreement contribute only when the answer is correct, under the assumption that a correct answer with faithful attribution is the target behavior; this assumption fails when the policy finds a correct answer via a shortcut that the probe also labels as sufficient, in which case r_ceca rewards a spurious but probe-consistent attribution.

## Contribution

(1) A counterfactual evidence-attribution certificate mechanism that replaces the scalar correctness-plus-efficiency reward in token-efficient multimodal RL with a ratio-invariant, per-token agreement signal checked against a frozen answer head's restoration probe. (2) A design principle — hard KEEP budget plus certificate reward — that separates efficiency from reasoning quality at the token level, showing that compression ratio need not appear in the reward for compression to occur. (3) A probe-subset sampling scheme with a proper-scoring-rule agreement objective that is cheap (≤8 frozen-head forward passes per rollout) and resistant to policy self-gaming.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Multimodal document QA with tables, captions, text (e.g., DocVQA-style) | Exercises heterogeneous elements where attribution is contentious |
| Primary metric | Answer EM at fixed KEEP budget B, stratified by dropped-modality composition | Detects whether faithfulness gain tracks attribution, not raw accuracy |
| Baseline | DistilDoc (RL compression, scalar correctness+ratio reward) | Tests claim that per-token certificates beat ratio-shaped rewards |
| Baseline | LeanContext (text-only RL compression) | Tests claim that multimodal attribution matters beyond text |
| Baseline | No-compression full-context answer head | Establishes the accuracy ceiling the budget must approach |
| Ablation-of-ours | CECA with random attribution head (agreement term decoupled from s_hat) | Isolates whether the certificate signal, not just RL, drives gains |

### Why this setup validates the claim

The central claim is that a per-token counterfactual certificate makes the compression policy's attribution surface identifiable, and that this identifiability — not merely RL training or a token-ratio penalty — is what produces faithful, budget-respecting compression. The design tests this by holding the KEEP budget B fixed across all methods, so any accuracy difference cannot be explained by trading ratio against accuracy. DistilDoc tests the sub-claim that a scalar (correctness, ratio) reward is insufficient: if it matches CECA at fixed B, the certificate is doing no work. LeanContext tests whether multimodality is load-bearing: if a text-only compressor matches CECA, the multimodal attribution surface is unnecessary. The no-compression baseline fixes the ceiling, so we can see how much of the achievable accuracy the budget preserves. The random-attribution ablation is the sharpest test: it keeps the RL loop, the budget, and the frozen answer head identical but severs the certificate from the policy's own scores. If CECA's gain survives this ablation, the gain is from RL fine-tuning, not from attribution faithfulness — which would refute the mechanism claim. Stratifying the primary metric by which modality dominates the dropped set is essential because the predicted effect is localized: CECA should help precisely where dropped elements are heterogeneous and individually ambiguous, and should be near-parity where the dropped set is trivially redundant. A uniform gain across strata would be inconsistent with the causal story.

### Expected outcome and causal chain

**vs. DistilDoc** — On a case where two table rows are individually redundant but jointly necessary (synergy), DistilDoc's scalar reward cannot distinguish a rollout that drops both from one that drops one, because its reward is a function of (correctness, ratio) with no per-element surface, so it converges to dropping both and losing the answer. CECA instead probes each dropped row, observes that restoring either alone does not recover EM (suff=0 for both), and its Brier agreement penalizes a high s_hat on either, pushing the policy to keep at least one. We expect a noticeable EM gap on synergy-heavy strata and near-parity on strata where dropped elements are individually decisive.

**vs. LeanContext** — On a case where the answer requires reading a table cell whose content is echoed in a caption, LeanContext's text-only view sees the caption as sufficient and drops the table, producing a wrong numeric answer because the caption paraphrases rather than states the value. CECA's probe restores the table row against the frozen multimodal answer head, observes EM improvement, and assigns suff=1, so its agreement term forces s_hat high on that row and the policy learns to keep it. We expect the gap to concentrate on table-plus-caption instances and vanish on text-only instances.

**vs. No-compression full-context** — On a case where the full context contains a distractor paragraph that the answer head latches onto, the no-compression baseline answers from the distractor and errs, because it has no mechanism to remove evidence. CECA at budget B drops the distractor, and its probe confirms suff=0 for it, so the certificate is consistent with dropping it; the compressed context steers the frozen head to the load-bearing element. We expect CECA to match or slightly exceed the no-compression ceiling on distractor-heavy instances while trailing it on instances requiring many jointly necessary elements that exceed B.

**vs. CECA with random attribution head** — On a case where the policy's s_hat is anti-correlated with true sufficiency (high score on redundant elements), the random-head ablation gives the same expected agreement as CECA because the head carries no signal, so its reward is uninformative about which elements to keep. CECA's head, trained by the Brier term, learns to rank load-bearing elements higher, so its KEEP decisions concentrate on them. We expect CECA to outperform the ablation specifically on instances with heterogeneous element importance and to tie on instances where all elements are equally load-bearing.

### What would falsify this idea

If CECA's gain over DistilDoc at fixed budget B is uniform across modality-composition strata rather than concentrated on synergy-heavy and table-plus-caption instances, then the certificate is not doing the causal work we claim and the improvement is attributable to generic RL fine-tuning. Equally, if the random-attribution ablation matches CECA on those same strata, the attribution-faithfulness mechanism is refuted.

## References

1. Distildoc: Deep Reinforcement Learning for Token-Efficient Multimodal Document Question Answering
2. VL-Rethinker: Incentivizing Self-Reflection of Vision-Language Models with Reinforcement Learning
