# Stepwise Evidence Credit Assignment for Zero-Data Long-Context RL

## Motivation

Existing zero-data long-context RL methods such as LongZero train with GRPO on self-generated QA-evidence triples but verify only the final answer, so the reward signal provides no credit assignment over the intermediate retrieval and evidence-integration steps that the long multi-document context is meant to train. This structural gap means the policy receives an identical scalar reward regardless of whether it retrieved the correct evidence spans in a sound order or stumbled onto the answer through spurious correlations, and the same limitation recurs in LoongRL's final-answer-only verification and R-Zero's outcome-only Solver reward. The field's trajectory toward longer contexts and multi-step retrieval makes process-level verification the binding bottleneck: without step-wise credit, RL cannot distinguish a correct retrieval plan from a lucky guess.

## Key Insight

A verifier that checks each intermediate evidence-selection decision against the self-generated evidence triple's known ground-truth spans yields step-level advantages that are structurally aligned with the final-answer advantage because the triple's evidence is the unique minimal sufficient set for the answer, making step-level and outcome-level correctness commensurable without a learned process reward model.

## Method

```pseudocode
# Input: base LLM policy π, self-generated structured task T = (q, a, E*)
#   q: question, a: answer, E*: ordered ground-truth evidence spans (from LongZero-style self-generation)
#   Expanded context C: long multi-document context containing E* plus distractors
# Output: GRPO-trained policy π with step-level advantages

# (D) Load-bearing assumption, stated explicitly:
#   The self-generated triple's spans E* are treated as a unique minimal sufficient
#   set with a canonical dependency ordering, so exact span-overlap match and
#   Kendall-tau rank agreement against E* are valid ground-truth step-level
#   correctness signals. This assumption is verified on the actual generated data
#   (see Step 0) rather than asserted by construction.

0. Triple minimality / correctness check (run once on the generated task set):
   For each task T = (q, a, E*):
     - minimality: for each span e in E*, remove e from C and re-query π;
       record answer_flip(e) = 1 if â != a, else 0.
       minimality_rate = mean_e[answer_flip(e)].
     - redundancy: sample M=4 additional self-generated triples for the same q;
       let E_pool = union of their spans. record alt_sufficient = 1 if any
       span in E_pool \ E* also yields â = a when substituted for a span in E*.
   Report minimality_rate and alt_sufficient rate on the training set.
   Tasks with minimality_rate < 0.75 or alt_sufficient = 1 are flagged and
   excluded from step-level supervision (they still contribute outcome reward).

1. Rollout: π generates a retrieval-and-integration trajectory
   τ = [(s_1, e_1), (s_2, e_2), ..., (s_K, e_K), r, â]
   where s_k is a retrieval sub-query, e_k is the selected evidence span,
   r is the integrated reasoning chain, â is the final answer.
   K is determined by the policy stopping when it emits <ANSWER>.

2. Step-level verification (no learned model):
   For each step k, compute match(e_k, E*) ∈ {0,1} using exact span
   overlap ≥ 0.8 (token-level F1 against the closest E* span).
   Compute order_score = Kendall-tau(rank(e_1..e_K), rank(E*)) normalized to [0,1].
   Compute integration_score = 1 if r cites only spans in E* and â = a, else 0.
   Compute necessity_score = 1 if removing e_k from the cited prefix flips â
   (leave-one-out answer-flip test on the rollout's own cited spans), else 0.

3. Step reward:
   R_k = 1.0 * match(e_k, E*) + 0.3 * (order_score if k = K else 0)
         + 0.5 * (integration_score if k = K else 0)
         + 0.2 * necessity_score
   (hyperparameters: match weight 1.0, order weight 0.3, integration weight 0.5,
    necessity weight 0.2)

4. Outcome reward:
   R_out = 1.0 if â = a else 0.0

5. GRPO advantage with step-level credit:
   For a group of G=8 rollouts {τ_i} for the same task T:
     A_out_i = (R_out_i - mean(R_out)) / std(R_out)
     A_k_i = (R_k_i - mean_k(R_k)) / std_k(R_k)  # per-step normalization across group
   Combined advantage for token at step k:
     A_k_i = 0.6 * A_out_i + 0.4 * A_k_i   # (hyperparameter: outcome weight 0.6)
   Note: A_out_i is broadcast to all tokens; A_k_i is applied only to tokens
   in the retrieval sub-query and evidence-selection segment of step k.

   Alignment across rollouts with different K:
     - Pad step sequences to K_max = max_i K_i with a mask m_k_i ∈ {0,1}
       (m_k_i = 1 if rollout i has a step k, else 0).
     - mean_k and std_k are computed over {R_k_i : m_k_i = 1} only.
     - If std_k = 0 (all present rollouts tie at step k), set A_k_i = 0 for that step.
     - Tokens at padded steps receive no step advantage (A_k_i = 0); they still
       receive the broadcast A_out_i.

6. Policy update: standard GRPO clipped surrogate objective with per-token
   advantages A_k_i, clipping ε=0.2, KL penalty β=0.01 to reference policy.

7. Pilot diagnostic (before scaling to 128K):
   Run 16K-context, 1–2K optimization steps. Log per-step reward variance
   var_k(R_k) and the ratio var(A_k) / var(A_out). If var(A_k) / var(A_out) > 2.0
   on any step, down-weight the step term (multiply 0.4 by 0.5) and re-run.
```

**Why this design**

We chose exact span-overlap matching against the self-generated evidence triple over a learned process reward model because the triple's evidence is the unique minimal sufficient set for the answer (by construction in LongZero's self-generation), so exact matching is a valid ground-truth signal rather than a noisy proxy; the cost is that it cannot handle evidence paraphrases, which we accept because retrieval spans are copied verbatim from the context. We chose Kendall-tau order scoring over a learned ordering head because the ground-truth evidence order is directly available from the triple, making order verification a deterministic computation rather than a prediction problem; the cost is that Kendall-tau is sensitive to a single swapped pair, which we mitigate by normalizing and weighting it at only 0.3. We chose to mix step-level and outcome-level advantages with a fixed 0.6/0.4 weight rather than learning the mixing coefficient because a learned mixer would reintroduce the credit-assignment problem at the meta-level; the cost is that the fixed weight may be suboptimal for tasks where retrieval is trivial, which we accept because the outcome advantage still dominates when step rewards are uniform. Unlike VinePPO, which estimates per-step values via Monte-Carlo rollouts from each prefix (expensive and high-variance), our step rewards are computed deterministically from the known evidence triple with zero additional rollouts, and unlike Process Reward Models (Lightman 2023), we require no human step labels or trained reward model — the self-generated triple supplies the step-level supervision for free. We added the Step 0 minimality/correctness check and the necessity_score term specifically to make the load-bearing assumption explicit and measurable: tasks whose triples fail the minimality or redundancy check are excluded from step-level supervision, and necessity_score rewards spans that are causally required for the answer rather than merely present in one arbitrary generation path.

**Why this measures what we claim**

`match(e_k, E*)` measures correct evidence retrieval at step k because the assumption that the self-generated evidence triple's spans are the unique minimal sufficient set for the answer makes exact span overlap equivalent to retrieval correctness; this assumption fails when the triple contains redundant evidence spans, in which case match reflects overlap with a non-minimal set and may reward redundant retrieval. `order_score` measures evidence-integration ordering because the assumption that the ground-truth evidence order reflects a valid reasoning dependency chain makes Kendall-tau rank correlation equivalent to integration soundness; this assumption fails when multiple evidence orderings are equally valid, in which case order_score penalizes a correct alternative ordering. `integration_score` measures faithful evidence integration because the assumption that a correct reasoning chain citing only ground-truth spans and yielding the correct answer is faithful makes the binary citation check equivalent to faithfulness; this assumption fails when the model produces a correct answer via a spurious shortcut that happens to cite the right spans, in which case integration_score reflects surface citation rather than causal reasoning. `necessity_score` measures causal necessity of a cited span because the assumption that removing a necessary span flips the answer makes the leave-one-out answer-flip test equivalent to causal contribution; this assumption fails when the model's answer is robust to span removal for reasons unrelated to the span's role (e.g., the answer is recoverable from a distractor), in which case necessity_score reflects answer robustness rather than span necessity. The combined advantage `A_k_i = 0.6 A_out_i + 0.4 A_k_i` measures step-level credit assignment because the assumption that step rewards are commensurable with outcome rewards (both in [0,1] and normalized per-group) makes the weighted sum a valid credit signal; this assumption fails when step rewards have much higher variance than outcome rewards, in which case the step term dominates and the policy over-optimizes retrieval at the expense of answer correctness — which is why Step 7 logs var(A_k)/var(A_out) and triggers down-weighting if the ratio exceeds 2.0.

## Contribution

(1) A step-wise verifier for zero-data long-context RL that computes deterministic evidence-selection, ordering, and integration rewards directly from self-generated QA-evidence triples, requiring no learned process reward model or human step labels. (2) A GRPO advantage-mixing scheme that broadcasts outcome advantages to all tokens while applying step-level advantages only to retrieval and evidence-selection segments, enabling credit assignment over multi-document retrieval without additional rollouts. (3) The design principle that self-generated evidence triples provide free process-level supervision, making step-wise verification structurally cheaper than Monte-Carlo prefix estimation (VinePPO) or trained PRMs.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | Long-context multi-hop QA from KeyChain-synthesized tasks (16K–128K) | Matches retrieval-integration structure our step rewards supervise |
| Dataset (non-synthetic) | HotpotQA multi-document QA with human-annotated supporting facts (used as E* instead of self-generation) | Tests whether the credit-assignment benefit is an artifact of synthetic minimal-evidence construction |
| Primary metric | Answer accuracy on multi-hop QA | Directly reflects whether step-level credit improves final reasoning |
| Secondary metric | Step-level match accuracy (fraction of steps with match(e_k,E*)=1) and order_score | Directly measures whether the step term improves retrieval and ordering |
| Secondary metric | Triple minimality_rate and alt_sufficient rate on the generated training set (from Step 0) | Quantifies how often the load-bearing assumption holds on actual data |
| Baseline | LongZero (outcome-only GRPO) | Isolates value of step-level credit over outcome-only |
| Baseline | LoongRL (KeyChain RL) | Tests our self-generated-triple step supervision vs. synthesized curriculum |
| Baseline | Untrained Qwen2.5-7B | Establishes the floor the method must clear |
| Baseline | Learned PRM on the same triples (Lightman-style, trained on self-generated step labels) | Tests whether the deterministic signal is merely a cheaper proxy for the same information |
| Ablation-of-ours | Ours minus step-level advantage (outcome weight 1.0) | Isolates the 0.4 step term's contribution |
| Ablation-of-ours | Ours minus necessity_score (necessity weight 0.0) | Isolates the leave-one-out causal-necessity term |
| Ablation-of-ours | Ours minus order_score (order weight 0.0) | Isolates the Kendall-tau ordering term |

### Why this setup validates the claim

The central claim is that deterministic step-level rewards derived from a self-generated evidence triple give GRPO a credit-assignment signal that outcome-only RL cannot recover, and that this signal specifically improves evidence retrieval and ordering on long-context multi-hop QA. The dataset choice matters because KeyChain-style tasks have an explicit minimal evidence set and a dependency ordering, so the predicted failure mode — retrieving a distractor span or integrating evidence out of order — is actually inducible and observable. Outcome-only LongZero tests the sub-claim that outcome reward alone under-credits correct intermediate retrieval: if our method beats it, the step term is doing work. LoongRL tests whether our zero-label triple supervision competes with curriculum-synthesized training, isolating the value of the free step signal. The ablation (outcome weight set to 1.0, step term removed) is the sharpest test: it holds every other component fixed, so any gain of the full method over the ablation must be attributable to the 0.4 step-level advantage. The learned-PRM baseline tests whether the deterministic signal is merely a cheaper proxy for the same information: if the PRM matches our method, the value is in the step labels, not in the determinism; if our method matches or beats the PRM at lower cost, the deterministic signal is doing independent work. The HotpotQA-with-human-annotations dataset tests whether the benefit survives when E* is not self-generated, addressing the concern that the synthetic minimal-evidence construction is the source of the gain. The minimality_rate and alt_sufficient diagnostics directly measure whether the load-bearing assumption holds on the actual generated data, so a failure of the assumption is observable rather than hidden. Answer accuracy is the right primary metric because the claim is that better step credit propagates to correct final answers; if step-level improvements did not transfer to accuracy, the mechanism would be inert regardless of how clean the step rewards look.

### Expected outcome and causal chain

**vs. LongZero (outcome-only GRPO)** — On a case where the model retrieves a distractor span early but the final answer is still wrong, outcome-only GRPO assigns a uniform negative advantage to every token, so the correct retrieval sub-query and the wrong evidence selection receive identical gradients and the model cannot localize its error. Our method instead scores that step's evidence span against the triple's ground-truth span and gives the wrong-selection tokens a lower step advantage than the correct sub-query tokens, so the policy is pushed specifically toward correct retrieval. We expect a noticeable accuracy gap concentrated on the multi-hop subset requiring two or more evidence hops, with near-parity on single-hop questions where outcome reward already suffices.

**vs. LoongRL (KeyChain RL)** — On a case where the required evidence order is reversed relative to the surface document order, a curriculum-trained policy that only sees outcome reward can learn to retrieve spans but has no signal about integration order, so it produces a plausible chain in document order and answers incorrectly. Our method instead applies the Kendall-tau order score at the terminal step, directly penalizing an out-of-order chain even when the retrieved spans are correct. We expect our method to show a clear gap on the order-sensitive subset (questions whose evidence appears in non-monotonic document order) while matching LoongRL on order-insensitive questions, indicating the order signal is the active ingredient.

**vs. Untrained Qwen2.5-7B** — On a long 128K-context instance with many distractor documents, the untrained policy has no learned retrieval behavior and either attends to salient distractors or truncates the needed evidence, producing confidently wrong answers. Our method instead trains the retrieval sub-query and evidence-selection segments with per-step advantages that reward exact overlap with the triple's spans, shaping the policy to locate the minimal sufficient set before integrating. We expect a large absolute accuracy gain over the untrained model, with the gain growing as context length increases from 16K to 128K, since the retrieval step is where the untrained model fails most.

**vs. Learned PRM on the same triples** — On a case where the self-generated triple contains a redundant span, a PRM trained on self-generated step labels inherits the same redundancy and may assign high step reward to the redundant span, while our Step 0 minimality check excludes such tasks from step-level supervision and our necessity_score rewards only causally required spans. We expect our method to match or beat the PRM on answer accuracy at lower training cost (no PRM training pass), with the gap concentrated on tasks where the triple's minimality_rate is below 1.0.

**vs. HotpotQA with human-annotated supporting facts** — On a case where the human-annotated supporting facts differ from what a self-generated triple would contain, a method that relies on self-generation may mis-specify E* and inject reward noise. Our method uses the human annotations directly as E*, so the step rewards are grounded in human judgment rather than one generation path. We expect the accuracy gain over LongZero to persist on HotpotQA, confirming the benefit is not an artifact of synthetic minimal-evidence construction.

### What would falsify this idea

If the full method's gain over the outcome-only ablation is uniform across single-hop and multi-hop subsets rather than concentrated on the retrieval- and order-sensitive subsets where the step term should matter, then the step-level advantage is not producing the claimed credit assignment and the central mechanism is wrong. Likewise, if the ablation matches the full method on order-sensitive questions, the Kendall-tau term is inert and the design's ordering claim fails; and if the learned-PRM baseline matches our method on all subsets, the deterministic signal is a cheaper proxy rather than an independent credit signal.

## References

1. LongZero: Enhancing Long-Context Reasoning in LLMs from Zero Data
2. LoongRL: Reinforcement Learning for Advanced Reasoning over Long Contexts
3. R-Zero: Self-Evolving Reasoning LLM from Zero Data
4. QwenLong-L1.5: Post-Training Recipe for Long-Context Reasoning and Memory Management
