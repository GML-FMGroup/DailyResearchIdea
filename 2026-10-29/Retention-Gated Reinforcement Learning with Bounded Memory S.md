# Retention-Gated Reinforcement Learning with Bounded Memory Summaries for Context-Overflow Reasoning

## Motivation

Existing long-context RL methods train models on tasks whose evidence fits inside the training context window: LongZero expands self-generated evidence into a long multi-document context but never requires the model to preserve distant details once the window cannot hold all evidence, and LoongRL trains at 16K and relies on the plan-retrieve-reason-recheck pattern generalizing to 128K rather than on any explicit memory-retention requirement. QwenLong-L1.5 adds a memory-augmented architecture but does not impose a retention invariant that forces distant facts to survive an overflow event. Structurally, these methods optimize answer accuracy over a fully visible evidence set, so a model can succeed without ever committing distant critical facts to a bounded memory that outlives the context that produced them — the field's trajectory toward ultra-long evidence exposes this missing retention property.

## Key Insight

If the training reward is computed only on post-overflow answers whose distant evidence has been evicted from the context, then a bounded memory summary becomes the unique channel through which distant facts can reach the answer, making retention a causally necessary condition for reward rather than a correlated proxy.

## Method

**What it is.** Retention-Gated RL (RGRL) trains a policy to maintain a fixed-size memory summary M (at most K tokens, K=512) across a context-overflow event and rewards it only on post-overflow questions whose evidence lies in the evicted prefix. Inputs: a question q, a long evidence stream E split into prefix E_pre (evicted) and suffix E_suf (visible), and a memory budget K. Outputs: a memory summary M written during the prefix phase and a post-overflow answer a.

**Load-bearing assumption (stated explicitly).** The 512-token memory M written during the prefix phase is the only channel by which evicted prefix facts can reach the post-overflow answer, so post-overflow exact-match correctness is a valid, non-gameable certificate that the policy actually retained the distant fact in M. This assumption is enforced by construction (counterfactual entities, entity-disjoint suffix) and verified empirically (no-memory ablation at chance, contrastive retention reward).

**How it works.**

```pseudocode
# Phase 1: encode prefix into bounded memory
M_0 = empty
for chunk c_i in E_pre (chunks of 2048 tokens):
    M_i = LLM.update(M_{i-1}, c_i)          # write, capped at K=512 tokens
M = M_last                                  # bounded memory, |M| <= K

# Phase 2: overflow event — E_pre is discarded, only M + E_suf remain
context = M + E_suf

# Phase 3: post-overflow answer
q = question whose supporting evidence is in E_pre
a = LLM.answer(context, q)                  # must recover distant fact from M

# Phase 4: retention-gated reward (contrastive)
r_retain = 1[answer matches gold]           # only post-overflow questions
r_no_mem = 1[answer_no_mem matches gold]    # matched rollout with M zeroed
r_contrast = max(0, r_retain - r_no_mem)    # structurally zero unless M is causal
r_format = 1[M is well-formed and <= K tokens]
r = r_contrast + lambda * r_format          # lambda = 0.1

# Phase 5: GRPO update on the group of rollouts
GRPO.update(policy, {r_j})                  # group-relative advantage, group size G=8

# Phase 6: retention verification probe (held-out, no gradient)
#   ask a question answerable ONLY from E_pre after overflow;
#   a correct answer with E_pre absent certifies retention of that fact.
```

Task construction: for each synthesized multi-hop question, we split its evidence so that at least one hop's supporting document falls in E_pre and the final answer requires that hop. Questions whose evidence is entirely in E_suf are filtered out, because they cannot certify retention. The memory budget K is held fixed across training so the model cannot expand its way out of the invariant. To make retention causally identifiable, we enforce three construction rules: (1) all entities in E_pre are counterfactual synthetic keys (randomized alphanumeric strings of length 8, e.g., 'X7K9Q2M4') that do not appear in the backbone's pretraining corpus, verified by checking that the untrained backbone answers retention-certifying questions at chance (≤ 1/|candidate set|); (2) E_suf is entity-disjoint from E_pre — no entity string from E_pre appears in E_suf, killing paraphrase leakage; (3) each retention-certifying question has a candidate answer set of size 8 (the gold key plus 7 distractor keys drawn from the same synthetic distribution), so chance accuracy is 12.5%.

**Why this design.** We chose a fixed K-token memory written during the prefix phase over letting the model attend to the full evidence, because the retention invariant only binds when the distant fact has no other route to the answer; the cost is that the model must learn a lossy compression that happens to preserve answer-relevant facts. We chose to gate the reward on post-overflow questions only over mixing pre- and post-overflow questions, because a mixed reward lets the model solve the easy in-context questions and ignore memory; the cost is a harder, sparser reward signal and slower early training. We chose GRPO over PPO because GRPO's group-relative advantage removes the need for a learned value head that would itself have to model memory quality, accepting the cost that group size must be large enough to give stable advantage estimates. We chose a held-out retention probe over using training reward as the retention certificate, because training reward can be gamed by memorizing the specific training questions; the probe uses fresh questions over the same evicted evidence. A domain expert would not describe this as LongZero with a memory head: LongZero never evicts evidence, so its reward is not gated on retention, whereas RGRL's reward is structurally zero unless the bounded memory carries the distant fact.

**Why it measures what we claim.** The quantity r_retain (post-overflow answer correctness with E_pre absent) measures distant-fact retention because the only channel from E_pre to the answer is M, under the assumption that the model cannot reconstruct the evicted fact from E_suf or from parametric knowledge; this assumption fails when the question's answer is guessable from world knowledge or when E_suf accidentally paraphrases the evicted fact, in which case r_retain reflects prior knowledge or redundancy rather than memory retention. The contrastive reward r_contrast = max(0, r_retain - r_no_mem) measures causal retention because it is structurally zero unless the with-M rollout succeeds where the no-M rollout fails, under the assumption that the no-M rollout is a valid counterfactual (same policy, same question, M zeroed); this assumption fails when the no-M rollout succeeds by guessing, in which case r_contrast is zero and the reward correctly withholds credit. The quantity |M| <= K measures boundedness because the memory is the sole persistent state, under the assumption that the model does not smuggle prefix information into the visible suffix; this fails if the model writes prefix facts into E_suf during phase 1, in which case the bound is vacuous — we prevent this by construction rule (2) (entity-disjoint suffix) and verify by checking that no E_pre entity string appears in E_suf after phase 1. The held-out probe measures generalization of retention because it uses unseen questions over the same evicted evidence, under the assumption that the probe's evidence distribution matches training; this fails when probe questions are systematically easier, in which case the probe overstates retention — we control this by drawing probe questions from the same synthetic generator with the same candidate-set size 8.

## Contribution

(1) A retention-gated RL training procedure that imposes a bounded-memory retention invariant by evicting the evidence prefix before the answer is produced, making distant-fact retention a necessary condition for reward. (2) A task-construction filter and held-out retention probe that certify that post-overflow correctness is attributable to memory rather than to in-context or parametric shortcuts. (3) An empirical design principle: retention-gated rewards produce measurably higher post-overflow accuracy on distant-evidence questions than accuracy-only long-context RL at matched compute.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | KeyChain-synthesized multi-hop QA, 16K context, counterfactual entities (8-char random keys), candidate set size 8 | Forces one hop into evicted prefix; non-guessable answers |
| Dataset | Multi-document QA with genuine overflow (e.g., 32K-token evidence streams from a real corpus, filtered for entity-disjoint prefix/suffix) | Tests retention beyond synthetic construction |
| Primary metric | Post-overflow exact-match on retention-certifying questions, reported as gain over leakage floor | Only channel from evicted prefix is bounded memory |
| Secondary metric | Contrastive retention rate: fraction of questions where with-M correct and no-M incorrect | Directly measures causal retention |
| Baseline | Untrained Qwen2.5-7B backbone | Tests whether RL is needed at all; establishes leakage floor |
| Baseline | LongZero (no eviction) | Tests whether eviction gating drives the gain |
| Baseline | LoongRL (full-context RL) | Tests whether bounded memory survives vs full attention |
| Baseline | Bounded-memory policy with full-context reward (no eviction) | Isolates the gate from the memory mechanism |
| Ablation-of-ours | RGRL with mixed pre/post-overflow reward | Tests whether retention gating is load-bearing |
| Ablation-of-ours | RGRL with non-contrastive reward (r_retain only) | Tests whether contrastive gating is load-bearing |
| Ablation-of-ours | RGRL with M zeroed at test time | Measures direct contribution of M |

### Why this setup validates the claim

The central claim is that gating reward on post-overflow questions over an evicted prefix forces the policy to learn lossy compression that preserves answer-relevant facts. The untrained backbone tests the null: if it already answers retention-certifying questions, the questions leak information and the metric is invalid; we report its accuracy as the leakage floor and require RGRL's gain to exceed it by at least 20 absolute points. The bounded-memory policy with full-context reward isolates the gate: same architecture, same K, but reward flows from full evidence, so if it matches RGRL on retention-certifying questions, the gate is not doing work. LongZero tests the eviction sub-claim: it trains long-context reasoning without ever evicting, so if it matches RGRL on retention-certifying questions, the bounded-memory invariant is doing no work. LoongRL tests the boundedness sub-claim: it keeps full evidence visible, so its advantage would come from attention, not memory. The mixed-reward ablation is the sharpest test: same architecture, same K, but reward also flows from easy in-context questions, so if it matches RGRL, the gate is not load-bearing. The non-contrastive ablation tests whether the contrastive term is necessary to make retention causal. Exact-match on retention-certifying questions is the right metric because it is structurally zero without memory, unlike aggregate long-context accuracy which can be carried by suffix-only questions; the contrastive retention rate directly measures the causal channel.

### Expected outcome and causal chain

**vs. Untrained Qwen2.5-7B backbone** — On a case where the final hop's document sits in E_pre and the question is asked after overflow, the untrained backbone produces a plausible but wrong entity because its parametric prior fills the gap left by the missing evidence; with counterfactual 8-char keys and candidate set size 8, its accuracy is at chance (12.5%). Our method instead writes the hop into M during the prefix phase because the retention-gated reward makes any rollout that drops it score zero, so we expect a large gap on retention-certifying questions (≥ 50 absolute points over the 12.5% leakage floor) but near-parity on suffix-only questions, which the backbone already handles.

**vs. LongZero (no eviction)** — On a case where E_pre spans 12K tokens and the distant fact is buried mid-prefix, LongZero keeps full evidence in context and succeeds by attending to it, so its reward never distinguishes memory from attention. Our method instead discards E_pre and must recover the fact from a 512-token M, because the reward is gated on post-overflow correctness, so we expect LongZero to look strong at 16K but collapse when the same policy is evaluated with E_pre removed (accuracy near the 12.5% leakage floor), while RGRL degrades gracefully — the gap should appear exactly at the eviction boundary.

**vs. LoongRL (full-context RL)** — On a case where the evidence stream exceeds the trained context length, LoongRL produces a truncated or positionally-confused answer because its policy learned to attend over the full stream and has no persistent summary to fall back on. Our method instead carries the fact in M across the overflow, because the fixed-K invariant is enforced during training, so we expect LoongRL to lead on in-context multi-hop accuracy but RGRL to lead on retention-certifying questions and to be the only one that does not degrade as prefix length grows past the training window.

**vs. Bounded-memory policy with full-context reward (no eviction)** — On a case where the policy is trained with the same bounded memory M but rewarded on full-context answers, it learns to ignore M because the full evidence is always visible. Our method instead gates reward on post-overflow questions, so we expect the full-context-reward policy to match RGRL on in-context accuracy but fall to the leakage floor on retention-certifying questions, demonstrating that the gate alone produces retention.

**vs. RGRL with mixed pre/post-overflow reward** — On a case where the reward also flows from easy in-context questions, the policy can achieve high reward by solving suffix-only questions and ignoring memory. Our method instead gates reward only on post-overflow questions, so we expect the mixed-reward ablation to match RGRL on suffix-only questions but underperform by at least 20 absolute points on retention-certifying questions, confirming the gate is load-bearing.

**vs. RGRL with non-contrastive reward (r_retain only)** — On a case where the no-M rollout guesses correctly (12.5% of the time), the non-contrastive reward gives credit to a non-retaining policy. Our method instead uses r_contrast = max(0, r_retain - r_no_mem), so we expect the non-contrastive variant to show higher variance and lower contrastive retention rate, confirming the contrastive term is necessary.

**vs. RGRL with M zeroed at test time** — On a case where M is zeroed after training, the policy has no channel for the evicted fact. Our method's accuracy should drop to the leakage floor (12.5%), directly measuring the contribution of M and validating that the reward is structurally zero without it.

### What would falsify this idea

If the mixed-reward ablation matches RGRL on retention-certifying questions, the retention gate is not load-bearing and the central claim collapses. Likewise, if RGRL's gain over the untrained backbone is uniform across retention-certifying and suffix-only subsets rather than concentrated on the former, the metric is not measuring memory retention. If the contrastive retention rate is near zero despite high r_retain, the no-M rollout is succeeding by guessing, and the reward is not causally gated on M.

## References

1. LongZero: Enhancing Long-Context Reasoning in LLMs from Zero Data
2. LoongRL: Reinforcement Learning for Advanced Reasoning over Long Contexts
3. R-Zero: Self-Evolving Reasoning LLM from Zero Data
4. QwenLong-L1.5: Post-Training Recipe for Long-Context Reasoning and Memory Management
