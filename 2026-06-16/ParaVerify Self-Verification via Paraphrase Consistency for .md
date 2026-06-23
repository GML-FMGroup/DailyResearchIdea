# ParaVerify: Self-Verification via Paraphrase Consistency for LLM Agents

## Motivation

Existing methods like MC-DML (Monte Carlo Planning with Large Language Model for Text-Based Game Agents) and LLM-MCTS assume that LLM-generated plans and actions are correct, leading to cascading errors when the LLM produces unreliable outputs. The root cause is the absence of an online verification mechanism: the agent has no way to detect when its own policy is inaccurate. This structural gap persists across multiple research trees, where the LLM's outputs are trusted without cross-checking against the environment's semantic invariants.

## Key Insight

The optimal action in a text-based game is invariant under paraphrases of the state description because the environment semantics are unchanged, so disagreement among actions sampled from different paraphrases signals policy uncertainty or unreliability.

## Method

**(A) What it is**
ParaVerify is a self-verification module that takes as input the current state description \(s\) and an LLM policy \(\pi\), and outputs a reliability score \(r\) and a recommended action \(a\) (or a rejection signal). It computes the consistency of \(\pi\) across multiple paraphrases of \(s\).

**(B) How it works**
```pseudocode
Input: state s, LLM, hyperparameters N=5, threshold τ=0.6

1. Generate paraphrases: for i in 1..N:
       p_i = LLM.generate_paraphrase(s)   # prompt: "Rephrase the following state description preserving all details:"
2. Query action for each paraphrase:
       a_i = LLM.get_action(p_i)            # prompt: "Given this state, what is the best action?"
3. Compute mode action a_mode = majority vote among a_i
       agreement = count(a_mode) / N
4. If agreement ≥ τ:
       output (a_mode, agreement)
   else:
       output ("REJECT", agreement)
```
Hyperparameters: \(N=5\) (cost–accuracy trade-off), \(\tau=0.6\) (conservativeness).

**(C) Why this design**
We chose to generate multiple paraphrases rather than using a single state description because a single query may yield an unreliable output due to sampling variance or misinterpretation; the trade-off is computational cost (N LLM calls per state). We chose agreement proportion over log-probabilities as the reliability score because log-probs are poorly calibrated for action correctness (see anti-pattern 3); the trade-off is that agreement requires multiple calls. We chose a hard rejection threshold instead of soft weighting because a hard stop prevents error propagation and triggers re-planning, accepting that some correct actions may be rejected (false positive). This design is not a variant of prior self-consistency methods (e.g., CoT self-consistency) because those average over multiple reasoning paths for the same input, while ParaVerify averages over semantically equivalent inputs, exploiting an invariance property of the environment rather than of the reasoning process.

**(D) Why it measures what we claim**
The quantity `agreement` measures the consistency of the policy's actions across paraphrases, which we claim measures policy reliability. This claim relies on the assumption that paraphrases are semantically equivalent to the original state and therefore the true optimal action is identical across all paraphrases. This assumption fails when the LLM generates paraphrases that omit or alter task-relevant details (e.g., changing "red block" to "block"); in that case, agreement may be low even for a reliable policy, causing false rejections. The `mode action` measures the invariant policy under the assumption that the majority of paraphrases are accurate; the failure mode is that if a systematic bias affects multiple paraphrases (e.g., all paraphrases misinterpret a pronoun), the mode may be unreliable but agreement remains high, leading to false acceptance. The `threshold τ` operationalizes the certificate: a value of 0.6 means that at least 60% of paraphrases must agree; this assumption fails if the true optimal policy is stochastic (multiple equally good actions), in which case agreement may hover near 50% and the method may always reject, even though actions are reasonable.

## Contribution

(1) ParaVerify, a self-verification module that exploits paraphrase invariance to detect unreliable LLM agent outputs without external rewards or oracles. (2) A design principle demonstrating that action consistency across semantically equivalent state descriptions serves as a proxy for policy reliability, enabling online rejection of uncertain decisions. (3) A rejection-based fallback strategy that, when integrated into MCTS planning, prevents cascading errors by triggering re-planning upon low consistency.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|-----------------------|
| Dataset | Jericho text-based games | Tests reasoning consistency |
| Primary metric | Task completion rate | Direct measure of action correctness |
| Baseline 1 | MCTS+RL | Planning without self-verification |
| Baseline 2 | LaPSRL | Posterior sampling without consistency |
| Ablation of ours | ParaVerify w/o paraphrases | Single query, no consistency check |

### Why this setup validates the claim

Jericho games require accurate reasoning about state changes and action effects, making them ideal to test whether consistency across paraphrases indicates reliable action selection. By comparing to MCTS+RL (a strong planning method without self-verification) and LaPSRL (a posterior sampling method that ignores semantic invariance), we can isolate the benefit of our consistency check. The ablation without paraphrases directly tests the contribution of multiple paraphrases. Task completion rate is the right metric because it captures ultimate action correctness, not proxy signals. If our method improves completion rate on states where paraphrases vary, it validates the claim that agreement measures reliability.

### Expected outcome and causal chain

**vs. MCTS+RL** — On a state description that is ambiguous (e.g., "the red block is on the table near the blue block"), MCTS+RL may interpret "near" incorrectly and plan a suboptimal action because it relies on a single interpretation. Our method generates paraphrases (e.g., "the red block rests on the table close to the blue block") and checks if actions agree; if high agreement, it takes the majority action. We expect noticeable gain on states with lexical ambiguity but parity on unambiguous states.

**vs. LaPSRL** — On a state where the LLM's representation is noisy (e.g., missing adjectives like "small red block"), LaPSRL's posterior sampling may waste exploration on irrelevant details because it treats each state as a single sample. Our method aggregates across paraphrases; if most paraphrases preserve the key detail, mode action is correct. We expect higher completion rate on states with noise-prone phrasing.

### What would falsify this idea

If the gain over the ablation (single query) is uniform across all state types rather than concentrated on states with high paraphrasability or ambiguity, then the consistency mechanism is not driving performance. Also, if our method performs worse than baselines on ambiguous states, the assumption that agreement indicates reliability is wrong.

## References

1. Monte Carlo Planning with Large Language Model for Text-Based Game Agents
2. Large Language Models as Commonsense Knowledge for Large-Scale Task Planning
3. Multi-skill Mobile Manipulation for Object Rearrangement
4. ProgPrompt: Generating Situated Robot Task Plans using Large Language Models
5. Inner Monologue: Embodied Reasoning through Planning with Language Models
6. Toward Efficient Exploration by Large Language Model Agents
7. Isoperimetry is All We Need: Langevin Posterior Sampling for RL with Sublinear Regret
8. Efficient Reinforcement Learning with Large Language Model Priors
