# Structurally-Weighted On-Policy Distillation for Reasoning

## Motivation

Direct-OPD (Agarwal et al., 2024) uses a log-ratio reward that treats all tokens equally, ignoring their roles in reasoning steps. This fails to capture reasoning structure because the reward is a uniform average of per-token teacher shifts, so tokens in irrelevant parts (e.g., punctuation) contribute equally to tokens in critical reasoning steps. The root cause is that the reward lacks any mechanism to identify which tokens carry reasoning signal, a capability that GSRM (RLKD) provides via step-level decomposition but only in an offline teacher-alignment setting.

## Key Insight

Step boundaries in the student's own on-policy rollouts can be detected via lightweight heuristics, and weighting the teacher log-ratio by the inverse variance of log-ratios within each step produces a reward that inherently emphasizes reasoning-coherent tokens without requiring a separate alignment phase.

## Method

**SWORD (Structurally-Weighted On-Policy Distillation)**

(A) **What it is:** SWORD computes a token-level reward for on-policy student rollouts by weighting the teacher log-ratio (post-RL vs. pre-RL) with step-level importance derived from a heuristic decomposition of the student's trajectory into reasoning steps.

(B) **How it works (pseudocode):**
```pseudocode
Input: student rollout S, teacher pre-checkpoint T_pre, teacher post-checkpoint T_post
Hyperparameters: step_boundary_regex = r'\n|\.|!|\?|Step \d+|Therefore|Finally',
                 importance_fn = 'inverse_step_variance'

1. Decompose S into steps: steps = split(S, regex=step_boundary_regex)
   (each step is a contiguous token span)

2. Compute per-token log-ratio:
   for each token t_i in S:
       lr_i = log P_Tpost(t_i | context) - log P_Tpre(t_i | context)

3. Compute step-level importance weights:
   for each step j:
       if importance_fn == 'inverse_step_variance':
           var_j = variance({lr_i for tokens in step j})
           w_j = 1 / (var_j + 1e-6)
       elif importance_fn == 'step_length':
           w_j = sqrt(len(step j tokens))
   w = softmax([w_1, ..., w_m])  # normalize to sum=1

4. Assign token reward:
   for each token t_i:
       step_idx = step containing t_i
       reward_i = w[step_idx] * lr_i

Output: reward_i for each token
```

(C) **Why this design:** We chose regex-based step detection over learned segmentation (e.g., using a classifier) to avoid additional training overhead and keep the method fully on-policy; the trade-off is that regex may miss subtle step boundaries or oversplit continuous reasoning, but for structured tasks like math or code this is acceptable. We adopted inverse_variance as the default importance function because steps with highly variable teacher log-ratios indicate uncertain alignment (some tokens the teacher strongly prefers, others not), which we want to downweight; the cost is that steps with uniformly low log-ratio but low variance (e.g., confidently wrong steps) are not penalized by variance alone, so we rely on the log-ratio magnitude. We chose softmax normalization of step weights to ensure the total reward budget is fixed per rollout, preventing longer rollouts from dominating the gradient; this forces the model to compete for importance across steps, but may compress the weight range. These decisions are grounded in the desire to integrate structure without requiring a separate teacher alignment phase, directly addressing Direct-OPD's limitation.

(D) **Why it measures what we claim:** The computational quantity `w_j * lr_i` measures structure-aware reasoning alignment because the step weight `w_j` emphasizes tokens in steps where the teacher's log-ratio is consistent (low variance), under the assumption that low variance in log-ratio across a step indicates that the teacher consistently prefers (or disprefers) that step's reasoning as a whole; this assumption fails when the step contains mixed-quality tokens (e.g., correct premise but wrong conclusion), in which case low variance may still occur but the step is not coherent. The log-ratio `lr_i` measures the teacher's relative preference for token `t_i` under the refined vs. initial policy, under the assumption that the difference between post-RL and pre-RL probabilities captures the direction of correct reasoning; this assumption fails when the teacher's RL training exploits spurious correlations (e.g., format changes), in which case lr_i may reward formatting artifacts. The step decomposition `steps` operationalizes the concept of reasoning structure by splitting at boundaries that correlate with reasoning units, under the assumption that punctuation, line breaks, and discourse markers align with semantic step boundaries; this assumption fails in free-form reasoning without clear markers, in which case steps become arbitrary spans and the weighting loses structural meaning.

## Contribution

(1) A novel reward formulation for on-policy distillation that integrates token-level reasoning structure via step decomposition and importance weighting. (2) A design principle that structural reasoning alignment can be achieved without a separate teacher alignment module by leveraging the student's own step boundaries. (3) An analysis of variance-based importance as a proxy for step coherence.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|----------------------|
| Dataset | MATH | Tests multi-step reasoning.
| Primary metric | Accuracy | Standard measure of correctness.
| Baseline 1 | Direct-OPD | Direct on-policy distillation without structure.
| Baseline 2 | KL Distillation | Standard teacher-student KL loss.
| Baseline 3 | SeqKD | Sequence-level black-box distillation.
| Ablation-of-ours | SWORD (uniform) | Removes step weighting, tests structure contribution.

### Why this setup validates the claim

This design isolates the core claim: SWORD's step-weighted token rewards improve reasoning alignment. MATH is chosen because its multi-step problems require coherent reasoning chains, where step structure is meaningful—the very scenario SWORD's regex-based decomposition targets. Comparing against Direct-OPD tests the value of structural weighting over uniform on-policy distillation; against KL Distillation tests off-policy imitation; and against SeqKD tests distillation from fixed teacher rollouts. The ablation (SWORD with uniform step weights) directly falsifies whether the variance-based weighting drives gains. Accuracy as the metric captures end-to-end task success, which should improve if SWORD better aligns student reasoning with teacher improvements.

### Expected outcome and causal chain

**vs. Direct-OPD** — On a problem with many filler tokens (e.g., "We can see that... therefore... "), Direct-OPD treats each token equally, amplifying noise from irrelevant tokens. SWORD computes step-level variance: filler-heavy steps have high variance (teacher log-ratios fluctuate), yielding low weight, while concise reasoning steps have low variance and high weight. Thus, SWORD suppresses noise and emphasizes core reasoning. We expect SWORD to outperform Direct-OPD by a noticeable margin (e.g., 5-8% accuracy on MATH), especially on problems with verbose student rollouts.

**vs. KL Distillation** — On a case where the teacher's pre- and post-RL policies diverge significantly for a key equation step, KL distillation minimizes overall divergence but may dilute the signal from the improved step. SWORD's log-ratio explicitly captures the improvement per token, and step weighting amplifies consistent improvements (low-variance steps). The result: SWORD better captures the direction of teacher refinement. We expect SWORD to exceed KL by a smaller but consistent gap (e.g., 3-5%), concentrated on problems where the teacher's RL training yielded large policy changes.

**vs. SeqKD** — On a problem where the teacher's rollout is correct but the student's own rollout differs, SeqKD forces the student to mimic the teacher's path even if the student's path is equally valid. SWORD uses student's on-policy rollouts, weighting steps by teacher log-ratio variance, so it rewards steps where the teacher's preference is clear regardless of path. Thus, SWORD allows diverse reasoning while still learning from teacher's confidence. Expect SWORD to outperform SeqKD by 4-6%, with gains on problems admitting multiple reasoning paths.

### What would falsify this idea

If SWORD's gains are uniform across all subsets (e.g., equally large on simple one-step problems and complex multi-step ones) rather than concentrated on problems where step structure matters (e.g., multi-step vs. single-step problems show no differential improvement), then the step-weighting mechanism is not acting as intended and the central claim is false.

## References

1. Weak-to-Strong Generalization via Direct On-Policy Distillation
2. Black-Box On-Policy Distillation of Large Language Models
3. RLKD: Distilling LLMs' Reasoning via Reinforcement Learning
4. Teaching Large Language Models to Reason with Reinforcement Learning
5. On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes
6. WizardMath: Empowering Mathematical Reasoning for Large Language Models via Reinforced Evol-Instruct
7. Reinforced Self-Training (ReST) for Language Modeling
8. Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback
