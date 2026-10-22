# SCoRD: Self-Consistent Redundancy Boundary Learning for Long-Horizon Agents

## Motivation

Across the agent-training literature, the rule that decides whether a repeated action is a wasteful loop or a necessary retry is a fixed, non-differentiable preprocessing heuristic whose output is then trusted as ground-truth supervision. DeLoop, for instance, labels any action that repeats an earlier tool call with an identical observation as a negative and the first subsequent non-looping action as a positive, then optimizes DPO on these pairs — but it never checks whether the policy actually needed that repetition to succeed, so the loop/non-loop boundary is exogenous and systematically miscalibrated. The same structural flaw recurs in Context-Folding's folding triggers, CAT's compression milestones, and RLTR's tool-use-completeness reward: the property 'learned identifiability of the redundancy-vs-necessary-repetition boundary' is never established because no branch optimizes the detector jointly with the policy against downstream correctness.

## Key Insight

The redundancy boundary is identifiable from task success alone once it is forced to be a latent variable that must simultaneously explain the policy's own conditional action likelihood and its downstream return — two constraints that pin down the boundary where a fixed heuristic cannot, because the heuristic is never contradicted by the return signal it is supposed to predict.

## Method

**What it is.** SCoRD (Self-Consistent Redundancy Detector) is a jointly-trained boundary model `b_φ(a_t | h_t) ∈ [0,1]` that outputs the probability that action `a_t` at history `h_t` is *redundant* (a loop) versus *necessary repetition*. It is trained end-to-end with the agent policy `π_θ` under a self-consistency objective; its outputs gate which trajectory segments become DPO preference pairs and which segments receive a return-based calibration loss.

**How it works.**

```pseudocode
# Rollout phase (per task, policy pi_theta, boundary b_phi)
for each task x:
    rollout trajectory tau = (h_1,a_1,o_1,...,h_T,a_T,o_T), return R = success(x)
    for each step t:
        r_t   = b_phi(a_t | h_t)                       # redundancy score in [0,1]
        l_t   = log pi_theta(a_t | h_t)                # policy's own action log-likelihood
        rep_t = 1[a_t repeats an earlier call with identical observation]  # raw repeat indicator

# Per-step counterfactual replay (replaces trajectory-mean R_drop)
for each step t with r_t > 0.5:
    R_drop_t = return of the same task with ONLY step t replaced by a no-op
               (re-evaluated by environment replay)
    d_t = (R - R_drop_t) / T                       # per-step marginal return drop

# Self-consistency constraint (differentiable, per step)
#   the boundary must agree with the policy's own likelihood of repeating,
#   but only as a weak prior whose weight is gated by the replay drop
w_t = exp(-|d_t| / sigma)                          # sigma = 0.1; down-weights when replay drop is zero
L_sc = mean_t [ w_t * ( r_t * (1 - exp(l_t)) + (1 - r_t) * rep_t ) ]
#   -> if b_phi says 'redundant' (r_t high), the policy should have LOW likelihood
#      of that action; if b_phi says 'necessary' (r_t low), the action should
#      actually be a repeat the policy needed. The gate w_t removes the
#      circular dependence on pi_theta's confidence when replay shows no effect.

# Return-calibration constraint (differentiable, per step)
#   a redundant step is one whose removal does not change the return
L_cal = mean_t [ ( r_t - sigmoid(d_t) )^2 ]        # per-step target from replay
#   -> the boundary's per-step redundancy must predict whether dropping
#      that step actually hurt success, not a hand-coded loop rule.

# Joint training
L = L_task(pi_theta)  +  lambda_sc * L_sc  +  lambda_cal * L_cal
#   lambda_sc = 0.5, lambda_cal = 0.5 (fixed, not tuned per task)
#   L_task is the standard DPO loss on pairs (a_t with r_t>0.5 as negative,
#   first a_t with r_t<0.5 as positive) -- but the PAIRS are now chosen by b_phi.
```

A small MLP head (2 layers, hidden 256, GeLU activation) reads the frozen policy's hidden state at `h_t` plus a 3-dim repeat feature (repeat count, observation-equality flag, step index) and outputs `r_t`; it is trained by the two consistency losses above, backpropagated through the policy's hidden state but with the policy's own parameters updated only by `L_task` (stop-gradient on `L_sc`/`L_cal` w.r.t. `θ`). Warm-up: first 500 steps use only `L_cal` (with `L_sc` weight 0) to initialize `r_t` from replay before the gated `L_sc` term activates.

**Why this design.** We chose a *jointly-trained latent boundary* over DeLoop's fixed loop detector because the detector's notion of redundant is never contradicted by the return signal, accepting the cost that we need environment replay to compute `d_t`. We chose a *self-consistency loss between the boundary and the policy's own action likelihood* over a separate classifier trained on loop labels because the policy's likelihood is the only signal that reflects whether the agent *needed* the repetition, accepting the cost that early in training the policy is noisy and `L_sc` is a weak signal (we mitigate with a warm-up of 500 steps where only `L_cal` is active). We chose a *return-calibration loss via counterfactual replay* over a reward-model judge because replay gives an exact interventional signal rather than a proxy, accepting the cost of one extra environment rollout per flagged step. A domain expert would not describe this as a variant of DeLoop: DeLoop's detector is frozen and its pairs are chosen by a hand rule; SCoRD's pairs are chosen by a learned boundary whose *only* supervision is agreement with the policy's likelihood and the return drop — the detector and the policy co-adapt, which DeLoop structurally cannot do.

**Why it measures what we claim.** The quantity `r_t` measures *redundancy of action `a_t`* because the self-consistency loss forces `r_t` to be high exactly when the policy assigns low likelihood to `a_t` (assumption: a necessary repetition is one the policy would choose again given the same history; this assumption fails when the policy is systematically over-confident about a loop, in which case `r_t` reflects policy confidence rather than task necessity). The quantity `r_t` also measures *per-step necessity* because `L_cal` forces it to match `sigmoid(d_t)`, i.e., the actual return drop from removing that step (assumption: the return drop is a sufficient statistic for whether the repetition was necessary; this assumption fails when removing a step changes the environment state in a way that cascades, in which case `r_t` reflects local perturbation sensitivity rather than global necessity). The DPO pair selection measures *which actions to prefer* because the boundary's `r_t > 0.5` threshold is calibrated by `L_cal` to the return-drop scale (assumption: the threshold is monotone in return drop; this fails when the return is binary and the drop is zero for both redundant and necessary steps, in which case the pairs reflect likelihood disagreement rather than correctness).

## Contribution

(1) A jointly-trained latent redundancy boundary `b_φ` whose supervision is generated by a self-consistency constraint between the policy's own conditional action likelihood and a return-calibration signal from counterfactual replay, replacing DeLoop's fixed loop detector. (2) A design principle: the redundancy-vs-necessary-repetition boundary is identifiable from downstream task success alone when it is forced to be a latent variable that must agree with both the policy's likelihood and the interventional return drop, rather than trusted as exogenous ground truth. (3) An open training recipe (SCoRD) that produces calibrated preference pairs for long-horizon agent DPO without any hand-designed loop rule, evaluated on CodeGym under history compression.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | CodeGym under history compression | Long-horizon loops are the core failure mode |
| Primary metric | Task success rate at fixed step budget | Directly measures whether redundancy removal helped |
| Baseline | Untrained Qwen3-8B agent | Tests whether SCoRD beats no loop handling |
| Baseline | DeLoop (frozen detector) | Tests whether co-adaptation beats fixed detector |
| Baseline | End-to-end RLTR planner | Tests whether joint boundary beats multi-objective RL |
| Baseline | Supervised loop classifier (same MLP capacity, trained on DeLoop labels) | Tests whether gain comes from objective, not any learned module |
| Ablation-of-ours | SCoRD with L_cal disabled | Isolates return-calibration contribution |
| Ablation-of-ours | SCoRD with L_sc disabled (L_cal only) | Isolates self-consistency contribution |

### Why this setup validates the claim

The central claim is that a boundary co-adapted with the policy, supervised only by policy likelihood agreement and counterfactual return drop, selects better DPO pairs than a frozen detector or an end-to-end planner — and that this translates into fewer redundant steps without sacrificing success. CodeGym under compression is the right testbed because loops are the dominant failure mode there, so any gain in pair quality shows up as success-rate movement rather than being masked by other errors. The untrained baseline tests the sub-claim that loop handling matters at all; DeLoop tests the sub-claim that *co-adaptation* (not just loop detection) is the source of the gain, since DeLoop already detects loops but freezes the detector; the supervised loop classifier (same MLP capacity, trained on DeLoop labels) tests whether any learned module would suffice, isolating the contribution of the self-consistency/return-calibration objective; RLTR tests the sub-claim that our self-consistency signal is a better training signal than multi-objective end-to-end optimization. The L_cal ablation is the diagnostic: if the return-calibration loss is what calibrates the r_t>0.5 threshold to actual necessity, removing it should produce the same pair-selection failure DeLoop exhibits — pairs chosen by likelihood disagreement rather than correctness. The L_sc ablation tests whether the gated likelihood prior adds anything beyond replay calibration. Success rate at a fixed step budget is the right metric because it simultaneously rewards removing wasteful steps and penalizes removing necessary ones; a method that merely shortens trajectories by dropping steps indiscriminately would not improve it.

### Expected outcome and causal chain

**vs. Untrained Qwen3-8B agent** — On a CodeGym task where the agent re-invokes an identical file-read after a failed edit, the untrained baseline has no mechanism to distinguish this loop from a legitimate retry, so it burns steps until the budget expires and fails. Our method flags the identical-observation repeat as high `r_t`, forms a DPO pair against the first low-`r_t` action, and the policy learns to prefer the productive action. We expect a success-rate gap concentrated on tasks with high loop density, with near-parity on short tasks where loops rarely occur.

**vs. DeLoop (frozen detector)** — On a task where the policy's *own* likelihood of a repeat is high because the task genuinely requires re-reading a file after a state change, DeLoop's hand-coded loop rule flags it as redundant and generates a preference pair that penalizes a necessary action, degrading the policy. Our method's `L_sc` term pushes `r_t` low exactly when the policy needs the repeat, and `L_cal` confirms via replay that dropping it hurts return, so the pair is never formed. We expect DeLoop to show a success *regression* on necessity-heavy tasks while SCoRD matches or exceeds baseline there, and both to improve on loop-heavy tasks.

**vs. End-to-end RLTR planner** — On a task where the planner's summarization objective and planning objective conflict (summarizing away a step the planner later needs), RLTR's joint multi-objective loss averages the two signals and the planner loses track of state, producing a wrong action. Our method keeps the boundary and policy losses separate with stop-gradient on `θ` for `L_sc`/`L_cal`, so the boundary adapts to the policy without corrupting the policy gradient. We expect RLTR to plateau below SCoRD on long-horizon tasks specifically, with parity on short tasks where no compression conflict arises.

**vs. Supervised loop classifier (same MLP capacity, trained on DeLoop labels)** — This baseline has the same architecture as SCoRD's boundary but is trained with a fixed cross-entropy loss on DeLoop's hand-coded loop labels, so it inherits DeLoop's miscalibration on necessity-heavy tasks. We expect SCoRD to outperform it on both loop-heavy and necessity-heavy subsets, with the largest gap on necessity-heavy tasks where the supervised labels are systematically wrong; this isolates the gain to the self-consistency/return-calibration objective rather than to replacing a rule with any learned module.

### What would falsify this idea

If SCoRD's success-rate gain over DeLoop is uniform across loop-heavy and necessity-heavy task subsets rather than concentrated on the loop-heavy subset where the co-adaptation mechanism is supposed to act, the central claim that learned co-adaptation (not just loop detection) drives the improvement is wrong. Equivalently, if the L_cal ablation matches full SCoRD on pair-selection quality, the return-calibration loss is not doing the work we claim.

## References

1. DeLoop: Repeated Actions as Self-Supervision for Long-Horizon Agents
2. Context as a Tool: Context Management for Long-Horizon SWE-Agents
3. Encouraging Good Processes Without the Need for Good Answers: Reinforcement Learning for LLM Agent Planning
4. AgenTracer: Who Is Inducing Failure in the LLM Agentic Systems?
5. Scaling Long-Horizon LLM Agent via Context-Folding
6. Co-Evolving Agents: Learning from Failures as Hard Negatives
7. Generalizable End-to-End Tool-Use RL with Synthetic CodeGym
8. AgentGen: Enhancing Planning Abilities for Large Language Model based Agent via Environment and Task Generation
