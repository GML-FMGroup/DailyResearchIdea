# Shapley-Closed Credit Assignment for Multi-Robot Learning from Team-Level Human Feedback

## Motivation

Existing multi-robot learning from human feedback (e.g., the frontier thesis and R2BC) applies team-level preference or demonstration signals directly to decentralized per-agent policies without a decomposition rule, so the per-agent updates need not sum to the team-level signal and can even oppose it. This is the same closure failure that recurs across branches: the feedback is well-posed only at the team level, but decentralized execution requires per-agent credit. R2BC sidesteps the issue by collecting per-agent demonstrations, which assumes single-agent data captures coordination and does not scale to team-level preference feedback.

## Key Insight

If per-agent credit is defined as the exact Shapley value of each agent's action in a joint Q-decomposition whose team value is the human-feedback score, then the sum of credits equals the team-level feedback by the efficiency axiom of Shapley values, giving a closed decomposition that can be checked exactly at training time rather than learned.

## Method

```pseudocode
# Inputs: joint policy pi (decentralized actors), team Q-head Q_team(s,a; theta) = 3-layer MLP
#         (obs_dim*n + act_dim*n -> 256 -> 256 -> 1, ReLU), team size n,
#         human feedback buffer D_hf = {(s, a_team, r_hf)},
#         held-out calibration split D_cal subset D_hf (|D_cal| = 0.2 |D_hf|, disjoint from fit set)
# Output: per-agent credit c_i(s, a_team) with sum_i c_i = r_hf, trained decentralized actors

# [Assumption A, stated explicitly] Q_team must be accurate on coalition inputs
# (a_S, a_{not S}) where absent agents' actions are drawn from the current joint policy pi.
# This is the load-bearing faithfulness assumption: if Q_team extrapolates arbitrarily
# off-support, the Shapley marginals reflect extrapolation error, not marginal contribution.
# We therefore (i) define v(S) as an ON-POLICY expectation (step 2b), and
# (ii) verify Q_team calibration on held-out full joint actions (step 1b) before trusting phi_i.

1. Fit team Q-head Q_team on D_hf \ D_cal by minimizing  (Q_team(s, a_team) - r_hf)^2
   (Adam, lr=3e-4, batch=256, 5k gradient steps or until fit-set MSE < 1e-3).
   1b. Calibration check on D_cal: report held-out MSE_cal = mean_{(s,a,r) in D_cal}(Q_team(s,a)-r)^2
       and Spearman rho between Q_team(s,a) and r_hf on D_cal. Log both every K iterations.
       (This replaces the vacuous efficiency assertion as the faithfulness check; the
        efficiency identity in 2d is retained only as an algebraic sanity check on phi_i.)

2. For each feedback tuple (s, a_team, r_hf):
   a. Enumerate all 2^n coalitions S subseteq {1..n} for n <= 6;
      for n > 6 use stratified sampling of M=64 coalitions (stratified by |S| so each
      coalition size 0..n is represented; fixed RNG seed per tuple, budget pre-registered).
   b. For each coalition S, define the ON-POLICY value
        v(S) = (1/M_mc) * sum_{m=1..M_mc} Q_team(s, a_S, a_{not S}^{(m)}),  a_{not S}^{(m)} ~ pi(.|s)
      with M_mc = 16 Monte-Carlo samples from the current joint policy (not a fixed a_ref).
      Every coalition evaluation is thus an on-distribution expectation.
   c. Compute exact Shapley credit:
        phi_i = sum_{S not containing i} [ |S|! (n-|S|-1)! / n! ] * ( v(S ∪ {i}) - v(S) )
   d. Algebraic sanity check (not a faithfulness check):
        | sum_i phi_i - (v({1..n}) - v(empty)) | < 1e-5
      and set c_i = phi_i + (r_hf - v({1..n})) / n   [residual spread so sum_i c_i = r_hf exactly]

3. Actor update: for each agent i, form per-agent advantage
      A_i = c_i - V_i(s_i)   where V_i is a per-agent value head (2-layer MLP, hidden=128)
      trained by regression to c_i
      Update actor pi_i by policy gradient  grad log pi_i(a_i | s_i) * A_i

4. Every K=10 iterations, re-fit Q_team on the union of D_hf and fresh rollouts labeled by
   team return to keep the Shapley base game aligned with the current joint policy;
   re-run the D_cal calibration check (step 1b) after each re-fit.

5. Return per-agent actors, the closure-check log (max |sum_i phi_i - (v(full)-v(empty))|),
   and the calibration log (MSE_cal, Spearman rho on D_cal over training).
```

**Why this design.** We chose to compute Shapley credits from a single jointly-fit team Q-head rather than training a per-agent decomposition head, because a learned head can drift from the team signal and reintroduce the closure gap, accepting the cost that Q_team must be accurate on off-coalition inputs. To make that cost explicit and checkable, we define every coalition value v(S) as an on-policy Monte-Carlo expectation over absent agents' actions drawn from the current joint policy (M_mc=16 samples), rather than substituting a single fixed reference action a_ref; this keeps every coalition evaluation on the support of the policy and makes the load-bearing faithfulness assumption (Assumption A) a stated precondition rather than a hidden one. We additionally hold out 20% of D_hf as a calibration split and report held-out MSE and Spearman rho of Q_team against r_hf, so that the faithfulness of Q_team on full joint actions is measured directly; the Shapley efficiency identity is retained only as an algebraic sanity check on phi_i, since it holds identically for any base game and therefore cannot by itself verify faithfulness. We chose exact coalition enumeration for n <= 6 and stratified sampling of M=64 coalitions (stratified by coalition size, fixed seed) for larger teams rather than a kernel-SHAP approximation, because exact enumeration makes the algebraic check a hard assertion rather than a statistical hope, accepting O(2^n) cost that bounds the method to moderate team sizes. We chose to spread the residual (r_hf - v(full set)) uniformly across agents rather than assigning it to a learned bias, because uniform spreading preserves the closure identity exactly and cannot be gamed by any single agent, accepting that it slightly dilutes per-agent signal when Q_team is poorly fit. A domain expert would not describe this as a variant of R2BC or of learned credit-assignment heads (e.g., COMA): those either avoid team-level feedback or learn a decomposition, whereas here the decomposition is an algebraic identity enforced by assertion, with the faithfulness precondition checked by a held-out calibration diagnostic.

**Why it measures what we claim.** The Shapley value phi_i measures per-agent credit because the efficiency axiom of cooperative game theory guarantees sum_i phi_i = v(full) - v(empty), under the assumption that Q_team is a faithful value function of the joint action on the coalition inputs visited by the on-policy expectation in step 2b; this assumption fails when Q_team extrapolates to unseen coalitions, in which case phi_i reflects Q_team's extrapolation error rather than true marginal contribution — which is why we report the held-out calibration MSE and Spearman rho on D_cal alongside phi_i, so that a large Shapley-vs-uniform gap accompanied by poor calibration is attributable to extrapolation rather than to marginal-contribution structure. The closure residual (r_hf - v(full set)) measures the gap between the human feedback and the fitted team value because r_hf is the ground-truth team score, under the assumption that the human feedback is a deterministic function of the joint action; this assumption fails when the human is noisy or context-dependent, in which case the residual reflects human noise rather than Q_team misfit. The uniform residual spread (r_hf - v(full))/n measures a per-agent correction that preserves closure because it sums to the residual by construction, under the assumption that no agent is more responsible for Q_team misfit than another; this assumption fails when one agent's actions dominate the team outcome, in which case the correction under-credits that agent.

## Contribution

(1) A Shapley-closure credit operator for multi-robot learning from team-level human feedback that computes per-agent credits from a jointly-fit team Q-head and enforces sum-of-credits equals team feedback as an exact training-time assertion. (2) A decentralized actor-update scheme that consumes these closed per-agent credits, with a residual-spreading rule that preserves closure even when the team Q-head is imperfect. (3) An open-source closure-check harness that logs the maximum Shapley-efficiency residual over training, enabling direct diagnosis of when team-level feedback is ill-posed for decentralized execution.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Cooperative multi-agent tasks (n=2..6 main; n=8,12 scaling) with scripted oracle team scores; small human study for validation | Team score is the only supervision signal assumed |
| Primary metric | Team return vs. number of human-feedback samples | Directly tests credit quality from sparse team signal |
| Secondary metric | Held-out calibration MSE and Spearman rho of Q_team on D_cal (20% of D_hf) | Measures faithfulness of Q_team on full joint actions |
| Diagnostic metric | Per-coalition epistemic variance (ensemble of 5 Q-heads) and correlation of calibration error with Shapley-vs-uniform gap | Tests whether gap is marginal-contribution or extrapolation artifact |
| Baseline | Single-agent RLHF applied independently per robot | Tests whether team credit is needed at all |
| Baseline | COMA-style learned counterfactual credit head (same Q_team, same D_hf) | Tests algebraic Shapley vs. learned decomposition |
| Baseline | Shapley Q-value with same Q_team and same D_hf | Controlled comparison of closure assertion + residual spread |
| Baseline | R2BC round-robin single-agent demonstrations | Tests feedback type: team score vs. per-agent demos |
| Ablation-of-ours | Uniform credit (r_hf/n) without Shapley decomposition | Isolates the Shapley decomposition's contribution |
| Ablation-of-ours | Shapley with fixed a_ref substitution (original step 2b) vs. on-policy MC (M_mc=16) | Isolates the on-policy coalition evaluation repair |

### Why this setup validates the claim

The central claim is that enforcing an exact Shapley closure identity over a jointly-fit team Q-head converts sparse team-level human scores into per-agent credits that decentralized actors can actually exploit. The metric—team return as a function of the number of human-feedback samples—is the right detector because the predicted effect is a sample-efficiency gain: if Shapley credits carry more per-agent signal than the alternatives, our curve should rise faster and plateau higher at matched feedback budgets, not merely differ at convergence. Each baseline isolates one sub-claim. Independent single-agent RLHF tests whether any team-level decomposition is needed; if it matches us, closure is doing nothing. The COMA-style learned decomposition and the Shapley Q-value baseline (both using the identical Q_team and D_hf) test whether the algebraic identity plus residual spread beats a learned head and a standard Shapley decomposition under controlled conditions. The ablation (uniform credit) tests whether the *Shapley structure* rather than the mere presence of a team signal drives the gain. Because the ablation preserves closure exactly (uniform spread sums to r_hf), any gap between it and full Shapley is attributable to marginal-contribution structure, not to closure. The held-out calibration metric and the per-coalition epistemic variance diagnostic directly test the load-bearing faithfulness assumption: if the Shapley-vs-uniform gap correlates with calibration error rather than with coordination coupling, the gain is an extrapolation artifact. Together these form a falsifiable ladder: the claim survives only if we beat independent RLHF, beat the learned head, beat uniform credit, and show the gap is not explained by calibration error.

### Expected outcome and causal chain

**vs. Single-agent RLHF** — On a case where two robots must coordinate (one blocks a corridor while the other retrieves), per-agent RLHF receives only a scalar per robot and cannot tell which robot's action caused the team score, so it assigns credit to whichever robot happened to act near the reward and reinforces the blocking robot's stall. Our method instead attributes credit through the joint Q-head's marginal contributions, so the blocking robot receives credit for enabling retrieval, and we expect a clear gap on coordination-required tasks with near-parity on independent tasks.

**vs. COMA-style learned credit head** — On a case where the learned decomposition head is fit on few team-score tuples, it drifts from the team signal and its per-agent credits fail to sum to the observed score, so actors receive systematically mis-scaled advantages and learning stalls early. Our method instead computes credits as an asserted algebraic identity over Q_team, so closure holds by construction and advantages stay calibrated to r_hf, and we expect our return curve to rise earlier in the low-feedback regime while converging similarly once data is abundant.

**vs. Shapley Q-value (same Q_team, same D_hf)** — On a case where the base game is identical, both methods compute Shapley credits, so any difference is attributable to the closure assertion and residual spread. We expect near-parity on well-calibrated Q_team regimes and a small advantage for our method when Q_team misfit is nonzero, because the residual spread keeps sum_i c_i = r_hf exactly while the standard Shapley Q-value leaves a nonzero closure gap that mis-scales advantages.

**vs. R2BC round-robin demonstrations** — On a case where sequential per-agent demonstrations are collected, each robot imitates a teammate's stale context, producing coordination mismatches at execution (agents expect partners that no longer exist). Our method instead learns from team-level scores on the *current* joint policy, so credits reflect the deployed teammates' behavior, and we expect a gap that widens as team size grows and coordination coupling increases, with parity on loosely coupled tasks.

**vs. Uniform credit ablation** — On a coordination-coupled task where marginal contributions diverge from uniform credit, the uniform ablation assigns equal credit to a blocking robot and a retrieving robot, so the blocking robot's stall is reinforced as strongly as the retrieval. Our method assigns credit proportional to marginal contribution, so the blocking robot is credited for enabling retrieval and the retrieving robot for the retrieval itself. We expect the gap to be concentrated on coordination-coupled tasks and near-zero on independent tasks; if the gap is uniform across task subsets, the Shapley-structure claim is falsified.

### What would falsify this idea

If our gain over the uniform-credit ablation and the learned-head baseline is uniform across task subsets rather than concentrated on coordination-coupled tasks where marginal contributions diverge from uniform credit, the claim that Shapley structure drives the improvement is wrong. Equivalently, if the held-out calibration MSE on D_cal is high and the Shapley-vs-uniform gap correlates with calibration error rather than with coordination coupling, then the gain is a Q_team extrapolation artifact and the design's central premise is falsified.

## References

1. Multi-Robot Learning from Human Feedback
2. R2BC: Multi-Agent Imitation Learning from Single-Agent Demonstrations
3. MAPF-GPT: Imitation Learning for Multi-Agent Pathfinding at Scale
4. Transformer-Based Imitative Reinforcement Learning for Multirobot Path Planning
