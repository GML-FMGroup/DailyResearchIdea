# Budget-Conserving Branch Allocation for History-Guided Code Agent Trajectory Collection

## Motivation

History-guided trajectory collectors such as SWE-Trail decide where to spend extra collection effort by looking at past rollouts, but they treat test-time compute as an aggregate sample count: each new branch consumes budget without any global accounting, so branching inherits no compositional invariant and compute drifts toward whichever branch happened to trigger first rather than being conserved and redistributed. Structurally, the collector has no representation of a global budget that branches must sum to, so a replace action changes the sampling distribution without ever making the total spend auditable or transferable across branches. This means the field's trajectory-collection effort is systematically misallocated: high-value states cannot be funded from low-value branches because there is no conserved quantity to move.

## Key Insight

If every branch carries an explicit per-branch budget and a global conservation invariant (sum of branch budgets equals the target) is enforced after every trigger-driven reallocation, then history-based branching becomes a compute-conserving redistribution operator rather than an unbounded sampler, so additional effort at a promising state is necessarily funded by withdrawal from measured low-value branches.

## Method

**What it is.** BUDGET-TRAIL is a trajectory collector for coding agents that maintains a global test-time compute budget `B` and a per-branch budget vector whose entries always sum to `B`. Its inputs are the issue, a history of prior trajectories, a calibration set of 512 held-out (state, extra-rollout-outcome) pairs, and `B`; its outputs are collected trajectories together with an auditable allocation ledger.

**How it works.**

```pseudocode
BUDGET-TRAIL(issue, history, B, n_branches=K, min_budget=b_min, trigger_quantile=q,
             calibration_set=512 pairs, novelty_floor=0.25):
  # Phase 0: calibrate trigger score against realized marginal value
  #   calibration_set = 512 (state, extra_rollout_outcome) pairs from held-out issues
  #   bin states by trigger score s(obs) into 5 quantile bins
  #   for each bin, estimate marginal_value(bin) = mean extra-rollout solve rate
  #   fit isotonic regression marginal_value ~ s(obs); record Spearman rho
  #   if Spearman rho < 0.3: set use_trigger = False (fall back to uniform)
  #   else: use_trigger = True; store isotonic map for Phase 3

  # Phase 1: initialize a conserving allocation
  budgets = [B/K] * K
  assert sum(budgets) == B
  ledger = []
  active = list(range(K))

  # Phase 2: roll out branches under their own budgets
  for step in range(max_steps):
      for i in active:
          if budgets[i] <= 0: active.remove(i); continue
          a = agent.act(obs_i)              # agent sees NO history
          obs_i, r_i = env.step(a)
          budgets[i] -= 1

  # Phase 3: history-based trigger at a state
  #   history = prior trajectories for this issue
  #   score s(obs) = fraction of prior trajectories that reached
  #   a state within distance d of obs AND then failed
  #   calibrated_value(obs) = isotonic_map(s(obs))  # from Phase 0
  for i in active:
      s_i = trigger_score(obs_i, history, d)
      v_i = calibrated_value(s_i) if use_trigger else 0.0
      if v_i >= quantile(calibrated_values, trigger_quantile):
          # Phase 4: conserving redistribution
          donors = [j for j in active if j != i
                    and calibrated_value(s_j) < median(calibrated_values)
                    and novelty(obs_j, history) >= novelty_floor]
          # novelty(obs, history) = min distance from obs to any history state;
          # branches below novelty_floor are protected from withdrawal
          take = min(sum(budgets[j] for j in donors),
                     floor(0.5 * sum(budgets[j] for j in donors)))
          for j in donors:
              # withdraw proportionally to low value
              w_j = (median(calibrated_values) - calibrated_value(s_j)) /
                    sum(median(calibrated_values) - calibrated_value(s_k) for k in donors)
              budgets[j] -= floor(w_j * take)
          budgets[i] += take
          assert sum(budgets) == B          # invariant
          ledger.append((i, donors, take, s_i, v_i))

  return trajectories, ledger, budgets
```

The trigger score `s(obs)` is computed by comparing `obs` to the states recorded in `history` using a fixed embedding distance `d=0.1` in the agent's observation embedding space; it is the empirical failure rate of prior trajectories that passed near `obs`. Redistribution withdraws at most half of the donors' remaining budget and caps the recipient at `B/2`. The invariant is re-asserted after every reallocation. The calibration set of 512 pairs is used to fit an isotonic regression from `s(obs)` to realized marginal value (mean extra-rollout solve rate in each of 5 quantile bins); if the Spearman correlation between `s(obs)` and marginal value is below 0.3, the trigger is disabled and the collector falls back to uniform allocation. Branches whose state is farther than `novelty_floor=0.25` from any history state are protected from withdrawal.

**Why this design.** We chose an explicit per-branch budget vector over an aggregate sample count because only a vector makes the conservation invariant checkable at every step, accepting the cost that the collector must track and update K budgets rather than one counter. We chose to fund reallocation only from branches whose trigger score is below the history median rather than from all branches uniformly, because uniform withdrawal would tax the very states that history says are promising, accepting the cost that some genuinely uncertain branches may be under-funded. We chose a proportional-to-low-value withdrawal weight rather than equal withdrawal because it concentrates the cost on the branches history most clearly marks as redundant, accepting the cost that a single mis-scored branch can be drained quickly. We chose to keep the agent's observation free of history (as in SWE-Trail) rather than feeding history into the policy, because the training data must match inference-time inputs, accepting the cost that the trigger signal can only influence allocation, not the agent's own decisions. This is not a variant of SWE-Trail's replace action: SWE-Trail changes *what distribution* a branch samples from, whereas BUDGET-TRAIL changes *how much compute* each branch holds under a conserved total, and a domain expert would describe the two as operating on different quantities (distribution vs. budget) rather than as one being a reparameterization of the other.

**Why it measures what we claim.** The trigger score `s(obs)` measures *state value for additional collection effort* because the assumption that prior-trajectory failure near a state is a sufficient statistic for that state's marginal value of extra compute makes the empirical failure rate equivalent to expected gain from more rollouts there; this assumption fails when the history is too small or the embedding distance `d` conflates semantically different states, in which case `s(obs)` reflects sampling noise rather than value. The calibration set of 512 pairs and the isotonic regression measure *whether `s(obs)` is a calibrated predictor of marginal value* because the assumption that the Spearman correlation between `s(obs)` and realized extra-rollout solve rate exceeds 0.3 makes the trigger score usable for allocation; this assumption fails when the correlation is below 0.3, in which case the collector disables the trigger and falls back to uniform allocation. The withdrawal weight `w_j` measures *redundancy of a branch* because the assumption that a branch scoring below the history median is exploring already-covered territory makes low score equivalent to low marginal information; this assumption fails when a low-scoring branch is early in a genuinely novel trajectory, in which case `w_j` reflects premature pessimism rather than redundancy, which is why branches with novelty below `novelty_floor=0.25` are protected from withdrawal. The invariant `sum(budgets) == B` measures *compositional conservation of test-time compute* because the assumption that compute is fungible across branches at the same step makes the sum the correct conserved quantity; this assumption fails when branches have different remaining-step horizons, in which case the invariant conserves step-counts rather than effective FLOPs. The ledger measures *auditability of allocation* because the assumption that every reallocation is recorded with its donors and amount makes the final budget vector reconstructible; this assumption fails if a branch terminates before spending its allocation, in which case the ledger over-reports realized spend.

## Contribution

(1) BUDGET-TRAIL, a trajectory collector that represents test-time compute as a per-branch budget vector subject to a global conservation invariant and reallocates it via a history-based trigger with proportional-to-low-value withdrawal. (2) The design principle that history-guided branching should be a compute-conserving redistribution operator, with an explicit trigger-score-to-withdrawal-weight mapping that makes every reallocation auditable. (3) An allocation ledger artifact that records donor branches, withdrawal amounts, and trigger scores for every redistribution, enabling post-hoc analysis of where collection compute was moved and why.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | SWE-bench issue set with repeated-run history | Enables trigger scores from prior-trajectory failures |
| Calibration set | 512 held-out (state, extra-rollout-outcome) pairs | Fits isotonic map from trigger score to marginal value |
| Primary metric | Pass@1 of post-trained agent on held-out issues | Measures whether collected trajectories improve learning |
| Secondary metric | Pass@1 vs. total steps (compute-normalized frontier) | Tests efficiency, not just single-point gain |
| Baseline | Independent multi-trajectory sampling | Tests whether allocation beats uniform sampling |
| Baseline | Uniform-budget collector (no redistribution) | Isolates value of the trigger-driven reallocation |
| Baseline | SWE-Trail replace-action collector | Tests budget vs. distribution as the operative lever |
| Baseline | UCB-style allocator (trigger score as UCB bonus) | Tests whether proportional withdrawal is a principled approximation |
| Ablation-of-ours | BUDGET-TRAIL with random trigger scores | Tests whether history-based scoring drives any gain |
| Ablation-of-ours | BUDGET-TRAIL without novelty floor | Tests whether protecting novel branches matters |

### Why this setup validates the claim

The central claim is that conserving a global compute budget while shifting it toward history-flagged high-value states yields training trajectories that produce a better coding agent than spending the same total compute uniformly. The dataset supplies both the issues and the prior-run history the trigger score depends on, so the mechanism can actually fire. The calibration set of 512 held-out pairs is the load-bearing validation of the trigger score: we bin states by `s(obs)` into 5 quantile bins, estimate the mean extra-rollout solve rate in each bin, fit an isotonic regression, and report the Spearman correlation; if it is below 0.3, the trigger is disabled and the method falls back to uniform, so the experiment directly tests the assumption that `s(obs)` predicts marginal value. The primary metric, held-out pass@1 after post-training, is the right detector because it is downstream of trajectory quality: if reallocation merely reshuffles compute without surfacing better states, pass@1 should stay flat. The uniform-budget collector is the load-bearing baseline, since it holds total compute and branch count fixed and differs only in redistribution, isolating the trigger mechanism. The independent sampler tests the coarser claim that structured collection beats naive sampling at all, while SWE-Trail tests whether the lever is budget rather than sampling distribution. The UCB-style allocator tests whether the proportional-withdrawal rule is a principled approximation to value-of-information allocation rather than mere bookkeeping. The random-trigger ablation is the falsification control: if gains survive when trigger scores are noise, the causal story is wrong. The novelty-floor ablation tests whether protecting novel branches prevents premature pessimism.

### Expected outcome and causal chain

**vs. Independent multi-trajectory sampling** — On an issue where the agent repeatedly stalls at the same failing test assertion, independent sampling spends its whole budget re-entering that dead region because it has no signal about where prior rollouts failed. Our method instead detects the high calibrated value at the stalled state and redirects donors' budget there, because the isotonic-calibrated failure rate marks that state as high marginal value. We expect a noticeable pass@1 gap concentrated on issues with repeated failure signatures, with near parity on single-attempt-solvable issues.

**vs. Uniform-budget collector** — On a case where one branch reaches a promising but under-explored subgoal while others wander, the uniform collector keeps all branches funded equally and lets the promising branch exhaust its slice before exploiting the subgoal. Our method withdraws from below-median branches and caps the recipient at B/2, so the promising branch keeps running. Because total compute is identical, we expect the gain to appear specifically as more solved issues per unit budget, not as more total steps, and we report the pass@1 vs. total steps frontier to show the efficiency gain.

**vs. SWE-Trail replace-action collector** — On a case where the agent's sampling distribution is already reasonable but compute is spread thin across many shallow branches, SWE-Trail changes what each branch samples without changing how much any branch holds, so it cannot deepen a promising branch. Our method instead reallocates conserved budget, letting one branch run longer. We expect our advantage to show on deep issues requiring long solution chains, with parity on shallow issues where distribution alone suffices.

**vs. UCB-style allocator** — On a case where the trigger score is noisy, the UCB-style allocator may over-explore high-uncertainty branches, while our proportional-withdrawal rule concentrates on below-median branches. We expect our method to match or exceed UCB on issues with clear failure signatures and to be more conservative on novel branches due to the novelty floor.

**vs. Random-trigger ablation** — If the random-trigger ablation matches the real trigger, the claimed value signal is noise. We expect the real trigger to outperform random by a margin concentrated on issues with repeated near-state failures.

**vs. No-novelty-floor ablation** — Without the novelty floor, branches early in novel trajectories may be drained prematurely. We expect the full method to outperform the no-floor variant on issues where the solution requires exploring a novel subgoal.

### What would falsify this idea

If BUDGET-TRAIL's pass@1 gain over the uniform-budget collector is uniform across issue subsets rather than concentrated on issues whose history contains repeated near-state failures, the trigger-score mechanism is not the cause. Equally, if the random-trigger ablation matches the real trigger, or if the calibration set shows Spearman correlation below 0.3 between `s(obs)` and marginal value, the claimed value signal is noise.

## References

1. SWE-Trail: History-Guided Trajectory Collection for Coding Agent Training
