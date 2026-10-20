# Belief-Conditioned Monotone Oracle Scoring for LLM Agent Trajectories in ORSim

## Motivation

Existing agent benchmarks, including OR-License, evaluate a final artifact against a single static oracle, so they cannot distinguish an agent that reasons well from one that merely guesses well or gets lucky with the order of information it happens to request; OR-License's ORSim added dialogue and information boundaries but kept the hidden expert solution as a fixed reference, leaving the structural property of monotone consistency under belief refinement untested. The root cause is that no existing scoring kernel conditions the oracle on the information revealed so far, so there is no way to ask whether a revised solution is better than the previous one given strictly more information — a property that recurs across dialogue benchmarks, interactive planning, and information-acquisition evaluations. Without it, a dialogue wrapper can pass while an agent's decisions are insensitive to what it learns.

## Key Insight

If the oracle is re-solved at every information state on exactly the revealed facts, then the sequence of oracle-relative scores forms a monotone reference ladder, and any agent whose score drops when its belief refines is provably making a strictly dominated revision — a signal a static final-artifact check cannot produce.

## Method

**What it is.** We introduce BAMOS (Belief-Aware Monotone Oracle Scoring), a trajectory-level scoring kernel for ORSim. Its inputs are the full interaction trace (the sequence of client-revealed fact sets and the agent's submitted solution at each state) and the instance's hidden ground-truth fact partition; its outputs are a per-state oracle-relative score, a monotonicity violation count, and an aggregate that penalizes dominated revisions.

**How it works.**

```pseudocode
Input: trace T = [(F_1, s_1), (F_2, s_2), ..., (F_n, s_n)]
       F_k = facts revealed up to turn k (monotone: F_1 subset ... subset F_n)
       s_k = agent solution submitted after F_k
       G = ground-truth fact partition of the instance
       ORA = fixed OR solver (exact MILP) used as oracle
       eps = 1e-6 feasibility tolerance

1. BELIEF-CONDITIONED ORACLE LADDER
   for k = 1..n:
       z*_k = ORA(F_k)                 # exact optimum on revealed facts only
       q_k  = quality(s_k, F_k)        # feasibility check + objective on F_k
       rho_k = q_k / z*_k              # oracle-relative score, clipped to [0,1]

2. MONOTONICITY TEST (counterfactual dominance check)
   viol = 0
   for k = 2..n:
       # Evaluate the agent's previous solution on the current fact set
       q_prev = quality(s_{k-1}, F_k)  # feasibility-projected if needed
       if q_k < q_prev - eps:
           viol += 1
           gap_k = q_prev - q_k        # magnitude of dominated revision

3. AGGREGATE
   BAMOS = mean_k(rho_k) - lambda * sum_k(gap_k)   # lambda = 0.5 (default; sensitivity over {0.1,0.25,0.5,1.0,2.0} reported)
   also report: final_score = rho_n, viol_rate = viol/(n-1)

4. ORACLE-SANITY GATE (per instance, offline)
   assert z*_k is non-decreasing in k under F_k subset F_{k+1}
   # if the oracle ladder itself is non-monotone, flag instance as
   # information-inconsistent and exclude from aggregate
```

**Assumption (faithfulness of revealed facts).** The quantity rho_k = q_k / z*_k measures belief-conditioned solution quality because the oracle z*_k is solved on exactly the facts revealed at turn k, under the assumption A that the revealed fact set is a faithful sufficient statistic for the client's information state (no redundant reveals, no hidden facts affecting feasibility). This assumption fails when the client simulator reveals a fact that is redundant given earlier facts, in which case z*_k is unchanged and rho_k reflects only the agent's re-optimization noise rather than genuine belief refinement; we quantify the frequency of redundant reveals per instance and report it alongside BAMOS. The violation count measures dominated revision because a drop in q_k relative to q_prev on the same fact set F_k implies the agent's new solution is worse than its own prior solution on a strictly larger feasible set, under the assumption that the oracle ladder is monotone (verified in step 4); this assumption fails when the oracle solve is censored by timeout, in which case the transition is skipped and the metric reflects only solved transitions. The magnitude gap_k measures the severity of information-insensitivity because a large gap means the agent discarded a solution that was near-optimal for a smaller fact set in favor of one far from optimal for a larger one, under the assumption that the agent's revision is deterministic given its context; this assumption fails when the agent samples stochastically, in which case gap_k reflects sampling variance and we report it alongside a fixed-seed rerun.

**Why this design.** We chose an exact MILP oracle re-solved per information state over reusing OR-License's single static expert solution because the static solution is only optimal for the full fact set, so it cannot rank partial-belief solutions and would make every early-state score an artifact of how much the agent happened to reveal; the cost is that oracle re-solving multiplies compute by the number of turns, which we accept because instances are small and the ladder is computed offline once per instance. We chose a ratio scale (q_k / z*_k) over a raw objective because raw objectives are incomparable across information states where the feasible region shrinks, accepting that ratios saturate near 1.0 and lose resolution for near-optimal agents. We chose a hard monotonicity violation count with a magnitude penalty over a soft ranking loss because violation is a discrete, auditable event that a human reviewer can inspect, accepting that a single noisy oracle solve can flip a violation; the censoring rule in step 4 and the eps tolerance mitigate this. We chose to gate on oracle-ladder monotonicity (step 4) rather than assume it, because if the oracle itself is non-monotone the instance's fact partition is inconsistent and any agent score is meaningless; this is a property no prior benchmark checks, and it is why BAMOS is not a variant of standard process-reward or per-step Monte-Carlo value estimation (VinePPO, PRMs), which estimate step value from rollouts of the policy rather than from an exogenous oracle conditioned on a controlled information partition.

**Why it measures what we claim.** The quantity rho_k = q_k / z*_k measures belief-conditioned solution quality because the oracle z*_k is solved on exactly the facts revealed at turn k, under the assumption that the revealed fact set is a faithful sufficient statistic for the client's information state; this assumption fails when the client simulator reveals a fact that is redundant given earlier facts, in which case z*_k is unchanged and rho_k reflects only the agent's re-optimization noise rather than genuine belief refinement. The violation count measures dominated revision because a drop in q_k relative to q_prev on the same fact set F_k implies the agent's new solution is worse than its own prior solution on a strictly larger feasible set, under the assumption that the oracle ladder is monotone (verified in step 4); this assumption fails when the oracle solve is censored by timeout, in which case the transition is skipped and the metric reflects only solved transitions. The magnitude gap_k measures the severity of information-insensitivity because a large gap means the agent discarded a solution that was near-optimal for a smaller fact set in favor of one far from optimal for a larger one, under the assumption that the agent's revision is deterministic given its context; this assumption fails when the agent samples stochastically, in which case gap_k reflects sampling variance and we report it alongside a fixed-seed rerun.

## Contribution

(1) BAMOS, a belief-conditioned scoring kernel that re-solves an exact oracle at every information state and produces a monotone reference ladder and a dominated-revision count, turning ORSim from a static final-artifact check into a trajectory-level test of information-sensitive decision-making. (2) An oracle-sanity gate that detects information-inconsistent instances whose fact partitions violate oracle monotonicity, providing a reusable design principle for any dialogue benchmark that conditions an oracle on revealed information. (3) A per-state diagnostic (oracle-relative score, violation rate, gap magnitude) that separates agents which improve with information from agents which merely guess well, applicable to any coding-agent harness evaluated on OR-License.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | OR-License instances via ORSim | Multi-turn belief refinement with hidden fact partitions |
| Primary metric | BAMOS aggregate (mean rho minus dominated-revision penalty) | Jointly captures quality and monotone belief use |
| Baseline | Expert-solution DAQ (OR-License) | Static-reference scoring ignores partial-belief states |
| Baseline | Final-turn rho_n only | Tests whether trajectory-level signal adds over endpoint |
| Baseline | Process-reward / PRM step scoring | Tests oracle-conditioned scoring versus rollout value |
| Ablation-of-ours | BAMOS without monotonicity penalty (mean rho only) | Isolates contribution of the dominated-revision term |
| Ablation-of-ours | BAMOS with ratio-based violation (original rho_k < rho_{k-1}) | Tests counterfactual dominance vs ratio comparison |
| Sensitivity | lambda in {0.1, 0.25, 0.5, 1.0, 2.0} | Reports rank correlation with held-out judgments |
| Compute budget | 50 instances, 10 turns each, 60s solver cap per state | Total solver-hours = 50*10*60/3600 ≈ 8.3 hours |
| Fallback solver | HiGHS/CBC with 1% optimality gap tolerance | Reproducible without commercial license |

### Why this setup validates the claim

The central claim is that belief-conditioned oracle-relative scoring plus a dominated-revision penalty detects a failure mode—agents that ignore newly revealed facts and revise to dominated solutions—that static-reference and endpoint metrics cannot see. OR-License is the right dataset because its ORSim client reveals facts incrementally and its hidden partition lets us re-solve the oracle per information state, which is exactly the controlled ladder BAMOS requires; without a genuine partial-belief structure the ladder collapses and the metric is vacuous. The expert-solution DAQ baseline tests the sub-claim that a single full-fact reference cannot rank partial-belief solutions, since it scores every turn against one static optimum. The final-turn-only baseline tests whether the trajectory aggregation is doing real work or whether the last state already carries the signal. The PRM baseline tests the sub-claim that an exogenous oracle conditioned on a controlled partition differs from policy-rollout step value, which is our sharpest distinction from prior process-reward work. The mean-rho ablation isolates the monotonicity penalty: if the full BAMOS does not separate agents more than mean-rho alone, the dominated-revision term is not measuring what we claim. The ratio-based violation ablation tests whether the counterfactual dominance check (q_k < q_prev on F_k) is necessary versus the original ratio comparison (rho_k < rho_{k-1}), which can be non-monotone even for optimal agents. BAMOS is the correct primary metric because it is the only one that can register a drop in q_k relative to q_prev on the same fact set, which is the observable signature of the failure mode; a metric that cannot decrease along the ladder cannot falsify or confirm the claim.

### Expected outcome and causal chain

**vs. Expert-solution DAQ (OR-License)** — On a case where the client reveals a binding constraint only at turn four, the baseline scores early submissions against the full-fact expert optimum, so an agent that happened to submit a strong guess early looks indistinguishable from one that genuinely re-optimized, because its mechanism has no partial-belief reference. Our method instead re-solves the oracle on exactly the revealed facts at each turn, so the early lucky guess scores near 1.0 on a small feasible set and the later dominated revision shows as a drop in q_k relative to q_prev on F_k. We expect a noticeable BAMOS gap between high- and low-reasoning models concentrated on instances with late-revealed binding facts, and near-parity on instances where all facts are revealed at turn one.

**vs. Final-turn rho_n only** — On a case where an agent submits a near-optimal solution at turn two and then revises to a worse one after a redundant reveal, the baseline records only the final state and reports the agent as mediocre, because its mechanism discards the trajectory and cannot distinguish a steady improver from a late collapse. Our method keeps the per-state ladder and the magnitude penalty, so the turn-two peak and the subsequent dominated revision are both visible and the penalty fires. We expect the largest separation on instances with redundant or late reveals, where endpoint scoring and BAMOS disagree in rank order, and parity on monotone improvers.

**vs. Process-reward / PRM step scoring** — On a case where the policy's own rollouts are uninformative because the agent's exploration is narrow, the baseline assigns near-uniform step values from policy rollouts and cannot tell a fact-ignoring revision from a fact-using one, because its mechanism estimates value from the policy rather than from an exogenous oracle on a controlled partition. Our method instead anchors each step to the exact MILP optimum on the revealed facts, so a revision that ignores a new constraint is scored against a strictly larger feasible set and registers as a violation. We expect BAMOS to flag dominated revisions that the PRM baseline scores as neutral, with the gap concentrated on instances where the oracle ladder is verified monotone in step four.

**vs. BAMOS without monotonicity penalty (mean rho only)** — On a case where an agent makes a dominated revision after a redundant reveal, the mean-rho ablation ignores the drop because it averages rho_k without penalizing the violation, so it cannot distinguish a fact-ignoring agent from a fact-using one when the average quality is similar. Our method adds the magnitude penalty, so the dominated revision is explicitly penalized. We expect the full BAMOS to separate high- and low-reasoning models more than mean-rho alone, with the gap concentrated on instances with redundant reveals.

**vs. BAMOS with ratio-based violation (original rho_k < rho_{k-1})** — On a case where a correct agent revises optimally at every turn but the ratio rho_k decreases because the newly revealed fact is binding and the denominator z*_k grows faster than the numerator, the ratio-based violation falsely flags a dominated revision, because it compares ratios with different denominators. Our method uses the counterfactual dominance check (q_k < q_prev on F_k), which holds the comparison set fixed and only flags a violation when the new solution is worse than the previous one on the same fact set. We expect the ratio-based violation to produce false positives on instances with binding late reveals, while the counterfactual check remains accurate.

### What would falsify this idea

If BAMOS's advantage over the expert-DAQ and final-turn baselines is uniform across all instance subsets rather than concentrated on instances with late or redundant fact reveals, then the metric is not detecting belief-conditioned dominated revision and the central claim is wrong. Likewise, if the mean-rho ablation matches full BAMOS in separating models, the monotonicity penalty is not measuring the failure mode we claim.

## References

1. OR-License: Evaluating LLM Agents as Real-World Operations Research Practitioners
