# Cross-Fold Transfer-Gap Operator Selection for MLE-Bench Search Policies

## Motivation

Existing LLM research agents on MLE-bench are formalized as search policies over isolated competition episodes, where each candidate solution is scored against a single held-out split and operators are rewarded by fold-local medal rate (E1, E5). This episode-as-isolated-task abstraction bakes in an i.i.d. assumption that no operator can be rewarded for cross-unit transfer, so the very generalization the field cares about is measured post hoc rather than optimized. The deployment-style evidence from PHM benchmarks (E4, E7) shows that an agent's advantage is a per-unit design that transfers only weakly, confirming that fold-local medal rate is a structurally biased objective for operator selection.

## Key Insight

Because a search operator's fold-local medal rate and its cross-fold transfer gap are separable quantities, selecting operators by the transfer gap creates a selection pressure for operators whose edits encode unit-invariant structure rather than fold-specific overfitting, which greedy medal-rate selection cannot express.

## Method

**What it is.** LOUO-Select is an operator-selection procedure for MLE-bench-style LLM research agents. Its inputs are a pool of candidate search operators (Greedy, MCTS, Evolutionary families, optionally combined with operator sets from E5) and a set of Kaggle competitions partitioned into unit-held-out folds; its output is a ranked operator set and a per-operator transfer-gap score used to drive the agent's search loop.

**How it works.**

```pseudocode
INPUT: operators O = {o_1..o_K}, competitions C, folds F = {f_1..f_M}
       (each fold holds out disjoint units/rows of C), budget B per fold
HYPER: K_ops = |O|, M folds, transfer-gap threshold tau, warm-start size W,
       variance weight lambda, reference operator o_ref (default: Greedy)

0. GATE: For each competition c in C, compute a unit-structure diagnostic:
   if c has no group/unit column (single-table i.i.d. rows), mark c as
   non-foldable and EXCLUDE from operator selection; report the fraction
   of foldable competitions. For foldable competitions, construct M=3
   folds by grouping on the unit column (e.g., patient, engine, user).

1. For each operator o_k in O:
     for each fold f_m in F:
        run agent search loop with o_k on C \ f_m, budget B
        record candidate solution s_{k,m} and its fold-local medal rate r_{k,m}
        evaluate s_{k,m} on held-out fold f_m -> score t_{k,m}
     # Paired, difficulty-normalized contrast:
     for each fold f_m:
        compute reference score t_ref,m = t_{k,m} for o_ref
        compute paired difference d_{k,m} = t_{k,m} - t_ref,m
     compute transfer gap as mean paired difference:
        G_k = mean_m d_{k,m}                    # lower = better transfer
     compute cross-fold variance of paired differences:
        V_k = std_m d_{k,m}
     transfer_score[k] = G_k + lambda * V_k     # lambda calibrated (see step 5)

2. Rank operators by transfer_score ascending (lower gap = better transfer).

3. Warm-start the search loop: for the current episode, seed the agent with
   the top-W operators (W=2) and their recorded s_{k,m} plans as plain-text
   design priors (following the plan-transfer mechanism in E4).

4. During the live episode, after each operator invocation, recompute the
   running transfer_score using the episode's own unit-split folds and
   down-weight operators whose gap exceeds tau (tau calibrated, see step 5).

5. CALIBRATION (held-out folds only): Using a disjoint set of calibration
   competitions C_cal (not used in the main evaluation), sweep
   lambda in {0.0, 0.25, 0.5, 1.0} and tau in {0.05, 0.10, 0.15, 0.20}
   and select the pair maximizing held-out medal rate on C_cal. Report the
   selected values and the full sensitivity grid in the experiment.

OUTPUT: ranked operator set, transfer_score vector, live operator schedule.
```

**Why this design.** We chose a leave-one-unit-out fold construction over a single held-out split because the single split conflates operator quality with fold-specific luck, accepting the cost that each operator must be run M times, multiplying search budget by M. We chose to score operators by a paired, difficulty-normalized contrast against a reference operator rather than by raw fold-local minus held-out medal rate because raw medal rates are thresholded against a fixed per-competition leaderboard and are not on a common scale across folds; the paired difference removes fold difficulty as a common-mode term, accepting the cost that the gap is noisier than a single score and needs variance regularization (the lambda term). We chose to warm-start the live episode with the top-W operators' plain-text plans rather than restarting search from scratch, following the plan-transfer evidence in E4, accepting the cost that a bad top-W selection propagates into the episode. We chose to calibrate lambda and tau on a disjoint calibration set rather than fixing them per competition because per-competition tuning reintroduces the fold-local overfitting we are trying to remove, while a single global calibration set keeps the constants fixed at evaluation time. A domain expert would not describe this as a variant of VinePPO or process-reward-model step scoring, because those methods assign credit to individual steps within one episode; LOUO-Select assigns credit to *operators* by comparing their behavior across held-out units, a cross-fold quantity that step-level value estimation does not compute.

**Why it measures what we claim.** The transfer gap G_k = mean_m(t_{k,m} - t_ref,m) measures an operator's cross-unit generalization because it assumes that the paired difference against a fixed reference operator on the same held-out fold cancels fold difficulty and evaluation-convention artifacts, leaving a quantity attributable to the operator's edits; this assumption fails when the reference operator itself behaves differently across folds (e.g., Greedy saturates on easy folds), in which case G_k reflects reference-operator instability rather than the target operator's overfitting. The cross-fold variance V_k measures operator stability because it assumes the M folds are exchangeable samples of the same unit-holdout process; this assumption fails when folds have very different sizes or class balances, in which case V_k reflects fold heterogeneity rather than operator instability. The warm-start plan transfer measures design-knowledge portability because it assumes the plain-text plan captures the operator's unit-invariant decisions (following E4's plan-transfer finding); this assumption fails when the plan encodes fold-specific hyperparameters, in which case the warm-start reflects leakage rather than transferable design. The gate in step 0 measures whether a competition admits meaningful unit-held-out folds because it assumes a group/unit column exists; this assumption fails on single-table i.i.d. competitions, in which case the transfer gap is identically zero and the method is not applied.

## Contribution

(1) A leave-one-unit-out meta-objective and operator-selection procedure (LOUO-Select) that scores search operators by cross-fold transfer gap rather than fold-local medal rate, making cross-unit generalization an optimized quantity inside the MLE-bench search loop. (2) The finding that operator ranking by transfer gap diverges from ranking by fold-local medal rate, establishing that the two objectives select structurally different operator sets. (3) A reusable fold-construction and transfer-gap scoring protocol for MLE-bench-style benchmarks that other agent designers can adopt to report operator-level generalization.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | MLE-bench lite, restricted to competitions with a unit/group column (foldable subset); report the foldable fraction | Real Kaggle tasks with foldable units; gate excludes i.i.d.-row competitions where the gap is identically zero |
| Dataset (cross-benchmark) | NextUnit-Bench (E4/E7) deployment-style folds | Tests whether transfer-gap operator selection is a general principle, not a Kaggle-specific trick |
| Primary metric | Held-out medal rate (fraction of competitions medaling) on foldable competitions | Directly reflects cross-unit generalization |
| Secondary metric | Signal-to-noise ratio of G_k: mean_k |G_k| / std_m(d_{k,m}) per competition | Reports whether the gap estimate is separable from fold noise |
| Secondary metric | Spearman correlation between G_k ranking and realized held-out medal rate across competitions | Directly tests the load-bearing assumption that G_k predicts held-out medal rate |
| Baseline | Best fixed operator set from E5 (Greedy/MCTS/Evolutionary) | Tests whether adaptive selection beats any static policy |
| Baseline | MLE-STAR-style targeted refinement (E2) | Tests whether operator-level credit beats coarse whole-code refinement |
| Baseline | Single held-out split operator selection | Tests whether LOUO paired contrast beats naive split scoring |
| Baseline | Raw-gap LOUO-Select (G_k = mean_m(r_{k,m} - t_{k,m}), unpaired) | Tests whether the paired, difficulty-normalized contrast is necessary |
| Ablation-of-ours | LOUO-Select without warm-start plan transfer | Isolates contribution of the E4 plan-transfer mechanism |
| Ablation-of-ours | LOUO-Select with lambda=0 and tau=0 (no variance penalty, no live down-weighting) | Isolates contribution of the regularization terms |
| Cost control | Total agent-episode budget capped identically across LOUO-Select and all baselines; main table uses K=3 operators, M=3 folds; budget-scaling curve reported | Ensures advantage is shown at matched compute, not at M-times the baseline budget |

### Why this setup validates the claim

The central claim is that selecting and scheduling operators by their cross-unit transfer gap yields better held-out medal rates than any fixed operator policy or single-split selection. The MLE-bench-lite foldable subset with unit-held-out folds is the right substrate because the medal-rate metric is exactly the quantity the transfer gap is designed to predict, so a mismatch between predicted ranking and realized held-out medal rate is directly observable; the gate ensures we do not evaluate on competitions where the gap is structurally zero. The best-fixed-operator baseline tests the sub-claim that adaptivity matters at all; if a single static operator set matches LOUO-Select, the ranking machinery is inert. The MLE-STAR-style baseline tests the sub-claim that operator-level credit assignment beats coarse whole-structure refinement. The single-split baseline tests the sub-claim that the gap score, not merely held-out score, is the right selection signal. The raw-gap baseline tests the sub-claim that pairing against a reference operator is necessary to cancel fold difficulty. The no-warm-start ablation isolates whether the E4 plan-transfer mechanism actually contributes, or whether the ranking alone carries the gain. The lambda/tau ablation isolates whether the regularization terms contribute. The Spearman-correlation diagnostic directly tests the load-bearing assumption that G_k predicts held-out medal rate better than held-out score alone; if the correlation is near zero, the selection signal is an unverified proxy. The NextUnit-Bench experiment tests whether the mechanism generalizes beyond Kaggle. Together these form a falsifiable test: the method must beat each baseline on the specific failure mode that baseline's mechanism cannot address, at matched compute.

### Expected outcome and causal chain

**vs. Best fixed operator set (E5)** — On a competition where the optimal search behavior differs by unit (e.g. one fold rewards broad feature exploration, another rewards deep hyperparameter refinement), the fixed operator set produces a single search trajectory tuned to the average fold, because it cannot condition operator choice on observed held-out behavior. Our method instead recomputes transfer scores per episode and down-weights operators whose gap exceeds tau, because the live unit-split folds expose which operator is overfitting the current fold. We expect a noticeable gap on competitions with heterogeneous unit structure and near-parity on homogeneous ones.

**vs. MLE-STAR-style targeted refinement (E2)** — On a competition where the initial retrieved model is strong but individual code blocks need component-specific exploration, the MLE-STAR-style agent modifies whole code structures coarsely, because it has no operator-level signal distinguishing which exploration family is generalizing. Our method instead ranks Greedy, MCTS, and Evolutionary operators by their paired cross-fold transfer gap and warm-starts with the top-W plans, because the gap score isolates operator-level overfitting from fold luck. We expect our gain to concentrate on competitions requiring deep component-level search, with parity where a single strong initial model suffices.

**vs. Single held-out split selection** — On a competition where one fold happens to have an unusually easy class balance, the single-split selector picks the operator that wins that fold, because it conflates operator quality with fold-specific luck. Our method instead averages the paired difference across M folds and penalizes cross-fold variance via lambda, because the LOUO construction separates operator quality from fold identity. We expect the single-split baseline to show high variance across competitions and our method to show a steadier medal rate, with the largest gap on competitions where fold heterogeneity is high.

**vs. Raw-gap LOUO-Select (unpaired)** — On a competition where fold difficulty varies (e.g. one fold has a harder leaderboard threshold), the raw gap G_k = mean_m(r_{k,m} - t_{k,m}) is dominated by the fold-difficulty term, because r_{k,m} and t_{k,m} are thresholded against different leaderboard scales. Our method instead uses the paired difference t_{k,m} - t_ref,m on the same held-out fold, because the reference operator cancels the common-mode fold difficulty. We expect the raw-gap baseline to show high variance and near-zero Spearman correlation with held-out medal rate, while our paired contrast shows a positive correlation.

### What would falsify this idea

1. If the no-warm-start ablation matches the full method, the E4 plan-transfer mechanism contributes nothing and the claimed causal chain is wrong. If LOUO-Select's gain over the best fixed operator set is uniform across homogeneous and heterogeneous competitions rather than concentrated on heterogeneous ones, the transfer-gap mechanism is not doing the work we claim. If the Spearman correlation between G_k ranking and realized held-out medal rate is near zero, the transfer gap is an unverified proxy and the selection signal is invalid. If the paired contrast does not outperform the raw-gap baseline, the difficulty-normalization is not necessary.

## References

1. Moving Generality from Model to Modeler: LLM Agents for PHM on a Deployment-Style Benchmark
