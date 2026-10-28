# Turn-Decomposed Credit Identification from Terminal Rewards via Observable Transition Kernels

## Motivation

Across the RL-for-language-agents literature, every method that assigns turn- or step-level credit — MPO's Multiscale Advantage Estimation, Math-Shepherd's step-wise PPO, GSPO's sequence-level advantages — presupposes a verifiable reward at the granularity at which it optimizes; MPO in particular explicitly anchors advantages at turns while assuming a turn-level reward signal exists, and never asks whether a coarser terminal-only signal uniquely determines those turn-level credits. The structural gap is that no work models the terminal reward as a decomposition over turns and asks the identifiability question: when does one scalar per trajectory pin down per-turn advantages? Because this question is never posed, turn-anchored estimation is only well-posed when a per-turn oracle is already available, which defeats the purpose for the many agent tasks where only final success is checkable.

## Key Insight

Treating the terminal verifiable reward as a sum of turn-level latent rewards, combined with the fact that the policy's own turn-transition kernel is observable from the same rollouts, yields an over-determined linear system whose per-turn advantage solution is unique up to exactly one gauge degree of freedom — the per-trajectory constant shift that is precisely the critic baseline — so terminal-only supervision identifies turn-level credit without any per-turn oracle.

## Method

**What it is.** TIDE (Turn-level Identification from Decomposed End-rewards) is an advantage estimator for turn-structured LLM agents that takes as input a batch of rollouts (each a sequence of turns with observable next-turn state transitions and a single terminal scalar reward) and outputs per-turn advantage estimates, which are then fed to a standard turn-level policy-gradient update. It requires no per-turn reward model.

**Load-bearing assumption (stated explicitly).** TIDE assumes the terminal scalar reward is *additive over turns*: R = sum_{t=1..T} r_t for latent per-turn rewards r_t, with a turn-value Phi(s_t) = E[sum_{k>=t} r_k | s_t] that is a function of the observable turn state alone. Under this assumption the stacked Bellman system A Phi = R holds exactly and its null space is only the per-trajectory constant gauge. This assumption is *not* guaranteed in tool-use/search agents, where a binary success flag is a conjunction/threshold over intermediate correctness rather than a sum. TIDE therefore treats additivity as a *testable* premise: it reports the per-trajectory residual ||A Phi* - R|| as a goodness-of-additivity diagnostic and gates trust in its advantages on that residual (Phase 5). When the residual exceeds a threshold, TIDE falls back to a counterfactual/leave-one-turn-out credit estimator (Phase 6) rather than the linear solve.

**How it works.**

```pseudocode
Input: rollouts D = {(s_1,a_1,s_2,...,s_T, R)}  # s_t = turn state (context+observation), a_t = turn action, R = terminal scalar
Hyperparams: ridge lambda=1e-2, gauge tol=1e-6, kernel bins M=64 per state-feature,
             state embedding dim d=32 (last-layer hidden state projected),
             residual threshold tau_res = 0.1 * mean_t |R_i|, CG tol=1e-4, CG max-iter=500

Phase 0 — State embedding (bounds |S|):
  Map each turn state s_t to a d=32 vector via a fixed linear projection of the
  last-layer hidden state (no training; projection matrix frozen at init).
  Phi is defined on these d=32 embeddings, so |S| <= number of distinct embedded states in D.

Phase 1 — Estimate observable turn-transition kernel:
  For each observed (s_t, a_t) pair, collect empirical distribution
  K_hat(s_{t+1} | s_t, a_t) by binning the d=32 embedded s_{t+1} into M=64 feature bins.
  (This is a lookup over the SAME rollouts; no extra sampling.)

Phase 2 — Build the decomposition system:
  Model R = sum_{t=1..T} r_t  with latent per-turn rewards r_t.
  For each turn t define the potential Phi(s_t) = E[ sum_{k>=t} r_k | s_t ] (turn-value).
  The Bellman relation r_t = Phi(s_t) - E_{s_{t+1}~K_hat}[Phi(s_{t+1})] holds for the true Phi.
  Substituting into R gives, for every trajectory i:
      R_i = Phi(s_1^i) + sum_{t=1..T_i-1} [ Phi(s_t^i) - E_{K_hat}Phi(s_{t+1}^i) ] - Phi(s_{T_i+1})   (terminal Phi=0)
  Stack all trajectories -> linear system A Phi = R over the |S|-dim vector Phi.

Phase 3 — Solve up to gauge:
  Solve ridge-regularized least squares: Phi* = argmin ||A Phi - R||^2 + lambda||Phi||^2.
  For |S| <= 2000 use dense solve; for |S| > 2000 use conjugate-gradient (tol=1e-4, max-iter=500).
  The null space of A is exactly the constant vector (one gauge d.o.f. per trajectory).
  Fix gauge by the advantage-centering constraint: for each rollout, subtract the
  rollout-mean of Phi so that sum_t A_t = 0 (this is the standard baseline).

Phase 4 — Emit turn advantages:
  A_t = Phi*(s_t) - E_{K_hat}[Phi*(s_{t+1})]  - baseline(s_t),  baseline = centered Phi*.
  Return {A_t} for the turn-level policy-gradient step.

Phase 5 — Additivity diagnostic and gating:
  For each trajectory i compute residual_i = ||A_i Phi* - R_i||.
  Report mean residual and condition number of A as a function of M and d.
  If residual_i > tau_res, mark trajectory i as non-additive.

Phase 6 — Fallback for non-additive trajectories:
  For trajectories flagged in Phase 5, replace the linear-solve advantage with a
  leave-one-turn-out counterfactual estimate computed from the SAME batch:
  A_t^CF = R_i - R_i^{(t removed)}, where R_i^{(t removed)} is the terminal reward
  of the nearest rollout in D that matches trajectory i on all turns except t
  (matched on the d=32 embedding within L2 distance 0.5).
  If no match exists, set A_t^CF = 0 for that turn.
```

**Why this design.** We chose to estimate the transition kernel empirically from the existing rollout batch rather than fitting a parametric dynamics model, because the identifiability argument only needs the kernel to be *observable* (a counting estimate suffices), and a parametric model would add a second estimation error that contaminates the linear system; we accept the cost that in high-dimensional state spaces the empirical kernel is coarse and the ridge term must absorb the resulting bias. We chose a ridge-regularized least-squares solve over an iterative Bellman backup because the decomposition system is a global linear system whose null space we need to characterize explicitly (the gauge), whereas iterative backup would silently absorb the gauge into initialization and never expose it; we accept the cost of an O(|S|^3)-ish solve on the reachable state set, mitigated by restricting to states visited in the batch and by the d=32 embedding that bounds |S|. We chose to fix the gauge with advantage-centering rather than an arbitrary anchor turn, because centering is exactly the critic baseline and is already required by the policy-gradient estimator, so the gauge becomes a free parameter with a known interpretation rather than a new hyperparameter; we accept that centering removes any genuine per-trajectory constant reward offset, which is harmless because a constant offset does not change the policy gradient. We chose to gate the linear solve on the residual and fall back to a counterfactual estimator for flagged trajectories, because the additive decomposition is a premise that the target domain can violate; we accept the cost of a second estimator, which is computed from the same batch and adds no new sampling. A domain expert would not describe this as a variant of MPO or VinePPO: MPO assumes the turn-level reward is given, and VinePPO estimates values by Monte-Carlo rollouts from each prefix — neither poses or solves the decomposition-identifiability problem, and neither exposes a gauge that is identified with the baseline.

**Why it measures what we claim.** The least-squares solution Phi* measures the turn-value function because the decomposition assumption R = sum_t r_t plus the Bellman relation makes the stacked system A Phi = R hold exactly for the true turn-value; this assumption fails when the terminal reward is not additive over turns (e.g., a single turn's correctness dominates the whole trajectory), in which case Phi* reflects a best-additive-fit projection rather than the true turn-value, and the Phase 5 residual ||A Phi* - R|| grows — which is why TIDE reports that residual and falls back to the Phase 6 counterfactual estimator for flagged trajectories. The empirical kernel K_hat measures the true turn-transition kernel because the rollouts are on-policy and therefore sample s_{t+1} from the policy's own transition distribution; this assumption fails under off-policy or heavily truncated rollouts, in which case K_hat reflects the behavior policy's transitions and the residual A Phi - R grows, which we monitor as a diagnostic. The gauge-fixing centering measures the critic baseline because any constant added to Phi leaves A Phi unchanged and centering is the unique choice making per-rollout advantages sum to zero; this assumption fails when a trajectory has a single turn, in which case the advantage is identically zero and that trajectory contributes no learning signal, which we handle by falling back to the terminal reward for single-turn rollouts. The Phase 6 counterfactual advantage A_t^CF measures per-turn credit under non-additivity because it directly ablates turn t and observes the change in terminal reward; this assumption fails when no matching rollout exists in the batch, in which case A_t^CF is set to zero and that turn contributes no learning signal.

## Contribution

(1) A turn-level advantage estimator, TIDE, that recovers per-turn credit from a single terminal verifiable scalar by solving an observable-kernel decomposition system, replacing the per-turn reward oracle assumed by MPO and step-wise PPO with a solvable inverse problem. (2) The identification result that the decomposition system's null space is exactly the per-trajectory constant shift, which coincides with the critic baseline, giving a principled and hyperparameter-free gauge fix. (3) A diagnostic residual (the norm of A Phi* - R) that flags when the additive-decomposition assumption is violated, enabling selective application of turn-level credit only where it is identifiable.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Multi-turn tool-use / search agent tasks with terminal-only binary reward, horizon T in {3,5,8,12,16} | Matches turn-structured assumption and sparse terminal signal; spans horizon scaling. |
| Primary metric | Success rate vs. turn-budget curve, reported per horizon bucket | Detects credit-assignment efficiency, not just final accuracy. |
| Secondary metric | Per-trajectory additivity residual ||A Phi* - R|| and condition number of A vs. M and d | Tests the load-bearing additivity assumption directly. |
| Baseline | GRPO (token-level group baseline) | Tests whether turn-level decomposition beats token-level credit. |
| Baseline | VinePPO (Monte-Carlo prefix values) | Tests whether decomposition beats rollout-based value estimation. |
| Baseline | Uniform terminal-reward broadcast | Tests whether any turn-credit signal beats naive spreading. |
| Ablation-of-ours | TIDE with shuffled-kernel (broken K_hat) | Isolates the empirical transition kernel as the causal ingredient. |
| Ablation-of-ours | TIDE with residual gating disabled (always linear solve) | Isolates the Phase 5/6 additivity gate. |
| Validation | Synthetic MDP with known ground-truth per-turn rewards (T=5, |S|=50, binary terminal reward) | Establishes decomposition recovers true turn-values before LLM benchmark. |

### Why this setup validates the claim

The central claim is that TIDE recovers a usable turn-value function from terminal-only rewards by solving the additive decomposition system, and that this beats alternatives precisely where credit assignment across turns is ambiguous. GRPO tests the sub-claim that token-level group normalization cannot attribute credit across turns; if TIDE's advantage is real, it should win on tasks requiring several correct intermediate turns before a terminal reward. VinePPO tests the sub-claim that Monte-Carlo prefix value estimation is noisy under sparse terminal reward; TIDE should win where prefix rollouts are expensive or high-variance. The uniform-broadcast baseline tests whether any nontrivial turn structure matters at all. The shuffled-kernel ablation is the key falsification lever: it keeps the solver and gauge-fixing identical but destroys the transition kernel, so any gain that survives shuffling is not due to the decomposition mechanism. The residual-gating ablation tests whether the Phase 5/6 additivity gate is load-bearing: if disabling it does not hurt, the gate is unnecessary; if it helps, the additivity assumption is indeed violated in the target domain. The synthetic-MDP validation with known ground-truth turn rewards is the direct test of the load-bearing assumption: it measures whether Phi* recovers the true turn-value when additivity holds exactly, and the residual ||A Phi* - R|| should be near zero there. Success rate versus turn-budget is the right metric because the predicted effect is sample-efficiency of credit assignment, which shows up as reaching a given success rate with fewer turns of interaction, not as a higher ceiling on an easy task; reporting per horizon bucket tests whether the gain grows with horizon as the credit-assignment problem worsens.

### Expected outcome and causal chain

**vs. GRPO** — On a case where a correct final answer requires three correct tool calls but the terminal reward is binary, GRPO assigns the same group-normalized scalar to every token in the trajectory because it has no turn-level structure, so it reinforces wrong intermediate turns that happened to co-occur with success. Our method instead solves for per-turn Phi and emits advantages that isolate the turns whose transition kernel actually moved the trajectory toward the terminal reward, so we expect a noticeable gap on multi-turn tasks requiring ordered correct intermediates and near-parity on single-turn tasks. The gap should grow with horizon T, since GRPO's token-level normalization has no mechanism to distinguish turns as T increases.

**vs. VinePPO** — On a case with a long prefix and sparse terminal reward, VinePPO estimates each prefix value by Monte-Carlo rollouts, which are high-variance when the remaining horizon is long and the reward is rare, so its advantages are noisy and it wastes compute on redundant prefix sampling. Our method instead uses the empirical transition kernel already present in the batch to propagate terminal reward backward through the linear system, so we expect VinePPO to be competitive at short horizons but TIDE to pull ahead as horizon grows and reward sparsity increases. The per-horizon-bucket curve should show TIDE's advantage over VinePPO widening from T=3 to T=16.

**vs. Uniform terminal-reward broadcast** — On a case where one late turn is decisive and earlier turns are neutral, uniform broadcast spreads the terminal reward equally across turns, so it dilutes the decisive turn's signal and reinforces neutral turns as if they mattered. Our method instead assigns near-zero advantages to neutral turns because their Bellman residual under the fitted Phi is small, concentrating gradient on the decisive turn, so we expect a clear separation on tasks with one dominant decisive turn and parity on tasks where all turns contribute equally. The residual diagnostic should confirm this: on the decisive-turn subset, the additivity residual should be small (the additive fit is good), and TIDE's gain should be largest there; on the conjunctive subset where the residual is large, TIDE's Phase 6 fallback should carry the gain.

### What would falsify this idea

If TIDE's gain over GRPO and uniform-broadcast is uniform across single-turn and multi-turn subsets rather than concentrated on multi-turn tasks requiring ordered correct intermediates, the credit-assignment claim is wrong. If the shuffled-kernel ablation matches full TIDE, the decomposition mechanism is not the cause of any improvement. If the synthetic-MDP validation shows Phi* does not recover the known ground-truth turn rewards (high residual even when additivity holds exactly), the linear-solve mechanism is mis-specified. If the residual-gating ablation matches full TIDE, the additivity gate is unnecessary and the load-bearing assumption is not violated in the target domain.

## References

1. Multiscale Policy Optimization for Long-Horizon LLM Agents
2. Truncated Proximal Policy Optimization
3. Group Sequence Policy Optimization
4. DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models
5. OpenWebMath: An Open Dataset of High-Quality Mathematical Web Text
6. Math-Shepherd: Verify and Reinforce LLMs Step-by-step without Human Annotations
7. Solving Quantitative Reasoning Problems with Language Models
