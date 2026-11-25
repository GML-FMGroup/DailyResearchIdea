# Distributionally-Robust Calibration Certificates for Co-Evolving CUA Evaluation Environments

## Motivation

CUA-SEE couples three evolution objectives — cohort-calibrated difficulty/coverage growth, agent-conditional capability-boundary probes, and state-grounded verifier checkpoints — but treats them as separable branches, so the verifier's precision is never guaranteed to survive the task-distribution shift that benchmark evolution itself creates. This is a meta-level structural failure recurring across adaptive evaluation work: Agent-as-a-Judge (E4) likewise produces intermediate process judgments without any statement of when those judgments remain valid as the judged task population drifts. Because no branch enforces a joint consistency condition, the rollout-evidence-to-checkpoint mapping is only an empirical heuristic, and a checkpoint can silently stop tracking true capability boundaries as difficulty and coverage expand.

## Key Insight

Coupling the verifier-acceptance decision to a monotonicity constraint against the diagnostic capability-boundary probes, evaluated under the exact distribution shift induced by benchmark evolution, makes the rollout-evidence-to-checkpoint mapping identifiable rather than merely fitted — verifier precision cannot be raised by exploiting the shift without simultaneously breaking probe monotonicity.

## Method

**What it is.** We introduce DRCC (Distributionally-Robust Calibration Certificate), a wrapper around CUA-SEE's verifier-evolution step. Its inputs are the current verifier checkpoint set `V_t`, the diagnostic probe set `P_t` produced by capability-boundary evolution, the cohort task distributions `{D_c}` from benchmark evolution at round `t`, and fresh rollout evidence `R_t`. Its output is a three-valued certificate `(accept / reject / out-of-support, certified radius ε*)` for each candidate checkpoint, plus the accepted checkpoint set `V_{t+1}` and an explicit uncovered-task list `U_t`.

**Load-bearing assumption (stated plainly).** The certificate is valid only if the Wasserstein ball of radius `delta_t` around each cohort distribution `D_c` — with `delta_t` set by the coverage-growth budget — contains every task distribution reachable by one round of cohort-calibrated coverage growth, so that the worst-case precision drift `sup_{D in D_c^shift}[prec_v(D) - prec_v(D_c)]` upper-bounds the true degradation of a verifier checkpoint under benchmark evolution. This assumption fails when coverage growth introduces a task whose feature distribution lies outside the ball (e.g., a brand-new application modality); in that regime the finite-sample DRO dual has no mass to work with and the bound is vacuous. DRCC therefore does not silently apply the in-ball bound to such tasks: it detects them (step 2b) and routes them to an explicit `out-of-support` certificate rather than certifying them.

**How it works.**

```pseudocode
Input: V_t (verifier checkpoints), P_t (probes), {D_c}_{c=1..K} (cohort task dists),
       R_t (rollout evidence), alpha=0.05 (confidence), gamma=0.1 (monotonicity slack),
       eps_budget=0.05, n_boot=1000, block_len=8 (episodes per block),
       m=256 (feature dim for shift detection), mmd_alpha=0.01
Output: V_{t+1}, certificates {cert_v}, uncovered-task list U_t

1.  # Build the shift set: task distributions reachable by benchmark evolution
    for each cohort c:
        D_c^shift = { D : W1(D, D_c) <= delta_t }   # delta_t = coverage-growth budget

2.  # Support-aware shift detection (repair for the load-bearing assumption)
    # 2a. Estimate delta_t per cohort from observed feature displacement of newly
    #     generated tasks: delta_t = 95th percentile of W1(D_c^new, D_c) over the
    #     round's newly generated tasks, computed on m=256-dim task embeddings.
    # 2b. Flag out-of-support tasks: run a kernel MMD two-sample test (RBF kernel,
    #     bandwidth = median pairwise distance) between each new task's embedding
    #     and D_c; if MMD p-value < mmd_alpha=0.01, route the task to U_t and do
    #     NOT include it in the in-ball precision estimate.

3.  # For each candidate verifier v in V_t, estimate two quantities on R_t:
    for v in V_t:
        prec_v(D)  = precision of v on rollouts drawn from D   # per-probe
        mono_v(D)  = P[ v accepts probe p_i ]  vs.  probe difficulty rank
                     # Spearman rho between v-acceptance and probe rank on D

4.  # Joint calibration loss (couples verifier + diagnostic objectives):
    L(v) = sup_{D in union_c D_c^shift} [ prec_v(D) - prec_v(D_c) ]
           + lambda * sup_{D} [ mono_v(D_c) - mono_v(D) - gamma ]_+
    # lambda=1.0 fixed; first term = precision invariance, second = monotonicity

5.  # Distributionally-robust certificate via worst-case over shift set:
    #    use the dual / Wasserstein-DRO closed form on the finite rollout sample
    #    with entropic regularization (reg=0.01) to stabilize the empirical dual.
    #    eps_star(v) = quantile_{1-alpha}( L(v) over n_boot=1000 block-bootstrap
    #    resamples of R_t with block_len=8 episodes, to respect episode autocorrelation).

6.  # Three-valued acceptance rule (closed form, no controller):
    if any task in cohort c is in U_t:
        cert_v = (out-of-support, eps_star(v))   # requires re-calibration, not certified
    elif eps_star(v) <= eps_budget:
        cert_v = (accept, eps_star(v))
    else:
        cert_v = (reject, eps_star(v))

7.  V_{t+1} = { v in V_t : cert_v is accept }; record cert_v = (eps_star(v), L(v))
    return V_{t+1}, certs, U_t
```

The two terms in `L(v)` are the cross-objective coupling: the first is the verifier-evolution objective (precision under shift), the second is the diagnostic-evolution objective (monotonicity of checkpoints w.r.t. probe difficulty). A checkpoint that improves precision only by exploiting the shift raises the first term; a checkpoint that violates monotonicity raises the second. Acceptance requires both to be small simultaneously. The `out-of-support` branch is the concrete repair for the load-bearing assumption: it prevents the in-ball bound from being applied where it has no mass.

**Why this design.** We chose a Wasserstein-DRO worst-case over a block-bootstrap quantile instead of a point estimate of precision because the whole failure mode is that benchmark evolution *moves* the task distribution, and a point estimate on the current cohort cannot see that move — accepting the cost that the certificate is conservative (some good checkpoints are rejected when `delta_t` is loose). We chose to make the monotonicity term a *slack-constrained hinge* (`[mono_v(D_c) - mono_v(D) - gamma]_+`) rather than a hard equality because probes are noisy and exact monotonicity is unattainable — accepting the cost that a small non-monotonicity is tolerated, which we bound by `gamma`. We chose a closed-form acceptance rule `eps_star <= eps_budget` rather than a learned gate or a router because a learned gate reintroduces the very unidentifiability we are removing — accepting the cost that `eps_budget` must be set by the practitioner, which we fix at 0.05 to match the nominal confidence level. We chose a three-valued certificate with an explicit `out-of-support` branch rather than a single accept/reject because the load-bearing ball assumption is known to fail for novel modalities, and silently certifying such tasks would under-report drift — accepting the cost that some tasks are deferred to re-calibration rather than certified in the same round. A domain expert would not describe this as a variant of Agent-as-a-Judge (E4): that work *adds* an agentic evaluator and reports agreement with humans, whereas DRCC does not add an evaluator at all — it adds a certificate that constrains *which* verifier checkpoints are admissible, and the admissible set is defined by a worst-case-over-shift condition that Agent-as-a-Judge has no analogue of.

**Why it measures what we claim.** The computational quantity `sup_{D in D_c^shift} [prec_v(D) - prec_v(D_c)]` measures *verifier calibration invariance under benchmark-induced shift* because the named assumption is that the Wasserstein ball of radius `delta_t` contains every task distribution reachable by one round of cohort-calibrated coverage growth; this assumption fails when coverage growth produces a task whose feature distribution lies outside the ball (e.g., a brand-new application modality), in which case the quantity reflects only in-ball precision drift and under-reports true degradation — which is why step 2b routes such tasks to `out-of-support` rather than certifying them. The quantity `mono_v(D)` (Spearman rho between verifier acceptance and probe difficulty rank) measures *consistency between the verifier and the diagnostic capability-boundary probes* because the named assumption is that the probe rank is a sufficient statistic for true task difficulty within a cohort; this assumption fails when the probe set is itself stale relative to the agent population, in which case the rho reflects agreement with a stale difficulty ordering rather than with true boundaries. The block-bootstrap quantile `eps_star(v)` measures *the certified radius of admissible checkpoints at confidence 1-alpha* because the named assumption is that block-bootstrap resampling of `R_t` with block length 8 is a valid approximation to the sampling distribution of the worst-case loss under within-episode autocorrelation; this assumption fails when autocorrelation extends beyond the block length (e.g., long-horizon episodes with cross-episode memory), in which case `eps_star` is anti-conservative and the acceptance rule admits checkpoints it should reject.

## Contribution

(1) DRCC, a distributionally-robust calibration certificate that couples CUA-SEE's verifier-evolution and diagnostic-evolution objectives through a joint loss with a Wasserstein-DRO worst-case term and a slack-constrained monotonicity term, producing a closed-form accept/reject rule for verifier checkpoints. (2) A design principle: verifier-checkpoint admissibility should be defined by invariance of precision under the benchmark-evolution-induced shift rather than by point-estimate precision on the current cohort, and this invariance is only meaningful when jointly constrained against probe monotonicity. (3) An analysis of when the certificate is conservative versus anti-conservative, tied to the radius `delta_t` and rollout autocorrelation, giving practitioners a stated validity region for the acceptance rule.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | CUA-SEE multi-round task suite (cohort task distributions) | Provides shifting distributions and process-level rollout evidence. |
| Dataset (transfer) | OmniBench OmniEval pipeline (graph-structured tasks, 20 scenarios) | Tests certificate as general meta-evaluation primitive, not CUA-SEE add-on. |
| Primary metric | Precision degradation under distribution shift (worst-case) | Directly targets verifier calibration invariance claim. |
| Secondary metric | Out-of-support rate (fraction of tasks flagged by MMD test) | Measures whether the ball's blind spot is detected, not hidden. |
| Secondary metric | Round-over-round precision collapse correlation with rejection decisions | Tests whether certificate predicts downstream degradation. |
| Baseline | Static fixed-threshold verifier | Point-estimate acceptance ignores benchmark-induced distribution move. |
| Baseline | Agent-as-a-Judge (agentic evaluator) | Adds evaluator but no shift-robust admissibility certificate. |
| Baseline | Unconstrained verifier evolution (no certificate) | Tests whether worst-case coupling is necessary at all. |
| Ablation-of-ours | DRCC without monotonicity hinge (λ=0) | Isolates diagnostic-evolution coupling term contribution. |
| Ablation-of-ours | DRCC without support-aware MMD routing (fixed delta_t, two-valued) | Isolates the load-bearing-assumption repair. |
| Ablation-of-ours | DRCC with i.i.d. bootstrap (block_len=1) | Tests whether episode autocorrelation correction matters. |

### Why this setup validates the claim

The central claim is that admissible verifier checkpoints can be certified against the distribution shift that benchmark evolution itself induces, and that coupling precision-invariance with diagnostic monotonicity rejects checkpoints that only appear calibrated on the current cohort. The CUA-SEE multi-round suite is the right primary testbed because it is the only setup here where the task distribution demonstrably moves across rounds via cohort-calibrated coverage growth, so a point-estimate verifier has a concrete opportunity to fail. The OmniBench OmniEval pipeline is added as a transfer testbed because its graph-structured tasks span 20 scenarios and are generated by subtask composition, giving an independent adaptive-evaluation framework on which the certificate's rejection decisions can be checked against downstream round-over-round precision collapse — this addresses whether DRCC is a general meta-evaluation primitive rather than a CUA-SEE add-on. The static fixed-threshold baseline tests the sub-claim that point estimates cannot see the move; the Agent-as-a-Judge baseline tests the sub-claim that adding an evaluator without a worst-case admissibility condition does not close this gap; the unconstrained verifier-evolution baseline tests whether the certificate is doing work beyond ordinary evolution. The λ=0 ablation isolates whether the monotonicity coupling term is load-bearing rather than decorative. The no-MMD-routing ablation isolates whether the support-aware repair is necessary or whether the in-ball bound alone suffices. The i.i.d.-bootstrap ablation isolates whether the block-bootstrap correction for episode autocorrelation matters. The worst-case precision-degradation metric is the correct detector because the predicted effect is a gap concentrated on shifted cohorts, not a uniform improvement — a uniform metric would not distinguish our mechanism from a better point estimate. The out-of-support rate is the correct secondary detector because the load-bearing assumption predicts that novel-modality tasks will be flagged rather than silently certified.

### Expected outcome and causal chain

**vs. Static fixed-threshold verifier** — On a round where coverage growth introduces tasks from a new application cohort, the static verifier produces inflated precision because its threshold was tuned on the prior cohort and cannot see the feature shift. Our method instead certifies each checkpoint against the Wasserstein ball of reachable distributions, rejecting checkpoints whose worst-case precision drift exceeds budget, and routes genuinely novel-modality tasks to `out-of-support` rather than certifying them. We expect a noticeable precision-degradation gap on the shifted cohort, with near-parity on the stable cohort, and a non-zero out-of-support rate on the round that introduces the new modality.

**vs. Agent-as-a-Judge (agentic evaluator)** — On a case where an intermediate solution step looks plausible but violates a state-grounded requirement, the agentic evaluator produces a high agreement score with humans because it reasons over the trajectory without any shift-robust admissibility condition, so it rewards checkpoints that generalize poorly. Our method instead constrains which checkpoints are admissible via worst-case precision and monotonicity, so poorly generalizing checkpoints are excluded before evaluation. We expect Agent-as-a-Judge to track human agreement yet admit checkpoints that later degrade, while our accepted set shows lower downstream degradation.

**vs. Unconstrained verifier evolution** — On a case where evolution finds a checkpoint that exploits a spurious correlation in the current cohort (e.g., a GUI element that co-occurs with success), the unconstrained verifier produces high current-cohort precision that collapses on the next round's cohort because nothing penalizes shift sensitivity. Our method instead raises the worst-case precision term for that checkpoint and rejects it. We expect the unconstrained baseline to show a sharp round-over-round precision drop on the shifted cohort, while our accepted set stays roughly flat.

**vs. DRCC without monotonicity hinge (λ=0)** — On a case where a checkpoint's precision is stable under shift but its acceptance ordering disagrees with the diagnostic probe difficulty rank (e.g., it accepts easy probes and rejects hard ones), the λ=0 variant certifies it because only the precision term is active. Our method raises the monotonicity hinge term and rejects it. We expect the λ=0 ablation to match full DRCC on stable-cohort precision but to admit checkpoints whose probe-rank correlation is negative, producing higher downstream degradation on the shifted cohort.

**vs. DRCC without support-aware MMD routing** — On a round that introduces a brand-new application modality, the fixed-`delta_t` two-valued variant applies the in-ball bound to tasks with no mass in the ball, producing a vacuous worst-case estimate and certifying checkpoints that should be flagged. Our method's MMD test routes those tasks to `out-of-support`, so the certificate reports a re-calibration requirement instead of a false accept. We expect the no-routing ablation to show near-zero out-of-support rate and inflated accepted-set size on the novel-modality round, with correspondingly higher downstream precision collapse.

**vs. DRCC with i.i.d. bootstrap (block_len=1)** — On a rollout set with strong within-episode autocorrelation, the i.i.d. bootstrap under-estimates the sampling variance of the worst-case loss, making `eps_star` anti-conservative and admitting checkpoints that the block-bootstrap rejects. Our method uses block length 8 to respect episode correlation. We expect the i.i.d.-bootstrap ablation to accept a superset of checkpoints and to show higher downstream degradation on the shifted cohort, with the gap concentrated on episodes longer than the block length.

### What would falsify this idea

If our precision-degradation advantage over the static and unconstrained baselines is uniform across stable and shifted cohorts rather than concentrated on the shifted cohort, the shift-robustness claim is wrong. If the λ=0 ablation matches full DRCC on shifted-cohort precision, the monotonicity coupling is not load-bearing and the central mechanism claim fails. If the no-MMD-routing ablation matches full DRCC on the novel-modality round (i.e., the out-of-support rate is near zero and downstream degradation is unchanged), the load-bearing-assumption repair is not doing work and the in-ball bound suffices.

## References

1. Self-Evolving Evaluation Environments for Computer-Use Agents
