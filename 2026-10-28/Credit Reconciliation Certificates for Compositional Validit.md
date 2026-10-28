# Credit Reconciliation Certificates for Compositional Validity in Multimodal Agent Self-Improvement

## Motivation

Existing agent self-improvement methods assume linear decomposability: turn-level rewards and skill-retrieval scores are treated as independently evaluable units whose aggregate equals the whole, with no testable condition linking the decomposition to terminal success. TARL (Process-Supervised RL for Interactive Multimodal Tool-Use Agents) assigns turn-level LLM-judge rewards without a causal link to the terminal database state, and VISUALSKILL retrieves skills per topic without verifying that the concatenation is executable against the target app. This meta-gap recurs across paper-trees: the field lacks a minimal structural object that makes the linear-decomposition assumption falsifiable at training time, causing self-improvement to reinforce decompositions that are locally plausible but globally invalid.

## Key Insight

A trajectory-level potential function whose marginals are exactly the turn rewards and skill-retrieval scores yields a per-trajectory compositional-validity certificate that is zero iff the decomposition is consistent with the terminal database-state criterion, enabling rejection or reweighting of invalid decompositions before they corrupt self-improvement.

## Method

# FARA: Faithful Attribution via Reconciliation of Aggregates
# Inputs: trajectory τ = (s_1..s_T, a_1..a_T, o_1..o_T), turn rewards r_t from TARL judge,
#         retrieved skill set K with per-skill scores q_k, terminal DB state D_τ, task success y ∈ {0,1}
# Output: certificate C(τ) ∈ [0,1], reconciled credit weights w_t, accepted/rejected flag

# Load-bearing assumption (stated explicitly):
#   The turn rewards r_t and skill-retrieval scores q_k are, up to a linear projection onto
#   the trajectory feature span, the marginals of a single trajectory potential P. A nonzero
#   reconciliation residual ρ(τ) therefore diagnoses an invalid decomposition rather than
#   arbitrary scale/calibration mismatch between two independently-produced, unnormalized
#   score vectors. To make this assumption testable, ρ(τ) is computed as a calibrated,
#   whitened residual against a null distribution estimated on held-out known-valid
#   trajectories, not as a raw least-squares residual.

1.  Compute terminal potential Φ(τ) = y * exp(-λ * d(D_τ, D*))   # λ=2.0, D* target DB state
    # d(D_τ, D*) is pre-registered as normalized edit distance over table rows:
    #   d = (insertions + deletions + substitutions) / max(|D_τ|, |D*|), with schema-matching
    #   by primary key; rows are compared field-by-field after type normalization.

2.  Build marginal vector m = [r_1..r_T, q_{k1}..q_{kM}]   # M = |K| retrieved skills
    # m is z-scored per component using mean/std estimated on the calibration set (see step 3a).

3.  Calibration (held-out, disjoint from training trajectories):
    3a. Fit the linear potential P on a calibration set of N_cal = 512 known-valid trajectories
        (y=1, C(τ) not yet defined) by constrained least squares:
          min_P || m - ∇P(τ)||^2   s.t.   P(τ) = Φ(τ)
        where ∇P(τ) = [∂P/∂s_t for t=1..T, ∂P/∂k_j for j=1..M] evaluated at τ
        (P is a linear potential over trajectory features: turn embeddings + skill embeddings).
        Solution via KKT with equality constraint; closed-form.
    3b. Compute the calibration residual covariance Σ = Cov(ρ_cal) over the N_cal valid trajectories,
        where ρ_cal = m - ∇P(τ) for each calibration trajectory.
    3c. Estimate the null distribution of the whitened residual statistic
          s(τ) = ρ(τ)^T Σ^{-1} ρ(τ)
        empirically from the N_cal valid trajectories (empirical quantiles).
    3d. Set θ_reject as the 95th percentile of s(τ) on the calibration set (i.e., a 5% false-reject
        rate on known-valid trajectories), replacing the fixed θ_reject=0.3.

4.  For a new trajectory τ, compute the calibrated reconciliation residual:
      ρ(τ) = m - ∇P(τ)   (using P fit in step 3a)
      s(τ) = ρ(τ)^T Σ^{-1} ρ(τ)

5.  Certificate: C(τ) = exp(-γ * s(τ))   # γ=1.0, monotone decreasing in the whitened residual

6.  If C(τ) < C_reject where C_reject = exp(-γ * θ_reject) (equivalently s(τ) > θ_reject):
        mark decomposition INVALID, reject from self-improvement buffer
    Else: reconciled weights w_t = softmax(∇P(τ)_t / T), use w_t * r_t in TARL policy gradient

7.  Return C(τ), w_t, valid_flag

**Why this design.** We chose a constrained least-squares reconciliation over an unconstrained learned potential because the equality constraint P(τ)=Φ(τ) is what forces the certificate to be zero exactly when marginals sum to the terminal potential, accepting the cost that the linear potential class may underfit nonlinear interactions. We chose a linear potential over trajectory features (turn embeddings + skill embeddings) over a neural potential because the KKT solution is closed-form and reproducible, accepting the cost that it cannot capture high-order skill-turn synergies. We chose a hard reject threshold (θ_reject=0.3) over soft reweighting for invalid decompositions because TARL's policy gradient is sensitive to corrupted credit, accepting the cost that some borderline-valid trajectories are discarded. We chose to reconcile turn rewards and skill scores in a single vector rather than separately because the compositional-validity condition couples them through the shared terminal potential — this is the structural property that separates FARA from VinePPO (per-step MC value rollouts) and Process Reward Models (Lightman 2023), which estimate step values in isolation and never test whether the sum of step credits equals the terminal outcome; a domain expert would not describe FARA as a variant of these because FARA's output is a falsifiable certificate over the whole decomposition, not a per-step value estimate. To address the load-bearing assumption that raw judge and retrieval scores are on incomparable scales, we calibrate the residual on a held-out set of known-valid trajectories: we fit P on N_cal=512 valid trajectories, estimate the residual covariance Σ, and set θ_reject from the empirical null distribution (95th percentile), so that the certificate is discriminative rather than a monotone function of misspecification.

**Why it measures what we claim.** The reconciliation residual ρ(τ) measures compositional validity because the assumption that turn rewards and skill scores are marginals of a single trajectory potential makes ρ(τ)=0 equivalent to the decomposition being consistent with the terminal DB state; this assumption fails when the linear potential class cannot represent the true credit assignment (e.g., a skill's contribution depends nonlinearly on the turn context), in which case ρ(τ) reflects potential-class misspecification rather than true invalidity. The terminal potential Φ(τ) measures task success alignment because the assumption that DB-state distance d(D_τ, D*) is a sufficient statistic for task success makes Φ(τ) monotone in y; this assumption fails when the DB state is under-specified (e.g., hidden application state not captured by D_τ), in which case Φ(τ) reflects observable DB proximity only. The reconciled weights w_t measure faithful credit because the assumption that the KKT solution's gradient ∇P(τ)_t captures each turn's marginal contribution to Φ(τ) makes softmax(∇P(τ)_t) a valid credit distribution; this assumption fails when turn features are collinear across turns, in which case w_t reflects feature redundancy rather than causal contribution. The calibrated certificate C(τ) measures invalidity because the assumption that the whitened residual s(τ) follows the null distribution estimated on known-valid trajectories makes s(τ) > θ_reject a valid rejection criterion; this assumption fails when the calibration set is not representative of the test distribution, in which case C(τ) reflects calibration-set shift rather than compositional invalidity.

## Contribution

['A compositional-validity certificate (FARA) computed via constrained least-squares reconciliation of turn rewards and skill-retrieval scores against a terminal database-state potential, yielding a per-trajectory scalar that is zero iff the linear decomposition is consistent with task success.', 'A rejection/reweighting mechanism that uses the certificate to filter or reweight decompositions before they enter the self-improvement buffer, with a closed-form KKT solution that makes the certificate reproducible and auditable.', 'An empirical design principle: compositional validity of credit decompositions is testable at training time from signals already produced by TARL and VISUALSKILL, without additional human annotation or environment rollouts.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | τ-bench + CUA-World (DB-backed tasks) | Terminal DB state gives ground-truth Φ |
| Dataset (transfer) | A code-execution or API-mock environment with a programmatic success checker (e.g., a unit-test-based code agent benchmark) | Tests whether Φ(τ) instantiated from a non-DB checker generalizes the certificate |
| Primary metric | Success rate on accepted-certificate subset | Isolates trajectories FARA deems valid |
| Secondary metric | False-reject rate on known-valid trajectories (calibration set) | Verifies the certificate is discriminative, not a monotone function of misspecification |
| Secondary metric | Success rate on rejected-certificate subset | Control to show the filter is not merely selecting easy tasks |
| Baseline | TARL (no certificate filtering) | Tests whether reconciliation filtering alone helps |
| Baseline | VinePPO per-step MC values | Tests whether whole-trajectory reconciliation beats isolated step values |
| Baseline | Process Reward Model (Lightman 2023) | Tests whether consistency-with-terminal beats per-step scoring |
| Baseline | Potential-based reward shaping (Ng et al. 1999) on the same features | Tests whether the certificate's rejection gate, not the potential reweighting, is the source of gain |
| Baseline | Linear credit assignment on the same features (no certificate) | Tests whether the LS machinery alone explains any gain |
| Ablation-of-ours | FARA w/o equality constraint (unconstrained LS) | Tests whether the constraint, not the LS, drives gains |
| Ablation-of-ours | FARA with fixed θ_reject=0.3 (no calibration) | Tests whether calibration is load-bearing |

### Why this setup validates the claim

The central claim is that reconciled credit weights derived from a single trajectory potential, gated by a falsifiable certificate, produce more faithful credit than methods that estimate step values in isolation. τ-bench and CUA-World are the right datasets because they expose a terminal DB state D_τ, which is exactly what Φ(τ) needs to be checkable — without a ground-truth terminal state, the certificate has no anchor and the claim is untestable. The code-execution/API-mock environment tests whether the certificate generalizes beyond DB-backed tasks by instantiating Φ(τ) from a programmatic success checker (unit tests), showing the mechanism is not a τ-bench-specific artifact. TARL (no certificate filtering) tests the sub-claim that filtering invalid decompositions improves the policy gradient signal; VinePPO tests the sub-claim that whole-trajectory reconciliation beats per-step Monte-Carlo value rollouts; the Process Reward Model tests the sub-claim that consistency-with-terminal beats isolated step scoring. The potential-based reward shaping and linear-credit-assignment baselines are load-bearing controls: if either matches FARA, the contribution is the LS machinery rather than the falsifiability object. The ablation — dropping the equality constraint P(τ)=Φ(τ) — is the load-bearing control: if the unconstrained least-squares potential performs identically, then the certificate is decorative and the mechanism is just feature-based reweighting. The fixed-threshold ablation tests whether calibration is necessary. Success rate on the accepted-certificate subset is the right primary metric because it directly measures whether the trajectories FARA admits are the ones that actually succeed; a metric on all trajectories would confound filtering quality with policy quality. The false-reject rate on known-valid trajectories and the success rate on the rejected subset are controls that demonstrate the certificate is discriminative and not merely selecting easy tasks.

### Expected outcome and causal chain

**vs. TARL (no certificate filtering)** — On a case where a mid-trajectory skill is retrieved with a high q_k but is contextually irrelevant (e.g., a form-filling skill retrieved on a navigation turn), TARL's per-turn rewards inflate that turn's credit because it never checks whether the sum of turn credits equals the terminal DB outcome. Our method instead solves for P(τ) constrained to Φ(τ), so the inflated skill score forces a large calibrated residual s(τ), the certificate C(τ) drops below C_reject, and the trajectory is rejected before it corrupts the policy gradient. We expect a noticeable success-rate gap concentrated on tasks with high skill-retrieval ambiguity, with near-parity on single-skill tasks where retrieval is unambiguous.

**vs. VinePPO (per-step MC values)** — On a case where a late turn's contribution is only realized through an earlier turn's state change (e.g., opening a dialog that a later turn commits), VinePPO's per-step Monte-Carlo rollouts assign that late turn high value in isolation but cannot see that the earlier turn was the marginal cause, producing credit that sums to more than the terminal outcome. Our method instead reconciles all turn and skill marginals against the shared terminal potential, so the collinear early/late turns are forced to split credit in a way consistent with Φ(τ). We expect a larger gap on multi-turn compositional tasks and parity on single-turn tasks.

**vs. Process Reward Model (Lightman 2023)** — On a case where a step looks locally correct (valid API call, plausible UI action) but the DB state silently diverges from D* due to a missing side-effect, the PRM scores that step high because it judges steps in isolation and never tests consistency with the terminal outcome. Our method instead computes Φ(τ) from actual DB distance, so the silent divergence produces a calibrated residual between the PRM-style marginals and Φ(τ), lowering C(τ) and rejecting the trajectory. We expect a gap concentrated on tasks with hidden side-effects (transactional or stateful operations) and parity on stateless read-only tasks.

**vs. Potential-based reward shaping (Ng et al. 1999)** — On a case where the shaping potential is fit to the same features but has no rejection gate, the shaped rewards can still inflate credit for contextually irrelevant skills because the potential is not constrained to equal the terminal DB potential. Our method adds the equality constraint and the calibrated rejection gate, so trajectories with large s(τ) are rejected. We expect FARA to outperform shaping on high-skill-ambiguity tasks, with parity on unambiguous tasks, demonstrating the certificate gate is the source of gain.

**vs. Linear credit assignment (no certificate)** — On a case where the linear potential is fit without the equality constraint or rejection, the credit weights w_t reflect feature redundancy rather than causal contribution. Our method's calibrated residual and rejection gate filter out such invalid decompositions. We expect FARA to outperform linear credit assignment on tasks with hidden side-effects and collinear turns, with parity on simple tasks.

**vs. FARA w/o equality constraint (unconstrained LS)** — On a case where the equality constraint is dropped, the unconstrained LS can drive the residual near zero for almost any m by absorbing scale into P's coefficients, making C(τ)≈1 uninformative. Our method's equality constraint forces the potential to match Φ(τ), so invalid decompositions produce large s(τ). We expect FARA to outperform the unconstrained ablation on all subsets where the residual mechanism predicts failure, and the ablation to match FARA on easy tasks, confirming the constraint is load-bearing.

**vs. FARA with fixed θ_reject=0.3 (no calibration)** — On a case where raw judge and retrieval scores are on incomparable scales, the fixed threshold may reject many valid trajectories or accept invalid ones. Our calibrated threshold from the null distribution (95th percentile) controls the false-reject rate at 5% on known-valid trajectories. We expect the calibrated version to have higher success rate on the accepted subset and a lower false-reject rate, demonstrating calibration is necessary.

### What would falsify this idea

If FARA's gain over TARL is uniform across all task subsets rather than concentrated on the high-skill-ambiguity and hidden-side-effect subsets where the residual mechanism predicts failure, then the certificate is not doing the work we claim and the improvement must come from something else. Equivalently, if the unconstrained-LS ablation matches FARA's success rate on the accepted subset, the equality constraint — the core of the claim — is not load-bearing. Additionally, if the false-reject rate on known-valid trajectories is not near the nominal 5% or if the success rate on the rejected subset is as high as on the accepted subset, the certificate is not discriminative.

## References

1. Process-Supervised Reinforcement Learning for Interactive Multimodal Tool-Use Agents
2. VISUALSKILL: Multimodal Skills for Computer-Use Agents
3. Agent S2: A Compositional Generalist-Specialist Framework for Computer Use Agents
4. OSWorld: Benchmarking Multimodal Agents for Open-Ended Tasks in Real Computer Environments
5. Agent S: An Open Agentic Framework that Uses Computers Like a Human
6. AgentStore: Scalable Integration of Heterogeneous Agents As Specialized Generalist Computer Assistant
7. OS-ATLAS: A Foundation Action Model for Generalist GUI Agents
8. OSCAR: Operating System Control via State-Aware Reasoning and Re-Planning
