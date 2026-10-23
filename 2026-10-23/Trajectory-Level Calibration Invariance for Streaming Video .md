# Trajectory-Level Calibration Invariance for Streaming Video Generation Judges

## Motivation

Automated judges for streaming video generation are validated only on static single-turn instances, yet deployment is increasingly closed-loop multi-turn. StreamEval (2025) provides scripted multi-turn test cases with state-change requirements, but no existing judge is certified to remain monotone in human preference across the full trajectory: a judge can be perfectly calibrated at each turn while systematically drifting as errors compound, because its scores are not coupled to the closed-loop state evolution. This is the meta-gap shared with autoregressive video models (e.g., Self-Forcing++, Diffusion Forcing) that optimize per-chunk quality without any trajectory-level invariant.

## Key Insight

A judge's scores are trustworthy in interactive regimes iff they form a monotone functional of the closed-loop state trajectory, a property that can be tested by injecting controlled state perturbations and checking that the judge's score ordering matches the induced human-preference ordering across turns.

## Method

**A. What it is**
We introduce TRACI (Trajectory-level Calibration Invariance), a certification procedure for an MLLM video judge. Input: a streaming video generation model M, a scripted multi-turn test case from StreamEval (initial assets, turn-wise requests, state-change requirements), and a judge J that outputs a scalar score per turn. Output: a binary certificate indicating whether J satisfies trajectory-level calibration invariance (TCI) on that case, plus a diagnostic profile of where it fails.

**B. How it works**

Phase 1: Generate the nominal trajectory.
  For t = 1..T:
    request_t = scripted request at turn t
    video_chunk_t = M(assets, request_1..t)
    state_t = extract_state(video_chunk_t)   # task-specific state vector (e.g., object positions, identity embedding, edit flags)
    score_t = J(video_chunk_t, request_t, context=video_chunk_1..t-1)
  Record nominal trajectory τ0 = (state_1..T, score_1..T).

Phase 2: Construct counterfactual trajectories by controlled state perturbation.
  For each turn k and each perturbation δ in a pre-specified set Δ (e.g., ±ε on object position, identity swap, edit omission):
    Re-run generation from turn k with request_k modified to induce δ in state_k, keeping all other requests fixed.
    Record perturbed trajectory τ_{k,δ} = (state_1..T, score_1..T).
  This yields a set of trajectories that differ from τ0 only in the causal future of turn k.

Phase 3: Estimate human-preference ordering via a held-out pairwise human study.
  For each pair (τ0, τ_{k,δ}), collect N=50 human pairwise preferences on "which trajectory better satisfies the scripted state-change requirements and preserves content?"
  Fit a Bradley-Terry model to obtain a latent preference score h(τ) for each trajectory.
  **Load-bearing assumption (stated explicitly):** h(τ) is a monotone functional of the scripted state-change requirements, i.e., humans rank trajectories by state-change satisfaction, not by visual fidelity or temporal smoothness. This assumption is tested in Phase 3b and audited in the experiment.
  **Phase 3b: Constrained/anchored preference elicitation (repair for fidelity bias).**
    For each turn t, elicit from each rater a binary judgment s_t(τ) ∈ {0,1} on whether the scripted state-change requirement at turn t is satisfied, plus a magnitude m_t(τ) ∈ {0,1,2} (0=not satisfied, 1=partially, 2=fully).
    Separately, elicit a visual-fidelity rating f_t(τ) ∈ {1..5} on a 5-point Likert scale (artifact severity, smoothness).
    Fit a two-factor Bradley-Terry model: P(τ_a ≻ τ_b) = σ( α·[Σ_t s_t(τ_a) − Σ_t s_t(τ_b)] + β·[Σ_t f_t(τ_a) − Σ_t f_t(τ_b)] ), with α, β ≥ 0 estimated by maximum likelihood.
    Define the decomposed utility h_state(τ) = Σ_t s_t(τ) (state-change satisfaction only) and h_full(τ) = α·Σ_t s_t(τ) + β·Σ_t f_t(τ).
    Certify TCI against h_state (the stated utility model), and report concordance against h_full as a secondary diagnostic. This makes the certificate defined relative to an explicitly stated utility model rather than an unconstrained human preference known to overweight fidelity.

Phase 4: Test trajectory-level calibration invariance.
  Compute the trajectory-level judge score S_J(τ) = Σ_t w_t * score_t, with weights w_t = 1/T (uniform) fixed a priori.
  TCI holds iff for all pairs (τ0, τ_{k,δ}), sign(S_J(τ0) - S_J(τ_{k,δ})) == sign(h_state(τ0) - h_state(τ_{k,δ})).
  Report the fraction of concordant pairs as the TCI concordance rate; certify if ≥ 0.95 with a binomial test at α=0.05.

Phase 5: Diagnostic decomposition.
  For each failing pair, attribute the violation to a turn t* where the per-turn score difference contradicts the per-turn state-change magnitude, using a Shapley decomposition over turns.

**C. Why this design**
We chose controlled state perturbations over natural variation because natural variation confounds judge miscalibration with genuine quality differences, accepting the cost that perturbations may be out-of-distribution for M. We chose a Bradley-Terry fit on human pairwise preferences over direct human scalar ratings because pairwise comparisons have higher inter-rater reliability, accepting the cost of more human queries. We chose a fixed uniform weighting over turns rather than a learned weighting because a learned weighting would let the judge hide trajectory-level drift in a few turns, accepting the cost of potentially underweighting early-turn errors. We chose a sign-concordance test over a correlation coefficient because the invariance condition is ordinal, not cardinal, accepting the cost of lower statistical power. Unlike StreamEval, which scores each turn independently, TRACI couples scores across turns via the closed-loop state trajectory; unlike Self-Forcing++, which optimizes long-horizon generation quality without a judge calibration condition, TRACI certifies the judge itself.

**D. Why it measures what we claim**
The trajectory-level judge score S_J(τ) measures human preference h_state(τ) because the two-factor Bradley-Terry fit assumes that pairwise human preferences are generated by a latent scalar utility that is monotone in the scripted state-change requirements (α > 0 and state satisfaction dominates fidelity by construction of h_state); this assumption fails when humans weight visual fidelity more than state consistency, in which case S_J reflects a different utility (h_full). The perturbation set Δ measures causal contribution of turn k to the trajectory because we assume that modifying request_k only affects states in the causal future of k; this assumption fails when the generation model has non-causal dependencies (e.g., bidirectional attention over the full video), in which case the perturbation leaks into past turns. The sign-concordance rate measures calibration invariance because we assume that a judge satisfying TCI will have concordant signs on all pairs; this assumption fails when the perturbation set Δ is too small to induce detectable preference differences, in which case the concordance rate is uninformative. The uniform weighting w_t = 1/T measures trajectory-level calibration because we assume that all turns contribute equally to the scripted state-change requirement; this assumption fails when early-turn errors compound more than late-turn errors, in which case uniform weighting underweights early-turn drift (tested by the learned-weight ablation).

## Contribution

['TRACI, a certification procedure that tests whether an MLLM video judge satisfies trajectory-level calibration invariance by comparing its score ordering to human preference ordering under controlled state perturbations across a streaming multi-turn session.', 'A diagnostic decomposition that attributes TCI violations to specific turns via Shapley values over per-turn score differences, enabling targeted judge repair.', 'An empirical finding on StreamEval test cases characterizing which judge architectures (e.g., per-turn vs. full-context) satisfy TCI and how violation rates scale with session length.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | StreamEval scripted multi-turn cases | Provides closed-loop state trajectories with turn-wise requirements. |
| Primary metric | TCI sign-concordance rate vs. decomposed BT utility h_state | Ordinal invariance against explicitly stated state-satisfaction utility. |
| Secondary metric | TCI sign-concordance rate vs. full BT utility h_full | Quantifies fidelity-bias gap between h_state and h_full. |
| Baseline | StreamEval per-turn independent judge | Tests whether trajectory coupling is necessary. |
| Baseline | Self-Forcing++-style quality-tuned judge | Tests whether quality optimization implies calibration. |
| Baseline | Raw per-turn scalar judge, no aggregation | Tests whether naive averaging already satisfies TCI. |
| Ablation-of-ours | TRACI with learned turn weights instead of uniform | Tests whether uniform weighting is load-bearing. |
| Ablation-of-ours | TRACI certified against h_full instead of h_state | Tests whether fidelity bias in the utility changes certification. |
| Assumption audit | Bidirectional-attention generator (non-causal) | Tests causal-confinement assumption (leakage into past turns). |
| Assumption audit | Human cohort instructed to prioritize fidelity | Tests BT monotonicity assumption (h_state vs. h_full divergence). |

### Why this setup validates the claim

The central claim is that a judge can be certified as trajectory-calibrated only when its ordering over perturbed closed-loop trajectories agrees with human preference ordering, and that per-turn or quality-optimized judges will fail this test in a structured, diagnosable way. StreamEval cases are the right dataset because they supply scripted turn-wise state-change requirements, so counterfactual perturbations have a ground-truth notion of what should change and what should stay fixed. The TCI concordance rate against the decomposed Bradley-Terry utility h_state is the right primary metric because the claim is ordinal and because h_state is defined explicitly as state-change satisfaction, orthogonal to the separately-estimated visual-fidelity factor; this directly addresses the load-bearing assumption that human preference is monotone in state-change requirements. The secondary metric against h_full quantifies how much the certificate would change if fidelity bias were allowed, making the assumption auditable. Each baseline isolates one sub-claim. The StreamEval per-turn judge tests whether coupling across turns matters; if it certifies at the same rate as TRACI, trajectory coupling is unnecessary. The Self-Forcing++-style judge tests whether optimizing long-horizon generation quality incidentally yields calibration; if it certifies, the calibration condition is redundant with quality. The naive raw-scalar judge tests whether uniform aggregation alone suffices. The learned-weight ablation tests whether uniform weighting is doing real work or whether any reasonable aggregation would pass. The assumption-audit conditions (bidirectional-attention generator, fidelity-prioritizing cohort) directly stress the causal-confinement and BT-monotonicity assumptions, quantifying their load-bearing role. Together these form a falsifiable design: TRACI must beat all three baselines on concordance against h_state, and the ablation must underperform TRACI specifically on cases where drift is concentrated in a few turns.

### Expected outcome and causal chain

**vs. StreamEval per-turn independent judge** — On a case where an early turn omits a required edit and later turns compensate visually, the per-turn judge scores each chunk high because each chunk looks locally plausible, missing the accumulated state violation. Our method instead compares the full perturbed trajectory against the nominal one, so the early omission propagates into the trajectory-level sign and is detected. We expect a clear concordance gap concentrated on cases with early-turn state drift and near-parity on cases with only single-turn perturbations.

**vs. Self-Forcing++-style quality-tuned judge** — On a case where a long-horizon generation degrades gracefully in fidelity but silently drops a scripted identity constraint, the quality-tuned judge rewards the smooth trajectory and ranks it above a slightly noisier but constraint-faithful one. Our method instead anchors the ordering to human preference over scripted state-change satisfaction (h_state), so the constraint-faithful trajectory wins. We expect TRACI to certify while the quality-tuned judge shows sign flips specifically on identity-swap and edit-omission perturbations, with parity on pure fidelity perturbations.

**vs. Raw per-turn scalar judge, no aggregation** — On a case where a single turn's score is noisy but the trajectory is otherwise correct, the raw judge's ordering flips on small perturbations because per-turn noise dominates the comparison. Our method instead aggregates with fixed uniform weights and tests sign concordance over the full trajectory, so noise averages out and the ordering is stable. We expect the raw judge to show low concordance driven by near-tie pairs, while TRACI's concordance is high and its failures concentrate on genuinely ambiguous perturbations rather than noise.

**vs. TRACI with learned turn weights (ablation)** — On a case where drift is concentrated in a few turns, learned weights can downweight those turns and hide the violation, yielding artificially high concordance against h_state. Our method with uniform weights cannot hide drift, so its concordance is lower on such cases but its failures are diagnosable via Phase 5. We expect the learned-weight ablation to certify at a higher rate than uniform TRACI specifically on concentrated-drift cases, confirming that uniform weighting is load-bearing.

**vs. TRACI certified against h_full (ablation)** — On a case where the perturbed trajectory has severe visual artifacts but satisfies the state-change requirement, h_full may rank it below a smoother but constraint-violating trajectory, flipping the sign relative to h_state. Our method certifies against h_state, so the constraint-faithful trajectory wins. We expect the h_full-certified variant to show lower concordance and to disagree with h_state on exactly the artifact-heavy perturbation pairs, quantifying the fidelity-bias gap.

**Assumption audit: bidirectional-attention generator** — On a case where the generator uses bidirectional attention, modifying request_k can leak into states before turn k, violating causal confinement. Our method assumes causal confinement, so the perturbation set Δ no longer isolates the causal future of k. We expect TCI concordance to degrade on this generator relative to the causal generator, with failures concentrated on turns preceding k, quantifying the load-bearing role of the causal-confinement assumption.

**Assumption audit: fidelity-prioritizing human cohort** — On a case where the perturbed trajectory has visible artifacts but satisfies the state-change requirement, a cohort instructed to prioritize fidelity will rank it lower, making h_full diverge from h_state. Our method certifies against h_state, so the certificate remains valid under the stated utility model. We expect the fidelity-prioritizing cohort to produce h_full rankings that disagree with h_state on artifact-heavy pairs, and we report the concordance of TRACI against h_full for this cohort to quantify the BT-monotonicity assumption's load-bearing role.

### What would falsify this idea

If TRACI's concordance advantage over the per-turn and quality-tuned baselines is uniform across all perturbation types rather than concentrated on the predicted failure modes (early-turn drift, identity swap, edit omission), the claimed causal mechanism is wrong. Likewise, if the learned-weight ablation certifies at the same rate as uniform weighting, uniform weighting is not load-bearing and the certification claim collapses to ordinary aggregation. Additionally, if the assumption-audit conditions (bidirectional-attention generator, fidelity-prioritizing cohort) do not degrade TCI concordance relative to the main conditions, the causal-confinement and BT-monotonicity assumptions are not load-bearing and the certificate's interpretation is undermined.

## References

1. StreamEval: Evaluating Interactive Streaming Video Generation Models
2. Looking Ahead: Modeling Future Evolution in Autoregressive Video Generation
3. Self-Forcing++: Towards Minute-Scale High-Quality Video Generation
4. Diffusion Forcing: Next-token Prediction Meets Full-Sequence Diffusion
5. LTX-Video: Realtime Video Latent Diffusion
6. GAIA-1: A Generative World Model for Autonomous Driving
7. AR-Diffusion: Auto-Regressive Diffusion Model for Text Generation
8. Phenaki: Variable Length Video Generation From Open Domain Textual Description
