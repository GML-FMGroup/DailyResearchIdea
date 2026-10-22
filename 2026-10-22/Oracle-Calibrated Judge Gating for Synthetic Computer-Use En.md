# Oracle-Calibrated Judge Gating for Synthetic Computer-Use Environments

## Motivation

WildAWM's cross-judge consistency gate accepts judge agreement as a proxy for correctness, but agreement among judges can be high even when all judges are systematically miscalibrated against the true environment state; this is a meta-level failure that recurs across synthetic-environment pipelines (WildAWM, R2E-Gym's execution-free verifiers, APIGen's semantic verification), where verifier reliability is assumed rather than measured against ground truth. The root cause is structural: the gate has no access to an environment-state oracle, so it can only enforce inter-judge consensus, not judge validity.

## Key Insight

A judge's agreement is only a valid quality signal when its discrimination against ground-truth environment-state labels has been established on a calibration sample, because calibration against an oracle converts an otherwise circular consensus signal into a measurement of judge validity.

## Method

## OCJG: Oracle-Calibrated Judge Gating

**What it is.** OCJG is a replacement for WildAWM's cross-judge consistency gate. Its inputs are a synthesized task, a set of candidate judge models, and a small budget of oracle-labeled environment states; its output is a binary accept/reject decision for each task, together with a per-judge calibration certificate (noise-corrected AUC with confidence interval, and threshold).

**Load-bearing assumption (stated explicitly).** OCJG assumes the environment-state oracle returns a stable, low-noise binary valid/invalid label per state, so that a judge's AUC against those labels measures true validity discrimination rather than agreement with oracle noise. In computer-use environments state validity is path-dependent, so we do not assume a single unconditional label is well-defined; instead the oracle is conditioned on the trajectory prefix and carries an explicit per-state ambiguity estimate, and the certificate is computed from that conditioned, noise-annotated label set (Phase 1 below).

**How it works.**

```pseudocode
# Phase 1: Oracle labeling (budget B_o = 200 tasks per environment class)
for task t in sample(B_o):
    states = rollout_heterogeneous_agents(t)          # reuse WildAWM rollouts
    for s in states.intermediate + [states.final]:
        # oracle is conditioned on the trajectory prefix, not the bare state
        y_raw[s] = environment_state_oracle(t, prefix(s), s)   # binary valid/invalid
        # per-state ambiguity estimate: K=3 independent oracle re-executions
        # (fresh execution branch each time); agreement = fraction matching y_raw[s]
        agree[s] = mean_k 1[ oracle_k(t, prefix(s), s) == y_raw[s] ]   # in {0,1/3,2/3,1}
        if agree[s] < 0.67:
            y[s] = ABSTAIN          # oracle-ambiguous states are abstentions, not labels
        else:
            y[s] = y_raw[s]
            # label-noise rate for this state, used to correct AUC below
            eps[s] = 1 - agree[s]

# Phase 2: Judge calibration (noise-corrected AUC)
for judge j in candidate_judges:
    labeled = [(s, y[s], eps[s]) for s in states if y[s] != ABSTAIN]
    scores_j = [j(t, s) for (s, y, eps) in labeled]
    # noise-corrected AUC: subtract the chance-level contribution of label noise
    # AUC_corr = (AUC_obs - 0.5 * mean(eps)) / (1 - mean(eps))
    auc_obs_j = AUC(scores_j, [y for (_,y,_) in labeled])
    eps_bar_j = mean([eps for (_,_,eps) in labeled])
    auc_corr_j = (auc_obs_j - 0.5 * eps_bar_j) / (1 - eps_bar_j)
    # 95% CI via 1000 bootstrap resamples over labeled states
    ci_j = bootstrap_ci(auc_corr_j, n_boot=1000, alpha=0.05)
    tau_j = threshold_maximizing_youden_j(scores_j, [y for (_,y,_) in labeled])
    cert_j = (auc_corr_j, ci_j.lower, tau_j, eps_bar_j)

# Phase 3: Gate (per task, no oracle access)
for task t in synthesized_tasks:
    rollouts = rollout_heterogeneous_agents(t)
    votes = []
    for j in judges_with_cert:
        if cert_j.ci_lower < AUC_MIN (0.80): continue   # require CI lower bound to clear
        votes.append(1[ j(t, final_state) >= cert_j.tau ])
    accept(t) iff mean(votes) >= 0.5 and agreement_kappa(votes) >= 0.6
```

**Why this design.** We chose a held-out oracle-labeled calibration set over in-the-loop oracle queries because oracle calls are expensive and can perturb the environment, accepting the cost that calibration is only as representative as the sample; we chose per-judge AUC thresholds over a single global agreement threshold because different judges have different discrimination profiles and a global threshold lets a weak judge free-ride on strong ones, accepting the cost of a larger calibration sample to estimate each AUC reliably; we chose to drop uncalibrated judges rather than reweight them because reweighting requires estimating inter-judge correlation, which is itself unidentifiable without more oracle labels, accepting the cost that the gate may become a single-judge gate in low-budget regimes. We chose a prefix-conditioned oracle with K=3 re-execution agreement over a single unconditional label because state validity is path-dependent, accepting the cost of 3x oracle calls per state; we chose to treat oracle-ambiguous states as abstentions and to noise-correct AUC rather than trust raw AUC because a judge that mimics oracle noise would otherwise receive an inflated certificate, accepting the cost of a smaller effective calibration set. This is not a variant of R2E-Gym's hybrid verifier combination: R2E-Gym combines execution-based and execution-free verifiers at test time without ever measuring either against an environment-state oracle, whereas OCJG's load-bearing step is the oracle-calibrated AUC certificate that licenses a judge's vote at all.

**Why it measures what we claim.** The noise-corrected AUC of a judge against prefix-conditioned, ambiguity-filtered oracle labels measures that judge's discrimination validity because the oracle label is defined as the environment's own state-validity predicate conditioned on the trajectory prefix, and the noise correction removes the chance-level contribution of residual oracle disagreement, so a judge with high corrected AUC is one whose score ordering agrees with the ground-truth ordering beyond what oracle noise can explain; this assumption fails when the oracle's noise is correlated with judge scores (e.g., a judge that mimics the oracle's re-execution quirks), in which case the correction under-corrects and AUC reflects agreement with oracle noise rather than true validity. The per-judge threshold τ_j measures the score at which the judge's vote becomes a valid classifier because Youden's J maximizes true-positive minus false-positive rate on the calibration sample; this fails when the calibration and deployment task distributions differ, in which case τ_j is miscalibrated for deployment. The agreement κ among calibrated judges measures residual uncertainty because conditional on each judge passing the AUC certificate, high disagreement flags states where the environment oracle's validity is ambiguous; this fails when all calibrated judges share a common training bias, in which case κ overstates confidence.

**(D) Assumptions behind κ.** κ among AUC-certified judges measures residual validity uncertainty under assumption A: calibrated judges' errors are conditionally independent given the oracle label. The failure mode F under which κ overstates confidence is shared training bias, which makes errors positively correlated and κ high even when all judges are wrong together. We test A directly in the experiment by measuring pairwise judge error correlation on oracle-labeled states.

## Contribution

["An oracle-calibrated judge gating protocol (OCJG) that replaces consensus-only gating with per-judge AUC certificates against ground-truth environment-state labels, including a budgeted oracle-labeling procedure that reuses WildAWM's heterogeneous rollouts.", "A design principle: judge agreement is only admissible as a quality signal after each judge's discrimination against an environment-state oracle exceeds a calibrated AUC threshold, and uncalibrated judges must be dropped rather than reweighted.", 'A reusable calibration certificate format (AUC + Youden threshold per judge per environment class) that can be plugged into existing synthetic-environment pipelines such as WildAWM, R2E-Gym, and APIGen.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Wild-AWM synthesized computer-use task suite | Matches the environment-synthesis pipeline OCJG replaces |
| Primary metric | Task accept-precision at fixed oracle budget (B_o=200) | Directly measures whether gated tasks are truly valid |
| Secondary metric | Budget-precision curve, B_o ∈ {25, 50, 100, 200, 400} | Quantifies oracle budget needed for target precision gain |
| Secondary metric | Accept-precision vs. embedding MMD between calibration and deployment task sets | Measures calibration-to-deployment transfer degradation |
| Secondary metric | Pairwise judge error correlation on oracle-labeled states | Tests conditional-independence assumption A behind κ |
| Baseline | WildAWM cross-judge consistency gate | Tests whether oracle calibration beats uncalibrated agreement |
| Baseline | R2E-Gym hybrid verifier combination | Tests whether oracle-certified gating beats test-time verifier mixing |
| Baseline | Single strongest judge, no gate | Tests whether gating adds value over best individual judge |
| Ablation-of-ours | OCJG with global agreement threshold, no per-judge AUC | Isolates the per-judge calibration certificate |
| Ablation-of-ours | OCJG with raw (uncorrected) AUC, no ambiguity abstention | Isolates the noise-correction and abstention steps |

**Oracle-construction protocol.** The environment-state oracle is a deterministic state-validity checker (the environment's own predicate) run K=3 times with fresh execution branches per state; a state is labeled valid/invalid only if at least 2 of 3 re-executions agree (agree ≥ 0.67), otherwise it is an abstention. A human spot-check is performed on 10% of labeled states, with inter-annotator agreement (Cohen's κ) reported; states where human and oracle disagree are counted as oracle noise and included in the per-state ε estimate.

### Why this setup validates the claim

The central claim is that a judge's vote should be licensed by measured discrimination against an environment-state oracle, not by agreement with other judges. The dataset must therefore be one where oracle-labeled states exist and where synthesized tasks vary in validity, which the Wild-AWM suite provides since it already rolls out heterogeneous agents and applies judge consistency checks. Accept-precision at a fixed oracle budget is the right metric because OCJG's entire value proposition is spending a small oracle budget once to buy reliable gating thereafter; if precision does not rise relative to baselines at equal budget, the calibration certificate is not doing work. The WildAWM gate baseline tests the sub-claim that uncalibrated cross-judge agreement is a weaker validity signal than oracle-calibrated AUC. The R2E-Gym hybrid verifier baseline tests the sub-claim that test-time verifier combination without oracle measurement cannot substitute for calibration. The single-strongest-judge baseline tests whether gating adds anything beyond picking the best judge. The global-threshold ablation isolates the per-judge certificate; the raw-AUC ablation isolates the noise-correction and abstention steps that make the certificate robust to oracle noise. The budget-precision curve tests practicality under scarce oracle access; the MMD stress test tests the calibration-to-deployment transfer assumption; the pairwise error-correlation measurement tests assumption A behind κ.

### Expected outcome and causal chain

**vs. WildAWM cross-judge consistency gate** — On a case where two judges share a training bias and confidently agree on a state that the environment oracle marks invalid, the baseline accepts the task because its consistency gate treats agreement as evidence. Our method instead drops or downweights a judge whose noise-corrected AUC certificate is low and requires calibrated judges to clear their own Youden thresholds, so the shared-bias pair cannot carry the vote alone. We expect a noticeable precision gap concentrated on tasks where judge agreement is high but oracle validity is low, with parity on tasks where all judges are individually well-calibrated.

**vs. R2E-Gym hybrid verifier combination** — On a case where an execution-free verifier scores a plausible-looking but environment-invalid final state highly and the execution-based verifier is silent because the task has no cheap execution check, the baseline combines the two scores without knowing either verifier's discrimination, so the confident wrong score dominates. Our method instead only admits a verifier that has demonstrated noise-corrected AUC above threshold against oracle-labeled states, so a verifier with no measured discrimination contributes no vote. We expect a precision gain on tasks lacking cheap execution checks, with parity on tasks where execution-based verification is already reliable.

**vs. Single strongest judge, no gate** — On a case where the strongest judge is well-calibrated on average but miscalibrated on a distribution shift within the task suite, the baseline accepts its high score uncritically because it has no per-task disagreement signal. Our method's agreement κ among calibrated judges flags exactly these states as ambiguous, so the gate rejects rather than trusting one judge. We expect the gain to appear as reduced false-accepts on the shifted subset, with the single-judge baseline matching OCJG on the in-distribution subset.

**vs. OCJG with raw (uncorrected) AUC, no ambiguity abstention** — On a case where a judge happens to mimic the oracle's re-execution quirks, the raw-AUC ablation assigns it an inflated certificate and admits its vote, while the full method's noise correction and abstention lower its corrected AUC below AUC_MIN and drop it. We expect the full method to show higher precision than this ablation on the subset of states with high oracle disagreement, with parity on low-disagreement states.

**Budget-precision and transfer.** We expect accept-precision to rise monotonically with B_o and to approach the full-method value by B_o=200; we expect precision to degrade as calibration-to-deployment MMD increases, with the degradation curve quantifying the transfer assumption. We expect pairwise judge error correlation to be positive but below the level at which κ would be fully driven by shared bias, and we report it alongside κ.

### What would falsify this idea

If OCJG's precision gain over the WildAWM gate is uniform across subsets rather than concentrated where judge agreement is high but oracle validity is low, the calibration certificate is not the causal mechanism and the idea is wrong. Likewise, if the global-threshold ablation matches OCJG at equal oracle budget, the per-judge AUC certificate adds nothing and the central claim collapses; and if the raw-AUC ablation matches the full method, the noise-correction and abstention steps are decorative.

## References

1. Agent World Model in the Wild: Computer-Use Agent Environment Synthesis
2. R2E-Gym: Procedural Environments and Hybrid Verifiers for Scaling Open-Weights SWE Agents
3. Generalizable End-to-End Tool-Use RL with Synthetic CodeGym
4. Training Software Engineering Agents and Verifiers with SWE-Gym
5. APIGen: Automated Pipeline for Generating Verifiable and Diverse Function-Calling Datasets
6. FireAct: Toward Language Agent Fine-tuning
7. Self-Instruct: Aligning Language Models with Self-Generated Instructions
