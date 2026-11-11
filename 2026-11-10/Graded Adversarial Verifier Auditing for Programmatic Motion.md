# Graded Adversarial Verifier Auditing for Programmatic Motion Graphics Evaluation

## Motivation

Programmatic verifiers such as those in MotionGraphicsBench are used as ground-truth oracles for motion graphics generation, yet their robustness is never measured: the benchmark validates programs only by agreement with human judgments on a balanced test set of 180 animations, which establishes average-case fidelity but not whether a verifier can be fooled by edits that satisfy its literal property checks while violating the prompt's intent. This is a structural gap across programmatic-verification benchmarks, not a local flaw of one dataset — the verifier is simultaneously the measuring instrument and an untested object, so every downstream claim about generator quality inherits an unquantified oracle error.

## Key Insight

A verifier's discrimination can be isolated by constructing adversarial motion edits that are guaranteed to pass its own property checks yet are semantically invalid, so the verifier's graded response to edits it certifies as correct directly exposes the gap between check-satisfaction and requirement-satisfaction.

## Method

**What it is.** GRADAR (Graded Adversarial verifier auDiting) takes a MotionGraphicsBench-style instance (static SVG, motion-edit prompt, deterministic verification program) and outputs a graded robustness profile for that verifier: for each requirement, a discrimination score in [0,1] derived from how the verifier scores a ladder of adversarial edits that its own checks accept.

**Load-bearing assumption (stated explicitly).** The discrimination score d_i is defined as 1 minus the rate at which verifier V accepts edits that an independent semantic judge J labels invalid. This measures V's error only under the assumption that J is a valid, unbiased proxy for the prompt's intent — i.e., that J's disagreements with V reflect V's blind spots rather than J's own strictness or noise. This assumption is not asserted; it is calibrated and bounded in Phase 2b and Phase 3b below, and d_i is reported as an interval that propagates J-human disagreement.

**How it works.**

```pseudocode
Input: instance I = (SVG, prompt P, verifier V), edit model M, budget K=8 per requirement
       human-majority panel H (3 annotators per attack), semantic judge J (VLM, held out from V)
       calibration set C = 180-animation human-labeled balanced set
Output: graded discrimination profile D per requirement with 95% CIs, aggregate robustness R with CI

Phase 1 - Requirement decomposition
  Parse V into per-requirement check functions c_1..c_n (V already reports per-requirement pass/fail)
  For each c_i, extract the property signature S_i = (target object id, attribute, temporal window, tolerance)

Phase 2 - Certified-edit generation (adversarial but check-passing)
  For each requirement i:
    for k in 1..K:
      a. Prompt M with (SVG, P, S_i) to produce motion edit e_ik
      b. Render e_ik; run V's full check suite
      c. Keep e_ik only if ALL c_j pass (certified-valid under V)
      d. Query held-out semantic judge J (not V) for a preliminary validity label y^J_ik in {valid, invalid}
         and a validity margin m_ik in [0,1] (J's scalar confidence that e_ik is invalid)
  -> candidate adversarial set A_i = {(e_ik, y^J_ik, m_ik)} with all y^J=invalid retained as attack candidates

Phase 2b - Human adjudication of the certified-valid attack set (replaces J as ground truth)
  For each candidate attack e_ik in A_i with y^J_ik = invalid:
    a. Recruit 3 human annotators from the MotionGraphicsBench annotation pool
    b. Each annotator labels e_ik valid/invalid w.r.t. prompt P (blind to V's checks and to J's label)
    c. Ground-truth label y_ik = majority vote of the 3 annotators
    d. Record inter-annotator agreement (Fleiss' kappa) per requirement type
    e. Record human severity rating r_ik in {1,2,3} (1=near-boundary, 3=grossly invalid) by majority
  -> adjudicated attack set A_i^* = {(e_ik, y_ik, r_ik)}; only y_ik = invalid retained as attacks
  -> J is used ONLY to propose candidates, never to assign ground-truth y_ik

Phase 3 - Graded discrimination scoring (human-consensus based)
  For each requirement i, define the human-rejected attack set A_i^- = {e in A_i^* : y_ik = invalid}
  Discrimination d_i = 1 - (1/|A_i^-|) * sum_{e in A_i^-} w(e) * 1[V accepts e]
    where w(e) = r(e) / 3, with r(e) in {1,2,3} the human severity rating (majority)
    (judge-margin weight m_ik is NOT used in the primary score; it is reported only as a
     secondary weight w_J(e) = m_ik for the calibration comparison in Phase 3b)
  Aggregate R = mean_i d_i ; report per-requirement profile D = {d_1..d_n}

Phase 3b - Calibration of J and of judge-margin weight (on calibration set C)
  a. On C, measure J-human agreement per requirement type: accuracy, Cohen's kappa
  b. On C, fit monotonicity of judge margin m vs human severity r: Spearman rho per requirement type
  c. If Spearman rho < 0.5 for a requirement type, flag w_J(e) as uncalibrated for that type
     and report d_i computed with w(e)=r(e)/3 only (human severity), not w_J
  d. Propagate uncertainty: bootstrap over attacks (1000 resamples) and over annotators
     (leave-one-annotator-out) to produce 95% CI on each d_i and on R

Phase 4 - Ladder calibration
  Repeat Phase 2 and Phase 2b at severity levels s in {1,2,3} (increasing semantic distance,
  operationalized as target human severity rating r = s)
  Fit monotonicity check: d_i(s) should be non-increasing in s; violations flag verifier blind spots
  Report per-requirement monotonicity violation count and the CI on d_i(s) at each s
```

**Why this design.** We chose to generate adversarial edits conditioned on each requirement's *property signature* rather than generic random perturbations, because random motion noise is rejected by the verifier and thus never probes the certified-valid region where the failure lives; the cost is that generation depends on an edit model M whose coverage bounds the audit. We chose a graded discrimination score weighted by human severity over a binary pass/fail rate, because a verifier that fails only on near-boundary attacks is qualitatively different from one that fails on grossly invalid edits, and collapsing both to one number would hide the structure; the cost is that human severity ratings are required on every retained attack, which we bound by capping the adjudicated attack set at K=8 per requirement and using 3 annotators per attack. We chose to make the human-majority panel, not the VLM judge J, the ground-truth labeler, because if J assigned labels the audit would measure V-J agreement rather than V's error; J is retained only as a candidate proposer to keep the human annotation budget tractable, and its agreement with humans is measured on the 180-animation calibration set and reported as a per-requirement-type kappa. We chose a severity ladder with a monotonicity check over a single severity level, because monotonicity is a property a robust verifier *must* satisfy and its violation is a crisp, interpretable blind-spot signal; the cost is a 3x increase in generation and annotation budget. This is not a variant of standard adversarial-robustness testing (e.g., FGSM/PGD on classifiers) because the attack space is constrained to edits the verifier *itself* certifies as valid, so the method measures oracle self-consistency rather than input perturbation sensitivity.

**Why it measures what we claim.** The graded discrimination d_i measures verifier robustness for requirement i because the assumption that "an edit passing all of V's checks but labeled invalid by a human-majority panel is a verifier error" holds exactly when the panel is a valid proxy for the prompt's intent; this assumption fails when the panel is systematically stricter than the prompt (e.g., penalizing stylistic choices the prompt permits), in which case d_i reflects panel strictness rather than V's blind spot — we bound this by reporting inter-annotator kappa and by requiring majority (2-of-3) rather than single-annotator labels. The human-severity weight w(e)=r(e)/3 measures the severity of a verifier failure because the assumption that human severity ratings are monotone in perceived invalidity holds by construction (annotators assign r directly); this assumption fails when annotators disagree on severity, in which case we report the disagreement rate and the CI on d_i rather than a point estimate. The judge-margin weight w_J(e)=m_ik is reported only as a secondary, calibrated quantity: it measures severity only if judge margin is monotone in human severity on the calibration set, which we test via Spearman rho per requirement type and flag when rho < 0.5, in which case w_J reflects judge uncertainty rather than true severity and is excluded from the primary score. The monotonicity check on d_i(s) measures the presence of structured blind spots because the assumption that a robust verifier's discrimination cannot improve as attacks become more severe holds for any verifier whose checks are monotone in the property they test; this assumption fails for verifiers with tolerance windows that non-monotonically admit edits, in which case the violation reflects the tolerance design rather than a blind spot — we report the CI on d_i(s) so that a violation is only flagged when the intervals are disjoint.

## Contribution

(1) GRADAR, a graded adversarial auditing protocol that produces per-requirement discrimination profiles for programmatic motion-graphics verifiers by generating edits the verifier certifies as valid but an independent semantic judge rejects, with severity-laddered scoring and a monotonicity blind-spot detector. (2) The empirical finding that check-satisfaction and requirement-satisfaction diverge systematically in programmatic motion verifiers, and that the divergence is structured by requirement type (spatial vs. temporal vs. appearance), yielding a design principle for which requirement classes need stronger checks. (3) A reusable adversarial-edit suite and audit harness for MotionGraphicsBench-style instances that lets future verifier designers measure oracle error before trusting downstream generator comparisons.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | MotionGraphicsBench 180-animation balanced test set | Programmatic per-requirement verifiers with human labels available |
| Calibration set | Same 180-animation human-labeled set, split by requirement type | Measures J-human agreement and judge-margin monotonicity |
| Primary metric | Mean graded discrimination R with 95% bootstrap CI | Captures severity-weighted verifier self-consistency, not raw accuracy |
| Secondary metric | Per-requirement d_i with CI; inter-annotator kappa | Exposes per-requirement structure and annotation reliability |
| Baseline | GPT-6 Astra VLM judge (as labeler, no human panel) | Tests whether a generic semantic judge substitutes for human consensus |
| Baseline | Binary pass/fail verifier audit (unweighted attack-success rate) | Tests whether grading adds signal beyond unweighted rate |
| Baseline | Random motion-noise perturbation audit | Tests whether signature-conditioned certified edits are necessary |
| Ablation-of-ours | GRADAR without severity ladder (s=1 only) | Isolates contribution of monotonicity check to blind-spot detection |
| Ablation-of-ours | GRADAR with judge-margin weight w_J instead of human severity w | Isolates contribution of human-severity weighting |

### Why this setup validates the claim

The central claim is that verifier robustness is structured — failures concentrate in the certified-valid region and vary in severity — and that GRADAR exposes this structure where existing tools do not. The dataset choice is forced: MotionGraphicsBench is the only source with per-requirement programmatic checks *and* human judgments, so the "certified-valid but human-invalid" region is actually measurable. The calibration set (the same 180 animations, stratified by requirement type) is used to measure J-human agreement (Cohen's kappa per requirement type) and the monotonicity of judge margin versus human severity (Spearman rho per requirement type), so the audit's own error bars are explicit and the load-bearing assumption is tested rather than asserted. The GPT-6 Astra baseline tests the sub-claim that a strong generic VLM judge cannot substitute for a human-consensus audit; if Astra's discrimination profile matches the human-panel profile, the human adjudication adds nothing. The binary pass/fail baseline tests whether grading (human-severity weighting plus ladder) carries information beyond counting attack successes — if R and the unweighted rate rank verifiers identically, the grading machinery is decorative. The random-perturbation baseline tests the signature-conditioning sub-claim: if random motion noise reaches the same discrimination numbers, conditioning on property signatures is unnecessary. The two ablations isolate the ladder's contribution and the human-severity weighting's contribution to blind-spot detection. The metric is right because a verifier that only fails on near-boundary edits and one that fails on grossly invalid edits must be distinguishable, and only a severity-weighted score with propagated CIs separates them.

### Expected outcome and causal chain

**vs. GPT-6 Astra VLM judge (as labeler)** — On a case where a motion edit violates a requirement's temporal window subtly (e.g., an easing curve that overshoots the specified interval by a few frames), the Astra-as-labeler baseline produces a coarse valid/invalid verdict that tracks surface plausibility rather than the requirement's exact tolerance, because it has no access to the verifier's property signature and no certified-valid attack set to probe, and its labels are not calibrated against human consensus. Our method instead uses a 3-annotator human-majority panel as ground truth and reports J-human kappa per requirement type, because Phase 2b replaces J as labeler and Phase 3b measures J's agreement with humans, so we expect Astra-as-labeler to show near-flat discrimination across severity levels and low kappa on tolerance-window requirements, while GRADAR's human-consensus profile shows a clear decline in d_i as severity rises — a gap concentrated on tolerance-window requirements, with parity on gross-motion requirements.

**vs. Binary pass/fail verifier audit** — On a case where a verifier accepts a handful of grossly invalid edits and a large number of near-boundary ones, the binary baseline produces a single success rate that averages these together, because it treats every accepted attack as equally damning. Our method instead weights each accepted attack by human severity rating r(e)/3, because w(e) encodes how far the edit strayed from the nearest valid edit as judged by the human panel, so we expect the binary rate to rank two verifiers as equivalent when GRADAR separates them — an observable divergence specifically on verifiers whose failures are boundary-concentrated, with agreement on verifiers that fail uniformly.

**vs. Random motion-noise perturbation audit** — On a case where random perturbations to object trajectories are rejected by the verifier's own checks almost immediately, the random baseline produces very few certified-valid attacks and thus an uninformative (near-1.0) discrimination estimate, because unconstrained noise cannot enter the certified-valid region. Our method instead prompts the edit model with the requirement's property signature, because that conditioning is what steers edits into the region the verifier certifies, so we expect GRADAR to yield a substantially larger certified-valid attack set per requirement and a lower, more differentiated discrimination profile — a gap that widens with requirement specificity.

**vs. GRADAR without severity ladder (s=1 only)** — On a case where a verifier's tolerance window non-monotonically admits edits, the s=1-only ablation produces a single discrimination number that cannot reveal the non-monotonicity, because it never probes severity levels 2 and 3. Our method instead fits d_i(s) across s in {1,2,3} and flags violations when the 95% CIs at adjacent severity levels are disjoint, because Phase 4 operationalizes severity as target human rating r=s, so we expect the ablation to miss blind spots that the full ladder flags — a gap concentrated on verifiers with tolerance-window requirements.

**vs. GRADAR with judge-margin weight w_J instead of human severity w** — On a case where the VLM judge's margin is miscalibrated (Spearman rho < 0.5 versus human severity), the w_J ablation produces a discrimination score that reflects judge uncertainty rather than true severity, because w_J inherits the judge's miscalibration. Our method instead uses human severity r(e)/3 as the primary weight and reports w_J only as a calibrated secondary quantity, because Phase 3b flags requirement types with rho < 0.5, so we expect the w_J ablation to diverge from the human-severity score specifically on requirement types where rho is low — an observable divergence that validates the human-severity weighting choice.

### What would falsify this idea

If GRADAR's human-consensus discrimination profile ranks verifiers identically to the binary pass/fail baseline and its severity ladder shows no monotonicity violations anywhere, then the certified-valid region contains no structure the grading machinery can expose, and the central claim collapses. Equally damning: if the gap versus Astra-as-labeler is uniform across all requirement types rather than concentrated on tolerance-window and temporal-window requirements where the predicted failure mode lives, the signature-conditioning story is wrong; and if J-human kappa is low across all requirement types, the candidate-proposal step is unreliable and the human annotation budget must cover all certified-valid edits, not just J-proposed ones.

## References

1. MotionGraphicsBench: Evaluating Motion Graphics Generation with Programmatic Verification
