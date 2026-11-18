# Switching-State Contact Observer for Mode-Gated Force Primitives

## Motivation

ForceHarness (arxiv:2611.21099) and related force-aware systems such as ForceVLA (E1) treat a single calibrated force scalar as a sufficient statistic for contact-state identification: self-terminating primitives fire on force thresholds and the physics translation layer reports scalar quantities with uncertainty, but neither establishes that a force reading uniquely determines the contact mode. This structural gap recurs across the force-manipulation literature — ForceVLA fuses 6-axis force as a modality but provides no mode-identifiability layer, and CR-DAgger (E2) relies on compliance control without certifying stick/slip/multi-contact boundaries — so termination events are unsound under bias and unmodeled compliance, and the frozen model receives no explicit belief over contact mode.

## Key Insight

A calibrated force scalar is not an injective function of contact mode under sensor bias and unmodeled compliance, so termination soundness requires an explicit switching-state posterior over stick/slip/free/multi-contact whose transition events — not force thresholds — define the primitive's return condition.

## Method

**What it is.** We introduce **MOCA** (Mode-Observer Contact Arbitration), a probabilistic contact-mode observer that replaces ForceHarness's single calibrated force scalar with a switching-state posterior over four modes {free, stick, slip, multi-contact}. MOCA takes as input a window of calibrated force traces (wrist sensor or joint-torque-derived, as in ForceHarness) plus proprioceptive velocity, and outputs a mode belief $b_t \in \Delta^3$ with a certified transition flag $\tau_t \in \{\text{none}, \text{boundary}\}$ that gates self-terminating primitives.

**Load-bearing assumption (stated explicitly).** MOCA assumes that, after bias/compliance residualization, the force residual is drawn from mode-specific, non-overlapping emission distributions (free/stick/slip/multi-contact), so that the switching-state posterior's argmax and entropy identify the true contact mode rather than sensor-noise-driven ambiguity. This assumption is not asserted; it is measured per task via the identifiability diagnostic in step 6, and $\tau_t$ is only certified where that diagnostic passes.

**How it works.**

```pseudocode
Input: force window F_{t-W:t} (W=16), velocity v_t, prior mode belief b_{t-1}
Hyper: bias prior N(0, sigma_b^2) with sigma_b=0.05 N, compliance prior Beta(2,5), mode persistence p_stay=0.95

1. Bias-compliant residualization:
   r_t = F_t - b_hat_t - k_c(m) * v_t          # subtract estimated bias & per-mode compliance drag
   (b_hat_t, k_c(m)) updated by recursive least squares with forgetting lambda=0.99
   # k_c(m) is a per-mode state (m in {free, stick, slip, multi}), initialized from Beta(2,5) prior

2. Mode likelihoods (switching-state emission, parameters identified from data):
   L_free   = N(r_t; 0, sigma_f^2)
   L_stick  = N(r_t; mu_s, sigma_s^2) * 1[|v_t| < v_eps]
   L_slip   = N(r_t; mu_k, sigma_k^2) * 1[|v_t| >= v_eps]
   L_multi  = mixture over 2-contact force superposition
   # mu_s, mu_k, sigma_s, sigma_k, v_eps are estimated by EM on a calibration set of 512 labeled windows
   # (ground-truth mode labels from HiddenPhys instrumentation), not hand-set.

3. Forward Bayesian update with persistence transition matrix P (diag p_stay):
   b_t = normalize( P^T b_{t-1} * L_t )

4. Boundary certification:
   tau_t = boundary  iff  argmax b_t != argmax b_{t-1}
                        AND  max b_t > 0.8
                        AND  entropy(b_t) < 0.3 nats

5. Gate primitive return: primitive terminates on tau_t == boundary, not on |F| > F_thresh

6. Identifiability diagnostic (per task, computed on the 512-window calibration set):
   For each mode pair (i,j), compute ROC-AUC between mode-conditioned residuals r_t | mode=i and r_t | mode=j.
   Let AUC_min = min over pairs.  If AUC_min < 0.85, the non-overlap assumption is flagged as failed for that task;
   tau_t is suppressed (set to none) and the task is reported as out-of-scope for certified termination.

Output: b_t, tau_t, AUC_min
```

**Why this design.** We chose a switching-state HMM with an explicit persistence matrix over a threshold-on-force rule because ForceHarness's self-terminating primitives implicitly assume the scalar is a sufficient statistic, and a Markov transition prior is the minimal structure that makes mode boundaries identifiable from noisy force under bias — accepting the cost that the transition matrix must be estimated per task family and mis-specification delays termination. We chose recursive least-squares bias/compliance residualization over a learned network because the bias is a slowly varying additive offset whose removal is what makes the emission model well-calibrated, accepting the cost that fast bias drift (thermal, battery sag) is only tracked with the forgetting factor and can leak into the slip likelihood. We chose entropy-and-margin gating for $\tau_t$ over a raw MAP switch because a single-step MAP flip is exactly the failure mode that current force-threshold primitives exhibit under noise, accepting the cost that genuine fast stick-slip oscillations may be under-reported as boundaries. A domain expert would not describe MOCA as a variant of ForceHarness: ForceHarness returns scalar physical quantities with uncertainty to a frozen model, whereas MOCA returns an explicit posterior over a discrete contact-mode state and uses its certified transitions as the primitive's control signal — the output lives on a different surface (mode simplex vs. force scalar) and the termination rule is derived from the posterior's entropy, not from the force magnitude.

**Why it measures what we claim.** The mode belief $b_t$ measures contact-mode identifiability because the switching-state emission model assumes that, conditional on mode and after bias/compliance residualization, the force residual is drawn from a mode-specific distribution with no cross-mode overlap — this assumption fails when unmodeled compliance makes stick and slip emissions overlap (e.g., soft gripper pads), in which case $b_t$ reflects sensor-noise-driven ambiguity rather than true mode. The certified transition $\tau_t$ measures contact-mode boundary soundness because the entropy-and-margin rule assumes that a true mode boundary produces a posterior that is both peaked and switched within one window — this assumption fails during slow stick-to-slip creep, in which case $\tau_t$ reflects a delayed boundary rather than a missed one. The bias/compliance residual $r_t$ measures the calibrated force's sufficiency gap because RLS assumes the bias is additive and slowly varying — this assumption fails under contact-induced compliance changes, in which case $r_t$ absorbs the compliance signal that should have driven the mode likelihood. The identifiability diagnostic AUC_min measures the non-overlap assumption directly because it compares mode-conditioned residual distributions on labeled calibration windows; when AUC_min < 0.85, the diagnostic reports that the assumption fails and $\tau_t$ is suppressed, so the method does not claim certified boundaries where the emission model cannot separate modes.

## Contribution

['A switching-state contact-mode observer (MOCA) that replaces the single calibrated force scalar with a four-mode posterior and a certified transition flag, providing an explicit uncertainty model over contact mode where prior force-based systems provide only scalar uncertainty.', 'A mode-gated termination rule for self-terminating force primitives in which return fires on certified posterior transitions rather than force thresholds, with the entropy-and-margin certification criterion specified as the soundness condition.', 'An empirical design principle: bias/compliance residualization is the load-bearing step that makes mode emission distributions separable, so primitive termination soundness is bounded by residualization quality rather than by force sensor resolution.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | HiddenPhys seven-task MuJoCo benchmark | Hidden physics stresses mode ambiguity directly |
| Primary metric | Task success rate under mode-boundary termination | Directly scores boundary soundness, not force magnitude |
| Secondary metric | Mode-boundary F1 and termination timing error (ms) against ground-truth mode labels | Quantifies boundary detection quality |
| Secondary metric | Posterior calibration: ECE and reliability diagram for b_t vs. ground-truth mode | Tests whether b_t is calibrated |
| Baseline | ForceHarness scalar self-terminating primitives | Tests whether mode posterior beats scalar sufficiency |
| Baseline | Expert force controllers | Tests whether learned posterior rivals hand-tuned thresholds |
| Baseline | Learned force-conditioned policy | Tests whether explicit mode state beats implicit encoding |
| Baseline | Classical friction-cone / LuGre-style contact observer | Tests against existing contact-state estimators |
| Baseline | Learned mode classifier (small transformer over same force window) | Tests whether switching-state posterior is more than a re-derivation |
| Ablation-of-ours | MOCA minus boundary certification (raw MAP switch) | Isolates entropy-and-margin gating contribution |
| Ablation-of-ours | MOCA minus identifiability diagnostic (always certify) | Isolates the effect of suppressing tau_t when AUC_min < 0.85 |

### Why this setup validates the claim

The central claim is that replacing ForceHarness's scalar force statistic with an explicit switching-state mode posterior makes contact-mode boundaries identifiable under bias and noise, and that certified boundaries improve primitive termination. HiddenPhys is the right testbed because its hidden physical variables (friction, compliance, mass) are precisely what make a single calibrated scalar an insufficient statistic — so any advantage of mode structure should surface there rather than on visually easy tasks. The scalar ForceHarness baseline tests the sufficiency sub-claim directly: if the posterior adds nothing, termination success should match. The expert force controller tests whether MOCA's learned posterior is merely re-deriving hand-tuned thresholds or actually capturing bias/compliance residualization that thresholds miss. The learned force-conditioned policy tests whether explicit discrete mode state beats an implicit encoding of the same force window. The classical friction-cone / LuGre-style observer and the learned mode classifier are added to demonstrate that the switching-state posterior plus certified transitions is not merely a re-derivation of existing contact-state estimators; mode-boundary F1 and termination timing error against ground-truth mode labels are reported for all methods. The ablation — MOCA with raw MAP switching instead of entropy-and-margin gating — is the key internal control: it holds the emission model and residualization fixed while removing only the certification rule, so any gap between it and full MOCA is attributable to boundary certification rather than to the mode representation itself. The identifiability diagnostic ablation (always certify) tests whether suppressing tau_t when AUC_min < 0.85 prevents unsound certifications. Success rate under boundary-triggered termination is the right metric because the claim is about when primitives stop, not about force accuracy; a force-tracking metric would be blind to premature or delayed termination, which is exactly the failure the method targets. Calibration (ECE, reliability diagrams) and boundary-detection latency/false-positive/false-negative rates against ground-truth mode labels quantify the gap between tau_t and true mode-boundary soundness.

### Expected outcome and causal chain

**vs. ForceHarness scalar self-terminating primitives** — On a HiddenPhys task where friction is hidden and the gripper pad is compliant, the baseline's self-terminating primitive fires on a raw force crossing because its scalar conflates compliance drag with contact onset, so it terminates early during the approach-to-stick transition. MOCA residualizes bias and compliance drag via RLS before emission, so the stick likelihood dominates only after true contact, and the certified boundary gates termination on a peaked switched posterior. We expect a clear success-rate gap concentrated on the high-compliance, high-friction-uncertainty task subset, with near-parity on the rigid, low-friction tasks where the scalar is nearly sufficient. On the secondary metrics, MOCA should show higher mode-boundary F1 and lower termination timing error, with calibration ECE below 0.1 on tasks where AUC_min ≥ 0.85.

**vs. Expert force controllers** — On a task requiring slow stick-to-slip creep, the expert controller's fixed threshold lags the true slip onset because its gains are tuned for a nominal friction that the hidden variable violates, producing late termination and overshoot. MOCA's persistence prior and slip emission conditioned on velocity let the posterior track the creep transition, though we expect the entropy gate to report it slightly late rather than miss it. The observable signal is a modest but consistent gap favoring MOCA on the creep-heavy subset and rough parity on tasks with sharp contact events, where expert thresholds are already well matched. Boundary-detection latency should be lower for MOCA on creep tasks, while false-positive rates should remain comparable.

**vs. Learned force-conditioned policy** — On a multi-contact task where two contacts superimpose, the learned policy's implicit encoding averages the superimposed force into a single feature and misclassifies the compound state, terminating on the wrong contact event. MOCA's explicit multi-contact mixture emission represents the superposition as a distinct mode, so the posterior separates compound from single contact before boundary certification. We expect MOCA to win specifically on the multi-contact subset while the learned policy remains competitive on single-contact tasks, confirming the gain comes from explicit mode structure rather than from extra capacity. Mode-boundary F1 should be higher for MOCA on multi-contact tasks.

**vs. Classical friction-cone / LuGre-style contact observer** — On tasks with hidden friction and compliance, the classical observer relies on a fixed friction-cone model and may misestimate stick/slip boundaries when the true friction coefficient deviates. MOCA's data-identified emission parameters and per-mode compliance term should adapt better, yielding higher boundary F1 and lower timing error on high-friction-uncertainty tasks, while the classical observer may be competitive on tasks where its model matches the hidden physics.

**vs. Learned mode classifier (small transformer)** — The learned classifier may overfit to training force windows and produce noisy mode predictions, leading to more false-positive boundaries. MOCA's switching-state prior and entropy gating should suppress spurious flips, resulting in higher precision on boundary detection and lower premature-termination counts, especially on noisy subsets.

**vs. MOCA minus boundary certification** — On a noisy contact task, the raw-MAP ablation flips mode on single-step force noise, firing spurious boundaries and terminating primitives prematurely, because it lacks the margin and entropy conditions. Full MOCA suppresses these flips, so we expect the ablation to show inflated premature-termination counts and lower success on the noisy subset, with the two matching on clean low-noise tasks — localizing the benefit to the certification rule.

**vs. MOCA minus identifiability diagnostic** — On tasks where AUC_min < 0.85 (e.g., soft-pad compliance causing stick/slip overlap), the always-certify ablation will issue unsound tau_t events, leading to incorrect terminations. Full MOCA suppresses tau_t on those tasks, so success rate and boundary F1 should be higher for full MOCA on the low-identifiability subset, while matching on high-identifiability tasks.

### What would falsify this idea

If MOCA's success-rate advantage over the scalar ForceHarness baseline is uniform across all HiddenPhys tasks rather than concentrated on the high-compliance, high-friction-uncertainty, and multi-contact subsets where the scalar sufficiency gap is predicted to bite, then the mode posterior is not doing the causal work claimed and the central claim is wrong. A second falsifier: if the raw-MAP ablation matches full MOCA everywhere, boundary certification contributes nothing and the entropy-and-margin rule is unjustified. A third falsifier: if the identifiability diagnostic (AUC_min) does not correlate with where MOCA's advantage appears, then the non-overlap assumption is not the governing factor.

## References

1. ForceHarness: Turning Force into Action and Physics for Frozen Frontier Models
