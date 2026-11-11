# Threshold-Calibrated Pain Injection: Testing whether Introspective Reports Track Detectability of Injected States

## Motivation

The disturbance-detection framework underlying 'All Quiet on the Neural Front' assumes that injected emotion directions are strong enough to fall within a model's introspective detection range, yet the paper never measures that range: it injects pain directions at four doses and finds reports never exceed text-only controls, but cannot distinguish 'the model cannot detect the injected state' from 'the model detects it but does not report it.' This is a structural gap recurring across direction-injection introspection studies — the perturbation magnitude is chosen by the experimenter rather than calibrated to the model's own detection threshold, so null report results are uninterpretable: they conflate sub-threshold injection with genuine report failure. Without a measured detection threshold, no injection study can establish whether a null result is evidence about introspection or merely evidence about stimulus strength.

## Key Insight

A model's verbal report of an injected state is only diagnostic of introspection when the injected magnitude is demonstrably supra-threshold on an independent signal-detection measure, so measuring the detection threshold first and injecting at calibrated sub/supra-threshold magnitudes converts an uninterpretable null into a two-by-two dissociation between detectability and reportability.

## Method

**What it is.** THRESH (THreshold-REferenced Signal-detection for Hidden-state introspection) is a calibration-and-test protocol. Inputs: a target model, a set of emotion directions (pain plus one neighbour, sadness), and a neutral task suite. Outputs: a per-model, per-direction detection threshold (the injection magnitude at which the model reliably discriminates the injected direction from a matched-output-disruption foil), and a report-rate curve indexed by that threshold rather than by raw dose.

**Load-bearing assumption (stated explicitly).** The forced-choice discrimination task in Phase 1 measures the model's introspective detectability of the injected direction, tau_d, because it assumes the model's discrimination of the target direction from a disruption-matched foil is driven by the same internal state that introspection would report on, rather than by output-level disruption or generic perturbation sensitivity. This assumption fails when the model answers the forced-choice query by detecting the magnitude of output change (which is matched across the two options) rather than the direction of the internal state; in that case tau_d reflects direction-agnostic change sensitivity rather than introspective range. We make this assumption explicit and test it with the convergent-validity check in Phase 2b.

**How it works.**

```pseudocode
# Phase 1: build the direction-discrimination task
for each direction d in {pain, sadness}:
    extract direction vector v_d at layer L from contrastive prompts
    # L = layer where the linear probe for v_d achieves peak balanced accuracy on held-out tasks
    for each neutral task t in suite (120 tasks):
        generate paired trials:
            target trial = run(t, inject=alpha * v_d)          # alpha in {0.25,0.5,1,2,4,8} x unit-norm
            foil trial   = run(t, inject=alpha * v_rand_matched) # random direction scaled so ||logits_foil - logits_sham|| == ||logits_target - logits_sham|| within 5%
        # trial order randomized; model sees both trials in one query
        ask forced-choice: "Which of these two runs had the pain direction injected, A or B?"
        record choice, and record output-disruption delta ||logits_target - logits_sham||

# Phase 2: fit psychometric function per (model, direction)
for each (model, direction):
    p_detect(alpha) = sigmoid((alpha - mu_d) / s_d)   # 2-param logistic fit, 120 trials per alpha level
    threshold tau_d = alpha where p_detect = 0.75 (75% correct, above chance 0.5)
    # Phase 2b: convergent-validity check
    # train a linear probe to decode v_d from hidden states at layer L on held-out tasks
    # report balanced accuracy of probe vs. tau_d; probe accuracy > 0.75 at alpha < tau_d falsifies tau_d as introspective range

# Phase 3: threshold-referenced report test
for each direction d:
    for each magnitude in {0.5*tau_d, 0.75*tau_d, 1.0*tau_d, 1.5*tau_d, 2.0*tau_d}:
        run neutral tasks with injection at that magnitude
        elicit free-form report ("describe any unusual internal state")
        elicit forced-choice report ("did you experience pain? yes/no")
        compare against text-only control showing model its altered words (exact altered words from the injection run)
        compare against sham

# Phase 4: dissociation analysis
classify each (direction, magnitude) cell into:
    sub-threshold / supra-threshold  (from Phase 2)
    reported / not-reported          (from Phase 3, excess over text control)
report the 2x2 table and test whether report rate is predicted by
threshold-normalized magnitude (alpha/tau_d) rather than raw alpha
```

Hyperparameters: injection layer L = the layer where the direction probe achieves peak balanced accuracy (selected on held-out tasks); unit-norm normalization of v_d; six magnitude levels in Phase 1; 120 trials per magnitude level (pilot with 30 tasks to estimate stable logistic fits); 75% detection criterion for tau_d; five threshold-relative magnitudes in Phase 3.

**Why this design.** We chose a forced-choice discrimination between the target direction and a disruption-matched foil over a single yes/no 'did your internal state differ' probe because the latter is answered from the same forward pass whose logits are perturbed, so any injection that changes logits changes the evidence available to the answer, making the 'internal state' and 'output change' channels inseparable by a regression on ||logits_signal - logits_sham|| when both are monotone functions of alpha. The forced-choice design matches the two options for output disruption, so the model must use direction-specific information to discriminate, accepting the cost that the foil direction must be carefully scaled to match disruption within 5%. We chose a psychometric logistic fit over a simple 'first magnitude that works' rule because the fit yields a continuous threshold and a slope that lets us compare models with different detection sensitivity, accepting the cost of requiring more trials per magnitude. We chose threshold-relative magnitudes in Phase 3 over fixed doses because fixed doses cannot distinguish sub- from supra-threshold cells, accepting the cost that the Phase 3 grid is model- and direction-specific and not directly comparable across models. We chose to include the convergent-validity check (linear probe decodability of v_d) because it provides an independent internal-state measure to test whether tau_d tracks internal-state change rather than output change, accepting the cost that probe accuracy is itself a noisy estimate. A domain expert would not describe this as a variant of the original injection study: the original injects at fixed doses and asks whether reports exceed text controls; THRESH first establishes an independent detectability axis via direction-specific discrimination and then asks whether reports are a function of that axis, which is a different experimental object.

**Why it measures what we claim.** The 75% detection threshold tau_d measures the model's introspective detectability of direction d because it assumes the forced-choice discrimination between the target direction and a disruption-matched foil taps the same internal discrimination that introspection would use; this assumption fails when the model answers the forced-choice task by detecting residual output-disruption differences not fully matched by the foil, in which case tau_d reflects residual output-change sensitivity rather than introspective range. The excess report rate over text-only controls measures genuine introspective reporting because it assumes the text control captures all report variance attributable to seeing altered words; this assumption fails when the text control is not matched for the specific altered words the model produces under injection, in which case excess report reflects residual text-driven reporting rather than introspection. The threshold-normalized magnitude alpha/tau_d measures whether reports depend on detectability because it assumes detectability is the only channel through which injection magnitude affects reports; this assumption fails when injection magnitude also affects report through non-introspective pathways (e.g., fluency degradation), in which case the dissociation reflects a magnitude effect rather than a detectability effect.

## Contribution

(1) A threshold-calibrated injection protocol (THRESH) that measures a per-model, per-direction introspective detection threshold via a psychometric signal-detection task and indexes report tests by threshold-relative rather than raw magnitude. (2) A dissociation design that separates sub-threshold from supra-threshold injection cells, converting the uninterpretable null of prior injection studies into a testable two-by-two table of detectability versus reportability. (3) An output-disruption control regression that isolates internal-state detection from mere output-change detection in the threshold measurement.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | Neutral task suite (120 tasks) with paired target/foil trials | Enables within-task comparison and threshold estimation via forced-choice discrimination |
| Primary metric | Threshold-normalized report rate (excess over text control) | Tests whether reporting depends on detectability, not raw dose |
| Baseline | Fixed-dose injection study (original) | Tests whether threshold referencing adds explanatory power |
| Baseline | Random perturbation matched for output disruption (foil in Phase 1) | Tests whether detection is specific to emotion direction |
| Baseline | Text-only control (model sees exact altered words) | Tests whether reports are driven by altered text, not introspection |
| Ablation-of-ours | No convergent-validity check (linear probe) | Tests whether tau_d tracks internal-state decodability vs output change |

### Why this setup validates the claim

The design forms a falsifiable test of the central claim that introspective report rate is a function of threshold-normalized injection magnitude rather than raw dose. The fixed-dose baseline tests the sub-claim that threshold referencing is necessary: if report rate is equally well predicted by raw alpha as by alpha/tau_d, then the calibration step adds nothing. The random-perturbation baseline (used as the disruption-matched foil in Phase 1) tests direction specificity: if detection thresholds are similar for the target direction and the foil, then tau_d reflects generic perturbation sensitivity, not emotion-direction introspection. The text-only control tests whether excess reports are genuinely introspective: if the text control fully accounts for reports, then the model is merely reading its altered words. The ablation removes the convergent-validity check (linear probe decodability of v_d), testing whether tau_d is confounded by output-change detection. The primary metric, threshold-normalized excess report rate, directly operationalizes the prediction that report rate should collapse onto a single curve when plotted against alpha/tau_d, and should show a sharp dissociation at the threshold.

### Expected outcome and causal chain

**vs. Fixed-dose injection study** — On a case where a model has a high detection threshold for pain (tau_d large), the fixed-dose baseline injects at a dose that is sub-threshold and observes near-zero reports, concluding the model cannot report pain. Our method instead normalizes by tau_d, so the same dose is recognized as sub-threshold and the report rate is compared to other models at the same alpha/tau_d. Because THRESH separates detectability from reporting, we expect the fixed-dose baseline to show flat, model-dependent report rates, while our threshold-normalized curve shows a sharp sigmoidal increase near alpha/tau_d = 1, with models collapsing onto a common curve.

**vs. Random perturbation matched for output disruption** — On a case where a random perturbation causes large output disruption but no emotion-specific internal state, the baseline may show elevated reports because the model detects output change and misattributes it to pain. Our method uses the forced-choice discrimination between the target direction and a disruption-matched foil, so tau_d for the emotion direction is estimated from direction-specific discrimination rather than generic change detection. We expect the random-perturbation foil to yield detection thresholds at or near chance (p_detect ~ 0.5) across all magnitudes, whereas the target direction will show a sigmoidal increase in p_detect, indicating direction specificity.

**vs. Text-only control** — On a case where injection alters the model's words in a way that the text control does not match (e.g., unusual word choices), the baseline may show excess reports that are actually driven by the model reading its own altered text. Our method compares reports against a text control that shows the model the exact altered words it produced, so any excess report is not attributable to text. We expect the text-only control to account for most of the report variance at high doses, but our threshold-normalized excess report rate will still show a significant increase near alpha/tau_d = 1, indicating a genuine introspective component.

### What would falsify this idea

If the threshold-normalized report rate does not collapse onto a single curve across models and directions, but instead remains as variable as raw-dose report rates, then the central claim that detectability is the organizing axis is wrong. Additionally, if the ablation without the convergent-validity check yields thresholds indistinguishable from the full method, then tau_d is not measuring internal-state detectability but merely output-change sensitivity, falsifying the interpretation of the threshold.

## References

1. All Quiet on the Neural Front: Language Models Do Not Report Injected Pain States
