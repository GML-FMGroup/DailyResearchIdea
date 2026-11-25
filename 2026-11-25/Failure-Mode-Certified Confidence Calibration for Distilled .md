# Failure-Mode-Certified Confidence Calibration for Distilled Medical Language Models

## Motivation

Distilled medical language models such as KMed-31B report large benchmark gains (HealthBench Hard 22.95→41.87), yet benchmark accuracy is structurally silent about whether the student's expressed confidence tracks correctness on the cases where medical models actually fail: MediConfusion shows all available medical MLLMs fall below random guessing on expert-distinct adversarial image pairs, and KMed-31B's rubric-judged clinical inquiry has no single reference answer at all. The root cause is that MOPD^2 integrates specialists by matching output distributions under feedback-appropriate rewards, but never instantiates a quantity that is monotone in empirical correctness conditional on a known failure mode, so the student's confidence inherits the specialists' aggregate behavior rather than being certified against the specific perturbations (expert-distinct image pairs, rubric-graded open inquiry) that define deployment reliability.

## Key Insight

Certifying confidence against a named failure mode is possible without retraining the student because the specialist decomposition induced by MOPD^2 provides a per-failure-mode disagreement statistic that is monotone in empirical error rate on held-out adversarial pairs, making the calibration map identifiable from a small probe set rather than requiring the student's raw log-probs to be calibrated.

## Method

**What it is.** We introduce FMC (Failure-Mode-Certified calibration), a lightweight post-hoc module attached to a frozen KMed-31B student. FMC takes as input (i) the student's top-1 answer token distribution and its entropy, (ii) the per-specialist answer agreement vector produced by the four MOPD^2 teachers at inference time, and (iii) a failure-mode tag from a probe classifier; it outputs a scalar certified confidence c ∈ [0,1] and a binary abstain flag. The probe classifier is trained once on the MediConfusion-style adversarial pair set and on rubric-scored Synthetic Health Dialogue items.

**Load-bearing assumption (stated explicitly).** FMC's central claim rests on the assumption that the mean fraction of MOPD^2 specialists agreeing with the student's top-1 answer, mean(a), is monotone in the student's empirical error probability conditional on the failure-mode tag, so that a 3-feature logistic + isotonic map fit on ~200 probe items per mode yields a certified confidence that transfers to deployment items of the same tag. This assumption fails when specialists agree for correlated reasons (shared pretraining contamination on the same adversarial pair), in which case mean(a) reflects shared bias rather than boundary proximity. We therefore replace mean(a) with a decorrelated disagreement statistic and validate monotonicity before fitting isotonic regression, as specified below.

**How it works.**

```pseudocode
# Phase 0: build the failure-mode probe set (offline, once)
D_probe = MediConfusion_pairs  ∪  SyntheticHealthDialogue_rubric_items
for each item x in D_probe:
    y_hat   = student(x)                       # frozen KMed-31B
    p_top   = max softmax(logits)[y_hat]        # raw confidence
    H       = entropy(softmax(logits))          # answer entropy
    a_raw   = [teacher_k(x)==y_hat for k in 1..4]   # MOPD^2 specialist agreement
    a_base  = [base_model(x)==y_hat]            # shared-base reference model answer
    # decorrelated disagreement: residual of a_raw after regressing out p_top and a_base
    a_resid = a_raw - (beta0 + beta1*p_top + beta2*a_base)   # fit beta on held-out half of Z
    m       = failure_mode_tag(x)               # 'expert_distinct_pair' | 'rubric_inquiry'
    correct = (y_hat == gold) or (rubric_score(x) >= tau_rubric)   # tau_rubric swept in [0.4,0.8]
    append to Z: (p_top, H, a_resid, m, correct)

# Phase 1a: validate monotonicity of a_resid (before fitting isotonic)
for m in {expert_distinct_pair, rubric_inquiry}:
    bin held-out half of Z_m by a_resid into 5 equal-frequency bins
    compute empirical error rate per bin with 95% bootstrap CI (1000 resamples)
    if error rate is not monotone increasing in a_resid (Spearman rho < 0.3 or CI overlap):
        mark specialist feature as non-informative for mode m
        fall back to mode-conditioned Platt map on p_top alone

# Phase 1b: fit per-mode isotonic calibration (no student gradients)
for m in {expert_distinct_pair, rubric_inquiry}:
    Z_m = {(p_top, H, a_resid) : correct} subset of Z with tag m
    s_m = logistic_regression(Z_m)              # 3 features, L2=1e-3
    iso_m = isotonic_regression(s_m -> correct) # monotone map
    certify threshold theta_m = argmax_t (TPR - FPR) on held-out half of Z_m

# Phase 2: inference-time certification
c(x) = iso_{m(x)}( s_{m(x)}( p_top, H, a_resid ) )
abstain(x) = 1[ c(x) < theta_{m(x)} ]
return (y_hat, c(x), abstain(x))
```

**Why this design.** We chose a per-failure-mode isotonic map over a single global temperature-scaling head because MediConfusion and rubric-scored inquiry have provably different error geometries (binary adversarial discrimination versus graded rubric), so a shared monotone map would average two non-commensurable correctness definitions, accepting the cost that FMC needs at least ~200 probe items per mode. We chose the decorrelated specialist disagreement a_resid as a feature over the student's own log-prob because KMed-31B's MOPD^2 teachers were trained on disjoint feedback regimes and their disagreement, after regressing out the student's own p_top and a shared-base reference model's answer, is a direct readout of the category boundary the student is straddling rather than shared pretraining bias, accepting the cost that FMC is tied to the four-specialist decomposition and cannot be ported to a single-teacher distillation. We chose isotonic regression over a learned neural calibration head because isotonic regression's monotonicity is a hard constraint rather than a soft penalty, accepting the cost of coarser resolution in sparse confidence bins. This is not a variant of temperature scaling (Guo et al.) or of Kadavath-style self-evaluation: temperature scaling has one global parameter and no failure-mode conditioning, and self-evaluation prompts the model for a verbalized probability without any specialist-agreement evidence; FMC's certified quantity is conditioned on a probe-classified failure mode and on cross-teacher disagreement, neither of which those methods consume.

**Why it measures what we claim.** The computational quantity `a_resid`, the residual of the fraction of MOPD^2 specialists that agree with the student's top-1 answer after regressing out the student's own p_top and a shared-base reference model's answer, measures *failure-mode proximity* because we assume specialist disagreement is monotone in the probability that the item lies on the category boundary where the student's error rate is elevated, and that this monotonicity survives the decorrelation step; this assumption fails when specialists agree for correlated reasons (shared pretraining contamination on the same adversarial pair), in which case `a_resid` reflects shared bias rather than boundary proximity, and we detect this failure by the Phase 1a monotonicity check (Spearman rho < 0.3 or overlapping bootstrap CIs) and fall back to a mode-conditioned Platt map on p_top alone. The computational quantity `iso_m(s_m(...))` measures *certified confidence* because we assume the held-out half of Z_m is exchangeable with deployment items of the same failure-mode tag; this assumption fails under distribution shift in the probe set's difficulty mix, in which case the isotonic map reflects the probe set's base rate rather than deployment accuracy. The computational quantity `theta_m` measures *the abstention boundary that maximizes TPR−FPR* because we assume the cost of a false confident answer equals the cost of a false abstention within a failure mode; this assumption fails in safety-critical triage where false confident answers dominate, in which case theta_m should be lowered by the deployment-specific cost ratio rather than the symmetric criterion used here.

## Contribution

(1) FMC, a failure-mode-conditioned post-hoc calibration module that consumes MOPD^2 specialist agreement as a calibration feature and produces per-mode certified confidence with an explicit abstention threshold. (2) A design principle: specialist disagreement from a multi-teacher distillation is a usable calibration signal precisely because the teachers were trained on disjoint feedback regimes, which no single-teacher or temperature-scaling baseline can exploit. (3) A probe protocol that pairs MediConfusion-style adversarial image pairs with rubric-scored Synthetic Health Dialogue items so that calibration is audited on two structurally different correctness definitions.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | MediConfusion pairs + Synthetic Health Dialogue rubric items | Two failure modes with distinct correctness geometry |
| Primary metric | Selective accuracy at fixed abstention rate | Isolates calibration gain from raw accuracy |
| Baseline | Global temperature scaling | Tests whether failure-mode conditioning matters |
| Baseline | Kadavath-style self-evaluation | Tests whether specialist agreement adds signal |
| Baseline | Raw top-1 confidence threshold | Tests whether calibration map adds beyond raw confidence |
| Baseline | Mode-conditioned Platt map on p_top alone | Tests whether decorrelated specialist feature adds beyond p_top |
| Ablation-of-ours | FMC with single shared isotonic map | Tests necessity of per-failure-mode maps |
| Ablation-of-ours | FMC with mean(a) instead of a_resid | Tests necessity of decorrelation step |
| Ablation-of-ours | FMC with corrupted failure-mode tag (5%, 10%, 20%, 30% label noise) | Quantifies probe-classifier error impact |
| Ablation-of-ours | FMC with tau_rubric swept in {0.4, 0.5, 0.6, 0.7, 0.8} | Tests sensitivity to rubric binarization |

### Why this setup validates the claim

The central claim is that FMC's certified confidence is better than existing calibration because it conditions on a probe-classified failure mode and on cross-teacher disagreement that is decorrelated from shared bias. The dataset is split by failure mode precisely so the metric can be read per-mode: if FMC's gain is real, it must show up where the two modes have non-commensurable correctness definitions. Global temperature scaling tests the sub-claim that a single monotone map is insufficient; if temperature scaling matches FMC, the per-mode conditioning is doing no work. Kadavath-style self-evaluation tests the sub-claim that specialist agreement carries information the student's own verbalized probability does not; if it matches FMC, the specialist feature is redundant. Raw top-1 threshold tests whether the isotonic map adds anything beyond the student's raw confidence. The mode-conditioned Platt map on p_top alone is the tightest test of the decorrelation step: it uses the same per-mode conditioning and fitting procedure but drops a_resid, so any gap between it and FMC isolates the value of the decorrelated specialist feature from the value of failure-mode conditioning. The shared-isotonic ablation uses identical features and fitting procedure but collapses the two modes, so any gap between it and FMC isolates the value of failure-mode conditioning from the value of the features themselves. The mean(a)-instead-of-a_resid ablation directly tests whether decorrelation is necessary for monotonicity. The tag-noise and tau_rubric sweeps quantify robustness to probe-classifier error and rubric binarization. Selective accuracy at a fixed abstention rate is the right metric because it forces every method to abstain at the same rate, so differences cannot be explained by one method simply abstaining more.

### Expected outcome and causal chain

**vs. Global temperature scaling** — On an expert-distinct adversarial pair where the student is confidently wrong because the two images are visually similar to it but clearly distinct to an expert, temperature scaling produces a high confidence because its single global parameter cannot distinguish this error geometry from ordinary rubric-scored inquiry errors, where the student's confidence is already reasonably ordered. Our method instead routes the item through the expert-distinct isotonic map, which was fit on exactly this pair type, and the decorrelated specialist disagreement feature a_resid is low because the specialist trained on the relevant modality disagrees and that disagreement is not explained by the student's p_top or the shared-base model, so we expect a noticeable gap on the adversarial-pair subset but parity on the rubric-inquiry subset.

**vs. Kadavath-style self-evaluation** — On a rubric-scored inquiry item where the student's answer is plausible but misses a key clinical caveat, self-evaluation produces an overconfident verbalized probability because the student is judging its own output with the same knowledge that produced the omission, and it has no access to external disagreement. Our method instead consumes a_resid: the specialist trained on clinical inquiry rubrics disagrees with the student's top-1, and this disagreement remains after regressing out p_top and the shared-base answer, so the agreement feature is low and the rubric-inquiry isotonic map, fit on rubric-scored items, lowers the certified confidence below theta. We expect a clear separation on rubric-inquiry items where specialist disagreement is present, with little change on items where all specialists agree.

**vs. Raw top-1 confidence threshold** — On a case where the student's top-1 probability is high but wrong because the item sits on the category boundary between two specialist regimes, the raw threshold produces a confident wrong answer because raw softmax confidence is not monotone in correctness across failure modes. Our method instead combines p_top with H and a_resid through the per-mode logistic map before isotonic calibration, so the boundary item's low decorrelated specialist agreement pulls the score down even when p_top is high. We expect the largest improvement in the low-abstention regime, where raw confidence is most misleading, and diminishing returns at high abstention rates.

**vs. Mode-conditioned Platt map on p_top alone** — On an expert-distinct adversarial pair where the student's p_top is high but wrong, the Platt map on p_top alone cannot lower confidence because p_top is the only input and it is high. Our method instead uses a_resid, which is low because the relevant specialist disagrees and this disagreement is not explained by p_top or the shared-base model, so the per-mode logistic map produces a lower score before isotonic calibration. We expect FMC to outperform the Platt map on the adversarial-pair subset, with the gap largest where p_top is high but a_resid is low, and parity on items where a_resid is near zero.

### What would falsify this idea

If FMC's gain over global temperature scaling is uniform across the adversarial-pair and rubric-inquiry subsets rather than concentrated on the subset whose failure mode the probe classifier tags, then the per-failure-mode conditioning is not doing the work we claim and the central mechanism is wrong. Similarly, if the shared-isotonic ablation matches FMC, the two correctness geometries are not actually non-commensurable and the design rationale collapses. If the Phase 1a monotonicity check fails (Spearman rho < 0.3 or overlapping bootstrap CIs) for a_resid and FMC's performance drops to that of the mode-conditioned Platt map on p_top alone, then the decorrelated specialist feature is non-informative and the load-bearing assumption is false.

## References

1. KMed-31B Technical Report
