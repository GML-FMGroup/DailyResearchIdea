# Web-Paraphrase Revalidation Audit: Per-Suite Contamination Resistance for Static Safety Item Banks

## Motivation

Frontier safety suites inherit static item-bank designs like GPQA's without re-validating web-access resistance at suite level, so a low loss-of-control or evaluation-integrity score cannot be attributed to capability rather than prior exposure to the exact items. The DeepSeek V4.1 Flash-1017 evaluation (34 benchmarks, 5 in-house suites, AEF-1 standard) reports safety improvements but inherits this unverified property, and the evaluation-awareness work of Phuong et al. shows models may also recognize and adapt to original-item formats, compounding the confound. The structural failure is that contamination resistance is treated as a one-time construction check on the source benchmark rather than a per-suite, per-item property that must be re-established whenever a new suite imports a static bank.

## Key Insight

A web-accessible paraphrase of a static item is a valid intervention on prior exposure only if the paraphrase preserves the item's answer-determining content while destroying surface-string retrievability, and this can be enforced per item by a two-sided check (answer-preservation and retrieval-miss), turning the accuracy gap into a per-item exposure indicator rather than a suite-level guess.

## Method

**What it is.** WRAITH (Web-Resistance Audit via Intervention-Tested Hardening) is an audit protocol that takes a static safety item bank and, per item, produces a web-accessible paraphrase whose answer-preservation and retrieval-miss are both verified, then computes a per-item accuracy gap between original and paraphrase and aggregates it into a per-suite contamination-resistance certificate with a pre-registered threshold. The load-bearing assumption is stated explicitly: a paraphrase that passes the answer-preservation and retrieval-miss filters is assumed to be difficulty-equivalent to the original, so that any systematic gap is attributable to training exposure rather than to format-induced capability differences. This assumption is not taken on faith; it is calibrated per style using a format-difficulty control set (Phase 0), and the exposure indicator is defined as the residual gap after subtracting the style-specific control offset.

**How it works.**

```pseudocode
INPUT: suite S = {(q_i, a_i, rubric_i)}_{i=1..N}, model M, threshold tau (pre-registered, e.g., 0.10),
       web index W (fixed open retriever: BM25 over a timestamped Common Crawl snapshot, e.g., CC-2024-10),
       verifier V (held-out reference model, disjoint from M), retrieval probe R (semantic retriever: E5-large),
       control set C = {(q_j, a_j, rubric_j)}_{j=1..K} of known-clean items (K=200, from a held-out clean suite with no web presence),
       style set ST = {narrative, tabular, multi-turn, cloze, applied}

PHASE 0 - Format-difficulty calibration (per style)
  for each style s in ST:
    for each control item j in C:
      p_js = LLM_paraphrase(q_j, style=s)  # same generator as Phase 1
      if V(p_js) == a_j and rubric_equivalence(V, p_js, q_j, rubric_j) >= 0.9:
        if max_{d in W} sim(p_js, d) < 0.85:  # retrieval-miss
          y_orig = M(q_j) under rubric_j ; y_para = M(p_js) under rubric_j
          gap_js = 1[y_orig correct] - 1[y_para correct]
  offset_s = mean_j gap_js over control items with valid paraphrase  # per-style format-difficulty offset
  sigma_s = std_j gap_js  # per-style control-set gap standard deviation

PHASE 1 - Paraphrase generation (per item, k=5 candidates)
  for each item i:
    cands = [LLM_paraphrase(q_i, style=s) for s in ST]
    keep = []
    for c in cands:
      if V(c) == a_i and rubric_equivalence(V, c, q_i, rubric_i) >= 0.9:   # answer-preservation
        if max_{d in W} sim(c, d) < 0.85:                                  # retrieval-miss (surface)
          if max_{d in W} sim_semantic(c, d) < 0.85:                       # retrieval-miss (semantic, via R)
            keep.append(c)
    if keep empty: mark i as NON-PARAPHRASABLE (excluded from gap, flagged)
    else: p_i = argmax_{c in keep} sim(c, q_i) subject to both checks  # closest preserved paraphrase
          s_i = style of p_i

PHASE 2 - Paired evaluation
  for each item i with paraphrase p_i:
    y_orig = M(q_i) under rubric_i ; y_para = M(p_i) under rubric_i
    gap_i = 1[y_orig correct] - 1[y_para correct]
    residual_i = gap_i - offset_{s_i}  # exposure indicator after format-difficulty control

PHASE 3 - Suite certificate
  G = mean_i residual_i over paraphrased items
  bootstrap 95% CI over items (1000 resamples)
  exposure_set E = {i : residual_i > sigma_{s_i} and y_para = 0}   # original-correct, paraphrase-wrong, beyond style control
  CERTIFY suite as CONTAMINATION-RESISTANT iff upper CI bound of G < tau AND |E|/N < tau
  report per-item E for manual review
```

Hyperparameters: k=5 paraphrase styles; answer-preservation threshold 0.9 rubric-equivalence; retrieval-miss similarity cutoff 0.85 against web index W (both surface BM25 and semantic E5-large); pre-registered gap threshold tau=0.10; 1000 bootstrap resamples; control set size K=200 known-clean items; per-style offset offset_s and control-set standard deviation sigma_s estimated from Phase 0.

**Why this design.** We chose a two-sided per-item filter (answer-preservation via a held-out verifier V plus retrieval-miss against a web index W) over a single paraphrase-then-evaluate pipeline because a paraphrase that changes the answer or that is itself retrievable would confound the gap with capability loss or with a second contamination channel, accepting the cost that some items become NON-PARAPHRASABLE and drop out of the estimate. We chose a paired within-item design (same item, original vs. paraphrase) over a between-suite comparison because item difficulty is the dominant variance source and pairing removes it, accepting a smaller effective sample size. We chose a pre-registered threshold with a bootstrap CI over items over a post-hoc significance test because the audit's purpose is a per-suite certificate that is fixed before seeing M's results, accepting that some suites will be certified inconclusive rather than passing. We added a format-difficulty control (Phase 0) because answer-preservation alone does not guarantee difficulty-equivalence; the per-style offset estimated on known-clean items subtracts the systematic format effect, so the residual gap is the exposure indicator. This is not a variant of GPQA's one-time Google-proofness construction check nor of standard benchmark decontamination (n-gram overlap filtering), because those operate on the source corpus at construction time and never produce a per-item intervention with verified answer-preservation; the closest named prior, GPQA, validates once and never re-validates the imported bank.

**Why it measures what we claim.** The per-item residual gap residual_i measures prior exposure rather than capability because the paraphrase p_i is constructed to be answer-equivalent under V, surface- and semantic-dissimilar from any web document, and its style-specific format-difficulty offset offset_{s_i} is subtracted using a known-clean control set; this assumption fails when the control set does not represent the item's format sensitivity (e.g., a cloze style that adds inference load beyond the control-set average), in which case residual_i reflects a residual capability differential between formats instead of exposure. The answer-preservation check measures paraphrase validity because rubric-equivalence under a held-out verifier V disjoint from M is assumed to track the same answer-determining content; this assumption fails when V shares M's blind spots, in which case a corrupted paraphrase passes and residual_i is noise. The retrieval-miss check measures exposure-channel closure because max cosine similarity below 0.85 against W under both surface (BM25) and semantic (E5-large) retrievers is assumed to imply the paraphrase is not retrievable verbatim or by meaning; this assumption fails for semantic retrieval engines that match meaning beyond the probe's recall, in which case the paraphrase is still web-accessible and residual_i under-estimates exposure. The suite certificate G measures per-suite contamination resistance because the pre-registered threshold tau is assumed to separate exposure-driven residuals from format-driven ones after the control-set subtraction; this assumption fails for suites with inherently format-sensitive items whose sensitivity exceeds the control-set spread, in which case the certificate reflects format sensitivity rather than contamination.

## Contribution

(1) WRAITH, a per-suite dynamic contamination-resistance audit protocol that converts a static item bank into paired original/paraphrase interventions with verified answer-preservation and retrieval-miss, yielding a pre-registered per-suite certificate and a per-item exposure set. (2) The design principle that contamination resistance must be re-established per suite via a two-sided intervention filter rather than inherited from a source benchmark's one-time construction check, with NON-PARAPHRASABLE items reported as a first-class audit outcome. (3) A reusable audit harness that emits per-item exposure sets for manual review, applicable to any static safety item bank without access to model training data.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | Held-out public safety suites with known web presence (e.g., a suite whose items appear in timestamped Common Crawl snapshots) plus a held-out clean suite with no web presence (K=200 items) for format-difficulty calibration | Provides items with real exposure signal and a control set for per-style offsets |
| Primary metric | Per-suite mean residual gap G (after subtracting style-specific control offset) with bootstrap upper bound vs tau | Directly tests the certificate claim while controlling for format difficulty |
| Baseline | GPQA-style one-time Google-proofness check | Tests whether static construction-time filtering suffices |
| Baseline | n-gram decontamination filtering | Tests whether surface overlap removal suffices |
| Baseline | Unfiltered paraphrase-then-evaluate (no answer-preservation / retrieval-miss filters, no format-difficulty control) | Tests whether the two-sided filter and control are what carry the signal |
| Ablation-of-ours | WRAITH minus retrieval-miss check | Tests whether exposure-channel closure matters |
| Ablation-of-ours | WRAITH minus format-difficulty control (i.e., original gap_i without subtracting offset_s) | Tests whether the load-bearing assumption repair is necessary |
| Calibration metric | Verifier answer-preservation false-accept rate and retrieval-miss false-miss rate against ground-truth exposure labels on a stratified sample of 100 items (50 known-contaminated, 50 known-clean) | Empirically justifies the 0.9 and 0.85 thresholds |

### Why this setup validates the claim

The central claim is that a per-item, intervention-tested paraphrase with verified answer-preservation and retrieval-miss, after subtracting a style-specific format-difficulty offset estimated on known-clean items, isolates prior exposure from capability, and that aggregating the resulting paired residual into a pre-registered certificate distinguishes contamination-resistant suites from contaminated ones. The dataset must contain suites where web exposure genuinely varies, so that a real residual signal exists to detect; the held-out clean suite serves as the parity control and as the source of per-style offsets. The two static baselines (GPQA-style one-time proof, n-gram filtering) each test a sub-claim: that construction-time checks are insufficient because they never re-validate the imported bank and never produce a verified per-item intervention, so they should fail to flag suites that the paired residual catches. The unfiltered paraphrase baseline tests whether the two-sided filter and format-difficulty control are load-bearing: without them, residuals should be inflated by capability loss and second-channel retrievability, producing false certificates. The retrieval-miss ablation isolates the exposure-channel-closure sub-claim, and the format-difficulty-control ablation isolates the load-bearing assumption repair. The calibration experiment directly measures each gate's validity: by injecting known-contaminated and known-clean items with ground-truth exposure labels (from timestamped Common Crawl snapshots and manual annotation), we report the verifier's answer-preservation false-accept rate and the retrieval-miss false-miss rate against a semantic retriever, so the 0.9 and 0.85 thresholds are empirically justified rather than asserted. The primary metric is the right one because it is exactly the quantity the certificate is defined on, and the pre-registered tau plus bootstrap upper bound makes the pass/fail decision falsifiable rather than post-hoc.

### Expected outcome and causal chain

**vs. GPQA-style one-time Google-proofness check** — On a suite whose items were Google-proof at construction but have since been paraphrased into web text (e.g., forum restatements, tutorial rewrites), the baseline certifies the suite as clean because its check ran once and never re-validates the imported bank. Our method instead re-derives a fresh verified paraphrase per item and measures the paired residual after subtracting the style-specific control offset, so we expect the baseline to pass suites where WRAITH reports a concentrated exposure-set E and a residual well above tau, with the divergence appearing specifically on suites with post-construction web drift.

**vs. n-gram decontamination filtering** — On an item whose original string is absent from the web index but whose meaning is retrievable through a semantic restatement, the baseline sees zero n-gram overlap and admits the item, so its certificate reflects surface absence rather than exposure. Our method instead requires retrieval-miss against W under both surface (BM25) and semantic (E5-large) probes and flags items whose paraphrase remains semantically accessible, so we expect n-gram filtering to certify suites that WRAITH marks inconclusive or contaminated, with the disagreement concentrated on semantic-retrieval-heavy items.

**vs. Unfiltered paraphrase-then-evaluate** — On a cloze-style or multi-turn paraphrase that adds inference load or is itself retrievable, the unfiltered pipeline records a large gap and attributes it to contamination, because it has no answer-preservation or retrieval-miss gate and no format-difficulty control. Our method instead drops or re-selects such candidates via the verifier V and the similarity cutoffs, and subtracts the style-specific offset from the known-clean control set, so we expect the unfiltered baseline to show uniformly inflated gaps and many false exposure-set entries, while WRAITH shows parity on format-sensitive suites and residuals only where exposure is real.

**vs. WRAITH minus retrieval-miss check** — Without the retrieval-miss gate, paraphrases that remain web-accessible (surface or semantic) are admitted, so the residual gap is attenuated toward zero because both original and paraphrase conditions are exposed. We expect this ablation to under-estimate exposure on suites with high web presence, producing false certificates of resistance, with the divergence largest on semantic-retrieval-heavy items.

**vs. WRAITH minus format-difficulty control** — Without subtracting the style-specific offset, the residual gap includes the format-difficulty differential, so suites with inherently format-sensitive items show inflated gaps and false contamination flags. We expect this ablation to produce false positives on the clean control suite and on format-sensitive safety suites, while WRAITH shows parity on the clean suite and residuals only where exposure is real.

### What would falsify this idea

If WRAITH's residual gap is uniform across suites regardless of known web exposure — i.e., no concentration of the exposure-set E on suites with documented web presence and parity on the held-out clean suite — then the paired paraphrase with format-difficulty control is not isolating exposure and the certificate claim is wrong. Likewise, if the retrieval-miss ablation or the format-difficulty-control ablation changes nothing, the corresponding sub-claim fails.

## References

1. Evaluating DeepSeek V4.1 Flash-1017 for loss of control and harmful manipulation risks
2. Large Language Models Often Know When They Are Being Evaluated
3. Frontier Models are Capable of In-context Scheming
