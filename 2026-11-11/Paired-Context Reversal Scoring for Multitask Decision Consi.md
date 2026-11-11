# Paired-Context Reversal Scoring for Multitask Decision Consistency

## Motivation

Existing multitask decision benchmarks evaluate each decision item exactly once, in a single context format: MMDM scores a model's structured probabilistic decision over options in one forward pass, and LongBench v2 likewise presents each item in one fixed format. Because no paired or counterfactual presentation of the same item exists, neither benchmark can detect whether a model's decision reverses when the same underlying item is embedded in an enriched or reformatted context that should leave the correct decision invariant. This structural one-presentation-per-item design means monotone consistency under belief refinement and format change is simply not measurable by any current scoring kernel.

## Key Insight

If a decision item's correct answer is invariant under belief-preserving enrichment and format change, then any reversal between the original and paired context is attributable to the model's context-sensitivity rather than to the item, making the paired presentation a self-contained consistency probe that requires no external ground-truth label.

## Method

**What it is.** We introduce PCRS (Paired-Context Reversal Scoring), a scoring procedure that takes a decision item with its original context and a set of belief-preserving transforms, produces a paired enriched/reformatted context for each, queries the model on both, and outputs a reversal rate plus a monotone-consistency certificate per item. Inputs: item (context, options, query format), transform set. Outputs: per-item reversal flags, aggregate consistency score, and a per-transform breakdown.

**Load-bearing assumption (stated explicitly).** PCRS assumes that the transform-validity procedure can certify that a transform is belief-preserving — i.e., leaves the correct option unchanged and introduces no new defensible option — so that any argmax flip between the paired presentations is attributable to model context-sensitivity rather than to a changed or newly-ambiguous item. This assumption is not taken on faith: it is operationalized as a measured transform-validity score (defined below) and audited via verifier-model disagreement on a held-out sample, and ReversalRate is reported only on transforms whose validity is established by a human-validated calibration set.

**How it works.**
```pseudocode
Input: item (context C, options O, format F), transform set T = {t1..tk}
Calibration set: 512 items with human-adjudicated invariance labels (invariant / newly-ambiguous / label-changed)

# --- Transform-validity scoring (replaces binary invariant check) ---
For each transform t in T:
    C_t, F_t = apply(t, C, F)          # belief-preserving enrichment/reformat
    v_score_t = validity_score(C, C_t, O)   # independent verifier model: P(correct option remains uniquely defensible), in [0,1]
    ambiguity_flag_t = (second-best option judged defensible by verifier)  # explicit ambiguity audit
    accept_t = (v_score_t >= 0.9) AND (NOT ambiguity_flag_t)
    if not accept_t: skip t for this item; log rejection

# --- Paired scoring on accepted transforms only ---
For each accepted transform t:
    p_orig = model.decide(C,  O,  F)   # structured prob over O
    p_pair = model.decide(C_t, O, F_t)
    a_orig = argmax(p_orig); a_pair = argmax(p_pair)
    flag_t = (a_orig != a_pair)
    margin_t = |p_pair[a_orig] - p_orig[a_orig]|

ReversalRate = mean over items of mean over accepted t of flag_t
MonotoneScore = 1 - ReversalRate
Report per-transform breakdown, margin_t distribution, and ReversalRate conditioned on v_score_t bins (0.9-0.95, 0.95-0.99, 0.99-1.0)
```
Transforms are drawn from a fixed library: (i) redundant premise restatement, (ii) option-order permutation, (iii) format switch (single-select to soft-label), (iv) distractor insertion that does not touch the correct answer. Each transform is scored by the transform-validity procedure above: an independent verifier model outputs a validity score v_score_t in [0,1] (probability that the correct option remains uniquely defensible), and an explicit ambiguity audit flags items where a second option is judged defensible; transforms with v_score_t < 0.9 or an ambiguity flag are rejected. The verifier protocol is pre-registered and piloted on 200 items, reporting verifier rejection rates and inter-verifier agreement (Cohen's kappa) before scaling.

**Verifier-model disagreement audit.** On a held-out sample of 200 accepted transforms, we compare the verifier's invariance judgment against the target model's own behavior: we record cases where the verifier accepts a transform but the target model's argmax flips, and manually adjudicate whether the flip reflects model context-sensitivity or a newly-defensible option. The disagreement rate is reported alongside ReversalRate so the linking assumption between 'validity score passed' and 'correct answer unchanged for the model' is empirically tested rather than assumed.

**Why this design.** We chose a paired same-item design over independent per-item scoring because the reversal signal is only identifiable when the item is held fixed and only the context changes, accepting the cost that we must construct and verify belief-preserving transforms. We chose a fixed transform library with an explicit transform-validity score and ambiguity audit over free-form LLM-generated paraphrases because free paraphrases can silently change the correct answer, accepting the cost of narrower transform coverage in exchange for label-valid reversals. We chose argmax-reversal as the primary flag over probability-difference thresholds because argmax reversal is directly interpretable as a decision flip and does not require calibrating the model's probabilities, accepting the cost that we lose sensitivity to near-ties. We report margin_t separately rather than folding it into the score because it captures confidence drift even when no flip occurs. Unlike MMDM's single-presentation scoring or standard paraphrase-robustness augmentation, which treats each variant as an independent test item, PCRS couples the two presentations as a matched pair whose disagreement is the measured quantity; a domain expert would not describe it as data augmentation because the pair is the unit of analysis and the reversal, not accuracy, is the target.

**Why it measures what we claim.** The argmax-reversal flag measures decision inconsistency under belief refinement because the transform-validity score and ambiguity audit guarantee the correct option is unchanged across the pair, so any flip cannot be attributed to a changed correct answer; this assumption fails when a transform passes the validity score but introduces an unintended ambiguity that makes two options defensible to the model (verifier-model disagreement), in which case the flag reflects item ambiguity rather than model inconsistency — which is why we report the verifier-model disagreement audit and condition ReversalRate on v_score_t bins. The transform-library stratification measures format-sensitivity because each transform isolates one axis (order, format, redundancy, distractors); this assumption fails when transforms interact, in which case the per-transform breakdown misattributes a joint effect to a single axis. The margin_t quantity measures confidence drift because it tracks the probability mass on the original answer across presentations; this assumption fails when the model's probability scale is itself format-dependent, in which case margin_t reflects format-induced rescaling rather than genuine belief change.

## Contribution

(1) A paired-context reversal scoring procedure (PCRS) that certifies monotone decision consistency by comparing the same item under original and belief-preserving enriched contexts, with an explicit transform-invariant check. (2) A design principle that reversal rate decomposes by transform axis, yielding a per-axis sensitivity profile that single-presentation benchmarks cannot produce. (3) A reusable transform library and invariant verifier for constructing label-valid paired decision items.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | MMDM (16 task categories, mixed formats) | Native single/soft-label formats and long contexts stress format-sensitivity |
| Primary metric | ReversalRate (1 − MonotoneScore), reported on accepted transforms only and conditioned on v_score_t bins | Directly counts decision flips under belief-preserving transforms |
| Baseline | MMDM single-presentation scoring | Tests whether paired reversal beats independent per-item accuracy |
| Baseline | Paraphrase-robustness augmentation | Tests whether matched-pair coupling beats independent variant testing |
| Baseline | LLM-generated free-form paraphrase | Tests whether validity-scored library beats unverified paraphrases |
| Ablation-of-ours | PCRS without transform-validity score / ambiguity audit (binary check only) | Isolates contribution of measured validity and audit |
| Ablation-of-ours | PCRS with v_score_t threshold swept (0.7, 0.8, 0.9, 0.95) | Tests sensitivity of ReversalRate to validity threshold |
| Calibration set | 512 MMDM items with human-adjudicated invariance labels (invariant / newly-ambiguous / label-changed) | Provides ground-truth for transform-validity calibration |
| Pilot | 200 items, pre-registered transform library and verifier protocol; report rejection rates and inter-verifier Cohen's kappa | Protects budget against unbounded transform-construction loop |
| Audit | 200 held-out accepted transforms, manual adjudication of verifier-model disagreement | Empirically tests linking assumption between validity score and model belief-preservation |

### Why this setup validates the claim

The central claim is that decision inconsistency under belief-preserving transforms is identifiable only when the item is held fixed as a matched pair and the transform is verified not to change the correct answer. MMDM is the right substrate because it already spans single-select, multi-select, and soft-label formats with long contexts, so the transform library's format-switch and redundancy axes are exercised on realistic items rather than synthetic ones. The 512-item human-adjudicated calibration set grounds the transform-validity score in ground truth rather than verifier judgment alone, and the 200-item pilot with pre-registered protocol and inter-verifier kappa ensures the verifier is reliable before scaling. MMDM single-presentation scoring tests the sub-claim that independent per-item accuracy cannot surface flips: it sees only one presentation and has no disagreement signal. Paraphrase augmentation tests the sub-claim that treating variants as independent test items dilutes the reversal signal into accuracy noise. The free-form paraphrase baseline tests the sub-claim that unverified transforms produce label-invalid flips that inflate ReversalRate spuriously. The no-validity-score ablation isolates whether measured validity and the ambiguity audit are doing real work or merely filtering noise, and the threshold sweep tests whether results are an artifact of the 0.9 cutoff. The verifier-model disagreement audit on 200 held-out transforms directly tests the load-bearing assumption that a passed validity score implies the correct answer is unchanged for the model. ReversalRate is the right primary metric because it is a direct count of the quantity the method claims to measure; accuracy and margin alone cannot distinguish a confident flip from a near-tie, which is why margin is reported separately rather than folded in, and why ReversalRate is conditioned on v_score_t bins to expose any residual ambiguity-driven flips.

### Expected outcome and causal chain

**vs. MMDM single-presentation scoring** — On a case where a model answers a multi-select item correctly under the original format but the soft-label reformat shifts probability mass toward a different option, the baseline reports one accuracy number and never sees the conflict because it scores each presentation independently. Our method pairs the two presentations and flags the argmax disagreement, so we expect a visible ReversalRate concentrated on format-switch and option-order transforms, with near-zero reversal on redundant-restatement transforms where the mechanism predicts stability.

**vs. Paraphrase-robustness augmentation** — On a case where a paraphrase variant is treated as a fresh test item, the baseline averages accuracy across variants and a single flip is lost in the aggregate, because the variant is not coupled to its original and the flip is indistinguishable from ordinary item difficulty. Our method holds the item fixed and makes the pair the unit of analysis, so we expect the reversal signal to be detectable on individual items rather than only as a small aggregate accuracy dip, with per-transform breakdown showing order-permutation flips that augmentation masks.

**vs. LLM-generated free-form paraphrase** — On a case where a generated paraphrase silently makes a second option defensible, the baseline records a flip that reflects changed item semantics rather than model inconsistency, inflating apparent inconsistency. Our method's transform-validity score and ambiguity audit reject such transforms before scoring, so we expect our ReversalRate to be lower than the free-paraphrase baseline specifically on ambiguous source items, while the two agree on clean items where no label drift occurs.

**vs. PCRS without transform-validity score / ambiguity audit** — On a case where a distractor-insertion transform inadvertently overlaps the correct option, the ablation counts a flip that is an artifact of a broken transform. Our method rejects the transform via the validity score and ambiguity audit, so we expect the ablation to show a uniformly higher ReversalRate with no concentration on the predicted format and order axes, while our per-transform breakdown stays localized to the axes the mechanism predicts.

**vs. PCRS with v_score_t threshold swept** — We expect ReversalRate to be stable across thresholds 0.8–0.95 on clean items but to drop sharply at 0.9 on ambiguous items, confirming that the 0.9 cutoff is not arbitrary and that residual flips below 0.9 are ambiguity-driven.

**Verifier-model disagreement audit** — We expect the disagreement rate on the 200 held-out accepted transforms to be low (target <10%) and concentrated on items where the calibration set flagged ambiguity, confirming that the validity score is a reliable proxy for model belief-preservation on the accepted subset.

### What would falsify this idea

If our ReversalRate gain over single-presentation and augmentation baselines is uniform across all transform axes rather than concentrated on format-switch and option-order transforms, then the paired design is not isolating decision inconsistency and the central claim is wrong. Likewise, if the no-validity-score ablation matches our method exactly, or if the verifier-model disagreement audit shows a high disagreement rate on accepted transforms, then the validity score is doing no work and the label-validity argument collapses.

## References

1. MMDM: Measuring Massive Multitask Decision Making
