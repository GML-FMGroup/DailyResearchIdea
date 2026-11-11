# Refresh-Invariance Certificates for Multitask Decision Benchmarks

## Motivation

MMDM and every single-pass benchmark of its lineage score each decision item with one forward pass, so the reported score is structurally a sum of independent single-pass predicates; the benchmark's own fresh-context and format-diversification machinery is used only at construction time and never as a test-time intervention. The consequence is that a model's decision policy is never modeled as an object whose invariance must be certified: a model that answers correctly only because of a surface format cue, a position artifact, or a context-specific shortcut receives full credit indistinguishable from a model with a stable policy. Because no ancestor treats item credit as conditional on multi-pass agreement, compositional decision-consistency under context refresh is invisible to the score by construction, not by oversight.

## Key Insight

A decision item's credit is only meaningful if the model's chosen action sequence is a fixed point of the benchmark's own refresh-and-reformat operators, so invariance across a paired variant family is the certificate that separates a stable policy from a single-pass coincidence.

## Method

**What it is.** We introduce RIC (Refresh-Invariance Certificate), a scoring kernel that replaces the single-pass item predicate of MMDM-style benchmarks. Its input is a decision item plus the benchmark's own refresh operators (context recomposition/rewrite) and format operators (single-select, multi-select, soft-label, option-order permutation); its output is a per-item certified credit in [0,1] and a per-item invariance vector, aggregated into a benchmark score.

**Load-bearing assumption (stated explicitly).** RIC assumes that the benchmark's refresh and format operators are *gold-preserving and difficulty-neutral* interventions: applying them leaves the gold action set unchanged and does not systematically alter task difficulty or introduce ambiguity the gold set cannot adjudicate. Under this assumption, any change in the model's canonicalized action set across the paired variant family is attributable to policy instability rather than to operator-induced label drift or difficulty shift. This assumption is not taken on faith: Phase 0 below calibrates each operator on a reference set and either filters or normalizes the agreement statistic so that `inv` measures excess instability over operator noise.

**How it works.**

```pseudocode
INPUT: item q with option set O, model M, K=4 paired variants, temperature 0
operators: R = {refresh_rewrite, refresh_recompose}   # benchmark-native
           F = {select_mode, option_order_perm}       # benchmark-native
reference set: S_ref = 512 items with human-verified gold, disjoint from eval items
reference model: M_ref = strongest available model (e.g., GPT-4-class) or human panel

PHASE 0 - operator calibration (run once per benchmark, before scoring)
  for each operator o in R ∪ F:
      # measure the operator's difficulty-shifting effect on the reference set
      acc_base = mean_{q in S_ref} 1[ M_ref(q) == A*(q) ]
      acc_o    = mean_{q in S_ref} 1[ M_ref(o(q)) == A*(q) ]
      delta_o  = acc_base - acc_o          # positive = operator makes items harder
      # null-policy agreement: expected agreement if M had no policy at all
      # estimated by permuting the reference model's action sets across items
      null_agree_o = mean_{q in S_ref} 1[ M_ref(o(q)) == M_ref(q') ]  for random q' != q
  # operator filter: keep only operators with |delta_o| <= tau_delta = 0.05
  R_kept = { o in R : |delta_o| <= tau_delta }
  F_kept = { o in F : |delta_o| <= tau_delta }
  # if fewer than 2 operators survive, fall back to all operators with normalization
  if |R_kept| < 1 or |F_kept| < 1:
      R_kept, F_kept = R, F   # use normalization instead of filtering

PHASE 1 - variant family construction
  base = q
  family = []
  for r in R_kept, f in F_kept:        # up to 2 x 2 = 4 cells, one variant each
      family.append( apply(f, apply(r, base)) )
  # each variant keeps the gold action set A* unchanged by construction
  assert gold(family[i]) == gold(base) for all i

PHASE 2 - action-sequence extraction
  for v in family + [base]:            # up to 5 passes, temperature 0
      a_v = canonicalize( M(v) )       # map to action set over O

PHASE 3 - invariance certificate (operator-calibrated)
  for i, v in enumerate(family):
      raw_agree[i] = 1[ a_v == a_base ]
      # soft credit for soft-label items: cosine on the probability simplex
      if item is soft-label:
          raw_agree[i] = max(0, cos( p_v, p_base ))
      # normalize by the operator's null-policy agreement
      # (excess agreement over what a policy-free model would show)
      agree[i] = (raw_agree[i] - null_agree_{o_i}) / (1 - null_agree_{o_i} + 1e-8)
      agree[i] = clip(agree[i], 0, 1)
  inv = mean_i agree[i]                # in [0,1]

PHASE 4 - certified credit
  correct = 1[ a_base == A* ]
  credit  = correct * inv               # certificate gates credit
  return credit, inv, {agree[i]}
```

Aggregation over items is the mean certified credit; the kernel additionally emits the marginal invariance distribution so that a high-accuracy/low-invariance model is separable from a high-accuracy/high-invariance one. The calibration constants are fixed: `S_ref` = 512 human-verified items, `tau_delta` = 0.05, `null_agree` estimated by 1000 random permutations of the reference model's action sets across items.

**Why this design.** We chose to reuse the benchmark's own refresh and format operators rather than authoring new paraphrases, because MMDM's operators are the only interventions whose gold-label preservation is already audited, accepting the cost that the variant family is bounded by whatever operators the benchmark exposes. We chose a multiplicative gate (correct x inv) over an additive penalty because a correct-but-non-invariant item is a single-pass coincidence and should not partially count, accepting the cost that near-invariant models receive a steep score reduction that an additive term would soften. We chose temperature-0 canonicalized action sets over sampling multiple stochastic passes because we want invariance to reflect sensitivity to context/format, not decoding noise, accepting the cost that the kernel cannot detect a policy that is stochastically unstable at nonzero temperature. We chose K=4 paired cells (2 refresh x 2 format) over a larger grid because the certificate's variance falls as 1/K while cost is linear in K, and 4 cells already expose the dominant refresh-vs-format failure axes. This is not a variant of MMDM's scoring: MMDM scores one forward pass per item and uses refresh only at construction, whereas RIC makes the model's action sequence the object under test and requires it to be a fixed point of the benchmark's operators; a domain expert would not describe RIC as MMDM with extra paraphrases, because the unit of credit changes from a predicate on one output to a certificate over a paired family.

**Why it measures what we claim.** The quantity `inv` (mean pairwise agreement between base and refreshed/reformatted variants, normalized by the operator's null-policy agreement) measures decision-policy stability because the assumption that the benchmark's refresh and format operators are gold-preserving and difficulty-neutral interventions makes any action change under them attributable to the model rather than to label drift; this assumption fails when a refresh operator accidentally alters task difficulty or introduces an ambiguity the gold set does not resolve, in which case `inv` reflects operator-induced label noise rather than policy instability — which is why Phase 0 filters operators with |delta_o| > 0.05 and normalizes by null-policy agreement. The quantity `correct` measures task competence because the assumption that the gold action set A* is the unique acceptable action under the base context makes equality with A* equivalent to a correct decision; this assumption fails for items with multiple defensible actions outside A*, in which case `correct` reflects gold-set narrowness. The product `credit` measures certified decision consistency because the assumption that a stable policy must produce identical actions across gold-preserving interventions makes the multiplicative gate equivalent to 'correct for a reason that survives context refresh'; this assumption fails when a model is stably wrong for a coherent reason, in which case `credit` reflects stable error rather than competence, which is why the kernel reports `inv` and `correct` separately alongside the product. A further failure mode is that a model exploits an operator-invariant but non-gold cue (e.g., answer-length or option-position artifact preserved across all cells), so `inv=1` despite a fragile policy; this is why the experiment includes a diagnostic that perturbs an unmodeled axis (option-order permutation beyond the 2x2 grid) to test whether `inv` over-certifies.

## Contribution

(1) A paired context-refresh invariance scoring kernel that converts a single-pass terminal-answer predicate into a multi-pass fixed-point certificate over the benchmark's own refresh and format operators, with a multiplicative credit gate and a per-item invariance vector. (2) The design principle that benchmark-native operators, whose gold preservation is already audited, are the correct intervention set for invariance certification, and the empirical separation this induces between high-accuracy/low-invariance and high-accuracy/high-invariance decision models. (3) A diagnostic decomposition of item-level credit into correctness, invariance, and per-operator sensitivity, exposing which of context refresh versus format diversification drives a model's instability.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | MMDM 16 categories (single/multi/soft-label), N=2000 items | Native refresh and format operators already audited |
| Reference set | 512 human-verified items, disjoint from eval | Operator calibration (Phase 0) |
| Primary metric | Mean certified credit (RIC) | Gates correctness by invariance, not raw accuracy |
| Baseline | MMDM single-pass predicate | Standard scoring RIC claims to supersede |
| Baseline | Accuracy + paraphrase-consistency (additive) | Tests whether multiplicative gate beats additive penalty |
| Baseline | Self-consistency at temperature 0.7 | Tests invariance-to-decoding-noise vs invariance-to-intervention |
| Ablation-of-ours | RIC with inv fixed to 1 | Isolates the certificate gate's contribution |
| Ablation-of-ours | RIC without Phase 0 calibration (raw agreement) | Tests whether operator calibration changes the ranking |
| Diagnostic | Unmodeled-axis perturbation (option-order permutation beyond 2x2) | Tests whether inv over-certifies operator-invariant shortcuts |

Compute budget: 5 passes x 2000 items x 4 models = 40,000 forward passes; at ~0.5s per pass on a single A100, this is ~5.5 GPU-hours per model, ~22 GPU-hours total. Power analysis: with N=2000 and expected effect size d=0.3 on the refresh-sensitive subset (estimated from pilot on 200 items), power > 0.9 at alpha=0.05.

### Why this setup validates the claim

The design sets up a three-way falsifiable contrast on the same item pool. The MMDM single-pass baseline tests the sub-claim that one forward pass cannot distinguish competence from coincidence, since it has no mechanism to penalize correct-but-non-invariant items. The additive-penalty baseline tests the sub-claim that the multiplicative gate is the right functional form, since it shares RIC's variant family but softens the reduction for near-invariant models. The self-consistency baseline tests the sub-claim that invariance should be measured against benchmark-native interventions rather than decoding noise, since it perturbs the decoder instead of the context. The ablation (inv=1) is the critical control: it reduces RIC to plain base-context accuracy, so any gap between full RIC and the ablation is attributable specifically to the certificate, not to the variant family or the canonicalization. The calibration ablation (raw agreement vs Phase-0-normalized) tests whether the load-bearing assumption of operator difficulty-neutrality is doing work: if the ranking is unchanged, calibration is unnecessary; if it changes, the raw statistic was confounded. The unmodeled-axis diagnostic tests the grid-sufficiency assumption: if a model scores high on inv but its action set changes under an option-order permutation outside the 2x2 grid, then inv over-certifies. Certified credit is the right primary metric because it is the only quantity that jointly encodes the two failure axes (wrong decision, unstable decision) in a single number, so a model that is accurate but fragile is separable from one that is accurate and stable. If the central claim is right, the ranking of models under certified credit should diverge from ranking under MMDM accuracy in a structured way, concentrated on the refresh-vs-format failure axis.

### Expected outcome and causal chain

**vs. MMDM single-pass predicate** — On a multi-select item where the base context yields the correct action set but a rewrite introduces a distractor that the model latches onto, the MMDM baseline records a full correct credit because it never re-queries under the rewrite. Our method instead extracts action sets across the paired family and observes disagreement on the rewrite cell, so `inv` drops and the multiplicative gate zeroes the credit. We expect a visible gap on refresh-sensitive items (rewrite/recompose cells) with near-parity on items whose variants the model handles identically, i.e., the gap concentrates where refresh actually changes the model's action set.

**vs. Accuracy + additive paraphrase-consistency** — On a soft-label item where the model's probability mass shifts modestly but consistently across all four cells (a coherent, stable-but-different distribution), the additive baseline subtracts a small penalty and still awards most of the credit, because additive terms let a correct-but-drifting item partially count. Our method instead uses cosine agreement on the simplex and the multiplicative gate, so a stable shift that is not a fixed point of the operators reduces credit more sharply. We expect the additive baseline to sit between MMDM and RIC on the soft-label subset, with the largest divergence on items whose drift is uniform rather than cell-specific.

**vs. Self-consistency at temperature 0.7** — On an item where the model is deterministically wrong under the base context but the intervention family leaves its action set unchanged, the self-consistency baseline scores high because resampling at nonzero temperature mostly reproduces the same wrong answer, mistaking decoding stability for policy stability. Our method instead reports `correct=0` and `inv=1`, yielding credit 0 while surfacing the high-invariance/low-correctness signature in the marginal distribution. We expect self-consistency to over-credit stably-wrong items, so the two methods should agree on genuinely competent items but diverge sharply on the stably-wrong subset.

**vs. RIC with inv fixed to 1 (ablation)** — On the full item pool, the ablation collapses to base-context accuracy and cannot separate a model that is correct-by-coincidence from one that is correct-by-policy, because it discards the per-cell agreement vector entirely. Our method retains that vector and gates credit on it, so the ablation should match full RIC on items where all cells agree and fall above full RIC exactly on the items where cells disagree. We expect the ablation-RIC gap to be small on high-invariance models and large on models whose accuracy is concentrated in single-pass coincidences.

**vs. RIC without Phase 0 calibration (raw agreement)** — On items where an operator has a measured difficulty delta of 0.10 (above tau_delta=0.05), the raw-agreement variant will show lower `inv` for all models simply because the operator makes items harder, not because the model is unstable. The calibrated RIC filters or normalizes that operator, so `inv` reflects excess instability over operator noise. We expect the calibrated RIC to rank models more consistently across operator subsets and to show a smaller gap between high- and low-competence models on the operator-affected items.

**vs. Unmodeled-axis perturbation diagnostic** — On items where a model exploits an option-position artifact preserved across all four cells of the 2x2 grid, `inv` will be 1.0 (all cells agree) but the action set will change under an option-order permutation outside the grid. The diagnostic flags these items, and we expect the fraction of such items to be non-zero (estimated 5-10% from pilot), demonstrating that the 2x2 grid is not a sufficient basis for the shortcut subspace and that `inv` can over-certify.

### What would falsify this idea

If full RIC's gain over the ablation is uniform across refresh-sensitive and refresh-insensitive subsets rather than concentrated where action sets actually change under the operators, the certificate is not measuring policy stability but something else (e.g., a global difficulty shift), and the central claim fails. Likewise, if the additive-penalty baseline matches or beats the multiplicative gate on the soft-label subset, the chosen functional form is wrong. Additionally, if the unmodeled-axis diagnostic shows that a large fraction of items have inv=1 but change under an out-of-grid perturbation, the grid-sufficiency assumption fails and inv over-certifies.

## References

1. MMDM: Measuring Massive Multitask Decision Making
