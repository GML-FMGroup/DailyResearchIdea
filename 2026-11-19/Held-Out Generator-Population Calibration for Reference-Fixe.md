# Held-Out Generator-Population Calibration for Reference-Fixed Video World-Model Benchmarks

## Motivation

Reference-fixed video evaluation benchmarks such as StrideWorld anchor scene rigidity on ground-truth start-view geometry and action consistency on a stride measured from each model's own first steps, but the reference parameters and the population of generators they are validated against are not separated: the same open-model set supplies both the calibration evidence and the judged scores, so anchor identifiability under generator distribution shift is never tested. The same structural conflation recurs across the tree — GRADEO curates its evaluation instruction set from the same video-generation models it later grades, Video-Bench and MJ-Video build prompt/criterion suites alongside the models they rank — so no existing benchmark can distinguish an anchor that is genuinely generator-invariant from one that is merely fitted to the generators it happens to see. The root cause is that reference construction and reference validation share a population, making the anchor's validity conditional on that population rather than on a stated invariance property.

## Key Insight

A reference-fixed anchor is only identifiable as a property of the physical world if its acceptance region is calibrated on a generator population disjoint from the one it judges and then frozen, so that any held-out generator's score is a test of anchor validity rather than a fit to it.

## Method

**What it is.** We introduce HOLD-STRIDE (Held-Out Generator-population Calibration for StrideWorld), a calibration protocol that splits interactive world models into a calibration population and a held-out evaluation population, fits the acceptance regions for StrideWorld's start-view-geometry rigidity residual and per-action stride-consistency residual only on the calibration population, freezes them, and then reports held-out scores plus a reference-identifiability test that asks whether the frozen anchor still separates good from bad motion when the generator population shifts.

**Load-bearing assumption (stated explicitly).** The protocol's validity rests on the assumption that a frozen acceptance region fitted on a disjoint calibration generator population captures a generator-invariant physical property (scene rigidity / stride consistency), so that held-out generators' scores and their Kendall-tau agreement with human ranks are a test of anchor validity rather than a fit to the calibration population's motion repertoire. This assumption is not taken for granted: it is operationalized as a homogeneity criterion on calibration residuals (Phase 3a) and a coverage diagnostic (Phase 4b), and the identifiability verdict is reported only when homogeneity holds; otherwise the protocol returns 'anchor not identifiable on this population'.

**How it works.**

```pseudocode
INPUT: start views S = {s_1..s_N} (140 views as in StrideWorld)
       generator pool G = {g_1..g_M} of open interactive world models
       held-out split rule (generator-level, not view-level)

PHASE 1  Generator-level split
  partition G into G_cal (calibration) and G_hold (held-out), disjoint
  by generator identity; no view from a calibration generator may
  appear in the held-out scoring set
  require |G_cal| >= 8 and |G_hold| >= 8

PHASE 2  Residual extraction on calibration population
  for each g in G_cal, each s in S, each command c:
    r_rig(g,s,c)   = || geom( first frame of g(s,c) ) - GT_geom(s) ||_F
    stride(g,c)    = camera translation magnitude of g's first K steps (K=3)
    r_act(g,s,c)   = max_t | stride(g,c) - step_translation_t | / stride(g,c)
  collect R_cal = { (r_rig, r_act, human_rating) }

PHASE 3a  Calibration homogeneity test (gate for identifiability)
  for each axis a in {rig, act}:
    compute per-generator residual distributions {r_a(g) : g in G_cal}
    run Kruskal-Wallis test across generators (alpha = 0.05)
    if p < 0.05 on either axis:
      OUTPUT 'anchor not identifiable on this population' and STOP
    else proceed to Phase 3b

PHASE 3b  Fit and freeze acceptance regions
  fit per-axis thresholds (tau_rig, tau_act) as the 90th percentile of
  calibration residuals among human-accepted videos (rating >= 4/5)
  fit a 2D acceptance ellipse via robust covariance (MCD, 75% subset)
  FREEZE (tau_rig, tau_act, ellipse) -- no refit on held-out data

PHASE 4  Held-out scoring
  for each g in G_hold: score_rig, score_act, score_joint
  score_joint = fraction of (s,c) pairs inside frozen ellipse

PHASE 4b  Calibration-coverage diagnostic
  for each held-out generator g in G_hold:
    coverage(g) = fraction of g's residuals falling within the
                  [min, max] support of R_cal on each axis
  report coverage(g) alongside score_joint(g)

PHASE 5  Reference-identifiability test
  (a) rank held-out generators by score_joint
  (b) rank by human preference on held-out videos
  (c) report Kendall tau between (a) and (b)
  (d) permutation test: shuffle generator labels 1000x, report null tau
  identifiability holds iff observed tau > 95th percentile of null
  AND Phase 3a homogeneity held

OUTPUT: frozen anchor A*, held-out scores, coverage diagnostics,
        identifiability verdict (or 'not identifiable' if Phase 3a failed)
```

**Why this design.** We chose a generator-level split over a view-level split because views generated by the same model share motion priors and rendering biases, so a view-level split leaks generator identity into calibration and would make the identifiability test vacuous; the cost is that calibration and held-out sets must each contain enough distinct generators (we require at least 8 per side), which restricts the protocol to benchmark suites with broad model coverage. We chose percentile-based acceptance regions over a learned classifier because a classifier trained on calibration residuals would re-absorb generator-specific statistics exactly as StrideWorld's self-fitted references absorb deformation, accepting the cost that the threshold is less sample-efficient than a parametric fit. We chose to freeze the anchor before held-out scoring rather than allow a final refit, because refitting would reproduce the population-conflation problem the protocol is designed to expose, accepting the cost that some held-out generators may fall outside the frozen region for benign reasons (e.g., a new rendering style) and be flagged as anchor failures rather than model failures — this is the intended diagnostic, not a bug. We added the Phase 3a homogeneity gate because the load-bearing assumption (that the calibration population contains the good-motion contrast needed to place a threshold) is otherwise unverified; if calibration generators share a single deformation failure mode, the frozen ellipse encodes that shared bias rather than a physical invariant, and the protocol must report non-identifiability rather than a tau score. A domain expert would not describe HOLD-STRIDE as a variant of StrideWorld: StrideWorld introduces the start-view and stride references but evaluates all models against references whose parameters are never separated from the judged population, whereas HOLD-STRIDE's contribution is the disjointness invariant, the homogeneity gate, and the identifiability verdict, none of which StrideWorld's construction produces.

**Why it measures what we claim.** The rigidity residual `r_rig` measures start-view geometric fidelity because the ground-truth geometry of the start view is fixed before any motion occurs, under the assumption that the first frame's geometry is a sufficient statistic for the scene's 3D structure; this assumption fails when a generator hallucinates plausible-looking but geometrically inconsistent first frames, in which case `r_rig` reflects first-frame rendering quality rather than scene rigidity. The stride residual `r_act` measures action consistency because each model's own first-K-step stride is a self-calibrating anchor for its subsequent steps, under the assumption that a model's initial motion is its most reliable motion; this assumption fails when a model's first steps are themselves anomalous (e.g., a slow start followed by a jump), in which case `r_act` reflects the anomaly of the first steps rather than inconsistency of later steps. The frozen acceptance ellipse measures reference identifiability because a region fitted on a disjoint generator population can only separate held-out generators if it captures a generator-invariant physical property, under the assumption that the calibration population is representative of the held-out population's motion repertoire; this assumption fails when the held-out generators use fundamentally different camera-control interfaces, in which case the ellipse reflects interface mismatch rather than anchor invalidity — the Phase 4b coverage diagnostic is reported precisely to expose this confound, and the Phase 3a homogeneity gate blocks the identifiability verdict when calibration residuals are not statistically indistinguishable across generators. The Kendall tau between frozen-anchor ranks and human ranks measures human alignment of the anchor because human preference is treated as the ground-truth ordering, under the assumption that human raters judge rigidity and consistency consistently across generators; this assumption fails when raters are biased toward visual smoothness, in which case tau reflects smoothness preference rather than physical fidelity.

## Contribution

['A held-out generator-population calibration protocol (HOLD-STRIDE) that separates anchor fitting from anchor validation for reference-fixed video world-model benchmarks, producing a frozen acceptance region and an identifiability verdict rather than a single population-fitted score.', 'A reference-identifiability test based on a generator-level permutation null that distinguishes anchors capturing generator-invariant physical properties from anchors fitted to the judged population, applicable to any benchmark with a reference-fixed residual.', "An analysis of how population conflation propagates across video-generation evaluation benchmarks, showing that GRADEO, Video-Bench, MJ-Video, and StrideWorld all share the calibration/evaluation population and that StrideWorld's start-view geometry is the most physically grounded candidate for a disjoint calibration protocol."]

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | StrideWorld 140 start views, open interactive world models | Broad generator coverage enables generator-level split |
| Primary metric | Kendall tau between frozen-anchor ranks and human ranks | Directly tests anchor identifiability under population shift |
| Secondary metric | Calibration-coverage fraction per held-out generator | Measures representativeness assumption rather than acknowledging it |
| Baseline | StrideWorld self-fitted references | Tests whether freezing references improves human alignment |
| Baseline | View-level split calibration | Tests whether generator-level disjointness is necessary |
| Baseline | Learned residual classifier | Tests whether percentile ellipse avoids population conflation |
| Ablation-of-ours | HOLD-STRIDE without freezing (refit on held-out) | Isolates contribution of the freeze invariant |
| Ablation-of-ours | HOLD-STRIDE without Phase 3a homogeneity gate | Isolates contribution of the homogeneity criterion |

### Why this setup validates the claim

The central claim is that a frozen anchor fitted on a disjoint generator population yields a reference that still separates good from bad motion when the generator population shifts, and that this separation tracks human preference. The StrideWorld self-fitted baseline tests the first sub-claim directly: if references fitted to the judged video absorb deformation, its ranks should drift from human ranks precisely on generators whose motion is subtly deformed. The view-level split baseline tests the disjointness invariant: if generator identity leaks through shared views, its identifiability test should be vacuous, showing high tau even under label shuffling. The learned-classifier baseline tests the percentile-ellipse choice: a classifier should re-absorb generator statistics and lose held-out alignment relative to the frozen ellipse. The no-freeze ablation isolates the freeze itself, so any gain can be attributed to the invariant rather than to residual extraction. The no-homogeneity-gate ablation isolates the Phase 3a gate, so any gain in identifiability verdict reliability can be attributed to the homogeneity criterion rather than to the split. The calibration-coverage diagnostic measures the representativeness assumption directly: if held-out generators with low coverage show systematically lower tau, the frozen anchor is failing for interface mismatch rather than anchor invalidity, and the identifiability verdict is reported conditional on coverage. Kendall tau against human ranks is the right metric because the claim is about ordering, not absolute scores: a frozen anchor that preserves the good-bad ordering under population shift is exactly what identifiability means, and a permutation null gives a principled threshold for whether the observed ordering exceeds chance. Together these choices form a falsifiable test: the claim predicts a specific pattern of elevated tau for HOLD-STRIDE concentrated on shifted generators, not uniform superiority.

### Expected outcome and causal chain

**vs. StrideWorld self-fitted references** — On a case where a generator gradually deforms its scene as the camera moves, the baseline produces inflated rigidity scores because its reference is reconstructed from the video's own frames and absorbs the deformation. Our method instead compares against the fixed start-view geometry, because the anchor is frozen before motion is judged, so we expect a noticeable tau gap on the deformed-generator subset but near-parity on generators with clean motion.

**vs. View-level split calibration** — On a case where calibration and held-out views come from the same generator, the baseline produces a high identifiability tau because shared motion priors leak generator identity into the acceptance region. Our method instead enforces generator-level disjointness, because no view from a calibration generator enters held-out scoring, so we expect the baseline's tau to stay high under label shuffling while ours drops to the null band, exposing the baseline's test as vacuous.

**vs. Learned residual classifier** — On a case where a held-out generator has a rendering style unseen in calibration, the baseline classifier produces confident rejections because it memorized calibration generator statistics rather than a physical property. Our method instead uses a frozen robust-covariance ellipse, because percentile thresholds on residuals do not encode generator identity, so we expect the classifier to show a larger tau drop on shifted generators while the ellipse retains ordering.

**vs. HOLD-STRIDE without freezing** — On a case where held-out generators have systematically larger strides, the no-freeze ablation produces a shifted acceptance region because refitting on held-out data re-centers the ellipse. Our method instead keeps the frozen anchor, because refitting would reproduce the population-conflation problem, so we expect the ablation to show inflated scores and a tau that fails to separate good from bad motion under shift.

**vs. HOLD-STRIDE without Phase 3a homogeneity gate** — On a case where calibration generators share a single deformation failure mode, the no-gate ablation proceeds to fit and freeze an ellipse that encodes that shared bias, producing a tau score that appears to validate the anchor. Our method instead runs the Kruskal-Wallis test across calibration generators and returns 'anchor not identifiable on this population' when residuals are not statistically indistinguishable, so we expect the no-gate ablation to report a tau on populations where our method correctly refuses to report one, exposing the gate as necessary for the identifiability verdict.

### What would falsify this idea

If HOLD-STRIDE's Kendall tau advantage over the self-fitted baseline is uniform across all generator subsets rather than concentrated on generators with predicted deformation or population shift, the central claim that freezing a disjoint anchor captures a generator-invariant property is wrong. Likewise, if the view-level split baseline's tau collapses to the null band under label shuffling just as ours does, the disjointness invariant is not doing the work we attribute to it; and if the Phase 3a homogeneity gate never fires on the available generator pool, the load-bearing assumption is untested rather than supported.

## References

1. StrideWorld: Benchmarking Scene Rigidity and Action Consistency in Interactive Video World Model
