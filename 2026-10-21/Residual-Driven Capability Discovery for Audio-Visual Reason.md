# Residual-Driven Capability Discovery for Audio-Visual Reasoning Benchmarks

## Motivation

Fixed-taxonomy audio-visual benchmarks such as EvaOmni organize evaluation around 15 hand-defined capabilities and three reasoning levels, yet the taxonomy itself is never validated for coverage: there is no procedure that can detect a reasoning type or model-specific shortcut that the designers did not anticipate, so a model can pass by exploiting capabilities outside the taxonomy and the benchmark reports coverage it cannot establish. This is a field-level structural problem, not a local EvaOmni flaw — every manually authored capability taxonomy (including the multimodal and unimodal-shortcut filtering EvaOmni performs) assumes the designer's enumeration is exhaustive, which makes 'comprehensive evaluation' an unfalsifiable claim.

## Key Insight

A taxonomy's coverage is exactly the fraction of model failure mass it can attribute to a named capability, so the unexplained residual failure mass after clustering is a directly measurable, monotonically shrinking quantity that exposes unanticipated reasoning types and shortcuts without requiring the designer to have anticipated them.

## Method

**What it is.** RESCAN (Residual-Scan Capability Discovery) takes as input a probe pool of audio-visual items (unlabeled or held-out EvaOmni-style clips with QA) and a diverse panel of audio-visual models, and outputs an expanded capability taxonomy plus a per-model residual-failure report quantifying how much failure mass remains unexplained. Its core loop is: probe → decompose → cluster residual failure → human-adjudicated promotion → re-measure residual.

**How it works.**

```pseudocode
INPUT: probe pool P (N items), model panel M = {m_1..m_K} (K>=6, spanning audio-only,
       visual-only, and joint AV models), seed taxonomy T_0 (EvaOmni's 15 caps),
       promotion threshold tau=0.05 of total failure mass, rounds R=4,
       attribution-calibration sample size S=200 residual failures per round

PHASE 1 - Capability-conditioned probing
  For each item p in P and model m in M:
      record answer correctness a(m,p) in {0,1}
      record evidence-localization score e(m,p) in [0,1] (audio & visual spans)
  Build modality-ablation tensor: for each p, re-run every m with audio muted,
  with video muted, and with both (text-only). Store correctness under each
  ablation as A(m,p,ablation).

PHASE 2 - Shortcut signature per (m,p)
  shortcut(m,p) = 'audio_sufficient' if A(m,p,no_video)=1 and A(m,p,full)=1
                = 'visual_sufficient' if A(m,p,no_audio)=1 and A(m,p,full)=1
                = 'text_only'        if A(m,p,no_audio,no_video)=1
                = 'genuine'          if A(m,p,full)=1 and all ablations fail
                = 'fail'             otherwise
  (This is a deterministic decision table, not a learned head.)

PHASE 3 - Residual failure extraction with attribution-uncertainty bucket
  For each (m,p) with a(m,p)=0 OR shortcut(m,p) in shortcut set:
      assign to an existing taxonomy node t in T_k if the item's annotated
      capability label matches t AND the failure is not attributable to a
      shortcut already in T_k.
      ELSE if the item's annotated capability label is ambiguous (multi-annotator
      agreement < 0.67 on the capability label) OR the item has no annotated
      capability label, assign to ATTRIBUTION_UNCERTAIN bucket U_k.
      ELSE assign to RESIDUAL R_k.
  RESIDUAL R_k = set of (m,p) failures not attributable to any node in T_k
                 and not in U_k.
  residual_mass_k = |R_k| / (total failures across M x P)
  uncertain_mass_k = |U_k| / (total failures across M x P)

PHASE 3b - Attribution calibration (noise correction)
  Sample S=200 failures from the union of attributed-to-node and R_k failures
  uniformly at random per round. Two annotators independently re-label each
  sampled failure's true capability (or 'unanticipated') without seeing the
  original label. Compute mis-attribution rate r_k = fraction of sampled
  failures whose original attribution disagrees with the adjudicated label.
  Report noise-corrected residual mass:
      residual_mass_k_corrected = residual_mass_k * (1 - r_k)
  with 95% bootstrap confidence interval (1000 resamples over the S calibration
  failures). Failures in U_k are reported separately and never counted as
  explained or as residual.

PHASE 4 - Residual clustering
  Embed each residual failure by its (capability label, shortcut signature,
  modality-ablation pattern, question type) tuple.
  Run agglomerative clustering (average linkage, cosine distance, cut at
  distance d=0.35 chosen on a held-out 20% of residuals).
  Each cluster c with mass >= tau is a CANDIDATE new capability or shortcut node.

PHASE 5 - Human adjudication + promotion
  For each candidate cluster: two annotators write a one-line capability
  definition and verify >=20 exemplar items genuinely require it (reject if
  the cluster is explained by an existing node). Promote accepted clusters into
  T_{k+1}. Record rejected clusters and why.

PHASE 6 - Re-measure and report
  After R rounds output T_R, per-model residual_mass_R (corrected and uncorrected),
  uncertain_mass_R, mis-attribution rate r_R with CI, and a per-model
  'unexplained failure profile' (which clusters remain).
```

Hyperparameters: K≥6 models, τ=0.05 residual-mass promotion threshold, d=0.35 cluster cut, R=4 rounds, 20% residual holdout for cluster-cut selection, S=200 calibration failures per round, multi-annotator agreement threshold 0.67 for the attribution-uncertain bucket.

**Why this design.** We chose a modality-ablation decision table over a learned shortcut classifier because the table is auditable and cannot itself introduce the shortcut it is meant to detect, accepting the cost that it only catches the three named shortcut families rather than arbitrary ones. We chose residual-mass clustering over simply asking annotators to brainstorm missing capabilities because clustering operates on actual model failures rather than designer imagination, accepting the cost that clusters require human adjudication and can be noisy at small mass. We chose a fixed promotion threshold τ over an automatic taxonomy-growth rule because an unbounded taxonomy inflates coverage trivially, accepting the cost that genuinely rare capabilities below τ are never promoted. We chose agglomerative clustering over k-means because the number of new capabilities is unknown a priori, accepting the cost of a distance-cut hyperparameter. We added an explicit attribution-uncertainty bucket and a per-round calibration sample because the label-match attribution in Phase 3 is itself derived from human annotation and can mis-bucket failures; without measuring that error the residual mass is not identifiable. A domain expert would NOT describe this as retrieval or as a learned-disentanglement head: the mechanism is a residual-mass accounting loop whose output is a measurable coverage quantity, which is the one thing EvaOmni's fixed taxonomy and standard retrieval-based capability selectors cannot produce.

**Why it measures what we claim.** The residual failure mass `residual_mass_k` measures *taxonomy coverage incompleteness* because it counts failures not attributable to any named node, under the assumption that the annotation+shortcut attribution in Phase 3 is exhaustive and error-free for the current taxonomy; this assumption fails when a failure is mis-attributed to an existing node that superficially matches, in which case residual mass under-reports true incompleteness — which is why we report the noise-corrected `residual_mass_k_corrected` with a bootstrap CI from the S=200 calibration sample and a separate `uncertain_mass_k`. The shortcut signature `shortcut(m,p)` measures *model-specific shortcut reliance* because the modality-ablation contrast isolates the minimal sufficient modality, under the assumption that muting a modality does not change the task in a way the model can detect from text alone; this fails when the QA text leaks the answer, in which case 'text_only' flags a dataset artifact rather than a model shortcut. The promoted cluster mass measures *newly discovered capability prevalence* because clusters above τ represent recurring failure patterns not covered before, under the assumption that the embedding tuple captures capability identity rather than surface item similarity; this fails when clusters split by video genre rather than reasoning type, in which case the promoted node reflects a domain split, not a capability.

## Contribution

['RESCAN, a residual-mass capability-discovery loop that converts taxonomy coverage into a measurable quantity (unexplained failure mass) and iteratively promotes human-adjudicated clusters of unexplained failures into new capability or shortcut nodes.', 'A modality-ablation shortcut signature that distinguishes genuine multimodal reasoning from audio-sufficient, visual-sufficient, and text-only shortcuts per model-item pair, enabling per-model shortcut profiling rather than dataset-level filtering.', 'An empirical finding on how much of EvaOmni-panel model failure mass is unexplained by its 15-capability taxonomy and which shortcut families account for the remainder, reported as a residual-failure profile per model.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | EvaOmni probe pool + held-out clips | Evidence-grounded AV items with 15-cap seed taxonomy |
| Primary metric | Noise-corrected residual failure mass after R rounds (with 95% bootstrap CI) | Directly quantifies unexplained coverage incompleteness |
| Secondary metric | Attribution mis-attribution rate r_k on S=200 calibration sample | Bounds the load-bearing attribution assumption |
| Baseline | EvaOmni fixed 15-cap taxonomy | Static taxonomy, no residual accounting |
| Baseline | Retrieval-based capability selector | Selects existing caps, cannot discover new ones |
| Baseline | Annotator brainstorm (no clustering) | Human-imagined caps, not failure-driven |
| Ablation-of-ours | RESCAN minus ablation tensor (no shortcut signature) | Tests whether shortcut removal drives residual discovery |
| Ablation-of-ours | RESCAN minus calibration (uncorrected residual mass) | Tests whether noise correction changes conclusions |

### Why this setup validates the claim

The central claim is that residual-mass accounting over real model failures discovers capability gaps that a fixed taxonomy and retrieval selectors structurally cannot surface. Each baseline isolates one sub-claim: the EvaOmni fixed-taxonomy baseline tests whether residual mass actually shrinks beyond the seed 15 capabilities; the retrieval selector tests whether discovery requires failure-driven clustering rather than selecting from a known cap set; the annotator-brainstorm baseline tests whether clustering on actual residuals outperforms designer imagination in producing promotable, non-redundant nodes. The ablation removes the modality-ablation shortcut signature, testing whether shortcut attribution is doing real work in concentrating residual mass onto genuine capability gaps rather than dataset artifacts. The second ablation removes the calibration correction, testing whether the headline metric is sensitive to attribution noise. Residual failure mass is the right primary metric because it is exactly the quantity the method claims to reduce and the one thing fixed taxonomies cannot report — a coverage incompleteness number that goes down only when failures are genuinely re-attributed to newly named, human-adjudicated nodes. Reporting the noise-corrected residual mass with a bootstrap CI and the mis-attribution rate r_k makes the load-bearing attribution assumption explicit and bounded: if r_k is large, the corrected residual mass will differ substantially from the uncorrected one, and the confidence interval will widen, preventing an over-confident coverage claim.

### Expected outcome and causal chain

**vs. EvaOmni fixed 15-cap taxonomy** — On a case where a model fails a multi-hop item requiring cross-modal temporal alignment (a capability not in the seed 15), the fixed taxonomy produces an unchanged residual mass because it has no mechanism to name the gap — the failure is either bucketed into a superficially matching node or left uncounted. Our method instead clusters that failure pattern, promotes it after human adjudication, and re-measures, so we expect the noise-corrected residual mass to drop noticeably by round 3-4, with the drop concentrated in the promoted clusters rather than uniformly distributed, and the mis-attribution rate r_k to remain below 0.15 so the correction does not erase the decline.

**vs. Retrieval-based capability selector** — On a case where the correct capability is absent from the seed taxonomy, the retrieval selector produces a wrong nearest-capability assignment because it can only choose from existing nodes, silently mis-attributing the failure. Our method instead leaves the failure in the residual pool (or the attribution-uncertain bucket if the label is ambiguous), clusters it with siblings, and surfaces it as a candidate new node, so we expect the retrieval baseline to show flat residual mass while ours declines, with the gap largest on items whose true capability is genuinely novel and whose calibration re-labeling confirms the original attribution was wrong.

**vs. Annotator brainstorm (no clustering)** — On a case where designers brainstorm missing capabilities without seeing failures, the brainstorm baseline produces plausible-sounding but non-prevalent or redundant nodes because it operates on imagination rather than observed failure mass. Our method instead promotes only clusters exceeding τ mass with verified exemplars, so we expect brainstorm nodes to show low exemplar-verification pass rates and high redundancy rejection, while our promoted nodes concentrate a larger share of residual mass per node.

**vs. RESCAN minus ablation tensor** — On a case where a model fails only because it leans on a text-only shortcut, the no-ablation variant produces inflated residual clusters that mix shortcut artifacts with genuine capability gaps because it cannot separate them. Our method instead attributes shortcut failures before clustering, so we expect the ablation variant to promote more clusters that human adjudicators reject as dataset artifacts, yielding a higher rejection rate and less residual-mass reduction per promoted node.

**vs. RESCAN minus calibration** — On a case where attribution noise is high (r_k > 0.2), the uncorrected residual mass will be systematically lower than the corrected one because mis-attributed failures are silently counted as explained. Our method reports both and a CI, so we expect the uncorrected variant to overstate coverage reduction and the corrected variant to show a smaller but statistically significant decline, with the gap between them tracking r_k across rounds.

### What would falsify this idea

If residual mass declines uniformly across all subsets rather than concentrating on the promoted clusters and on items whose capability is genuinely absent from the seed taxonomy, the residual-accounting mechanism is not doing what we claim — the decline would be an artifact of re-bucketing rather than genuine discovery. A second falsifier: if the no-ablation variant matches our residual-mass reduction and promotion acceptance rate, the shortcut-signature step is inert and the central causal chain collapses. A third falsifier: if the mis-attribution rate r_k is so high that the 95% bootstrap CI on corrected residual mass includes zero decline across rounds, the coverage claim is not identifiable and the method fails its own calibration test.

## References

1. EvaOmni: An Evidence-Grounded Benchmark for Audio-Visual Reasoning
