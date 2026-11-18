# Component-Attributed Oracle Feedback for Iterative Research Ideation

## Motivation

Across LLM ideation research, feedback on a candidate idea is delivered as a scalar whole-idea judgment — a blind review score (as in the 100+ NLP researcher study) or a binary yes/no oracle answer (as in Re-Search) — never decomposed into attributable sub-criteria. Structurally, this forces the revising model to treat the entire idea as the unit of credit assignment, so a single 'no' carries no information about whether the hypothesis, method, or novelty claim is the defective part; Re-Search's own trace analysis shows models therefore over-exploit one direction or over-explore many, because the feedback surface cannot localize the revision target. The root cause is the scalar attachment assumption, not insufficient model capacity.

## Key Insight

If every oracle 'no' is accompanied by a forced localization to one of a fixed set of idea components (hypothesis, method, novelty claim) and a sub-criterion verdict within that component, then the feedback signal becomes a per-component credit assignment that lets the model revise the defective component while holding the others fixed — a structural property scalar feedback cannot provide regardless of how many questions are asked.

## Method

**What it is.** We introduce CAFE (Component-Attributed Feedback Engine), which replaces Re-Search's scalar binary oracle with a component-factorized oracle. Inputs: the research problem plus the model's current candidate idea (a structured tuple of hypothesis, method, novelty claim). Outputs: for each query, a verdict vector `(v_hyp, v_meth, v_nov) ∈ {yes,no}³` together with, for every `no`, a forced localization `ℓ ∈ {hypothesis, method, novelty}` and a sub-criterion string drawn from a fixed per-component rubric.

**Load-bearing assumption (stated explicitly).** CAFE assumes that for any candidate idea, each `no` can be assigned to exactly one of the three components via the fixed 3-binary-sub-criterion rubric, such that the resulting component verdict vector reflects the true defective component rather than rubric-boundary artifacts — i.e., the components are separable and the rubric is component-private and exhaustive. This assumption is false when a method defect makes a hypothesis unfalsifiable in practice, or when novelty is defined relative to the method; in those cases the forced localization attributes a `no` to a component that is only marginally at fault, injecting attribution noise. The design below makes this noise observable via a disagreement-based confidence score and a first-class `cross-cutting` abstain label, so the revision step can act on attribution confidence rather than on a deterministic-looking attribution of judge noise.

**How it works.**

```pseudocode
# Fixed rubric R_c per component c (hypothesis / method / novelty), each with 3 binary sub-criteria
# e.g. R_hypothesis = [well-posed, falsifiable, non-trivial]
#      R_method     = [feasible, appropriate, specified]
#      R_novelty    = [distinct-from-prior, non-obvious, claim-scoped]

function CAFE_query(problem, idea_tuple, question):
    # Phase 1: decompose the question into component targets
    targets = LLM_decompose(question)        # subset of {hypothesis, method, novelty}
    # Phase 2: per-target sub-criterion scoring (forced, no abstain)
    verdicts = {}
    for c in targets:
        for k in R_c:                         # 3 sub-criteria, no threshold tuning
            verdicts[(c,k)] = ORACLE_binary(problem, idea_tuple[c], k)
    # Phase 3: forced localization for every 'no'
    for c in targets:
        if any(verdicts[(c,k)] == no for k in R_c):
            verdicts[(c,'localize')] = c      # the 'no' is attributed to c, not the whole idea
    # Phase 4: aggregate into a component verdict vector
    v = {c: all(verdicts[(c,k)] for k in R_c) for c in targets}
    # Phase 5: attribution confidence via disagreement across rubric sub-criteria
    #   For each component c with v[c]==no, compute the fraction of sub-criteria that disagree
    #   with the majority verdict for that component. If disagreement > 0.5 (i.e., at least 2 of 3
    #   sub-criteria conflict with the component-level 'no'), label the attribution as 'cross-cutting'.
    attrib = {}
    for c in targets:
        if v[c] == no:
            sub_verdicts = [verdicts[(c,k)] for k in R_c]
            # majority vote among sub-criteria for this component
            maj = majority(sub_verdicts)
            disagree_frac = sum(1 for x in sub_verdicts if x != maj) / len(sub_verdicts)
            if disagree_frac > 0.5:
                attrib[c] = 'cross-cutting'   # first-class abstain label
            else:
                attrib[c] = c                 # forced localization
    return v, attrib

# Ideation loop (model side)
for t in 1..T:                                # T=100, matching Re-Search budget
    q_t = MODEL_ask(problem, idea_tuple, history)
    v_t, attrib_t = CAFE_query(problem, idea_tuple, q_t)
    # Revise ONLY components flagged 'no' with a non-'cross-cutting' attribution;
    # if any component is 'cross-cutting', fall back to whole-idea revision for that component.
    for c in components_with_no(v_t):
        if attrib_t[c] == 'cross-cutting':
            idea_tuple[c] = MODEL_revise_whole(problem, idea_tuple, history)  # fallback
        else:
            idea_tuple[c] = MODEL_revise(problem, idea_tuple[c], attrib_t[c], history)
    history.append((q_t, v_t, attrib_t))
```

**Why this design.** We chose a *forced* localization over an *optional* one because an optional localization can be skipped exactly when the model is most confused, which is the failure mode Re-Search documents; the cost is that the oracle sometimes attributes a 'no' to a component that is only marginally at fault, injecting attribution noise. We chose a fixed 3-criterion rubric per component over a learned rubric because a learned rubric would reintroduce the scalar-aggregation problem under a different name (Anti-Pattern 1: a learned head cannot manufacture separability that the rubric does not enforce); the cost is coarse granularity. We chose to hold non-flagged components fixed during revision rather than re-generate the whole idea, because whole-idea regeneration is precisely what scalar feedback already permits and what Re-Search shows leads to over-exploration; the cost is that a genuinely cross-cutting defect must be surfaced through multiple queries, which the 100-question budget accommodates. A domain expert would not describe CAFE as a variant of Re-Search-with-a-richer-oracle, because Re-Search's oracle returns a scalar and the model's action space is 'ask another question'; CAFE's oracle returns a *vector over a fixed component basis* and the model's action space is 'revise this component while freezing the others' — a different credit-assignment surface, not a richer label on the same surface. The disagreement-based confidence and `cross-cutting` label make the attribution noise observable, so the revision step can act on attribution confidence (revise only when the component posterior is peaked; otherwise fall back to whole-idea revision), which preserves the contribution even if components are not perfectly separable.

**Why it measures what we claim.** The component verdict vector `v` measures *attributable defect localization* because the forced-localization step makes each 'no' map to exactly one component under the assumption that the rubric's sub-criteria are component-private (a sub-criterion of the hypothesis rubric is not evaluable against the method alone); this assumption fails when a method defect makes a hypothesis unfalsifiable in practice, in which case `v` reflects rubric-boundary artifacts rather than the true defective component. The number of components revised per round measures *revision focus* because holding non-flagged components fixed makes the count equal to the number of independently defective components under the assumption that components are revised independently; this assumption fails when a revision to the hypothesis forces a method rewrite, in which case the count understates the true revision scope. The plateau point (first t after which no component verdict changes) measures *ideation convergence* because a stable component vector means the oracle can no longer localize a defect under the rubric, which is equivalent to convergence only under the assumption that the rubric is exhaustive for the target idea space; this assumption fails for genuinely novel ideas that fall outside all three rubrics, in which case the plateau reflects rubric exhaustion rather than idea quality. The disagreement fraction (Phase 5) measures *attribution confidence* because it quantifies the fraction of sub-criteria that conflict with the component-level verdict under the assumption that sub-criteria within a component are mutually consistent when the component is truly defective; this assumption fails when the component is cross-cutting (e.g., a method defect that makes the hypothesis unfalsifiable), in which case the disagreement fraction is high and the `cross-cutting` label correctly signals that the forced localization is unreliable.

## Contribution

["CAFE, a component-factorized oracle that replaces scalar binary feedback with a per-component verdict vector and forced localization, turning each 'no' into a credit-assignment signal over hypothesis, method, and novelty claim.", "A design principle — freeze-and-revise — under which the ideation model revises only oracle-flagged components, providing a structural alternative to whole-idea regeneration that is testable against Re-Search's over-exploration failure mode.", 'A component-attributed ideation trace dataset built from the same 200-paper Re-Search construction, enabling analysis of which components drive progress and where models plateau.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Re-Search 200-paper ICML problem set | Same problems and budget as the target baseline |
| Primary metric | Component-attribution accuracy of localized defects | Directly measures the claimed credit-assignment surface |
| Baseline | Re-Search scalar binary oracle | Isolates effect of vector vs scalar feedback |
| Baseline | Re-Search + optional localization | Tests whether forced localization is the causal ingredient |
| Baseline | Re-Search + whole-idea regeneration | Tests whether freezing components drives focus |
| Baseline | Independent LLM judge producing the same component verdict vector | Isolates structural claim from 'more labels help' |
| Ablation-of-ours | CAFE with single-criterion rubric per component | Tests whether the 3-criterion rubric adds separability |
| Ablation-of-ours | CAFE without disagreement-based confidence and `cross-cutting` label | Tests whether calibration is necessary for attribution accuracy |
| Human ground truth | Stratified 40-problem subset, two annotators, report inter-annotator agreement (Cohen's κ) | Grounds forced-localization mechanism in measured reliability |

### Why this setup validates the claim

The central claim is that a component-factorized, forced-localization oracle changes the credit-assignment surface enough that the model revises the right component instead of over-exploring or over-exploiting. Re-Search's scalar oracle is the right primary baseline because it holds problem set, question budget (100), and player model fixed, so any difference must come from the oracle's output structure rather than from more information or more queries. The optional-localization baseline is the sharpest sub-claim test: if forced localization is doing the work, removing the forcing should degrade attribution accuracy specifically on the subset of cases where the model's question is ambiguous, not uniformly. The whole-idea-regeneration baseline tests the freezing mechanism: if holding non-flagged components fixed is what produces revision focus, this baseline should show higher revision-scope counts and earlier plateau on cross-cutting-defect cases. The independent LLM judge baseline produces the same component verdict vector but without the oracle's forced localization (i.e., it uses a separate LLM to assign component labels directly from the idea text), so if CAFE's advantage persists only when localization is oracle-forced, the structural claim is isolated from a generic 'more labels help' account. The single-criterion ablation tests whether the 3-criterion rubric is what makes components separable; if separability comes from the forced localization alone, the ablation should match CAFE. The no-calibration ablation tests whether the disagreement-based confidence and `cross-cutting` label are necessary for attribution accuracy; if removing them does not hurt, the calibration step is not load-bearing. Component-attribution accuracy is the right primary metric because it is the quantity the mechanism claims to produce — not final idea novelty, which confounds oracle quality with the player model's generation ability and would not distinguish our causal story from a generic 'richer feedback helps' account. The human ground-truth subset (40 problems, two annotators) provides a measured reliability check: we report oracle localization precision and recall per component against the annotators' majority label, and we exclude cases where annotators disagree on the defective component (cross-cutting-defect exclusion rule) from the primary accuracy computation, reporting the exclusion rate separately.

### Expected outcome and causal chain

**vs. Re-Search scalar binary oracle** — On a case where the model's candidate idea has a well-posed but non-falsifiable hypothesis and a feasible method, the scalar oracle returns `no` with no location, so the model's revision action is to ask another question or regenerate, and Re-Search's documented over-exploration kicks in as it drifts across directions. Our method instead returns `(no, yes, yes)` with localization `hypothesis` and the sub-criterion `falsifiable`, so the model revises only the hypothesis and holds the method fixed. We expect a noticeable gap in component-attribution accuracy concentrated on multi-defect and near-miss cases, with near-parity on cases where the idea has a single obvious defect that scalar feedback already isolates by elimination.

**vs. Re-Search + optional localization** — On a case where the model is genuinely confused about which component is at fault, the optional-localization baseline skips the localization exactly when it is most needed, because the model's uncertainty about the question is highest precisely where the defect is ambiguous, so the baseline collapses back to scalar behavior on the hard subset. Our method forces a localization for every `no`, so even a marginally-attributed verdict produces a revision target. We expect the optional baseline to match CAFE on easy single-defect cases but to show a clear attribution-accuracy drop on the ambiguous subset, and we expect its revision-scope counts to be noisier because skipped localizations leave defects unattributed.

**vs. Re-Search + whole-idea regeneration** — On a case where only the novelty claim is defective but the hypothesis and method are sound, whole-idea regeneration re-samples all three components, so a sound hypothesis can be replaced by a worse one and the model's trajectory shows the over-exploration Re-Search documents. Our method freezes the two sound components and revises only the novelty claim, so the trajectory should show revision-scope counts near one on single-defect cases and a later plateau point than whole-idea regeneration. We expect the gap to be largest on cases with a single localized defect and to shrink on cases with genuinely cross-cutting defects, where our freezing assumption is stressed and the whole-idea baseline is comparatively advantaged.

**vs. Independent LLM judge producing the same component verdict vector** — On a case where the idea has a clear single-component defect, the independent judge can assign the correct component label from the idea text alone, so both methods should perform similarly. However, on ambiguous or multi-defect cases, the independent judge lacks the oracle's forced localization and must infer the defective component from the idea text without the structured sub-criterion rubric, so its attribution accuracy should be lower. We expect CAFE to outperform the independent judge specifically on the ambiguous subset, demonstrating that the structural forced localization (not merely the presence of component labels) drives the gain.

**vs. CAFE with single-criterion rubric per component** — On a case where a component has multiple potential defects (e.g., a hypothesis that is both non-falsifiable and trivial), the single-criterion rubric can only flag one sub-criterion, so the component verdict may be `yes` if the single criterion passes, missing the defect. The 3-criterion rubric aggregates over three sub-criteria, so it is more likely to detect a defective component. We expect the single-criterion ablation to show lower attribution accuracy on multi-defect cases, with near-parity on single-defect cases where the single criterion happens to align with the defect.

**vs. CAFE without disagreement-based confidence and `cross-cutting` label** — On a case where a method defect makes the hypothesis unfalsifiable in practice (a cross-cutting defect), the no-calibration variant forces a localization to either hypothesis or method, producing an attribution that may be wrong. The calibrated CAFE detects high disagreement among sub-criteria and labels the attribution as `cross-cutting`, falling back to whole-idea revision. We expect the no-calibration variant to show lower attribution accuracy on cross-cutting cases and to have a higher rate of misattributed revisions, while the calibrated version maintains accuracy by correctly abstaining.

### What would falsify this idea

If CAFE's attribution-accuracy gain over the scalar oracle is uniform across easy and ambiguous subsets rather than concentrated on the ambiguous and multi-defect cases where forced localization is predicted to matter, then the gain is not coming from the component basis and the central claim is wrong. Likewise, if the single-criterion ablation matches CAFE on attribution accuracy, the 3-criterion rubric is not what enforces separability and the design rationale collapses. Additionally, if the independent LLM judge baseline matches CAFE on the ambiguous subset, the structural forced localization is not the causal ingredient.

## References

1. Re-Search: Dissecting LLM Ideation Through a Two-Player Questioning Game
