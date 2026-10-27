# Perceptual State Grounding for Verifiable Cross-Episode GUI Work

## Motivation

Work Agents characterizes reliable digital work by a persistence dimension but never specifies how a persistent workspace state is grounded in what the agent actually perceives; CogAgent perceives only within a single episode from screenshots, and PPTAgent's decomposition assumes the workspace schema is given rather than recovered from observation. The structural root cause is that every existing system treats the persistent state as an internal, unobservable variable, so there is no condition under which two agents (or two episodes) can be shown to agree on the same workspace state from screenshots alone — cross-episode state identifiability is simply undefined, not merely unmeasured.

## Key Insight

If the state-grounding function is constrained to be invariant across the perceptual aliasing classes that the GUI itself generates (same state rendered differently, different states rendered identically), then the persistent state becomes a measurable function of the screenshot sequence, and cross-episode agreement is a testable identifiability condition rather than an assumption.

## Method

**What it is.** We introduce **GRASP** (Grounded Representation And State Partitioning), a state-grounding function `g: (screenshot sequence) -> persistent state representation` plus an identifiability certificate. Input: an ordered sequence of screenshots `o_1..o_T` from one or more episodes over the same workspace. Output: a discrete state label `z_t` per observation and a partition `P` of observations into equivalence classes claimed to be workspace states, together with a certificate listing which aliasing classes were exercised and which remain confounded. The certificate is reported **conditional on a declared perturbation family** `P_fam` (window size, theme, scroll offset, transient toast) and on a **calibrated null residual distribution**; observations flagged as out-of-distribution (OOD) with respect to the probe set are marked **uncertifiable** rather than forced through the residual rule.

**Load-bearing assumption (stated explicitly).** The fitted renderer `R(state, perturbation)` is assumed accurate enough that its pixel residual, evaluated against a **calibrated null distribution** of same-state residuals, cleanly separates nuisance variation from genuine workspace-state change. This assumption fails when the GUI renders content the probe set never exercised (live data tables, timestamps, ads, animations, anti-aliasing/font-stack differences across OS and DPI), in which case the residual reflects unmodeled content rather than a state change. We therefore (i) fit `R` only over the declared closed perturbation family `P_fam`, (ii) replace the fixed `tau_render = 0.05` with a per-region, per-app calibrated threshold `tau_render(app, region)` obtained from a held-out set of same-state pairs, and (iii) gate the certificate on an OOD detector for content outside the probe set.

**How it works.**

```pseudocode
# Phase 0: harness probes the GUI to build an alias model (offline, per app)
# Declared perturbation family P_fam = {window size, theme, scroll offset, transient toast}
for each (state s, perturbation p) in AliasProbeSet (20k renders per app, p in P_fam):
    render s under p  -> o            # p = window size, theme, scroll offset, transient toast
    record (s, p, o)
fit renderer R: (state, perturbation) -> observation   # contrastive encoder, InfoNCE, temperature 0.07
fit inverse    I: observation -> (state, perturbation) # same encoder, two heads

# Phase 0b: calibration of the null residual distribution (per app, per screen region)
# Held-out same-state pairs: 512 pairs per app, disjoint from AliasProbeSet
for each held-out same-state pair (o_a, o_b) with known p_ab in P_fam:
    r_ab = ||R(z_a, p_ab) - o_b||
fit null distribution D_null(app, region) over {r_ab}
set tau_render(app, region) = 95th percentile of D_null(app, region)   # calibrated FP rate = 5%
set tau_ambig(app, region)  = 99.5th percentile of D_null(app, region) # ambiguous band upper edge

# Phase 0c: OOD detector for content outside the probe set
# Fit a density estimator (e.g., 2-layer MLP, hidden=256, GeLU) on probe-set observation embeddings
fit OOD: embedding -> in-distribution score
set tau_ood = 5th percentile of in-distribution scores on held-out probe renders

# Phase 1: grounding a live episode (online)
z_prev = I(o_1).state
for t in 2..T:
    (s_hat, p_hat) = I(o_t)
    if OOD(o_t) < tau_ood:
        emit UNCERTIFIABLE(o_t)        # content outside probe set; do not assign z_t
        z_prev = z_prev                # carry forward last certified state
        continue
    # identifiability check: does the recovered perturbation explain the delta?
    r = ||R(s_prev, p_hat) - o_t||
    if r < tau_render(app, region):    # calibrated same-state threshold
        z_t = s_prev                   # same workspace state, different rendering
    else:
        z_t = s_hat                    # genuine state transition
    if tau_render(app, region) <= r < tau_ambig(app, region):
        emit ALIAS_WARNING(z_prev, s_hat, o_t)  # unidentifiable pair
    z_prev = z_t

# Phase 2: identifiability certificate over a screenshot log
P = partition(observations by z_t, excluding UNCERTIFIABLE)
for each pair (a,b) in P with same z but different pixels:
    assert exists p in P_fam s.t. R(z, p) ~= o_a and R(z, p) ~= o_b   # render-consistency
for each pair with different z but similar pixels:
    assert NOT exists p in P_fam s.t. R(z_a, p) ~= o_b               # separation
certify(P) = fraction of pairs passing both, plus list of ALIAS_WARNINGs and UNCERTIFIABLEs
```

Hyperparameters: `tau_render(app, region)` = 95th percentile of held-out same-state residual distribution `D_null(app, region)` (calibrated false-positive rate 5%); `tau_ambig(app, region)` = 99.5th percentile of `D_null(app, region)`; InfoNCE temperature `0.07`; encoder trained on 20k probe renders per app over declared `P_fam`; calibration set of 512 held-out same-state pairs per app; OOD detector = 2-layer MLP, hidden=256, GeLU, threshold `tau_ood` at 5th percentile of in-distribution scores.

**Why this design.** We chose an explicit *renderer/inverse pair* over a single contrastive embedding because the identifiability claim requires a generative direction — you must be able to say *what perturbation would produce this pixel difference* — accepting the cost that the renderer must be fit per application and will be imperfect on unseen UI toolkits. We chose a *calibrated residual-threshold decision rule* over a learned transition classifier because the threshold has a direct interpretation as a calibrated hypothesis test (stated false-positive rate) rather than an arbitrary constant, accepting the cost that the null distribution must be estimated per app and per region and that a badly chosen `tau_ambig` masks real transitions. We chose to *emit an ALIAS_WARNING rather than force a label* in the ambiguous band because a wrong state label silently corrupts the certificate, accepting the cost that some genuinely identifiable transitions are reported as unresolved. We chose to *mark OOD observations as UNCERTIFIABLE* rather than route them through the residual rule because the renderer's residual on unseen content reflects renderer capacity, not state identity, accepting the cost that the certificate denominator shrinks on apps with live content. A domain expert would not describe this as Toolformer/Gorilla-style retrieval or as a standard contrastive state encoder (e.g., the latent-state heads used in world-model agents): those map observations to embeddings but never construct the perturbation-explicit renderer needed to *prove* two observations are the same state, which is exactly the object our certificate checks.

**Why it measures what we claim.** The calibrated render residual `||R(z_prev, p_hat) - o_t||` compared against `D_null(app, region)` measures *workspace-state identity across observations* because we assume the fitted renderer captures all nuisance variation in the declared perturbation family `P_fam` and that the held-out null distribution is representative of same-state residuals; this assumption fails when the app renders content the probe set never exercised (e.g., a live data table), in which case the residual reflects unseen content rather than a state change — which is why OOD-gated observations are marked UNCERTIFIABLE rather than forced through the rule. The partition `P` measures *cross-episode state identifiability* because we assume every observation in a class is reachable from the class's state by some perturbation in `P_fam`; this fails when a state is only reachable through a perturbation family absent from probing, in which case `P` reflects probe coverage rather than identifiability. The certificate fraction measures *the fraction of state pairs whose identity is verifiable from screenshots alone* because we assume the separation check (no perturbation in `P_fam` maps `z_a` to `o_b`) is sound under the fitted renderer; this fails under renderer underfitting, in which case the certificate reflects renderer capacity rather than true separability.

## Contribution

(1) GRASP, a state-grounding function with an explicit perturbation-aware renderer/inverse pair that maps screenshot sequences to persistent workspace states and emits per-pair identifiability certificates. (2) The identifiability condition itself — a render-consistency plus separation test over the recovered perturbation — which converts cross-episode state agreement from an internal assumption into a screenshot-only checkable property, together with the finding that the ambiguous residual band is where existing GUI agents silently disagree on state. (3) An alias-probe protocol and certificate format for auditing GUI agent state persistence across episodes.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | Multi-episode GUI screenshot logs over shared workspaces (web + desktop apps); first experiment scoped to two apps with scriptable state spaces: one web app with a DOM-accessible state oracle and one desktop app with a fixed schema | Enables cross-episode state-identity claims, not single-trajectory labels; scriptable state oracles provide ground-truth state for calibration and certificate scoring |
| Primary metric | Certificate fraction: verified state-pair fraction minus alias-warning rate, computed over certified (non-UNCERTIFIABLE) observations; reported alongside UNCERTIFIABLE rate and calibration false-positive rate on held-out same-state pairs | Directly measures the identifiability claim, penalizing silent aliasing; calibration FP rate bounds the renderer-exactness assumption |
| Baseline | Contrastive state encoder (world-model latent head) | Tests whether perturbation-explicit renderer is necessary |
| Baseline | Perceptual-hash / pixel-diff deduplication | Tests whether naive nuisance invariance suffices |
| Baseline | Supervised transition classifier on labeled state deltas | Tests whether a learned decision rule beats calibrated residual thresholds |
| Ablation-of-ours | GRASP without ALIAS_WARNING (force label in ambiguous band) | Isolates the certificate-corruption cost of forced labels |
| Ablation-of-ours | GRASP with fixed `tau_render = 0.05` instead of calibrated per-app/per-region threshold | Isolates the contribution of calibration to certificate soundness |
| Ablation-of-ours | GRASP without OOD gating (route all observations through residual rule) | Isolates the contribution of OOD gating to certificate soundness |
| Calibration experiment | Hold out perturbation families (dark theme, 150% zoom, live table updates) from the probe set; measure certificate degradation and residual distribution shift (KL divergence between held-out and calibration `D_null`) | Empirically bounds the probe-coverage assumption rather than asserting it |

### Why this setup validates the claim

The central claim is that a perturbation-explicit renderer is what lets you *prove* two observations are the same workspace state, and that this proof is what makes the partition trustworthy. Each baseline attacks a different sub-claim. The contrastive encoder baseline tests whether the generative direction is actually load-bearing: it can cluster observations by appearance but cannot answer "what perturbation would produce this delta," so it must fail precisely on alias pairs where pixels differ but state is identical. The pixel-hash baseline tests whether nuisance invariance is trivial; if deduplication already recovered the partition, the renderer would be unnecessary. The supervised transition classifier tests whether the calibrated residual-threshold rule is doing real work or is a stand-in for a learnable decision boundary. The ablation forcing labels in the ambiguous band is the key falsification probe for the ambiguity machinery: if removing ALIAS_WARNING does not degrade the certificate, then the ambiguity machinery is decorative. The fixed-threshold ablation tests whether calibration is load-bearing: if a fixed `tau_render = 0.05` matches the calibrated threshold, then the renderer-exactness assumption is not being exercised. The no-OOD-gating ablation tests whether the OOD gate is load-bearing: if routing unseen content through the residual rule does not degrade the certificate, the gate is unnecessary. The held-out perturbation-family calibration experiment directly bounds the probe-coverage assumption by measuring how far the residual distribution shifts when a perturbation family is absent from probing. The certificate fraction is the right metric because it is defined over exactly the pairs the claim concerns — same-state-different-pixels and different-state-similar-pixels — so a method that merely clusters well cannot score high without actually separating and unifying the right pairs.

### Expected outcome and causal chain

**vs. Contrastive state encoder** — On a case where a live data table scrolls within one workspace state, the contrastive encoder produces two distinct state labels because its embedding keys on pixel content and has no way to attribute the delta to a scroll perturbation. Our method instead recovers the scroll offset via the inverse head, renders the previous state under it, and confirms the residual is below the calibrated `tau_render(app, region)`, so it keeps one label. We expect a concentrated gap on the alias-heavy subset (scrolling, theme, transient toasts) with near-parity on clean single-render transitions.

**vs. Perceptual-hash deduplication** — On a case where two genuinely different workspace states differ only by a small region (e.g., a single toggled checkbox or a one-row selection change), pixel-hash dedup collapses them into one class because its distance is dominated by unchanged pixels. Our method instead checks separation: no perturbation in `P_fam` maps the first state to the second observation, so the pair fails the render-consistency assertion and is correctly split. We expect dedup to score high on same-state unification but low on separation, yielding a large certificate gap driven almost entirely by the different-state-similar-pixels pair bucket.

**vs. Supervised transition classifier** — On a case where an unseen UI toolkit renders a state transition the training labels never covered, the classifier produces a confident but wrong transition label because it extrapolates a learned boundary rather than checking renderability. Our method instead emits ALIAS_WARNING when the residual lands in the calibrated ambiguous band, leaving the pair unresolved rather than silently mislabeled. We expect the classifier to show higher raw label coverage but a markedly lower certificate fraction, with its errors concentrated in the ALIAS_WARNING region our method flags.

**vs. GRASP without ALIAS_WARNING (ablation)** — On a case where a live-data residual sits between the calibrated `tau_render(app, region)` and `tau_ambig(app, region)`, the forced-label ablation assigns a state label and thereby asserts render-consistency that the renderer cannot satisfy, corrupting the certificate for every pair in that class. Our method instead withholds the label and records the alias warning, so the certificate denominator shrinks but its numerator stays sound. We expect the ablation to show a modestly higher partition size but a lower certificate fraction, with the degradation concentrated in apps whose probe set under-covers content perturbations.

**vs. GRASP with fixed `tau_render = 0.05` (ablation)** — On an app whose held-out same-state residual distribution has a 95th percentile well above 0.05 (e.g., due to anti-aliasing or font-stack differences), the fixed threshold declares same-state pairs as transitions, inflating partition size and corrupting the certificate. Our method instead uses the calibrated per-app/per-region threshold, so the false-positive rate on same-state pairs is held at 5%. We expect the fixed-threshold ablation to show a higher alias-warning rate and a lower certificate fraction on apps with high render noise, with near-parity on apps whose residual distribution is tightly concentrated below 0.05.

**vs. GRASP without OOD gating (ablation)** — On an app with live content (e.g., a data table with timestamps), the no-gating ablation routes unseen content through the residual rule, producing residuals that reflect renderer capacity rather than state change, and thereby corrupting the certificate for every pair involving that content. Our method instead marks such observations UNCERTIFIABLE, so the certificate denominator shrinks but its numerator stays sound. We expect the no-gating ablation to show a higher certified-pair count but a lower certificate fraction, with the degradation concentrated in apps whose probe set under-covers content perturbations.

**Calibration experiment (held-out perturbation families)** — When dark theme, 150% zoom, or live table updates are held out from the probe set, the residual distribution on held-out same-state pairs shifts away from `D_null` (measured by KL divergence), and the calibrated threshold `tau_render(app, region)` no longer holds its 5% false-positive rate. We expect certificate fraction to degrade monotonically with the KL divergence between held-out and calibration residual distributions, and we report this degradation as an empirical bound on the probe-coverage assumption.

### What would falsify this idea

If the contrastive-encoder and pixel-hash baselines achieve certificate fractions statistically indistinguishable from GRASP on the alias-heavy and separation-heavy subsets, then the perturbation-explicit renderer is not what produces identifiability and the central claim is wrong. Likewise, if the no-ALIAS_WARNING ablation, the fixed-threshold ablation, or the no-OOD-gating ablation matches GRASP's certificate fraction, the corresponding machinery is not load-bearing and the design rationale collapses.

## References

1. Work Agents: Systems, Capabilities, and Optimization
2. ScienceAgentBench: Toward Rigorous Assessment of Language Agents for Data-Driven Scientific Discovery
3. SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering
