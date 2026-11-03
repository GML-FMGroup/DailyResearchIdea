# Monotone Corroboration Coupling for Falsifiable Event Boundary Detection in Video Captioning

## Motivation

Existing structural-decision pipelines treat a single channel as sufficient evidence: SBS (Seeing Before Synthesizing) places boundaries purely from semantic variation in VLM-generated frame narratives, so a hallucinated or off-topic narrative can shift or create a boundary with nothing to challenge it, and CodeCap-Bench accepts code captions without an independent executable-semantics check. The meta-gap is that across these trees no branch establishes that a structural decision (boundary placement, representation acceptance) is corroborated by an independent modality, so single-channel noise is structurally indistinguishable from genuine event change. This work makes boundary placement falsifiable by requiring a semantic-change signal and an independent low-level visual-change signal to agree under a learned monotone coupling before any boundary is accepted.

## Key Insight

If a learned monotone map is trained so that the semantic-change signal can only be high when the independent optical-flow discontinuity signal is high (and vice versa) on non-boundary frames, then agreement becomes a necessary condition rather than a post-hoc fusion, so a boundary that survives the coupling is one that no single noisy channel could have placed alone.

## Method

**What it is.** We introduce MCDC (Monotone Corroboration for Dense Captioning), a boundary-acceptance mechanism that takes as input a video and its frame-level VLM narratives and outputs a set of accepted event boundaries, where each accepted boundary carries a corroboration score g = min(ŝ, v̂) obtained by passing the semantic-change signal s and the optical-flow discontinuity signal v through a learned monotone coupling before any threshold is applied.

**How it works.**

```pseudocode
Input: video frames F[1..T], VLM narratives N[1..T], window w=5
Output: accepted boundaries B with corroboration score g

# Phase 1: two independent channels
for t in 1..T:
    s[t] = cosine_dist( embed(N[t]), embed(N[t-1]) )      # semantic change
    v[t] = || mean_flow(F[t-w:t]) - mean_flow(F[t:t+w]) ||  # flow discontinuity
normalize s, v to [0,1] per video

# Phase 2: monotone coupling (isotonic regression, trained on NON-boundary frames only)
# Load-bearing assumption: f is a valid upper envelope of s given v on non-boundary frames,
# i.e., no event transition leaks into the fitting window. We make this explicit and verify it.
# Fitting set: frames t such that t is NOT within radius r=10 of any candidate peak (candidate peaks
# are local maxima of s+v with prominence > 0.3) AND t is in a held-out temporal split (every 5th
# second of video, disjoint from evaluation segments).
f = IsotonicRegression(increasing=True, out_of_bounds='clip')   # fitted on {v[t] -> s[t]} for t in fitting set
# Per-video monotonicity check: compute residuals r[t] = s[t] - f(v[t]) on held-out non-boundary frames
# (a separate 10% of non-boundary frames not used for fitting). Reject f if the 90th percentile of r on
# held-out non-boundary frames is >= the 10th percentile of r on candidate frames (i.e., if the residual
# distribution on non-boundary frames is not stochastically dominated by that on candidate frames).
# If rejected, fall back to a global f fitted on all videos' non-boundary frames.

# Phase 3: contrastive identifiability constraint
# positives: (s[t], v[t]) at candidate peaks; negatives: temporally shuffled pairs
L_id = InfoNCE( proj(s[t], v[t]), proj(s[t'], v[t]) )  # t' != t, temperature tau_id=0.07, negative ratio 1:5

# Phase 4: corroboration-gated acceptance
g[t] = min( s_hat[t], v[t] )              # boundary only if BOTH channels high
B = { t : g[t] > tau (tau=0.6) and t is local max of g }
```

**Why this design.** We chose an isotonic (monotone) map fitted on non-boundary frames over a free-form learned fusion head because the structural claim is that semantic change must be *bounded above* by visual change away from boundaries, and isotonic regression encodes exactly that ordering constraint while a fusion MLP would let the head learn to ignore v when s is confident, accepting the cost that f is only as expressive as a monotone step function. We chose the min() corroboration gate over a weighted sum because min() is the unique aggregation under which a high value on one channel cannot compensate for a low value on the other, accepting the cost that genuine boundaries with weak flow (e.g., gradual dissolves) are missed. We chose the InfoNCE contrastive constraint over a simple correlation penalty because shuffling temporal pairs forces the coupling to be *identifiable* (the specific (s,v) pairing must match, not just co-vary), accepting the cost of extra negative sampling. A domain expert would not describe this as SBS with an added flow feature: SBS uses semantic variation alone and refines masks post-hoc, whereas here the flow channel is a *falsification gate* whose low value vetoes a semantic boundary, and the isotonic fit is trained to be violated only at true boundaries.

**Why it measures what we claim.** `g[t] = min(s_hat[t], v[t])` measures *corroborated boundary evidence* because the isotonic map f is assumed to be a valid upper envelope of semantic change on non-boundary frames, so any s exceeding f(v) signals a semantic change not explained by visual change; this assumption fails when the VLM narrative changes for reasons unrelated to visual content (e.g., a topic shift in narration over a static shot), in which case g reflects narrative drift rather than a true event boundary. The InfoNCE term measures *identifiability of the coupling* because shuffled temporal pairs are assumed to be non-boundary pairings, so a coupling that separates them has learned the boundary-specific joint distribution; this assumption fails when the video has near-periodic structure, in which case shuffled pairs resemble true pairs and the constraint reflects periodicity rather than identifiability. The isotonic fit on non-boundary frames measures *the null distribution of semantic change given visual change* because non-boundary frames are assumed to contain no event transition; this assumption fails at gradual transitions that leak into the fitting window, in which case f is biased upward and g under-reports those boundaries. To verify the upper-envelope assumption, we perform a coverage test: on held-out non-boundary frames, we check that the empirical 90th percentile of s given v is below f(v) for at least 95% of v-bins; we report the fraction of true boundaries that violate the envelope (s > f(v)) and the sensitivity to the exclusion radius r.

## Contribution

(1) A monotone-corroboration boundary-acceptance mechanism (MCDC) that makes semantic boundary placement falsifiable by an independent optical-flow channel through an isotonic coupling and a contrastive identifiability constraint, replacing single-channel thresholding. (2) A design principle: fitting the monotone coupling only on non-boundary frames turns the coupling into a null-model test, so a boundary is accepted only when it violates the learned null — a structural property post-hoc fusion cannot produce. (3) An analysis of when cross-modal corroboration fails (gradual dissolves, narrative drift over static shots) and how the isotonic null-model framing exposes these failure modes.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | ActivityNet Captions | Standard dense captioning with clear event boundaries. |
| Primary metric | Boundary localization F1 (tIoU=0.5) | Directly measures accepted boundary correctness. |
| Baseline | SBS (semantic-only) | Tests whether flow veto reduces false positives. |
| Baseline | Free-form fusion MLP | Tests whether monotone coupling is necessary. |
| Baseline | Weighted-sum gate | Tests whether min() corroboration is necessary. |
| Baseline | Learned non-compensatory fusion (monotone-constrained NN) | Tests whether isotonic+min is just one instance of a generic non-compensatory family. |
| Baseline | Probabilistic joint-density model (bivariate KDE) | Tests whether a density-based non-compensatory score matches the isotonic construction. |
| Ablation-of-ours | MCDC w/o InfoNCE | Tests identifiability constraint's contribution. |
| Ablation-of-ours | MCDC w/o per-video monotonicity check | Tests whether the calibration check is load-bearing. |

### Why this setup validates the claim

The central claim is that a monotone coupling between semantic change and optical-flow discontinuity, combined with a min() corroboration gate, yields fewer false-positive boundaries than semantic-only detection or unconstrained fusion. ActivityNet Captions provides a mix of abrupt cuts and gradual transitions, so the predicted failure modes (weak-flow boundaries, narrative drift) are actually present. SBS tests the sub-claim that the flow channel acts as a falsification gate: if SBS's false positives on static-shot narration are not reduced by our veto, the gate is not doing its job. The free-form fusion MLP tests whether the isotonic ordering constraint is load-bearing; if the MLP matches MCDC, the monotone assumption is unnecessary. The weighted-sum gate tests whether min()'s non-compensatory property matters; if a sum matches MCDC, corroboration is not the mechanism. The learned non-compensatory fusion and bivariate KDE baselines test whether the specific isotonic+min construction yields gains beyond any non-compensatory aggregator, addressing novelty. The ablation w/o InfoNCE tests whether identifiability of the (s,v) pairing adds anything beyond co-variation. The ablation w/o per-video monotonicity check tests whether the calibration check is necessary to avoid circularity. Boundary localization F1 at tIoU=0.5 is the right metric because it penalizes both missed boundaries and spurious ones, and the predicted effect is a precision gain on subsets where one channel is misleading, which F1 will surface as a gap concentrated on those subsets rather than a uniform lift. We also report the fraction of real-world false positives attributable to single-channel noise (by manual audit of 200 false positives) to establish the practical magnitude of the problem.

### Expected outcome and causal chain

**vs. SBS (semantic-only)** — On a case where the VLM narrative shifts topic over a static shot (e.g., narrator changes subject without a visual cut), SBS produces a false boundary because it uses semantic variation alone and has no visual falsification channel. Our method instead vetoes that candidate because the isotonic map f, fitted on non-boundary frames, predicts high s_hat only when v is high, so min(s_hat, v) stays low when flow is flat. We expect a noticeable precision gap on the static-shot-narration subset, with parity on abrupt-cut subsets where both channels agree.

**vs. Free-form fusion MLP** — On a case where a gradual dissolve produces moderate semantic change and moderate flow, the MLP fusion head learns to trust s when it is confident and can accept the boundary even though v is below the corroboration level, because its only training signal is the final boundary label and it can ignore v. Our method instead enforces the monotone upper-envelope constraint via isotonic regression on non-boundary frames, so s exceeding f(v) is treated as unexplained and g is capped by v. We expect a recall gap on gradual dissolves for the MLP (it over-accepts) and a matching precision gain for MCDC on that subset, with parity on sharp cuts.

**vs. Weighted-sum gate** — On a case where semantic change is very high but flow is near zero (narrative drift), a weighted sum can still exceed tau because the high s term compensates for low v. Our method instead uses min(), which is non-compensatory, so a low v vetoes regardless of s. We expect a precision gap on the narrative-drift subset specifically, with no gap on subsets where both channels are jointly high, confirming that the gain is concentrated where the predicted failure mode occurs.

**vs. Learned non-compensatory fusion (monotone-constrained NN)** — On a case where the joint distribution of (s,v) is complex but still non-compensatory, a monotone-constrained NN might match MCDC if the isotonic ordering is not the key. We expect MCDC to outperform on subsets where the isotonic upper-envelope assumption holds (verified by coverage test), and to be comparable elsewhere, indicating that the specific isotonic constraint adds value beyond generic non-compensatory fusion.

**vs. Probabilistic joint-density model (bivariate KDE)** — On a case where the joint density of (s,v) is multimodal, a KDE might capture non-compensatory structure without the isotonic assumption. We expect MCDC to be competitive or better on subsets where the upper-envelope assumption is validated, and to underperform if the assumption is violated (e.g., gradual dissolves), highlighting the importance of the calibration check.

### What would falsify this idea

If MCDC's gain over SBS is uniform across all subsets rather than concentrated on the static-shot-narration and narrative-drift subsets where the flow veto is supposed to act, the central claim that the flow channel is a falsification gate is wrong. Likewise, if the free-form fusion MLP, weighted-sum gate, learned non-compensatory fusion, or bivariate KDE matches MCDC on those same subsets, then the monotone coupling and min() corroboration are not the causal mechanisms.

## References

1. Seeing Before Synthesizing: VLM-Guided Transition Event Discovery for Weakly-Supervised Dense Video Captioning
2. CodeCap-Bench: Can Code Serve as a Good Video Caption?
3. Implicit Location-Caption Alignment via Complementary Masking for Weakly-Supervised Dense Video Captioning
4. SmolVLM: Redefining small and efficient multimodal models
5. NVILA: Efficient Frontier Visual Language Models
