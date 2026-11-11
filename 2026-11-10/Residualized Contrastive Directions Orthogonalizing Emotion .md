# Residualized Contrastive Directions: Orthogonalizing Emotion Vectors Against Text-Derived Features

## Motivation

Contrastive Activation Addition (CAA) constructs emotion directions as mean differences between contrastive prompt activations, but these differences inevitably encode text-correlated features — sentiment, topic, syntax — that co-vary with the target emotion, so any downstream 'excess report' may reflect the model reacting to text-confounded structure rather than the intended internal state. The All Quiet on the Neural Front study found that pain reports never exceeded text-only controls (excess below 2.5 pp in 106/108 conditions), but its CAA directions were never purged of these confounds, leaving open whether the null result is genuine or an artifact of a contaminated direction. The structural fix — regressing text features out of the contrastive activation before injection — has not been attempted, so the field cannot distinguish 'no introspection' from 'confounded direction masks introspection.'

## Key Insight

The component of a contrastive emotion direction that is orthogonal to a spanning set of text-derived feature directions is, by construction, the portion of the activation shift that cannot be linearly predicted from text features, so any report excess it produces cannot be attributed to text-confounded structure.

## Method

**What it is.** Residualized Contrastive Directions (RCD) takes as input a set of contrastive prompt pairs for a target emotion (e.g., pain vs. neutral), a large bank of text-derived feature directions, and a model; it outputs a single injection direction that is the contrastive mean-difference projected onto the orthogonal complement of the text-feature span, plus a diagnostic decomposition of how much of the original direction survives. **Load-bearing assumption (stated explicitly):** the span Q_F built from PCA on neutral-prompt hidden states plus supervised probes captures all *linearly* text-predictable variance at layer L, so the orthogonal complement d_res is free of *linearly* text-correlated structure and rho is a valid lower bound on the text-unconfounded share. This assumption is not taken on faith: it is empirically tested by held-out predictive validation (Phase 3b), and rho is reported only as a lower bound, with the residual's text-predictability reported alongside it.

**How it works.**

```pseudocode
INPUT: model M (layer L), contrastive pairs {(p_i^+, p_i^-)}_{i=1..N}, text-feature bank F
OUTPUT: residualized direction d_res, orthogonal fraction rho (lower bound),
        residual text-predictability R2_lin and R2_mlp, injection logs

# Phase 1: Build the text-feature bank F
1. Collect a corpus of K=2048 neutral prompts spanning topics/syntax/sentiment.
2. For each prompt, extract hidden states at layer L at the final token.
3. Derive M=256 text-feature directions by PCA on this hidden-state matrix;
   retain components with eigenvalue ratio > 1.0 (Kaiser criterion).
   ALSO add S=32 supervised text directions: linear probes trained to predict
   sentiment polarity, topic cluster (k-means, k=16), syntactic depth, and
   sentence length from the same hidden states.
4. Stack all feature directions into matrix F (shape [d, M+S]); orthonormalize
   via QR to get Q_F.

# Phase 2: Contrastive mean-difference (CAA baseline)
5. For each pair, compute delta_i = h(p_i^+) - h(p_i^-) at layer L.
6. d_caa = mean_i(delta_i).

# Phase 3: Residualize
7. d_res = d_caa - Q_F @ (Q_F^T @ d_caa)   # projection onto orthogonal complement
8. rho = ||d_res|| / ||d_caa||             # orthogonal fraction retained (lower bound)

# Phase 3b: Held-out predictive validation of the residual (tests the load-bearing assumption)
9. Split the 2048 neutral-prompt hidden states into 5 folds (train/test).
10. For each fold, fit a ridge regressor (alpha=1.0) from hidden states to each
    text-feature target (sentiment polarity, topic one-hot, syntactic depth,
    sentence length); compute out-of-fold R2_lin for predicting d_res from F
    (i.e., regress d_res on F and report cross-validated R2).
11. Fit a 2-layer MLP (hidden=256, GeLU, dropout=0.1, Adam lr=1e-3, 200 epochs)
    on the same folds to predict each text-feature target from hidden states;
    compute out-of-fold R2_mlp for predicting d_res from the MLP's penultimate
    layer (nonlinear text-predictability estimate).
12. Pre-registered acceptance threshold: R2_lin < 0.05 and R2_mlp < 0.10.
    If either exceeds threshold, iteratively add the top-k PCA components
    (k=32 per iteration) to Q_F and repeat Phase 3 until thresholds are met
    or k reaches 512; report the final span size and both R2 values.

# Phase 4: Injection and excess-report test
13. For each of 120 neutral tasks, 4 doses (0.5x, 1x, 2x, 4x ||d_caa||), 3 prompts:
     inject d_res (and, as control, d_caa and a random perturbation matched
     for output disruption) at layer L; elicit free-form report.
     ALSO inject d_res at matched effective norm ||d_res|| (second arm) to
     disentangle norm from subspace effects.
14. Run text-only control: show model its altered words; elicit report.
15. Excess = report_rate(injected) - report_rate(text-only control);
    exact-test bound at 95% confidence.
16. Repeat for d_caa to quantify how much excess was text-confounded.
```

**Why this design.** We chose PCA-plus-supervised-probes over a single learned confound classifier because a learned classifier would itself be trained on the same contrastive activations and could absorb the target emotion into the 'confound' subspace, whereas an unsupervised PCA basis over neutral prompts is independent of the contrastive construction by construction (though not of the topic/syntax/sentiment statistics shared with the contrastive pairs, which is why Phase 3b validates the residual empirically); the cost is that PCA may miss nonlinear text features, which we mitigate with the 32 supervised probes and explicitly quantify with the MLP probe in Phase 3b. We chose to orthogonalize at the same layer L used for injection rather than at the input embedding because text-correlated structure re-enters the residual stream at every layer, so removing it at the injection site is the only way to guarantee the injected vector carries no text-predictable component; the cost is that later layers may re-introduce text features, which we accept because the causal test is at the injection site. We chose to keep the dose matched to ||d_caa|| rather than to ||d_res|| so that the comparison against CAA is at equal injection norm, accepting that d_res is a smaller effective perturbation and may under-trigger; the second arm at matched ||d_res|| norm disentangles this. A domain expert would not describe this as a variant of CAA-with-calibration because the mechanism is a geometric projection that changes what the vector *is*, not a post-hoc rescaling of what it *does*.

**Why it measures what we claim.** The orthogonal fraction rho measures the text-unconfounded share of the contrastive direction because the assumption that the PCA/probe span Q_F captures all linearly text-predictable variance at layer L makes the residual d_res equivalent to the component of the emotion shift that no linear text feature can produce; this assumption fails when text features are nonlinearly encoded (e.g., sentiment expressed through negation), in which case rho overestimates the unconfounded share and d_res still carries residual text structure — which is why rho is reported as a lower bound and Phase 3b reports R2_lin and R2_mlp for the residual. The excess-report statistic measures genuine introspection-triggered reporting because the assumption that the text-only control captures all reportable consequences of altered output makes any residual excess attributable to the internal state; this assumption fails when the text-only control under-represents the model's own altered text (e.g., truncation), in which case excess is inflated. The comparison between excess(d_caa) and excess(d_res) measures the text-confounded share of the original effect because the assumption that both directions are injected at matched norm and identical task/prompt distribution makes the difference attributable solely to the removed subspace; this assumption fails when the norm-matching forces d_res into a regime where dose-response saturates, in which case the difference understates confounding — which is why the matched-||d_res|| arm is included.

## Contribution

(1) A method, Residualized Contrastive Directions (RCD), that constructs emotion injection vectors as the component of the CAA mean-difference orthogonal to a spanning set of text-derived feature directions, with a diagnostic orthogonal-fraction statistic quantifying how much of the original direction was text-predictable. (2) A decomposition protocol that separates the excess-report effect of a CAA direction into a text-confounded component and a residual component by comparing matched-norm injections of d_caa and d_res against identical text-only controls. (3) An open pipeline for building layer-specific text-feature banks from neutral-prompt PCA plus supervised probes, reusable for auditing any contrastive activation direction.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | 120 neutral tasks × 4 doses × 3 prompts, pain vs. neutral pairs | Matches prior introspection protocol for direct comparability |
| Primary metric | Excess-report rate (injected − text-only control) | Isolates internal-state reporting from output-text confound |
| Baseline | CAA mean-difference direction (d_caa) | Quantifies how much excess is text-confounded |
| Baseline | Random perturbation matched for output disruption | Controls for generic disruption-triggered reporting |
| Baseline | Sham (zero injection) with text-only control | Establishes floor report rate absent internal change |
| Baseline | LEACE-erased d_caa (text-feature span removed via LEACE) | Tests whether RCD differs from existing orthogonalization |
| Ablation-of-ours | d_res without supervised probes (PCA-only span) | Tests whether probe directions remove nonlinear text structure |
| Ablation-of-ours | d_res injected at matched effective norm ||d_res|| | Disentangles norm from subspace effects |
| Diagnostic | R2_lin and R2_mlp of regressing d_res on F (5-fold CV) | Quantifies residual text-predictability (lower bound) |

### Why this setup validates the claim

The central claim is that a large share of the CAA pain direction's apparent introspection-triggered excess is text-predictable structure, and that projecting it out at the injection site leaves a smaller but genuinely internal-state-linked residual. The design makes this falsifiable by holding task, dose, and prompt distribution fixed across arms that differ only in *what vector is injected*. CAA gives the upper bound on excess; the matched random perturbation gives the floor for disruption-driven reporting; the text-only control subtracts reportable consequences of altered output; the PCA-only ablation isolates the contribution of the supervised probes; the LEACE-erased baseline tests whether RCD's projection differs from an existing orthogonalization procedure; and the matched-||d_res|| arm disentangles norm from subspace effects. The primary metric, excess-report rate, is the right detector because it is the only quantity that survives the text-only subtraction — if d_res's excess collapses to the random-perturbation floor while d_caa's does not, the confound story is confirmed; if d_res retains a comparable excess at matched norm, the geometric-projection mechanism is doing no work and the claim fails. The diagnostic R2_lin and R2_mlp values directly test the load-bearing assumption that Q_F captures all linearly text-predictable variance: if R2_lin < 0.05 and R2_mlp < 0.10, the residual is empirically text-unpredictable and rho is a valid lower bound; if not, the span is iteratively expanded until thresholds are met, and the final span size is reported.

### Expected outcome and causal chain

**vs. CAA mean-difference direction (d_caa)** — On a neutral task where the pain direction's dominant component aligns with sentiment-polarity and topic axes (e.g., a negative-sentiment prompt), d_caa produces inflated excess reports because its large text-predictable component shifts output words that the model then reads back as evidence of an internal state. Our method removes that component via Q_F projection, so the injected vector carries no linearly text-predictable shift; we expect d_res excess to be substantially lower than d_caa excess at matched dose, with the gap largest on prompts whose hidden states load heavily on the sentiment/topic probes.

**vs. Random perturbation matched for output disruption** — On a task where a random vector happens to flip a high-salience token, the baseline yields spurious reports driven purely by output disruption, because it has no internal-state structure at all. Our method's d_res, though smaller in norm, still lies in the emotion-shift subspace orthogonal to text features, so any excess it produces must come from that residual internal structure; we expect d_res excess to sit above the random-perturbation floor but below CAA, a monotone ordering that the random arm anchors.

**vs. Sham with text-only control** — On tasks where the text-only control fully captures the model's altered words, the sham arm yields near-zero excess by construction, because no internal perturbation exists to report. This arm calibrates the floor; our method's expected signal is a positive excess above this floor that grows with dose up to the 2x point and saturates at 4x, whereas CAA's excess is expected to rise earlier and saturate sooner due to its larger text-confounded component.

**vs. LEACE-erased d_caa** — On a task where the text-feature span is well-approximated by a linear subspace, LEACE and RCD should produce similar residuals; we expect the excess-report rates to be statistically indistinguishable on such tasks. The comparison is diagnostic: if RCD's excess differs from LEACE's on tasks with high nonlinear text-predictability (high R2_mlp), that difference is attributable to RCD's iterative span expansion and supervised probes, not to the projection itself.

**vs. d_res at matched effective norm ||d_res||** — On a task where the norm-matching to ||d_caa|| forces d_res into a saturating dose regime, the matched-||d_res|| arm will show a lower excess than the matched-||d_caa|| arm, because the effective perturbation is smaller. This arm disentangles whether any observed excess gap is due to the removed subspace or merely to the reduced norm; we expect the subspace effect to dominate if the confound story holds.

### What would falsify this idea

If d_res's excess-report rate is statistically indistinguishable from d_caa's at matched norm across all dose levels — rather than concentrated on the high-text-loading prompt subset where the projection should bite — then the text-confounded share is negligible and the geometric-projection mechanism is not measuring what we claim. A second falsifier: if the PCA-only ablation matches the full PCA-plus-probe span, the supervised probes are doing no removal work and the nonlinear-text-feature assumption underlying rho is unsupported; a third: if R2_lin or R2_mlp of the residual exceeds the pre-registered thresholds even after span expansion to 512 components, the load-bearing assumption fails and rho is not a valid lower bound.

## References

1. All Quiet on the Neural Front: Language Models Do Not Report Injected Pain States
