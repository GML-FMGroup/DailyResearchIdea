# ClosureCert: A Learned Composition-Invariance Certificate for RAG Component Signals

## Motivation

Across the RAG evaluation literature, each component signal is validated only inside its ancestor's frozen pipeline: M3/Dense X Retrieval validates proposition-level retrieval granularity only under a fixed passage-based reader, Self-RAG validates reflection-token critiques only under its own fixed retriever and chunker, and LASAGNA measures aggregate composition effects across 450 configurations but treats every component as a frozen black box whose signal validity is assumed invariant to recomposition. The structural root cause is that no component ever carries a certificate of its own validity, so when components are recomposed the field has no way to distinguish a genuine non-additive interaction from a component whose signal has silently stopped being a sufficient statistic for its downstream decision. The result is that LASAGNA's observed +47% predicted vs +17% measured gap is uninterpretable: it is an aggregate residual with no attribution to which component's validity broke.

## Key Insight

A component signal's validity under recomposition is not a property of the component alone but of the conditional distribution of its downstream decision given the other components' signals, so it can be certified by a learned operator trained to be invariant across recompositions that preserve that conditional (closure-compatible) and to flag recompositions that change it — a compositional invariance that no frozen black-box evaluation can express.

## Method

## ClosureCert: Learned Composition-Invariance Certificate for RAG Component Signals

**(A) What it is.** ClosureCert is a learned operator `C_θ` that takes as input a recomposition description `r = (e, k, c)` — embedding model `e`, chunker/proposition extractor `k`, reflection-token critic `c` — together with the three component signals produced on a probe set `P` (embedding relevance scores `s_e`, proposition factoid-validity labels `v_k`, reflection-token critiques `q_c`), and outputs a certificate vector `(z, p_break)` where `z ∈ R^d` is a composition-invariance embedding and `p_break ∈ [0,1]` is a calibrated probability that the tuple of signals is no longer jointly valid (i.e., at least one signal has ceased to be a sufficient statistic for its intended downstream decision).

**(B) How it works.**

```pseudocode
# Phase 1: Probe-set signal extraction (offline, once per recomposition)
for each recomposition r = (e, k, c) in LASAGNA-harness(R):
    for each probe item (x, y*, D) in P:            # D = gold evidence set
        s_e[x]  = cos(emb_e(x), emb_e(d)) for retrieved d in top-k_e(D)
        v_k[x]  = factoid_validity(k(D), y*)          # proposition-unit atomicity check
        q_c[x]  = reflection_token_critique(c, x, retrieved_context)
    # Phase 2: Downstream decision labels (the target of sufficiency)
    for each probe item:
        a_r[x] = downstream_decision(reader, s_e[x], v_k[x], q_c[x])  # e.g. answer/abstain

# Phase 3: Train the closure operator C_θ (contrastive + calibration)
Initialize θ; build positive/negative recomposition pairs:
  POS = pairs (r, r') where r' differs from r only in a component declared
        closure-compatible by the empirically estimated closure relation
        (see below), AND the conditional-preservation test passes
  NEG = pairs (r, r') where a component is swapped across families
        (e.g., proposition chunker -> passage chunker, Self-RAG critic -> no critic)
for epoch in 1..N_epochs (=200):
    for batch of (r, r', a_r, a_r') in POS ∪ NEG:
        z_r  = encoder_θ(r, s_e, v_k, q_c)            # 4-layer transformer, d=256
        z_r' = encoder_θ(r', s_e', v_k', q_c')
        L_inv  = ||z_r - z_r'||^2   for POS pairs       # invariance objective
        L_sep  = max(0, m - ||z_r - z_r'||^2) for NEG   # margin m=1.0
        # Sufficiency head: does z predict downstream decision agreement?
        p_break = sigmoid(MLP_θ(z_r))                  # 2-layer MLP, hidden 128
        L_cal  = BCE(p_break, 1[agreement(a_r, a_r') < τ])  # τ=0.5
        L = L_inv + λ_sep * L_sep + λ_cal * L_cal      # λ_sep=1.0, λ_cal=2.0
        θ ← θ - lr * ∇L                                # lr=1e-4, AdamW

# Phase 4: Certify a new recomposition r_new
z_new, p_break_new = C_θ(r_new, signals(P))
if p_break_new < 0.5:  emit CERTIFIED (signals jointly valid)
else:                  emit FLAGGED (which signal: argmax over per-signal
                       sufficiency-head ablations)
```

**Empirical closure relation (replaces hand-authored compatibility table).** For each candidate positive pair `(r, r')` that differs only in a component from the same family (same embedding family, same granularity class, same critic token set), we test conditional preservation on a held-out set of 512 probe items `P_held` (disjoint from the 2048 training probe items). The test computes the downstream decision agreement rate `agreement(a_r, a_r')` on `P_held` and admits the pair to POS only if a two-sided binomial test fails to reject `H0: agreement = 1` at significance level α=0.05 (i.e., the number of disagreements is not significantly greater than zero). Pairs failing this test are moved to NEG. This yields an estimated closure relation that is data-driven rather than declared.

**Per-pair conditional-divergence estimate.** For each admitted POS pair, we additionally compute a scalar `d_cond(r, r') = 1 - agreement(a_r, a_r')` on `P_held` and store it as a target for an auxiliary regression head `d_hat = MLP_θ(z_r, z_r')` (2-layer MLP, hidden 128). The training loss adds `L_div = MSE(d_hat, d_cond)` with weight λ_div=0.5. At certification time, the operator outputs `d_hat` alongside `p_break` as a measured conditional-divergence estimate, grounding the invariance claim in a quantity computed from held-out downstream decisions rather than only in the learned distance.

**(C) Why this design.** We chose a contrastive invariance objective over a supervised break-prediction classifier because the LASAGNA harness yields far more closure-compatible recomposition pairs (same-family swaps) than labeled break events, and invariance is the property we actually need — a classifier trained only on breaks would memorize the 450 configurations rather than learn what invariance means, accepting the cost that we must estimate a closure relation (a design decision that trades generality for a well-defined positive set). We chose a per-signal ablation head (masking `s_e`, `v_k`, or `q_c` in turn) over a single scalar break score because the motivation requires attributing a break to a specific component signal, accepting the cost of three forward passes at certification time. We chose to condition the encoder on the recomposition description `r` as well as the signals, rather than on signals alone, because two recompositions can produce numerically similar signals while differing in whether the signal is a sufficient statistic — the description carries the structural context that disambiguates them, accepting the cost that the operator cannot certify a recomposition whose component families were never seen in training. This is deliberately not a variant of LASAGNA's aggregate composition measurement: LASAGNA regresses combined performance from configuration features, whereas ClosureCert learns an invariance map whose output is a per-component sufficiency certificate, and a domain expert would not describe an invariance-trained certificate as a performance regressor.

**(D) Why it measures what we claim.** The contrastive embedding distance `||z_r - z_r'||^2` measures closure-compatibility because we assume that recompositions admitted to POS by the empirical closure relation (binomial test on held-out downstream decision agreement) preserve the conditional distribution of the downstream decision given the component signals; this assumption fails when a same-family swap changes the signal's scale (e.g., a longer-context embedding model inflates relevance scores) and the binomial test has low power on 512 items, in which case the distance reflects nuisance scale shift rather than a genuine validity break. The calibrated break probability `p_break` measures joint sufficiency because we assume the probe-set downstream decision agreement `1[agreement(a_r, a_r') < τ]` is a faithful proxy for whether each signal remains a sufficient statistic; this assumption fails when the reader is itself robust to a broken signal (masking the break), in which case `p_break` reflects reader robustness rather than signal validity. The per-signal ablation attribution measures which component broke because we assume the sufficiency head's sensitivity to masking a signal is monotone in that signal's contribution to downstream agreement; this assumption fails under strong signal redundancy (e.g., the critic and the embedding both encode relevance), in which case attribution reflects redundancy rather than the broken component. The conditional-divergence estimate `d_hat` measures the degree of conditional-preservation violation because we assume the held-out agreement rate `1 - agreement(a_r, a_r')` is an unbiased estimate of the total variation distance between the downstream decision conditionals; this assumption fails when the held-out set is not representative of the probe distribution, in which case `d_hat` reflects sampling bias rather than conditional divergence.

## Contribution

['A learned closure operator (ClosureCert) that outputs a calibrated per-component sufficiency certificate for the tuple (embedding relevance, proposition factoid validity, reflection-token critique) under arbitrary RAG recomposition, trained via a compositional invariance objective over closure-compatible and break-inducing recomposition pairs.', 'A design principle: component-signal validity under recomposition is certifiable as an invariance property of the conditional downstream decision, not as a property of the component in isolation — operationalized through a pre-registered compatibility table and a per-signal ablation attribution head.', 'A method for attributing LASAGNA-style aggregate composition residuals to specific component-validity breaks, turning the predicted-vs-measured gap into a per-component diagnostic.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | LASAGNA 450-configuration recomposition sweep (8 benchmarks) | Provides closure-compatible and break pairs at scale |
| Primary metric | Per-signal sufficiency-break detection AUROC | Directly measures the certificate's discriminative claim |
| Baseline | LASAGNA additive-performance regressor | Tests whether invariance beats performance regression |
| Baseline | Signal-only break classifier (no recomposition description) | Tests whether structural context is load-bearing |
| Baseline | Single-scalar break score (no ablation attribution) | Tests whether per-signal attribution is needed |
| Ablation-of-ours | ClosureCert without contrastive invariance loss | Isolates contribution of invariance objective |
| Ablation-of-ours | ClosureCert with hand-authored compatibility table (no empirical closure relation) | Tests whether empirical closure relation is necessary |
| Ablation-of-ours | ClosureCert without conditional-divergence head | Tests whether measured divergence adds beyond learned distance |

### Why this setup validates the claim

The central claim is that a learned composition-invariance map, trained on closure-compatible recomposition pairs, yields a calibrated per-signal sufficiency certificate that a performance regressor cannot produce. Each baseline isolates one sub-claim. The LASAGNA additive regressor tests whether predicting combined performance from configuration features is sufficient — if it matches our certificate's break-detection AUROC, invariance learning adds nothing. The signal-only classifier tests whether conditioning on the recomposition description `r` is necessary, i.e., whether numerically similar signals from structurally different recompositions are genuinely indistinguishable. The single-scalar baseline tests whether per-signal ablation attribution carries information beyond a joint break score. The ablation-of-ours (no contrastive loss) tests whether the invariance objective, rather than the encoder capacity or calibration loss alone, produces the certificate. The ablation with hand-authored compatibility table tests whether the empirical closure relation (binomial test on held-out agreement) is necessary to avoid contaminated positive pairs. The ablation without conditional-divergence head tests whether the measured divergence estimate adds discriminative power beyond the learned embedding distance. AUROC is the right metric because the claim is discriminative — separating jointly-valid recompositions from broken ones — not regressional; a regressor can be accurate in aggregate while systematically missing the break events that matter. The falsifiable structure is: if all three baselines and the ablation match our AUROC on the break subsets, the invariance framing is doing no work.

### Expected outcome and causal chain

**vs. LASAGNA additive-performance regressor** — On a case where a proposition chunker is swapped for a passage chunker under a fixed embedding model, the additive regressor predicts combined performance from per-technique gains and therefore assigns the same predicted score to both granularities because its features do not encode whether `v_k` remains a sufficient statistic for the reader's answer/abstain decision. Our method instead places the two recompositions far apart in `z` because the contrastive objective was trained on exactly this cross-family negative, and the sufficiency head fires on the masked `v_k` ablation. We expect a noticeable AUROC gap on cross-family-swap subsets but parity on same-family swaps, where both methods should agree the recomposition is valid.

**vs. Signal-only break classifier (no recomposition description)** — On a case where a longer-context embedding model and a standard embedding model yield numerically similar `s_e` distributions on the probe set, the signal-only classifier sees near-identical inputs and outputs near-identical break probabilities, because it has no channel for the structural fact that the two embedding families have different score scales and different sufficiency relationships to the downstream decision. Our method conditions on `r`, so the encoder can separate the two despite signal similarity, and `p_break` diverges. We expect the gap to concentrate on the subset of pairs with high signal-space cosine similarity but cross-family component swaps — precisely the disambiguation case the description channel was added for.

**vs. Single-scalar break score (no ablation attribution)** — On a case where the critic `q_c` is broken (reflection tokens no longer track factuality) but the embedding and chunker remain valid, the scalar baseline correctly flags a break but cannot say which signal broke, so a downstream repair step must re-probe all three components. Our method's per-signal ablation head localizes the break to `q_c` because masking `q_c` collapses downstream agreement while masking `s_e` or `v_k` does not. We expect the scalar and our method to match on break-detection AUROC but for our method to show a clear attribution-accuracy advantage on single-component-break cases, with the advantage shrinking on multi-component breaks where redundancy confounds attribution.

**vs. ClosureCert with hand-authored compatibility table** — On a case where a same-family swap (e.g., two embedding models from the same family) actually changes the downstream conditional (as evidenced by a held-out agreement rate significantly below 1), the hand-authored table would admit this pair to POS, and the invariance loss would pull together representations of genuinely broken recompositions, corrupting `z` and `p_break`. Our method's empirical closure relation rejects this pair (binomial test p<0.05) and moves it to NEG, so the contrastive objective learns to separate it. We expect the hand-authored-table ablation to show lower AUROC on the subset of same-family swaps that are genuine breaks, while matching our method on truly compatible pairs.

**vs. ClosureCert without conditional-divergence head** — On a case where two recompositions have similar embedding distance but different degrees of conditional preservation (e.g., one pair has 2% disagreement, another has 10% disagreement on held-out items), the no-divergence-head model cannot distinguish them because its only output is the binary-calibrated `p_break`. Our method's auxiliary regression head predicts `d_hat` and provides a graded estimate. We expect our method to show better calibration (lower expected calibration error) on the continuous divergence prediction task, while both methods may match on binary break-detection AUROC.

### What would falsify this idea

If the signal-only classifier — which never sees the recomposition description — matches ClosureCert's break-detection AUROC across all subsets, then the structural-context channel is doing no work and the invariance framing collapses to ordinary signal monitoring. Similarly, if the no-contrastive-loss ablation matches full ClosureCert, the invariance objective is not what produces the certificate and the central mechanism claim is wrong. If the hand-authored-table ablation matches full ClosureCert, then the empirical closure relation is unnecessary and the positive-set contamination concern is not empirically material.

## References

1. Retrieval Augmented Generation Compositionality
2. Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection
3. Dense X Retrieval: What Retrieval Granularity Should We Use?
4. Few-shot Learning with Retrieval Augmented Language Models
5. Attributed Question Answering: Evaluation and Modeling for Attributed Large Language Models
6. When Not to Trust Language Models: Investigating Effectiveness of Parametric and Non-Parametric Memories
