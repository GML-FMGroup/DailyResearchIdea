# Counterfactual Edit Attribution for Prompt-Enhancer Supervision

## Motivation

Prompt-enhancer (PE) training treats every sentence deleted from a dense caption as redundant supervision, but P2E's own counterintuitive finding — that high-gain and low-gain caption pairs train comparable enhancers — shows the supervision signal is confounded: deletions that preserve user intent and deletions that merely strip incidental image-specific detail are pooled under one heuristic redundancy score. The root cause is structural: P2E selects deletions by a text-only redundancy heuristic with no interventional link to generation quality, so the trained enhancer cannot distinguish edits that causally drive gains from edits that are merely correlated with caption verbosity. The same confound recurs in online prompt optimization (E1) and in prompt-difficulty prediction (E3), where proxy signals (likelihood, caption redundancy, human difficulty labels) stand in for causal contribution without identifiability.

## Key Insight

If each pruned-caption SFT pair is paired with a counterfactual regeneration under a fixed seed and the deletion's effect on generation quality is isolated by a within-pair contrast that holds the base prompt and seed fixed, then the intent-preserving vs. incidental distinction becomes an identifiable within-pair contrast rather than a cross-example correlation, because the fixed-seed regeneration removes the sampling noise that makes naive gain comparisons unidentifiable.

## Method

**What it is.** CEASE (Counterfactual Edit Attribution for Supervision Enhancement) takes P2E-style dense-caption SFT pairs and, before training the prompt enhancer, assigns each candidate sentence deletion a counterfactual attribution score computed from paired generations under K=5 fixed seeds. It outputs a filtered SFT set in which only deletions whose attribution exceeds a calibrated threshold are retained as supervision; the enhancer is then trained on this filtered set with unchanged architecture and hyperparameters.

**How it works.**

```pseudocode
Input: dense captions C = {c_i}, short user prompts u_i, frozen T2I model G (e.g., Stable Diffusion 1.5 with CFG scale 7.5), quality scorer Q = 0.5*CLIP-Score(x,u_i) + 0.5*aesthetic_predictor(x), seed set S = {s_1,...,s_K} with K=5, deletion bound B=3, attribution threshold tau, noise floor epsilon
Output: filtered SFT pairs D_filtered; trained prompt enhancer PE

Phase 0 — sampler noise floor estimation
for each of 50 randomly sampled (c_i, u_i):
    for each seed s in S:
        x1 = G(c_i, seed=s); x2 = G(c_i, seed=s)   # caption-identical re-generations
        delta_noise = |Q(x1,u_i) - Q(x2,u_i)|
    epsilon = 95th percentile of all delta_noise values   # within-seed noise floor

Phase 1 — candidate deletion enumeration (greedy/beam, not exhaustive)
for each (c_i, u_i):
    S_i = all sentences in c_i not containing explicit user requirements (token overlap with u_i < 0.5)
    candidates = all single-sentence deletions d with |d|=1 from S_i
    plus beam-search extensions: iteratively add one sentence to d if |d|<B and the partial attribution (computed in Phase 2) improves
    record candidate edit (c_i, u_i, d, c_i^d)

Phase 2 — counterfactual attribution (multi-seed paired generation)
for each candidate edit:
    a_seeds = []
    for each seed s in S:
        x_base = G(c_i,   seed=s)
        x_edit = G(c_i^d, seed=s)
        q_base = Q(x_base, u_i)
        q_edit = Q(x_edit, u_i)
        a_seeds.append(q_edit - q_base)
    a(d) = mean(a_seeds)
    sigma(d) = std(a_seeds)
    # monotonicity check across guidance scales w in {5.0, 7.5, 10.0}
    for each w:
        compute a_w(d) using G with CFG scale w and seed s_1
    if sign(a_w(d)) not consistent across all w: mark d as guidance-unstable and set a(d) = -inf
    # noise floor gate
    if |a(d)| < epsilon: set a(d) = 0

Phase 3 — identifiability filtering
for each (c_i, u_i):
    d* = argmax_{d} a(d) subject to |d| <= B and d not guidance-unstable
    if a(d*) >= tau and sigma(d*) < 0.5*epsilon:   # stable above threshold
        D_filtered += (u_i -> c_i^{d*})
    else:
        D_filtered += (u_i -> c_i)   # fall back to unpruned caption

Phase 4 — enhancer training
PE = SFT(D_filtered; identical config to P2E baseline: same architecture, learning rate, batch size, epochs)
return PE
```

**Why this design.** We chose fixed-seed paired generation over independent-sample gain comparisons because P2E's own high-gain/low-gain equivalence finding (E2) is exactly what independent sampling produces when seed variance dominates the edit effect; fixing the seed makes the deletion the only varying factor, accepting the cost that the attribution is conditional on one seed and may miss edits whose effect is seed-dependent. We chose a per-example argmax with a global threshold tau over per-edit regression because the enhancer consumes one pruned caption per user prompt, not a per-sentence weight vector, accepting the cost that secondary useful deletions are discarded. We chose a CLIP-Score plus aesthetic-predictor composite Q over a learned reward model because a learned reward introduces its own confound and would make the attribution circular with respect to the enhancer's training objective, accepting the cost that Q is a coarser proxy than a human-preference reward. A domain expert would not describe CEASE as a variant of P2E: P2E prunes by a text-only redundancy heuristic with no interventional signal, whereas CEASE's supervision is selected by a within-pair counterfactual contrast against a frozen generator — the deletion is the intervention, not the descriptor.

**Why it measures what we claim.** The quantity a(d) = mean_{s in S} [Q(G(c_i^d, s), u_i) − Q(G(c_i, s), u_i)] measures the causal contribution of deletion d to generation-quality gain because the fixed seed set S and the within-seed noise floor epsilon make G(c_i^d, s) and G(c_i, s) differ only through the caption edit under the named assumption that the diffusion sampler is deterministic given (caption, seed) and that Q is monotone in the quality dimensions the enhancer is trained to improve; this assumption fails when the sampler's guidance scale amplifies small caption differences non-monotonically, in which case a(d) reflects sampler sensitivity rather than intent-preserving content removal. The threshold tau measures the boundary between intent-preserving and incidental deletions because it is calibrated on a held-out split by maximizing agreement with human preference judgments on the same pairs under the named assumption that human preference is a stable target for the quality dimension Q approximates; this assumption fails when the held-out prompts are drawn from a different verbosity distribution, in which case tau reflects the calibration split's redundancy level rather than a transferable intent boundary. The filtered SFT pair (u_i → c_i^{d*}) measures intent-preserving supervision because it retains only the deletion with maximal a(d) above tau under the named assumption that the single highest-attribution deletion carries the dominant intent-preserving signal; this assumption fails when two deletions are jointly necessary and individually low-attribution, in which case the pair reflects a partial edit rather than the full intent-preserving transformation.

## Contribution

['A counterfactual edit-attribution layer (CEASE) that converts P2E-style dense-caption pruning from a text-only redundancy heuristic into a fixed-seed interventional selection over candidate deletions, producing a filtered SFT set for prompt-enhancer training.', "An empirical design principle: supervision for prompt enhancers should be selected by within-pair counterfactual contrast against a frozen generator rather than by cross-example gain correlation, because the latter is confounded by seed variance — the mechanism behind P2E's high-gain/low-gain equivalence.", 'A calibrated attribution threshold procedure on a held-out prompt split that separates intent-preserving from incidental deletions without requiring a learned reward model.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Caption-derived SFT pairs with held-out human-preference split (512 calibration examples, 2,048 test prompts) | Matches P2E supervision source; enables tau calibration |
| Primary metric | Human-preference alignment on held-out prompts (pairwise win rate vs. baseline) | Directly targets the intent-preserving claim |
| Baseline | P2E text-only redundancy pruning | Tests whether interventional signal beats descriptor heuristic |
| Baseline | Conventional PE on unpruned caption pairs | Tests whether filtering helps at all |
| Baseline | Random-deletion SFT with matched deletion bound (B=3) | Controls for deletion volume, isolates attribution |
| Baseline | Likelihood-objective prompt optimization (E1-style) adapted to SFT filtering | Tests against existing interventional/attribution method |
| Ablation-of-ours | CEASE with shuffled attribution scores | Breaks causal ordering while preserving pipeline |
| Ablation-of-ours | CEASE with single-seed attribution (K=1) | Tests whether multi-seed averaging is necessary |
| Ablation-of-ours | CEASE without monotonicity check | Tests whether guidance-stability gating matters |

### Why this setup validates the claim

The central claim is that supervision selected by a within-pair counterfactual contrast yields better intent-preserving prompt enhancers than supervision selected by text-only redundancy. The conventional-PE baseline tests whether pruning matters at all; if CEASE only matches it, the attribution machinery is inert. The P2E baseline is the sharp test: both methods prune under the same deletion bound, so any gap must come from the selection signal, not from deletion volume. The random-deletion baseline controls for the possibility that deleting anything helps; if random deletion matches CEASE, the attribution score is not doing the work. The likelihood-objective baseline (E1-style) tests whether CEASE's fixed-seed contrast yields a distinct, non-redundant gain over an existing interventional method. The shuffled-attribution ablation preserves the full pipeline but destroys the causal ordering of a(d), isolating whether the contrast itself, rather than the filtering step, produces the gain. The single-seed ablation tests whether multi-seed averaging is necessary to overcome sampler noise. The no-monotonicity-check ablation tests whether guidance-stability gating removes spurious attributions. Human-preference alignment is the right primary metric because the claim is about intent preservation and compositional quality, which CLIP-Score alone does not capture; it is also the metric P2E reports, making the comparison direct. Together these form a falsifiable test: the predicted pattern is a gap over P2E, random deletion, and likelihood-objective baseline, with the shuffled and single-seed ablations collapsing toward P2E.

### Expected outcome and causal chain

**vs. P2E** — On a case where a dense caption contains a visually vivid but intent-irrelevant sentence (e.g., a specific lighting descriptor the user never asked for), P2E's text-only redundancy heuristic may retain it because it is not lexically redundant with other sentences, so the enhancer learns to reproduce incidental detail. CEASE instead measures that deleting this sentence raises Q(G(c^d,s),u) over the paired baseline generation, assigns it high attribution, and prunes it, so the SFT pair drops the incidental detail. We expect a noticeable preference-alignment gap concentrated on prompts whose dense captions carry high-attribution non-redundant detail, with parity on captions where redundancy and attribution agree.

**vs. Conventional PE on unpruned pairs** — On a case where a dense caption is verbose with several image-specific details weakly tied to user intent, conventional PE trains the enhancer to imitate all of them, producing over-specified outputs that drift from the user prompt. CEASE filters to the single highest-attribution deletion above tau, so the enhancer sees a tighter target. We expect conventional PE to underperform broadly on verbosity-heavy prompts, while CEASE shows a smaller but consistent advantage that shrinks on short, already-terse captions.

**vs. Random-deletion SFT** — On a case where a caption contains one genuinely load-bearing sentence and several incidental ones, random deletion under the same bound sometimes removes the load-bearing sentence, degrading the SFT pair and teaching the enhancer to drop intent-relevant content. CEASE's argmax attribution avoids this because the load-bearing deletion scores low or negative, so it is not selected. We expect random deletion to show erratic per-prompt quality with a heavy lower tail, while CEASE shows a tighter distribution and a clear mean advantage on prompts with mixed load-bearing and incidental content.

**vs. Likelihood-objective prompt optimization (E1-style)** — On a case where the dense caption contains a sentence that is likely under the model but not causally relevant to user intent, the likelihood objective may assign high score to it because it is fluent and common, whereas CEASE's fixed-seed contrast measures its actual effect on generation quality. We expect CEASE to outperform the likelihood baseline on prompts where fluency and causal relevance diverge, with parity on prompts where they align.

### What would falsify this idea

If CEASE's gain over P2E is uniform across prompts rather than concentrated on cases where text-only redundancy and counterfactual attribution disagree, or if the shuffled-attribution ablation matches CEASE, or if the single-seed ablation matches multi-seed CEASE, then the counterfactual contrast is not the causal driver and the central claim is wrong.

## References

1. Prune to Expand: Training Prompt Enhancers with Less to Imitate
