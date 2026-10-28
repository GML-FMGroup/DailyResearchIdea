# Falsifiable Prompt Enhancement via Dual-Path Condition Re-derivation for Interactive Video Generation

## Motivation

Across the interactive world-generation paper-tree, Hy-Nomas, Yume-1.5, and their siblings all inherit the same unchecked assumption: a VLM Prompt Enhancer can reliably translate multimodal interaction history into structured conditions, and no independent mechanism verifies that the mapping is well-posed. Because the generated video segment is a deterministic function of the condition plus the diffusion backbone, the mapping can be audited by inverting the segment, yet no prior work exploits this causal independence — leaving under-determined conditions silently mis-specified at generation time.

## Key Insight

A condition is only meaningful if it is the unique intersection of what the raw multimodal history implies and what the generated segment implies; divergence between these two causally independent derivations is a direct, calibration-free certificate of condition under-determination.

## Method

**What it is.** FALCON (Falsifiable Condition Oracle) wraps Hy-Nomas's Prompt Enhancer with a self-consistency audit: given an interaction history $H$ (raw multimodal stream) and a candidate structured condition $c$, it re-derives $c$ from two causally independent views — $H$ itself and the video segment $v = G(c)$ produced by the frozen diffusion backbone $G$ — and constrains generation to the intersection of surviving conditions. The load-bearing assumption is stated explicitly: **G is injective on the structured-condition space** — i.e., for every condition token type in the schema (camera parameters, counts, action verbs, object attributes), distinct values of that token produce distinguishable segments, so that a VLM reading $V(G(c))$ recovers $c$ and any divergence between $V(H)$ and $V(G(c))$ is attributable to condition under-determination rather than to the backbone's failure to render $c$. Because this assumption is known to fail for some token types (e.g., camera parameters outside training range, rare verbs), FALCON first runs a **renderability calibration** (step 0 below) that measures, per token type, whether $G$ actually renders that token, and restricts the divergence test to the subspace of tokens that pass the probe; tokens that fail are routed to an 'unrenderable' flag rather than the 'under-determined' flag.

**How it works.**
```pseudocode
# Inputs: history H (frames + actions + text), frozen enhancer E, frozen backbone G, frozen VLM V
# Hyperparams: K=8 (candidate conditions), tau=0.15 (divergence threshold), alpha=0.5 (intersection weight)
# Token schema T = {camera, count, action_verb, object_attr} (4 token types)
# Renderability probe set P: 64 synthetic (c, v=G(c)) pairs per token type, c drawn from schema

0. RENDERABILITY CALIBRATION (once, offline):
   for each token type t in T:
       acc_t = mean_{ (c,v) in P_t } [ V(v) recovers c_t ]   # VLM token-recovery accuracy
       renderable[t] = (acc_t >= 0.80)                        # 0.80 = probe threshold
   # tokens with renderable[t]=False are excluded from d(c); their divergence
   # is routed to 'unrenderable' flag, not 'under-determined' flag

1. c0 = E(H)                              # baseline condition from Hy-Nomas enhancer
2. C  = {c0} u {E(H) with temperature T=0.7}^(K-1)   # K candidate conditions
3. for each c in C:
4.     v_c = G(c, cached_history)         # deterministic segment (DDIM, eta=0)
5.     c_raw  = V(H)                       # view-1: re-derive from raw stream
6.     c_seg  = V(v_c)                     # view-2: re-derive from generated segment
7.     d(c)  = D_KL( c_raw || c_seg ) restricted to tokens t with renderable[t]=True
           # tokens with renderable[t]=False are masked out of the KL sum
8.     C* = { c in C : d(c) <= tau }        # survivors = intersection of both views
9. if C* empty:
       c* = argmin_c d(c)
       if any token in c* has renderable[t]=False: flag 'unrenderable'
       else: flag 'under-determined'         # rejected, not silently accepted
10. else: c* = argmin_{c in C*} [ alpha * d(c) + (1-alpha) * ||c - c0||_1 ]
11. generate final segment v* = G(c*, cached_history); append (c*, v*) to history
```

**Why this design.** We chose to re-derive from the *generated segment* rather than from a second VLM pass over $H$ because only the segment is a deterministic function of $c$ (DDIM with $\eta=0$), so $c_{seg}$ carries information about $c$ that is causally independent of the enhancer's own encoding of $H$; a second VLM pass would share the enhancer's inductive bias and could not falsify it, accepting the cost that the audit is only as well-posed as the backbone's invertibility — which is why step 0 measures that invertibility per token type and masks out tokens that fail the probe. We chose an *intersection over a candidate set* rather than a single-sample divergence because a single $c_0$ cannot distinguish 'the enhancer got it wrong' from 'the condition is under-determined' — the candidate set lets $C^*$ be empty precisely when no condition survives both views, which is the falsifiability signal; the cost is $K$ forward passes per segment, mitigated by caching and by $K=8$ being enough in practice because survivors cluster. We chose to *flag rather than repair* under-determined cases (step 9) instead of falling back to $c_0$, because silently repairing would reintroduce the exact unchecked assumption the oracle is meant to remove; the cost is that some generations are surfaced as failures rather than produced, which is the intended behavior. A domain expert would not describe FALCON as a variant of Hy-Nomas: Hy-Nomas's Prompt Enhancer is a one-pass translator, whereas FALCON makes the enhancer's mapping *falsifiable at generation time* by requiring a second, causally independent inverse to agree — a property the enhancer structurally cannot provide on its own.

**Why it measures what we claim.** The quantity $d(c) = D_{KL}(c_{raw} \| c_{seg})$ measures **condition under-determination** because the assumption that $G$ is injective on the condition space (DDIM with $\eta=0$ and causal attention makes the segment a deterministic, near-invertible function of $c$) implies that if two derivations of $c$ disagree, no single $c$ is consistent with both views — i.e., the condition is not identified by the data; this assumption fails when the backbone is not injective (e.g., a condition token that $G$ ignores, such as a camera parameter outside its training range), in which case $d(c)$ reflects the backbone's blind spot rather than the enhancer's under-determination, and the oracle will flag a condition that is actually well-specified but unrenderable — which is why step 0's renderability probe masks such tokens out of $d(c)$ and routes them to the 'unrenderable' flag. The survivor set $C^*$ measures the **intersection of causally independent implications** because the assumption that $H$ and $v_c$ are conditionally independent given $c$ (the segment is generated only from $c$ and cached history, not from $H$ directly) makes agreement between $V(H)$ and $V(v_c)$ evidence that $c$ is pinned down by both; this assumption fails when cached history leaks $H$ into $G$, in which case $c_{seg}$ is not independent of $c_{raw}$ and the audit degenerates to a consistency check rather than a falsification test. The flag in step 9 measures **falsifiability** because the assumption that an empty $C^*$ implies no condition in the candidate neighborhood satisfies both views makes 'empty' equivalent to 'the enhancer's mapping is not well-posed for this history'; this fails when $K$ is too small to cover the true condition, in which case 'empty' reflects candidate-set poverty rather than genuine under-determination.

## Contribution

["FALCON: a self-consistency oracle that audits a VLM Prompt Enhancer's action-to-condition mapping by requiring dual-path re-derivation (raw multimodal stream vs. generated video segment) and constraining generation to the intersection of surviving conditions, making the mapping falsifiable at generation time.", 'A design principle for interactive world models: condition under-determination, not VLM miscalibration, is the correct failure mode to detect, and it is detectable via divergence between two causally independent inverses of a deterministic diffusion backbone — sidestepping the log-prob calibration anti-pattern.', 'An audit protocol and flagging mechanism that surfaces under-determined conditions as explicit generation failures rather than silent mis-specifications, applicable to any Hy-Nomas-style pipeline with a deterministic backbone.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|-----------------------|
| Dataset | Hy-Nomas interactive episodes (held-out, 512 episodes) | Real multimodal histories with known condition structure. |
| Primary metric | Condition F1 on under-determined subset | Directly measures falsification of enhancer mapping. |
| Secondary metric | False-flag rate per token type (camera, count, action_verb, object_attr) | Bounds backbone blind-spot failure mode. |
| Baseline | Hy-Nomas Prompt Enhancer (single-pass) | Tests whether audit adds value over one-pass translation. |
| Baseline | Second VLM pass over H | Tests whether causal independence of segment view matters. |
| Baseline | FALCON with K=1 (single candidate c0) | Tests whether candidate set enables falsifiability signal. |
| Baseline | Naive cycle-consistency (re-derive c from G(c) only, no raw-history view) | Tests whether two-view intersection is the novel ingredient. |
| Ablation-of-ours | FALCON with tau=∞ (no divergence filter) | Isolates the effect of the intersection constraint. |
| Ablation-of-ours | FALCON without renderability calibration (step 0 disabled) | Isolates the effect of the injectivity probe. |

### Why this setup validates the claim

The central claim is that FALCON makes the enhancer's mapping falsifiable at generation time by requiring agreement between two causally independent inverses of the condition. The dataset (512 held-out Hy-Nomas episodes) is the right testbed because it contains real interaction histories where the enhancer can fail, and the condition structure is known enough to score correctness. The primary metric—condition F1 on the under-determined subset—isolates the cases where the enhancer is most likely to be wrong, so a gain there is direct evidence that the audit detects and rejects bad conditions. The secondary metric—false-flag rate per token type—directly bounds the load-bearing injectivity assumption: if a token type (e.g., camera) has high false-flag rate even on well-specified synthetic histories, the divergence test is measuring the backbone's blind spot, not under-determination, and the renderability calibration must mask that token. The single-pass Hy-Nomas baseline tests whether the audit adds anything beyond the enhancer; the second-VLM-pass baseline tests whether the segment view's causal independence is necessary (if a second pass over H works equally well, the design rationale is wrong); the K=1 baseline tests whether the candidate set is what enables the falsifiability signal (if K=1 matches K=8, the candidate set is unnecessary); the naive cycle-consistency baseline (re-derive c from G(c) only, no raw-history view) tests whether the two-view intersection is the novel ingredient (if it matches FALCON, the raw-history view is doing no work). The ablation (tau=∞) removes the divergence filter, so if performance matches full FALCON, the intersection constraint is doing no work; the ablation without renderability calibration tests whether step 0's masking is necessary to avoid false flags on unrenderable tokens. Together these form a falsifiable test: the predicted pattern is a gain concentrated on the under-determined subset, with parity elsewhere, degradation when the causal-independence or candidate-set components are removed, and a false-flag rate on well-specified synthetic histories that is low for renderable token types and high for unrenderable ones (confirming the need for step 0).

### Expected outcome and causal chain

**vs. Hy-Nomas Prompt Enhancer (single-pass)** — On a case where the history contains a rare action verb (e.g., "slide" vs. "push") that the enhancer maps to a plausible but wrong condition token, the baseline produces a segment that follows the wrong interaction because its one-pass translator has no internal check. Our method instead re-derives the condition from the generated segment and rejects the candidate when the two views diverge, so we expect a noticeable F1 gap on the under-determined subset but parity on well-specified episodes.

**vs. Second VLM pass over H** — On a case where the enhancer's inductive bias systematically misreads a multimodal cue (e.g., a gesture that looks like "grab" but is "release"), the second-pass baseline produces the same wrong condition because both passes share the enhancer's encoding bias, so its divergence is near zero and it accepts the error. Our method instead uses the generated segment as the second view, which is causally independent of the enhancer, so divergence is high and the candidate is rejected; we expect the second-pass baseline to show near-zero flag rate on these cases while FALCON flags them.

**vs. FALCON with K=1 (single candidate c0)** — On a case where the true condition is under-determined (e.g., ambiguous depth cue), the K=1 baseline produces either a silent acceptance of c0 or an arbitrary flag because it cannot distinguish "enhancer wrong" from "condition under-determined." Our method instead uses K=8 candidates so that an empty survivor set is a genuine falsifiability signal; we expect K=1 to show high false-flag rate on well-specified cases and low flag rate on truly under-determined cases, while FALCON separates the two.

**vs. Naive cycle-consistency (re-derive c from G(c) only, no raw-history view)** — On a case where the enhancer is wrong but the segment is self-consistent (e.g., a wrong action verb that G renders faithfully), the cycle-consistency baseline re-derives c from G(c) and finds agreement with itself, so it cannot flag the error because it lacks the raw-history view. Our method instead requires agreement between V(H) and V(G(c)), so the raw-history view exposes the divergence and the candidate is rejected; we expect the cycle-consistency baseline to show near-zero flag rate on enhancer-error cases while FALCON flags them, isolating the two-view intersection as the novel ingredient.

**vs. FALCON with tau=∞ (no divergence filter)** — On a case where the enhancer is wrong, the ablation produces the wrong condition because it never rejects candidates, so its F1 matches the single-pass baseline. Our method instead filters by divergence, so we expect the ablation to lose most of the gain on the under-determined subset, confirming that the intersection constraint is the active ingredient.

**vs. FALCON without renderability calibration (step 0 disabled)** — On a case where the condition is well-specified but contains an unrenderable token (e.g., a camera parameter outside G's training range), the no-calibration ablation computes d(c) over all tokens, so the unrenderable token's divergence triggers a false 'under-determined' flag even though the condition is correct. Our method instead masks unrenderable tokens out of d(c) and routes them to the 'unrenderable' flag, so we expect the no-calibration ablation to show a higher false-flag rate on well-specified episodes with unrenderable tokens, while FALCON's false-flag rate stays low for renderable token types and its 'unrenderable' flag rate rises for the unrenderable ones.

### What would falsify this idea

If FALCON's gain over the single-pass baseline is uniform across all episodes rather than concentrated on the under-determined subset, or if the second-VLM-pass baseline matches FALCON's flag rate on enhancer-error cases, then the causal-independence claim is wrong and the audit is not doing what we claim. Similarly, if the K=1 baseline matches FALCON's separation of under-determined from well-specified cases, the candidate-set mechanism is unnecessary; and if the renderability calibration's false-flag rate on well-specified synthetic histories is high for token types that pass the probe, the injectivity assumption is not salvaged by step 0 and the divergence metric does not certify under-determination.

## References

1. Hy-Nomas: Towards Free-form Interactive World Generation
2. Decoupled DMD: CFG Augmentation as the Spear, Distribution Matching as the Shield
3. Yume-1.5: A Text-Controlled Interactive World Generation Model
