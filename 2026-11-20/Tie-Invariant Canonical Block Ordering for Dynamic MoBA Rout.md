# Tie-Invariant Canonical Block Ordering for Dynamic MoBA Routing

## Motivation

LightningMoBA's dynamic routing thresholds per-query MoBA block scores to emit variable-cardinality sparse metadata, but because selection inherits MoBA's index-ordered scan, permuting equally-scored KV blocks can change which columns are gathered and therefore change the attention output. This is a structural selection-invariance gap shared across the MoBA lineage: the retained set is a function of score order, not of the score multiset, so identical score distributions yield different sparse patterns depending on arbitrary block indices. Closing it requires an assumption-level repair to the routing rule rather than a new architecture.

## Key Insight

If the threshold is applied to a canonical ordering keyed by the score multiset rather than by raw block index, the retained KV-block set becomes a function of the multiset of scores alone, making the gathered columns and attention output invariant to permutations of equally-scored blocks.

## Method

**What it is.** Tie-Invariant Canonical Routing (TICR) is a drop-in replacement for the thresholding step in LightningMoBA's dynamic routing. Its input is the per-query vector of MoBA block scores plus block content hashes; its output is the same variable-cardinality sparse metadata format LightningMoBA already emits, but the retained set is now a function of the score multiset alone.

**Load-bearing assumption (stated explicitly).** TICR's quality claim rests on the assumption that permuting equally-scored (within the 1e-4 quantization grid) KV blocks changes which blocks LightningMoBA's index-ordered threshold retains in a way that materially changes attention output and downstream RULER score, so that resolving these ties by content hash yields a measurable quality gain at matched retention. This assumption is load-bearing and is tested directly by the tie-frequency calibration and the value-spread oracle below; if it fails, TICR delivers determinism but not quality gain, and the contribution is reframed accordingly.

**How it works.**

```pseudocode
INPUT: scores s[1..B] for one query head, block hashes h[1..B], threshold tau
OUTPUT: retained block set R, variable-cardinality metadata

0. Calibrate delta on a held-out calibration set of 512 queries drawn from the
   target checkpoint and RULER subset: compute all adjacent score gaps
   |s[i]-s[j]| for pairs within the top-32 blocks, set delta = 10th percentile
   of these gaps (clipped to [1e-5, 1e-3]); default delta = 1e-4 if calibration
   is skipped.
1. Quantize each score to the calibrated grid: q[i] = round(s[i] / delta)
2. Build canonical key for each block: k[i] = (q[i], h[i])           # lexicographic
3. Sort blocks by k descending: order = argsort_desc(k)
4. Walk order; retain block order[j] while q[order[j]] >= tau
   (ties broken deterministically by h[i], never by block index)
5. Emit R = {order[0..m-1]} and per-query cardinality m as sparse metadata
6. Pass R to LightningMoBA's KV-centric ownership and pipelined execution
```

**Why this design.** We chose score quantization to a calibrated grid (default 1e-4, set from the 10th percentile of observed adjacent score gaps on 512 calibration queries) over raw floating-point comparison because exact float ties are rare in practice and a quantized grid makes the tie class explicit and hashable, accepting the cost that near-ties within delta are treated as ties. We chose a content hash as the secondary key over block index because index is exactly the arbitrary quantity whose permutation we want to neutralize; the cost is one hash per block, computed once at KV-cache construction. We chose to layer canonical ordering on top of LightningMoBA's threshold rather than replace it with fixed Top-k, because fixed Top-k reintroduces a hard budget that LightningMoBA's dynamic routing was designed to remove; the cost is that the retained cardinality remains workload-dependent. This differs from FlashMoBA's fixed Top-k and from LightningMoBA's index-ordered thresholding: neither prior method makes the retained set a function of the score multiset, so a domain expert would not describe TICR as a variant of either — it changes the selection rule's invariance class, not its budget policy.

**Why it measures what we claim.** The quantized score q[i] measures the tie-class membership of block i because the assumption that scores within delta are exchangeable makes q[i] equal to the tie class; this assumption fails when two blocks have genuinely different importance but fall within delta, in which case q[i] reflects a coarse importance bucket rather than a true tie. The content hash h[i] measures the canonical identifier of block i because the assumption that block content is unique makes h[i] a permutation-invariant label; this assumption fails when duplicate blocks exist, in which case h[i] collides and the secondary sort falls back to stable order, which is itself permutation-invariant. The retained set R measures the score-multiset-determined selection because the assumption that the sort key (q[i], h[i]) is a total order on the multiset makes R independent of input block ordering; this assumption fails when hashes collide and quantized scores tie simultaneously, in which case R reflects the stable-sort order of the input rather than the multiset alone. Crucially, the content hash is an arbitrary permutation-invariant label, not a semantic key: resolving ties by hash selects an arbitrary member of the tie class, not the higher-value block. TICR therefore claims deterministic, reproducible selection under permutation (bitwise-identical output for permuted equally-scored blocks), not a quality gain from tie-breaking; any quality gain would require the tie class to be value-homogeneous, which the value-spread oracle below tests directly.

## Contribution

["A tie-invariant canonical block-selection rule (TICR) that replaces raw per-query score thresholding with a two-stage quantize-then-canonically-order procedure, preserving LightningMoBA's variable-cardinality sparse metadata format.", 'An empirical invariance check showing that permuting equally-scored KV blocks leaves the gathered columns and attention output unchanged under TICR, whereas index-ordered thresholding changes them.', 'A design principle for the MoBA lineage: selection rules should be specified as functions of the score multiset, not of score order, to make sparse attention outputs reproducible across block layouts.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | RULER long-context benchmark | Stresses block selection under long contexts |
| Primary metric | Mean RULER score at matched retention | Isolates selection quality from budget |
| Secondary metric | Bitwise output-invariance rate under block permutation | Directly tests the invariance claim |
| Baseline | LightningMoBA (index-ordered thresholding) | Direct predecessor; tests invariance claim |
| Baseline | FlashMoBA (fixed Top-k) | Tests removal of hard budget |
| Baseline | Twilight (adaptive top-p) | Tests adaptive-budget alternative |
| Ablation-of-ours | TICR without content hash (score-only key) | Isolates permutation-invariance contribution |
| Calibration | 512 queries from target checkpoint and RULER subset | Sets delta from observed score gaps |
| Oracle | Value-spread within tie classes | Tests whether tied blocks are value-homogeneous |

### Why this setup validates the claim

The central claim is that making the retained block set a function of the score multiset alone — rather than of block index or a fixed budget — yields deterministic, reproducible selection under permutation, and that this invariance is measurable at matched retention. RULER is the right dataset because it exposes long-context retrieval and reasoning where block selection actually determines what the model can attend to, so differences in selection rule surface as score differences rather than being masked by short-context saturation. LightningMoBA is the critical baseline: it shares our dynamic thresholding and KV-centric execution, so any gap against it isolates the invariance change rather than confounding it with routing or scheduling. FlashMoBA tests the sub-claim that removing the fixed Top-k budget helps, and Twilight tests whether our gain is distinguishable from generic adaptivity. The score-only ablation is the sharpest test: if permutation invariance is the mechanism, dropping the content hash should collapse our advantage toward LightningMoBA's behavior, since ties would then resolve by input order. The bitwise output-invariance rate is the direct test of the invariance claim: for each query, we permute the KV blocks within each quantized tie class and check whether the attention output is bitwise identical; TICR should achieve 100% by construction, while LightningMoBA should not. The value-spread oracle measures whether tied blocks differ in importance: for each tie class, we compute the spread of block contribution scores (pooled dot-product estimates) and report the fraction of tie classes with spread above a threshold; if this fraction is low, the tie class is value-homogeneous and hash-based tie-breaking is value-neutral, supporting the reframed determinism claim rather than a quality claim. The calibration set of 512 queries sets delta from the 10th percentile of observed adjacent score gaps, replacing the a priori 1e-4 with a data-driven grid. Mean RULER at matched retention is the right metric because it holds the efficiency budget constant, so the comparison measures what we claim to improve — which blocks get kept — not how many.

### Expected outcome and causal chain

**vs. LightningMoBA (index-ordered thresholding)** — On a case where several blocks score within the quantization grid but carry different content, LightningMoBA's index-ordered walk retains whichever blocks happen to sit at favorable indices, so the kept set shifts under block reordering. Our method instead resolves the tie class by content hash, so the retained set depends only on the score multiset and is stable under permutation. We expect TICR to achieve a bitwise output-invariance rate of 100% under block permutation, versus a rate below 100% for LightningMoBA, with any RULER score difference concentrated on retrieval subsets where tied blocks are frequent and near-parity elsewhere. If the value-spread oracle shows tie classes are value-homogeneous, we expect near-zero RULER gap and attribute the contribution to determinism, not quality.

**vs. FlashMoBA (fixed Top-k)** — On a case where a query's score distribution is sharply peaked, fixed Top-16 forces retention of low-scoring blocks to fill the budget, diluting attention with irrelevant KV tiles. FlashMoBA's mechanism gap is the hard budget: it cannot shrink below k. Our method inherits LightningMoBA's threshold, so the retained cardinality contracts when few blocks clear tau, and the canonical ordering only decides among those that qualify. We expect our advantage to appear as a quality edge on peaked-distribution subsets while matching FlashMoBA on flat distributions where the budget is not binding.

**vs. Twilight (adaptive top-p)** — On a case where the score distribution has a long tail of near-equal mid-tier blocks, Twilight's nucleus criterion admits a large fraction of them because cumulative probability mass accumulates slowly, inflating the retained set and eroding speed. Twilight's gap is that its budget adapts to mass rather than to an explicit tie class. Our method's quantized threshold makes near-equal blocks an explicit tie class resolved deterministically by hash, so the retained set stays compact. We expect a modest score edge plus a retention reduction on tail-heavy subsets, with parity on sharply peaked ones.

### What would falsify this idea

If our gain over LightningMoBA is uniform across all RULER subsets rather than concentrated where tied or near-tied block scores are frequent, then permutation invariance is not the operative mechanism and the central claim is wrong. Likewise, if the score-only ablation matches our full method, the content hash contributes nothing and the invariance story collapses. Additionally, if the value-spread oracle shows tie classes are value-heterogeneous and hash-based tie-breaking selects arbitrary members, then the quality claim is unsupported and the contribution must be reframed as determinism only.

## References

1. Accelerating Mixture of Block Attention via Dynamic Routing, KV-Centric Execution and Runtime Pipelining
