# BraidWord: Differentiable Braid-Word Optimization for Identifiable Strand Connectivity

## Motivation

Across both the braid-parametric tree (Unsupervised 3D Braided Hair Reconstruction) and the Gaussian-segment tree (HairGS, GaussianHair, Gaussian Haircut), discrete strand/braid connectivity is never treated as an identifiable variable coupled to the image loss: the braid method fixes a synthetic braid template so topology is assumed rather than inferred, while HairGS merges Gaussian segments by local geometric-continuity heuristics, making topology a post-hoc artifact with no certificate. The structural reason is that neither lineage states a condition under which the photometric+silhouette loss is injective on the discrete topology, so there is no signal telling the optimizer whether a given crossing sequence is recoverable at all — the merge step is therefore forced to be heuristic or templated.

## Key Insight

The braid group's finite presentation reduces all strand-connectivity ambiguity to a crossing sequence, and a single transverse crossing in the silhouette breaks the translation/rotation/strand-relabeling gauge symmetry that otherwise makes the discrete topology unidentifiable, so the crossing word becomes a discrete variable that the photometric+silhouette loss can rank.

## Method

**What it is.** BraidWord takes a single-view (or multi-view) image set and a set of segment endpoints from an unstructured Gaussian/strand initialization and outputs a braid word (a sequence of generators and their inverses) plus continuous geometry, by optimizing a Gumbel-softmax relaxation of braid generators jointly with strand geometry. Inputs: images, silhouette, initial strand segments; outputs: a discrete braid word, per-strand curves, and a connectivity certificate (crossing count and residual gauge check).

**Load-bearing assumption (stated explicitly).** The method assumes that the photometric+silhouette loss, evaluated on the rendered curves decoded from a braid word, is injective on the discrete braid word up to global translation/rotation/strand-relabeling gauge once at least one transverse crossing is visible in the silhouette. This assumption is not proven; it is tested empirically by the ambiguity-class calibration below. When it fails (e.g., symmetric braids, near-non-transverse views), the method returns an equivalence class of words rather than a single word, and the certificate reports the class size and loss gap.

**How it works.**

```pseudocode
# Phase 0: initialization (from existing pipeline)
segments = init_gaussian_segments(images)          # e.g. HairGS-style merge seed
endpoints = segment_endpoints(segments)

# Phase 1: build braid-word lattice
#   lattice L = words over {sigma_i, sigma_i^{-1}} with |word| <= K_max
K_max = schedule_max_crossings(silhouette)          # from crossing-count schedule
lattice = enumerate_words(generators=sigma_1..sigma_{n-1}, max_len=K_max)

# Phase 2: joint discrete+continuous optimization
logits = zeros(len(lattice))                        # word-selection logits
tau = tau_0                                          # Gumbel temperature
for step in 1..T:
    # anneal temperature per crossing-count schedule
    tau = tau_0 * exp(-step / T) * crossing_schedule(step, K_max)
    # relaxed word selection
    w = gumbel_softmax(logits / tau)                # differentiable word distribution
    # decode word -> strand permutation/connectivity
    perm = decode_braid(w)                          # maps word to endpoint pairing
    curves = fit_strands(endpoints, perm)           # continuous geometry
    # differentiable render + losses
    I_hat = render(curves, camera)
    L_photo = ||I_hat - I||_1
    L_sil   = silhouette_loss(I_hat, S)
    L_word  = entropy_regularizer(w)                # encourage discreteness
    L = L_photo + lambda_s * L_sil + lambda_w * L_word
    backprop(L) -> logits, curves

# Phase 3: certificate
w_star = argmax(logits)
cert = check_crossing_count(w_star, silhouette)     # >=1 transverse crossing?
cert_gauge = residual_gauge_check(curves, w_star)   # translation/rotation/relabel
# ambiguity-class calibration (tests the load-bearing assumption)
losses = [L_photo(w) + lambda_s * L_sil(w) for w in lattice]  # re-render each word
best = min(losses)
ambiguity_class = {w in lattice : losses[w] <= best + delta}   # delta = 0.05 * best
cert_ambiguity = (len(ambiguity_class), best, max(losses[w] for w in ambiguity_class) - best)
return w_star, curves, cert, cert_gauge, cert_ambiguity
```

Hyperparameters inline: `K_max` from silhouette crossing count; `tau_0` initial temperature; `lambda_s`, `lambda_w` loss weights; `T` annealing steps; `delta` ambiguity tolerance (set to 0.05 * best loss).

**Why this design.** We chose a Gumbel-softmax relaxation over the braid-word lattice rather than a fixed braid template (as in Unsupervised 3D Braided Hair Reconstruction) because the template assumes topology and cannot infer it, accepting the cost that the lattice must be bounded by `K_max` and words outside it are unreachable. We chose a crossing-count-derived annealing schedule over a fixed temperature decay because the identifiability argument says a transverse crossing is what breaks gauge symmetry, so temperature must stay high until the optimizer has committed enough crossings for the photometric+silhouette signal to disambiguate the word, accepting the cost that early steps are noisy. We chose to decode the word into an endpoint permutation and fit continuous curves through that permutation rather than merging segments by geometric continuity (as in HairGS) because continuity heuristics are post-hoc and unverifiable, accepting the cost that the permutation decode must be differentiable and the lattice enumeration grows as (2(n-1))^K_max. A domain expert would not describe this as HairGS plus a Gumbel head: HairGS's merge is a fixed geometric rule applied after the fact, whereas here the crossing word is a variable inside the same loss that shapes the curves, and the certificate is a property of the optimized word, not of a heuristic.

**Why it measures what we claim.** The crossing-count certificate `check_crossing_count(w_star, silhouette)` measures topological identifiability because the injectivity argument assumes at least one transverse crossing appears in the silhouette; this assumption fails when the braid is viewed exactly along a crossing (a non-transverse view), in which case the certificate reflects only the visible crossing count and not whether the word is recoverable. The residual gauge check `residual_gauge_check(curves, w_star)` measures whether the solution is unique up to global translation/rotation/strand relabeling because the loss is invariant under those gauge actions by construction; this assumption fails when two strands coincide geometrically (a degenerate braid), in which case the residual reflects a genuine geometric collision rather than a gauge ambiguity. The word entropy regularizer measures discreteness of the recovered topology because the Gumbel-softmax relaxation converges to a one-hot word as temperature anneals; this assumption fails when the photometric loss is flat across multiple words (e.g., a symmetric braid seen from one side), in which case the entropy reflects optimization noise rather than a committed topology. The ambiguity-class calibration `cert_ambiguity` measures the size of the equivalence class of words whose loss is within `delta` of the best, because the load-bearing assumption is that this class is a singleton up to gauge; this assumption fails when multiple distinct words render to near-identical images, in which case `cert_ambiguity` reports the class size and loss gap instead of claiming a unique recovery.

## Contribution

(1) A differentiable braid-word optimization framework that treats strand/braid connectivity as a discrete variable coupled to the photometric+silhouette loss via a Gumbel-softmax relaxation of braid generators with a crossing-count annealing schedule. (2) A connectivity certificate based on crossing count and residual gauge check that flags when the recovered topology is identifiable versus when the view degenerates. (3) A design principle: the crossing word, not the geometric merge heuristic, is the correct optimization variable for braided hair topology, demonstrated by optimizing it jointly with geometry.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | Single-view braided hairstyle images with varied crossing counts (n=4-8 strands, K_max=6-10 crossings) | Directly exercises topology inference, not just geometry |
| Primary metric | Braid-word recovery accuracy (exact word match) | Discrete topology is the core claim, not pixel fidelity |
| Secondary metric | Ambiguity-class size (number of words within delta=0.05*best loss) and loss gap | Tests the load-bearing assumption of loss-injectivity up to gauge |
| Baseline | Unsupervised 3D Braided Hair Reconstruction | Template-based method assuming fixed braid topology |
| Baseline | HairGS | Geometric-continuity merge with no topology certificate |
| Baseline | Strand-Aligned 3D Gaussians | Strand reconstruction without crossing-aware connectivity |
| Ablation-of-ours | Fixed-K_max, no crossing-count annealing schedule | Tests whether annealing is what enables word commitment |
| Ablation-of-ours | Gumbel-softmax word selection on HairGS merge seed without braid-group decode (permutation decode replaced by identity) | Tests whether the braid-group structure, not the relaxation alone, yields word recovery |

### Why this setup validates the claim

The central claim is that braid topology can be recovered as a discrete variable jointly with geometry, and that a crossing-count-derived annealing schedule is what makes the discrete word identifiable. The dataset must therefore contain braids whose crossing counts vary, because if all instances had the same topology a fixed template would suffice and the claim would be untestable. The primary metric — exact braid-word match rather than a geometric distance — is the right detector because the claim is about recovering a discrete word, and geometry metrics can be high while the word is wrong (a visually plausible but topologically incorrect braid). The secondary ambiguity-class metric directly tests the load-bearing assumption: if the true word is not in the reported ambiguity class, or if the class is large for transverse-heavy views, the injectivity claim is falsified. The template baseline tests the sub-claim that topology cannot be assumed; the two strand-based baselines test the sub-claim that post-hoc geometric merging does not yield a verifiable word; the fixed-K_max ablation tests the sub-claim that the annealing schedule, not merely the Gumbel relaxation, is responsible for commitment; the no-braid-decode ablation tests whether the braid-group structure (generator relations, permutation decode) is necessary beyond the Gumbel head. Together these form a falsifiable test: if the template baseline already matches our word accuracy, or if either ablation matches the full method, the mechanism claim collapses.

### Expected outcome and causal chain

**vs. Unsupervised 3D Braided Hair Reconstruction** — On a case where the braid has an unusual crossing count or a non-standard over/under pattern, the template baseline produces a geometrically smooth but topologically wrong braid because its fixed template cannot represent words outside its assumed family. Our method instead searches the braid-word lattice up to K_max and selects the word by photometric-plus-silhouette fit, so we expect a large word-accuracy gap concentrated on instances whose crossing count deviates from the template's assumption, with near-parity on instances that happen to match the template.

**vs. HairGS** — On a case where two strands pass close together but do not actually cross, the baseline produces an incorrect merge because its geometric-continuity rule cannot distinguish proximity from a true transverse crossing. Our method instead decodes an explicit word into an endpoint permutation and fits curves through that permutation, so we expect our crossing-count certificate to agree with ground-truth crossing count on these near-miss cases while the baseline's implied connectivity diverges, showing a gap that grows with the number of near-miss strand pairs.

**vs. Strand-Aligned 3D Gaussians** — On a case where the braid is viewed from an angle that makes one crossing nearly non-transverse, the baseline produces plausible geometry but ambiguous connectivity because it has no discrete topology variable to commit. Our method instead keeps temperature high until enough crossings are committed, so we expect our word entropy to drop sharply after the crossing-count schedule passes the visible crossing count, with word accuracy holding on transverse-heavy views and degrading gracefully (not collapsing) on near-non-transverse views; the ambiguity-class size should increase on the latter, confirming that the method reports ambiguity rather than over-committing.

**vs. Ablation: Gumbel head on HairGS seed without braid-group decode** — On instances with multiple crossings, the ablation can select a word but cannot map it to a valid endpoint permutation, so its rendered curves do not respect the selected crossing sequence. We expect the full method to outperform this ablation on exact word match, with the gap growing with K_max, demonstrating that the braid-group structure (generator relations, permutation decode) is what yields word recovery, not the Gumbel relaxation alone.

### What would falsify this idea

If our braid-word accuracy gain over the template baseline is uniform across crossing-count subsets rather than concentrated where the template's assumed topology fails, or if the fixed-K_max no-annealing ablation matches the full method's word accuracy, or if the true word is frequently outside the reported ambiguity class on transverse-heavy views, then the claimed mechanism — topology as a jointly optimized discrete variable enabled by crossing-derived annealing and identifiable up to a certified ambiguity class — is not what produces the result.

## References

1. Unsupervised 3D Braided Hair Reconstruction from a Single-View Image
2. Neural Haircut: Prior-Guided Strand-Based Hair Reconstruction
3. HairGS: Hair Strand Reconstruction based on 3D Gaussian Splatting
4. 3D Gaussian Splatting for Real-Time Radiance Field Rendering
5. Human Hair Reconstruction with Strand-Aligned 3D Gaussians
6. GaussianHair: Hair Modeling and Rendering with Light-aware Gaussians
7. Dr.Hair: Reconstructing Scalp-Connected Hair Strands without Pre-Training via Differentiable Rendering of Line Segments
8. Point-NeRF: Point-based Neural Radiance Fields
