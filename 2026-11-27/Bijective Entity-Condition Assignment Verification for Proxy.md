# Bijective Entity-Condition Assignment Verification for Proxy-Guided Video Generation

## Motivation

Proxy-guided video generators are evaluated by ProxyGenBench using separable, case-specific checklists, so a binding error that preserves every per-item score — such as swapping which entity an audio cue is attached to while each cue still matches some entity — remains invisible to the protocol; ProxyGenBench itself reports that strong structural adherence does not ensure correct binding, and FloVD and IM-Zero likewise optimize structural/motion control without any joint multimodal binding constraint. The root cause is structural: checklist aggregation assumes binding decomposes into independent per-item checks, but binding is a global bijection between proxy elements, entities, and modality conditions, and no per-item score can certify a joint assignment invariant.

## Key Insight

Binding correctness is a bijective assignment constraint over the joint (proxy-element × entity × modality-condition) space, so it is detectable exactly when the assignment matrix deviates from a permutation structure — a property that separable per-item checklist scores are invariant to and therefore cannot certify.

## Method

**What it is.** We introduce BIND-V (Bijective INvariant verifier for Deployment of Video generators), a post-hoc verifier that takes a generated video plus its structural proxy (blockout regions, motion trajectories, camera path), its text/image/audio conditions, and the generator's per-entity detection/tracking/attribution outputs, and returns a scalar binding-violation score together with the argmax violating assignment. It replaces checklist aggregation with a single global assignment invariant.

**Load-bearing assumption (stated explicitly).** A correctly bound generation induces an affinity matrix whose Sinkhorn relaxation is close to a permutation matrix, so that the Frobenius residual ||S − Π*||_F^2 (and the marginal residual) is a valid, monotone certificate of binding correctness rather than a reflection of tracker/embedding ambiguity. This assumption fails when two entities share near-identical visual or audio signatures (e.g., two identical blockout cubes) or when two conditions have colliding embeddings, in which case L_perm reflects ambiguity of the tracker or encoder rather than an actual binding error. To make the verifier abstain under such ambiguity instead of scoring it as a violation, we condition the violation decision on a margin between the best and second-best permutation, calibrated by an affinity-uncertainty band estimated from tracker and encoder variance.

**How it works.**

```pseudocode
INPUT: video V, proxy P = {blockout regions R, trajectories T, camera path C},
       conditions M = {text spans X, image refs I, audio cues A},
       generator entity tracks E (from an off-the-shelf tracker on V)
OUTPUT: violation score s, violating assignment pair, abstention flag

# Phase 1: build per-pair affinity tensor with uncertainty
for each proxy element p in P, entity e in E, condition m in M:
    a[p,e,m] = cos( f_vis(V, p, e),  f_cond(m) )       # f_vis: region-track CLIP-style embedding; f_cond: modality encoder
    tau[p,e,m] = temporal_overlap(p, e) * spatial_overlap(p, e)   # proxy-element/entity co-location
    A[p,e,m] = softmax over (e,m) of ( a[p,e,m] + lambda * tau[p,e,m] )   # lambda = 1.0

# Phase 1b: estimate per-pair affinity uncertainty via ensemble
sigma[p,e,m] = std over K=5 tracker seeds and 3 modality encoder seeds of a[p,e,m]

# Phase 2: joint assignment invariant
S = sinkhorn(A, epsilon=0.05, iters=50)   # doubly-stochastic relaxation of permutation constraint
# S is |P| x (|E|*|M|); row sums -> 1, column sums -> 1

# Phase 3: consistency loss over the JOINT matrix
L_bind = - sum_{p,e,m} S[p,e,m] * log A[p,e,m]         # entropy of assignment
L_marg = sum_p | sum_{e,m} S[p,e,m] - 1 | + sum_{e,m} | sum_p S[p,e,m] - 1 |
L_perm = || S - Pi* ||_F^2   where Pi* = hungarian(S)   # distance to nearest permutation
s = L_bind + beta * L_marg + gamma * L_perm             # beta=1.0, gamma=0.5

# Phase 3b: margin-based calibration and abstention
# Compute best and second-best permutation costs via Hungarian on A
cost1, Pi1 = hungarian(-A)   # best assignment
cost2, Pi2 = hungarian(-A with Pi1 entries masked)   # second-best
margin = cost2 - cost1
# Affinity-uncertainty band: propagate sigma to permutation cost via first-order approximation
band = 2 * sqrt( sum_{p,e,m} sigma[p,e,m]^2 * (Pi1[p,e,m] - Pi2[p,e,m])^2 )
if margin < band:
    abstain = True   # ambiguity dominates; do not report violation
else:
    abstain = False

# Phase 4: counterfactual swap probe (detects score-preserving swaps)
for each candidate swap (e1,m1)<->(e2,m2) with S near-tied:
    s_swap = score with A columns permuted accordingly
    if |s_swap - s| < delta and s_swap > s_thresh and not abstain:  flag swap as undetected-by-checklist
return s, argmax violating (p,e,m), abstain
```

**Why this design.** We chose a Sinkhorn-relaxed permutation constraint over independent per-pair thresholding because the motivation-level failure (a swap that preserves every per-item score) is by construction a violation of the *marginal* doubly-stochastic structure, not of any single affinity — accepting the cost that Sinkhorn introduces an entropy temperature epsilon whose value trades off strictness against numerical stability. We chose a single joint assignment matrix over separate per-modality assignment matrices (one for text, one for image, one for audio) because cross-modality swaps — the exact failure ProxyGenBench's checklist cannot see — only manifest when modalities share the same column space; the cost is that the tensor scales as |P|·|E|·|M| and must be sparsified for large scenes. We chose the Hungarian distance term L_perm over a pure soft-assignment loss because we need a discrete certificate of the nearest permutation, accepting that Hungarian is O(n^3) and is applied only to the final |P|×(|E||M|) matrix rather than during training. We added the margin-based abstention (Phase 3b) to make the load-bearing assumption explicit and operational: when the affinity-uncertainty band exceeds the permutation margin, the verifier abstains rather than scoring ambiguity as a violation, preserving the joint-bijection contribution while removing dependence on the false calibration assumption. A domain expert would not describe this as a variant of ProxyGenBench's checklist scoring or of retrieval-style nearest-condition matching, because neither computes a *joint* permutation residual: the checklist marginalizes over entities and conditions independently, and retrieval picks a nearest neighbor per item without enforcing that the collection of picks is a bijection.

**Why it measures what we claim.** The quantity `L_perm = ||S − Π*||_F^2` measures **binding correctness** because the named assumption is that a correctly bound generation induces an affinity matrix whose Sinkhorn relaxation is close to a permutation matrix — i.e., each proxy element is assigned to exactly one (entity, condition) pair; this assumption fails when two entities share near-identical visual or audio signatures (e.g., two identical blockout cubes), in which case L_perm reflects ambiguity of the tracker rather than an actual binding error. The quantity `L_marg` measures **bijectivity** because the assumption is that row and column marginals of a valid assignment are all one; it fails when the generator omits an entity entirely, in which case a missing column reflects a completion failure rather than a mis-binding. The counterfactual swap probe measures **score-preserving binding error** because the assumption is that a true binding error changes the joint assignment residual while leaving per-item affinities unchanged; it fails when the swap is between two conditions with identical embeddings, in which case the probe reflects embedding collision rather than a real violation. The margin-based abstention (Phase 3b) measures **calibrated binding violation** because the assumption is that the affinity-uncertainty band captures tracker/encoder noise; it fails when the ensemble variance underestimates true ambiguity, in which case the verifier may still flag a collision as a violation.

## Contribution

(1) A joint bijective assignment verifier (BIND-V) that formulates multimodal binding in proxy-guided video generation as a single Sinkhorn-relaxed permutation problem over the (proxy-element × entity × condition) tensor, replacing separable checklist aggregation. (2) A counterfactual swap probe that surfaces binding errors which preserve every per-item affinity score, providing a diagnostic that separable protocols structurally cannot produce. (3) An analysis of when the permutation residual reflects genuine binding error versus tracker ambiguity or embedding collision, delimiting the verifier's operational validity.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | ProxyGenBench (312 cases, 60 scenes) | Multi-entity, multi-modal binding stress cases |
| Primary metric | Binding-violation AUC vs checklist-flagged errors | Directly targets the score-preserving-swap failure |
| Baseline | ProxyGenBench checklist aggregation | The independent-marginal scorer we claim to beat |
| Baseline | Nearest-condition retrieval matcher | Per-item greedy matching, no bijection |
| Baseline | Independent per-modality assignment | Text/image/audio scored separately, no cross-modality coupling |
| Ablation-of-ours | BIND-V without L_perm (soft-only) | Isolates the discrete permutation certificate's contribution |
| Ablation-of-ours | BIND-V without margin-based abstention (Phase 3b) | Isolates the calibration/abstention contribution |
| Ablation-of-ours | BIND-V with collision injection (controlled embedding collisions) | Tests whether the residual separates real swaps from collisions |

### Why this setup validates the claim

The central claim is narrow and structural: binding errors that preserve every per-item affinity score are invisible to marginal/independent scorers but detectable as a joint permutation residual. ProxyGenBench is the right dataset because its controlled variations (proxy fidelity, text detail, multimodal inputs) deliberately create scenes where a proxy element, an entity track, and a condition can be permuted without changing any single similarity — exactly the trigger case. The checklist baseline tests the sub-claim that marginal aggregation misses these; the retrieval baseline tests the sub-claim that per-item nearest-neighbor matching cannot certify a bijection; the per-modality baseline tests the sub-claim that cross-modality swaps only surface in a shared column space. The ablation isolates whether the Hungarian distance term, not just Sinkhorn softness, carries the discrimination. The margin-based abstention ablation isolates whether the calibration step is necessary to avoid false positives under embedding collision. The collision-injection ablation directly tests the load-bearing assumption: we inject controlled embedding collisions (e.g., replace two entity embeddings with their average) and measure whether the residual separates real swaps from collisions. The metric is binding-violation AUC against human-verified violations, because it rewards ranking the score-preserving swaps above clean cases — a uniform-quality metric like FVD would be blind to the predicted effect by construction. To ground the human-verified ground truth, we will annotate a subset of 100 cases (stratified across scene tasks and proxy fidelity levels) with 3 annotators per case, reporting inter-annotator agreement (Fleiss' kappa) and resolving disagreements by majority vote; annotation cost is estimated at 2 hours per annotator per 10 cases, totaling 60 annotator-hours. Runtime and memory of the affinity tensor on the largest ProxyGenBench scene (|P|≈20, |E|≈10, |M|≈15) are measured; sparsification via top-k=5 per row reduces memory from O(|P||E||M|) to O(|P|k) with negligible AUC change.

### Expected outcome and causal chain

**vs. ProxyGenBench checklist aggregation** — On a case where two blockout cubes with near-identical appearance each carry distinct text spans and the generator swaps which cube receives which span, the checklist produces full marks because each span still matches some cube at high affinity and each cube still has a span. Our method instead sees the joint affinity matrix admit a lower-residual permutation under the swap, so L_perm rises even though per-item scores are unchanged. We expect a clear AUC gap concentrated on the swap-triggered subset, with near-parity on single-entity scenes.

**vs. Nearest-condition retrieval matcher** — On a case where an entering entity's image reference and an exiting entity's audio cue are both plausibly matched by either track, greedy retrieval assigns each condition to its own nearest track independently and never checks that the resulting collection is a bijection, so a crossed assignment passes. Our Sinkhorn relaxation enforces row and column marginals jointly, so the crossed assignment inflates L_marg and L_bind. We expect the gap to appear specifically on multi-reference scenes and to vanish on scenes with a single reference per modality.

**vs. Independent per-modality assignment** — On a case where text is satisfied by entity A and audio by entity B but the correct binding requires both on the same entity, per-modality scoring is perfect because each modality's marginal is satisfied. Our shared column space couples modalities, so the cross-modality swap changes the joint permutation distance. We expect a moderate gap on multimodal cases and parity on text-only cases, which is the signature that the coupling, not general scoring power, drives the gain.

**vs. BIND-V without margin-based abstention** — On cases with near-identical entities or colliding embeddings, the no-abstention variant will flag ambiguity as a violation, inflating false positives. With abstention, the verifier abstains when the affinity-uncertainty band exceeds the permutation margin, so AUC on collision cases should improve while AUC on clean swap cases remains unchanged. We expect the abstention ablation to show a significant AUC drop on the collision-injection subset, confirming the calibration step is necessary.

**vs. BIND-V with collision injection** — We inject controlled embedding collisions (e.g., replace two entity embeddings with their average) and measure whether the residual separates real swaps from collisions. We expect that without abstention, the residual will fail to separate (AUC near 0.5), while with abstention, the verifier will abstain on collision cases and maintain high AUC on real swaps, demonstrating that the margin-based calibration addresses the load-bearing assumption.

### What would falsify this idea

If BIND-V's advantage over the checklist baseline is uniform across all subsets rather than concentrated on swap-triggered and multimodal cases, the joint-permutation mechanism is not what produces the gain and the central claim is wrong. Equally fatal: if the no-L_perm ablation matches full BIND-V, the discrete permutation certificate contributes nothing and the framing collapses to soft-assignment scoring. Additionally, if the margin-based abstention does not improve AUC on collision-injection cases, the calibration step fails to address the load-bearing assumption.

## References

1. ProxyGenBench: Benchmarking Proxy-Guided Video Generation
