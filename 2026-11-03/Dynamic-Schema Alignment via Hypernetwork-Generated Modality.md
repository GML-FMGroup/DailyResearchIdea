# Dynamic-Schema Alignment via Hypernetwork-Generated Modality Encoders with Anchor-Based Identifiability

## Motivation

The omnimodal embedding roadmap in 'From Fragmentation to Fusion' identifies comparability as the central unsolved capability, yet every alignment method it surveys assumes the modality set is fixed at training time; Dyg2vec (2024) makes this assumption explicit by hard-coding a node-type schema, so any newly observed type has no path into the shared metric space. The structural root cause is that the shared space is defined by the parameters of a fixed set of encoders rather than by a set of relations that must hold among embedded objects — so when the encoder set grows, the space itself moves and previously established comparability silently breaks. This is not a data-scarcity problem but a schema-rigidity problem: the embedding space has no invariant that survives modality-set expansion.

## Key Insight

Comparability under schema growth is preserved if the shared space is pinned by pairwise distance constraints among a small set of permanently-retained anchor modalities, so that a newly generated encoder is admissible exactly when it reproduces the anchor sub-geometry — making the space's identity a property of the anchors rather than of whichever encoders happen to exist.

## Method

**What it is.** We introduce **SCOPE** (Schema-Conditioned Projection with Encoder synthesis), a mechanism that takes as input a modality's metadata (a small vector of structural descriptors: tokenization granularity, value cardinality, temporal autocorrelation, sparsity) together with a stream of raw observations, and outputs an encoder that maps those observations into a shared metric space whose pairwise geometry on a frozen anchor set is preserved by construction.

**How it works.**

```pseudocode
# Phase 0: one-time anchor establishment (first K modalities)
# K is chosen so that the anchor centroid Gram matrix G[a,b] = <c_a, c_b> has
# rank equal to the embedding dimensionality d (checked via SVD: number of
# singular values > 0.01 * max singular value). If rank < d, add anchor
# modalities until full rank is achieved or the pool is exhausted.
for each anchor modality a in {a_1..a_K}:
    train encoder E_a by contrastive alignment to a reference anchor a_1
    store centroid c_a and pairwise anchor distances D[a,b] = ||c_a - c_b||
FREEZE {E_a}, {c_a}, D          # never updated after this phase

# Phase 1: hypernetwork that synthesizes encoders from metadata
H_psi( m ) -> theta_m             # m = metadata vector, theta_m = encoder params
E_m = Encoder(theta_m)

# Phase 2: training a new modality m (arrives at any time)
for each batch of paired (x_m, x_a) with anchor a:
    z_m = E_m(x_m);  z_a = E_a(x_a)          # anchors frozen
    L_align   = InfoNCE(z_m, z_a, tau=0.07)   # pull matched pairs together
    L_topo    = sum_{a,b} w_ab * ( ||mu(z_m)-c_a|| - ||mu(z_m)-c_b||
                                  - (D[a,b] - 0) )^2
                # w_ab = 1/(D[a,b]+eps); mu = batch mean of z_m
    L_anchor  = sum_a ( ||mu(z_m) - c_a|| - d_target(m,a) )^2
    L = L_align + lambda_1 * L_topo + lambda_2 * L_anchor
         # lambda_1 = 0.5, lambda_2 = 0.1
    backprop through H_psi only (anchors frozen)

# Phase 3: admissibility test (identifiability check)
# Held-out probe set: P modalities never used in L_topo or L_anchor.
# For each probe p, compute its centroid c_p using its frozen encoder E_p.
# Compute residual r = min over rigid transforms T of  || T(anchor-subgeometry of E_m) - D ||_F
# PLUS probe residual r_probe = min over rigid transforms T of || T(probe-subgeometry of E_m) - D_probe ||_F,
# where D_probe is the pairwise distance matrix among probe centroids.
admit E_m into the shared space iff r < delta AND r_probe < delta_probe
# delta = 0.05 * mean(D); delta_probe = 0.05 * mean(D_probe)
reject and re-fit with augmented anchor pairs otherwise
```

The topological loss `L_topo` is the load-bearing term: it forces the new modality's centroid to sit at distances from every anchor centroid that reproduce the anchor–anchor distance matrix, i.e. it forces the new point into the *same* metric configuration the anchors already define. `L_anchor` supplies the absolute scale that `L_topo` leaves free (a distance-difference loss is invariant to translation of the whole configuration). The hypernetwork is trained jointly across all admitted modalities so that metadata-to-parameter mapping generalizes to unseen metadata.

**Why this design.** We chose a *frozen* anchor set over periodically re-aligning all modalities because re-alignment lets the shared space drift each time a modality is added — the exact failure Dyg2vec exhibits when its node-type schema is extended — accepting the cost that the anchor set must be chosen well up front and that anchor modalities must remain available for paired supervision. We chose a hypernetwork over per-modality fine-tuning because per-modality fine-tuning gives no mechanism to embed a modality with no training pairs, accepting the cost that the hypernetwork's metadata vector must actually carry enough structural signal to distinguish modalities (we verify this with a metadata-ablation). We chose a distance-reproduction loss over a direct parameter-regularization toward anchor encoders because parameter-space proximity does not imply geometric consistency in embedding space, accepting the cost that the topological term is invariant to rigid motion and therefore requires the auxiliary `L_anchor` term. A domain expert would not describe SCOPE as a variant of Dyg2vec: Dyg2vec learns one joint encoder over a fixed node-type vocabulary, whereas SCOPE never trains a joint encoder at all — it synthesizes per-modality encoders and constrains them through a frozen anchor geometry, so the object being optimized is the *relation* between embeddings, not the embeddings themselves.

**Why it measures what we claim.** The topological residual `L_topo` measures *schema-consistency of the new modality* because the assumption is that the anchor–anchor distance matrix D is a sufficient statistic for the shared space's identity — if that assumption holds, reproducing D from the new modality's position is equivalent to placing it in the same space; this assumption fails when the anchor set is too small or too mutually correlated to span the space's degrees of freedom, in which case `L_topo` reflects only consistency with a degenerate subspace and a new modality can be admitted while being mis-placed along an unconstrained direction. The admissibility residual `r` measures *identifiability of the new encoder* because the assumption is that any two encoders inducing the same anchor sub-geometry are equivalent up to a rigid transform, which is what makes the anchor configuration an identifier; this assumption fails when the new modality's true position is near-collinear with an anchor, in which case `r` stays small for genuinely different encoders and the test admits non-identifiable ones. The metadata-conditioned hypernetwork output `theta_m` measures *generalization to unseen modalities* because the assumption is that metadata is a sufficient statistic for the encoder's functional role; this assumption fails for modalities whose behavior is driven by content not captured in metadata (e.g. two text-like modalities with identical descriptors but different semantics), in which case `theta_m` reflects the metadata prior rather than the modality's actual structure. Additionally, the batch-mean centroid `mu(z_m)` measures *the modality's population position* because the assumption is that the batch is an unbiased sample from the modality distribution; this assumption fails under small batch size or heavy-tailed modality distributions, in which case `mu(z_m)` reflects batch noise rather than the true centroid.

## Contribution

(1) SCOPE, a dynamic-schema alignment mechanism in which per-modality encoders are synthesized by a metadata-conditioned hypernetwork and constrained to a frozen anchor geometry via a topological distance-reproduction loss, allowing modalities unseen during training to be embedded without retraining the shared space. (2) An anchor-residual admissibility criterion that converts the geometric consistency of a candidate encoder into an accept/reject decision, together with the finding that distance-reproduction alone is rigid-motion-invariant and must be paired with an absolute-scale anchor term to pin the space. (3) A metadata-ablation protocol that isolates how much of the hypernetwork's generalization to unseen modalities is attributable to metadata signal versus shared training, providing a diagnostic for when dynamic-schema alignment is actually operating.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Cross-modal retrieval over held-out modality stream | Tests late arrival of unseen modalities |
| Primary metric | Retrieval recall@10 on newly admitted modality | Directly probes geometric placement quality |
| Baseline 1 | Dyg2vec joint encoder | Fixed-vocabulary joint encoder that cannot extend |
| Baseline 2 | Per-modality fine-tuning | No mechanism for unpaired modality embedding |
| Baseline 3 | Re-align-all retraining | Drift-prone when schema is extended |
| Baseline 4 | Shared encoder + per-modality LoRA adapter | Tests if hypernetwork is just expensive reparameterization |
| Ablation-of-ours | SCOPE without L_topo | Isolates the load-bearing topological constraint |
| Ablation-of-ours | SCOPE with varying anchor-set size K (K=4,8,16,32) | Tests identifiability vs. anchor span |
| Ablation-of-ours | SCOPE with varying batch size (16,64,256) | Tests centroid estimation stability |
| Ablation-of-ours | SCOPE with correlated anchors (correlation ρ=0.0,0.5,0.9) | Tests failure mode of degenerate subspace |

### Why this setup validates the claim

This design turns the central claim — that a frozen anchor geometry plus a metadata-conditioned hypernetwork places unseen modalities into a stable shared space — into a falsifiable test by pairing a held-out modality stream (modalities never seen during anchor establishment or hypernetwork training) with retrieval recall@10 on those newly admitted modalities. Dyg2vec tests the sub-claim that a fixed joint encoder cannot absorb new modalities without schema extension, since it has no vocabulary slot for them; per-modality fine-tuning tests the sub-claim that gradient-based fitting requires paired supervision and thus fails for modalities arriving with few or no anchor pairs; re-align-all retraining tests the sub-claim that re-aligning anchors lets the shared space drift, so previously admitted modalities should degrade as more arrive. The shared encoder + LoRA baseline tests whether the hypernetwork's metadata-to-parameter mapping is necessary or merely an expensive reparameterization of adapter-based alignment. The anchor-set size and correlation ablations directly probe the load-bearing assumption: if the anchor Gram matrix is not full-rank or anchors are highly correlated, the admissibility residual r should correlate with downstream placement error, and recall@10 should degrade. The batch-size ablation tests whether the batch-mean centroid mu(z_m) is a faithful estimator of the modality's population position. The ablation SCOPE without L_topo is the key control: if the topological loss is truly load-bearing, removing it should collapse the geometric consistency that the other two losses cannot supply, so recall on the new modality should fall back toward what alignment alone achieves. Recall@10 is the right metric because the claim is about *placement* — whether the new modality's embeddings land near semantically matched anchor embeddings in the shared space — and retrieval is exactly the observable that degrades when placement is off-axis, whereas a classification head or linear probe could mask placement error by relearning a decision boundary.

### Expected outcome and causal chain

**vs. Dyg2vec joint encoder** — On a case where a new modality arrives whose token structure does not fit the fixed node-type vocabulary, Dyg2vec produces embeddings that are either collapsed onto an existing type or routed through a mismatched slot, because its joint encoder has no parameter pathway for an unseen type. Our method instead synthesizes a fresh encoder from that modality's metadata and constrains its centroid through the frozen anchor geometry, because the hypernetwork maps metadata to parameters and L_topo pins the new point into the existing metric configuration. We expect a large recall@10 gap on the newly arrived modality and near-parity on the anchor modalities themselves.

**vs. Per-modality fine-tuning** — On a case where a new modality arrives with very few paired anchor samples, per-modality fine-tuning produces an encoder that overfits those pairs and drifts in unconstrained directions, because it has no prior over where the modality should sit and no structural signal beyond the sparse pairs. Our method instead conditions the encoder on metadata and anchors its position via L_topo and L_anchor, so even with few pairs the synthesized encoder inherits the shared geometry. We expect per-modality fine-tuning to trail noticeably in the low-pair regime and to converge only as pair count grows, while our method stays flat across pair counts.

**vs. Re-align-all retraining** — On a case where several modalities are added in sequence, re-align-all retraining produces embeddings for early modalities that shift between admissions, because every re-alignment perturbs the whole configuration and lets the shared space drift. Our method instead freezes the anchor encoders and centroids after Phase 0, so early-modality embeddings are untouched by later admissions. We expect re-align-all to show recall@10 for the earliest modalities degrading monotonically with the number of later admissions, while our method's early-modality recall stays essentially constant.

**vs. Shared encoder + per-modality LoRA adapter** — On a case where a new modality arrives, the LoRA baseline adapts a shared encoder via low-rank updates trained on the same anchor pairs, but it lacks a mechanism to synthesize a full encoder from metadata and may overfit to the limited anchor pairs. Our method instead generates the entire encoder from metadata and constrains it via the frozen anchor geometry, which should yield better recall@10 on newly admitted modalities, especially when anchor pairs are scarce. We expect the LoRA baseline to perform comparably on anchor modalities but to trail on new modalities, demonstrating that the hypernetwork is not merely a reparameterization.

### What would falsify this idea

If removing L_topo leaves recall@10 on newly admitted modalities essentially unchanged, the claim that the topological term is load-bearing is wrong. Likewise, if our advantage over per-modality fine-tuning is uniform across pair-count regimes rather than concentrated in the low-pair regime where the predicted failure occurs, the mechanism story is wrong and the gain is not coming from metadata-conditioned anchoring. Additionally, if the admissibility residual r does not correlate with downstream placement error across anchor-set sizes, or if the anchor Gram matrix cannot be made full-rank in practice, the identifiability claim is unsupported.

## References

1. From Fragmentation to Fusion: A Survey and Roadmap for Omnimodal Embeddings
