# Place-Bound Snapshot Memory: Cross-Visit Identity Certification for Embodied 3D Scene Memory

## Motivation

Persistent spatial understanding requires agents to recognize a revisited place as the same physical room despite viewpoint change, similar-looking distractors, and object changes between visits; ReSpaceBench (Lost in the Hallway) shows the strongest zero-shot model reaches only 50.5% overall accuracy versus 90.1% for humans, and agentic frameworks give no improvement over direct inference, with execution traces exposing evidence omissions and incorrect cross-visit reasoning. 3D-Mem's incremental construction pipeline aggregates Memory Snapshots by visual similarity and frontier coverage, but it has no mechanism that certifies two snapshots taken on separate visits refer to the same physical place, so visually similar same-type rooms are merged and genuinely revisited rooms are not bound together. The root cause is structural: snapshot retrieval is a similarity operation over appearance features, and appearance is not an identity signal across revisits because furniture moves, lighting shifts, and same-type rooms share appearance.

## Key Insight

Place identity across visits is a property that must be certified by an explicit binding invariant over a place-recognition decision boundary rather than inferred from snapshot feature similarity, because the invariant makes identity a transitive closure over labeled snapshots that similarity retrieval cannot represent.

## Method

**What it is.** Place-Bound Snapshot Memory (PBSM) augments 3D-Mem's Memory Snapshot store with a persistent place label per snapshot and a retrieval invariant. Inputs: a stream of Memory Snapshots (multi-view images plus frontier snapshots) produced by 3D-Mem's incremental construction pipeline, and a place-recognition module P that maps a snapshot to a place label or to a new label. Outputs: a place-labeled snapshot graph in which every retrieval returns the union of all snapshots sharing the queried snapshot's place label, and a certified identity set for each physical room.

**Load-bearing assumption (stated explicitly).** PBSM assumes that a frozen DINOv2 ViT-B/14 place encoder, mean-pooled over multi-view snapshots and thresholded at a single global cosine cutoff (tau_match=0.82) against incrementally updated per-place centroids, yields a distance that is monotone in physical room identity across revisits — i.e., same-room cross-visit distances fall below same-type-different-room distances for this encoder on the ReSpaceBench distribution. This assumption is verified empirically in the calibration study below (Phase 0); if separability is poor, the assumption is reported as failing and the identity-certification claim is scoped to the subsets where it holds.

**How it works.**

```pseudocode
# Phase 0: calibration of tau_match on ReSpaceBench revisit pairs (val split)
# Inputs: labeled revisit pairs (same-room across visits, same-type-different-room)
# Procedure:
#   - Compute place_encoder distances for all same-room pairs and all same-type-different-room pairs
#   - Report AUC of same-room vs same-type-different-room distance separability
#   - Sweep tau_match over grid {0.70, 0.72, ..., 0.90} (step 0.02) on val split
#   - Select tau_match maximizing balanced accuracy (mean of same-room recall and
#     same-type-different-room specificity); record false-merge and false-split rates
#   - Pre-register selected tau_match and grid before test-set evaluation
# Output: tau_match (default 0.82 if calibration selects it), AUC, false-merge rate, false-split rate

# Phase 1: place-recognition module P (frozen encoder; threshold from Phase 0)
# Input: snapshot s (multi-view images), candidate place labels L_known
# Output: label l in L_known or NEW
P(s, L_known):
    f = place_encoder(s)                 # frozen DINOv2 ViT-B/14, mean-pooled over views
    scores = {l: cosine(f, centroid[l]) for l in L_known}
    if max(scores) >= tau_match:         # tau_match from Phase 0 calibration
        return argmax(scores)
    return NEW

# Phase 2: incremental construction with identity binding
for each incoming snapshot s_t from 3D-Mem pipeline:
    l_t = P(s_t, L_known)
    if l_t == NEW:
        L_known.add(l_t); centroid[l_t] = place_encoder(s_t); n[l_t] = 1
    else:
        centroid[l_t] = (n[l_t]*centroid[l_t] + place_encoder(s_t)) / (n[l_t]+1)
        n[l_t] += 1
    store s_t with label l_t

# Phase 3: retrieval under the binding invariant
def retrieve(query_snapshot q):
    l_q = P(q, L_known)
    if l_q == NEW:                       # never seen this place
        return top_k_by_similarity(q, all_snapshots, k=5)
    return all snapshots with label l_q   # merged across visits, by invariant
```

**Why this design.** We chose an explicit place-recognition module with a calibrated match threshold over 3D-Mem's similarity-only retrieval because similarity conflates appearance with identity, accepting the cost that P can mislabel and the invariant then propagates the error to all snapshots in that place. We chose a centroid-based label store with incremental updates over a full pairwise graph because the invariant must be checkable in O(1) per snapshot for lifelong exploration, accepting the cost that a place whose appearance drifts (e.g., furniture rearranged) may require a threshold re-check. We chose to train P on ReSpaceBench's controlled revisit pairs rather than on generic image retrieval data because those pairs are the only supervision that explicitly labels same-room/different-room across visits, accepting the cost that P inherits ReSpaceBench's scene distribution. A domain expert would not describe PBSM as a variant of 3D-Mem because 3D-Mem's retrieval is a nearest-neighbor over snapshot features, whereas PBSM's retrieval is a set-union under a certified equivalence relation; the operation is different, not the features. Nor is it a variant of standard re-identification: the binding invariant makes identity transitive across visits, which nearest-neighbor retrieval cannot express.

**Why this measures what we claim.** The match threshold tau_match measures cross-visit place identity because the assumption that a snapshot's place encoder distance to a place centroid is monotone in physical-room identity holds when the encoder is trained contrastively on ReSpaceBench revisit pairs; this assumption fails when two distinct same-type rooms share furniture layout, in which case tau_match reflects appearance similarity rather than physical identity. The set-union retrieval measures persistent spatial memory because the assumption that all snapshots sharing a certified place label belong to the same physical room holds by construction of the invariant; this assumption fails when P mislabels a snapshot, in which case the retrieved set reflects the labeler's error rather than the room's true contents. The centroid update measures memory updating across visits because the assumption that a place's appearance is a running average of its observed snapshots holds when revisits are representative; this assumption fails when a place is observed once under atypical lighting, in which case the centroid reflects that single observation. The Phase 0 calibration study measures whether the encoder-distance-to-identity equivalence holds because the assumption that same-room cross-visit distances fall below same-type-different-room distances is directly testable as AUC on ReSpaceBench revisit pairs; this assumption fails when AUC is near 0.5, in which case tau_match reflects appearance similarity rather than physical identity and the identity-certification claim is scoped to subsets where separability holds.

## Contribution

["A place-bound snapshot memory framework that replaces 3D-Mem's similarity-based snapshot retrieval with a certified equivalence relation over persistent place labels, making cross-visit identity a transitive closure rather than a nearest-neighbor decision.", "A place-recognition module trained on ReSpaceBench's controlled same-room/different-room revisit pairs, with a calibrated match threshold that separates identity from appearance similarity.", 'An incremental construction pipeline that maintains place centroids and enforces the binding invariant at retrieval time, yielding a place-labeled snapshot graph suitable for lifelong exploration.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | ReSpaceBench (590 scenes, 3,244 questions) | Only benchmark labeling cross-visit room identity explicitly. |
| Primary metric | Accuracy on cross-visit room-association subset | Directly measures persistent place identity, the claimed target. |
| Secondary metric | False-merge rate and false-split rate at tau_match=0.82 | Quantifies the load-bearing encoder-distance assumption. |
| Secondary metric | AUC of same-room vs same-type-different-room encoder distances | Tests separability required by the identity assumption. |
| Baseline | 3D-Mem (similarity retrieval) | Tests whether similarity-only retrieval conflates appearance and identity. |
| Baseline | Direct video-LLM inference (best zero-shot) | Tests whether explicit place binding beats implicit model memory. |
| Baseline | Agentic video-understanding framework | Tests whether scaffolding without identity binding suffices. |
| Baseline | NetVLAD place-recognition head with same tau_match | Tests whether gain is from binding structure, not encoder. |
| Ablation-of-ours | PBSM with tau_match=0 (always NEW) | Isolates the contribution of the binding invariant. |
| Ablation-of-ours | PBSM with same labels but nearest-neighbor retrieval (label-preserving NN) | Isolates set-union invariant from encoder. |

### Why this setup validates the claim

The central claim is that explicit place-label binding, not richer features or longer context, is what fixes persistent spatial memory across revisits. ReSpaceBench is the right dataset because its controlled walkthroughs define visits by physical room entry and deliberately include same-type room distractors and cross-session object changes — exactly the conditions under which similarity-based retrieval should fail and an identity invariant should help. The primary metric isolates the room-association task, so gains cannot be attributed to unrelated reasoning improvements. 3D-Mem tests the sub-claim that nearest-neighbor snapshot retrieval conflates appearance with identity; the video-LLM and agentic baselines test whether implicit or scaffolded memory can substitute for explicit binding; the NetVLAD baseline tests whether the gain is from the binding structure rather than the specific DINOv2 encoder; and the two ablations — tau_match=0 (every snapshot its own place) and label-preserving nearest-neighbor retrieval (same labels, but retrieval ignores the set-union invariant) — together isolate the binding invariant itself from the encoder. The Phase 0 calibration study reports AUC, false-merge rate, and false-split rate at tau_match=0.82, making the load-bearing encoder-distance assumption explicit and testable; if separability is poor, the identity-certification claim is scoped to subsets where it holds. If the gain appears only where the predicted failure mode occurs — same-type distractors and revisits — the causal story is supported; if it is uniform, the mechanism claim is wrong.

### Expected outcome and causal chain

**vs. 3D-Mem (similarity retrieval)** — On a case where the agent revisits a kitchen that looks like a second, same-type kitchen across the hallway, 3D-Mem retrieves the nearest snapshot by feature similarity and returns contents from the wrong kitchen because its retrieval has no notion of physical place identity. Our method instead assigns both visits the same place label via P and returns the set-union of snapshots under that label, so we expect a clear accuracy gap concentrated on the same-type-distractor subset, with near-parity on unambiguous single-room queries.

**vs. Direct video-LLM inference** — On a case requiring state tracking across separated visits (e.g., an object moved between sessions), the best zero-shot video model answers from the current clip's appearance and omits the earlier visit because its context does not bind the two visits as the same place. Our method's labeled store makes the earlier snapshots retrievable by label, so we expect a noticeable gain on persistent-state-tracking questions and little change on single-visit captioning.

**vs. Agentic video-understanding framework** — On a case where the agent must decide whether a room seen now is the room seen ten minutes ago, the agentic framework's tool calls re-query by visual similarity and, per the benchmark's trace analysis, omit cross-visit evidence, yielding no improvement over direct inference. Our method's O(1) label check makes the cross-visit link explicit, so we expect the framework's evidence-omission errors to drop and the overall gain to exceed the framework's own reported null effect.

**vs. NetVLAD place-recognition head with same tau_match** — On the same-type-distractor subset, a NetVLAD head with the same threshold tests whether the gain comes from the binding structure or from DINOv2 features. If PBSM outperforms NetVLAD with the same threshold, the binding invariant — not the encoder — is the source of the gain; if they match, the encoder is doing the work and the invariant claim is weakened.

**vs. PBSM with tau_match=0 (ablation)** — On a case where the same room is revisited under changed lighting, the ablation creates a new label per visit and retrieves only the current visit's snapshots, reproducing 3D-Mem's failure. Our method's centroid update and calibrated threshold merge the revisits, so we expect the ablation to match 3D-Mem on revisits and our full method to show the gap, confirming the invariant — not the encoder — drives the effect.

**vs. PBSM with label-preserving nearest-neighbor retrieval (ablation)** — On a case where the same room is revisited and correctly labeled, the label-preserving NN ablation uses the same labels but retrieves only the nearest snapshot by feature similarity rather than the set-union of all snapshots under the label. If the full method outperforms this ablation on cross-visit state-tracking questions, the set-union invariant — not the labeler alone — is responsible; if they match, the invariant contributes nothing beyond labeling.

### What would falsify this idea

If PBSM's accuracy gain over 3D-Mem is uniform across all ReSpaceBench subsets rather than concentrated on same-type-distractor and cross-visit questions, the claim that place-identity binding is the causal mechanism is wrong. Equally, if the tau_match=0 ablation or the label-preserving nearest-neighbor ablation matches the full method, the binding invariant contributes nothing and the idea is falsified.

## References

1. Lost in the Hallway: Benchmarking Persistent Spatial Memory Across Revisits
