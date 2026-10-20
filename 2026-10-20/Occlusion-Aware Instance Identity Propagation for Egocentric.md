# Occlusion-Aware Instance Identity Propagation for Egocentric World-Record Construction

## Motivation

HoliWorld-Foundry (Holi-World) constructs ~80K object-centric QA pairs from egocentric recordings by treating per-frame segmentation masks as independent, so the same physical entity receives inconsistent references under occlusion and scene cuts; the decoupled video segmentation assumption that masks can be processed frame-by-frame without a persistent identity layer is the structural root cause. This is not a local data-cleaning issue but a meta-level failure of record construction: identity is never enforced as a first-class invariant, so every downstream question inherits untraceable object references. The same failure recurs in any system that builds temporally extended records from decoupled per-frame perception, including Holi-World's static+dynamic perception pipeline.

## Key Insight

A stable object-instance ID is an invariant that must be enforced at record-construction time by a learned occlusion-aware re-identification module whose identity assignments are validated by a cycle-consistency constraint across the full temporal window, rather than assumed to emerge from per-frame segmentation.

## Method

**What it is.** We introduce **IPL** (Instance-Persistence Layer), a module inserted between HoliWorld-Foundry's decoupled per-frame segmentation output and its QA generator. IPL takes as input a sequence of per-frame masks with appearance embeddings and outputs, for every mask, a stable integer instance ID that persists across frames and scene cuts; the QA generator then conditions object-centric questions on these IDs and rejects any QA pair whose referenced ID is not traceable through the entire temporal window.

**How it works.**

```pseudocode
# Input: frames F[1..T], per-frame masks M[t] = {m_t^1..m_t^{k_t}}
#        with appearance emb e_t^i in R^256 (from a frozen DINOv2)
#        and depth d_t^i (from a frozen depth model)
# Output: ID assignment a_t^i in Z+ for every mask, plus traceability flag

Phase 1: Per-frame descriptor
  for each mask m_t^i:
    e_t^i  = DINOv2(m_t^i)                       # 256-d appearance
    s_t^i  = [area, centroid_x, centroid_y, mean_depth]  # 4-d geometry
    h_t^i  = MLP([e_t^i ; s_t^i])                # 260 -> 128, learned

Phase 2: Occlusion-aware matching between consecutive frames
  for t in 1..T-1:
    C = cosine_sim(h_t, h_{t+1})                 # k_t x k_{t+1}
    C = C - lambda_occ * OcclusionPenalty(m_t, m_{t+1})  # lambda_occ=0.3
    # OcclusionPenalty = 1 if mask m_t^i's projected region is covered
    # by a *different* mask in frame t+1 (depth-ordered), else 0
    A = Hungarian(C)                             # one-to-one assignment
    assign IDs via A; unmatched masks get new IDs

Phase 3: Scene-cut handling
  detect cut when mean cosine_sim(h_t, h_{t+1}) < tau_cut (tau_cut=0.2)
  at a cut: do NOT propagate IDs across the cut by default;
  instead run a re-identification pass over a window of +/-W frames (W=15)
  using the same matching in Phase 2, so an object that reappears
  after the cut receives its prior ID only if matched.

Phase 4: Cycle-consistency validation (the invariant)
  for each ID with >=3 occurrences at frames t1 < t2 < t3:
    require match(t1->t2) o match(t2->t3) == match(t1->t3)
    if violated: split the ID into two IDs at the violating frame
    record traceability[id] = (first_frame, last_frame, num_gaps)

Phase 5: QA gating
  for each candidate QA pair referencing instance ID k:
    accept iff traceability[k] covers the question's temporal window
    AND num_gaps <= G_max (G_max=1)
```

**Why this design.** We chose a learned appearance+geometry descriptor over a pure appearance descriptor because egocentric scenes contain many visually similar objects (cups, hands, packaging) and geometry (depth-ordered centroid) is the only cue that survives occlusion, accepting the cost that depth-estimation errors propagate into matching. We chose a depth-ordered occlusion penalty over a learned occlusion classifier because the penalty must be computed per candidate pair at inference time and a classifier would need per-scene training data that HoliWorld-Foundry does not provide, accepting the cost that the penalty is a hard 0/1 signal rather than a graded one. We chose to *not* propagate IDs across scene cuts by default and instead re-identify over a +/-15-frame window, because naive propagation across cuts is exactly the failure mode that decoupled video segmentation exhibits; the cost is that objects appearing only briefly after a cut may be assigned a new ID. We chose cycle-consistency as a hard validation constraint rather than a soft loss because it is the only signal that can detect an identity swap without ground-truth IDs, accepting the cost that it splits some true identities when a genuine appearance change occurs. This is not a variant of Tracking Anything with Decoupled Video Segmentation: that method performs per-frame association and leaves identity persistence as an emergent property, whereas IPL enforces persistence as a validated invariant and uses the validation to gate downstream QA.

**Why it measures what we claim.** The cycle-consistency violation rate measures *identity persistence* because the assumption is that a correct identity assignment is transitive across time (match(t1->t2) composed with match(t2->t3) equals match(t1->t3)); this assumption fails when an object undergoes a genuine appearance change (e.g., a cup is rotated to show its unlabeled side), in which case the violation reflects appearance change rather than an identity error. The occlusion penalty measures *occlusion-awareness* because the assumption is that a mask covered by a different depth-ordered mask in the next frame cannot be the same physical entity at that location; this assumption fails when two objects genuinely swap depth order (e.g., a hand passes behind then in front of a cup), in which case the penalty incorrectly suppresses a valid match. The traceability window measures *temporal traceability* because the assumption is that an ID with no gaps over the question window refers to the same physical entity throughout; this assumption fails when a gap is filled by an incorrect re-identification, in which case traceability reports a false positive.

**Contribution.** (1) IPL, an occlusion-aware instance-persistence layer with a cycle-consistency invariant that converts decoupled per-frame masks into traceable instance IDs and gates QA generation on traceability. (2) A design principle: identity persistence must be enforced as a validated invariant at record-construction time, not assumed from per-frame segmentation, and cycle-consistency is a sufficient validation signal without ground-truth IDs. (3) A traceability-filtered version of HoliWorld-Data with a reported rejection rate and per-ID traceability metadata, usable as a diagnostic for future record-construction pipelines.

## Contribution

(1) IPL, an occlusion-aware instance-persistence layer with a cycle-consistency invariant that converts decoupled per-frame masks into traceable instance IDs and gates QA generation on traceability. (2) A design principle: identity persistence must be enforced as a validated invariant at record-construction time, not assumed from per-frame segmentation, and cycle-consistency is a sufficient validation signal without ground-truth IDs. (3) A traceability-filtered version of HoliWorld-Data with a reported rejection rate and per-ID traceability metadata, usable as a diagnostic for future record-construction pipelines.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | HOLIWORLD-Data held-out recordings with human-annotated HOLIWORLD-Bench | Provides independent identity/QA reference on real ego scenes |
| Primary metric | QA accuracy on object-centric questions gated by traceability | Directly tests whether validated persistence yields correct answers |
| Baseline | Decoupled per-frame segmentation + Hungarian matching (no invariant) | Tests whether validation, not matching, drives the gain |
| Baseline | TADeD-style emergent identity propagation | Tests persistence-as-emergent-property against enforced invariant |
| Baseline | IPL without QA gating (IDs produced, all QA kept) | Isolates the gating contribution from ID quality |
| Ablation-of-ours | IPL with cycle-consistency validation removed | Tests whether the invariant is the causal ingredient |

### Why this setup validates the claim

The central claim is that identity persistence must be enforced as a validated invariant and that this validation, not better matching alone, is what makes downstream object-centric QA correct. The design isolates that claim by holding the descriptor, matching, and depth cues fixed while varying only the presence of the cycle-consistency invariant and the QA gate. The decoupled-segmentation baseline tests the sub-claim that per-frame association leaves persistence emergent and therefore produces identity swaps that corrupt QA. The TADeD-style baseline tests whether a strong existing decoupled tracker already achieves persistence without an explicit invariant. The no-gating baseline separates the value of producing good IDs from the value of refusing untraceable QA pairs. The ablation removes only the invariant, so any accuracy drop it produces is attributable to the invariant rather than to the descriptor or matcher. QA accuracy on object-centric questions is the right metric because it is the downstream quantity the pipeline actually claims to protect: if validated persistence does not improve answers on questions whose temporal windows span occlusions and cuts, the claim is empty. Because the gate rejects untraceable pairs, we also track rejection rate to confirm the mechanism is active rather than inert.

### Expected outcome and causal chain

**vs. Decoupled per-frame segmentation + Hungarian matching** — On a case where a hand occludes a cup and then withdraws, per-frame matching reassigns the cup's mask to a new ID because appearance alone is ambiguous among similar objects and no cross-frame check exists. Our method applies the depth-ordered occlusion penalty and then the cycle-consistency check, splitting or repairing the swap before it reaches QA. We expect a clear accuracy gap concentrated on occlusion-heavy question subsets, with near-parity on unoccluded subsets.

**vs. TADeD-style emergent identity propagation** — On a case where a scene cut separates two views of the same packaging, emergent propagation either drops the identity or merges it with a visually similar neighbor because persistence is never verified. Our method re-identifies over the +/-15-frame window and only accepts the match if the transitivity check holds. We expect our gain to be largest on cut-spanning questions, with a modest but visible edge on continuous segments.

**vs. IPL without QA gating** — On a case where an identity is split by a genuine appearance change, the ungated pipeline still emits QA pairs referencing the now-untraceable ID, producing confident but wrong answers. Our gated pipeline rejects those pairs, so remaining answers are more often correct. We expect higher accuracy at the cost of a nonzero rejection rate, with the accuracy gain scaling with the fraction of pairs the gate removes.

### What would falsify this idea

If the accuracy gain over the decoupled baseline is uniform across occlusion-heavy, cut-spanning, and clean subsets rather than concentrated where the predicted identity-swap failure occurs, then the invariant is not the causal mechanism and the central claim is wrong. Likewise, if removing cycle-consistency (the ablation) leaves accuracy unchanged while the gate alone accounts for all gains, the claim that validation is the sufficient signal collapses.

## References

1. Holi-World: Building Holistic World Intelligence from Human-Ego Experience
