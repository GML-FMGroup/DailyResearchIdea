# Hard Temporal Contrastive Learning for Robust Video Understanding

## Motivation

Video-Oasis (from the Idea Cards) reveals that state-of-the-art Video-LLMs perform only marginally above random on video-native challenges after removing shortcut-laden benchmark samples. This demonstrates that current memory-based architectures lack robust temporal reasoning, even when benchmarks are cleaned. The root cause is that training procedures do not explicitly enforce sensitivity to temporal order in cases where shortcuts are absent, leaving models reliant on spurious correlations. We need a training intervention that directly targets the temporal reasoning deficit identified by diagnostic audits.

## Key Insight

Contrastive learning on temporally hard pairs—diagnosed by shortcut audit as video-native where temporal order is critical—forces the model to encode temporal structure because the pairs are perceptually similar but temporally distinct, eliminating surface-form shortcuts.

## Method

**(A) What it is**: Hard Temporal Contrastive Learning (HTCL) is a training method that takes as input a video encoder, a diagnostic audit suite (e.g., Video-Oasis), and a pool of video clips. It outputs an encoder with improved temporal reasoning fidelity by optimizing a contrastive loss on pairs of clips that the audit identifies as temporally challenging and free of non-visual shortcuts.

**(B) How it works** (pseudocode):
```pseudocode
# Input: Encoder f_θ, dataset D, audit suite A
# Step 1: Shortcut audit (one-time, using Video-Oasis methodology)
For each clip c in D:
    score = A.audit(c)  # returns video-native score and shortcut flags
    if score.high_video_native and score.shortcut_free:
        add c to set H (hard temporal candidates)

# Step 2: Construct temporal contrastive pairs
For each clip h in H:
    # Use temporal order swapping to generate a negative
    h_neg = temporal_reorder(h)  # e.g., reverse key segment order
    verify that A.audit(h_neg) is also video-native and not shortcut-laced
    pair = (h, h_neg)

# Step 2.5: Pair verification (calibration)
Sample 512 pairs from constructed pairs; for each, human workers label whether the temporal order is the only distinguishing factor (i.e., they cannot identify the negative without temporal cues). Retain only pairs where human accuracy ≥80% (i.e., at least 80% of workers can correctly identify the temporally correct clip).

# Step 3: Contrastive training (batch size B=64, temperature τ=0.07)
For each batch B pairs:
    # Encode both clips
    z_pos = f_θ(h)      # positive anchor
    z_neg = f_θ(h_neg)  # hard negative
    # Also sample additional negatives from other clips in batch (optional)
    # NT-Xent loss:
    sim = cosine_similarity(z_pos, z_neg)
    loss = -log( exp(sim/τ) / (exp(sim/τ) + sum_{k!=neg} exp(sim(z_pos, z_k)/τ)) )
    update θ via SGD (learning rate 1e-4, momentum 0.9)
```

**(C) Why this design**: We chose contrastive learning over generative objectives (e.g., video prediction) because generative modeling often focuses on pixel-level consistency rather than temporal abstraction, and is computationally heavy. The trade-off is that contrastive learning requires careful negative sampling to avoid trivial solutions—we address this by using audit-verified hard negatives. We chose temporal reordering as the augmentation for negatives over random frame shuffling because random shuffling destroys both temporal and spatial semantics, making the task too easy; temporal reordering preserves low-level perceptual features while altering the causal event structure, forcing the encoder to reason about sequence. We chose to filter candidates through the diagnostic audit rather than using all clips because including shortcut-laced clips would allow the model to rely on non-temporal features, defeating the purpose; the cost is a smaller training set, but the increased difficulty yields better generalization to video-native tasks. Finally, we used a standard NT-Xent loss with temperature τ=0.07 (common in contrastive learning) rather than a margin-based triplet loss because NT-Xent naturally handles multiple negatives and is more stable during training; the trade-off is that it treats all negatives equally, but our hard-negative selection ensures that the most informative negatives are already emphasized through the pair construction. Additionally, we incorporate a human verification step to ensure that hard negative pairs are indeed temporally discriminable and free of non-temporal shortcuts, mitigating the risk that residual shortcuts undermine the contrastive objective.

**(D) Why it measures what we claim**: The computational quantity `cosine_similarity(z_pos, z_neg)` measures **temporal discriminability** because we assume that two clips differing only in temporal order of events should have lower similarity than clips with the same temporal structure; this assumption fails when the temporal reordering does not change the semantic content (e.g., static scenes), in which case the similarity gap reflects spurious visual differences instead. The loss `NT-Xent` measures **temporal reasoning fidelity** because it enforces that the anchor embedding is closer to the positive (which has the correct temporal order) than to the hard negative (which has an incorrect order); this assumes that the positive and hard negative are matched on all non-temporal factors (visual appearance, objects, etc.) via the audit-selected video-native property, so any difference in similarity must be due to temporal order. This assumption fails when the reordering inadvertently introduces a new visual segment that was not present in the original (e.g., if the video is not perfectly segmentable), in which case the loss may reflect low-level visual novelty rather than temporal reasoning. The audit step `A.audit` measures **shortcut absence** by checking solvability without video input; the claim is that clips with high video-native scores and no shortcut flags are truly reliant on visual-temporal reasoning. This assumption fails when the audit's definition of 'video-native' is incomplete (e.g., it misses certain textual shortcuts), in which case the training pairs may still have residual shortcuts, and improvement in contrastive loss could be due to learning those shortcuts rather than temporal reasoning. Overall, HTCL operationalizes temporal reasoning fidelity as the contrastive margin between temporally permuted and correct sequences on diagnostically cleaned data, with the key assumption that the audit removes all non-temporal confounds and that reordering preserves visual identity. The human verification step provides an explicit check: if human workers can identify the negative without temporal cues, the pair is discarded, directly validating that the remaining pairs are temporally discriminable.

## Contribution

['(1) A novel training framework that combines diagnostic shortcut detection (Video-Oasis) with targeted contrastive learning on temporally hard video-native pairs to improve temporal reasoning fidelity.', '(2) A design principle that temporally reordered clips, when verified to be shortcut-free, serve as effective hard negatives for contrastive learning, forcing models to encode event order rather than visual appearance.', '(3) A curated set of temporally challenging video-native pairs derived from existing benchmarks (e.g., Something-Something, Kinetics) using the diagnostic audit, enabling reproducible evaluation.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|-----------------------|
| Dataset | Video-Oasis video-native splits; Something-Something v2; Diving48 | Tests temporal reasoning across diverse event types |
| Primary metric | Temporal ordering accuracy | Directly measures temporal discrimination |
| Baseline 1 | Random guessing | Lower bound for reasoning |
| Baseline 2 | Video-LLaVA (standard training) | Representative state-of-the-art |
| Baseline 3 | SimCLR video (no audit) | Ablates contrastive learning effect |
| Ablation-of-ours | HTCL w/o audit filtering | Isolates audit benefit |
| Ablation-of-ours 2 | HTCL with varying audit threshold (video-native score >0.8, >0.9, >0.95) | Measures sensitivity to shortcut removal |

### Why this setup validates the claim
This combination directly tests the central claim that audit-filtered hard temporal contrastive learning improves temporal reasoning fidelity. The Video-Oasis curated splits ensure that only video-native samples are used, removing non-visual shortcuts that could otherwise explain away performance gains. Adding Something-Something v2 and Diving48 tests generalization to diverse temporal reasoning tasks (fine-grained actions and intentionality). Comparing against random guessing establishes a floor; Video-LLaVA represents the ceiling of current methods that lack our temporal contrastive design; SimCLR video shows whether our gains come from contrastive learning alone or from the audit. The ablation of audit filtering isolates the contribution of clean hard negatives, and varying the audit threshold tests how much shortcut removal drives improvement. The metric of temporal ordering accuracy is falsifiable: if our method truly improves temporal reasoning, accuracy should be higher on temporally challenging clips, but remain equal on temporally ambiguous static clips.

### Expected outcome and causal chain

**vs. Random guessing** — On a case where a clip shows a person picking up a cup then drinking, random guessing expects 50% accuracy. The baseline cannot reason about order because it has no mechanism for temporal understanding. Our method uses contrastive pairs where the negative is temporally reordered (e.g., drinking then picking up), forcing the encoder to distinguish correct sequence. Thus we expect a large accuracy gap (e.g., 70% vs. 50%) on dynamic clips.

**vs. Video-LLaVA** — On a case where a clip contains a subtle temporal anomaly (e.g., a ball rolling then reversing), Video-LLaVA may fail because its training includes shortcut-ridden data that allow it to rely on object presence rather than order. Our method only trains on audit-cleaned clips, so it must learn true temporal cues. We expect our model to outperform Video-LLaVA by a noticeable margin on the video-native subset (e.g., 65% vs. 55%), while performance may be similar on easy static clips.

**vs. SimCLR video** — On a case where two clips differ only in temporal order (e.g., a door opening closing vs. closing opening), SimCLR trained on all clips may learn to rely on shortcuts even if using temporal augmentation, because negative pairs are randomly sampled from other clips, not hard temporally reordered ones. Our method constructs explicit hard negatives from the same clip, forcing temporal discrimination. We expect HTCL to show higher accuracy on temporally sensitive clips (e.g., 70% vs. 60%), while on static clips both may perform near chance.

**vs. Ablation (varying audit threshold)** — As the threshold increases (more stringent shortcut removal), the training set shrinks but contains harder temporal natives. We expect a non-monotonic effect: threshold too low (e.g., >0.8) includes residual shortcuts, reducing gains; threshold too high (e.g., >0.95) may remove too many clips, limiting data. Optimal performance likely near >0.9. This pattern would confirm that audit filtering is crucial and that our threshold choice is valid.

### What would falsify this idea
If HTCL's accuracy gain over SimCLR video is uniform across all clip types rather than concentrated on temporally challenging clips, or if it performs worse than random on any video-native subset, then the central claim is falsified. Additionally, if the human verification step reveals that many pairs are distinguishable by non-temporal cues (i.e., human accuracy <80% on temporal order), then the audit is insufficient and the method's foundation collapses. If varying the audit threshold shows no effect on performance, then shortcut removal is not driving improvements.

## References

1. Video-Oasis: Rethinking Evaluation of Video Understanding
