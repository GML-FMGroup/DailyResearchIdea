# Checkpoint-Consistent SAE Feature Labels via Semantic Drift Certification

## Motivation

Auto-interpretation in SAE-based detector pipelines (e.g., the one-shot labeling oracle of Building Auditable AI Text Detectors with Sparse Autoencoders) treats feature labels as a fixed vocabulary, yet the detector's residual stream is repeatedly perturbed by fine-tuning, so a label that was faithful at checkpoint t may silently denote a different concept at t+k. This unchecked label drift structurally invalidates downstream audits of coverage and necessity, because an unstable label makes any claim that the concept set is sufficient or necessary meaningless — the audit is measuring a vocabulary that no longer exists.

## Key Insight

Label stability under detector perturbation is not a property of the language model's descriptions but of the local geometry of the SAE feature's activation support: if the feature's top-activating probe examples remain semantically clustered around the same concept across checkpoints, the label is an invariant of the feature, and drift can be certified by measuring the movement of that support in a fixed embedding space rather than by re-reading the label text.

## Method

```pseudocode
# CDC-SAE: Checkpoint Drift Certification for SAE feature labels
Input: detector checkpoints {theta_0, theta_1, ..., theta_T} (theta_0 = labeled checkpoint),
       SAE trained on theta_0, auto-interpretation labeler L, fixed sentence encoder E (all-MiniLM-L6-v2, 384-d, frozen),
       probe set P = {(x_i, y_i)} held out from detector training, |P| = 4096,
       per-feature soft-support bandwidth sigma (set to median pairwise E-distance in P),
       semantic drift thresholds tau_support, tau_label (calibrated per SAE width; see step 0),
       fidelity floor phi = 0.6 (auto-interp fidelity score in [0,1]).
Output: certified label set C, flagged feature set F, frozen-weight mask M.

# Load-bearing assumption (stated explicitly):
#   The activation-weighted embedding centroid of a feature over the probe set P is a
#   sufficient statistic for the concept that feature encodes, so that its cosine drift
#   between checkpoints measures genuine concept migration rather than probe-set sampling
#   noise or global activation-scale changes. This assumption is tested in step 0 and in
#   the calibration experiment; it fails when P is unrepresentative of the detector's
#   training distribution or when activations rescale under fine-tuning.

0. Calibration of tau_support and tau_label (run once before step 1):
     a. Null distribution for support drift: for each feature f, compute centroid_0[f]
        on P, then recompute centroids on B=20 random half-splits of P at theta_0;
        tau_support[f] <- 95th percentile of {1 - cos(centroid_half_b[f], centroid_0[f])}.
     b. Null distribution for label drift: for each feature f, draw 10 paraphrases of
        label_0[f] from L, embed with E, and set tau_label[f] <- 95th percentile of
        {1 - cos(E(paraphrase), emb_0[f])}.
     c. Report the null distributions; features whose null 95th percentile exceeds 0.15
        are marked low-confidence and excluded from C regardless of later drift.

1. For each SAE feature f at theta_0:
     a. label_0[f] <- L(description of f)                 # original auto-interp label
     b. emb_0[f] <- E(label_0[f])                          # fixed reference embedding
     c. Normalize activations: a_0[f, x] <- relu(act_f(x)) / max_{x' in P} relu(act_f(x'))
        for all x in P (per-checkpoint max-normalization, recomputed at each theta_t).
     d. Soft support weights: w_0[f, x] <- exp(-||E(x) - E(x*)||^2 / (2 sigma^2)) where
        x* = argmax_{x in P} a_0[f, x]; equivalently use kernel density over full P.
     e. centroid_0[f] <- sum_x w_0[f,x] * a_0[f,x] * E(x) / sum_x w_0[f,x] * a_0[f,x]
        (activation-weighted embedding centroid over the full probe set).

2. For each fine-tuning update theta_t -> theta_{t+1}:
     a. Recompute normalized activations a_{t+1}[f, x] on P under theta_{t+1}.
     b. For each feature f:
          w_{t+1}[f, x] <- exp(-||E(x) - E(x*_{t+1})||^2 / (2 sigma^2)),
            x*_{t+1} = argmax_x a_{t+1}[f, x]
          centroid_{t+1}[f] <- weighted mean of E(x) with weights w_{t+1}*a_{t+1}
          drift[f] <- 1 - cos(centroid_{t+1}[f], centroid_0[f])   # support drift
          relabel[f] <- L(description of f at theta_{t+1})
          sem_drift[f] <- 1 - cos(E(relabel[f]), emb_0[f])        # label drift
     c. For each feature f, certify if BOTH:
          drift[f] <= tau_support[f]   AND   sem_drift[f] <= tau_label[f]
          AND fidelity(relabel[f]) >= phi
        else flag f into F.

3. Downstream classifier training on certified features:
     - Train logistic regression on activations of C only.
     - For f in F: freeze its classifier weight at its theta_0 value (no gradient updates)
       for the first W = 500 warm-up steps, then allow re-learning with a doubled L2 penalty
       (lambda_F = 2 * lambda_C, lambda_C = 1e-3).

4. Report per-checkpoint certification rate, drift distribution, and a
   label-vocabulary churn score = |F| / |C ∪ F|.

# Cost bound: full sweep over T=10 checkpoints, |P|=4096, SAE width 16k features:
#   activation recompute ~0.5 GPU-hr/checkpoint on one A100; labeler L ~0.1 GPU-hr/checkpoint;
#   total ~6 GPU-hr. Subsampling option: certify only features with activation mass
#   (sum_x a_t[f,x]) above the 50th percentile, or every m=2nd checkpoint with linear
#   interpolation of drift; ablation reports certification quality under subsampling.
```

**Why this design.** We chose a fixed sentence encoder E (all-MiniLM-L6-v2) over the labeler's own embeddings because the labeler is the object under audit and cannot be its own referee, accepting the cost that E may under-represent domain-specific concept distinctions; we chose activation-weighted support-centroid drift over pure label-text drift because a label can be paraphrased identically while the feature's activation support migrates to a different concept, accepting the cost of a second drift signal and a threshold tau_support that must be calibrated per SAE width; and we chose freeze-then-relearn with doubled L2 over immediate re-learning because immediate re-learning would let flagged features re-enter the classifier before their drift is resolved, accepting the cost that certified-feature training is slower to adapt to genuine distribution shift. A domain expert would not describe this as standard checkpoint ensembling or as the original one-shot auto-interpretation pipeline of Building Auditable AI Text Detectors with Sparse Autoencoders, because the mechanism is a two-signal certification gate (support-centroid drift AND label-embedding drift) applied after every update, not a single labeling pass at theta_0.

**Why this measures what we claim.** The support-centroid drift `1 - cos(centroid_t[f], centroid_0[f])` measures label stability because it assumes the activation-weighted probe-set centroid is a sufficient statistic for the concept the feature encodes; this assumption fails when the probe set is not representative of the detector's training distribution or when activations rescale globally under fine-tuning, in which case the metric reflects probe-set leakage or activation-scale change rather than concept migration. The label-embedding drift `1 - cos(E(relabel[f]), emb_0[f])` measures semantic faithfulness of the re-read label because it assumes E places paraphrases of the same concept close together and distinct concepts far apart; this assumption fails for fine-grained concept pairs (e.g., 'hedging' vs. 'qualification'), in which case the metric reflects encoder coarseness rather than genuine relabeling. The fidelity floor phi measures whether the re-read label is interpretable at all because it assumes the auto-interpretation fidelity score is monotone in human agreement; this assumption fails for features with sparse activation support, in which case phi reflects labeler verbosity rather than label quality. Certification requires all three signals to agree, so a feature enters the certified set only when support geometry, label semantics, and fidelity jointly support the original label.

## Contribution

['A checkpoint-consistency certification protocol (CDC-SAE) that defines label stability as a two-signal invariant — activation-support centroid drift and label-embedding drift — and gates downstream classifier weights on it, turning an unchecked assumption into an auditable per-feature property.', 'The finding that SAE feature labels drift non-uniformly across fine-tuning updates, with a small subset of features accounting for most vocabulary churn, and that freezing the classifier weights of flagged features during warm-up preserves OOD detection accuracy relative to re-learning all features.', 'A released diagnostic suite reporting per-checkpoint certification rates and a label-vocabulary churn score for SAE-based detector pipelines, enabling comparison of SAE training and detector fine-tuning schedules by label stability rather than only by detection accuracy.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | AI-text detection with OOD shift (in-dist vs. out-of-domain generators); probe set P = 4096 held-out examples | Tests whether certification tracks real concept migration |
| Primary metric | Certified-set accuracy gap vs. original SAE classifier on OOD subset | Directly measures value of the certification gate |
| Baseline | Original one-shot SAE auto-interp classifier (theta_0 labels) | Tests sub-claim: single labeling pass degrades under drift |
| Baseline | Global-fidelity-threshold SAE filter (phi=0.6 only) | Tests sub-claim: fidelity alone misses semantic drift |
| Baseline | Full fine-tuned neural detector | Tests sub-claim: opaque detector is the drift source |
| Baseline | Checkpoint-ensembling SAE (average feature activations across checkpoints, label once) | Tests novelty: gate is not just ensembling |
| Baseline | Naive re-label-every-checkpoint SAE (re-run L at each theta_t, no drift gate) | Tests novelty: gate is not just re-labeling |
| Ablation-of-ours | Certification with label-drift signal removed (support-centroid only) | Isolates whether both signals are jointly needed |
| Ablation-of-ours | Certification with hard top-k support (k=32) instead of soft weighted centroid | Tests the sufficient-statistic assumption repair |
| Calibration experiment | For 200 sampled features, vary k in {8,32,128} and probe composition (full P vs. random 25% subset); correlate support-centroid drift with expert-judged concept migration (binary label from 2 annotators, Cohen's kappa reported) | Directly tests the load-bearing assumption and reports the k/probe regime where it holds |
| Encoder validation | For 100 concept pairs (e.g., 'hedging' vs. 'qualification'), measure E cosine separation vs. human similarity ratings; report Spearman correlation | Validates E's concept geometry before trusting the gate |
| Transfer check | Repeat certification on a second SAE pipeline (different detector architecture) and a non-detection classification task; report churn-score correlation with downstream audit validity (coverage/necessity audit pass rate) | Tests pipeline-specificity |

### Why this setup validates the claim

The central claim is that a two-signal certification gate applied after every checkpoint update preserves downstream detection accuracy where a single theta_0 labeling pass does not, specifically because it catches features whose activation support or label semantics migrate during fine-tuning. The one-shot auto-interp baseline tests the first sub-claim: if that baseline's accuracy collapses on the OOD subset while ours holds, the degradation is attributable to stale labels rather than to the detector itself. The global-fidelity baseline tests the second sub-claim: fidelity alone cannot see support or semantic migration, so its accuracy should track the one-shot baseline rather than ours. The checkpoint-ensembling and naive re-label baselines test the novelty claim: if our gate does not beat both on the OOD label-migration subset, the mechanism is not doing work beyond averaging or re-reading. The opaque detector establishes the ceiling and confirms the drift originates in fine-tuning. The support-centroid-only ablation removes label drift while keeping support drift, so if our gain persists without it, the two-signal claim is unsupported. The hard-top-k ablation and the calibration experiment directly test the load-bearing sufficient-statistic assumption: if soft weighting and per-checkpoint normalization do not improve correlation with expert-judged concept migration over hard top-k, the repair is unnecessary; if they do, the assumption is validated in the reported k/probe regime. The encoder validation checks that E's geometry preserves the concept distinctions the gate relies on. Accuracy gap on the OOD subset is the right metric because the predicted failure mode is drift-driven mislabeling, which surfaces as OOD-specific accuracy loss, not in-distribution loss.

### Expected outcome and causal chain

**vs. Original one-shot SAE auto-interp classifier** — On a case where fine-tuning shifts a feature's activation support from 'hedging' to 'qualification' while its theta_0 label still reads 'hedging', the baseline produces confidently wrong classifier weights on that feature because its label was frozen at theta_0 and never re-audited. Our method instead flags that feature into F via support-centroid drift above the calibrated tau_support[f], freezes its weight during warm-up, and re-learns with doubled L2, so we expect a noticeable accuracy gap on the OOD generator subset but near-parity on in-distribution text.

**vs. Global-fidelity-threshold SAE filter** — On a case where a re-read label is fluent and scores high fidelity but now names a different concept than at theta_0, the baseline produces a feature admitted as stable because its gate is fidelity-only and blind to semantic movement. Our method instead computes label-embedding drift against the fixed E reference and flags the feature when cosine distance exceeds the calibrated tau_label[f], so we expect the baseline's certified set to shrink slowly while ours shrinks sharply at the checkpoint where the concept actually migrates, producing a divergence in certified-set accuracy concentrated at that checkpoint.

**vs. Full fine-tuned neural detector** — On a case where the detector's residual stream drifts under OOD fine-tuning, the opaque baseline produces accurate in-distribution predictions but unstable feature semantics that cannot be localized, because it exposes no auditable feature set at all. Our method instead surfaces per-checkpoint drift distributions and a label-vocabulary churn score, so we expect comparable in-distribution accuracy but a measurable OOD accuracy advantage plus a diagnostic churn signal that the opaque detector cannot supply.

**vs. Checkpoint-ensembling SAE** — On a case where averaging activations across checkpoints blurs a feature that migrated from 'hedging' to 'qualification', the baseline produces a single label that fits neither concept and classifier weights that are miscalibrated on both. Our method instead certifies per checkpoint and flags the migrating feature, so we expect our OOD accuracy on the label-migration subset to exceed the ensembling baseline by a margin that grows with the number of checkpoints spanned by the migration.

**vs. Naive re-label-every-checkpoint SAE** — On a case where the re-read label is a paraphrase that E places close to the original but the activation support has moved, the baseline produces a certified label because it has no drift gate and accepts every re-read. Our method instead requires support-centroid drift to also pass, so we expect the naive baseline to admit features that our gate flags, yielding lower OOD accuracy for the baseline concentrated on support-migration cases.

**vs. Ablation (support-centroid only)** — On a case where activation support stays put but the re-read label silently shifts to a near-synonym with different downstream semantics, the ablation produces a certified label that passes the gate because it lacks the label-drift check. Our method instead requires both signals to agree, so we expect the ablation to recover most of our gain on support-migration cases but lose it on label-migration cases, yielding a gap concentrated on the label-drift subset.

**vs. Ablation (hard top-k support)** — On a case where global activation rescaling under fine-tuning changes the identity of the top-k set without semantic change, the hard-top-k ablation produces spurious support drift and over-flags features. Our method instead uses activation-weighted soft support with per-checkpoint normalization, so we expect the hard-top-k ablation to have a higher false-flag rate and lower certified-set accuracy, with the gap largest at checkpoints with the largest activation-scale change.

**Calibration experiment** — We expect support-centroid drift to correlate with expert-judged concept migration (Spearman rho > 0.5) in the soft-weighted, full-P regime, and to degrade (rho < 0.3) under hard top-k or 25% probe subsampling, identifying the regime where the sufficient-statistic assumption holds.

**Encoder validation** — We expect E cosine separation to correlate with human similarity ratings (Spearman rho > 0.6) for coarse concept pairs but to degrade for fine-grained pairs, bounding where the label-drift signal is trustworthy.

**Transfer check** — We expect the churn score to correlate negatively with downstream audit pass rate (coverage/necessity) on the second pipeline and task, indicating the diagnostic transfers beyond the original detector.

### What would falsify this idea

If our accuracy advantage over the one-shot, global-fidelity, checkpoint-ensembling, and naive re-label baselines is uniform across all subsets rather than concentrated on the OOD and label-migration subsets where the predicted failure mode lives, the certification gate is not doing the causal work we claim. Equally falsifying: if the support-centroid-only ablation matches our full method everywhere, the two-signal claim is wrong; and if the calibration experiment shows support-centroid drift does not correlate with expert-judged concept migration in the soft-weighted regime, the load-bearing sufficient-statistic assumption is false.

## References

1. Building Auditable AI Text Detectors with Sparse Autoencoders
