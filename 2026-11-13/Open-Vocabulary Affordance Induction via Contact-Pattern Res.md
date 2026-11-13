# Open-Vocabulary Affordance Induction via Contact-Pattern Residual Clustering from Scaled Demonstrations

## Motivation

Existing affordance-grounded language methods, including the fixed-vocabulary grounding of Ahn et al. (2022, 'Do As I Can, Not As I Say'), assume a closed affordance vocabulary defined before data collection, so any interaction pattern absent from that vocabulary is either mislabeled or discarded rather than becoming a new grounded primitive. Ropedia Xperience 2 scales demonstration coverage across 1,589 scenes and 74 tasks but provides no mechanism to expand or validate affordances across that distribution, so affordance identifiability under demonstration distribution shift — the shared residual gap across the affordance-grounding and demonstration-data branches — remains structurally untestable: the vocabulary cannot grow, and generalization cannot be measured against categories that were never allowed to exist.

## Key Insight

Affordances are identifiable exactly when a contact-interaction pattern's residual against the current vocabulary is predictable from object geometry and task context but not from any existing primitive — a condition that yields a closed-form, geometry-conditioned novelty test rather than a learned confidence score, so vocabulary expansion is driven by an identifiable statistical residual instead of a retrieval match or a log-probability gate.

## Method

**What it is.** We introduce OVAI (Open-Vocabulary Affordance Induction), which takes as input a stream of retargeted whole-body demonstrations from Ropedia Xperience 2 — each a trajectory of end-effector poses, contact events, object point clouds, and a task instruction — and outputs a growing affordance vocabulary plus a grounding function mapping instruction phrases to induced primitives. It has three coupled stages: contact-pattern featurization, residual-driven induction, and held-out distribution-shift validation.

**Load-bearing assumption (stated explicitly).** The method assumes that within-primitive contact variation is bounded and geometry-explainable, so any residual that matched geometry cannot explain signals a genuinely unmodeled interaction mode rather than geometric out-of-distributionness. This assumption fails when an object's geometry is simply out-of-distribution (e.g., a novel deformable or thin-shelled object), in which case the residual reflects geometric surprise rather than a new affordance. The calibration procedure in Phase 2 is designed to detect and suppress this failure mode by comparing candidate residuals against a matched-geometry control set, rather than relying on an absolute fit threshold.

**How it works.**

```pseudocode
INPUT: demos D = {(traj, contacts, object_cloud, instruction)}
       base vocabulary V0 (from Ahn et al. 2022)
       residual threshold tau, min support k, geometry regressor G
       control-set size m = 512 episodes, permutation count P = 1000
OUTPUT: expanded vocabulary V*, grounding map g

PHASE 1 - Contact-pattern featurization (per demo segment):
  for each contact episode e in D:
    # geometry-excluded feature vector: no object-frame curvature or contact normal
    phi(e) = [relative end-effector pose to object frame (6D),
              approach direction (3D),
              pre/post-gripper aperture (2D)]
    assign primitive v = argmin_{v in V} dist(phi(e), mu_v)
    residual r(e) = dist(phi(e), mu_v)          # distance to nearest centroid

PHASE 2 - Residual-driven induction (batch over D):
  # Step 2a: build matched-geometry control set
  for each episode e with r(e) > tau:
    find m nearest neighbors in object_cloud feature space (PointNet global descriptor, 1024-d)
    among those neighbors, retain only episodes whose assigned primitive v is in V0
    control_set(e) = retained neighbors (target size m=512; if fewer, use all)
  # Step 2b: geometry-conditioned novelty test
  for each unassigned episode e with r(e) > tau:
    compute null distribution of residuals: {r(e') for e' in control_set(e)}
    fit G: object_cloud features -> r(e') on control_set(e) (PointNet-style regressor, 3-layer MLP head, hidden=256)
    compute held-out R^2 of G on control_set(e) via 5-fold cross-validation
    # calibrated gate: residual must exceed what matched geometry explains
    if r(e) > 95th percentile of null residuals AND G's held-out R^2 < 0.3:
       mark e as candidate novel
  # Step 2c: cluster candidates
  cluster {phi(e) for candidate novel e} with DBSCAN (eps = 5th-nearest-neighbor distance, min_samples = k)
  for each cluster with support >= k:
    add new primitive v_new with centroid mu_new = cluster mean
    V <- V + {v_new}
  # Step 2d: refit G on residuals after each vocabulary update (iterate to fixpoint)
  repeat Steps 2a-2c until no new primitives added or max 5 iterations

PHASE 3 - Held-out distribution-shift validation:
  split D by scene into D_train / D_shift (disjoint scenes, overlapping tasks)
  induce V* on D_train only
  on D_shift: measure (a) fraction of episodes explained by V*,
              (b) grounding accuracy of instruction phrases to V*,
              (c) whether the calibrated novelty test (95th percentile + R^2 < 0.3) on D_shift residuals fires at the same rate as on D_train
  report delta vs. fixed V0 baseline
```

Hyperparameters: tau set per-primitive as the 90th percentile of within-primitive residual on D_train; k = 15 episodes minimum support; DBSCAN eps from 5th-nearest-neighbor distance; G is a small PointNet-style regressor (PointNet global descriptor 1024-d followed by 3-layer MLP with hidden=256, GeLU activation) trained per iteration on the control set with 5-fold cross-validation; control-set size m = 512 episodes; permutation count P = 1000 for null distribution estimation.

**Why this design.** We chose a geometry-conditioned residual test over a learned novelty classifier because a classifier would confound 'novel affordance' with 'hard-to-fit geometry', and the residual-vs-geometry regression separates them by construction, accepting the cost that G must be refit at each vocabulary update (a few minutes per iteration on the Ropedia scale). We chose DBSCAN over k-means for induction because the number of new affordances is unknown a priori and DBSCAN's noise label correctly leaves one-off contact patterns unclustered, accepting the cost that eps must be set from a k-NN heuristic rather than tuned per task. We chose a held-out scene split over a random episode split for validation because random splits leak the same objects into train and shift sets, which would make any vocabulary look general, accepting the cost that scene-disjoint splits reduce effective training support. A domain expert would not describe this as a variant of Ahn et al. (2022) because that work fixes the vocabulary and grounds language into it, whereas here the vocabulary is an output of the residual structure of the demonstration distribution itself — the grounding map is a consequence, not an input.

**Why it measures what we claim.** The residual r(e) measures *affordance novelty* because the assumption is that within-primitive contact variation is bounded and geometry-explainable, so a residual that geometry cannot predict signals a genuinely unmodeled interaction mode; this assumption fails when an object's geometry is simply out-of-distribution (e.g., a novel deformable), in which case r(e) reflects geometric surprise rather than a new affordance, and the calibrated novelty test (95th percentile of matched-geometry control residuals plus G-R^2 < 0.3) is what catches that case. The held-out scene split measures *generalization across demonstration distributions* because the assumption is that scene-disjoint splits induce the same distribution shift that Ropedia's scaling claims to cover; this assumption fails when task overlap between train and shift sets is high enough that the shift is only in appearance, in which case the metric reflects appearance robustness rather than affordance generalization. The DBSCAN support threshold k measures *affordance identifiability* because the assumption is that a real affordance recurs across independent demonstrations; this fails for rare but valid affordances (long-tail tasks in Ropedia's 74-task set), in which case k suppresses them and the vocabulary under-counts.

## Contribution

(1) OVAI, an open-vocabulary affordance induction mechanism that grows a grounded primitive vocabulary from demonstration contact patterns using a geometry-conditioned residual test, replacing the fixed vocabulary assumed by Ahn et al. (2022). (2) A design principle — residual-vs-geometry separability — that distinguishes genuinely novel affordances from geometrically hard-but-known ones, operationalized as a refit-to-fixpoint loop between the vocabulary and the geometry regressor. (3) A held-out scene-disjoint distribution-shift protocol for validating whether an induced vocabulary generalizes across demonstration distributions rather than memorizing scenes.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Ropedia Xperience 2, scene-disjoint split | Retargeted demos with contacts; scene split blocks object leakage |
| Primary metric | Held-out scene-shift affordance coverage delta vs. V0 | Directly tests whether induced vocabulary generalizes |
| Baseline | Fixed V0 (Ahn et al. 2022 grounding) | Tests whether vocabulary expansion is necessary at all |
| Baseline | Residual-only induction (no geometry gate) | Tests whether geometry gate separates novelty from hard geometry |
| Baseline | k-means induction (fixed cluster count) | Tests whether unknown-count DBSCAN matters |
| Ablation-of-ours | OVAI with calibrated novelty test disabled (no 95th percentile + R^2 gate) | Isolates contribution of the novelty-vs-geometry separation |
| Diagnostic | Synthetic dataset with known ground-truth affordances (e.g., 10 object categories, 5 true affordances per category, 100 episodes per affordance) | Provides precision/recall of novelty test against known labels |

### Why this setup validates the claim

The central claim is that the affordance vocabulary should be an *output* of residual structure in the demonstration distribution, and that this output generalizes across scene-disjoint shifts better than a fixed vocabulary. The fixed-V0 baseline tests the necessity sub-claim: if a frozen vocabulary already explains held-out scene episodes, then induction adds nothing and the whole premise collapses. The residual-only baseline tests the separation sub-claim: without the calibrated novelty test, it will absorb geometrically-hard-but-known interactions as "novel," inflating the vocabulary with spurious primitives — so any coverage gain it shows on shift scenes should be accompanied by vocabulary bloat and grounding degradation. The k-means baseline tests the unknown-count sub-claim: forcing a fixed cluster count either splits real affordances or merges distinct ones, which should show up as grounding errors concentrated on the merged/split primitives. The ablation (calibrated test off) is the sharpest test — it is our method minus exactly the mechanism we argue is load-bearing, so if it matches full OVAI, the geometry-conditioned residual test is decorative. The synthetic diagnostic provides a direct precision/recall measurement of the novelty test against known ground-truth labels, which is the only way to independently falsify the load-bearing assumption that the residual separates novelty from geometric surprise. The primary metric — fraction of held-out scene episodes explained by the induced vocabulary — is the right one because it is measured on scenes never seen during induction, so it cannot be gamed by memorization; a gain there is precisely the generalization the claim predicts, and the calibrated novelty test transfer check (does the 95th percentile + R^2 gate fire at the same rate on shift residuals?) confirms the gain is from genuine novelty capture rather than overfitting the geometry regressor.

### Expected outcome and causal chain

**vs. Fixed V0 (Ahn et al. 2022 grounding)** — On a case where a held-out scene contains a contact pattern (e.g., a hook-and-pull interaction) whose featurized contact signature falls far from any V0 centroid, the baseline produces a forced assignment to the nearest existing primitive because its vocabulary is frozen and grounding is a one-way map into it. Our method instead induces a new primitive from the residual cluster because that residual exceeds the matched-geometry control distribution and is not geometry-predictable, so we expect a modest but concentrated coverage gap on shift-scene episodes whose contact signatures are far from V0 centroids, and near-parity on episodes close to known primitives — i.e., the gain should be *localized*, not uniform.

**vs. Residual-only induction (no geometry gate)** — On a case where an object has unusual local curvature (a novel deformable or a thin-shelled object) that makes the contact featurization hard to fit, the baseline produces spurious new primitives because it treats every large residual as novelty, so its vocabulary inflates with geometry artifacts. Our method instead suppresses those residuals via the calibrated novelty test because they are predictable from matched geometry (residual falls below the 95th percentile of the control set or G's R^2 exceeds 0.3), so we expect the residual-only baseline to show a similar or slightly higher raw coverage number but a visibly worse grounding accuracy and a much larger induced vocabulary — the diagnostic is vocabulary size vs. grounding accuracy, not coverage alone.

**vs. k-means induction (fixed cluster count)** — On a case where the true number of novel affordances in a scene batch is, say, three but the fixed count is set to five, the baseline produces split primitives (one real affordance carved into two) because k-means must place all centroids, and on batches with fewer true novelties it merges distinct ones. Our method instead lets DBSCAN's noise label absorb one-off patterns and cluster only supported ones, so we expect grounding errors on shift scenes to be concentrated on the primitives that k-means split or merged, with OVAI showing cleaner per-primitive grounding on exactly those clusters.

**vs. OVAI with calibrated novelty test disabled** — On a case where a novel deformable produces high residuals that are geometry-predictable (because the control set contains similar deformables), the ablation fires the gate and induces spurious primitives, whereas full OVAI suppresses them via the 95th percentile + R^2 < 0.3 test. We expect the ablation to show higher vocabulary size and lower grounding accuracy on shift scenes, with the synthetic diagnostic showing precision/recall degradation (e.g., precision drops from 0.85 to 0.60) because the gate is the only mechanism separating novelty from geometric surprise.

### What would falsify this idea

If the ablation with the calibrated novelty test disabled matches full OVAI on held-out scene coverage *and* grounding accuracy, then the geometry-conditioned residual test is not doing the work we claim and the central mechanism is wrong. Equivalently, if the synthetic diagnostic shows the novelty test has precision below 0.5 against known ground-truth labels, the load-bearing assumption that the residual separates novelty from geometric surprise is falsified.

## References

1. Ropedia Xperience 2: A Gold Standard for Scaling Physical AI Environments and Behaviors
