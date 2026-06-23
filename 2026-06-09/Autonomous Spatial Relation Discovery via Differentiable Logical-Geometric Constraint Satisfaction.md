# Autonomous Spatial Relation Discovery via Differentiable Logical-Geometric Constraint Satisfaction

## Motivation

Current spatial reasoning models, such as those trained on InternSpatial, rely on predefined question-answer pairs and cannot autonomously discover spatial properties of novel scenes. The fundamental limitation is that spatial relation extraction is always guided by human-defined queries, which restricts agents to answering only those spatial questions that have been explicitly anticipated and annotated. Without a self-supervised signal that does not require human queries, models cannot generalize to unlabeled environments or uncover latent spatial structures.

## Key Insight

Spatial relations are governed by axiomatic properties (asymmetry, transitivity, composition) and geometric projective invariants, which together form a self-consistent supervisory signal that does not depend on human-annotated queries.

## Method

### Explicit Load-Bearing Assumption
We assume that ground-truth spatial relations in any scene satisfy logical axioms (asymmetry, transitivity) and geometric priors (e.g., 'above' implies lower y-coordinate). However, this may fail under occlusion or perspective distortion. To mitigate, we introduce a learned confidence weighting per constraint per pair, downweighting unreliable constraints.

### Method Pseudocode (with specifications)
```python
# Hyperparameters: λ_logical=0.1, λ_geometric=0.5, λ_sparsity=0.01, margin=0.1, confidence_scale=0.5
# Predictor: 2-layer MLP, hidden=512, ReLU, output logits for R=6 relations (left, right, front, behind, above, below)
# Confidence network: 2-layer MLP, hidden=256, ReLU, output scalar weight per constraint per pair (clipped to [0,1])

for each batch of 32 unlabeled images:
    # Step 1: Object detection (frozen Faster R-CNN ResNet-50, pre-extracted features stored on disk)
    objects = detect_objects(image)  # returns bounding boxes, classes, and feature vectors
    
    # Step 2: Predict spatial relations
    relation_logits = predictor(objects['features'])  # shape: (N, N, R)
    relation_probs = softmax(relation_logits, dim=-1)
    
    # Step 3: Compute confidence weights
    confidence_weights = confidence_network(objects['features'])  # shape: (N, N, C) where C= number of constraint types
    # Constraint types: asymmetry per relation (6), transitivity per transitive class (2: left/right, front/behind), geometric per relation (6) => total 14 weights
    
    # Step 4: Logical consistency loss (weighted)
    L_logical = 0
    for each relation type r:
        # Asymmetry: weight w_asym = confidence_weights[i,j,r]
        L_logical += mean( w_asym * P(i,j,r) * (1 - P(j,i,opposite(r))) )
        # Transitivity (for each transitive class): weight w_trans = confidence_weights[i,j,k,trans_class]
        L_logical += mean( w_trans * P(i,j,left) * P(j,k,left) * (1 - P(i,k,left)) )
    
    # Step 5: Geometric consistency loss (weighted)
    L_geometric = 0
    for each pair (i,j) and each relation r with geometric prior:
        # weight w_geo = confidence_weights[i,j,6+index_of_r]
        if r == 'above': violation = relu( (center_y(i) - center_y(j) + margin) )
        L_geometric += mean( w_geo * P(i,j,r) * violation )
    
    # Step 6: Sparsity regularization
    L_sparsity = mean( entropy(relation_probs) )
    
    total_loss = λ_logical * L_logical + λ_geometric * L_geometric + λ_sparsity * L_sparsity
    update predictor and confidence network parameters via gradient descent (Adam, lr=1e-4)
```

(C) Why this design: We chose differentiable logical constraints over post-hoc rule verification because it allows end-to-end training and scales to large scenes where combinatorial search is prohibitive, accepting the cost that not all logical axioms are perfectly continuous (e.g., transitivity with graded probabilities). We used softmax probabilities rather than hard decisions for constraints because it maintains differentiability during training, accepting the trade-off that crisp logical deduction is approximated. We added a sparsity regularization (entropy minimization) to avoid the trivial all-uniform prediction that would satisfy average consistency but not yield informative relations; the cost is a hyperparameter that may need tuning. We excluded human-defined queries entirely, forcing the model to rely solely on internal consistency, which is a deliberate departure from supervised approaches like InternSpatial that depend on extensive annotation. The confidence weighting mechanism allows the model to downweight constraints when geometric evidence is weak, addressing potential failures of the axiomatic assumptions. This design parallels prior work on unsupervised visual relation discovery but is the first to combine logical axioms with geometric priors and adaptive confidence in a differentiable framework.

(D) Why it measures what we claim: The logical consistency loss (L_logical) measures the degree to which predicted relations satisfy asymmetry and transitivity axioms, operationalizing the concept of 'logical validity' under the assumption that ground-truth spatial relations are perfectly consistent with these axioms; this assumption fails when scenes contain cyclic or ambiguous relations (e.g., objects in a circle), in which case the loss reflects the model's internal coherence rather than external truth. The geometric consistency loss (L_geometric) measures alignment between predicted relations and projective geometry (e.g., 'above' implies lower y-coordinate), operationalizing 'geometric plausibility' under the assumption that objects are not rotated or distorted by perspective; this assumption fails when the camera viewpoint introduces foreshortening, in which case the loss penalizes valid relations. The confidence weighting explicitly models these failure modes by downweighting constraints that are unlikely to hold. The sparsity regularization encourages high-confidence predictions, operationalizing 'informativeness' under the assumption that optimal representation uses categorical relations; this assumption fails when soft relations are more appropriate (e.g., 'vaguely left'), in which case the regularization forces unnatural decisions. Together, these components ensure the discovered relations are logically, geometrically, and representationally consistent in well-conditioned scenes.

## Contribution

(1) A self-supervised framework that discovers spatial relations from unlabeled images by enforcing differentiable logical and geometric constraints, eliminating the need for human-annotated queries. (2) The finding that logical axioms (asymmetry, transitivity) combined with geometric priors provide a sufficient supervisory signal for training spatial relation predictors on real-world scenes without any labeled data. (3) Design principles for constructing consistency losses that avoid trivial solutions and scale to multi-object scenes.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | InternSpatial-Bench single-view tasks | Tests spatial relation prediction accuracy. |
| Primary metric | Relation prediction accuracy | Direct measure of correct spatial relations. |
| Baseline 1 | Random uniform prediction | Chance level lower bound. |
| Baseline 2 | Geometric heuristic (threshold on bbox centers) | Tests necessity of learned constraints. |
| Baseline 3 | Fully supervised relation predictor (same architecture, trained with cross-entropy on InternSpatial) | Upper bound with labels. |
| Ablation-of-ours | Ours w/o logical consistency loss | Isolates contribution of logical axioms. |

### Why this setup validates the claim

This experimental design tests the central claim that self-supervised logical-geometric consistency can discover accurate spatial relations without labels. InternSpatial-Bench provides ground-truth relations for direct accuracy measurement. The random baseline sets chance performance; geometric heuristic isolates the benefit of learning beyond fixed thresholds; supervised baseline bounds achievable performance; ablation of logical loss tests whether consistency constraints are the driving factor. The chosen metric (accuracy) directly reflects whether the predicted relation matches the ground truth, making it a falsifiable test: if our method fails to outperform the geometric heuristic, or if the ablation matches the full method, the claim that logical consistency helps is disproven. The addition of a confidence network does not change the fundamental test; it enhances robustness to assumption failures.

### Expected outcome and causal chain

**vs. Random baseline** – On a scene with three objects in a line (A left of B left of C), random guessing yields ~1/8 accuracy per pair (assuming 8 relations, but we use 6). Our method uses transitivity to infer A left of C, giving consistent prediction. We expect a large gap: >60% vs ~16.7% accuracy on all pairs.

**vs. Geometric heuristic** – On a scene where object A is partially occluded behind B, bounding box centers may suggest A is right of B, but geometry is ambiguous. The heuristic fails because it thresholds on centers. Our method applies asymmetry and transitivity from nearby unambiguous pairs to correct this, and the confidence network downweights ambiguous geometric constraints. We expect 10–20% accuracy improvement on occlusion-heavy subsets.

**vs. Supervised** – On a scene with rare relation (e.g., 'above' rarely appears), supervised model may misclassify due to data imbalance. Our method uses geometric prior and logical consistency (e.g., if A is above B, B must be below A) to robustly predict. The confidence network may further help by focusing on reliable constraints. We expect our method to be within 5–10% of supervised accuracy on rare relations, though supervised wins on frequent ones.

**vs. Ablation (ours w/o logical loss)** – On a scene requiring transitive chain (A left-of B, B left-of C), ablation without logical loss may predict A left-of B and B left-of C but fail to enforce A left-of C, leading to inconsistent probability vectors. Our full method enforces transitivity, raising accuracy on triples. We expect a 5–10% gap favoring full method on transitive subsets.

### What would falsify this idea

If our full method performs no better than the geometric heuristic, or if the ablation without logical loss matches the full method (especially on transitive subsets), then the central claim that logical consistency improves discovery is falsified. Similarly, if our method fails to outperform random on any nontrivial subset, the self-supervised approach is unsound. The confidence weighting is a safeguard, but if it downweights too many constraints, it may collapse to geometric heuristic; thus failure on the same subsets would also be evidence against the idea.

## References

1. InternSpatial: A Comprehensive Dataset for Spatial Reasoning in Vision-Language Models
2. Are We on the Right Way for Evaluating Large Vision-Language Models?
3. Is A Picture Worth A Thousand Words? Delving Into Spatial Reasoning for Vision Language Models
4. MMMU: A Massive Multi-Discipline Multimodal Understanding and Reasoning Benchmark for Expert AGI
