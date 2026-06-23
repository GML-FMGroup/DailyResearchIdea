# Weakly-Supervised Hierarchical Narrative Parsing via Self-Training with Soft Dimension Constraints

## Motivation

Existing narrative analyzers like NarraBERT (Characterizing Narrative Content in Web-scale LLM Pretraining Data) provide flat, sentence-level predictions of 11 narrative dimensions but fail to capture hierarchical structure such as plot arcs and scene transitions. Supervised hierarchical parsers require expensive annotations of tree structures, which is unscalable. The root cause is that narrative structure is inherently hierarchical, yet current methods treat each dimension independently across sentences, ignoring the compositional transitions that define higher-level narrative units.

## Key Insight

Hierarchical narrative structure emerges from the joint distribution of narrative dimension transitions, where major structural boundaries correspond to simultaneous changes in multiple dimensions, providing a soft supervisory signal that can guide self-training without explicit hierarchical labels.

## Method

### (A) What it is
**NarraHier** is a weakly-supervised hierarchical narrative parser that takes a text passage as input and outputs a binary tree where each leaf corresponds to a narrative segment (e.g., scene) and each internal node represents a composite narrative unit (e.g., plot arc). The parser is trained via self-training using soft constraints from a pre-trained flat classifier (NarraBERT) on narrative dimensions.

### (B) How it works
We use a recursive splitting algorithm guided by a scoring function. Below is a pseudocode representation:
```python
def NarraHier(passage, dimension_predictor=NarraBERT, max_depth=5, min_seg_len=2):
    # Step 1: Compute per-sentence dimension scores
    sentences = tokenize(passage)
    # dimension_scores: (n_sentences, n_dimensions) with values in [0,1]
    dimension_scores = dimension_predictor(sentences)
    
    # Step 2: Recursive optimal segmentation via dynamic programming
    # score_seg(span) evaluates goodness of a segment based on dimension distribution
    # split_score(span, split) measures how much splitting at 'split' improves hierarchical structure
    
    def score_seg(start, end):
        # Compute mean dimension vector over span
        mean_dim = mean(dimension_scores[start:end], axis=0)
        # Compute intra-span variance (lower is more coherent)
        variance = var(dimension_scores[start:end] - mean_dim)
        # Reward low variance and penalize length? No, just variance.
        return -variance  # negative variance so higher is better
    
    def split_score(start, split, end):
        # Score of splitting into [start,split) and [split,end)
        return score_seg(start, split) + score_seg(split, end)
    
    # DP table: best_score[start][end] and best_split[start][end]
    n = len(sentences)
    best_score = [[-inf]*n for _ in range(n)]
    best_split = [[None]*n for _ in range(n)]
    for length in range(1, n+1):
        for start in range(0, n-length+1):
            end = start + length
            if length < min_seg_len:
                best_score[start][end] = score_seg(start, end)
            else:
                best_score[start][end] = score_seg(start, end)  # no split baseline
                for split in range(start+1, end):
                    candidate = (split_score(start, split, end) 
                                 + alpha * _hierarchical_bonus(start, split, end))
                    if candidate > best_score[start][end]:
                        best_score[start][end] = candidate
                        best_split[start][end] = split
    
    # Step 3: Reconstruct tree from DP table
    tree = reconstruct_tree(0, n-1, best_split)
    
    # Step 4: Self-training loop (repeat until convergence)
    for iteration in range(max_iter):
        # Generate pseudo-hierarchical structures for unlabeled passages
        pseudo_trees = []
        for passage in unlabeled_corpus:
            pseudo_tree = parse(passage)  # using current model
            pseudo_trees.append(pseudo_tree)
        # Train a neural recursive network (e.g., Tree-LSTM) to predict splits
        # using the pseudo-trees as targets, with additional soft constraints
        # from dimension scores: the predicted splits should align with dimension change points
        # Loss = cross-entropy on split decisions + lambda * soft_constraint_loss
        # soft_constraint_loss = sum over splits of (cosine_distance between mean dim vectors of left and right child)
        # Update model parameters
    return best_tree
```
Hyperparameters: `alpha=0.5` (hierarchical bonus weight), `lambda=0.1` (constraint weight), `max_iter=10`, learning rate 1e-4.

**Load-bearing assumption (explicit):** Intra-segment variance of NarraBERT's per-sentence narrative dimension scores is a reliable proxy for narrative coherence, and the largest decrease in this variance at a split point corresponds to a true narrative boundary. To calibrate this, before self-training, we compute the distribution of variance changes at true boundaries vs. random splits on a calibration set (e.g., 100 annotated trees from StorySeeker) and set a threshold for split significance (e.g., variance decrease must exceed the 90th percentile of random splits). This calibration is applied as a filter on DP splits: only splits with variance decrease above the threshold are considered.

### (C) Why this design
We chose a recursive segmentation based on dimension variance over a fixed-depth neural encoder because it directly operationalizes the theoretical notion that narrative transitions occur when dimension profiles shift (fewer assumptions about neural architecture). The trade-off is that variance-based scoring is a coarse proxy that may miss subtle transitions; we accept this cost because it enables interpretability and avoids overfitting given the small supervised signal. We used self-training to iteratively refine the parser, rather than a single-pass clustering, because the initial splits from variance alone are noisy and self-training with bootstrapped pseudo-trees can improve accuracy by acting as a denoising mechanism. The hierarchical bonus (`alpha`) encourages splitting over monolithic segments, but setting it too high leads to over-segmentation; we tune it on a held-out set of annotated stories from the StorySeeker dataset (Where Do People Tell Stories Online?). Finally, we use a Tree-LSTM for neural refinement in the self-training step, choosing it over a transformer because tree-structured representations naturally align with our hierarchical output, though at the cost of sequential computation that is slower than parallel attention.

### (D) Why it measures what we claim
The computational quantity `variance of dimension scores within a segment` measures *narrative coherence* because a uniform distribution of narrative features across a span indicates a consistent narrative frame; this assumption fails when a single narrative dimension (e.g., temporal reference) is flat while others vary, in which case variance reflects only the most volatile dimension. The `cosine distance between mean dimension vectors of left and right children at a split` measures *narrative transition strength* because a large shift in the multidimensional profile indicates a structural boundary (e.g., change in agency or setting); this assumption fails when the shift is due to random noise rather than a true narrative boundary, in which case the measure reflects feature variability without semantic grounding. The `soft_constraint_loss` in self-training measures *alignment with flat classifier predictions* under the assumption that NarraBERT's per-sentence predictions are locally accurate; this assumption fails when NarraBERT misclassifies a sentence’s narrative role (e.g., due to distribution shift), in which case the constraint incorrectly reinforces the error.

## Contribution

(1) A novel weakly-supervised hierarchical narrative parser (NarraHier) that learns to discover plot-level structure using only soft dimension constraints from a flat classifier. (2) An empirical design principle: narrative dimension transitions (as captured by cosine distance between mean vectors) serve as a reliable signal for hierarchical segmentation, validated across multiple narrative genres. (3) The resulting hierarchical annotations on a large web-scale corpus (NarraDolma), enabling downstream analysis of narrative structure in pretraining data.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset 1 | StorySeeker (100 hierarchical trees) | Tests hierarchical narrative parsing. |
| Dataset 2 | NarrativeQA (news stories, 50 human-annotated trees) | Tests generalizability to news narratives. |
| Dataset 3 | OralStoryBank (oral narratives, 50 human-annotated trees) | Tests generalizability to oral storytelling. |
| Primary metric | Leaf-span F1 | Directly measures segment boundary accuracy. |
| Baseline 1 | Flat thresholding (NarraBERT) | Tests if hierarchy adds value over flat. |
| Baseline 2 | Agglomerative clustering | Tests variance-based DP vs. similarity clustering. |
| Baseline 3 | Random segmentation | Tests if method beats chance. |
| Baseline 4 | NarraHier w/ supervised split predictor (trained on 50 annotated boundaries) | Tests if self-training provides unique benefit over direct supervision. |
| Ablation 1 | NarraHier w/o self-training | Tests contribution of self-training. |
| Ablation 2 | NarraHier w/o calibration threshold | Tests contribution of variance calibration. |
| Human validation | Human coherence ratings on 20 StorySeeker stories correlated with intra-segment variance | Validates the load-bearing assumption. |

### Why this setup validates the claim
The dataset combination covers multiple genres (written stories, news, oral tales) to test generalizability. The flat thresholding baseline isolates the benefit of hierarchical splitting; agglomerative clustering tests whether variance-based DP outperforms a simple similarity-based approach; random segmentation ensures the method captures real patterns. The ablation of self-training and supervised split predictor determines whether the iterative refinement improves beyond initial variance-driven splits and whether soft constraints are beneficial. The human validation directly tests the core assumption that variance correlates with coherence. Leaf-span F1 is chosen because it is the most interpretable and directly comparable metric for segmentation quality, and it reflects the method's core output.

### Expected outcome and causal chain

**vs. Flat thresholding** — On a case where narrative transitions are gradual (e.g., slow shift in agency across several sentences), flat thresholding using cosine distance on NarraBERT outputs may fail to detect a boundary because the dimensional shift is spread out. Our method's variance-based scoring on spans detects that a longer span has higher variance, prompting a split at the point of maximal dimensional change, even if gradual. We expect NarraHier to achieve higher recall on gradual transitions while maintaining precision, resulting in a noticeable F1 gain (e.g., 0.15–0.25 absolute) on stories with such transitions.

**vs. Agglomerative clustering** — On a case where two distinct narrative scenes have similar average dimension profiles (e.g., both are high in character agency and low in temporal reference), agglomerative clustering merges them into one segment because global mean similarity is high. Our method's intra-segment variance penalty forces a split if the combined span exhibits high variance, correctly separating them. We expect NarraHier to have higher precision (e.g., 0.20–0.30 absolute) on clusters of scenes with similar mean but high internal variability.

**vs. Random segmentation** — Random segmentation will produce arbitrary boundaries with F1 near chance (~0.10–0.20). Since NarraHier leverages dimension scores, it should substantially outperform random, serving as a sanity check. We expect a large gap (e.g., 0.30–0.40 absolute F1 gain) confirming the method captures non-random structure.

**vs. Supervised split predictor** — On StorySeeker, the supervised predictor (trained on 50 boundaries) may perform well on those 50 boundaries but generalize poorly. Self-training leverages unlabeled data and soft constraints, so we expect NarraHier to achieve higher F1 (e.g., 0.10–0.15 absolute) on held-out stories, especially where unlabeled data is abundant.

**Human validation** — We expect a significant positive correlation (Spearman ρ > 0.4) between intra-segment variance and average human coherence rating (1–5 scale) on 20 StorySeeker stories, supporting the load-bearing assumption.

### What would falsify this idea
If NarraHier's gains over flat thresholding are uniform across all boundary types (rather than concentrated on gradual transitions), or if the self-training ablation shows no improvement over the initial DP, then the hierarchical mechanism is not addressing the intended failure mode. If human validation shows no significant correlation, the core assumption is invalidated.

## References

1. Characterizing Narrative Content in Web-scale LLM Pretraining Data
2. Stories That Heal: Characterizing and Supporting Narrative for Suicide Bereavement
3. Where Do People Tell Stories Online? Story Detection Across Online Communities
4. Toward a Data-Driven Theory of Narrativity
