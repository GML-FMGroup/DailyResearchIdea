# Minimal Causal Patch Search: Finding Sufficient and Necessary Intervention Targets in Transformer Computational Graphs

## Motivation

Activation patching methods like self-patching (Meng et al., Meng et al., etc.) identify intervention locations that improve model accuracy but provide no guarantees that the identified set is causally sufficient (i.e., patching those nodes alone fixes the output) or minimal (no smaller set suffices). This structural lack of causal grounding means interventions may miss critical components or include redundant ones, limiting reliability for debugging or editing. We need a method that explicitly optimizes for a set of nodes whose value changes are both necessary and sufficient to correct a generalization failure, ensuring interventions are causally justified.

## Key Insight

The causal effect of patching a set of nodes is a monotone submodular function of the set, enabling a greedy algorithm that finds a near-minimal sufficient set with theoretical guarantees, and the necessity of each node can be verified via ablation checks.

## Method

### Minimal Causal Patch Search (MCPS)

**(A) What it is**: MCPS takes a transformer model, an input causing a generalization failure (output incorrect), and a reference activation (from a correct example or a forward pass with corrected context). It outputs a set of computational graph nodes (e.g., attention heads or MLP outputs at specific layers and positions) such that simultaneously patching these nodes with reference activations makes the model output correct, and no proper subset does.

**(B) How it works**: The algorithm uses a greedy beam search over subsets of nodes, relying on the monotonicity assumption that adding a node never worsens accuracy.

```python
def mcps(model, input, ref_activations, threshold=0.95, beam_width=5, pairwise_lookahead=True):
    # Input: model, input (token ids), ref_activations dict mapping node->activation
    # Output: set of nodes to patch
    
    # Get all candidate nodes (e.g., all attention heads + MLPs at each layer and each token)
    nodes = get_all_candidate_nodes(model)  # list of node identifiers
    
    # Define f(S) = probability of correct output after patching nodes in S with their ref_activations
    def f(S):
        with patch(model, S, ref_activations):
            logits = model(input)
            return softmax(logits)[correct_token_index]
    
    # Calibration: check monotonicity on a held-out set of 100 random subsets
    calibration_subsets = [random.sample(nodes, k) for k in [1,2,3,4,5] for _ in range(20)]
    monotonicity_violations = 0
    for S in calibration_subsets:
        for v in nodes - S:
            if f(S | {v}) < f(S) - 1e-6:
                monotonicity_violations += 1
    violation_rate = monotonicity_violations / (len(calibration_subsets) * len(nodes))
    if violation_rate > 0.05:
        # Fallback: exhaustive search over pruned candidate set (top-100 nodes by single-node gain)
        gains = [(n, f({n}) - f(set())) for n in nodes]
        top_nodes = [n for n, g in sorted(gains, key=lambda x: -x[1])[:100]]
        # Exhaustive search over all subsets of top_nodes up to size 10
        best_S = None
        best_f = 0
        for size in range(1, 11):
            for S in itertools.combinations(top_nodes, size):
                val = f(set(S))
                if val > best_f:
                    best_f = val
                    best_S = set(S)
                    if val >= threshold:
                        break
            if best_f >= threshold:
                break
        S = best_S if best_f >= threshold else set()
    else:
        # Beam search: maintain top-k subsets by f(S)
        beams = [set()]
        best_S = set()
        best_f = 0
        while beams:
            new_beams = []
            for S in beams:
                if f(S) >= threshold:
                    if f(S) > best_f:
                        best_f = f(S)
                        best_S = S
                    continue
                # Consider adding a new node
                candidates = list(nodes - S)
                for v in candidates:
                    S_new = S | {v}
                    if pairwise_lookahead:
                        # Also consider adding a pair (v, w) to capture non-additive interactions
                        for w in candidates:
                            if w != v:
                                S_new_pair = S | {v, w}
                                new_beams.append((S_new_pair, f(S_new_pair)))
                    new_beams.append((S_new, f(S_new)))
            # Keep top beam_width candidates by f(S)
            new_beams.sort(key=lambda x: -x[1])
            beams = [S for S, _ in new_beams[:beam_width]]
            if not beams:
                break
        S = best_S
    
    # Prune for necessity: after achieving threshold, try removing each node; if removal keeps f >= threshold, keep removed
    S = prune_necessary(S, f)  # returns minimal subset with f >= threshold
    return S
```

Hyperparameters: `threshold=0.95`, `beam_width=5`, `pairwise_lookahead=True`. The `prune_necessary` function iterates through nodes in random order and removes a node if f(S - {node}) still >= threshold.

**(C) Why this design**: We chose a greedy algorithm over exhaustive search because the number of candidate nodes is large (e.g., 144 heads × 12 layers × sequence length = thousands), and exact minimal set is NP-hard. Greedy leverages monotonicity and submodularity to achieve a (1-1/e) approximation guarantee, accepting that the result may be larger than the true minimal set. We use a patch-and-evaluate procedure rather than gradient-based attribution because interventions are counterfactual and discrete; gradients would require continuous relaxation. We include a necessity pruning step because greedy alone can include redundant nodes; pruning adds computational cost but ensures each node in the output set is individually necessary—a key design for causal interpretability. We choose to patch at the granularity of attention head and MLP outputs rather than individual neurons because previous work (Knowledge Circuits, Memory Injections) shows head-level interventions are effective, and finer granularity would explode search space. The calibration check for monotonicity ensures the algorithm's foundation is validated before proceeding; if violated, we fallback to exhaustive search on a pruned set, making the approach robust to non-monotone interactions.

**(D) Why it measures what we claim**: The set S found by MCPS is causally sufficient because patching S raises the probability of the correct token above threshold, and the counterfactual test directly checks the output—no proxy. The f(S) computation measures **causal sufficiency** of S, under the assumption that patching a set of nodes only affects downstream computations through the normal forward pass; this assumption fails when patching induces distributional shift (e.g., attention patterns change unpredictably), in which case f(S) reflects the model's behavior under unnatural activation combinations rather than true repair. We explicitly assume: (1) reference activations are correct for the counterfactual context, (2) no destructive interference beyond monotonicity (i.e., patching a node does not degrade performance when combined with others), (3) the effect of multiple patches is additive (no super-additive or sub-additive interactions). The `prune_necessary` step measures **causal necessity** of each node: if removing a node still yields correct output, that node was not necessary. This relies on the assumption that interventions are modular and do not interact compositionally; when node interactions are non-linear and super-additive, a node may appear unnecessary individually but be necessary in combination—then our necessity check may return false negatives. Despite these failure modes, the method provides the first mechanistically grounded minimal patch set, explicitly naming the assumptions.

## Contribution

(1) A novel algorithm, MCPS, that finds a near-minimal set of causal intervention targets in transformer computational graphs with theoretical approximation guarantees via submodularity. (2) A quantitative framework for evaluating causal sufficiency and necessity of intervention sets, applicable to debugging generalization failures. (3) Empirical demonstration that MCPS recovers smaller intervention sets than prior self-patching heuristics while achieving similar or better correction rates.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | CounterFact dataset (500 test examples) | Tests generalization of individual facts; failure mode is clear (factual error) |
| Primary metric | Top-1 accuracy on counterfactual queries | Measures correct output after patching |
| Baseline 1 | Standard fine-tuned model (no intervention) | Baseline without any intervention |
| Baseline 2 | Self-patching (related work, Meng et al.) | Prior intervention method that patches all high-attribution nodes |
| Baseline 3 | Random node patching (same size as MCPS output) | Control for patch effect |
| Ablation 1 | MCPS without pruning | Ablate necessity check to measure its impact on minimality |
| Ablation 2 | Exhaustive search on small model (2-layer, 2-head GPT-2) | Ground truth minimal set for evaluating greedy approximation error |
| Additional test | GPT-2 medium (6 layers) and LLaMA-7B | Test robustness across model families |
| Additional test | Reasoning tasks (e.g., subject-verb agreement) | Test beyond factual errors |

### Why this setup validates the claim
This experimental design tests the central claim that MCPS finds a causally sufficient and necessary minimal patch set. The CounterFact dataset isolates factual generalization failures, where the model's error can be traced to specific internal nodes. The primary metric, top-1 accuracy, directly measures whether patching restores correct output, establishing causal sufficiency. The baselines compare against no intervention (showing the need for any patch), a sanity check (random patching showing specificity), and a prior method (self-patching) to assess minimality—self-patching patches all high-impact nodes, likely exceeding minimality. The ablation without pruning isolates the necessity check's contribution. Exhaustive search on a small model provides a ground truth minimal set to measure the greedy algorithm's approximation factor. Testing on GPT-2 and LLaMA families with both factual and reasoning tasks evaluates robustness beyond a single setup. We will release open-source code (PyTorch, HuggingFace Transformers) with a tutorial notebook to ensure reproducibility.

### Expected outcome and causal chain

**vs. Standard fine-tuned model** — On a case where the model incorrectly predicts the original relation (e.g., "Messi plays for Barcelona") instead of the counterfactual ("Messi plays for Real Madrid"), the baseline outputs the wrong answer because no intervention is applied. Our method patches the minimal set of nodes (e.g., specific MLP or attention head that stores the original fact), steering the output to the correct token. We expect a large accuracy gain on the subset of failed cases (e.g., >30% improvement) but near-zero change on already-correct examples.

**vs. Self-patching (related work)** — On the same failure case, self-patching patches all nodes with high attribution scores (e.g., via gradients), which may include irrelevant nodes and miss necessary interactions. Our method greedily searches for a minimal set, often finding a smaller patch that achieves correctness. We expect MCPS to match or exceed self-patching's accuracy while using significantly fewer nodes (e.g., 5 vs 15 on average), and to outperform on cases where self-patching introduces interference.

**vs. Random node patching** — On that case, patching a random set of nodes (same size as MCPS output) will rarely fix the error because the chosen nodes are unrelated to the stored fact. The baseline accuracy remains near the original level. Our method's patches are targeted, so we expect a clear gap: MCPS >50% accuracy on failed cases vs. random patching ≤5%.

**Ablation: MCPS without pruning** — On the same failure case, the algorithm will output a superset of the pruned set, sometimes including redundant nodes. The pruned version will use fewer nodes while maintaining the same accuracy (by construction). We expect the pruned set to be 10-30% smaller than the unpruned set, confirming that the necessity check is effective.

**Ablation: Exhaustive search on small model** — On a 2-layer, 2-head GPT-2, we can compute the exact minimal sufficient set via exhaustive search. We expect the greedy algorithm's output to be within a factor of 2 of the true minimal size (consistent with the (1-1/e) approximation under submodularity). If the gap is larger, it indicates a violation of submodularity in this small model.

**Cross-model and cross-task** — We expect similar trends on GPT-2 medium and LLaMA-7B, though the absolute number of patched nodes may scale with model size. On reasoning tasks, the failure may involve multiple interacting nodes; MCPS should still find a minimal set, but the size may be larger. This would validate the method's generality.

### What would falsify this idea
If MCPS fails to consistently find smaller patch sets than self-patching while achieving similar or better accuracy on counterfactual queries, or if its accuracy gain is uniform across all test examples regardless of failure mode, then the claim of minimal causal sufficiency is invalid. Additionally, if the greedy algorithm's output on the small model exceeds the true minimal size by more than a factor of 3, or if monotonicity violations are frequent and fallback exhaustive search still cannot find a solution, then the core assumptions do not hold.

## References

1. Towards Mechanistically Understanding Why Memorized Knowledge Fails to Generalize in Large Language Model Finetuning
2. Hopping Too Late: Exploring the Limitations of Large Language Models on Multi-Hop Queries
3. Knowledge Circuits in Pretrained Transformers
4. How much do language models memorize?
5. Just How Flexible are Neural Networks in Practice?
6. Physics of Language Models: Part 3.3, Knowledge Capacity Scaling Laws
7. Scaling Laws for Fact Memorization of Large Language Models
8. Memory Injections: Correcting Multi-Hop Reasoning Failures During Inference in Transformer-Based Language Models
