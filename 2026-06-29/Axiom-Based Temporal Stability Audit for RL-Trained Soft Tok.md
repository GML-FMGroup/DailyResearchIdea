# Axiom-Based Temporal Stability Audit for RL-Trained Soft Token Representations

## Motivation

The 'Formalizing Latent Thoughts' framework provides four axioms to evaluate representational quality, but it assesses only static model snapshots, ignoring the representational drift inherent to RL training. For RL-trained soft token representations, which are optimized via reward signals rather than supervised objectives, stability across training is critical yet unmeasured. This omission means existing evaluations may miss structural failures that emerge during training, such as catastrophic forgetting of separability or collapse of minimality.

## Key Insight

The invariance of axiom-compliance across training phases is a structural signature of representational stability in RL-trained soft tokens, distinct from noise invariance alone.

## Method

### Axiom-Based Temporal Stability Audit (ATSA)

**Load-bearing assumption:** The four axiom metrics (causality, minimality, separability, stability) from Formalizing Latent Thoughts are valid and comparable across RL training checkpoints when applied to soft token representations. We validate this via a calibration step on synthetic RL benchmarks with known drift properties before applying variance-based stability scoring.

**(A) What it is**: Axiom-based Temporal Stability Audit (ATSA) is a method that takes a set of RL-trained model checkpoints and their soft token representations, and outputs a stability score for each of the four axioms (causality, minimality, separability, stability) by tracking their variation across training time. Inputs: list of checkpoints T = {t0, t1, ..., tn} (e.g., every 10% of training steps, n=9 for 10 checkpoints), layer indices L = {l1, l2, ..., lk} (e.g., layers 1, 3, 5, 7, 10 of a 12-layer transformer), a set of diverse reasoning tasks D (e.g., 10 tasks covering spatial, factual, arithmetic, and commonsense reasoning). Output: stability scores S_axiom per layer. Resource estimate: For a 1B parameter model with 10 checkpoints, 10 tasks, and 5 layers, ATSA requires approximately 200 GPU-hours on A100 (40 GB), assuming each axiom computation takes ~4 hours per layer-task-checkpoint.

**(B) How it works**:
```pseudocode
Input: Checkpoints T = {t0,t1,...,tn}, layers L = {l1,l2,...,lk}, tasks D
Output: Stability scores S for each axiom and layer

# Calibration step: Validate axiom metrics on synthetic RL benchmark
# (e.g., controlled drift via noise injection, policy oscillation)
# Ensure calibration AUC > 0.7 for drift detection before proceeding

for each checkpoint t in T:
    load model weights at t
    for each layer l in L:
        reps = get_soft_token_representations(model, l, D)  # list of vectors per task
        causality[t][l] = compute_causality(reps)        # intervention-based metric (Pearl, 2000)
        minimality[t][l] = compute_minimality(reps)      # mutual information with compressed representation
        separability[t][l] = compute_separability(reps)  # average inter-task cluster distance
        stability[t][l] = compute_stability(reps)        # similarity under Gaussian noise (σ=0.01)

for each layer l in L:
    for each axiom in {causality, minimality, separability, stability}:
        scores = [axiom[t][l] for t in T]
        variance = var(scores)
        S[axiom][l] = 1 - variance / max_possible_variance  # normalized so 1=perfect stability
return S

Hyperparameters: checkpoint granularity (e.g., every 10% of training steps), number of tasks (at least 10 to cover diverse reasoning types), noise standard deviation for stability metric (σ=0.01).
```

**(C) Why this design**: We chose to compute axiom metrics at multiple checkpoints rather than only at initialization or final because static evaluation fails to capture training-phase drift, which is the core concern for RL-trained representations. We reused the exact metrics from Formalizing Latent Thoughts rather than designing axioms specific to RL because maintaining comparability with prior work allows direct interpretation of deviations; the trade-off is that these metrics may not be sensitive to RL-specific effects like reward hacking or policy oscillation, so our stability scores may miss some failures. We aggregated stability as inverse variance across time rather than using a trend slope or minimum threshold because variance is a symmetric measure that treats both increases and decreases as instability, appropriate when we care about any departure from consistency; the cost is that we do not distinguish between beneficial improvements (e.g., increasing causality) and harmful degradation (e.g., decreasing minimality), both contribute equally to instability. We chose to evaluate on multiple layers because representational properties can be layer-specific, reflecting the cascade of abstraction; this adds computational overhead but provides a finer-grained diagnostic than a single layer summary.

**(D) Why it measures what we claim**: The variance of causality scores across training measures training-phase stability of causal influence because the causality metric operationalizes the degree to which a representation causally affects output under intervention (e.g., replacing the representation and measuring output change); this assumption fails when the intervention is imperfect (e.g., replacement disrupts internal model coherence beyond the targeted representation), in which case the metric reflects intervention corruption rather than true causal change. The variance of minimality scores across training measures the conservation of information redundancy because minimality quantifies the mutual information between the representation and a compressed version (e.g., via dimensionality reduction); this assumption fails when the compression method is not calibrated to the representation's distribution (e.g., linear PCA on non-linear manifold), reflecting compression bias instead of redundancy. The variance of separability scores across training measures the persistence of task-type discrimination because separability computes the average distance between representation clusters of different task types (e.g., spatial vs. factual); this assumption fails when the task distribution itself shifts across training (e.g., RL focuses on a subset of tasks), reflecting task selection changes rather than representational degradation. The variance of stability scores across training measures the robustness of noise invariance because stability computes representation similarity under small input perturbations (e.g., Gaussian noise); this assumption fails when the noise magnitude is mismatched to the model's sensitivity (too large causes representational collapse, too small yields trivial invariance), reflecting measurement artifact rather than true invariance. **Additionally, the variance-based stability score assumes that training-phase drift is captured by fluctuations in axiom values; this assumption fails when drift is non-stationary (e.g., periodic oscillations), in which case variance may underestimate instability because repeated cycles average out. We therefore include a trend-aware aggregation (monotonicity test) as an ablation to address this failure mode.**

## Contribution

(1) A temporal extension of the four-axiom evaluation framework that measures representational stability across RL training checkpoints, enabling diagnosis of representational drift. (2) An empirical finding that RL-trained soft token representations exhibit significant temporal variance in separability and minimality, while causality remains largely stable. (3) A diagnostic tool to monitor axiom compliance during RL training, allowing early detection of representational collapse or overfitting.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
|------|--------|-----------|
| Dataset | 10 RL reasoning tasks (spatial, factual, arithmetic, commonsense) | Covers diverse reasoning |
| Primary metric | AUC for drift detection | Measures diagnostic accuracy |
| Baseline 1 | Static evaluation (t_final) | Ignores temporal dimension |
| Baseline 2 | Single axiom (causality) | Ignores other axioms |
| Baseline 3 | Average across checkpoints | Ignores variance pattern |
| Baseline 4 | Dynamic Stability Predictor (DSP) — LSTM that predicts stability scores from raw token trajectory statistics (mean, var, autocorrelation) trained on synthetic drift instances | Tests unique value of axiomatic approach |
| Ablation | Min-max range aggregation | Tests variance specificity |
| Ablation 2 | Trend-aware aggregation (monotonicity test) | Tests assumption that variance captures all drift |

### Why this setup validates the claim
The proposed method attributes diagnostic power to temporal variance and axiom diversity. To test this, we create a ground truth of stable vs. unstable training runs by injecting controlled representation drift (e.g., noise injection or policy oscillation) into RL checkpoints. The dataset covers diverse reasoning tasks to ensure generalizability. The primary metric (AUC) directly evaluates whether ATSA can discriminate these runs better than baselines. Each baseline isolates a specific design choice: static evaluation tests the need for temporal tracking, single-axiom tests the value of multiple perspectives, and average tests the insufficiency of mean-level summarization. The DSP baseline tests whether a learned predictor can match the axiomatic approach. The ablations test whether variance aggregation is superior to range-based or trend-aware alternatives. This design ensures that any observed advantage can be traced back to the claimed mechanism.

### Expected outcome and causal chain

**vs. Static evaluation (t_final)** — On a case where representation oscillates early then returns to original at final checkpoint, static evaluation sees no issue, falsely deeming the representation stable. Our method captures variance across all checkpoints, so it correctly flags oscillation as instability. We expect ATSA to achieve AUC >0.8 on oscillatory instances while static evaluation yields AUC ~0.5 (chance).

**vs. Single axiom (causality)** — On a case where causality remains constant but separability degrades (e.g., task clusters merge due to reward hacking), the single-axiom baseline misses the degradation. Our method uses all four axioms, so it detects the separability drop. We expect ATSA to have AUC >0.8 on instances where non-causal axioms are informative, while single-axiom performs near chance (AUC ~0.5) on those same instances.

**vs. Average across checkpoints** — On a case where representations degrade monotonically (causality steadily decreases), the average across checkpoints yields a moderate score, implying false stability. Our method's variance-based aggregation produces low stability, correctly indicating degradation. We expect ATSA to achieve AUC >0.9 on monotonic degradation instances, while average baseline yields AUC <0.6.

**vs. Dynamic Stability Predictor (DSP)** — On cases where representational drift is simple (e.g., monotonic), DSP may match ATSA; but on complex drift (e.g., oscillation), ATSA's axiomatic approach should outperform. We expect ATSA AUC >0.7 on complex drift instances while DSP AUC ~0.6.

### What would falsify this idea
If ATSA's detection AUC is statistically indistinguishable from the average-across-checkpoints baseline on monotonic degradation cases, or if its advantage over static evaluation is absent on oscillatory cases, or if ATSA's AUC is not significantly above DSP on complex drift instances, then the central claim that temporal variance uniquely captures stability is false.

## References

1. Formalizing Latent Thoughts: Four Axioms of Thought Representation in LLMs
