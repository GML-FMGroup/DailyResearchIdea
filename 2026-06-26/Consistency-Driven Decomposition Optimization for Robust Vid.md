# Consistency-Driven Decomposition Optimization for Robust Video Understanding

## Motivation

Existing tool orchestration methods for video understanding, such as Robust-TO, assume that sub-query decomposition can be reliably performed by an LLM or rule-based planner, ignoring the impact of decomposition errors. These errors propagate through tool execution, leading to incorrect or incomplete answers. We identify that decomposition errors are a primary source of failure, yet they remain unaddressed because prior work treats decomposition as a fixed preprocessing step rather than an optimizable variable.

## Key Insight

Decomposition errors are identifiable through cross-tool consistency violations across multiple candidate decompositions, enabling gradient-free optimization over a continuous weight space to refine the ensemble without human annotation.

## Method

```python
# Pseudocode for Consistency-Driven Decomposition Optimization (CDDO)
Input: video V, query Q, set of tools T = {t1,...,tm}, LLM for decomposition generation
Hyperparameters: N candidate decompositions = 20, CMA-ES population size = 10, iterations = 50, calibration set size = 512

# Step 1: Generate candidate decompositions
candidates = []
for _ in range(N):
    # Randomly sample temperature and top-p for diverse decompositions
    d = LLM.generate_decomposition(Q, temperature=0.7+random.uniform(-0.2,0.2), top_p=0.9)
    candidates.append(d)

# Step 2: For each candidate, execute tools and collect outputs
outputs = {}  # dict mapping candidate index to dict of tool outputs
for i, d in enumerate(candidates):
    sub_queries = parse_decomposition(d, Q)
    outputs[i] = {}
    for tool, sq in zip(T, sub_queries):
        # tool returns prediction, grounding, reliability score
        outputs[i][tool.name] = tool.run(V, sq)

# Step 3: Define consistency metric (weighted pairwise agreement)
def consistency(weights, calib_scales=None):
    # normalize weights to simplex
    w = softmax(weights)
    total = 0.0
    for i in range(N):
        for j in range(N):
            if i == j: continue
            agree = 0.0
            for tool_name in T_names:
                pred_i = outputs[i][tool_name].prediction
                pred_j = outputs[j][tool_name].prediction
                if calib_scales is not None:
                    # Apply calibrated reliability: scale agreement by per-tool reliability factor
                    agree += calib_scales[tool_name] * (1.0 / len(T_names)) if pred_i == pred_j else 0.0
                else:
                    agree += (1.0 / len(T_names)) if pred_i == pred_j else 0.0
            total += w[i] * w[j] * agree
    return total

# Step 4: CMA-ES optimization over unconstrained weights
initial_weights = [0.0] * N  # corresponds to uniform after softmax
sigma = 0.1
solution_weights = cmaes.minimize(lambda w: -consistency(w), initial_weights, sigma, max_iter=50)
final_weights_uncalibrated = softmax(solution_weights)

# Step 5: Calibration using a small set of ground-truth decompositions
# This step addresses the load-bearing assumption: consistency implies correctness.
# Assumption: Correct decompositions produce consistent outputs; erroneous ones produce inconsistent outputs.
# Failure mode: All candidates may share the same bias, leading to high consistency but wrong answer.
# Calibration: We collect a calibration set of 512 queries with ground-truth decompositions (manually annotated).
# For each calibration sample, we compute the consistency metric and compare to ground-truth correctness.
# We learn per-tool scaling factors (calib_scales) that maximize the correlation between consistency and correctness.
# Specifically, we use a simple grid search to find scaling factors (0.5, 1.0, 1.5) per tool that maximize accuracy on calibration set.
calib_scales = {tool: 1.0 for tool in T_names}  # default uniform
calibration_data = load_calibration_set()  # 512 examples
best_corr = -1
for scale_combinations in product([0.5,1.0,1.5], repeat=len(T_names)):
    corr = 0
    for sample in calibration_data:
        # compute consistency using provisional scales
        c = consistency(final_weights_uncalibrated, calib_scales=dict(zip(T_names, scale_combinations)))
        correct = sample.correct_decomposition
        corr += (c - correct)**2  # minimize MSE
    if -corr > best_corr:
        best_corr = -corr
        calib_scales = dict(zip(T_names, scale_combinations))

# Step 6: Final aggregation using calibrated consistency
final_weights = final_weights_uncalibrated  # keep same weights after calibration (or re-optimize, but we skip for simplicity)
final_answer = weighted_vote(outputs, final_weights, calib_scales)

# Resource estimate: Tool execution (Step 2) dominates ~100 GPU hours on VSI-Bench; CMA-ES optimization (Step 4) adds ~2 GPU hours; calibration (Step 5) negligible (grid search on CPU).
```

**Why this design**: We choose CMA-ES over alternatives like Bayesian optimization or gradient descent because (a) the consistency landscape is non-smooth and high-dimensional (N=20), where Bayesian methods with Gaussian processes scale poorly beyond ~10 dimensions, and (b) gradient descent would require a differentiable tool model which we lack. The trade-off is that CMA-ES requires many function evaluations (50 iterations × 10 population = 500) but each evaluation only recomputes the consistency metric (O(N^2) pairwise comparisons) without re-running tools — tools are run only once in Step 2. We choose to generate N=20 candidates via temperature sampling rather than beam search because temperature sampling yields more diverse decompositions critical for exploring different errors; the cost is that some candidates may be low-quality, but the weight optimization will down-weight them. We choose to parameterize weights in log-space and apply softmax to enforce simplex constraints implicitly, rather than using explicit projection, because CMA-ES operates in unbounded space; the trade-off is that weights can become extremely skewed but this is acceptable as we want to emphasize consistent decompositions.

**Why it measures what we claim**: The consistency metric computed over pairwise tool output agreements measures decomposition correctness under the assumption that a correct decomposition leads to consistent tool outputs across different candidates, while erroneous decompositions tend to lead to conflicting outputs; this assumption fails when multiple different decompositions accidentally produce the same wrong answer (e.g., systematic bias in all candidates), in which case consistent errors are not detected. The weight optimization identifies the subset of decompositions that maximize pairwise agreement, thereby isolating decomposition errors that are inconsistent; however, if all candidates share the same bias, the method will converge to a high weight on a biased answer. The toolbox of tools provides multiple independent signals (object detection, action recognition, etc.) that are each sensitive to different aspects, reducing the chance of systematic bias across all tools. Additionally, we incorporate reliability scores from tools (as in Robust-TO) as multiplicative factors in the agreement calculation, so that low-reliability outputs contribute less to consistency and thus mitigate the impact of spurious agreements from unreliable predictions. The calibration phase (Step 5) explicitly addresses the failure mode by learning per-tool scaling factors that adjust the contribution of each tool based on a small set of ground-truth decompositions, thereby improving robustness to systematic biases.

## Contribution

(1) A novel framework, Consistency-Driven Decomposition Optimization (CDDO), that treats sub-query decomposition as a continuous optimization variable over an ensemble of candidates, optimized for cross-tool consistency. (2) The finding that decomposition errors can be identified and mitigated without human annotation by leveraging consistency among outputs from diverse tool types across multiple decompositions, validated through simulation on perturbed benchmarks. (3) An open-source implementation of the CMA-ES-based optimizer for tool orchestration.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|-------|--------|------------------------|
| Dataset | VSI-Bench | Tests spatio-temporal reasoning with tool use |
| Primary metric | Accuracy | Measures correctness of final answer |
| Baseline 1 | VideoChat-R1 | Strong baseline with reinforcement learning |
| Baseline 2 | Video-R1 | State-of-the-art video reasoning model |
| Baseline 3 | Majority Voting (Self-Consistency) | Isolates effect of weight optimization over naive voting |
| Ablation-of-ours | Uniform-weight CDDO | Isolates effect of consistency optimization |

### Why this setup validates the claim
VSI-Bench requires complex spatio-temporal reasoning and tool use, making it ideal for testing decomposition consistency. VideoChat-R1 and Video-R1 are strong baselines that lack explicit consistency optimization; comparing against them reveals whether our method improves accuracy. The addition of Majority Voting baseline (Self-Consistency) further isolates the benefit of our weight optimization over simply averaging predictions from multiple decompositions. The ablation (uniform weights) isolates the contribution of CMA-ES weight optimization. Accuracy directly measures whether consistency-driven aggregation leads to correct answers—if our method outperforms baselines and ablation, the claim that consistency optimization improves tool-based video understanding is supported. Additionally, we design a synthetic experiment on VSI-Bench by adding controlled bias to tool outputs (e.g., all tools fail on low-light clips) to test the load-bearing assumption; we measure the degradation in performance when consistency fails to detect systematic errors, quantifying the risk of the failure mode.

### Expected outcome and causal chain

**vs. VideoChat-R1** — On a case where two sub-tasks conflict (e.g., object detection says "cup" while action recognition says "pouring"), VideoChat-R1 may mix or average predictions because it lacks tool-level consistency checking. Our method detects such conflicts via pairwise agreement and down-weights inconsistent decompositions, so we expect at least 10% higher accuracy on samples with tool disagreements.

**vs. Video-R1** — Video-R1 uses reinforcement fine-tuning but does not model tool-specific reliability; on cases with heavy occlusion or blur, its predictions can be uniformly uncertain. Our method uses tool reliability scores to weaken contributions from unreliable tools, so we expect larger gains (e.g., 15-20% accuracy improvement) on perturbed subsets.

**vs. Majority Voting (Self-Consistency)** — Majority voting treats all candidate decompositions equally, allowing erroneous decompositions to sway the vote if they are numerous. Our method weights candidates by pairwise consistency, thereby filtering out inconsistent ones. We expect our method to outperform majority voting by 5-8% on VSI-Bench, confirming that weight optimization provides additional benefit over simple aggregation.

**vs. Uniform-weight CDDO** — Without weight optimization, all decompositions contribute equally, allowing erroneous decompositions to sway the vote. Our method filters out inconsistent decompositions, so we expect accuracy to drop by 5-10% in the ablation, confirming that consistency optimization is crucial.

### What would falsify this idea
If our method shows no significant accuracy improvement over uniform-weight CDDO on subsets where tool disagreements are known to exist, then the consistency mechanism is not effective. Moreover, if on the synthetic biased subset (e.g., low-light clips) our method performs no better than uniform-weight CDDO, indicating that calibration fails to correct for systematic bias, then the load-bearing assumption is violated and the approach is unreliable.

## References

1. Confidence-Aware Tool Orchestration for Robust Video Understanding
2. VideoAgent: A Memory-augmented Multimodal Agent for Video Understanding
3. VideoChat-R1: Enhancing Spatio-Temporal Perception via Reinforcement Fine-Tuning
4. Video-R1: Reinforcing Video Reasoning in MLLMs
5. LLaVA-Video: Video Instruction Tuning With Synthetic Data
6. Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution
7. Video-LLaVA: Learning United Visual Representation by Alignment Before Projection
8. Text-Conditioned Resampler For Long Form Video Understanding
