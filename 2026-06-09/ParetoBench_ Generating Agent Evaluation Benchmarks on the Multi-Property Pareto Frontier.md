# ParetoBench: Generating Agent Evaluation Benchmarks on the Multi-Property Pareto Frontier

## Motivation

Existing benchmark generation methods, such as BeTaL, optimize for a single property (e.g., difficulty) while neglecting others like diversity and robustness, leading to benchmarks that may be narrow or brittle. This ignores the inherent trade-offs among quality dimensions, which is a structural limitation shared across multiple research trees (e.g., automated benchmark design, co-evolution environments, and function-calling datasets). Without explicitly modeling these trade-offs, generated benchmarks cannot guarantee optimal balance across properties, limiting their diagnostic power for agent evaluation.

## Key Insight

The Pareto frontier provides a structural decomposition that transforms multi-objective benchmark design into a conditional generation problem, where each generated benchmark corresponds to a unique optimal trade-off among properties, eliminating the need to manually balance conflicting objectives.

## Method

(A) **What it is**: ParetoBench is a conditional generative framework that takes a desired property vector (a point on the Pareto frontier defined by anchor benchmarks) and produces a benchmark configuration (e.g., task template and parameters) that optimally achieves those properties. (B) **How it works**:

```pseudocode
Input: Anchor benchmarks {B_i} with property vectors {p_i} (difficulty, diversity, robustness); target property vector p_target on frontier.
Output: Benchmark configuration B_gen.

1. Train a property predictor P (2-layer MLP with GeLU, hidden size=256) on anchor benchmarks to map benchmark configurations to property vectors.
2. Train a Pareto classifier C (binary logistic regression with L2 regularization, C=1.0) on anchor property vectors: label frontier points (non-dominated) as 1, dominated points as 0.
   - **Load-bearing assumption**: The Pareto classifier trained on finite anchor benchmarks accurately captures the true Pareto frontier of the entire benchmark space. This assumption is tested in a calibration step (see step 4).
3. Train a conditional generative model G (denoising diffusion probabilistic model; noise schedule: linear from 1e-4 to 0.02 over 1000 steps) conditioned on property vectors. G is trained to minimize:
   L = L_recon + λ_frontier * L_frontier + λ_diversity * L_diversity
   where:
   - L_recon: MSE reconstruction loss between generated benchmark and ground-truth anchor (when conditioning on its property vector).
   - L_frontier = max(0, 1 - C(P(G(p_target)))): hinge loss that penalizes generated benchmarks whose predicted properties are classified as dominated (off-frontier).
   - L_diversity = -var(predicted properties of generated benchmarks from a batch): encourages coverage.
4. **Calibration and verification**: After training, evaluate the accuracy of P and C on a held-out calibration set of 512 synthetic benchmarks with known ground-truth properties (generated from a procedural task template with controllable difficulty, diversity, and robustness parameters). Compute mean absolute error (MAE) between predicted and true properties for P, and classification accuracy for C. If MAE > 0.1 or accuracy < 80%, iteratively expand the anchor set with synthetic benchmarks that are mispredicted or misclassified, retrain P and C, and repeat until thresholds are met or 10 iterations are reached. This ensures that P and C are reliable for the conditional generation step.
Hyperparameters: λ_frontier = 0.5, λ_diversity = 0.1, diffusion steps = 1000, batch size = 64, learning rate = 1e-4 for G, 1e-3 for P and C. Training on a single NVIDIA A100 GPU takes approximately 24 hours.
```
(C) **Why this design**: We chose a diffusion model over a VAE because diffusion provides higher-fidelity generation for structured benchmark configurations (e.g., parameterized task descriptions), accepting higher computational cost. We used a separate Pareto classifier instead of a differentiable frontier loss (e.g., hypervolume) because classifier-based loss is simpler to train with binary supervision from anchor points, at the risk of classifier inaccuracies. We included a diversity term (L_diversity) to avoid mode collapse to a single frontier region, a trade-off that slightly reduces per-benchmark frontier adherence but ensures coverage across the trade-off space. (D) **Why it measures what we claim**: The predicted property vector P(G(p_target)) measures the generated benchmark's actual properties under the assumption that P is accurately trained on anchor benchmarks; this assumption is verified via calibration on synthetic benchmarks (see step 4). The frontier classifier C(P(G(p_target))) measures whether the benchmark lies on the Pareto frontier, under the assumption that anchor benchmarks adequately define the true frontier; this is validated by the calibration accuracy of C on synthetic frontier points. If calibration fails (MAE > 0.1 or accuracy < 80%), the generated benchmarks' properties are not trustworthy, and the anchor set is expanded. Together, they operationalize the motivation that generated benchmarks should achieve optimal trade-offs, but only to the extent that the calibration process ensures the quality of P and C.

## Contribution

(1) ParetoBench, a framework that generates agent evaluation benchmarks by conditioning on a desired property vector on the Pareto frontier, enabling control over multiple quality dimensions simultaneously. (2) A Pareto loss function that uses a binary classifier trained on anchor benchmarks to penalize off-frontier generated benchmarks, ensuring adherence to optimal trade-offs. (3) A design principle that benchmark generation should explicitly model property interdependencies via the Pareto frontier, moving beyond single-objective optimization.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Anchor benchmarks with property vectors | Captures multi-objective trade-offs. |
| Primary metric | Pareto frontier adherence | Measures generated benchmark's frontier status. |
| Calibration set | 512 synthetic benchmarks with known ground-truth properties | Validates property predictor and Pareto classifier. |
| Calibration metric | Mean absolute error (MAE) for P; accuracy for C | Quantifies prediction and classification reliability. |
| Baseline 1 | Random benchmark generation | No optimization, tests need for conditioning. |
| Baseline 2 | Enhanced-POET (CPPN) | Prior generative method for benchmarks. |
| Baseline 3 | Manual design | Human-driven selection, gold standard. |
| Ablation (ours) | ParetoBench w/o diversity loss | Isolates diversity term contribution. |

### Why this setup validates the claim

This combination tests the core claim: that conditional generation on the Pareto frontier reliably produces optimal benchmarks. Using anchor benchmarks with known property vectors provides ground-truth for training and evaluation. The primary metric, frontier adherence, directly quantifies whether generated benchmarks achieve the target trade-off. Random generation serves as a null baseline to show that conditioning is essential; Enhanced-POET tests generalization beyond evolutionary methods; manual design sets a human expert reference. The ablation isolates the diversity loss, verifying its role in broad coverage. The calibration set and metrics ensure that the property predictor and Pareto classifier are reliable, addressing the circularity concern: if calibration shows low error and high accuracy, then the generated benchmarks' properties are trustworthy.

### Expected outcome and causal chain

**vs. Random generation** — On a case where the target property vector requires high difficulty and moderate diversity, random generation produces benchmarks with random properties, rarely landing on the frontier. Because it lacks any learning or conditioning, it cannot consistently match the target. Our method instead leverages the trained generative model conditioned on p_target, learned from anchor patterns and validated via calibration, so it produces benchmarks with properties near the frontier. We expect a large gap in frontier adherence (e.g., >30 percentage points) and high variance for random.

**vs. Enhanced-POET** — On a case where the frontier region is sparse (few anchor points), Enhanced-POET relies on CPPN mutation, which may not preserve frontier properties if the encoding fails to capture multi-objective trade-offs. Because CPPNs are less expressive for structured parameters, they often generate dominated benchmarks. Our method uses a diffusion model with explicit frontier classifier loss, directly optimizing for frontier membership under the anchor-derived frontier, with calibration ensuring the classifier is accurate. Thus we expect ParetoBench to show superior frontier adherence, especially in sparse regions, with a moderate gap (10-20 percentage points).

**vs. Manual design** — On a case where the target requires balanced properties across three axes, manual design may inadvertently favor one axis due to human bias. Because humans cannot easily optimize high-dimensional trade-offs, manual benchmarks often deviate from the true frontier. Our method explicitly optimizes for the predicted property vector via the classifier loss, aligning closer to the target. However, manual design may excel on well-known benchmarks. We expect ParetoBench to match or exceed manual design on frontier adherence for novel target vectors, with a gap of ≤5 percentage points on familiar points.

### What would falsify this idea

If ParetoBench's generated benchmarks are consistently classified as dominated (off-frontier) by the same classifier used in training, then the whole reliance on anchor-defined frontier is flawed. Alternatively, if the ablation without diversity loss performs equally well, the diversity term is unnecessary and the method is just a conditional generator. Additionally, if calibration reveals that the property predictor or Pareto classifier has high error/low accuracy even after expansion, the approach fails to produce trustworthy benchmarks.

## References

1. Automating Benchmark Design
2. τ-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains
3. LLM-POET: Evolving Complex Environments using Large Language Models
4. APIGen: Automated Pipeline for Generating Verifiable and Diverse Function-Calling Datasets
5. Automatic Generation of Benchmarks and Reliable LLM Judgment for Code Tasks
6. MarioGPT: Open-Ended Text2Level Generation through Large Language Models
7. Evolution Gym: A Large-Scale Benchmark for Evolving Soft Robots
8. AgentBench: Evaluating LLMs as Agents
