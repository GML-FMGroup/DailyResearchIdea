# Fixed-Point Post-Processing: A Projection-Operator Framework for Resource-Bounded Automated Medical Imaging Pipelines

## Motivation

Current automated medical imaging pipelines, such as the AMID framework in 'Towards Autonomous and Auditable Medical Imaging Model Development', either omit domain-specific post-processing or rely on handcrafted heuristics, offering no guarantee of performance optimality under fixed computational budgets. This structural gap arises because post-processing is treated as an ad-hoc final step rather than as an operation that can be formally integrated to ensure pipeline closure and monotonic improvement.

## Key Insight

By modeling post-processing as a monotone projection onto a convex feasibility set derived from anatomical priors and resource constraints, the entire pipeline becomes a contraction mapping, guaranteeing convergence to a unique fixed point that is both feasible and optimal within the budget.

## Method

### (A) What it is
Fixed-Point Post-Processing (FPP) is a post-processing module that takes a raw pipeline output (e.g., segmentation map) and projects it onto a convex set S encoding domain-specific anatomical constraints and resource budgets. The projection is assumed to be monotone with respect to a task-specific performance metric (validated empirically), ensuring each application improves performance until convergence to a unique fixed point. **Load-bearing assumption:** The method assumes that anatomical priors can be encoded as a convex set S and that the task metric f is monotone under projection onto S. We validate these assumptions empirically in the experiment.

### (B) How it works (Pseudocode)
```pseudocode
Input: Raw output x_0, feasibility set S (convex, closed; constructed from convex hull of aligned pancreas segmentations from 10 atlas scans, resampled to fixed spacing, then intersected with bounding box and resource constraints), metric f (Dice score), resource budget vector B = [max inference time 0.1s, max memory 10MB], tolerance epsilon (0.001), max_iter (50).
Initialize k = 0.
While k < max_iter:
  1. Compute projection: x_{k+1} = argmin_{z in S} ||z - x_k||^2  subject to g(z) <= B.
  2. If f(x_{k+1}) > f(x_k) + epsilon:
       k = k + 1
     Else:
       break
Output: x* = x_k
```
**Hyperparameters:** epsilon = 0.001, max_iter = 50, S constructed from a labeled atlas (10 scans) and resource budget B.

### (C) Why this design
We chose a projection-operator framework over heuristic post-processing rules because it provides formal guarantees of convergence and optimality, accepting the cost that constructing S requires domain knowledge and may be computationally expensive. We opted for a metric-based monotonicity check (step b) instead of a residual-based stopping criterion because f directly measures task performance, ensuring iterations are only accepted when they improve the objective; this trade-off introduces dependence on a reliable f, but avoids premature termination when residual is small but performance is poor. We selected L2 projection over other distances (e.g., L1, KL) due to its closed-form solution for convex polyhedral sets, enabling efficient computation; the cost is that L2 may not align with perceptual quality, but we mitigate this by constructing S to reflect perceptual priors. Finally, we enforce resource constraints as hard constraints within the projection rather than as soft penalties, because hard constraints guarantee budget satisfaction at each iteration, at the risk of making the projection non-expansive if constraints are too tight. Unlike classical POCS methods for image restoration, our projection operator enforces not only shape constraints but also resource budgets, and includes a monotonicity check to ensure performance improvement.

### (D) Why it measures what we claim
**Projection onto S** measures feasibility with respect to anatomical priors and resource budgets because it maps x_k to the closest point in the convex set S, which is defined as the intersection of constraints derived from domain knowledge; this assumption fails when the domain knowledge used to construct S is incomplete or inaccurate, in which case the projection may map to a point that is anatomically plausible but clinically incorrect, reducing to a mere smoothing operation. **Monotonicity condition f(x_{k+1}) > f(x_k)** measures performance improvement because we assume f is monotone under the projection operator, i.e., f(P_S(x)) ≥ f(x) for all x when S is feasible; this assumption holds when the projection onto S always moves x toward points with higher f. The failure mode occurs if S contains points with lower f than x, then projection may decrease f. To mitigate, we empirically verify on a calibration set (10 scans) that f is non-decreasing for the chosen S and task; if not, we adjust S (e.g., shrink convex hull) or switch to a learned shape constraint. **Fixed point condition P_S(x*)=x*** measures pipeline closure because it indicates that no further projection can improve feasibility or performance; this equivalence relies on S being convex and f being monotone, which together ensure that any fixed point is a global minimizer of the projection distance subject to constraints; this assumption fails when S is non-convex or f is non-monotone, in which case the fixed point may correspond to a local rather than global optimum, reducing the guarantee to a local convergence claim.

## Contribution

(1) The Fixed-Point Post-Processing (FPP) framework, a projection-operator method that integrates domain-specific post-processing into automated pipelines with provable monotonic improvement and resource-budget guarantees. (2) A methodology to construct convex feasibility sets from anatomical priors and resource constraints, enabling the application of projection-based post-processing in medical imaging.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
|------|--------|-----------|
| Dataset | NIH Pancreas-CT (82 CT scans, resampled to 1mm isotropic) | Public CT segmentation benchmark with challenging non-convex anatomy |
| Primary metric | Dice score | Standard overlap metric for segmentation |
| Baseline | Thresholding (0.5) + largest connected component | Simple heuristic baseline |
| Baseline | DenseCRF post-processing (potts weight=3, bilateral_strength=5, bilateral_spatial_sigma=5, bilateral_color_sigma=5) | Widely used graphical model for segmentation refinement |
| Ablation | FPP with random anatomical set: S constructed from random binary masks of same size | Tests influence of domain priors |

**Construction of S:** S is the convex hull of 10 randomly selected training scans' pancreas segmentations, after affine alignment to a common reference space (ITK-SNAP). The convex hull is then intersected with a bounding box (margin 10 pixels) and resource budget constraints (max inference time 0.1s, max memory 10MB). For the ablation, random masks are generated by thresholding Gaussian noise at 0.5 and taking the largest connected component.

### Why this setup validates the claim

The combination of a challenging segmentation dataset (NIH Pancreas-CT) with anatomical shape constraints and a primary metric (Dice) that directly measures overlap creates a direct test of FPP's ability to improve segmentation by enforcing domain-specific constraints. Baselines include a simple heuristic (thresholding) and a popular probabilistic method (DenseCRF), each representing different levels of prior incorporation. Comparing FPP to these baselines isolates the benefit of structured convex constraints with monotonicity guarantees. The ablation with random anatomical set tests whether observed gains come from the specific domain knowledge embedded in S or merely from any projection operation; if FPP with random S fails to improve, it confirms that domain-appropriate constraints are crucial. This design falsifies the claim if FPP fails to outperform baselines on cases where anatomical priors are informative, or if the ablation performs similarly to the full method.

### Expected outcome and causal chain

**vs. Thresholding + largest CC** — On a case where the raw segmentation produces multiple disconnected false positives or holes, thresholding + largest CC may select the largest false region or miss true structure, because it only uses size and connectivity. Our method instead projects the raw map onto the convex hull of plausible pancreas shapes, filling holes and removing outliers while respecting resource budgets, so we expect a noticeable gain in Dice (e.g., ~5-10 points) on irregular or fragmented predictions, but parity on already clean cases.

**vs. DenseCRF post-processing** — On a case where the raw segmentation has low confidence but correct shape, DenseCRF may mistakenly reinforce background pixels due to pairwise potentials, because it uses pixel-level affinities without global shape constraints. Our method instead enforces global shape feasibility via projection onto anatomical priors, preserving correct shape even when local cues are weak, so we expect a smaller but consistent gain (e.g., ~2-3 Dice points) on ambiguous boundaries, with similar performance on clear cases.

### What would falsify this idea

If FPP shows no improvement over thresholding or CRF on cases with fragmented or shape-distorted outputs (e.g., Dice gain < 1 point), or if the ablation with random S performs as well as the full method (within 1 Dice point), then the central claim that domain-specific convex projection monotonic improvement is the driver would be false. Additionally, if the monotonicity condition f(x_{k+1}) > f(x_k) fails (i.e., Dice decreases) in more than 5% of iterations on the calibration set, the method's assumption is violated and would need redesign.

## References

1. Towards Autonomous and Auditable Medical Imaging Model Development
2. AutoMLGen: Navigating Fine-Grained Optimization for Coding Agents
3. An AI system to help scientists write expert-level empirical software
4. SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering
5. DS-Agent: Automated Data Science by Empowering Large Language Models with Case-Based Reasoning
6. ExpeL: LLM Agents Are Experiential Learners
7. Automatic Chain of Thought Prompting in Large Language Models
