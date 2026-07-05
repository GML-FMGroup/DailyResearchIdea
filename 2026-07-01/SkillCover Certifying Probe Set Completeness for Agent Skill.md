# SkillCover: Certifying Probe Set Completeness for Agent Skill Discovery via Rank Conditions on Behavioral Embeddings

## Motivation

Existing skill discovery methods like SkillHone rely on practice probes without guaranteeing coverage of the full skill manifold, leading to potential blind spots (as acknowledged in SkillHone's limitations). The underlying structural problem is that these systems assume the probe space sufficiently covers all relevant task behaviors but provide no method to measure or ensure comprehensive coverage.

## Key Insight

The rank of a behavioral embedding matrix constructed from probe trajectories equals the dimension of the skill manifold spanned by those probes; thus, full-rank relative to the intrinsic dimensionality of the skill space is both necessary and sufficient for probe set completeness under a linearity assumption.

## Method

### (A) What it is
SkillCover is a certification module that takes a set of probe trajectories (each described by a behavioral feature vector) and outputs a binary completeness guarantee along with a set of missing directions. Input: a list of behavioral embeddings (each vector in ℝ^d). Output: a Boolean indicating whether the probe set spans the skill manifold, and if not, a set of orthonormal vectors that span the orthogonal complement.

### (B) How it works
```python
import numpy as np

def certify_probe_set(probes, threshold_ratio=0.01):
    """
    probes: list of n behavioral embeddings, each shape (d,)
    returns: (is_complete: bool, missing_directions: np.ndarray or None)
    
    Assumption: The skill manifold is a linear subspace of the embedding space.
    If this assumption fails, the guarantee only applies to linear coverage.
    """
    X = np.stack(probes)  # n x d
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    # Estimate intrinsic dimension: count singular values above threshold
    threshold = threshold_ratio * S[0]
    intrinsic_dim = int(np.sum(S > threshold))
    # Numerical rank
    rank = np.linalg.matrix_rank(X, tol=threshold)
    if rank >= intrinsic_dim:
        return True, None
    else:
        # Missing directions: rows of Vt beyond rank
        missing_directions = Vt[rank:]
        return False, missing_directions
```
Hyperparameters: threshold_ratio (default 0.01) controls sensitivity to noise.

### (C) Why this design
We chose singular value decomposition over eigenvalue decomposition because it handles non-square matrices and provides both left and right singular vectors, which directly give the span of probes and the orthogonal complement. We chose a fixed threshold ratio of 0.01 × σ_max to define intrinsic dimension over a more complex Bayesian estimator because it is simple and interpretable, accepting the risk of misestimating dimension when the gap between signal and noise is ambiguous. We chose to return the missing directions as nullspace vectors rather than a scalar coverage score because this directly informs subsequent exploration: the agent can generate probes that are orthogonal to existing ones. Using matrix rank instead of a distance-based coverage metric (e.g., maximum similarity) avoids the need to set a distance threshold and provides a linear-algebraic guarantee of spanning, at the cost of assuming the skill manifold is approximately linear in the embedding space. This design explicitly assumes that the skill manifold is a linear subspace of the behavioral embedding space. If this assumption fails, the rank condition only certifies linear coverage, not full skill coverage.

### (D) Why it measures what we claim
The computational quantity rank(X) measures probe set completeness because it equals the dimension of the linear span of probes; under Assumption A (the skill manifold is a linear subspace of the behavioral embedding space), full rank implies the probes span the manifold. This assumption fails when the skill manifold is nonlinear or when the behavioral embedding is insufficiently expressive, in which case rank measures only linear coverage and may underestimate incompleteness (false positive guarantee). The singular value threshold for intrinsic dimension estimates the effective dimensionality of the skill manifold under the assumption that signal singular values dominate noise; this assumption fails when the noise distribution is heavy-tailed, in which case the estimated dimension may be inflated or deflated. We also include a linearity diagnostic: we compute the reconstruction error of projecting held-out probe trajectories onto the linear subspace spanned by the training probes. A low reconstruction error indicates that the linearity assumption is likely valid for the given embedding space.

## Contribution

(1) SkillCover, a certification mechanism that uses rank conditions on behavioral embeddings to diagnose probe set completeness and identify missing directions. (2) A method to translate missing directions into targeted exploration objectives by generating probes orthogonal to the current span. (3) An analysis of the linearity assumption's scope, with a procedure to extend the certification to local linear patches via clustering and per-cluster SVD.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | WebWalkerQA-EN | Diverse web navigation tasks with behavioral embeddings. |
| Primary metric | Task success rate | Direct measure of agent performance. |
| Baseline 1 | Random exploration | No coverage guidance; isolates benefit. |
| Baseline 2 | Direct LLM task proposal | Prior method without formal coverage. |
| Baseline 3 | Distance-based coverage | Alternative with similarity threshold. |
| Ablation | SkillCover w/o missing-direction feedback | Tests importance of guided exploration. |
| Linearity diagnostic | Reconstruction error of linear subspace projection on held-out probes | Quantifies validity of linearity assumption. |

### Why this setup validates the claim

This combination tests whether SkillCover’s linear-algebraic completeness guarantee enables more efficient skill acquisition than heuristic or no guidance. WebWalkerQA-EN provides a rich set of behavioral trajectories that can be projected to embeddings, allowing direct measurement of coverage. Random exploration tests the necessity of any coverage guidance. Direct LLM task proposal tests whether structured completeness detection outperforms ad-hoc diversity. The distance-based baseline tests whether linear spanning is superior to similarity thresholds. The ablation isolates the value of missing-direction feedback. Task success rate captures the ultimate benefit of covering the skill manifold. The linearity diagnostic (reconstruction error) quantifies how well the linearity assumption holds, enabling interpretation of results. The falsifiable prediction is that SkillCover’s advantage will be largest on tasks requiring rare, orthogonal skills and when the reconstruction error is low (linear manifold).

### Expected outcome and causal chain

**vs. Random exploration** — On a task requiring precise multi-step planning, random exploration scatters trajectories across diverse but non-covering behaviors because it has no signal to fill gaps. Our method explicitly identifies missing orthonormal directions and directs exploration there, so we expect a large success rate gap (e.g., 70% vs 20%) on tasks with high coverage deficiency.

**vs. Direct LLM task proposal** — On a case where the agent has covered common skills but misses an obscure subroutine, the LLM proposes tasks similar to existing ones due to lack of coverage awareness, failing to fill the gap. Our SkillCover detects the missing direction and generates a probe to span it, so we expect a moderate advantage (e.g., 65% vs 45%) on tasks requiring that subroutine.

**vs. Distance-based coverage** — On a task set where skills are nearly orthogonal, a similarity threshold may overestimate coverage by missing directions only slightly beyond the cutoff, while our SVD-based rank correctly detects them. Thus we expect SkillCover to have a small but consistent advantage (e.g., 62% vs 58%) on medium-difficulty tasks.

**Linearity diagnostic** — We expect to observe low reconstruction error (e.g., <0.1 after normalization) on tasks where SkillCover provides guarantees that correlate with actual performance; high reconstruction error indicates the linearity assumption is violated, and the guarantee may be unreliable. This diagnostic will be reported alongside success rates to contextualize results.

### What would falsify this idea

If SkillCover’s success rate gain is uniform across all task difficulty levels rather than concentrated on tasks requiring rarely seen orthogonal skills, then the coverage-guided exploration claim is unsupported. Additionally, if the linearity reconstruction error is high (e.g., >0.3) but SkillCover still shows large gains, then the linearity assumption is not driving the benefit, and the explanation is incorrect.

## References

1. SkillHone: A Harness for Continual Agent Skill Evolution Through Persistent Decision History
2. Multi-Agent Collaboration: Harnessing the Power of Intelligent LLM Agents
3. Automated Skill Discovery for Language Agents through Exploration and Iterative Feedback
4. Live-SWE-agent: Can Software Engineering Agents Self-Evolve on the Fly?
5. Chain of Thought Prompting Elicits Reasoning in Large Language Models
6. OpenWebVoyager: Building Multimodal Web Agents via Iterative Real-World Exploration, Feedback and Optimization
7. DataEnvGym: Data Generation Agents in Teacher Environments with Student Feedback
8. Proposer-Agent-Evaluator(PAE): Autonomous Skill Discovery For Foundation Model Internet Agents
