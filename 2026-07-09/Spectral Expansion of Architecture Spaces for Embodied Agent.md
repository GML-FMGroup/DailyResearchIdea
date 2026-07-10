# Spectral Expansion of Architecture Spaces for Embodied Agents

## Motivation

Existing embodied agent architecture search methods such as AgentCanvas (from Automating the Design of Embodied Agent Architectures) assume a fixed set of module types and connection patterns, limiting adaptation to novel tasks. The root cause is that the architecture space is predefined and enumerable, preventing the discovery of functional modules not anticipated during design. This paper addresses the convergent_gap that both AgentSquare and MaAS suffer from: they cannot generate new module types that fill functional gaps in the performance landscape.

## Key Insight

The eigenvectors of the performance Hessian over architecture parameters reveal under-explored functional dimensions, enabling principled generation of new module types that are orthogonal complements to the current set.

## Method

**A) What it is:** SAELA (Spectral Architecture Expansion via Landscape Analysis) takes a set of existing module types and their connection patterns, and dynamically expands the architecture space by generating new module types that target functional gaps identified via spectral analysis of the performance landscape. Its output is an augmented architecture space with new module types.

**B) How it works:**
```pseudocode
Input: Initial architecture space S (modules M, connections C, search space)
       Performance predictor P (e.g., LLM-Distill-PP from LLM-PP paper)
       Simulator E (for embodied tasks)
Procedure:
1. Sample a set of architectures A from S using random search (e.g., 100 architectures).
2. For each a in A, estimate performance p_a using P (or simulator E if cheap).
3. Compute a local approximation of the performance Hessian H at the current best architecture a*:
   - For each trainable module parameter (e.g., weight, prompt), perturb by δ and measure performance change.
   - H_{ij} = (p(a*+δe_i+δe_j) - p(a*+δe_i) - p(a*+δe_j) + p(a*)) / δ^2 (with δ=0.01)
4. Perform eigendecomposition of H: H = Q Λ Q^T.
5. Identify eigenvectors v_k corresponding to eigenvalues λ_k with small magnitude (|λ_k| < τ, where τ=0.1). These directions indicate low curvature → functional gaps.
6. For each such v_k, generate a new module type m_new:
   - Use a conditional diffusion model (e.g., DiffModule) that takes v_k as conditioning and outputs a module specification (e.g., a neural network architecture or LLM prompt).
   - Train DiffModule on existing module specifications to learn the manifold of feasible modules.
7. Validate m_new via simulator: run a few episodes to ensure feasibility and basic performance.
8. Add m_new to S, and continue architecture search (e.g., KDLoop from AgentCanvas) with the expanded space.
9. Repeat steps 1-8 every T search steps (e.g., T=20) to progressively fill gaps.
Hyperparameters: δ=0.01, τ=0.1, T=20, number of new modules per cycle = 2.
```

**C) Why this design:** We chose spectral analysis over random expansion of the space because random generation may add redundant or irrelevant modules that do not fill functional gaps, wasting search budget. The Hessian approximation, though computationally expensive (requiring per-dimension perturbations), is preferred over simpler gradient-based methods (e.g., Fisher information) because it captures second-order curvature, which directly measures how performance changes when module parameters are jointly varied, revealing directions where small changes yield little improvement (functional gaps). We use a conditional diffusion model to generate new modules conditioned on eigenvectors, rather than a simple linear interpolation, because diffusion models can produce diverse, complex modules that respect the structural constraints of the architecture (e.g., input/output types). The trade-off is increased generation cost; we mitigate this by only generating modules for the top-k eigenvectors with smallest eigenvalues. The validation step with the simulator accepts the risk that generated modules may be infeasible; we incur a small simulation cost to filter them. Overall, this design prioritizes principled expansion over efficiency, aiming to discover truly novel modules that existing automated design methods would miss.

**D) Why it measures what we claim:** The eigenvalue magnitude λ_k measures the curvature of the performance landscape along eigenvector v_k; small |λ_k| indicates that performance is insensitive to changes in that direction, implying that the current architecture space lacks mechanisms to exploit that dimension, i.e., a functional gap. This equivalence holds under the assumption that the Hessian is a locally faithful approximation of the true performance function; this assumption fails when the performance landscape is highly non-smooth (e.g., due to stochastic simulator noise), in which case small eigenvalues may reflect noise rather than genuine gaps. In such cases, we regularize the Hessian with Tikhonov damping to suppress noise-induced eigenvalues. The conditional diffusion model's output m_new is then a module targeted to fill the gap; we measure its effectiveness by the increase in performance after its addition, which directly quantifies gap closure.

## Contribution

(1) A method for dynamic architecture space expansion via Hessian spectral analysis, enabling discovery of new module types that fill functional gaps. (2) An empirical demonstration that generated modules lead to non-trivial performance gains on embodied tasks, validated on VLN and manipulation benchmarks. (3) A conditional diffusion model that generates feasible module specifications conditioned on eigenvectors of the performance Hessian.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|----------------------|
| Dataset | Habitat ObjectNav | Standard embodied navigation benchmark. |
| Primary metric | Success Rate | Directly measures task completion. |
| Baseline 1 | Handcrafted | Fixed architecture, no adaptation. |
| Baseline 2 | Random Search Expansion | Expands space without spectral guidance. |
| Baseline 3 | Agentic Supernet | Prior automated search method. |
| Ablation | SAELA w/o spectral analysis | Random module generation instead. |

### Why this setup validates the claim

This combination tests the core claim that SAELA's spectral analysis identifies and fills functional gaps in the architecture space. Habitat ObjectNav is a challenging embodied task with diverse failure modes requiring specialized modules. Success Rate is a direct measure of task completion, sensitive to the presence or absence of critical modules. The handcrafted baseline shows the necessity of search; Random Search Expansion tests whether simple random expansion suffices, isolating the benefit of spectral targeting; Agentic Supernet is a strong prior method that operates within a fixed space, serving as a comparison for generating novel modules. The ablation removes the spectral component, using random module generation instead, to confirm that any performance gain from SAELA is due to targeted gap filling rather than mere expansion. Together, these comparisons create a falsifiable test: if SAELA's gains are concentrated in tasks requiring missing capabilities, the claim is supported; if gains are uniform or absent, the spectral analysis is ineffective.

### Expected outcome and causal chain

**vs. Handcrafted** — On a case where the scene has unseen object arrangements requiring new interaction skills (e.g., opening a drawer before grasping), the handcrafted architect has no mechanism to add a module for that skill, leading to failure. Our method instead detects a functional gap via a small Hessian eigenvalue along a direction representing interaction skills, and generates a module to fill that gap, enabling success. We expect a noticeable gap in success rate on scenes requiring skill discovery (e.g., +15%) vs. parity on standard scenes.

**vs. Random Search Expansion** — On a case where the current space lacks a crucial module (e.g., subgoal decomposition), random expansion may create modules that are irrelevant or redundant, not targeting the missing subgoal capability. SAELA's spectral analysis identifies the exact direction of missing functionality, generating a subgoal module. Thus, we expect SAELA to outperform random expansion specifically on tasks where subgoal decomposition is needed (e.g., multi-step tasks) by ~10%, while being similar on simpler tasks.

**vs. Agentic Supernet** — Agentic Supernet searches within a fixed space of predefined modules, so it cannot create new module types outside that space. When the optimal architecture requires a novel module (e.g., a learned world model predictor), it fails. SAELA expands the space with new modules, capturing that capability. We expect SAELA to show superior performance on tasks demanding novel module types (e.g., +20% on partially observable domains), with parity on tasks already covered by the initial space.

**vs. SAELA w/o spectral analysis** — Without spectral guidance, the method generates random modules, which may not fill true gaps, adding noise at best. The ablation therefore cannot systematically improve performance. SAELA's gain comes from targeted gap filling. We expect SAELA to consistently outperform its ablation by at least 5% across all tasks, with the gap widening on tasks with clear missing capabilities.

### What would falsify this idea

If SAELA's performance gain over Random Search Expansion is uniform across all tasks rather than concentrated on tasks where functional gaps exist (e.g., scenes requiring new skills), then the central claim that spectral analysis identifies genuine gaps is falsified.

## References

1. Automating the Design of Embodied Agent Architectures
2. Multi-agent Architecture Search via Agentic Supernet
3. EvoAgentX: An Automated Framework for Evolving Agentic Workflows
4. Symbolic Learning Enables Self-Evolving Agents
5. MapCoder: Multi-Agent Code Generation for Competitive Problem Solving
6. Adaptive In-conversation Team Building for Language Model Agents
7. Language Agent Tree Search Unifies Reasoning Acting and Planning in Language Models
8. LLM Performance Predictors are good initializers for Architecture Search
