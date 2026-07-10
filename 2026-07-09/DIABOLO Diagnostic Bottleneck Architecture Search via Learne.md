# DIABOLO: Diagnostic Bottleneck Architecture Search via Learned Inverse Edits

## Motivation

Existing embodied agent architecture search methods like AgentCanvas rely on iterative reflection and full simulator rollouts per candidate, requiring many evaluations to converge. The reflection module produces high-dimensional diagnostics but does not directly guide edit generation, creating a structural bottleneck between diagnosis and improvement. This observation suggests that if we can learn an offline inverse mapping from compact deficiency representations to effective architecture edits, we can amortize the search cost and reduce rollouts significantly.

## Key Insight

Diagnostic deficiencies compress the minimal information needed to identify effective architecture edits, making the inverse mapping from deficiencies to edits learnable offline and enabling zero-rollout proposal during search.

## Method

**(A) What it is:** DIABOLO (Diagnostic Bottleneck Architecture Search) is a method that learns an inverse mapping from a deficiency vector (output of a reflection module trained as an overcomplete autoencoder with latent dimension 512, expandable to 1024) to a sequence of architecture edit operations using a conditional variational autoencoder (CVAE). Its inputs are the deficiency vector from a single rollout; its outputs are proposed architecture edits that can be directly applied without further rollouts during search. The deficiency vector is assumed to be a sufficient statistic for the edit, i.e., it compresses all information needed to identify effective edits. This assumption is load-bearing; if it fails (e.g., due to nonlinear interactions), we include a fallback rollout to refine the diagnosis when the proposed edit fails to improve performance.

**(B) How it works:**
```
# Phase 1: Offline data collection
for each architecture A in training set:
    d = reflect(A)  # deficiency vector from overcomplete autoencoder (latent dim=512)
    E = full_search(A)  # AgentCanvas KDLoop to find best edit
    store (d, E)

# Phase 2: Train CVAE
# Encoder: q(z|d) = N(μ(d), σ²(d)) with 2-layer MLP (hidden=256, GeLU)
# Decoder: p(E|z) autoregressive over edit tokens (max length 20) with Gumbel-softmax (τ=1.0)
# Loss: ELBO = reconstruction_loss(E, p(E|z)) + β * KL(q(z|d) || N(0,I)), β=0.1
# Training: Adam lr=1e-4, batch size=64, 100K steps on 1 NVIDIA A100 GPU (~10 hours)

# Phase 3: Search with DIABOLO
for new architecture A':
    d' = reflect(A')
    sample z ~ q(z|d')
    generate edit E' = p(E|z)  # greedy decoding
    # optionally sample K=5 edits, pick top via predicted reward
    rollout A'+E' to evaluate (only 1 rollout)
    if improvement: store (d', E') in buffer for periodic fine-tuning
    else:  # Fallback: refine diagnosis
        d'' = reflect(A'+E')  # rollout again to get corrected deficiency
        sample z'' ~ q(z|d'')
        generate new edit E'' = p(E|z'')
        rollout A'+E'' to evaluate (2nd rollout)
        if improvement: add (d'', E'') to buffer
```

**(C) Why this design:** We chose a conditional VAE over a deterministic mapper because the edit space is combinatorially large and one deficiency may map to multiple valid edits; stochastic sampling provides diversity, accepting the cost of occasional invalid edits. We trained the CVAE on pairs derived from AgentCanvas search to ground the mapping in real improvements, trading off dataset quality for initial coverage. We limited rollouts to one per proposal to maximize savings, but to mitigate inaccuracy we periodically re-run K=5 samples every 10 iterations as a fallback. Latent interpolation was chosen for dynamic expansion without retraining, accepting that some interpolated edits may be semantically invalid; these are filtered by a validity check (e.g., graph constraints) before rollout. We added a fallback rollout when the initial edit fails, which refines the diagnosis and re-queries the CVAE, addressing the load-bearing assumption of deficiency sufficiency.

**(D) Why it measures what we claim:** The deficiency vector d measures diagnostic sufficiency because it aggregates architectural failure modes into a compressed representation via a trained overcomplete autoencoder; this assumes failure modes are linearly separable, failing when critical deficiencies are not linearly separable (e.g., rare nonlinear interactions), in which case d may omit information needed for effective edits. The CVAE decoder p(E|z) measures edit relevance because it is trained to minimize reconstruction loss of edits that demonstrably improved performance on the training set; this assumes the training distribution covers the test deficiency space, failing when out-of-distribution deficiencies produce nonsensical edits. The latent interpolation operation measures novelty because it generates edits along linear paths that preserve semantic similarity if the latent space is smooth; this fails when the edit space is not convex (e.g., discrete module existence), leading to intermediate edits that violate architectural constraints.

## Contribution

(1) DIABOLO, a method that learns an inverse mapping from reflection deficiency vectors to architecture edits offline, enabling zero-rollout proposal during search. (2) A training procedure using a conditional VAE with Gumbel-softmax for discrete edit generation and latent interpolation for dynamic space expansion. (3) An empirical demonstration that DIABOLO reduces search cost by an order of magnitude while maintaining or improving success rates on embodied agent tasks.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|-----------------------|
| Dataset | Habitat ObjectNav | Standard embodied task with diverse failures |
| Primary metric | Success Rate | Direct measure of task completion |
| Secondary metric | Mutual Information (MI) between deficiency vector and edit outcome | Empirically verify sufficiency of deficiency vector |
| Baseline 1 | Hand-designed architecture | Represents expert-crafted baselines |
| Baseline 2 | EvoAgentX (evolutionary search) | State-of-the-art automated architecture search |
| Ablation of ours | DIABOLO w/o latent interpolation | Tests importance of stochastic diversity |
| Additional ablation | DIABOLO w/o fallback refinement | Tests importance of fallback for diagnosis |
| Additional task | RoboTurk manipulation | Tests generality to different failure modes |

### Why this setup validates the claim
This experimental design directly tests the central claim that DIABOLO can efficiently generate effective architecture edits from a diagnostic deficiency vector. Habitat ObjectNav provides a rich set of failure modes (e.g., perceptual errors, planning loops) that the deficiency vector must capture. The mutual information metric quantifies how much information about the edit outcome is contained in d, grounding the sufficiency assumption. The additional manipulation task (RoboTurk) tests whether the approach generalizes to different embodied domains with distinct failure types. Comparing to a hand-designed baseline reveals whether search is beneficial at all; the EvoAgentX baseline tests whether DIABOLO's one-rollout search competes with multi-iteration evolution. The ablation without latent interpolation isolates the role of stochastic diversity, while the ablation without fallback refinement tests whether the fallback is necessary for cases where d is insufficient. Success Rate is the appropriate primary metric because it directly reflects task performance, which is the ultimate goal of architecture search.

### Expected outcome and causal chain

**vs. Hand-designed architecture** — On a case where the agent fails due to a rare perceptual occlusion (e.g., a door left ajar), a hand-designed architecture lacks a specialized plan-repair module and simply replans from scratch, often repeating the same error. Our method's deficiency vector captures the occlusion as a "perceptual bottleneck" pattern, and the CVAE generates a targeted edit (e.g., adding a memory module) that avoids the failure. Thus we expect DIABOLO to achieve a noticeable success rate improvement (e.g., +15-20%) on scenes with uncommon obstacles, while parity on simple scenes.

**vs. EvoAgentX** — On a case where the agent's current architecture has a suboptimal submodule (e.g., a high-level planner that ignores depth cues), EvoAgentX would require many rollouts to mutate that submodule, and mutations are random. Our method directly maps the deficiency vector (which highlights the depth-ignoring issue) to an edit that replaces the planner with a depth-aware variant, all in one rollout. Therefore we expect DIABOLO to achieve similar final success rates in far fewer total rollouts (e.g., 5× fewer), but with higher variance due to occasionally invalid edits.

**vs. DIABOLO w/o latent interpolation** — On a case where multiple valid edits exist (e.g., add memory OR add attention), the deterministic mapper may collapse to one edit and miss the other, reducing diversity. Our method with latent interpolation allows sampling different edits from the same deficiency. We expect DIABOLO to find a higher-quality edit more often (e.g., 10% higher success rate) when the search space is multi-modal, and identical performance when only one edit dominates.

**vs. DIABOLO w/o fallback refinement** — On a case where the initial deficiency vector is insufficient (e.g., due to nonlinear interactions), the fallback rollout refines the diagnosis and produces a better edit. We expect the full DIABOLO to achieve higher success rate on complex scenes (e.g., +8%) compared to the variant without fallback, especially in early search iterations.

### What would falsify this idea
If DIABOLO's success rate gains are uniform across all scene types (i.e., no concentration on scenes where its deficiency-driven diagnosis should matter most), then the deficiency vector is not capturing meaningful diagnostic information and the central claim is false. Furthermore, if mutual information between d and edit outcome is low (<0.1 bits), the sufficiency assumption is invalid.

## References

1. Automating the Design of Embodied Agent Architectures
2. Multi-agent Architecture Search via Agentic Supernet
3. EvoAgentX: An Automated Framework for Evolving Agentic Workflows
4. Symbolic Learning Enables Self-Evolving Agents
5. MapCoder: Multi-Agent Code Generation for Competitive Problem Solving
6. Adaptive In-conversation Team Building for Language Model Agents
7. Language Agent Tree Search Unifies Reasoning Acting and Planning in Language Models
8. LLM Performance Predictors are good initializers for Architecture Search
