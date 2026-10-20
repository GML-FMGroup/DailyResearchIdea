# Symmetry-Adaptive Abstraction Learning for Goal-Shifted World Models

## Motivation

World models trained on expert rollouts systematically fail on AnyGoal's goal-shifted tasks because they memorize factor combinations rather than learning composable abstractions; the interaction asymmetry principle, which assumes asymmetric factor dependencies, does not hold when factors interact symmetrically, so abstraction granularity is misaligned with the true symmetry structure. This structural gap—shared across interaction asymmetry, Lejepa, and PWM—means that without a symmetry-aware composability metric, models cannot determine when to merge or split factors, leading to brittle generalization on recombinations of individually familiar factors.

## Key Insight

The degree of interaction asymmetry between state factors directly determines the optimal abstraction granularity, and by measuring this symmetry from finite-horizon interaction data we can adaptively compose factor abstractions that remain invariant under goal shifts.

## Method

**What it is**  
SAL (Symmetry-Adaptive Abstraction Learning) is an algorithm that takes as input a set of state factors (e.g., object positions, velocities) and a replay buffer of expert trajectories, and outputs a hierarchical abstraction tree where each node groups factors that interact with high symmetry (i.e., their joint effect is approximately invariant under permutation of their roles). The tree is used to define a composable latent space for a world model, with abstraction granularity adapted per factor group.

**How it works**  
```pseudocode
# Phase 1: Estimate pairwise interaction symmetry
for each pair of state factors (i, j):
    # Collect transition tuples where factor i changes and j is held fixed, and vice versa
    # Compute the effect of i on j's future state: E[Δj | do(i+=δ), context] vs E[Δj | do(i-=δ), context]
    # Symmetry score s_ij = 1 - |E[Δj|do(i+=δ)] - E[Δj|do(i-=δ)]| / (|E[Δj|do(i+=δ)]| + |E[Δj|do(i-=δ)]| + ε)
    # Use a small MLP to estimate the conditional expectations from data (δ = 0.1 * std(i))
    s_ij = estimate_symmetry(i, j, buffer, δ=0.1*std(i))

# Phase 2: Build symmetry-aware composability graph
G = graph with nodes = factors, edge weight = s_ij
# Merge factors with high symmetry (s_ij > τ_sym = 0.8) into a single abstraction node
# Use hierarchical clustering with average linkage on 1 - s_ij
abstraction_tree = hierarchical_cluster(G, threshold=0.2, linkage='average')

# Phase 3: Train world model with adaptive abstraction granularity
for each node in abstraction_tree (bottom-up):
    if node is leaf: latent_dim = base_dim (e.g., 16)
    else: latent_dim = base_dim * (1 + log(1 + num_children))  # more capacity for symmetric groups
    train an encoder-decoder for the node's factor group with a dynamics model conditioned on parent abstraction
    # Use a reconstruction loss + a symmetry regularization: penalize variance of dynamics predictions under permutations of symmetric factors
    loss = recon_loss + λ * symmetry_regularizer (λ = 0.1)

# Phase 4: Goal-shifted planning
# At test time, given a new goal, use the abstraction tree to compose factor predictions
# Plan by rolling out the hierarchical dynamics model, treating symmetric groups as single entities
```

**Why this design**  
We chose to estimate pairwise interaction symmetry via conditional expectation differences (rather than via a learned independence head) because symmetry is a property of the data-generating process that can be probed by interventions; a learned head would conflate symmetry with statistical correlation, which is exactly the failure mode of prior disentanglement methods. We chose hierarchical clustering with a fixed threshold (τ_sym = 0.8) over a learned gating mechanism because the threshold directly controls the trade-off between abstraction coarseness and fidelity—lower thresholds merge more factors, reducing model complexity but risking underfitting to asymmetric interactions. We chose to allocate latent dimensions proportionally to the log of group size because symmetric groups require more capacity to represent joint permutations, while asymmetric factors can be modeled independently; this trades off parameter efficiency against representational power. We chose to use a symmetry regularizer during world model training (rather than post-hoc merging) because it ensures that the learned dynamics respect the estimated symmetry, preventing the model from exploiting spurious asymmetries in the finite data. Unlike the interaction asymmetry principle, which assumes a fixed asymmetric structure, SAL adapts the abstraction to the observed symmetry, avoiding the misalignment that causes AnyGoal failures.

**Why it measures what we claim**  
The symmetry score s_ij measures the degree of interaction asymmetry between factors i and j because it compares the counterfactual effect of increasing versus decreasing factor i on factor j's future state; this assumes that the conditional expectation difference is a sufficient statistic for symmetry, which fails when the dynamics are nonlinear and the perturbation δ is too small to capture global symmetry, in which case s_ij reflects local sensitivity rather than global symmetry. The hierarchical clustering threshold τ_sym measures the composability of factor groups because it groups factors whose pairwise symmetry exceeds a threshold; this assumes that pairwise symmetry is transitive, which fails when three factors have pairwise symmetry but triple-wise interactions are asymmetric, in which case the clustering may merge factors that are not jointly composable. The latent dimension allocation measures the representational demand of a factor group because it scales with the log of group size; this assumes that symmetric groups require more capacity to represent permutations, which fails when the group's dynamics are low-rank, in which case the allocation over-parameterizes. The symmetry regularizer measures the consistency of learned dynamics with estimated symmetry because it penalizes prediction variance under permutations; this assumes that the estimated symmetry is accurate, which fails when the symmetry estimate is noisy, in which case the regularizer may enforce spurious symmetry and degrade performance.

## Contribution

['A symmetry-aware composability metric that quantifies the degree of interaction asymmetry between state factors from finite-horizon counterfactual data.', 'A symmetry-adaptive abstraction learning algorithm (SAL) that builds a hierarchical abstraction tree by clustering factors based on the composability metric and allocates representational capacity proportionally to group symmetry.', 'An open-source implementation of SAL integrated with the AnyGoal benchmark, enabling evaluation of compositional generalization under goal shifts.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | AnyGoal goal-shifted variants | Tests generalization beyond expert rollouts directly |
| Dataset | Procedurally generated block-world with known ground-truth symmetry groups | Provides recoverable true abstraction tree for validation |
| Primary metric | Compositional goal-reaching success rate | Measures downstream planning under novel factor combinations |
| Baseline | Flat monolithic world model | No abstraction; tests value of any structure |
| Baseline | Interaction-asymmetry disentanglement model | Fixed asymmetric structure; tests adaptivity claim |
| Baseline | Learned-gating hierarchical world model | Learned merging; tests intervention-based symmetry estimate |
| Baseline | Correlation-based gating baseline (orthogonalized) | Tests SAL vs correlation on constructed orthogonal environment |
| Ablation-of-ours | SAL with fixed uniform latent dims | Tests capacity-allocation claim |
| Ablation-of-ours | SAL with allocation exponent swept over {0, 0.5, 1, 2} | Tests log scaling is not arbitrary |
| Ablation-of-ours | SAL with symmetry regularizer disabled | Tests regularizer contribution |

### Why this setup validates the claim

The central claim is that estimating symmetry by intervention-probed conditional expectations, then adapting abstraction granularity per symmetric group, yields better compositional generalization beyond expert rollouts than methods that assume a fixed structure or learn structure from correlation. The AnyGoal goal-shifted variants are the right testbed because they require composing factor predictions in configurations absent from expert data, which is exactly where abstraction misalignment bites. The procedurally generated block-world with known ground-truth symmetry groups provides a controlled setting where the true abstraction tree is recoverable, allowing direct measurement of clustering accuracy and validation of the symmetry score against ground-truth permutation invariance. The flat monolithic baseline isolates whether structure helps at all; the interaction-asymmetry baseline isolates whether *adaptive* symmetry beats *fixed* asymmetry; the learned-gating baseline isolates whether intervention-based symmetry estimation beats correlation-based gating. The correlation-based gating baseline on a constructed environment where pairwise intervention symmetry and statistical correlation are provably orthogonal isolates whether SAL's clustering recovers the true symmetry group while the gate provably cannot. The ablation with fixed uniform latent dims isolates the latent-dimension allocation sub-claim, and the exponent sweep tests whether the log scaling is arbitrary. Success rate on goal-shifted tasks is the right primary metric because it directly measures whether the composed latent dynamics reach novel goals, not merely reconstruction fidelity. Together these form a falsifiable test: gains must be concentrated on the subsets where symmetry misalignment is predicted to occur, and the ablation must lose ground specifically on symmetric groups.

### Expected outcome and causal chain

**vs. Flat monolithic world model** — On a case where the goal requires composing two symmetric object-position factors in a permutation not seen in expert rollouts, the flat model produces entangled latent dynamics that fail to transfer because it treats factors independently with no shared structure, so its rollout drifts. Our method instead groups the symmetric factors into one abstraction node with symmetry-regularized dynamics, so permutations are represented jointly, and we expect a clear success-rate gap on compositional subsets with near-parity on single-factor goals.

**vs. Interaction-asymmetry disentanglement model** — On a case where the environment's true symmetry differs from the fixed asymmetric structure the baseline assumes, the baseline produces misaligned abstractions and plans along the wrong factor decomposition because its structure is not adapted to observed symmetry. Our method instead estimates symmetry per pair via counterfactual probes and clusters adaptively, so the abstraction matches the data-generating process, and we expect the gap to concentrate on environments whose symmetry deviates from the baseline's prior, with parity where they coincide.

**vs. Learned-gating hierarchical world model** — On a case where two factors are statistically correlated but dynamically asymmetric, the learned gate merges them because it conflates correlation with symmetry, producing dynamics that cannot separate their effects and failing goal-shifted planning. Our method instead probes intervention effects directly, so correlated-but-asymmetric factors stay separate, and we expect a notable gap on the correlation-asymmetry-conflict subset while both methods tie on cleanly symmetric subsets.

**vs. Correlation-based gating baseline (orthogonalized)** — On a constructed environment where pairwise intervention symmetry and statistical correlation are provably orthogonal (e.g., two factors with zero correlation but high intervention symmetry, and two factors with high correlation but low intervention symmetry), the correlation-based gate merges the correlated-but-asymmetric pair and splits the symmetric-but-uncorrelated pair, producing an abstraction tree that provably differs from the true symmetry group. Our method's intervention-probed clustering recovers the true symmetry group, and we expect a large gap on goal-shifted tasks requiring the correct grouping, with both methods failing on tasks that require neither grouping.

**vs. SAL with fixed uniform latent dims** — On a case where a symmetric group has high permutation complexity (e.g., three or more factors with non-trivial joint permutations), the uniform-dimension ablation under-parameterizes the group's dynamics, leading to poor reconstruction and planning. Our method allocates latent dimensions proportionally to log group size, providing sufficient capacity, and we expect a gap on tasks involving large symmetric groups, with parity on small groups.

**vs. SAL with allocation exponent swept** — On the same tasks, sweeping the exponent over {0, 0.5, 1, 2} tests whether the log scaling (exponent 1) is optimal. We expect the success rate to peak at exponent 1, with degradation for exponent 0 (uniform) and exponent 2 (over-allocation to large groups causing underfitting elsewhere), confirming the log scaling is not arbitrary.

**vs. SAL with symmetry regularizer disabled** — On a case where the estimated symmetry is accurate but the model could exploit spurious asymmetries in finite data, disabling the regularizer leads to dynamics that violate the true symmetry, causing planning failures on goal-shifted tasks. Our method with the regularizer enforces consistency, and we expect a gap on tasks where symmetry is critical, with parity on tasks where symmetry is irrelevant.

### What would falsify this idea

If SAL's gain over the interaction-asymmetry and learned-gating baselines is uniform across all goal-shifted subsets rather than concentrated on the subsets where symmetry misalignment is predicted, the central claim that adaptive intervention-based symmetry drives the improvement would be wrong. Likewise, if the uniform-latent-dimension ablation matches full SAL everywhere, the capacity-allocation sub-claim is falsified. Additionally, if the correlation-based gating baseline matches SAL on the constructed orthogonal environment, the claim that intervention-based symmetry estimation is necessary would be falsified.

## References

1. AnyGoal: On World Model Generalization Beyond Expert Rollouts
