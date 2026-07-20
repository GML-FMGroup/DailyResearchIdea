# Multi-Environment Self-Supervised Skill Discovery via Cross-Environment Invariance

## Motivation

Existing skill acquisition methods, such as RESOURCE2SKILL, depend on human-created resources (tutorials, code, articles) which introduce bias and limit diversity. These resources are curated and may not cover rare or emerging skills. The root cause is reliance on a single, static resource repository, which cannot adapt to novel interactions. To overcome this, we need an autonomous framework that explores multiple diverse environments (e.g., web, robotics, games) and distills skills from heterogeneous experience without human guidance, ensuring diversity and bias-free induction.

## Key Insight

Skills are defined by behavioral patterns that transfer across environments; enforcing cross-environment invariance in skill representations via contrastive alignment extracts environment-agnostic skills from self-supervised exploration.

## Method

### (A) What it is: We propose **MESSI** (Multi-Environment Self-Supervised Skill Induction), a framework that takes multiple unlabeled interaction streams from diverse environments and outputs a set of discrete skill prototypes and an environment-invariant skill encoder. 

### (B) How it works:
```python
# Pseudocode: MESSI training
# Environments: Robotics (MetaWorld, 10 tasks), Games (Atari, 5 games), Web (MiniGrid, 4 tasks)
for each environment e in {MetaWorld, Atari, MiniGrid}:
    agent explores using random policy, collects 1000 trajectories τ_e = (s_1,a_1,...,s_T) per task  # no human guidance

# Build skill-aligned pairs across environments
for each trajectory τ_a in environment A:
    find best-matching trajectory τ_b in environment B (B≠A) using dynamic time warping (DTW) on action sequences (threshold=0.7)
    # Calibration: DTW threshold set via small labeled set (20 trajectories per environment, verified by 2 annotators)
    positive_pair = (τ_a, τ_b)
    sample negative τ_neg from any environment with DTW similarity <0.3
    # Verification: Manual inspection of 50 random positive pairs per environment pair; if >10% misaligned, adjust threshold

# Train VAE + triplet + adversarial objective
Encoder: 2-layer MLP, hidden=256, GeLU, output dim=32
Decoder: 2-layer MLP, hidden=256, GeLU, output same as input
DomainClassifier: 2-layer MLP, hidden=64, output 3 classes (environments) with gradient reversal layer (λ=0.1)
for each minibatch (batch size=64):
    z = Encoder(τ)  # VAE encoder maps trajectory to latent z (dim=32) via reparameterization
    reconstruction_loss = MSE(Decoder(z), τ)  # sum over states and actions
    triplet_loss = max(0, ||z_a - z_pos|| - ||z_a - z_neg|| + margin=0.5)
    domain_loss = CrossEntropy(DomainClassifier(z), environment_label)  # adversarial
    total_loss = reconstruction_loss + λ_trip*triplet_loss - λ_adv*domain_loss  # λ_trip=1.0, λ_adv=0.1
    update Encoder, Decoder, DomainClassifier via gradient reversal layer
    # Use Adam optimizer, lr=1e-4, train for 100 epochs

# Skill clustering: after training, cluster z vectors using Chinese Restaurant Process (concentration=1.0) to get K skills
# Final output: skill prototypes μ_k (K cluster means) and Encoder
```

### (C) Why this design: We chose a VAE with contrastive alignment over a simple clustering of trajectories because the VAE provides a probabilistic mapping and generative capacity, allowing us to sample new trajectories for each skill. We chose triplet loss over a classifier because we do not have skill labels; the positive pairs are mined by action similarity, which is a noisy heuristic but enforces environment invariance indirectly. We accept the cost that action similarity may not capture all skills (e.g., different action sequences achieving same goal) but mitigate by using a large margin to tolerate within-skill variations. We chose a non-parametric prior (Chinese Restaurant Process) instead of fixing K because in exploratory settings the number of skills is unknown; the cost is computational complexity of inference. We added adversarial domain adaptation to further remove environment-specific features, acknowledging training instability but leveraging its theoretical guarantee of invariance. 

### (D) Why it measures what we claim: 
- **Triplet loss with cross-environment positive pairs**: Operationalizes environment invariance. Assumption A: Action similarity (via DTW) measures skill identity. Failure mode F: Equifinality — different action sequences achieve the same goal, causing false negatives in positive pair mining. In that case, the triplet loss may push apart same-skill trajectories, leading to over-segmentation of skills. We calibrate the DTW threshold using a small labeled set (20 trajectories per environment) to mitigate, but assumption A remains load-bearing.
- **Adversarial domain loss**: Measures bias-free representation. Assumption B: Environment information is a confound, not a necessary component of skill. Failure mode F2: Skills that are inherently environment-specific (e.g., web navigation requiring mouse clicks) — then the adversary removes useful information, losing skill fidelity. We accept a trade-off between completeness and invariance.
- **VAE reconstruction loss**: Measures skill content conservation. It ensures z retains enough information to generate behavior, but reconstruction includes environment-specific low-level details, which we intentionally discard via adversarial loss, accepting a trade-off.

### Calibration/Verification of load-bearing assumption:
We verify Assumption A via a human annotation study. Three annotators label 100 random positive pairs (50 from each environment pair) as same-skill or not. Fleiss' kappa measures inter-annotator agreement. Pairs with majority agreement are used to compute precision/recall of our DTW-based mining. If precision <0.8 or recall <0.6, we adjust the DTW threshold or augment with cycle-consistency (future work).

## Contribution

(1) A self-supervised multi-environment exploration framework for skill acquisition that eliminates dependence on human-curated resources. (2) A cross-environment contrastive learning objective combined with adversarial domain adaptation to learn environment-invariant skill representations. (3) A variational skill discovery method that automatically determines the number of skills via a non-parametric prior, scaling to heterogeneous environments.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | MetaWorld (10 tasks), Atari (5 games), MiniGrid (4 tasks) | Diverse dynamics, existing simulators reduce cost |
| Primary metric | Task success rate on unseen environments | Measures skill transfer and generalization |
| Baseline 1 | No-skill baseline (random policy + planning) | Primitive actions without skill library |
| Baseline 2 | Single-environment VAE (trained on one env, tested on others) | No cross-environment alignment |
| Baseline 3 | Clustering-only (k-means on raw sequences, K=20) | No representation learning |
| Ablation | MESSI w/o adversarial loss | Removes domain invariance mechanism |

### Why this setup validates the claim
This combination tests the core claim that MESSI learns environment-invariant skill representations. The no-skill baseline quantifies the lower bound of performance without skills. Single-environment VAE tests whether skills discovered in one environment transfer to another; if they do not, the cross-environment alignment is crucial. Clustering-only baseline assesses whether raw action similarity suffices or if representation learning is needed. The ablation isolates the effect of adversarial domain adaptation. The primary metric, task success rate on unseen environments, directly measures the utility of discovered skills in novel settings. A significant gap between MESSI and baselines would support the claim that our method induces reusable, invariant skills. Additionally, we validate the load-bearing assumption (action similarity measures skill identity) via a human annotation study: 3 annotators label 100 random positive pairs (50 from each environment pair) as same-skill or not; Fleiss' kappa is computed; precision and recall of the DTW mining are reported. If precision <0.8 or recall <0.6, we flag the assumption as violated.

### Expected outcome and causal chain

**vs. No-skill** — On a task requiring multi-step coordination (e.g., stacking blocks in MetaWorld), the no-skill baseline must plan each action from scratch, leading to high sample complexity and low success. MESSI retrieves pre-learned skill prototypes and composes them, yielding high success. Expect a >50% success gap on complex tasks.

**vs. Single-environment VAE** — On a web task (e.g., MiniGrid door opening) after training on MetaWorld only, the single-environment VAE encodes action sequences with robotic motion features; when tested on web, those features are irrelevant and cause poor retrieval. MESSI aligns representations across environments via triplet loss, so web tasks become accessible. Expect success rate >30% vs. <10%.

**vs. Clustering-only** — On tasks with variable execution speed (e.g., stacking quickly vs. slowly), k-means on raw action sequences groups by length rather than skill, causing misassignment. MESSI's VAE with temporal warping in positive pair mining groups by skill intent. Expect clustering purity improvement >20%.

**vs. Ablation (w/o adversarial loss)** — On tasks where environment identity is a strong confound (e.g., mouse clicks vs. keyboard shortcuts), the ablation retains environment-specific features, causing poor transfer to a new environment with different action modalities. MESSI with adversarial loss forces invariance, so transfer succeeds. Expect a 10-15% success gap in cross-modal tasks.

### What would falsify this idea
If MESSI shows no performance improvement over single-environment VAE on cross-environment tasks, the core claim of environment-invariant skill learning is unsupported.

## References

1. RESOURCE2SKILL: Distilling Executable Agent Skills from Human-Created Multimodal Resources
