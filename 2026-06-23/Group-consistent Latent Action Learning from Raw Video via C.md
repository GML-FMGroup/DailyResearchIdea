# Group-consistent Latent Action Learning from Raw Video via Compositional Invariance

## Motivation

Existing latent action learning methods such as PoLAR and LAPO require paired observations with known temporal offsets to infer actions from transitions. This assumption limits applicability to videos where precise temporal alignment is unavailable or where actions are not linearly correlated with time. We aim to remove this requirement by exploiting the compositional structure of latent actions across arbitrary frame triples, enabling fully unsupervised action emergence from monolithic video without any temporal labels.

## Key Insight

The composition property of latent actions (the action from A to C equals the composition of actions from A to B and B to C) is a self-supervised invariant that holds for any three temporally ordered frames, enabling learning without temporal offset labels.

## Method

**Group-consistent Latent Action Learning (GCLAL)**

(A) **What it is:** GCLAL is a self-supervised method that learns latent action vectors from raw video by enforcing a group composition property across randomly sampled frame triples, using only the inherent temporal order of frames.

(B) **How it works (pseudocode):**
```python
# Input: Video V with frames x_1,...,x_T (temporally ordered)
# Hyperparameters: embedding_dim d = 64, variance_strength λ_var = 0.1

# Encoder E: a Siamese ResNet-18 (shared weights) that outputs a d-dimensional vector
# Normalize output to unit sphere (||a|| = 1)

for each batch of videos:
    for each video:
        # Sample a random triple (i,j,k) with i<j<k
        a_ij = E(x_i, x_j)
        a_jk = E(x_j, x_k)
        a_ik = E(x_i, x_k)
        # Composition loss (assumes additive composition: a_ij + a_jk ≈ a_ik)
        L_comp = ||a_ij + a_jk - a_ik||^2
        # Inverse loss (sample reversed pair)
        a_ji = E(x_j, x_i)
        L_inv = ||a_ji + a_ij||^2
        # Variance regularization over batch
        all_actions = [a from all pairs in batch]
        std = std(all_actions)
        L_var = -std  # maximize standard deviation
    L_total = L_comp + L_inv + λ_var * L_var
    update E with Adam (learning_rate=1e-4)
```

(C) **Why this design:** We chose a simple additive composition model over a learned associative operation because it imposes a strong inductive bias that the latent action space is Euclidean and actions compose by vector addition, which is invertible and commutative. The trade-off is that this choice assumes transitions are roughly linear in the latent space, which may not hold for complex dynamics; however, it enables direct arithmetic correspondence with the group property. We chose triplet sampling over pairwise only because it directly enforces transitivity across arbitrary frames. We chose variance regularization over a reconstruction decoder to avoid that the decoder's inductive biases contaminate latent action learning; the cost is that we do not directly model observation generation, so the latent actions may not be fully grounded in the visual appearance. The unit norm normalization prevents collapse to zero magnitude and stabilizes training. **Load-bearing assumption:** The design assumes that latent actions compose via vector addition (a_ij + a_jk = a_ik) for any three temporally ordered frames; this holds for deterministic linear dynamics but may fail for nonlinear dynamics such as robotic pushing with friction.

(D) **Why it measures what we claim:** The composition loss L_comp measures group consistency because it operationalizes the invariance that the action from frame i to k should equal the composition of actions from i to j and j to k, assuming that the latent action space is a vector space where addition corresponds to composition. This assumption holds when the dynamics are deterministic and the encoder extracts the true underlying action; it fails when the dynamics are stochastic or the encoder produces spurious correlations, in which case L_comp instead reflects the degree of inconsistency in the learned representation. The inverse loss L_inv measures reversibility of actions, operationalizing the requirement that going from j to i is the opposite action of i to j. Variance regularization measures non-collapse, assuming that the variance of a batch is indicative of a non-degenerate representation—but if the batch is too small, variance may be artifactually high. **Failure mode:** Under nonlinear dynamics (e.g., pushing with varying friction), additive composition may break down, causing L_comp to be high even if the encoder captures valid latent actions; we therefore verify the assumption by monitoring L_comp on held-out triples during training.

## Contribution

(1) A novel self-supervised objective for learning latent actions from monolithic video without temporal offset labels, based on group compositional invariance across arbitrary frame triples. (2) Empirical demonstration that the learned latent actions reconstruct transitions and improve downstream policy learning efficiency compared to training from scratch or with temporal-offset-based methods. (3) A new benchmark setting for unsupervised latent action learning that strips away the assumption of known temporal pairing, enabling broader applicability.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | RoboNet pushing dataset (unlabeled) | Large-scale unlabeled robot interaction videos. |
| Dataset (additional) | Something-Something-v2 (human-object interactions) | Diverse dynamics to test generalization. |
| Primary metric | Task success rate on push-to-target | Measures action understanding directly. |
| Baseline 1 | Random policy (no pretraining) | Lower bound: chance performance. |
| Baseline 2 | VAE-based latent action pretraining | Reconstruction baseline without group consistency. |
| Baseline 3 | Contrastive TCN pretraining | Common self-supervised video representation. |
| Ablation of ours | GCLAL w/o inverse loss | Tests importance of invertibility constraint. |
| Ablation of ours | GCLAL w/o triplet (pair only) | Tests importance of composition over pairs. |

### Why this setup validates the claim

This combination of dataset, metric, and baselines forms a falsifiable test of the central claim: that enforcing group composition in latent action space leads to more meaningful action representations than alternative self-supervised objectives. RoboNet pushing provides unlabeled videos with consistent dynamics, enabling fair comparison; Something-Something-v2 tests generalization to complex, nonlinear dynamics. Success rate directly reflects action quality. The random baseline establishes a lower bound. VAE tests whether reconstruction alone is sufficient; TCN tests whether temporal alignment but not composition works. The ablations test the contributions of the inverse loss and the composition structure. If GCLAL outperforms all baselines on both datasets, it confirms that group consistency is key. If not (e.g., VAE matches GCLAL), the claim is falsified.

### Expected outcome and causal chain

**vs. Random** — On a case where the object must be pushed precisely to a target, the random policy produces erratic, ineffective pushes because it lacks any action understanding. Our method instead learns smooth, goal-directed actions from unlabeled video because the composition loss encourages consistent transitions, so we expect a large gap (e.g., 80% success vs. <10% on RoboNet; similarly large on Something-Something).

**vs. VAE** — On a case requiring multi-step planning (e.g., push around an obstacle), the VAE baseline produces latent actions that collapse to a mean representation because reconstruction alone does not enforce composability. Our method instead maintains distinct, additive actions because the composition loss forces transitive structure, so we expect a noticeable gap on long-horizon tasks (e.g., 70% vs. 30% on RoboNet) but parity on single-step pushes. On Something-Something, the gap may narrow due to nonlinearities.

**vs. TCN** — On a case with action reversals (push left then right), TCN produces temporally smooth but non-invertible embeddings because it only aligns frames locally. Our method captures the inverse relation through the inverse loss, enabling precise backward actions, so we expect a gap on tasks requiring reversals (e.g., 75% vs. 40% on RoboNet).

**Ablation: w/o inverse loss** — Without the inverse loss, actions may not be invertible, leading to degraded performance on reversal tasks.

**Ablation: w/o triplet (pair only)** — Without triplet composition, the model only enforces pairwise consistency, not transitivity, so performance on long-horizon tasks drops.

### What would falsify this idea

If GCLAL's success rate is uniformly similar to VAE across all task subsets (not just on simple cases), that would indicate that group composition does not provide additional benefit over reconstruction, falsifying the claim that additive composition is essential for learning meaningful latent actions. Additionally, if GCLAL performs poorly on Something-Something-v2 while baselines perform well, it would suggest that the additive assumption fails on complex dynamics.

## References

1. PoLAR: Factorizing Extent and Mode in Latent Actions for Robot Policy Learning
2. Learning to Act without Actions
3. Video PreTraining (VPT): Learning to Act by Watching Unlabeled Online Videos
