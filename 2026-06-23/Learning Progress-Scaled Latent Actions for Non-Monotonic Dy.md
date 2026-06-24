# Learning Progress-Scaled Latent Actions for Non-Monotonic Dynamics

## Motivation

PoLAR factorizes latent actions into radial extent and directional mode but relies on temporal offset as a proxy for extent, which fails when the relationship between time and transition magnitude is non-monotonic (e.g., oscillations). This structural assumption is violated in many physical processes, limiting pretraining applicability. We need a method that infers extent without assuming monotonic time-extent correlation.

## Key Insight

By learning a scalar progress variable from pairwise temporal ordering and enforcing that the radial component of the latent action equals this progress, we recover a metric of transition extent that is decoupled from temporal offset and can handle non-monotonic dynamics.

## Method

We propose Progress-Regularized Latent Actions (PoLAR-P). (A) **What it is**: PoLAR-P is an unsupervised latent action pretraining method that replaces temporal offset supervision with a learned progress scalar. It takes pairs of observations (o_t, o_{t+∆t}) and outputs a latent action (r, θ) where r is the radial extent and θ is the directional mode. (B) **How it works**: Pseudocode in Python style.

```python
# Hyperparameters: λ_ord = 0.1, λ_kl = 0.01, prior rate = 1.0, temperature τ = 1.0
# Encoder: 4-layer CNN with 256-dim output
# Decoder: 4-layer deconv network
# Hyperbolic operations use Poincaré ball model with curvature 1.0:
#   exp_map(v) = tanh(||v||) * v / ||v||
#   add_hyperbolic(z, a) = mobius_add(z, exp_map(a))
#   where mobius_add uses the standard formula.
# Progress predictor MLP: 2-layer MLP with hidden=256 and GeLU activation, output scalar p.

for each batch of N observation pairs (o_i, o_j) with unknown temporal offset:
    # Encode observations
    z_i = encoder(o_i)  # [batch, 256]
    z_j = encoder(o_j)
    # Sample latent action from hyperbolic space: radius r and direction θ
    # Direction θ is a point on the unit sphere in Euclidean space sampled from von Mises-Fisher
    # Exponential map to hyperbolic: θ_hyp = exp_map(θ * scale) where scale is 1.0
    # Progress scalar p is predicted via MLP: p = MLP(concat([z_i, z_j]))
    # Constraint: r = p (norm equality), so r ≥ 0 enforced via ReLU
    # Decode next observation: o_pred = decoder(add_hyperbolic(z_i, (r, θ_hyp)))
    # Losses:
    # 1. Reconstruction loss: L_rec = MSE(o_pred, o_j)
    # 2. Ordering contrastive loss: for each pair (i,j), sample a negative pair (i,k) from a different trajectory (uniform time offset)
    #    L_order = -log(sigmoid((p_ij - p_ik) / τ))
    # 3. Prior regularization: KL divergence between p and Exponential(rate=1.0) (using closed-form)
    # Total loss: L = L_rec + λ_ord * L_order + λ_kl * L_kl
```

(C) **Why this design**: We chose to predict progress from observation embeddings rather than from a separate temporal module because it naturally integrates with the encoder and avoids an extra time-encoding head that would require explicit time stamp inputs. We enforce r = p directly rather than using a soft penalty (e.g., L2 loss between r and p) because equality constraints are necessary to ensure the radial component truly represents progress; a soft penalty could allow r to drift and still correlate with temporal offset. We use an ordering contrastive loss on p rather than regression to a known temporal offset because in non-monotonic dynamics, the actual temporal offset is not a reliable measure of progress; ordering provides a weaker but more robust signal that only requires that progress increases along the trajectory. The trade-off for the equality constraint is that it restricts the expressivity of the radial component — it must exactly match the scalar progress, which may limit the capacity to capture other aspects of extent, but we accept this because progress is the only aspect we want to capture. The ordering loss avoids temporal supervision but requires pairs with known ordering, which is available in any sequential data. We chose a prior exponential distribution for p to reflect that smaller steps are more common, which aligns with typical dynamics. **Load-bearing assumption**: We assume that the ordering contrastive loss alone, without metric supervision, can learn a progress scalar that reflects actual transition extent (path length) in non-monotonic dynamics. To verify this during training, we monitor the Spearman correlation between p and the Euclidean distance ||z_i - z_j|| on a held-out validation set (1K pairs). If the correlation falls below 0.5 after 10 epochs, we flag a potential assumption violation.

(D) **Why it measures what we claim**: The scalar progress p measures the amount of change relative to the trajectory direction, as inferred from temporal ordering; the assumption is that progress increases along the trajectory direction even if the temporal offset is non-monotonic. This assumption fails if the trajectory reverses direction (e.g., a pendulum swings back), in which case p may increase even though the system is returning to a previous state; in that case, p would reflect cumulative distance traveled along the path rather than displacement. Additionally, if dynamics are highly non-uniform (e.g., long pauses followed by sudden movements), the ordering signal may be insufficient to disentangle magnitude from time; p may then correlate with time rather than change. To diagnose this, we compute the correlation between p and the Euclidean distance between latent embeddings on a validation set. The radial component r is forced to equal p by the norm equality constraint, so r measures the same quantity; thus, the latent action extent is decoupled from temporal offset and reflects path length in latent space. This ensures that extent is not misled by oscillations in time.

## Contribution

(1) A novel unsupervised latent action pretraining method that replaces temporal offset supervision with a learned progress scalar trained via temporal ordering constraints and coupled to the radial component via a norm equality constraint. (2) The first latent action framework that is robust to non-monotonic dynamics, enabling pretraining on diverse dynamical data where time and extent are not monotonically related. (3) Empirical demonstration that the learned progress scalar does not correlate with temporal offset when dynamics are non-monotonic, while still enabling downstream policy learning.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
|------|--------|-----------|
| Dataset | MetaWorld MT25 | Diverse tasks with non-monotonic dynamics (e.g., push, door-open) |
| Dataset (real-world) | Real robot pushing sequences (e.g., RoboTurk) | Tests non-monotonic dynamics in realistic setting with high-dimensional observations |
| Primary metric | Task success rate (average over 10 seeds) | Direct measure of utility |
| Baseline 1 | Unsup. latent actions (PoLAR) | Tests benefit of progress scalar |
| Baseline 2 | Supervised latent actions (T-offset) | Tests advantage of unsupervised ordering |
| Baseline 3 | Behavioral cloning (BC) | Lower bound with full supervision |
| Ablation 1 | PoLAR-P w/o ordering loss | Isolates ordering loss contribution |
| Ablation 2 | PoLAR-P with temporal offset as progress (fixed p = ∆t) | Isolates novelty of learned progress scalar |

### Why this setup validates the claim

This experimental design tests the central claim that PoLAR-P learns meaningful latent actions by decoupling extent from temporal offset via a learned progress scalar and ordering contrastive loss. The dataset includes non-monotonic dynamics where temporal offset is misleading, making it a critical test for the progress scalar. The real-world robotic task provides a more challenging domain with high-dimensional pixel inputs and physical non-monotonicity. Comparing to unsupervised latent actions (no progress scalar) tests whether the scalar provides additional structure. Comparing to supervised latent actions (using known temporal offsets) tests whether the weak ordering supervision is sufficient. Behavioral cloning serves as a lower bound with full action supervision. Ablation 1 removes the ordering loss to isolate its contribution; ablation 2 replaces the learned progress with raw temporal offset to measure the value of learning p. Task success rate directly measures how well the learned latent actions facilitate downstream policy learning, providing a falsifiable metric.

### Expected outcome and causal chain

**vs. Unsup. latent actions (PoLAR)** — On a case where dynamics reverse (e.g., pendulum swinging back), PoLAR predicts large latent actions for large temporal offsets even if little actual change, misrepresenting extent. Our method's progress scalar p tracks path length, so latent actions remain small during reversal. We expect PoLAR-P to achieve noticeably higher success on tasks with oscillations (e.g., pushing, door-open) while matching on monotonic tasks. Quantitatively, we expect >5% absolute improvement in success rate on the oscillation-heavy tasks in MetaWorld.

**vs. Supervised latent actions (T-offset)** — On a task with variable execution speed (e.g., slow then fast), the supervised method ties extent to time, overestimating for slow steps. Our ordering loss ignores magnitude, so progress adapts to actual change. We expect PoLAR-P to match or exceed the supervised baseline on variable-speed tasks and be comparable on constant-speed tasks. On real robot pushing where speed varies, we expect PoLAR-P to outperform T-offset by at least 3%.

**vs. Behavioral cloning (BC)** — On a long-horizon task with high-dimensional action space, BC suffers from compounding errors due to distribution shift. Our latent actions encode lower-level dynamics, reducing error accumulation. We expect PoLAR-P to outperform BC on long-horizon tasks (e.g., >50 steps) while BC may be better on short-horizon tasks. In MetaWorld, we expect PoLAR-P to exceed BC on tasks requiring more than 30 steps.

### What would falsify this idea

If PoLAR-P fails to outperform unsupervised latent actions on tasks with non-monotonic dynamics, or if its gains are uniform across all tasks rather than concentrated on subsets where temporal offset is misleading, then the progress scalar and ordering loss are not functioning as intended. Additionally, if the correlation between p and transition norm on the validation set remains low (<0.5) after training, it would indicate that the ordering loss alone is insufficient without metric supervision.

## References

1. PoLAR: Factorizing Extent and Mode in Latent Actions for Robot Policy Learning
2. Learning to Act without Actions
3. Video PreTraining (VPT): Learning to Act by Watching Unlabeled Online Videos
