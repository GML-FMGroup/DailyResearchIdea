# Multi-sample Weight-Space Adaptation for Multi-modal Domain Shifts in VLA Policies

## Motivation

Existing one-shot weight-space adaptation methods (e.g., DART) assume a single target sample captures the full distribution shift, which fails for multi-modal environmental variations (e.g., simultaneous changes in lighting and background). This limitation arises because a single demonstration cannot represent multiple distinct modes of variation. We propose to leverage multiple target samples to characterize the domain shift as a weighted combination of per-sample directions, with weights optimized to minimize worst-case output variance, thereby addressing multi-modal shifts that prior single-sample methods cannot handle.

## Key Insight

A convex combination of per-sample gradient directions, with weights optimized to minimize worst-case loss across samples, yields a more robust domain direction than any single-sample direction because the combination can interpolate between modes without assuming unimodality.

## Method

**Multi-sample Domain Arithmetic (MuDA)**

**(A) What it is:** MuDA adapts a pre-trained VLA policy to a target domain by computing a domain direction as a convex combination of per-sample directions from multiple target demonstrations, where the combination weights are optimized to minimize the worst-case output variance across those demonstrations.

**(B) How it works:**

```pseudocode
Input: Source model weights θ_s, N target demonstrations {(x_i, a_i)}
Output: Adapted weights θ_adapted

Step 1: For each demonstration i, fine-tune the policy on (x_i, a_i) for K gradient steps to obtain θ_i.
         (Hyperparameter: K = 5, learning rate = 1e-5)
Step 2: Compute per-sample direction d_i = θ_i - θ_s  (normalized to unit norm to control scale).
Step 3: Parameterize the combined direction as Δθ = Σ_{i=1}^N w_i d_i, where w = (w_1,...,w_N) is in the N-simplex (w_i ≥ 0, Σ w_i = 1).
Step 4: Solve the minimax optimization:
         min_w max_{i=1..N} L_i(θ_s + Δθ(w))
         where L_i is the action prediction loss on demonstration i.
         Use projected gradient descent on w (learning rate = 0.1, 50 iterations); project w back to simplex each step.
Step 5: Set θ_adapted = θ_s + Δθ(w*).
```

**(C) Why this design:** We chose to compute per-sample directions via fine-tuning rather than single-image gradients because fine-tuning captures nonlinear adaptation effects (e.g., feature reuse) critical for VLA policies, accepting the cost of multiple fine-tunings per domain shift. We chose a convex combination (simplex constraint) over an unweighted average because the optimization can assign zero weight to outlier samples, increasing robustness; this comes at the cost of a more complex optimization. We chose the minimax objective over minimizing average loss because worst-case performance is the key metric for reliable deployment, though it may sacrifice average performance. We selected projected gradient descent over closed-form solution because the objective is non-convex in w, but PG is tractable for small N (<20). Compared to DART, which uses a single direction from one sample, MuDA explicitly addresses multi-modal shifts by learning weights that balance multiple directions; a domain expert would not describe MuDA as a variant of DART because the core operation (weight optimization over multiple directions) is absent in DART.

**(D) Why it measures what we claim:** The combined direction Δθ = Σ w_i d_i operationalizes the concept of "multi-modal domain characterization" because each d_i captures a different mode of variation (via fine-tuning on that sample); the convex combination allows interpolation between modes. The minimax objective min_w max_i L_i(θ_s + Δθ) operationalizes "robust adaptation" because optimizing for the worst case ensures the adapted model performs well on all samples, even the hardest; this assumes that the per-sample directions span the true domain shift modes. This assumption fails when the demonstrations are not representative of the full shift (e.g., missing a mode), in which case the optimized direction may overfit to the observed samples. The normalization of d_i to unit norm ensures equal influence across samples, preventing a sample with large gradient magnitude from dominating; this assumes all samples are equally important for characterizing the shift, which is violated when samples have varying noise levels, causing some to be over-weighted despite being less representative.

## Contribution

(1) A novel multi-sample weight-space adaptation framework that represents domain shifts as convex combinations of per-sample fine-tuned directions, with weights optimized via minimax convex programming. (2) An empirical demonstration that this approach outperforms single-sample DART on multi-modal domain shifts (e.g., simultaneous lighting and background changes) in simulated robotic manipulation tasks. (3) A publicly available set of multi-modal domain shift benchmarks for VLA policies.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Libero benchmark with 4 camera angles | Multi-modal shift; standard VLA evaluation. |
| Primary metric | Success rate across camera angles | Measures robustness to viewpoint variation. |
| Baseline 1 | Source only (no adaptation) | Quantifies necessity of adaptation. |
| Baseline 2 | Full fine-tuning (FT) on single demo | Shows overfitting to one mode. |
| Baseline 3 | Domain Arithmetic (DART) (one-shot) | Tests single-direction baseline. |
| Ablation of ours | MuDA with uniform weights | Isolates benefit of weight optimization. |

### Why this setup validates the claim

This setup tests MuDA‘s central claim: it handles multi-modal domain shifts by optimizing a convex combination of per-sample directions to minimize worst-case loss. The Libero benchmark provides multiple camera angles as explicit modes, making the multi-modal shift observable. The source-only baseline quantifies the need for adaptation. FT on a single demo reveals overfitting to one mode. DART represents a single-direction approach. The uniform-weight ablation measures the contribution of the minimax weight optimization. Success rate across all angles is the right metric because it directly reflects worst-case performance—the key claim. Together, the baselines and ablation isolate whether MuDA’s mechanism (weighted combination with minimax) indeed improves robustness across diverse modes.

### Expected outcome and causal chain

**vs. Source only** — On a sample from an unseen camera angle, the source policy produces a high action loss (e.g., 0.8) because its features are misaligned with the new viewpoint. Our method adapts the policy using multiple demonstrations, shifting features via the optimized direction, so we expect a large success rate gain (e.g., +40%) on that angle, while source only stays near zero.

**vs. Full fine-tuning (FT) on single demo** — On a sample from a camera angle different from the single FT demo, FT overfits to that demo‘s angle, yielding poor performance (e.g., 20% success) on other angles because it only captures one mode. Our method uses multiple directions and minimax weighting to balance across modes, so we expect our method to outperform FT on the non-finetuned angles (e.g., 70% vs. 20%) while being comparable on the finetuned one.

**vs. Domain Arithmetic (DART)** — On a scenario with two distinct background modes (e.g., kitchen and lab), DART uses a single direction from one demo, so it fails on the other mode (e.g., 30% success on the unseen background). Our method combines directions from both modes and optimizes weights for worst-case, so we expect high success on both backgrounds (e.g., 80% each), leading to a noticeable gap in the hardest mode.

**vs. MuDA with uniform weights** — On a dataset where one sample is noisy (e.g., blurry image), uniform weighting gives it equal influence, harming performance (e.g., 60% worst-case). Our minimax optimization downweights that sample, yielding better worst-case (e.g., 75%). Thus, we expect a clear gap on the subset where a distracting sample exists.

### What would falsify this idea

If MuDA’s worst-case success rate is not higher than that of uniform weighting or DART on the hardest subset (e.g., the most extreme camera angle), or if the improvement is uniform across all angles rather than concentrated on the hardest modes, then the minimax weight optimization is not providing the claimed robustness.

## References

1. Domain Arithmetic: One-Shot VLA Adaptation under Environmental Shifts
2. Robust Finetuning of Vision-Language-Action Robot Policies via Parameter Merging
3. VLA Models Are More Generalizable Than You Think: Revisiting Physical and Spatial Modeling
4. LiNeS: Post-training Layer Scaling Prevents Forgetting and Enhances Model Merging
5. π0: A Vision-Language-Action Flow Model for General Robot Control
6. Vision-Language Foundation Models as Effective Robot Imitators
7. 3D-VLA: A 3D Vision-Language-Action Generative World Model
8. Conditional Prompt Learning for Vision-Language Models
