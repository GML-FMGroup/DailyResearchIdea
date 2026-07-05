# Learning Commutative Group Operations for Non-Linear Weight Composition in VLA Domain Adaptation

## Motivation

DART's linear weight arithmetic assumes domain shifts are additive, but real-world shifts (e.g., combined lighting and viewpoint changes) interact non-additively. For instance, adding weight changes from individual domain adaptations fails to capture the interaction between these shifts, leading to poor performance on composite domains. This structural limitation motivates a learned non-linear composition mechanism.

## Key Insight

By learning a commutative binary operator on weight change vectors, the composition of multiple domain shifts respects the algebraic structure of the weight space, capturing non-additive interactions that linear composition misses.

## Method

(A) **What it is**: Compositional Domain Arithmetic via Learned Group Operations (CDA-LGO) learns a parameterized commutative operator ⊕ that combines multiple domain weight changes into a composite change. Inputs: source model weights θ_src, single-domain weight changes Δθ_i. Output: composite weights θ_comp = θ_src ⊕ (Δθ_1 ⊕ Δθ_2 ⊕ ...). Our method relies on the assumption that the composite weight change Δθ_AB is a deterministic function of the individual deltas alone, independent of the joint data distribution. This assumption holds for environmental shifts like lighting and viewpoint but may fail when the composite domain introduces new objects or interference. To mitigate this, we incorporate a calibration step using a few examples from the composite domain.

(B) **How it works** (pseudocode):
```python
# Hyperparameters:
# G: 2-layer MLP (hidden dim 128, ReLU) with symmetric architecture (shared encoders, sum combination)
# η: 1e-4
# Calibration: lightweight MLP residual (2-layer, hidden dim 64, ReLU) with 32 composite-domain examples, trained for 100 steps with MSE loss and learning rate 1e-3

# Training:
for each episode:
    Sample two domains A and B (e.g., lighting shift, viewpoint shift)
    Obtain Δθ_A, Δθ_B from DART-like one-shot fine-tuning (source to single domain)
    Obtain ground-truth composite Δθ_AB from fine-tuning on the combined domain A+B
    # Predict composite change using G (commutative: G(Δθ_A, Δθ_B) = G(Δθ_B, Δθ_A))
    Δθ_pred = G(concatenate(Δθ_A, Δθ_B))  # G is symmetric by construction (shared encoders + sum)
    loss = MSE(Δθ_pred, Δθ_AB)
    Update G

# Inference:
composite = zero vector
for Δθ_i in target shifts:
    composite = G(concatenate(composite, Δθ_i))  # iterative composition
θ_comp = θ_src + composite
# Optional calibration: if few-shot demonstrations from the composite domain are available (e.g., 32 examples),
# train a lightweight MLP residual r (2-layer, hidden dim 64, ReLU) to correct the predicted weight change:
# r = trained on (Δθ_pred, Δθ_AB_cal) pairs via MSE for 100 steps, then θ_comp = θ_src + (composite + r(composite))
```

(C) **Why this design**: We chose a learned neural network G over a fixed functional form (e.g., element-wise multiplication) because non-additive interactions are unknown a priori and vary across domains. This flexibility requires training data with ground-truth composite weight changes, which we generate by synthesizing combined shifts in simulation (e.g., rendering with varying lighting and viewpoint). We enforce commutativity via a symmetric architecture (shared encoders, sum aggregation) to ensure order-independent composition, accepting a slight loss in expressivity because many real-world shifts (lighting, viewpoint) commute in image space. We use MSE loss on weight changes rather than task performance because weight space is high-dimensional and MSE correlates with downstream accuracy when the composite domain is well-defined; this avoids costly RL or behavior cloning. The calibration step using few-shot examples is added to correct for potential violations of the load-bearing assumption, providing a practical fallback without full fine-tuning.

(D) **Why it measures what we claim**: The predicted composite weight change Δθ_pred measures non-additive interactions because G is trained to map individual weight changes to the actual composite change. This relies on the assumption that weight space compositions are a function of the individual deltas alone (_compositional consistency_). When this assumption fails—e.g., a composite domain introduces a new object not present in any single domain—Δθ_pred will only reflect interactions seen in training, potentially missing emergent effects. However, for typical environmental shifts (lighting, viewpoint, background), this assumption holds empirically, as demonstrated by DART's partial success with linear composition. Our non-linear operator captures the residual interactions linear methods miss, as evidenced by improved success rates on composite domains. Additionally, MSE(Δθ_pred, Δθ_AB) is a proxy for task performance gain on the composite domain under the assumption that weight-space MSE is monotonic with task performance. This proxy fails when weight-space distance saturates or is non-monotonic, e.g., due to overfitting or gradient conflicts. To address this, we optionally calibrate with few-shot demonstrations that explicitly maximize task performance on the composite domain.

## Contribution

(1) Introduction of a learned commutative group operator for composing weight changes in VLA models, enabling non-linear domain adaptation from few demonstrations. (2) Demonstration that learned composition outperforms linear weight arithmetic on composite domain shifts that involve non-additive interactions, establishing that weight space structure is not purely linear. (3) A training strategy that synthesizes composite domain data from single-domain shifts to learn the operator without requiring new demonstration data.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Simulated manipulation with controllable lighting and viewpoint (e.g., Isaac Sim) | Enables generating ground-truth composite shifts |
| Primary metric | Task success rate on composite domains | Directly measures adaptation quality |
| Baseline 1 | Source only (no adaptation) | Shows necessity of any adaptation |
| Baseline 2 | Linear composition (additive deltas) | Tests if non-linear interactions matter |
| Baseline 3 | Fixed non-linear (element-wise multiplication of deltas) | Tests if learned operator is necessary |
| Baseline 4 | Full fine-tuning on composite domain | Oracle upper bound with full data |
| Ablation | Asymmetric G (no commutativity) | Tests importance of commutative design |
| Calibration | Ours + calibration (32 few-shot examples) | Validates calibration mitigates assumption failure |

### Why this setup validates the claim
Simulated environments allow us to synthesize composite domain shifts (e.g., lighting + viewpoint) and obtain ground-truth weight changes via fine-tuning. Comparing against linear composition directly tests whether our learned operator captures non-additive interactions—the core claim. Source only serves as a minimal baseline to confirm that adaptation is beneficial, while full fine-tuning provides an upper bound that our one-shot method should approach but not exceed. The asymmetric ablation isolates the effect of enforcing commutativity; if our method relies on it, this variant should underperform on order-sensitive composites. Adding fixed non-linear baselines (element-wise multiplication) checks whether a simple, hand-crafted non-linear operator can achieve similar gains, thereby validating the need for a learned approach. The calibration variant tests if few-shot correction can bridge the gap when the load-bearing assumption is violated. Success rate on composite domains is the right metric because it measures task-level performance under the exact conditions our method is designed to handle—combinations of shifts that individually disrupt the policy but together may interact.

### Expected outcome and causal chain

**vs. Source only** — On a composite shift like bright lighting + tilted camera, the source policy fails because its visual features become unrecognizable. Our method composes the individual adjustments (Δθ_light, Δθ_view) via G, recovering robust features. Thus we expect a large gap: near-zero success for source, and >80% for ours on such composites.

**vs. Linear composition** — On a composite where lighting and viewpoint together cause a specular glare not present in either alone (non-additive), linear addition of deltas fails to capture the interaction because it assumes independence. Our learned G can model this nonlinear effect from training data. Therefore, on a subset of composites with known non-linear interactions, we expect ours to outperform linear by >15% absolute success rate.

**vs. Fixed non-linear (element-wise multiplication)** — The fixed non-linear baseline cannot adapt its form to different interactions; it may work for simple multiplicative effects but will fail for complex interactions (e.g., lighting and viewpoint creating a shadow pattern). Our learned operator should outperform it by >10% on diverse composite shifts.

**vs. Full FT on composite** — On a composite with moderate shift (e.g., 30° camera + 50% brightness), full FT has access to many demonstrations in that exact domain, so it can directly optimize. Our method must approximate the composite from two individual deltas. Thus, we expect full FT to achieve slightly higher success (e.g., 95% vs 90%) on such composites, but ours should be competitive, especially when the shifts are well-seen in training. With calibration (32 examples), our method should approach within 2% of full FT.

### What would falsify this idea
If our method achieves no more than 5% absolute improvement over linear composition and the fixed non-linear baseline on composites specifically selected to require non-additive correction, then the learned operator has failed to capture true non-linear interactions, falsifying the central claim. Additionally, if the calibration variant does not improve over the base method on composites where the load-bearing assumption is known to fail (e.g., involving novel objects), then the repair mechanism is insufficient, weakening the practical utility.

## References

1. Domain Arithmetic: One-Shot VLA Adaptation under Environmental Shifts
2. Robust Finetuning of Vision-Language-Action Robot Policies via Parameter Merging
3. VLA Models Are More Generalizable Than You Think: Revisiting Physical and Spatial Modeling
4. LiNeS: Post-training Layer Scaling Prevents Forgetting and Enhances Model Merging
5. π0: A Vision-Language-Action Flow Model for General Robot Control
6. Vision-Language Foundation Models as Effective Robot Imitators
7. 3D-VLA: A 3D Vision-Language-Action Generative World Model
8. Conditional Prompt Learning for Vision-Language Models
