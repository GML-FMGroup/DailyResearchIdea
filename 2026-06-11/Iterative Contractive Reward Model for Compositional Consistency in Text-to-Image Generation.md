# Iterative Contractive Reward Model for Compositional Consistency in Text-to-Image Generation

## Motivation

Existing reward models for text-to-image generation, such as SpatialScore and PIGReward, rely on single-pass evaluation—either a global scalar score or a chain-of-thought reasoning that produces one final output without iterative refinement. This single-pass paradigm fails to capture compositional or stepwise inconsistencies; for example, SpatialScore cannot verify that spatial relations hold across multiple objects, and PIGReward's CoT reasoning lacks stepwise verification of its own reasoning chain. The root cause is that these methods do not enforce transitivity or decomposition consistency, leading to reward estimates that may contradict themselves when evaluated on subcomponents or across multiple trials.

## Key Insight

A contractive fixed-point iteration over a decomposed reward function guarantees a unique, self-consistent reward score that is invariant to the order of evaluation, thereby enforcing transitivity and compositional consistency.

## Method

**Iterative Contractive Reward Model (ICRM)**

(A) **What it is:** ICRM is an implicit neural network that takes an image-text pair (optionally with intermediate decompositions like spatial bounding boxes or sub-prompts) and outputs a scalar reward score via a fixed-point iteration. The iteration is defined by a contractive mapping that combines sub-reward predictions from a shared scoring network.

(B) **How it works:**
```python
# Pseudocode for ICRM forward pass
def icrm_reward(text, image, decomposition=[], epsilon=1e-4, max_iter=50, gamma=0.9):
    # decomposition: list of sub-tasks or sub-relations (e.g., object counts, spatial relations)
    # Initialize reward estimate
    r = 0.5  # initial guess
    for i in range(max_iter):
        # For each decomposition element, compute a sub-reward using a shared network
        sub_rewards = []
        for d in decomposition:
            sub_r = predict_sub_reward(text, image, d)  # returns scalar in [0,1]
            sub_rewards.append(sub_r)
        # Compute new reward as weighted combination of sub_rewards and previous r
        r_new = gamma * mean(sub_rewards) + (1 - gamma) * r
        # Stopping criterion: contractive update guarantees convergence
        if abs(r_new - r) < epsilon:
            break
        r = r_new
    return r
```
The shared network `predict_sub_reward` is trained to predict a scalar alignment for a given sub-relation (e.g., 'left of', 'blue circle'). The hyperparameter gamma in (0,1) controls the contraction rate; gamma close to 1 emphasizes current sub-reward mean, while gamma close to 0 smooths over iterations.

(C) **Why this design:**
We chose iterative refinement over a single-pass evaluator because single-pass methods cannot detect inconsistencies that only appear when evaluations are repeated or composed. The contractive update (with gamma < 1) guarantees a unique fixed point by Banach's fixed-point theorem, ensuring that different evaluation orders or starting points converge to the same reward—this directly enforces transitivity. We use a mean of sub-rewards as the update target rather than a learned fusion, because a learned fusion might introduce a another source of inconsistency; the mean is interpretable and requires no additional parameters. The stopping criterion based on epsilon provides a controllable trade-off between computational cost and precision: a small epsilon guarantees near-exact fixed point but increases iterations. We chose to share the sub-reward predictor across all decomposition types rather than using specialized heads, accepting that the predictor may be less accurate for rare sub-relations, in exchange for parameter efficiency and better generalization across sub-tasks.

(D) **Why it measures what we claim:**
The fixed-point reward r* after convergence measures **compositional consistency** because the iteration forces agreement between the aggregated sub-rewards and the global reward; if the sub-rewards are contradictory (e.g., 'left of' and 'right of' both high), the mean will be moderate and the fixpoint will reflect the tension. The contractive property ensures **transitivity** because the uniqueness of the fixed point implies that evaluating the same image-text pair under different orders of sub-reward combinations yields the same result; this assumption fails if the reward function is not Lipschitz continuous in the space of sub-reward means, in which case the fixed point may be sensitive to the decomposition order, but our design mitigates this by using a simple averaging rather than a learned nonlinear aggregation. The **stepwise verification** is operationalized by the iterative process itself: each iteration acts as a verification step that the current reward is consistent with the sub-rewards; the number of iterations provides an upper bound on how many verification steps were needed, and if convergence fails (max_iter reached without epsilon), it indicates that the decomposition or sub-reward model is fundamentally inconsistent with a consistent global reward.

## Contribution

(1) A novel iterative contractive reward model (ICRM) that replaces single-pass evaluation with a fixed-point iteration enforcing compositional consistency and transitivity. (2) A design principle showing that a contractive mapping over decomposed sub-rewards yields a unique, self-consistent reward score. (3) An open-source implementation and evaluation protocol for compositional reward evaluation on text-to-image benchmarks (to be released).

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|----------------------|
| Dataset | CLEVR-Compositional | Synthetic, controlled relations for consistency tests |
| Primary metric | Consistency Accuracy | Measures transitivity and compositionality |
| Baseline 1 | CLIPScore | Single-pass similarity, lacks iterative refinement |
| Baseline 2 | LearnedAgg | Neural network aggregation of sub-rewards |
| Ablation-of-ours | Ours w/o iteration | Single-pass mean of sub-rewards |

### Why this setup validates the claim
CLEVR-Compositional provides ground-truth relations, enabling exact evaluation of compositional consistency. Consistency Accuracy directly measures transitivity (if A>B and B>C then A>C) and agreement across decompositions. CLIPScore tests whether a global single-pass method can capture composition; LearnedAgg tests whether learned fusion helps; our ablation isolates the benefit of iteration. This design creates a falsifiable test: if our method truly enforces transitivity via contractive fixed points, it must outperform baselines on consistency, especially on multi-relation prompts where sub-rewards may conflict.

### Expected outcome and causal chain

**vs. CLIPScore** — On a prompt like "a red cube left of a blue sphere and right of a green cylinder", CLIPScore produces a scalar that averages all visual features, missing the relational contradiction if both "left of" and "right of" are incorrectly scored high. Our method iteratively refines by averaging sub-rewards for each relation; if sub-rewards are contradictory, the fixed point settles at a moderate value, yielding a consistent score across permutations. We expect a noticeable gap (e.g., 15-20% higher Consistency Accuracy) on prompts with ≥2 relations, but parity on single-relation prompts.

**vs. LearnedAgg** — On a novel composition like "a red cube above a blue sphere that is left of a green cylinder", a learned aggregation network may extrapolate poorly because it never saw that specific combination during training, leading to inconsistent global rewards. Our method uses a simple mean (no learned parameters), so the fixed point depends only on sub-reward predictions, not on training distribution. We expect our method to show lower variance (e.g., 10% higher Consistency Accuracy) on unseen compositions, and comparable performance on seen ones.

**vs. Ours w/o iteration** — On ambiguous cases where sub-rewards are nearly equal (e.g., all around 0.5), the single-pass mean fluctuates with decomposition order, causing inconsistency. Our iterative refinement converges to a unique fixed point regardless of order, stabilizing the reward. We expect our full method to have near-perfect Consistency Accuracy (e.g., >95%), while the ablation may hover around 80% on such ambiguous prompts.

### What would falsify this idea
If our method's Consistency Accuracy is not significantly higher than the single-pass ablation on multi-relation prompts, or if it performs worse than LearnedAgg on any systematic subset (e.g., prompts with exactly two relations), then the central claim that iteration enforces transitivity and consistency is false.

## References

1. Enhancing Spatial Understanding in Image Generation via Reward Modeling
2. Pref-GRPO: Pairwise Preference Reward-based GRPO for Stable Text-to-Image Reinforcement Learning
3. Hunyuan-DiT: A Powerful Multi-Resolution Diffusion Transformer with Fine-Grained Chinese Understanding
4. CapsFusion: Rethinking Image-Text Data at Scale
5. Improved Baselines with Visual Instruction Tuning
6. Personalized Reward Modeling for Text-to-Image Generation
7. VisionReward: Fine-Grained Multi-Dimensional Human Preference Learning for Image and Video Generation
8. Improve Vision Language Model Chain-of-thought Reasoning
