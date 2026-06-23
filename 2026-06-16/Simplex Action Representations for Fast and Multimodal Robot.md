# Simplex Action Representations for Fast and Multimodal Robot Control

## Motivation

Existing action representations either rely on fast unimodal regression (e.g., L1 loss in Fine-Tuning VLA) or expensive multimodal diffusion (e.g., Diffusion Policy), leaving a trade-off between speed and multimodality in contact-rich manipulation. This structural failure arises because neither approach models the output as a convex combination of learned action modes, which would allow a single feedforward pass to capture multiple valid outcomes.

## Key Insight

By representing an action as a convex combination of a learned dictionary of anchor actions, the model can express multiple distinct behaviors in a single forward pass while inheriting the speed of regression, because each combination corresponds to a different mode depending on the weight distribution.

## Method

(A) **What it is**: SimAct predicts an action a as a convex combination of K learned anchor actions via a feedforward network that outputs weights w = softmax(z).

(B) **How it works**:
```python
# Training
Initialize anchor actions A = [a_1, ..., a_K] as learnable tensors of same dimension as action.
Initialize weight predictor network f_theta (e.g., 2-layer MLP with 128 hidden units, GeLU activation).
Backbone: OpenVLA vision-language encoder (ViT-L+LLaMA pretrained).
For each batch (obs, target_action):
    features = encode(obs)   # from VLA backbone
    logits = f_theta(features)   # K-dimensional
    w = softmax(logits)   # convex combination weights
    a_pred = sum_i w_i * a_i   # predicted action
    loss = L1_loss(a_pred, target_action) + lambda * entropy(w)
    # entropy(w) = -sum w_i log w_i
    Update parameters.
# Inference
Obs = encode(obs)
logits = f_theta(obs)
w = softmax(logits)
a = sum_i w_i * a_i
```
Hyperparameters: K=5, lambda=0.1 (initial), learning rate=1e-4, trained for 50k steps, batch size=64.

(C) **Why this design**: We chose convex combination over a mixture model (e.g., MoE) because convex combination yields a single continuous action at inference without needing to sample or average multiple outputs, accepting the cost that the representation is inherently linear in the anchor space, which may limit expressivity for highly nonlinear action manifolds. We chose softmax to enforce sum-to-one and non-negativity, and entropy penalty to encourage sparse weight vectors (few active anchors) for interpretability and to prevent collapse to one anchor; this introduces an extra hyperparameter lambda that requires tuning. We chose to learn anchors jointly with the weight predictor rather than fixing them via clustering, accepting the risk that anchors may drift to uninterpretable regions, but allowing them to adapt to task-specific actions. We used L1 loss for robustness to outliers, as is common in VLA fine-tuning.

**Load-bearing assumption**: A convex combination of K=5 learned anchors can represent multiple distinct action modes because each anchor is learned to represent a prototypical action, and the weight distribution captures the mode's likelihood. This is only true if anchors are well-separated. To calibrate, we monitor pairwise cosine similarity between anchors during training; if any pair exceeds threshold 0.8, we increase lambda by 0.05 (up to max 1.0) to encourage separation. This diagnostic is computed every 500 steps.

(D) **Why it measures what we claim**: The computational quantity w (weight vector) measures multimodality because a high-entropy w indicates multiple anchors are active, corresponding to multiple plausible actions; this equivalence assumes that each learned anchor represents a distinct action mode and that the weight distribution captures the relative likelihood of each mode. This assumption fails when anchors are not well-separated (e.g., if the network learns similar anchors), in which case high entropy reflects uncertainty rather than true multimodality; we detect this via the cosine similarity diagnostic (threshold 0.8). The quantity a_pred (convex combination) measures speed because it is computed in a single forward pass without iterative sampling; this assumes that the feedforward computation is the bottleneck, ignoring potential overhead from the anchor dictionary lookup. The entropy penalty ensures sparse activation, operationalizing the concept of 'mode selection' under the assumption that a small set of anchors can cover the action space; when the action space requires many modes, the penalty may force false sparsity, and the representation degrades to unimodal.

## Contribution

['(1) A novel action representation that unifies fast regression and multimodality by representing actions as convex combinations of learnable anchor actions.', '(2) A training objective combining L1 regression loss with entropy regularization to encourage sparse mode selection, enabling interpretable identification of action modes.', '(3) Empirical demonstration that the simplex representation matches the inference speed of unimodal regression while achieving multimodal behavior comparable to diffusion policies (to be evaluated).']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|----------------------|
| Dataset | LIBERO-90 (90 tasks) | Tests multimodality and speed |
| Primary metric | Success rate (%) | Direct measure of task completion |
| Baseline 1 | OpenVLA (no fine-tuning) | Strong VLA baseline without adaptation |
| Baseline 2 | Standard fine-tuned VLA | Shows benefit of fine-tuning |
| Baseline 3 | MoE-VLA (mixture of linear experts) | Compares convex combination to gated mixture |
| Ablation | SimAct w/o entropy penalty | Isolates effect of sparsity regularization |

### Why this setup validates the claim

The combination of LIBERO-90 benchmark, success rate metric, three complementary baselines including a mixture-of-experts variant, and an ablation forms a falsifiable test of SimAct's central claim: that convex combination of learned anchor actions with entropy regularization improves both speed and multimodality handling in real-time VLA interaction. LIBERO-90 provides diverse long-horizon tasks that inherently require multiple action modes, making it ideal to detect multimodality improvements. Success rate captures overall task performance, which is the ultimate goal. OpenVLA (no fine-tuning) tests whether any fine-tuning is beneficial, standard fine-tuned VLA isolates the effect of our specific architecture, and MoE-VLA directly compares convex combination against an alternative mixture approach. The ablation tests whether entropy regularization is necessary for sparsity and mode selection. If SimAct outperforms both baselines and the ablation fails, the causal role of our design choices is validated.

### Expected outcome and causal chain

**vs. OpenVLA (no fine-tuning)** — On a task like "pick up the rice bowl and place it in the microwave" in LIBERO, the initial action distribution is multimodal (reach vs. grasp vs. move). OpenVLA's single feedforward output averages these modes, resulting in hesitant or inefficient motion. Our method activates multiple anchors via softmax weights; the entropy penalty forces sparsity after a few steps, sharpening to a single decisive action. Thus, we expect a 10–15% higher success rate on tasks with high multimodality, but parity on simple pick-and-place tasks.

**vs. Standard fine-tuned VLA** — On a task requiring rapid mode switching, such as LIBERO's "open the drawer and then pick up the object inside", standard fine-tuned VLA must learn separate output weights for each subtask, often blending transitions. Our method dynamically adjusts anchor weights per observation, enabling sharp transitions without forgetting. The same feedforward speed is maintained. We expect a 5–10% success rate improvement on tasks with subtask structure, as observed in the `functional` suite of LIBERO.

**vs. MoE-VLA** — MoE-VLA uses a gated mixture of linear experts (each expert predicts an action) and samples one expert per timestep, which introduces stochasticity and slower inference. Our method's deterministic convex combination is faster and avoids sampling noise. We expect comparable success rates but faster inference (measured in ms per inference step).

### What would falsify this idea

If our method's success rate is not significantly higher on tasks identified a priori as multimodal compared to all baselines, or if the entropy penalty ablation performs equally well, the central claim that convex combination with sparsity improves multimodality handling would be invalidated. Additionally, if the method is highly sensitive to lambda (e.g., performance varies >10% across lambda in [0.01, 0.5]), the design's robustness is questionable.

## References

1. Fine-Tuning Vision-Language-Action Models: Optimizing Speed and Success
2. Language Conditioned Multi-Finger Dexterous Manipulation Enabled by Physical Compliance and Switching of Controllers
3. RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control
4. Diffusion policy: Visuomotor policy learning via action diffusion
5. VIMA: General Robot Manipulation with Multimodal Prompts
6. RT-1: Robotics Transformer for Real-World Control at Scale
7. Is Conditional Generative Modeling all you need for Decision-Making?
8. Planning with Diffusion for Flexible Behavior Synthesis
