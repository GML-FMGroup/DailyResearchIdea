# Learning Task-Agnostic Action Representations via Contrastive Predictive Coding for Zero-Shot Generalization in Vision-Language-Action Models

## Motivation

Existing VLAs (e.g., OpenVLA-OFT, TinyVLA) require task-specific demonstration data for fine-tuning to achieve high performance on new tasks, limiting zero-shot generalization. The root cause is that these models learn a direct mapping from task instructions to actions, capturing task-specific correlations rather than general environment dynamics. This structural assumption—that supervised fine-tuning on target-task demos is necessary—perpetuates across models and prevents adaptability to novel tasks without additional data.

## Key Insight

Contrastive predictive coding on future observations extracts temporal structure that is invariant to task identity, providing a universal action representation that can be used to infer actions for arbitrary goal conditions.

## Method

### FRAME (Future Representation for Action from Motion and Environment)

**Input:** A sequence of raw video frames from diverse, unlabeled robot demonstrations (o_1,...,o_T) with corresponding actions (a_1,...,a_{T-1}), and a goal image G.

**Output:** A continuous action sequence that achieves the goal.

**Training phase:**

```python
# Hyperparameters:
# - Encoder: ResNet-50, output dim=512
# - Temporal transformer: GPT-like, 6 layers, 8 heads, embedding dim=512
# - CPC predictor MLP: 2-layer MLP, hidden=256, output dim=512, GeLU activation
# - InfoNCE temperature: tau=0.07
# - Inverse model MLP: 2-layer MLP, hidden=256, output dim=action_dim, ReLU
# - Loss weight lambda=0.1

for each batch of 32 videos:
    o_seq = [o_1, o_2, ..., o_T]
    z_seq = ResNet50(o_seq)  # each z_t is a 512-dim vector
    c_seq = Transformer(z_seq)  # context vectors, 512-dim
    # CPC loss
    L_cpc = 0
    for t in 1..T-k:
        pred = MLP_predictor(c_t)  # predict z_{t+k}
        positive = z_{t+k}
        # sample 512 negatives from other timesteps/sequences
        negative = random_sample_negatives(512)
        L_cpc += -log( exp(sim(pred, positive)/tau) / sum(exp(sim(pred, neg)/tau)) )
    # Inverse dynamics: predict action a_t from (z_t, z_{t+1})
    for t in 1..T-1:
        a_pred = inverse_model(concat(z_t, z_{t+1}))
        L_inv = MSE(a_pred, a_t)
    total_loss = L_cpc + lambda * L_inv
    # Optimizer: AdamW, lr=3e-4, weight decay=1e-3
```

**Testing phase for new task:**

Given initial observation o_0 and goal image G:
1. Encode o_0 -> z_current = ResNet50(o_0)
2. Encode G -> z_goal = ResNet50(G)
3. Repeat until goal reached:
   a_pred = inverse_model(concat(z_current, z_goal))
   Execute a_pred, get next observation o_next
   z_current = ResNet50(o_next)

**Why this design:** We chose CPC over autoencoders because CPC explicitly maximizes mutual information between context and future observations, producing representations that capture predictable dynamics rather than static reconstructions. We chose a temporal transformer over LSTM for its ability to capture long-range dependencies and efficient parallel training, accepting a higher parameter count. We train inverse dynamics separately from CPC rather than jointly to prevent the CPC representation from overfitting to action prediction and losing temporal generality; this trade-off means the inverse model must be trained with action labels, but these are naturally available in robot demos.

**Why it measures what we claim:** The CPC contrastive loss *measures* mutual information between context and future observation representations, which *operationalizes* 'task-agnostic dynamics representation' under the assumption that higher mutual information correlates with richer predictive features applicable across tasks; this assumption fails when visual prediction dominates actionable dynamics (e.g., pushing vs. grasping look similar but require different forces), in which case the metric reflects visual predictability rather than action relevance. The inverse model mapping (z_t, z_goal) -> action *measures* the ability to generate goal-directed actions under the assumption that the CPC representation is a sufficient statistic for Markovian dynamics; this fails when dynamics involve unobserved state elements (e.g., object slip), in which case the mapping learns spurious correlations. We accept these bounded failures. **Load-bearing assumption**: Unconditional CPC (without action conditioning) yields a representation that is a sufficient statistic for Markovian dynamics and enables zero-shot goal-directed action via a separate inverse model. **Verification**: We will test this assumption by measuring whether CPC representations cluster by dynamics type (e.g., pushing vs. grasping) rather than task identity, using mutual information between representation and dynamics type on a held-out set of 50 trajectories. If the clustering is poor, the assumption is violated.

**Training compute:** 4x A100 GPUs, 100 hours.

**Hyperparameter tuning:** We tune learning rate {1e-4, 3e-4, 1e-3}, batch size {16, 32, 64}, and lambda {0.01, 0.1, 1.0} on a held-out validation set of 10 LIBERO tasks.

## Contribution

(1) A novel framework FRAME that combines contrastive predictive coding on unlabeled video demonstrations with inverse dynamics to enable zero-shot action generalization without fine-tuning. (2) A design principle showing that temporally predictive representations learned via CPC can serve as task-agnostic action priors for vision-language-action models. (3) An analysis of the conditions under which this representation suffices for action generation, highlighting the assumptions about visual observability and Markov dynamics.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
|------|--------|-----------|
| Dataset | LIBERO simulation benchmark (4 suites: LIBERO-10, LIBERO-90, LIBERO-100, LIBERO-Long) | Standard, diverse manipulation tasks including pushing, picking, inserting, sliding; varying horizon and object variety |
| Primary metric | Success rate across tasks | Directly measures goal achievement |
| Baseline 1 | Goal-Conditioned BC (ResNet+LSTM, 2-layer LSTM hidden=512) | Tests CPC benefit over naive temporal model |
| Baseline 2 | OpenVLA (fine-tuned on LIBERO-10 for 10k steps) | Strong VLA baseline with language input |
| Baseline 3 | TinyVLA (trained from scratch on LIBERO-10 for 50k steps) | Compact VLA, tests against efficiency |
| Ablation 1 | FRAME-AE (autoencoder instead of CPC, reconstruction loss MSE, same encoder/transformer) | Isolates CPC contribution |
| Ablation 2 | FRAME w/ CPC+action (joint training: add auxiliary action prediction loss to CPC loss with weight 0.1) | Tests whether action-conditional CPC improves dynamics representation; note this changes training procedure |
| Training compute | 4x A100 GPUs, 100 hours per run | Ensures reproducibility |
| Hyperparameter tuning | Grid search over lr {1e-4, 3e-4, 1e-3}, batch size {16,32,64}, lambda {0.01,0.1,1.0} on 10 held-out LIBERO tasks | Reports best hyperparameters |

### Why this setup validates the claim

The combination of LIBERO's diverse tasks (pushing, picking, inserting, sliding) allows testing spatial reasoning across different dynamics and required skills. Comparing to Goal-Conditioned BC isolates the benefit of CPC's mutual information maximization for long-horizon tasks. Comparing to VLAs (OpenVLA, TinyVLA) tests whether visual-only representations can compete with language-augmented models on spatial reasoning. The ablation FRAME-AE directly tests the hypothesis that CPC produces more transferable dynamics features than reconstruction. Success rate is the appropriate metric because it directly evaluates whether the generated actions achieve the goal, which is the central claim of FRAME. Additionally, we include a verification experiment: we compute mutual information between CPC representations and dynamics type (e.g., push vs. grasp) vs. task identity on a held-out set of 50 trajectories per dynamics type; if MI(dynamics) > MI(task), the assumption is supported.

### Expected outcome and causal chain

**vs. Goal-Conditioned BC** — On a multi-step task like "pick and place with obstacle", the LSTM baseline may lose context over long horizons (e.g., forgetting the obstacle location after first step), leading to premature grasping or dropping. Our method's temporal transformer retains global dependencies, ensuring smooth, coherent actions. We expect a noticeable gap (e.g., 20-30% higher success) on tasks with more than 4 steps, but parity on short tasks.

**vs. OpenVLA** — On tasks with unseen object colors or shapes, OpenVLA's language grounding may misinterpret instructions (e.g., "blue cube" but cube is white), causing incorrect initial actions. FRAME's visual-only CPC representation focuses on motion patterns, not semantic labels, so it generalizes better to novel appearances. We expect FRAME to match OpenVLA on seen objects but surpass it by ~15% on unseen object variants.

**vs. TinyVLA** — On tasks requiring precise spatial alignment (e.g., peg insertion), TinyVLA's compressed VLM may lack fine-grained spatial understanding due to limited parameters, resulting in frequent alignment failures. FRAME's dedicated temporal encoder captures pixel-level motion cues, enabling precise actions. We expect FRAME to achieve ~10% higher success on precision tasks, with parity on coarse tasks.

**vs. FRAME-AE** — On tasks where visual background changes, autoencoder may overfit to static features, while CPC focuses on predictive dynamics. We expect FRAME to outperform by ~15% on tasks with distracting background changes.

### What would falsify this idea

If FRAME's success gain is uniform across all task subsets rather than concentrated on long-horizon or novel-object tasks, then the improvement is likely due to increased model capacity rather than CPC's theoretical benefit, contradicting the central claim that mutual information maximization yields superior dynamics representations. Additionally, if the verification experiment shows that MI(task) > MI(dynamics), the assumption of task-agnostic dynamics is violated.

## References

1. Fine-Tuning Vision-Language-Action Models: Optimizing Speed and Success
2. TinyVLA: Toward Fast, Data-Efficient Vision-Language-Action Models for Robotic Manipulation
3. π0: A Vision-Language-Action Flow Model for General Robot Control
4. Visual Instruction Tuning
5. MiniGPT-v2: large language model as a unified interface for vision-language multi-task learning
6. Scaling Instruction-Finetuned Language Models
7. Self-Instruct: Aligning Language Models with Self-Generated Instructions
8. Unnatural Instructions: Tuning Language Models with (Almost) No Human Labor
