# Deterministic Multi-Modal Action Anchors for Fast VLA Inference

## Motivation

Existing deterministic VLA methods like OFT use L1 regression, which collapses to the mean and fails to capture multi-modal action distributions common in robotics (e.g., multiple valid grasps). In contrast, stochastic methods like TinyVLA's diffusion decoder generate diverse actions but require iterative sampling, incurring high latency. The structural root cause is that deterministic objectives cannot represent multiple modes, while iterative sampling is inherently sequential. Thus, a method that outputs multiple distinct actions in a single forward pass is needed.

## Key Insight

Training a predictor to output a fixed set of K action candidates with a repulsive diversity loss forces the candidates to span the true action modes, and a learned mode selector then identifies the correct mode, enabling deterministic multi-modal prediction.

## Method

### MADA (Multi-Action Deterministic Anchors)

(A) **What it is:** MADA is a deterministic VLA model that, given visual and language features, outputs K action anchors and a mode selector distribution over them in a single forward pass. The model is trained end-to-end with a reconstruction loss on the selected anchor and a repulsive loss to keep anchors distinct.

(B) **How it works:**
```python
# Input: observation o_t, language instruction l, ground truth action a_t
# Encoder (pretrained VLM backbone, e.g., OpenVLA's SigLIP ViT-L/14 + 7B LLM)
h = VLM_encoder(o_t, l)  # shared feature vector (dimensionality h_dim=4096)

# Anchor predictor: 2-layer MLP with GeLU activation, hidden_dim=256
anchor_mlp = MLP(input_dim=h_dim, hidden_dim=256, output_dim=K * action_dim)
anchors = anchor_mlp(h).reshape(K, action_dim)  # [K, D] (D=7 for LIBERO)

# Mode selector: 2-layer MLP with GeLU activation, hidden_dim=128
selector_mlp = MLP(input_dim=h_dim, hidden_dim=128, output_dim=K)
selector_logits = selector_mlp(h)  # [K]
selector_probs = softmax(selector_logits, dim=-1)

# Loss computation
# Reconstruction loss: expected L1 distance under selector distribution
L_rec = sum(selector_probs * L1_loss(anchors, a_t.unsqueeze(0)))  # broadcast
# Repulsive loss: Gaussian kernel on pairwise anchor distances
pairwise_dists = pairwise_cosine_distances(anchors)  # [K, K]
L_rep = (1/(K*(K-1))) * sum(exp(-pairwise_dists.unsqueeze(0) / tau))  # exclude diag
# Total loss
L_total = L_rec + lambda * L_rep
# Hyperparameters: K=5, lambda=0.1, tau=1.0

# Training: AdamW optimizer, lr=1e-4, batch_size=64, 100k steps, 4×A100 GPUs (~12 hours)

# Inference: deterministic selection
action = anchors[argmax(selector_logits)]
```

(C) **Why this design:** We chose softmax-weighted L1 loss over hard assignment (e.g., only update the closest anchor) because it provides a differentiable training signal that trains the selector to weight the most relevant anchor, while allowing multiple anchors to be updated; the cost is that the selector may spread probability across anchors if multiple fit equally well, but the repulsive loss mitigates this. We chose a Gaussian kernel repulsive loss (over contrastive or margin-based losses) because it smoothly penalizes near pairs while ignoring far pairs, avoiding forced separation beyond the kernel width; the cost is sensitivity to the kernel width tau, which requires tuning. We chose a separate selector MLP (rather than using distances to anchors) because the selector can leverage contextual features (e.g., task progression) to disambiguate modes, whereas geometric distance alone may be insufficient; the cost is increased parameters and potential overfitting with limited data.

(D) **Why it measures what we claim:** The reconstruction loss (softmax-weighted L1) measures the model's ability to place at least one anchor near the ground truth action; specifically, the selector's softmax distribution weighted by L1 loss quantifies the expected distance to the true action under the assumption that the selector can identify the correct mode. This assumption fails when the true action is equidistant from multiple anchors (e.g., two symmetric grasps), in which case the loss reflects a blend of distances rather than a single mode, but the repulsive loss reduces this risk by enforcing separation. The repulsive loss (Gaussian kernel over anchor pairs) measures anchor distinctness by counting nearby pairs weighted by distance; it operationalizes the concept of 'mode coverage' under the assumption that diverse true modes are separated by at least the kernel width. This assumption fails when true modes are very close (e.g., fine-grained translations), in which case repulsion may reduce reconstruction accuracy. The selector logits measure contextual mode confidence; they operationalize 'mode identification' under the assumption that the observation contains discriminative cues. This assumption fails under severe partial observability, in which case the selector becomes random, and performance degenerates to random anchor choice. 

*Sub-block: Selector argmax measures mode identification.* At inference, taking argmax over selector logits operationalizes 'mode identification' under the assumption that the selector's confidence from training distribution matches inference. This assumption fails when the observation is ambiguous (e.g., symmetric environment) and the selector assigns near-uniform probability; then argmax yields a deterministic but potentially incorrect choice, reducing success rate. We diagnose this failure by monitoring selector entropy: if average entropy over a validation set exceeds a threshold (e.g., 0.7), we flag ambiguous inputs.

(E) **Load-bearing assumption:** The mode selector can reliably identify the correct action mode from the current observation and language instruction alone, despite possible partial observability. This assumption is load-bearing because if the observation is indistinguishable from multiple valid actions (e.g., symmetric grasps), the selector cannot disambiguate, leading to random choice and degraded performance. We verify this by evaluating on tasks with varying ambiguity (e.g., LIBERO tasks with symmetric vs. asymmetric initial states). If success drops significantly on symmetric tasks, the assumption is violated, and temporal context would be needed (future work).

## Contribution

(1) Introduces MADA, a deterministic VLA architecture that outputs multiple action anchors with a learned mode selector and repulsive diversity loss, enabling one-shot multi-modal action generation. (2) Provides a design principle that combining a reconstruction loss with a repulsive loss on anchor candidates effectively induces diversity without iterative sampling, establishing a trade-off between mode coverage and anchor distinctness.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
|------|--------|-----------|
| Dataset | LIBERO simulation benchmark | Standardized multi-task manipulation, includes tasks with varying multimodality |
| Primary metric | Task success rate | Direct measure of task completion |
| Baseline 1 | OpenVLA | State-of-the-art VLA model (single deterministic action) |
| Baseline 2 | TinyVLA | Efficient VLA variant (discretized action) |
| Baseline 3 | MiniGPT-v2 | VLM baseline without action decoder |
| Baseline 4 | Multiple Choice Learning (MCL) | Ensemble of K independent deterministic predictors trained with oracle assignment, representative of classic multi-hypothesis methods |
| Ablation 1 | MADA w/o repulsive loss | Isolates repulsive loss contribution |
| Ablation 2 | MADA with hard assignment loss (only update closest anchor) | Compare with softmax-weighted loss |
| Ablation 3 | Vary K in {2,5,10,20} | Evaluate speed-coverage trade-off |

**Additional specifications:** Training on 4 NVIDIA A100 GPUs, batch size 64, 100k steps (~12 hours). Action dimension D=7 for LIBERO (6-DoF pose + gripper). Evaluation on 50 episodes per task (500 tasks total).

### Why this setup validates the claim
This combination forms a falsifiable test: LIBERO contains tasks with varying degrees of multimodality (e.g., picking from different orientations). Comparing MADA to OpenVLA, which outputs a single continuous action, isolates the benefit of multi-anchor representation; a larger gain on multimodal tasks supports the mode coverage claim. TinyVLA provides a strong efficiency baseline; if MADA matches its speed while achieving higher success, it validates deterministic multi-anchor as efficient. MiniGPT-v2 tests whether a VLM lacking an action-specific head can be adapted; poor performance would justify the need for explicit action anchors. The ablation without repulsive loss tests whether anchor diversity is crucial; if it performs similarly, then repulsion is unnecessary. Task success rate is the appropriate metric because it directly reflects real-world utility and aggregates over multimodal decisions.

### Expected outcome and causal chain

**vs. OpenVLA** — On a task with two symmetric grasps (e.g., picking cup from left or right), OpenVLA averages them into a single action between the two, leading to poor execution because it cannot represent multimodality. MADA outputs separate anchors for left and right, and selects the appropriate one based on context, so we expect a noticeable success gap on tasks with multimodal actions (e.g., ~20% higher) but parity on deterministic tasks.

**vs. TinyVLA** — TinyVLA uses action discretization with limited bins, losing fine-grained precision on continuous actions. On tasks requiring precise placement (e.g., stacking), TinyVLA may fail due to quantization error. MADA outputs continuous anchors, preserving precision, so we expect MADA to outperform on fine-grained tasks by ~10-15% while matching speed.

**vs. MiniGPT-v2** — MiniGPT-v2 lacks a dedicated action decoder; if used to predict actions via language output, it struggles with fine-grained motor commands. On precise manipulation, its success rate will be low (<50%). MADA's explicit action anchors provide proper motor representation, so we expect MADA to significantly outperform (e.g., >80% vs <50%).

**vs. Multiple Choice Learning (MCL)** — MCL trains K independent predictors with oracle assignment (each sample assigned to best predictor). This encourages mode specialization but does not learn a selector; at inference, it requires a separate oracle (e.g., distance to demo) or random selection. MADA's learned selector provides context-conditioned mode identification, so we expect MADA to outperform MCL+random by ~10% on ambiguous tasks, while MCL+oracle (if available) provides an upper bound.

### What would falsify this idea
If MADA's success gain over OpenVLA is uniform across all LIBERO tasks rather than concentrated on tasks with clear multimodal action distributions, then the improvement is not due to mode coverage and the idea is falsified.

## References

1. Fine-Tuning Vision-Language-Action Models: Optimizing Speed and Success
2. TinyVLA: Toward Fast, Data-Efficient Vision-Language-Action Models for Robotic Manipulation
3. MiniGPT-v2: large language model as a unified interface for vision-language multi-task learning
4. EVA: Exploring the Limits of Masked Visual Representation Learning at Scale
