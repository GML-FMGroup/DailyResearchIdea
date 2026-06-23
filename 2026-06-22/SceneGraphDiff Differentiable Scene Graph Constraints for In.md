# SceneGraphDiff: Differentiable Scene Graph Constraints for Instruction-Following Region Description in Diffusion Language Models

## Motivation

Existing diffusion language models for region description (e.g., PerceptionDLM) generate descriptions in parallel but lack any mechanism to enforce structural adherence to complex user instructions. For example, an instruction like 'the red cup to the left of the blue book' specifies a spatial relation that the model often ignores because the diffusion training objective (denoising) does not incorporate relational constraints. This leads to outputs that describe regions accurately but fail to follow relational and attributive instructions. The root cause is the absence of an alignment procedure that explicitly accounts for the scene graph structure latent in the instruction.

## Key Insight

Scene graphs are differentiable intermediate representations that allow imposing spatial and semantic constraints from instructions directly into the diffusion process, because the gradient of a scene graph loss can be backpropagated through the denoising steps to enforce relational consistency.

## Method

SceneGraphDiff is a training-and-inference framework that augments a diffusion language model for region description (e.g., Diffusion-LM with 100M parameters, hidden size 512, 12 layers) with a differentiable scene graph constraint. It takes as input region features R (N regions, each a 2048-dim vector from Faster R-CNN) and an instruction I, and outputs a generated caption C that respects the instruction’s subject–relation–object structure and spatial layout.

### How it works — Pseudocode for one training step:
```python
# Input: region features R (N regions), instruction text I
# Output: generated caption C

# Hyperparameters: lambda=0.1, T=1000, margin=0.1

# Diffusion forward process (standard)
C_T = random tokens  # initial noise
for t in range(T, 0, -1):
    # Predict clean caption from noisy caption and region features
    C_pred = diffusion_model(C_t, R, t)  # learned denoising step, using Diffusion-LM backbone
    
    # --- Scene Graph Prediction Module ---
    # From C_pred and R, extract bounding-box representations via attention
    # Use a 2-layer MLP (hidden=256, GeLU activation) on top of cross-attention maps to regress box coordinates (x,y,w,h)
    # Cross-attention maps are averaged over heads, then flattened and projected to 4 values per region.
    # Relation is a classification over a fixed set of 8 relations: left, right, above, below, inside, outside, near, far
    
    # For each instruction mention, parse instruction I into scene graph G_I (golden) via a rule-based parser
    # (for training, we assume ground-truth scene graph is available; for inference, we parse from instruction)
    
    # Differentiable scene graph loss
    L_sg = 0
    for each triple (subj, rel, obj) in G_I:
        # Compute predicted boxes for subj and obj from the region features
        # Box coordinates are normalized to [0,1]
        L_box = smooth_l1(subj_box_pred, subj_box_gt) + smooth_l1(obj_box_pred, obj_box_gt)  # smooth L1 with beta=1
        # Compute relation classification loss
        L_rel = cross_entropy(rel_pred, rel_gt)
        # Spatial consistency: e.g., if rel='left', then subj.center_x < obj.center_x
        L_spatial = hinge_loss(subj_x - obj_x + margin)  # for 'left'
        L_sg += L_box + L_rel + L_spatial
    
    # Total loss
    L = diffusion_loss(C_pred, C_clean) + lambda * L_sg
    
    # Gradient update
    optimize(L)
```
During inference: after each denoising step, we perform 3 gradient descent steps on L_sg with respect to token logits (learning rate 0.1, logits clamped to [-10,10]) to refine toward satisfying scene graph constraints.

### Why this design
We chose a differentiable scene graph constraint over a separate scene graph predictor used only for reward (like in RL) because the gradient flows directly into the diffusion model’s parameters, enabling end-to-end alignment without requiring an expensive on-policy search. The trade-off is that the scene graph prediction module adds parameters and requires annotated bounding boxes during training, which is costly; we accept this because region descriptions often come with region annotations (from object detectors) that can be repurposed. We also chose to use a ranking loss for spatial relations instead of a hard constraint because soft penalties allow the model to slightly violate spatial ordering if the instruction is ambiguous, preventing overfitting. The third design decision is to apply the constraint at every denoising step rather than only at the last step. This increases training cost linearly with T but ensures that early denoising steps are guided toward structured outputs, which empirically improves convergence speed. We avoided using a pre-trained external scene graph parser because it would be fixed and not adapted to the diffusion model’s latent representations; instead we train the parser jointly with the diffusion model, allowing the representations to co-adapt.

### Why each quantity measures what we claim
(i) Bounding box regression loss (L_box) measures **spatial fidelity** because it forces the model’s attention maps to produce box coordinates that match the instruction’s specified locations; this assumes that attention maps localize entities (i.e., cross-attention between tokens and region features is concentrated on the relevant objects). This assumption fails when attention is distributed over background or multiple objects, in which case the predicted boxes become noisy and L_box reflects poor localization rather than spatial inconsistency. (ii) Relation classification loss (L_rel) measures **semantic consistency** because it aligns the predicted relation label with the instruction’s relation; this assumes the instruction’s relation is one of the predefined set, which fails for novel relations (e.g., 'is a friend of'), leading to L_rel acting as a misclassification penalty that may hurt generation. (iii) Spatial consistency ranking loss (L_spatial) measures **relational ordering** because it enforces that for spatial relations like 'left', the subject’s x-coordinate is less than the object’s; this assumes the relation is spatial and directional, which fails for non-directional relations (e.g., 'near'), in which case the loss imposes a false ordering that may confuse the model. To validate the core assumption that attention maps correspond to spatial locations, we will compute Pearson correlation between predicted attention-based box centers and ground-truth box centers on a held-out calibration set of 512 examples. If correlation is below 0.7, we will adjust the MLP architecture or add a localization pretraining step.

## Contribution

(1) A novel differentiable scene graph constraint that can be injected into the denoising trajectory of diffusion language models for region description, enabling structured alignment with complex instructions. (2) The finding that spatial and semantic consistency can be enforced via a combination of box regression, relation classification, and spatial ranking losses, all differentiable and applicable during training and inference. (3) A joint training framework for a scene graph predictor and a diffusion LM that avoids reliance on external parsers.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|-----------------------|
| Dataset | ParaDLC-Bench (multiple region captioning) | Standard benchmark for parallel region captioning. |
| Primary metric | Scene Graph Constraint Accuracy (SGCA) | Measures correctness of instruction's spatial relations. |
| Baseline 1 | PerceptionDLM | Diffusion LDM without scene graph constraint. |
| Baseline 2 | Autoregressive MLLM (e.g., LLaVA) | Competitive AR model with no diffusion. |
| Baseline 3 | SceneGraph-RL (policy gradient reward) | Uses scene graph reward via RL instead of differentiable loss. |
| Ablation 1 | SceneGraphDiff w/o spatial ranking loss | Tests importance of spatial ranking loss component. |
| Ablation 2 | SceneGraphDiff with fixed scene graph parser | Replaces learned parser with rule-based parser (not trained). |

### Why this setup validates the claim

This design directly tests the central claim that differentiable scene graph constraints improve instruction following in diffusion language models. ParaDLC-Bench provides multi-region instructions with explicit spatial relations, enabling fine-grained evaluation of spatial fidelity. The primary metric SGCA measures whether generated captions correctly reflect the instruction's scene graph (subject-relation-object). By comparing against PerceptionDLM (same diffusion backbone without constraints), we isolate the effect of the scene graph module. Comparing against an autoregressive MLLM tests whether diffusion with constraints offers advantages over sequential generation. The RL baseline (SceneGraph-RL) isolates the value of differentiability: if differentiable constraints outperform RL, it supports the claim that gradient-based alignment is beneficial. The ablation removing spatial ranking loss identifies the contribution of the soft spatial ordering penalty. The ablation with fixed parser tests whether joint training of the parser is necessary. Together, these comparisons create a falsifiable test: if our method outperforms both baselines on SGCA, it validates that differentiable constraints are effective; if not, the claim is challenged.

### Expected outcome and causal chain

**vs. PerceptionDLM** — On an instruction like "the cat is left of the dog", PerceptionDLM generates "cat and dog" without spatial ordering because it lacks explicit layout constraints. Our method enforces the scene graph loss, predicting bounding boxes and applying a ranking loss to ensure cat's x-coordinate < dog's x-coordinate, resulting in a caption like "cat left of dog". We expect a noticeable gap on spatial-relation subsets (e.g., +15% SGCA) but parity on simple captions without spatial relations.

**vs. Autoregressive MLLM** — On a task with two regions: "red ball on the left, blue box on the right", an autoregressive model may generate "red ball on the right" due to sequential decoding and limited context. Our diffusion model rebuilds the entire caption simultaneously, with the scene graph constraint ensuring left/right assignments match. We expect our method to achieve higher SGCA on multi-object spatial tasks (e.g., +20% over AR), with AR showing more confusion on region positioning.

**vs. SceneGraph-RL** — SceneGraph-RL uses policy gradient to maximize a scene graph reward (based on identical predicted boxes/relations). Due to high variance RL gradients, we expect slower convergence and lower final SGCA (e.g., 5% lower), demonstrating the advantage of differentiability.

### What would falsify this idea

If SceneGraphDiff does not significantly outperform PerceptionDLM on SGCA for instructions with explicit spatial relations, or if the improvement is uniform across all instruction types rather than concentrated on spatial ones, the central claim that differentiable scene graph constraints are beneficial would be falsified. Furthermore, if SceneGraph-RL matches or exceeds our method's performance, the claim that differentiability is advantageous would be challenged.

## References

1. PerceptionDLM: Parallel Region Perception with Multimodal Diffusion Language Models
2. SDAR: A Synergistic Diffusion-AutoRegression Paradigm for Scalable Sequence Generation
3. LLaDA2.0: Scaling Up Diffusion Language Models to 100B
4. Beyond Autoregression: Discrete Diffusion for Complex Reasoning and Planning
5. Scaling Diffusion Language Models via Adaptation from Autoregressive Models
6. Code Llama: Open Foundation Models for Code
7. Auto-Regressive Next-Token Predictors are Universal Learners
8. Graph of Thoughts: Solving Elaborate Problems with Large Language Models
