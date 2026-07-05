# Causal Frame Ordering via Inverse Dynamics for Sparse Keyframe VideoQA

## Motivation

Existing inverse dynamics models for agentic tasks (e.g., Watch&Learn) exploit temporal causality from continuous frame sequences, but in VideoQA, keyframes are sparse and unordered, breaking this assumption. TASKER extracts informative keyframes but does not recover temporal order. We address this by learning to infer temporal order from visual causality cues alone, enabling inverse dynamics to operate on unordered keyframes.

## Key Insight

Visual causality cues (object state changes, scene dynamics) are temporally consistent across frame densities, allowing an inverse dynamics model to infer correct order from sparse keyframes without explicit temporal supervision.

## Method

## Causal Frame Ordering via Inverse Dynamics for Sparse Keyframe VideoQA

### A) What it is
CFO-ID takes an unordered set of keyframes (e.g., from TASKER) and outputs an ordered action sequence with segmentation. It combines a differentiable permutation network with an inverse dynamics module trained on source-domain continuous frames.

### B) How it works

**Load-bearing assumption:** The entropy (or contrastive confidence) of inverse dynamics predictions (trained on dense continuous frames) monotonically decreases with the temporal correctness of frame ordering for sparse keyframes. This assumption is calibrated via a contrastive loss that compares forward and reversed ordering confidence.

```python
# Pseudocode for CFO-ID training on sparse keyframes

def train_step(K, inv_dyn_model, ordering_net, trans_head):
    # K: set of unordered keyframes (N frames, each as image)
    # 1. Encode frames
    F = vit_encoder(K.images) # shape: (N, d), ViT-L/14, d=768
    
    # 2. Ordering network outputs permutation matrix P via Gumbel-Sinkhorn
    #    hyperparams: tau=1.0, n_iter=10, Sinkhorn algorithm with 10 iterations
    P = ordering_net(F) # shape: (N, N), differentiable
    
    # 3. Apply permutation to get ordered sequence
    F_ordered = P @ F
    
    # 4. Form consecutive pairs and predict transitions
    pairs = [(F_ordered[i], F_ordered[i+1]) for i in range(N-1)]
    trans_logits = trans_head(pair_features) # shape: (N-1, 2), 2-layer MLP hidden=256, GeLU
    trans_mask = torch.argmax(trans_logits, dim=-1) # binary: 1=action transition
    
    # 5. Inverse dynamics predicts action for transition pairs
    actions = []
    for i in range(N-1):
        if trans_mask[i]:
            act = inv_dyn_model(pairs[i]) # softmax over K action classes, K=80 for MSRVTT
            actions.append(act)
    
    # 6. Loss computation
    #    - Ordering consistency: contrastive loss comparing forward vs reversed pair confidence
    contrastive_loss = 0.0
    for i in range(N-1):
        if trans_mask[i]:
            ordered_pair = pairs[i]
            reversed_pair = (F_ordered[i+1], F_ordered[i])
            fwd_conf = torch.max(torch.softmax(inv_dyn_model(ordered_pair)/T, dim=-1))
            rev_conf = torch.max(torch.softmax(inv_dyn_model(reversed_pair)/T, dim=-1))
            contrastive_loss += -torch.log(fwd_conf / (fwd_conf + rev_conf + 1e-8))
    #    - Transition sparsity: encourage few transitions (L1 on trans_logits)
    sparsity_loss = torch.sum(torch.abs(trans_logits[:,1])) / N
    #    - Total loss
    loss = contrastive_loss + 0.1 * sparsity_loss
    loss.backward()
    optimizer.step()
    
    # Inference: hard permutation (argmax of P) and hard transitions (argmax of trans_logits)
    # Segment sequence: consecutive transition pairs form an action segment
```

**Hyperparameters:** Gumbel-Sinkhorn temperature tau=1.0, Sinkhorn iterations = 10, contrastive temperature T = 1.0, contrastive loss weight = 1.0 (combined with sparsity weight 0.1).

### C) Why this design
We chose a differentiable permutation via Gumbel-Sinkhorn over greedy sorting or RL because it allows end-to-end gradient flow through the ordering decision, avoiding high-variance policy gradients (trade-off: Gumbel-Sinkhorn is approximate and blurry at low temperatures). We use a separate transition head rather than assuming every consecutive pair is an action step, because sparse keyframes may capture within-action states; this adds a binary decision that prevents over-segmentation (trade-off: the transition head may misclassify, but we regularize with sparsity). We pretrain inv_dyn_model on source-domain continuous frames using cross-entropy on action labels, then freeze it during target training, because the action space is shared and we want the ordering network to exploit the already-learned causal mapping (trade-off: frozen inv_dyn may be suboptimal for target domain, but fine-tuning could overfit to sparse data). Unlike Watch&Learn which relies on temporal adjacency, our method removes that dependency by learning order from causality, enabling application to VideoQA where keyframes are given in arbitrary order.

### D) Why it measures what we claim
The computational quantity `contrastive_loss` measures **temporal consistency** because low contrastive loss implies that the inverse dynamics model confidently predicts a single action given the ordered frame pair compared to the reversed pair; this assumes that a causally consistent ordering yields high confidence while reversed order yields low confidence. This assumption fails when multiple actions share similar visual changes (e.g., both 'pouring' and 'stirring' involve hand motion), in which case contrastive loss reflects ambiguity rather than order correctness. The quantity `sparsity_loss` measures **segmentation parsimony** because it penalizes excessive transition predictions, assuming that true actions are relatively few compared to within-action states; this assumption fails in rapidly changing scenes where many quick actions occur. The permutation matrix `P` operationalizes **ordering recovery** by assigning each frame a position; it is trained to make the inverse dynamics outputs confident. The transition mask `trans_mask` operationalizes **action boundaries** by converting consecutive pairs into segments; it is regularized to produce sparse yet continuous groupings. The load-bearing assumption (entropy/contrastive confidence decreases with order correctness) is explicitly calibrated via the contrastive loss formulation; failure occurs when multiple actions share similar visual dynamics, causing confident but wrong predictions.

## Contribution

(1) CFO-ID, a framework that recovers temporal order and action segmentation from unordered keyframes by training a differentiable permutation network via inverse dynamics consistency. (2) The finding that visual causality cues alone are sufficient to infer temporal order from sparse keyframes, enabling inverse dynamics models to transfer from agentic tasks to VideoQA without temporal supervision. (3) A self-supervised training paradigm combining source-domain inverse dynamics pretraining with target-domain ordering entropy minimization.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|-----------------------|
| Dataset | MSRVTT-QA, Next-QA, TGIF-QA | Standard video QA with temporal reasoning; multiple datasets for generalization. |
| Synthetic dataset | SyntheticOrder | Ground-truth order and actions to directly measure ordering accuracy. |
| Primary metric | Top-1 accuracy, Kendall's tau (for synthetic) | Top-1 for QA; Kendall's tau measures ordering quality. |
| Baseline 1 | Random order ordering | Tests if any ordering helps. |
| Baseline 2 | Greedy visual similarity order | Heuristic baseline using frame similarity. |
| Baseline 3 | Learned ordering from video prediction (VideoDirector) | External learned ordering method from other domain. |
| Ablation-of-ours | Without transition head | Tests sparsity regularization effect. |

### Why this setup validates the claim
This setup forms a falsifiable test because it isolates the core claim: learning order from causality improves VideoQA. Using standard datasets (MSRVTT-QA, Next-QA, TGIF-QA) with keyframe extraction from TASKER ensures realistic evaluation. The synthetic dataset provides ground-truth order to directly measure ordering accuracy and its correlation with contrastive loss, addressing the load-bearing assumption. Random order baseline tests whether any ordering matters; greedy similarity tests whether simple heuristics suffice; VideoDirector baseline tests if our learned ordering is competitive with cross-domain methods. The ablation of the transition head tests whether sparsity regularization is necessary. Top-1 accuracy directly measures downstream impact of ordering on question answering. If our method fails to outperform these baselines or fails on synthetic order recovery, the central hypothesis—that inverse-dynamics-guided ordering benefits VideoQA—is invalidated.

### Expected outcome and causal chain

**vs. Random order ordering** — On a video where action sequence is crucial (e.g., pouring then stirring), random order may place stirring before pouring, causing the inverse dynamics model to predict an implausible action transition (e.g., stirring to pouring, which is rare). Random handles this by ignoring order; our method learns to order frames so that contrastive loss is minimized (forward confidence high, reversed low). Thus we expect a clear gap (e.g., +10% accuracy) on videos with distinct actions, but parity on static videos.

**vs. Greedy visual similarity order** — On a video where successive frames look similar but actions differ (e.g., stirring followed by shaking, both hand motions), greedy similarity may incorrectly group them as same or order wrongly due to visual similarity. Our method uses inverse dynamics to detect action boundaries, leading to correct ordering even when visual similarity is high. We expect our method to outperform greedy by a noticeable margin (e.g., +5%) specifically on such ambiguous transitions.

**vs. Learned ordering from video prediction (VideoDirector)** — On synthetic data with ground-truth order, VideoDirector may overfit to continuous temporal dynamics, while our method trained on sparse keyframes with contrastive loss should recover order more accurately. We expect Kendall's tau at least 0.7 for our method vs. 0.5 for VideoDirector.

**vs. Without transition head** — On a video with long within-action sequences (e.g., pouring water continuously), the ablation predicts a transition at every consecutive pair, leading to over-segmentation and multiple action predictions for a single true action. Our method uses sparsity regularization to suppress false transitions. We expect our method to show higher accuracy on videos with extended repetitive actions, with a drop of at least 3-5% in the ablation condition.

### What would falsify this idea
If our method achieves no significant improvement over random ordering or greedy similarity, or if the gain is uniform across all subsets rather than concentrated on videos where temporal order is critical, or if on synthetic data the contrastive loss does not correlate with order accuracy, then the central claim that causal order recovery via inverse dynamics enhances video QA would be false.

## References

1. Bridging VideoQA and Video-Guided Agentic Tasks via Generalized Keyframe Extraction
2. Tool-Augmented Spatiotemporal Reasoning for Streamlining Video Question Answering Task
3. Scalable Video-to-Dataset Generation for Cross-Platform Mobile Agents
4. VideoChat-A1: Thinking with Long Videos by Chain-of-Shot Reasoning
5. Watch and Learn: Learning to Use Computers from Online Videos
6. Computer-Use Agents as Judges for Generative User Interface
7. AgentTrek: Agent Trajectory Synthesis via Guiding Replay with Web Tutorials
8. AMEX: Android Multi-annotation Expo Dataset for Mobile GUI Agents
