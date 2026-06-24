# SCoRe: Self-Consistency-guided Correctness Representation for Step-Level Self-Distillation

## Motivation

Existing step-level error-guided self-distillation methods like TAPO require at least one correct rollout per query, identified via an external correctness signal (e.g., ground-truth answer). This dependency limits applicability to tasks without reliable evaluation or oracle knowledge. The root cause is the lack of a self-supervised signal to identify correct reasoning steps without supervision. We propose to leverage the model's own output consistency across multiple rollouts to learn a step-level correctness classifier, eliminating the need for any external signal.

## Key Insight

When multiple rollouts of the same query yield consistent final answers, the steps preceding that consensus are more likely to be correct, enabling a contrastive self-supervised signal to learn step-level correctness without labels.

## Method

**SCoRe: Self-Consistency-guided Correctness Representation**

(A) **What it is:** SCoRe is a self-supervised method that trains a step-level correctness classifier purely from the model's own generations, using the consistency of final answers among multiple rollouts as pseudo-labels. The classifier then provides the correctness signal needed to construct micro-reflective trajectories for self-distillation (as in TAPO).

(B) **How it works:**
```
For each query q:
  1. Generate N rollouts (N=10) using current policy π_θ.
  2. Group rollouts by their final answer; let group G be the largest group (majority).
  3. Extract the hidden state h_{i, t} from the last layer at the end of each step t in each rollout i. (Steps are delimited by newlines or 'Step k:' tokens.)
  4. Construct contrastive pairs:
     - Positive: (h_i,t, h_j,t') where both rollouts i,j belong to the same final-answer group (i.e., majority).
     - Negative: (h_i,t, h_k,t') where rollout k belongs to a different group (minority).
  5. Train a small MLP classifier C (2-layer, 256 hidden) with a contrastive loss (e.g., NT-Xent) on these pairs. The classifier outputs a scalar score s ∈ [0,1] indicating step-level correctness.
  6. For each rollout, compute average step score S_i = mean_t C(h_i,t).
  7. Select the rollout with highest S_i as the "correct rollout" for that query.
  8. Construct micro-reflective trajectories following TAPO: for each incorrect rollout j, find the first step τ where its output diverges from the selected correct rollout, and build a trajectory that keeps steps 1..τ-1, inserts a diagnostic sentence (e.g., "Wait, this step seems wrong. Let me correct..."), and appends the correct steps from τ onward.
  9. Fine-tune π_θ on these trajectories using KL divergence minimization (self-distillation).
```
Hyperparameters: N=10, contrastive temperature τ=0.07, MLP learning rate 1e-4, batch size 64 over steps, frozen backbone.

(C) **Why this design:** We chose contrastive learning over direct binary classification (e.g., using majority vote as label) because contrastive learning learns representations that separate consistent from inconsistent steps without requiring a hard decision boundary, which is more robust to noise when the majority group is small. We chose to freeze the backbone during contrastive training to prevent representation collapse and preserve the model's pretrained knowledge, accepting that the classifier may be less expressive. We chose N=10 as a trade-off between capturing sufficient diversity and computational cost; fewer rollouts may yield unreliable majority groups, while more increase inference time. Finally, we opted for a simple MLP rather than a trained reward model because the MLP is lightweight and can be trained on the fly, avoiding the overhead of separately training a large model.

(D) **Why it measures what we claim:** The average step score S_i, computed from the classifier's output, measures the degree to which a rollout's steps are consistent with the majority final answer, which we claim is a proxy for correctness. This operationalizes the concept of "correctness" under the assumption that majority consistency correlates with ground-truth correctness; this assumption fails when the model is systematically biased toward a wrong answer (e.g., all rollouts converge to the same incorrect final answer), in which case S_i instead measures mere consensus, not truth. The contrastive objective pulls together representations from steps that lead to the same final answer, thereby measuring "causal contribution to consistency" — i.e., how much a step's hidden state encodes the property of being part of a path that yields a particular consensus outcome. The final selection of the rollout with highest S_i assumes that the classifier's scores generalize across steps; this fails if the classifier overfits to spurious patterns (e.g., position-based cues), in which case S_i might select a rollout by chance.

## Contribution

(1) A self-supervised method to learn step-level correctness from unlabeled model generations using contrastive learning on final-answer consistency, eliminating the need for external correctness signals. (2) An extension of the TAPO framework that makes step-level error-guided self-distillation fully self-contained, applicable to tasks without reliable evaluation. (3) A demonstration that the learned correctness classifier can reliably identify correct rollouts, enabling construction of micro-reflective trajectories without any labeled data.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
|------|--------|-----------|
| Dataset | GSM8K | Tests multi-step reasoning |
| Primary metric | Accuracy | Direct correctness indicator |
| Baseline 1 | Vanilla self-distillation | Tests benefit of micro-reflective trajectories |
| Baseline 2 | TAPO with random selection | Tests value of SCoRe selection |
| Baseline 3 | ReST | Tests against RL-based self-improvement |
| Ablation | SCoRe with direct classifier | Tests contrastive vs. supervised |

### Why this setup validates the claim

The combination of a reasoning benchmark (GSM8K) and accuracy as the primary metric directly tests the core claim: selecting rollouts with high step-level consistency (via contrastive learning) improves self-distillation. Vanilla self-distillation isolates whether micro-reflective trajectories help at all. TAPO with random selection controls for trajectory construction, isolating the selection mechanism. ReST provides a strong RL-based baseline using self-generated data with outcome rewards. The ablation (direct classifier) tests if contrastive learning is superior to simpler supervised classification. If SCoRe outperforms these baselines, it validates that the contrastive correctness representation effectively identifies correct steps for self-distillation.

### Expected outcome and causal chain

**vs. Vanilla self-distillation** — On a problem where the model consistently makes the same error across all rollouts (e.g., misinterpreting a quantitative relationship), vanilla self-distillation reinforces that error because it treats all generations equally without correction. Our method instead uses consistency across rollouts to detect that the majority answer is likely correct, and selects a rollout leading to that answer. Since the majority is often correct, we expect SCoRe to show a significant accuracy gain (e.g., 5–10% absolute) on problems where the model is confidently wrong.

**vs. TAPO with random selection** — On a problem where only a few rollouts are correct, random selection often picks an incorrect rollout, constructing a flawed micro-reflective trajectory that teaches the model to "correct" from a correct step to an incorrect one. Our SCoRe selection picks the rollout with highest average step score, which correlates with correctness, leading to better trajectory quality. We expect SCoRe to outperform random selection by a noticeable margin (e.g., 3–5% accuracy) on problems with high variance in rollout quality.

**vs. ReST** — ReST uses offline RL with outcome rewards (e.g., binary correctness). On problems where sparse rewards are insufficient to pinpoint step-level errors (e.g., long chains with only final correctness), ReST may plateau, correcting only terminal steps. Our method provides dense step-level supervision, enabling finer-grained corrections. We expect SCoRe to achieve better accuracy on long reasoning chains where intermediate steps matter.

### What would falsify this idea

If SCoRe's gains are uniform across all problem difficulties rather than concentrated on problems where majority consistency is a reliable signal (i.e., where the model is not systematically biased toward a wrong answer), the core claim that consistency measures correctness would be unsupported. For instance, if accuracy improves equally on problems where the majority is correct and where it is wrong, that would indicate the selection mechanism is not actually discerning correctness.

## References

1. Learning from Your Own Mistakes: Constructing Learnable Micro-Reflective Trajectories for Self-Distillation
2. On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes
3. Reinforced Self-Training (ReST) for Language Modeling
4. Scaling Instruction-Finetuned Language Models
5. Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback
