# Dynamical System Distillation: Preserving In-Context Learning via Layer-Wise Linear Isomorphism Alignment

## Motivation

Standard knowledge distillation for language models aligns teacher and student representations at specific layers or through prediction matching, but these approaches fail to transfer emergent abilities like in-context learning (ICL). The survey 'Knowledge distillation and dataset distillation of large language models' explicitly notes that preserving emergent LLM abilities is not fully addressed. The root cause is that these methods ignore the dynamical structure of how representations iteratively refine across layers—ICL emerges from the teacher's layer-wise transformation dynamics, not from static states. Aligning only states or outputs cannot guarantee that the student inherits the teacher's ability to update context representations in a way that enables ICL.

## Key Insight

The teacher's in-context learning ability is a property of its layer-wise transformation dynamics, and aligning those dynamics up to a learned linear isomorphism preserves the structural behavior that gives rise to ICL, because linear isomorphisms maintain the relative geometry of the state space while allowing for differences in basis.

## Method

### (A) What it is
**Dynamical System Distillation (DSD)** is a knowledge distillation method that aligns the teacher's and student's hidden state dynamical systems by learning a per-layer linear isomorphism that maps the student's hidden states to the teacher's, and then enforcing consistency between the layer-wise transformation functions under that map. Inputs: teacher and student language models (same architecture but different sizes), a corpus of natural language sequences. Output: a student model whose layer-wise dynamics approximate the teacher's, thereby transferring in-context learning ability.

### (B) How it works
```pseudocode
# Let T_l and S_l be the teacher and student hidden states at layer l (for a given input).
# Let f_T^l: T_l -> T_{l+1} be the teacher's transformation (layer l to l+1), e.g., a transformer block.
# Let f_S^l: S_l -> S_{l+1} be the corresponding student transformation.
# For each layer l, learn an affine linear map A_l: S_l -> T_l (parameterized by a matrix and bias).

Initialize student model weights (can be from scratch or from teacher via standard KD).

for each batch of sequences do:
    # Forward pass through both models to collect all hidden states
    Compute teacher hidden states {T_1, ..., T_L} and student hidden states {S_1, ..., S_L}
    # For each layer, compute the aligned student state and the teacher's state prediction
    for l = 1 to L-1 do:
        # Map student's current state to teacher's space
        aligned_S_l = A_l(S_l)
        # Map student's next state using teacher's transformation? No: we want to align the student's transformation to the teacher's.
        # Compute the teacher's predicted next state from the aligned student state
        predicted_T_{l+1} = f_T^l(aligned_S_l)  # note: f_T^l is frozen teacher
        # Compute the student's actual next state (in teacher's space via A_{l+1})
        actual_S_{l+1} = A_{l+1}(S_{l+1})
        # Distillation loss: minimize the difference between predicted and actual
        loss_l = MSE(predicted_T_{l+1}, actual_S_{l+1})
    end for
    # Also add a state-matching loss to help shape the linear maps? Optionally include a small L2 regularization on the linear maps.
    total_loss = sum(loss_l) + lambda * sum(||A_l||_F^2)   # lambda = 1e-4
    Update student model parameters (including S_l transformation f_S^l) and linear maps A_l via backprop.
end for
```
Hyperparameters: lambda = 1e-4 (regularization strength), learning rate = 1e-4, Adam optimizer.

### (C) Why this design
We chose to align transformation functions via a learned linear isomorphism rather than directly matching hidden states because: 1) **Linear isomorphism instead of identity mapping**: the student has fewer parameters and a different dimensionality, so forcing exact state match is too restrictive and hurts performance; the linear map allows the student to learn a compressed but structurally equivalent representation. The trade-off is that we introduce additional parameters (the A_l maps) which increase training cost. 2) **Aligning dynamics (f) instead of states**: standard intermediate matching only ensures that the student's states approximate the teacher's at fixed layers, but does not enforce that the student's transformation from one layer to the next behaves like the teacher's. By minimizing MSE between the teacher's prediction (from aligned student state) and the student's actual next aligned state, we force the student's transition function to approximate the teacher's. The trade-off is that this requires access to teacher's internal computations, which may be computationally expensive. 3) **Per-layer maps instead of a single global map**: ICL emerges from the entire sequence of layer-wise updates; a single map cannot capture layer-specific nonlinearities. Per-layer maps provide flexibility to correct mismatches at each depth, but increase memory and training time. 4) **Affine maps (linear + bias) instead of simple linear**: the bias accounts for constant offsets between teacher and student representations, improving alignment accuracy without overfitting. The trade-off is a slight increase in parameters.

### (D) Why it measures what we claim
*Computational quantity*: the aligned student state A_l(S_l) *measures* the student's internal representation in a space comparable to the teacher's, *because* we assume that a linear (affine) map can transform the student's representation space to match the teacher's up to an isomorphism; this assumption fails when the student's representational geometry is not linearly related to the teacher's (e.g., due to highly nonlinear compression), in which case A_l(S_l) reflects a distorted view of the student's state rather than a faithful alignment. *Computational quantity*: the difference between predicted_T_{l+1} (teacher's transformation on aligned student state) and actual_S_{l+1} (aligned student's actual next state) *measures* the discrepancy between the teacher's and student's layer-wise dynamics, *because* we assume that the teacher's transformation applied to the aligned student's current state should produce the same next state as the student's actual transformation, when both are expressed in the teacher's space; this assumption fails when the student's transformation is not approximately linear in the teacher's space (e.g., due to different activation patterns), in which case the difference reflects both dynamical mismatch and linearization error. The total loss across layers *measures* the degree to which the student's iterative processing mirrors the teacher's, *because* the sum over layers captures the cumulative effect of dynamics alignment; this assumption fails if the teacher's ICL ability relies on early-layer processing that is not captured by the loss because the student's early layers are poorly aligned (since maps are learned jointly), leading to a low total loss but poor ICL transfer.

## Contribution

['(1) A novel distillation method (DSD) that aligns the layer-wise transformation dynamics of teacher and student language models via a learned linear isomorphism per layer, going beyond static representation matching.', '(2) The key design principle that in-context learning ability can be transferred by preserving the structure of iterative refinement across layers, rather than aligning only outputs or intermediate states.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|-----------------------|
| Dataset | Few-shot NLU benchmark (10 tasks) | Tests in-context learning transfer. |
| Primary metric | Average few-shot accuracy | Directly measures ICL ability. |
| Baseline | Standard logit distillation | Common KD baseline, matches output distribution. |
| Baseline | Hidden state MSE distillation | Intermediate feature matching baseline. |
| Baseline | No distillation (from scratch) | Lower bound, shows absolute gain. |
| Ablation-of-ours | DSD with identity linear maps | Isolates benefit of learned isomorphism. |

### Why this setup validates the claim

This setup validates the claim that aligning the teacher's and student's hidden state dynamical systems transfers in-context learning ability. The few-shot NLU benchmark directly tests ICL across diverse tasks, and average accuracy captures overall transfer quality. By comparing against standard logit and hidden state distillation, we isolate the effect of dynamics alignment over mere output or state matching. The ablation (identity maps) tests whether learned linear isomorphism is essential. The metric is sensitive to the expected improvement: if DSD truly improves the iterative processing, it should show a systematic gain over baselines, especially on tasks requiring multi-step reasoning, where dynamics matter most. A failure to outperform on such tasks would falsify the claim.

### Expected outcome and causal chain

**vs. Standard logit distillation** — On a multi-step reasoning task (e.g., adding numbers in context), standard KD matches only final output probabilities, so the student may memorize answer patterns but fails on new prompts because it lacks iterative processing. Our method aligns layer-wise dynamics, so the student learns to transform context step-by-step like the teacher, enabling generalization. We expect DSD to show a noticeable gap (e.g., 5-10% higher accuracy) on reasoning tasks but parity on simple classification.

**vs. Hidden state MSE distillation** — On a task with diverse prompts (e.g., varying syntactic structures), direct MSE forces exact hidden state match, causing the student to overfit to specific representations and generalize poorly. Our method uses learned linear maps, allowing flexible alignment without forcing exact values. We expect DSD to outperform on tasks with high prompt variability, showing a 3-5% gain.

**vs. No distillation** — On any few-shot task, training from scratch yields low accuracy due to weak ICL ability. Our method transfers teacher dynamics, providing a strong initialization. We expect DSD to outperform scratch by a large margin (e.g., 15-20% accuracy improvement) across all tasks.

### What would falsify this idea

If DSD fails to outperform hidden state MSE distillation on tasks requiring multi-step reasoning, or if the identity-map ablation matches learned-map DSD, then the central claim that dynamics alignment via learned linear isomorphism improves ICL transfer is wrong. A uniform gain across all subsets (rather than concentrated on reasoning tasks) would also contradict the expected mechanism.

## References

1. Knowledge Distillation for Language Models
2. Knowledge distillation and dataset distillation of large language models: emerging trends, challenges, and future directions
3. TAGCOS: Task-agnostic Gradient Clustered Coreset Selection for Instruction Tuning Data
4. A Comprehensive Survey of Small Language Models in the Era of Large Language Models: Techniques, Enhancements, Applications, Collaboration with LLMs, and Trustworthiness
5. From Quantity to Quality: Boosting LLM Performance with Self-Guided Data Selection for Instruction Tuning
6. The Internal State of an LLM Knows When its Lying
7. MCC-KD: Multi-CoT Consistent Knowledge Distillation
8. AWQ: Activation-aware Weight Quantization for On-Device LLM Compression and Acceleration
