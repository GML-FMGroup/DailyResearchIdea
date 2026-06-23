# FISH: Fisher Information Self-distillation via Hidden States for Language Models

## Motivation

Existing self-distillation methods for language models rely on task-specific heuristics like attention head selection (SEEKR) or semantic clustering (SSD) to identify important teacher knowledge, lacking a principled, task-agnostic criterion. This reliance on heuristics limits generalizability and introduces assumptions that may not hold across tasks. The Fisher Information Matrix (FIM) of the teacher's predictive distribution with respect to its hidden states offers a natural, task-agnostic importance measure that captures how sensitive outputs are to perturbations in each hidden dimension, enabling principled knowledge transfer.

## Key Insight

The Fisher Information of the teacher's output distribution with respect to its hidden states quantifies the local curvature of the teacher's log-likelihood, directly measuring the importance of each hidden dimension for the teacher's predictions independent of any downstream task.

## Method

Assumption: The diagonal Fisher Information of the teacher's predictive distribution with respect to its hidden states, computed on unlabeled data, serves as a task-agnostic importance measure that generalizes across tasks.

(A) **What it is**: FISH is a self-distillation method that computes the Fisher Information Matrix (FIM) of the teacher's predictive distribution with respect to its hidden states on a large unlabeled corpus (e.g., 500K tokens), then uses the diagonal of the FIM to weight the distillation loss, focusing the student on the most informative teacher knowledge.

(B) **How it works**:
```python
def fish_distillation(teacher, student, unlabeled_data, val_calibration_set=None):
    # Step 1: Compute Fisher Information for each hidden state dimension
    fisher_accum = 0
    num_batches = 0
    for batch in unlabeled_data:
        hidden_states, logits = teacher.forward_with_hidden(batch)
        probs = softmax(logits, dim=-1)
        predicted_class = argmax(probs, dim=-1)
        log_prob = log_softmax(logits)[:, predicted_class]
        grad_h = torch.autograd.grad(log_prob, hidden_states, retain_graph=True)[0]  # (seq_len, hidden_dim)
        fisher_diag = grad_h ** 2  # element-wise square, (seq_len, hidden_dim)
        fisher_accum += fisher_diag.mean(dim=0)  # average over sequence length
        num_batches += 1
    fisher_weights = fisher_accum / num_batches  # (hidden_dim,)
    # Normalize to sum to hidden_dim (optional)
    fisher_weights = fisher_weights / fisher_weights.mean()

    # Verification step: calibrate Fisher weights on small labeled validation set (512 examples)
    if val_calibration_set is not None:
        # Prune bottom 20% of dimensions (by Fisher weight) and measure teacher accuracy change
        teacher.eval()
        threshold = torch.quantile(fisher_weights, 0.2)
        mask = fisher_weights > threshold
        # Evaluate teacher on val set with pruned hidden states (zero out low-weight dims)
        accuracy_full = evaluate(teacher, val_calibration_set)
        accuracy_pruned = evaluate_with_pruned_hidden(teacher, val_calibration_set, mask)
        drop = accuracy_full - accuracy_pruned
        print(f"Accuracy drop after pruning bottom 20% Fisher dims: {drop:.2f}%")
        # If drop < 1%, weights may be unreliable; if drop > 5%, weights are informative

    # Step 2: Distill student with weighted hidden-state loss
    for batch in training_data:
        with torch.no_grad():
            teacher_hidden, teacher_logits = teacher.forward_with_hidden(batch)
        student_hidden, student_logits = student.forward_with_hidden(batch)
        loss_hidden = (fisher_weights * (teacher_hidden - student_hidden) ** 2).mean()
        loss_kl = kl_div(student_logits, teacher_logits)
        loss = loss_kl + lambda_hidden * loss_hidden  # lambda_hidden default=1.0
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```
Hyperparameters: `lambda_hidden` controls the weight of hidden-state loss (default=1.0); Fisher weights computed once before distillation or updated every 10K batches; calibration threshold for pruning set at bottom 20%.

(C) **Why this design**: We chose Fisher Information over gradient norm (e.g., TAGCOS) because Fisher Information has a principled interpretation as the local curvature of the log-likelihood, directly measuring information content, whereas gradient norm is scale-dependent and task-specific. We use the diagonal of the FIM rather than the full matrix to keep computation tractable, accepting the loss of off-diagonal covariance information (which captures interactions between dimensions) because weights remain positive and can be used directly as a per-dimension importance signal. We compute Fisher on unlabeled data only, not the training set, to ensure task-agnostic importance; this reduces domain-specific bias but may miss task-relevant dimensions if the unlabeled data distribution differs from the target domain. The weighting is applied to hidden-state MSE rather than logits because hidden states are the direct carriers of knowledge; logit loss already uses KL divergence, making hidden-state weighting complementary.

(D) **Why it measures what we claim**: The diagonal Fisher information `fisher_weights[d] = E[ (∂ log p(y|h)/∂h_d)^2 ]` measures **knowledge importance** of hidden dimension `d` because it quantifies how much the teacher's output probability changes when `h_d` is perturbed, under the assumption that the teacher's log-likelihood is locally quadratic in h. This assumption fails when the teacher's predictions are discontinuous w.r.t. h (e.g., due to sharp nonlinearities like ReLU or hard attention), in which case `fisher_weights` instead reflects the average squared gradient magnitude, which may be large even for irrelevant dimensions if the function is noisy. Additionally, when hidden dimensions are highly correlated, the diagonal approximation ignores off-diagonal interactions: a dimension may have low individual Fisher but high importance when combined with another dimension. In such cases, `fisher_weights` underestimates importance of correlated groups. The weighting of hidden-state MSE by `fisher_weights` operationalizes **concentrated knowledge** because dimensions with high Fisher information force the student to match them more closely, focusing capacity on the most predictive parts of the teacher's representation. The use of unlabeled data ensures the measure is **task-agnostic**: `fisher_weights` does not depend on any downstream labels, consistent with the motivation of generalizable importance.

## Contribution

(1) A principled, task-agnostic knowledge importance identification method via Fisher Information on teacher hidden states for self-distillation of language models. (2) A weighted distillation loss that automatically emphasizes the most informative hidden dimensions, reducing reliance on heuristics. (3) An empirical demonstration that FISH improves student uncertainty calibration and downstream task performance compared to standard self-distillation and heuristic-based selection.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|----------------------|
| Dataset | MMLU (5-shot) | Tests generalization across diverse knowledge domains |
| Primary metric | Accuracy | Direct measure of student task performance |
| Baseline | Vanilla KD | Only logit distillation, no hidden-state guidance |
| Baseline | FitNet | Uniform hidden-state MSE, no importance weighting |
| Baseline | TAGCOS | Gradient-norm weighting, related importance method |
| Ablation | FISH (Fisher on labeled) | Tests effect of using labeled vs unlabeled data |

### Why this setup validates the claim

MMLU's 57 diverse subjects provide a broad testbed for knowledge distillation. Accuracy directly reflects student ability to replicate teacher knowledge. Vanilla KD isolates the effect of hidden-state supervision; FitNet tests uniform weighting; TAGCOS tests an alternative importance metric; and our ablation assesses the necessity of unlabeled-data Fisher computation. Together, these baselines form a falsifiable test: if Fisher weights truly capture informative dimensions, FISH should outperform all baselines and the ablation, especially on subjects requiring precise factual knowledge.

### Expected outcome and causal chain

**vs. Vanilla KD** — On a case where the teacher's logits are uncertain (e.g., ambiguous multiple-choice questions with close probabilities), vanilla KD provides poor guidance because it only matches soft probabilities, which contain limited information. Our method additionally uses Fisher-weighted hidden states, which capture fine-grained predictive structure. Thus we expect a noticeable accuracy gap on ambiguous subjects (e.g., ethics or social sciences) but parity on clear-cut factual subjects (e.g., mathematics).

**vs. FitNet** — On a case where hidden dimensions vary in importance (e.g., in biology, certain neuron activations are critical for species classification), FitNet treats all dimensions equally, wasting capacity on noisy features. Our method concentrates learning on high-Fisher dimensions via weighted MSE. We expect a clear gap on knowledge-intensive subjects (e.g., biology, physics) but smaller differences on reasoning-heavy tasks (e.g., logic).

**vs. TAGCOS** — On a case where gradient norm is large due to outlier activations (e.g., rare tokens with high variance), TAGCOS overweights those dimensions, distorting student representations. Our Fisher weighting is principled and scale-invariant, avoiding such biases. We expect a modest but consistent improvement across all subjects, with largest gains in low-resource domains where outliers are more frequent.

### What would falsify this idea
If FISH performs worse than FitNet or shows uniform gain across all subjects (rather than concentration on knowledge-intensive subsets), the central claim that Fisher weights measure knowledge importance would be invalid.

## References

1. Semantic Self-Distillation for Language Model Uncertainty
2. Knowledge distillation and dataset distillation of large language models: emerging trends, challenges, and future directions
3. LLM Distillation for Efficient Few-Shot Multiple Choice Question Answering
4. DiffLM: Controllable Synthetic Data Generation via Diffusion Language Models
5. TAGCOS: Task-agnostic Gradient Clustered Coreset Selection for Instruction Tuning Data
6. FIRST: Teach A Reliable Large Language Model Through Efficient Trustworthy Distillation
7. SEEKR: Selective Attention-Guided Knowledge Retention for Continual Learning of Large Language Models
8. Controlled Text Generation via Language Model Arithmetic
