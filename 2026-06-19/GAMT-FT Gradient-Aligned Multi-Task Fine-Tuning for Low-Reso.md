# GAMT-FT: Gradient-Aligned Multi-Task Fine-Tuning for Low-Resource Code Generation

## Motivation

Current approaches for low-resource code generation (e.g., MultiPL-T, No Resource No Benchmarks) rely on a two-stage pipeline: pre-training on English code followed by fine-tuning on target-language data. This sequential process assumes that transfer is additive and that fine-tuning does not degrade English performance — an assumption that remains untested and likely false due to catastrophic forgetting. Moreover, the separation prevents English code supervision from benefiting target-language learning during joint optimization. We hypothesize that a single-stage approach can achieve better cross-lingual transfer by simultaneously optimizing both tasks with gradient alignment regularization.

## Key Insight

Enforcing positive cosine similarity between the gradients of English code generation and target-language instruction following ensures that parameter updates move in consistent directions, thereby preventing forgetting and enabling shared representation learning.

## Method

(A) **What it is**: GAMT-FT (Gradient-Aligned Multi-Task Fine-Tuning) is a single-stage training procedure that jointly optimizes an LLM on English code generation and target-language instruction data by projecting conflicting gradients to ensure positive correlation before updating. Inputs: a pretrained LLM, English code dataset D_en, target-language instruction dataset D_low. Outputs: a fine-tuned model with improved performance on both tasks.

(B) **How it works**:
```python
# Pseudocode for GAMT-FT
# Hyperparameters: lr, batch_size, conflict_threshold=0.0

θ = init_pretrained_llm()
for iter in range(max_steps):
    # Sample batches from each task
    B_en = sample(D_en, batch_size)
    B_low = sample(D_low, batch_size)
    
    # Compute gradients separately
    g_en = ∇θ L(f_θ(x_en), y_en)   # English code generation loss
    g_low = ∇θ L(f_θ(x_low), y_low) # Target language instruction loss
    
    # Compute cosine similarity
    cos_sim = (g_en · g_low) / (||g_en|| * ||g_low||)
    
    if cos_sim < conflict_threshold:  # gradients conflict
        # Project g_low onto the nullspace of g_en (PCGrad-style)
        g_low_proj = g_low - ( (g_low · g_en) / (||g_en||^2 + 1e-8) ) * g_en
    else:
        g_low_proj = g_low
    
    # Average aligned gradients
    g_avg = (g_en + g_low_proj) / 2.0
    
    # Update model
    θ = θ - lr * g_avg
```

(C) **Why this design**: We chose to project only the low-resource gradient onto the nullspace of the English gradient (rather than modifying both or using a weighted average) because English code data is typically more abundant and reliable; this preserves the English update direction intact, accepting the cost that the low-resource update may be overly constrained, potentially slowing target adaptation. We set the conflict threshold to 0.0 (strict positive correlation) to avoid any negative transfer; a softer threshold might allow small conflicts but was avoided to be conservative. We compute gradients on separate batches rather than a joint loss to avoid dominance by one task's scale; this introduces a hyperparameter for batch sizes but ensures equal influence per iteration. We average rather than sum the aligned gradients to keep the learning rate consistent with single-task training. Finally, we use the PCGrad projection formula (Yu et al., 2020) due to its simplicity and proven effectiveness in multi-task learning, rather than a more complex gradient surgery (e.g., orthogonal projection of both gradients), because our preliminary analysis showed English gradients dominate and projecting both could discard valuable English signal.

(D) **Why it measures what we claim**: The computational quantity `cos_sim` measures the alignment between English and target gradients; it operationalizes the concept of 'positive transfer' because if gradients are positively correlated, updating along their average benefits both tasks (by moving toward a shared representation that minimizes both losses). This assumption fails when gradient similarity arises from spurious correlations in small batches, in which case `cos_sim` may be positive but the shared update harms performance on a held-out set; we mitigate by using large batches (e.g., 32) and monitoring validation loss. The `g_low_proj` quantity measures the component of the low-resource gradient that is non-conflicting with the English gradient; it operationalizes the concept of 'non-forgetting' because any update orthogonal to the English gradient will not harm English performance (by the Taylor approximation). This assumption fails when the orthogonal update accidentally increases English loss due to non-linear interactions; we accept this risk and empirically verify that English loss does not increase. The `g_avg` quantity measures the final update direction; it operationalizes the concept of 'joint optimization' because it combines both tasks' non-conflicting signals, assuming that the average of aligned gradients points toward a Pareto-optimal solution. This assumption fails when the tasks have fundamentally incompatible minima; in that case, the method will oscillate and converge to a compromise with suboptimal performance on both tasks.

## Contribution

(1) A single-stage training framework (GAMT-FT) for low-resource code generation that jointly optimizes English code and target-language instructions via gradient alignment, directly addressing the untested two-stage pipeline assumption. (2) An empirical finding that enforcing gradient positive correlation prevents catastrophic forgetting and improves target-language code generation accuracy compared to two-stage fine-tuning baselines. (3) An analysis of gradient conflict patterns during joint training, showing that conflicting gradients are prevalent in naive joint training but are effectively mitigated by the proposed alignment, providing insights into cross-lingual transfer dynamics.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|------------------------|
| Dataset | LowResCode-Bench (target language) | Evaluates code generation in low-resource language. |
| Primary metric | Pass@1 (functional correctness) | Direct measure of code generation accuracy. |
| Baseline 1 | Zero-shot LLM | Measures initial performance without adaptation. |
| Baseline 2 | Naive multi-task fine-tuning | Tests standard joint training without gradient alignment. |
| Baseline 3 | Fine-tune only on target instruction | Shows target-only adaptation, may forget English code. |
| Ablation-of-ours | Ours without gradient projection | Tests effect of gradient projection component. |

### Why this setup validates the claim
The central claim is that GAMT-FT achieves positive transfer between English code generation and target instruction tasks, improving both without forgetting. The dataset directly measures target language code generation, while the pass@1 metric captures functional correctness essential for real-world use. The baselines isolate distinct failure modes: zero-shot quantifies the adaptation gap; naive multi-task fine-tuning tests if simple gradient averaging suffices; target-only fine-tuning reveals catastrophic forgetting of English. The ablation removes gradient projection to test its necessity. Additionally, we evaluate English code generation (not shown in table) to verify that English performance is maintained or improved. If GAMT-FT outperforms all baselines on target pass@1 while preserving English accuracy, the claim is supported. Conversely, if it fails, the cause (e.g., ineffective projection or task interference) can be identified by comparing to the ablation and baselines.

### Expected outcome and causal chain

**vs. Zero-shot LLM** — On a case where the target language uses unique syntax (e.g., postpositional phrases instead of prepositions), the zero-shot model produces incorrect code or hallucinations because it has never seen such structures during pre-training. Our method adapts using target instruction data, so it learns the correct syntax via gradient updates that align with English code gradients. We expect a large gap in pass@1 (e.g., ~35% improvement on target language examples with divergent syntax).

**vs. Naive multi-task fine-tuning** — On a case where the gradients for English code and target instructions push the model in opposite directions (e.g., one wants to increase weight on a shared feature while the other wants to decrease it), naive joint training averages the conflicting gradients, causing the update to harm both tasks. Our method projects the target gradient onto the nullspace of the English gradient, preserving the English update while removing the conflicting component. We expect our method to achieve 5-10% higher pass@1 on both tasks compared to naive multi-task fine-tuning, especially on examples where gradient conflict is high (e.g., tasks requiring different vocabulary biases).

**vs. Fine-tune only on target instruction** — On a case where the model strongly relies on English coding patterns (e.g., variable naming conventions), fine-tuning only on target data causes the model to overfit to target idioms, drastically reducing English code generation performance. Our method jointly trains on English code data, ensuring the model retains English patterns through aligned gradients. We expect similar target pass@1 (within 2%) but significantly higher English pass@1 (e.g., >20% improvement) compared to target-only fine-tuning.

### What would falsify this idea
If our method fails to outperform naive multi-task fine-tuning on target pass@1, or if the performance gain is uniform across all examples rather than concentrated on instances with conflicting gradients (e.g., where English and target syntax diverge), then the gradient projection is not the cause of improvement, and the central claim of positive transfer via conflict resolution is invalid.

## References

1. No Resource, No Benchmarks, No Problem? Evaluating and Improving LLMs for Code Generation in No-Resource Languages
2. Efficient Model Development through Fine-tuning Transfer
3. Enhancing Code Generation for Low-Resource Languages: No Silver Bullet
4. Knowledge Transfer from High-Resource to Low-Resource Programming Languages for Code LLMs
5. Investigating the Performance of Language Models for Completing Code in Functional Programming Languages: A Haskell Case Study
6. Code Llama: Open Foundation Models for Code
7. Layer Swapping for Zero-Shot Cross-Lingual Transfer in Large Language Models
8. Examining Modularity in Multilingual LMs via Language-Specialized Subnetworks
