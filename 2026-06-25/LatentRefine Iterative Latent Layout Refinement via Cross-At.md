# LatentRefine: Iterative Latent Layout Refinement via Cross-Attention Consistency for Structure-Aware Text-to-Image Generation

## Motivation

IV-CoT achieves structure-aware generation through implicit latent plans, but its single-pass nature prevents iterative correction, leading to errors in complex scenes. GoT-R1 demonstrates the value of multi-step reasoning, but requires explicit intermediate outputs and external reward models. We bridge these by introducing a self-supervised iterative refinement loop for latent structural queries, eliminating external supervision while enabling multi-step correction.

## Key Insight

Cross-attention consistency between structural queries and generated image features provides a self-supervised correction signal because layout errors cause misaligned attention patterns, which can be corrected by updating queries to minimize attention entropy while maintaining diversity.

## Method

### (A) What it is
**LatentRefine** is a recurrent module that refines latent structural queries in the IV-CoT framework using cross-attention consistency feedback. It takes as input the initial set of structural queries and the generated image features (from a single forward pass) and outputs updated queries that produce more coherent layouts, with no external sketch supervision required.

### (B) How it works
```pseudocode
# Let Q_s be structural queries (N x d), F be image feature map (H*W x d) from IV-CoT decoder.
# Parameters: refinement transformer R (2 layers, d_model=512, 8 heads), T=3 steps.
# Hyperparameters: λ_entropy=0.1, λ_div=0.5, diversity threshold τ=0.7.
# Calibration: a small calibration set of 512 annotated layout examples from T2I-CompBench train set.
# Train a logistic regression classifier C: [mean_entropy, mean_similarity] -> correctness probability.
# At inference, C is used to decide whether to refine: only if C(predicted_prob) < 0.5.

function LatentRefine(Q_s, F, text_emb, calibration_classifier):
    # Compute initial attention statistics for all queries
    A_init = softmax(Q_s @ F^T / sqrt(d))
    entropy_init = -sum(A_init * log(A_init + 1e-8), dim=-1)
    sim_init = cosine_similarity(A_init, A_init)  # exclude diagonal
    mean_ent = mean(entropy_init)
    mean_sim = mean(sim_init[~eye(N)])
    correctness_prob = calibration_classifier([mean_ent, mean_sim])
    if correctness_prob >= 0.5:
        return Q_s  # skip refinement, layout already predicted correct
    for t in 1..T:
        # Compute cross-attention from Q_s to F
        A = softmax(Q_s @ F^T / sqrt(d))   # N x (H*W)
        # Compute per-query entropy
        entropies = -sum(A * log(A + 1e-8), dim=-1)   # shape N
        # Compute pairwise cosine similarity between attention maps
        sim = cosine_similarity(A, A)   # N x N, exclude diagonal
        # Loss for refinement (used only in training; during inference we update Q_s)
        L = λ_entropy * mean(entropies) + λ_div * mean(clamp(sim - τ, min=0))
        # Prepare conditioning signal: concatenate current Q_s, mean(entropies, keepdim) broadcasted, and global text embedding (pooled)
        context = [Q_s, entropies.unsqueeze(-1).expand(-1, d), text_emb.expand(N, -1)]
        delta = R(context)   # MLP or small transformer, output shape (N x d)
        Q_s = Q_s + 0.1 * delta   # learning rate 0.1 to stabilize
    end
    return Q_s
```
During training, we minimize L over refinement steps using gradient descent through the unrolled loop. At inference, we run the loop conditionally based on calibration classifier and use final Q_s to generate the image.

### (C) Why this design
We chose a transformer-based refinement module over an MLP because it can capture interactions between different structural queries (e.g., to prevent two queries from collapsing to the same attention region), accepting higher parameter count. We fixed T=3 as a trade-off between correction depth and computational cost; more steps could improve accuracy but risk overfitting to the self-supervised loss. Using attention entropy as the primary self-supervised signal (rather than an external reward model like in GoT-R1) avoids reliance on MLLM evaluation and keeps training efficient, though it may not capture full layout correctness (e.g., missing objects). The diversity penalty is crucial to prevent all queries from attending to the same salient region; we set τ=0.7 to allow moderate similarity while penalizing near-identical attention. We apply a small update step (0.1 × delta) to stabilize iterative refinement, which slows convergence but prevents divergence from the initial plan. The load-bearing assumption is that minimizing cross-attention entropy and maximizing attention diversity across structural queries directly corresponds to improving layout coherence for all text-to-image compositions. To verify this assumption, we introduce a lightweight calibration classifier trained on a small annotated set (512 examples) that predicts layout correctness from initial attention statistics; at inference, we only refine when the classifier indicates potential error, providing a concrete validation that the self-supervised signals align with ground-truth layout quality.

### (D) Why it measures what we claim
Per-query attention entropy measures **layout focus**: lower entropy implies a query attends strongly to a specific image region, which (under the assumption that each structural query corresponds to a distinct object or part) indicates that the query has settled on a coherent location for that element. This assumption fails when multiple objects overlap or share similar appearances, causing high entropy despite correct layout; in that case entropy reflects uncertainty rather than error. The pairwise similarity of attention maps measures **layout distinctness**: low similarity across queries ensures that different structural queries capture different image regions, which is essential for a structured scene. The assumption is that a correct layout assigns each object to a distinct region; failure occurs when objects are naturally adjacent or identical, forcing similarity despite correct layout, leading to an unnecessary penalty. The refinement delta is computed by a transformer that ingests both attention statistics and current query embeddings, which together operationalize **iterative correction** by allowing the model to adjust queries based on how well they currently align with the generated image features. The calibration classifier operationalizes **verification of the load-bearing assumption**: if the classifier predicts correctness based on initial statistics, we skip refinement; this tests whether entropy/diversity correlate with ground-truth layout quality.

## Contribution

(1) A novel iterative latent refinement module that updates structural queries via cross-attention consistency, enabling multi-step layout correction without external supervision. (2) A self-supervised training objective combining attention entropy and diversity loss that automatically drives layout improvement. (3) A training procedure that uses solely self-supervised signals to train the refinement module, removing dependence on sketch supervision or external reward models.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|-------------------------|
| Dataset | T2I-CompBench (all subsets: spatial, non-spatial, attribute binding, complex) | Tests compositional generation with spatial relations and hard cases. |
| Primary metric | Spatial accuracy (OSR score) on hard subsets (≥3 objects) | Directly measures layout coherence on challenging compositions. |
| Baseline 1 | IV-CoT (base) | Base method without refinement. |
| Baseline 2 | Janus-Pro | Strong unified model without explicit structure. |
| Baseline 3 | GoT-R1 | Uses MLLM feedback for layout refinement. |
| Ablation-of-ours | LatentRefine w/o diversity loss | Isolates diversity penalty contribution. |
| Ablation-of-ours | LatentRefine w/o calibration classifier | Isolates calibration contribution. |

### Why this setup validates the claim
T2I-CompBench is specifically designed to evaluate compositional understanding, including spatial relations, object attributes, and multiple objects. The hard subsets (≥3 objects) provide a stringent test where layout errors are frequent. IV-CoT serves as the direct baseline to test whether iterative refinement improves upon the initial structural queries. Janus-Pro and GoT-R1 represent strong alternatives: Janus-Pro lacks explicit structural queries, while GoT-R1 uses external MLLM feedback rather than self-supervised signals. The primary metric, spatial accuracy (OSR score) on hard subsets, directly measures whether generated layouts correctly reflect spatial relations from the text. The ablation without diversity loss isolates whether the diversity term is necessary for distinct object placement. The ablation without calibration classifier tests whether the lightweight verification step is beneficial. This combination provides a falsifiable test: if our method improves spatial accuracy, it must be due to the refinement mechanism, and each ablation reveals which component drives the gain.

### Expected outcome and causal chain

**vs. IV-CoT** — On a case where the prompt specifies two objects in a left-right relation (e.g., "a cup on the left of a book"), IV-CoT's initial structural queries may collapse to overlapping attention regions because they are not refined based on cross-attention consistency. Our method iteratively updates queries using entropy minimization, diversity penalty, and a calibration gate that skips refinement when the initial layout is already predicted correct. The observable signal: a noticeable gap on hard spatial relation subsets (e.g., +10-15% OSR accuracy on subsets with ≥3 objects) but parity on single-object prompts.

**vs. Janus-Pro** — On a case with multiple objects and attributes (e.g., "red cube next to blue sphere behind green cylinder"), Janus-Pro, lacking explicit structural planning, often conflates attributes or misplaces objects because it relies on implicit attention without disentangled queries. Our method explicitly refines latent queries to attend to distinct regions matching text features. We expect a noticeable gap on compositional prompts with ≥3 objects and attribute binding (e.g., +15-20% OSR accuracy), but similarity on simple single-object prompts.

**vs. GoT-R1** — On a case with a subtle spatial arrangement (e.g., "a clock above a mirror"), GoT-R1 uses an MLLM to evaluate layout and adjust via RL, which is computationally expensive and may not generalize to rare arrangements due to limited reward model coverage. Our self-supervised refinement uses intrinsic signals (entropy, diversity) plus a calibration classifier to verify assumption, being universally applicable and efficient. Expected: comparable OSR accuracy on common arrangements, but our method achieves faster inference (no MLLM calls) and slightly better accuracy on rare or complex spatial relations (+5-10%).

**Visualization analysis** — We will provide attention map visualizations for initial vs. refined queries on representative failure cases, showing entropy reduction and diversity increase, with quantitative bar charts of average entropy and similarity across refinement steps for correct vs. incorrect layouts.

### What would falsify this idea
If our method shows improvement on all metrics uniformly, including non-layout metrics like FID, rather than a concentrated gain on spatial accuracy hard subsets, or if spatial accuracy does not improve over IV-CoT on T2I-CompBench hard subsets despite similar computational budget, or if the calibration classifier's skip decision does not correlate with layout quality (e.g., false positives skip needed refinement), then the claim that entropy and diversity refinement specifically improves layout coherence is falsified.

## References

1. IV-CoT: Implicit Visual Chain-of-Thought for Structure-Aware Text-to-Image Generation
2. Janus-Pro: Unified Multimodal Understanding and Generation with Data and Model Scaling
3. GoT-R1: Unleashing Reasoning Capability of MLLM for Visual Generation with Reinforcement Learning
4. Emu3: Next-Token Prediction is All You Need
5. PUMA: Empowering Unified MLLM with Multi-Granular Visual Generation
6. Emu Edit: Precise Image Editing via Recognition and Generation Tasks
7. VL-GPT: A Generative Pre-trained Transformer for Vision and Language Understanding and Generation
8. mPLUG-OwI2: Revolutionizing Multi-modal Large Language Model with Modality Collaboration
