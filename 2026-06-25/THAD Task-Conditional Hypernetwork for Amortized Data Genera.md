# THAD: Task-Conditional Hypernetwork for Amortized Data Generation Adaptation

## Motivation

Gradient-based meta-learning (e.g., MAML) for synthetic data generation, as used in Autodata, requires per-task inner-loop gradient updates that are computationally prohibitive when scaling to many tasks. This bottleneck arises because the meta-learner must repeatedly compute second-order gradients or truncated backpropagation for each new task, limiting real-time adaptation. Existing approaches like Autodata suffer from this cost despite their effectiveness.

## Key Insight

The mapping from task embedding to optimal generator parameters is smooth and can be learned via a contrastive alignment objective that bypasses the need for inner-loop optimization.

## Method

Load-bearing assumption: The reference parameters θ*_i, obtained via 50-step gradient descent (learning rate=0.01) on a support set of 32 examples, approximate the true optimal generator parameters. To ensure this, we filter meta-training tasks where the reference generator's loss on a held-out set (16 examples) exceeds 0.5. Tasks failing this filter are discarded.

(A) **What it is**: THAD is a hypernetwork that takes a task embedding as input and directly outputs the parameters of a data generator, allowing immediate generation without any gradient-based adaptation.

(B) **How it works**:
```python
# Training
for each task T_i in meta-training set:
    # Step 1: Compute task embedding e_i using a pretrained encoder (e.g., sentence-BERT on task description)
    e_i = sentence-BERT(task_description_i)
    
    # Step 2: Generate candidate generator parameters θ_i = H(e_i) where H is hypernetwork (2-layer MLP, hidden=512, output=D, GeLU activation)
    θ_i = hypernetwork(e_i)
    
    # Step 3: Compute contrastive loss using a set of reference parameters
    # Reference: optimal parameters θ*_i obtained from 50 steps of gradient descent on a support set (only for training)
    # Positive pair: (e_i, θ*_i). Negative pairs: (e_i, θ*_j for j≠i)
    loss = -log( exp(sim(θ_i, θ*_i)/τ) / (exp(sim(θ_i, θ*_i)/τ) + sum_{j≠i} exp(sim(θ_i, θ*_j)/τ) ) )
    # sim = cosine similarity, τ=0.1, batch size N=64
    
    # Update hypernetwork to minimize loss using Adam (lr=1e-4)
    # Note: θ*_i is precomputed and fixed during hypernetwork training
```
At test time: for new task T_new, compute e_new, then θ_new = H(e_new), and generate data using generator with parameters θ_new. Hyperparameters: encoder fixed (sentence-BERT); hypernetwork architecture (2-layer MLP, hidden=512, output=D, GeLU); temperature τ=0.1; batch size N=64; optimizer Adam lr=1e-4.

(C) **Why this design**: We chose a contrastive objective over regression to the reference parameters because regression would force the hypernetwork to memorize exact parameter values, which are high-dimensional and noisy, while contrastive learning encourages the hypernetwork to produce embeddings that cluster with correct parameters, accepting the trade-off that the generated parameters may not be exactly optimal but will be functionally similar. We used a fixed pretrained encoder for task embeddings to avoid end-to-end training complexity, accepting that the encoder's representation may not be fully aligned with the generator space. We opted for cosine similarity in parameter space instead of L2 distance because parameters can vary in scale across tasks, and cosine similarity is invariant to scaling which is desirable for comparing direction. The cost is that we ignore magnitude information, but we can normalize generator parameters to unit norm.

(D) **Why it measures what we claim**: The contrastive loss <sim(θ_i, θ*_i)> measures alignment between the hypernetwork output and the optimal generator parameters for task i, because we assume the reference θ*_i (obtained via gradient-based meta-learning on a support set) is a good proxy for the true optimal parameters; this assumption fails when the reference is not truly optimal (e.g., due to limited steps or overfitting to support set), in which case the metric reflects how close the hypernetwork is to an imperfect reference rather than the true optimum. The use of negative pairs <sim(θ_i, θ*_j)> measures discriminability across tasks, because we assume that different tasks require different generator parameters; this assumption fails when tasks are very similar, so that negative pairs become similar to positive, leading to a poor contrastive signal. Overall, the method aims to amortize the meta-learning process by replacing gradient updates with a learned mapping, and the contrastive objective drives the hypernetwork to match the intended adaptation. Crucially, we assume that generator parameters with high cosine similarity produce similar data distributions; this is tested in ablation by comparing with L2 similarity and negative cosine similarity.

## Contribution

(1) A novel amortized meta-learning framework THAD that eliminates per-task inner-loop gradients by directly predicting generator parameters from task embeddings via a hypernetwork. (2) A contrastive training objective that aligns hypernetwork outputs with reference parameters without requiring online gradient adaptation. (3) Demonstration that this approach can reduce per-task computation to a single forward pass while preserving generation quality.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|------------------------|
| Dataset | Diverse task suite (CS, law, math) | Covers varied domains for generalization |
| Primary metric | Downstream task accuracy | Measures quality of generated data |
| Baseline 1 | Autodata | End-to-end meta-optimization baseline |
| Baseline 2 | GEPA | Prompt evolution without hypernetwork |
| Baseline 3 | Self-Instruct | Classical zero-shot generation |
| Ablation of ours | THAD w/o contrastive loss | Isolates effect of contrastive objective |
| Ablation 2 | THAD w/ L2 similarity | Tests effect of similarity measure |
| Ablation 3 | THAD w/ negative cosine similarity | Tests direction invariance |
| Metric 2 | Parameter cosine similarity to optimal | Measures alignment directly |
| Compute | 100 GPU hours (4x A100-80GB) | For 500 tasks, 100k hypernetwork updates |

### Why this setup validates the claim

This setup forms a falsifiable test of the claim that THAD can amortize the meta-learning process for synthetic data generation. By comparing against Autodata (which uses explicit meta-optimization), we test whether the hypernetwork can match or exceed the performance of a system that explicitly adapts to each task. GEPA and Self-Instruct test whether the hypernetwork approach outperforms simpler prompt-based methods that do not learn a mapping from task embeddings to generator parameters. The ablation (removing contrastive loss) tests the necessity of the contrastive objective: if it performs similarly, then the core claim that contrastive learning enables better amortization is unsupported. The additional ablations (L2 and negative cosine similarity) test the assumption that cosine similarity in parameter space is appropriate for measuring functional similarity. The downstream task accuracy directly reflects the quality of generated data, as it measures how well a model trained on synthetic data performs on real test sets. The diverse task suite ensures that improvements are not domain-specific. The compute row provides feasibility context.

### Expected outcome and causal chain

**vs. Autodata** — On a case where the new task has a novel structure not seen in meta-training (e.g., combining legal reasoning with mathematical objects), Autodata's gradient-based meta-optimization may overfit to its support set or require many steps, producing unstable generator parameters. Our THAD instead directly maps the task embedding (derived from description) to parameters via the hypernetwork, which has learned a shared latent space; even for novel structures, the embedding may lie near known tasks, yielding reasonable parameters. We expect THAD to show a noticeable gain on tasks with high diversity but parity on tasks very similar to training.

**vs. GEPA** — On a case where task requires generating data with precise numerical relations (e.g., scientific formulas), GEPA's reflective prompt evolution may struggle to refine prompts to capture exact numerical patterns because it operates in natural language space. Our method directly outputs generator parameters that can encode such relations precisely via the decoder. Thus we expect THAD to significantly outperform GEPA on numerical or structured tasks, while both may perform similarly on open-ended text.

**vs. Self-Instruct** — On a case where the task is highly specific (e.g., generating adversarial examples for a particular model), Self-Instruct, which relies on a template or instruction, lacks the ability to adapt its generation strategy per task, leading to low-quality data. Our method, by conditioning on task embedding, can generate tailored parameters. We expect THAD to show a large gap over Self-Instruct across all tasks, especially on niche domains.

### What would falsify this idea

If the ablation THAD w/o contrastive loss achieves comparable performance to THAD on diverse tasks, then the contrastive objective is not essential, contradicting our claimed mechanism. Additionally, if the similarity measure ablations show no significant difference, then the assumption about cosine similarity measuring functional similarity is unsupported. Alternatively, if THAD is consistently inferior to Autodata on tasks requiring high precision, then the hypernetwork fails to capture fine-grained parameter details.

## References

1. Autodata: An agentic data scientist to create high quality synthetic data
2. GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning
