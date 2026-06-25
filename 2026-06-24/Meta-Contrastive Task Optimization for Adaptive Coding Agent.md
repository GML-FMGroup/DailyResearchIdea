# Meta-Contrastive Task Optimization for Adaptive Coding Agent Evaluation

## Motivation

Current coding agent benchmarks such as NatureBench and MLE-bench assume a single-source static task set is representative of overall agent capability, but this allows agents to overfit to the task distribution, masking systematic weaknesses. The root cause is the lack of a mechanism to adaptively probe the agent's capability frontier: static sets sample the space coarsely and cannot discover failure modes absent from the curated set. For instance, NatureBench's 90 tasks from Nature-family papers may miss domains where agents underperform, while MLE-bench's reliance on Kaggle competitions biases evaluation toward competition-level tasks.

## Key Insight

An agent's performance across tasks is encoded in a low-dimensional capability manifold where distances between tasks correspond to performance differences, so generating tasks that maximize expected performance gap on the manifold's boundary provably exposes the agent's weakest capabilities.

## Method

# Hyperparameters: lambda_div = 0.3 (diversity weight), num_candidates = 100, encoder_dim = 128, meta_lr = 0.001, meta_steps = 5, calib_size = 16, K = 30 (limit iterations), generator model = CodeBERT (initialized with GitHub code prompts)
# Input: Agent A, initial task set T0 (e.g., HumanEval+ seeds), number of iterations K=30
# Output: Evaluated task set T

**Load-bearing assumption**: The contrastive manifold learned on initial tasks generalizes to new tasks such that distances in latent space correspond to performance differences. To mitigate distribution shift, we add a meta-learning calibration step (MAML-like) after each iteration that adapts the encoder using a small validation set of generated tasks.

1. T = T0
2. For each task t in T, compute agent performance s_t = A(t)  # e.g., pass@1
3. Learn capability manifold M via contrastive learning:
   - Encode each task t into z_t using a Transformer-based encoder E (BERT-base-uncased, hidden=768)
   - Pair tasks with similar performance differences into positive pairs, dissimilar into negatives
   - Train E with InfoNCE loss (temperature τ=0.07) to align z_t with performance differences
   - Use a 2-layer MLP projection head (hidden=256, ReLU) on top of E for contrastive embedding
4. Train a gap predictor P on (z_t, 1 - s_t) to estimate performance gap: 2-layer MLP (hidden=128, ReLU) with L2 loss
5. Initialize a task generator G (CodeBERT fine-tuned on task descriptions) with prompt template: "Write a Python coding task that tests {weakness_category}"
6. For i = 1 to K:
   a. Sample candidate tasks {t_c} from G conditioned on current manifold boundary (z_t's at extremes: top 10% by predicted gap)
   b. For each t_c, encode z_c = E(t_c), predict gap g_c = P(z_c)
   c. Compute diversity score d_c = min_{t in T} cosine_dist(z_c, z_t)
   d. Select t* = argmax_{t_c} g_c + lambda_div * d_c
   e. Execute A on t*, get score s*, update T = T ∪ {t*}
   f. Update M and P with new (z*, 1 - s*)
   g. (Calibration step) Sample calib_set of size calib_size=16 from {t in T} uniformly; for each, compute s_cal = A(t), 
      compute gradient of P on z_cal, update encoder E via 5 inner steps of meta-learning (MAML) using meta_lr=0.001 to adapt to new tasks
7. Return T

(C) **Why this design**: We chose contrastive learning over a variational autoencoder because contrastive learning directly captures relative performance differences between tasks, which is more aligned with identifying capability gaps than reconstructing task features. We trade off the ability to generate novel tasks (which a VAE could do) for a more discriminative representation that highlights weaknesses. We use a predictor P trained on manifold representations rather than raw features to avoid overfitting to task content – this forces P to rely on capability-relevant factors. We chose a diversity-seeking selection objective (weighted sum of predicted gap and diversity) over pure optimization of gap because it prevents the generator from repeatedly producing tasks in the same region of the capability space, which would lead to overfitting. The cost is tuning lambda_div, but we mitigate this with a schedule decreasing lambda_div linearly from 0.3 to 0.1 over K iterations. We selected a Transformer encoder (BERT-base-uncased) for E because it handles variable-length task descriptions and code; alternatives like graph neural networks would require custom parsing and lose sequence structure. The key trade-off is representation power versus computational cost – Transformer is expensive but captures long-range dependencies essential for coding tasks.

(D) **Why it measures what we claim**: The predicted performance gap g(t_c) measures the agent's *vulnerability* (likelihood of failure) because it is derived from the capability manifold that encodes systematic performance variations; this assumes the manifold generalizes to new tasks sharing latent factors – when this fails (e.g., a new capability dimension unseen in training), g(t_c) reflects predictor uncertainty rather than actual weakness. To detect such failures, we estimate the local Lipschitz constant L(t_c) = max_{t' in N_k(t_c)} |s(t)-s(t')| / ||z(t)-z(t')||_2 (k=5 nearest neighbors in training set); if L(t_c) > threshold 0.5, we flag g(t_c) as unreliable and increase lambda_div to encourage exploration. The diversity metric d_c measures *coverage* of the capability frontier because it enforces spacing in latent space, assuming the manifold is continuous – if the true space is disconnected (e.g., discrete task modalities), d_c may oversample in one region while missing isolated islands, so coverage becomes latent density rather than frontier extent. The contrastive loss aligns task embeddings with performance differences, so distances in latent space correspond to *capability distance*; this works under the assumption that performance is a smooth function of the latent code – if performance is non-smooth (e.g., brittle prompting), embeddings encode spurious correlations and distances no longer reflect capability. The local Lipschitz check provides a practical test of this smoothness assumption on-the-fly.

## Contribution

(1) A meta-evaluation framework (MCTO) that dynamically generates tasks to probe an agent's capability frontier, eliminating reliance on static benchmarks. (2) A contrastive-based capability manifold learning method that represents agent performance across tasks in a latent space, enabling task generation via diversity-seeking RL. (3) A design principle: using performance gap prediction on a learned manifold guides task generation toward the decision boundary of agent competence.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
|------|--------|-----------|
| Dataset | HumanEval+ seed tasks | Widely used coding benchmark. |
| Primary metric | Pearson correlation predicted vs actual gap | Directly measures estimation accuracy. |
| Random task generation | No optimization baseline. |
| VAE-based generation | Tests contrastive vs generative. |
| Ours without diversity term | Isolates diversity contribution. |

### Why this setup validates the claim
This design creates a falsifiable test of MCTO's claim to generate tasks that reveal capability gaps and estimate them accurately. Using HumanEval+ seeds ensures a controlled initial distribution. The correlation metric directly quantifies how well predicted gaps match actual performance drops on generated tasks. Random baseline shows the necessity of any optimization; VAE baseline tests if contrastive representation is superior; and the diversity ablation tests whether the diversity term broadens coverage. If MCTO achieves higher correlation than all baselines, it validates that the capability manifold learning and gap prediction are effective. Conversely, failure to outperform the ablation would indicate that diversity alone drives gains, undermining the gap prediction claim.

### Expected outcome and causal chain

**vs. Random** — On a task where the agent fails due to a specific weakness (e.g., concurrency bugs), random generation may never create such a task, so the baseline misses the gap entirely. Our method actively searches for tasks near the capability frontier, generating a task that triggers the failure. We expect a noticeably higher correlation on the subset of tasks requiring that weakness, while random shows near-zero correlation.

**vs. VAE** — On a pair of tasks with similar surface features (e.g., both use recursion) but different difficulty (e.g., one requires memoization), VAE embeddings may collapse the two due to reconstruction focus, failing to distinguish their difficulty. Our contrastive learning aligns embeddings with performance differences, so predicted gaps differentiate them. We expect our method to show higher correlation specifically on such feature-only similar but difficulty-different pairs.

**vs. Ours without diversity** — On a scenario where the agent has multiple distinct weaknesses (e.g., both in regex and memory management), the ablation focuses only on the first weakness discovered, overestimating gaps in that region while missing others. Our diversity term forces exploration across the manifold, so generated tasks cover both weaknesses. We expect our full method to produce more balanced coverage and higher overall correlation, with the ablation underperforming on tasks related to the second weakness.

### What would falsify this idea
If our method's correlation is not significantly higher than the ablation across all task subsets, or if the gain is uniform rather than concentrated on tasks where the predicted gaps are high, then the central claim that MCTO identifies capability gaps is false.

## References

1. NatureBench: Can Coding Agents Match the Published SOTA of Nature-Family Papers?
2. OpenScholar: Synthesizing Scientific Literature with Retrieval-augmented LMs
3. Autonomous chemical research with large language models
4. MLE-bench: Evaluating Machine Learning Agents on Machine Learning Engineering
5. SWE-bench: Can Language Models Resolve Real-World GitHub Issues?
6. MLAgentBench: Evaluating Language Agents on Machine Learning Experimentation
7. Closed-loop optimization of general reaction conditions for heteroaryl Suzuki-Miyaura coupling
8. CodeGen: An Open Large Language Model for Code with Multi-Turn Program Synthesis
