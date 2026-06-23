# Joint Clustering of Multi-Teacher Representations for Knowledge Atom Distillation in Reinforcement Learning

## Motivation

Existing multi-teacher knowledge distillation methods for reinforcement learning often average teacher predictions uniformly or via learned weights, ignoring the functional specialization each teacher develops across different regions of the state space. This leads to redundant coverage of some skills and blind spots in others, as noted in multi-teacher frameworks surveyed by [Knowledge distillation and dataset distillation of large language models]. The root cause is the absence of a structural decomposition of teacher knowledge into complementary, non-overlapping atoms that can be assigned to separate student subnetworks.

## Key Insight

Joint clustering of hidden representations from multiple teachers reveals a natural partition of the state space into regions where each teacher's expertise dominates, which correspond to distinct knowledge atoms that can directly define both the student expert heads and the routing policy.

## Method

We propose **MACo** (Multi-teacher Atom Clustering distillation), a framework that decomposes multi-teacher knowledge into disjoint atoms via clustering and trains a mixture-of-experts student with one head per atom.

### (A) What it is
MACo is a knowledge distillation method for RL that, given a set of teacher policies with identical architecture, learns a student policy composed of: (1) a shared encoder that maps states to a low-dimensional embedding, (2) a set of expert heads {h₁,...,hₖ}, each responsible for imitating teacher behavior in a distinct region of the state space, and (3) a router network r(s) that selects the active head(s) for a given state. The routing structure and expert initialization are derived from joint clustering of teacher hidden representations augmented with policy divergence to ensure clusters correspond to behavioral differences.

### (B) How it works

**Input**: K teacher policies π₁,...,πₖ, student architecture (encoder φ, router r, heads h₁..hₖ), number of clusters m, temperature τ, balancing hyperparameter λ=0.5.
**Output**: Trained student policy.

**Phase 1: Collect teacher representations and policy divergences**
1. Sample a diverse set of states S from the environment (e.g., via random rollouts of a uniform policy).
2. For each state s ∈ S, feed s through each teacher πₜ, extract the hidden activation vector aₜ(s) from a chosen layer (e.g., the penultimate layer), and compute the policy divergence dₜ(s) = KL(πₜ(s) || π_avg(s)), where π_avg(s) is the uniform average of all teacher action probabilities for state s.
3. For each state-teacher pair, create an augmented feature vector fₜ(s) = [aₜ(s); λ·dₜ(s)]. (Optionally, apply PCA to reduce dimension of aₜ(s) before concatenation.)

**Phase 2: Joint clustering to define atoms**
4. Run K-means (Euclidean distance) on the set X = {fₜ(s) | t∈[1,K], s∈S} with m clusters. Obtain cluster centers c₁,...,cₘ.
5. Assign each state-teacher pair (s,t) to the cluster with nearest center; let C_j be the set of pairs assigned to cluster j.

**Phase 3: Initialize student components**
6. **Expert heads**: For each cluster j, initialize head h_j as a 2-layer MLP (128-128, ReLU activations). Set its target policy to be the average of all teacher action distributions over (s,t) in C_j, weighted by teacher confidence if available (e.g., softmax temperature).
7. **Router network**: Train a classifier (2-layer MLP, 256-64-m, ReLU + softmax) on the state embeddings φ(s) to predict the cluster assignment of the *best* teacher for s (the teacher whose fₜ(s) was closest to the assigned cluster center). Use cross-entropy loss. This defines r(s) = softmax(CLF(φ(s))/τ).
8. **Encoder φ**: Initialize randomly (3-layer MLP, 256-256-128, ReLU); will be updated end-to-end.

**Phase 4: Distillation training**
9. For each state s in a replay buffer, compute teacher outputs π₁(s),...,πₖ(s) (action probabilities).
10. For each state s, compute router logits r(s) = softmax(CLF(φ(s))/τ).
11. Compute student output as mixture-of-experts: π_student(s) = Σⱼ rⱼ(s) · hⱼ(φ(s)).
12. Minimize loss: L = KL(π_student(s) || π_aggregated(s)), where π_aggregated(s) is the aggregated teacher output (e.g., uniform average over teachers, or teacher-specific routing based on ground-truth clusters). In practice, we use the cluster-wise averaged teacher output as target: for each state s, the target is the average of teachers whose (s,t) pairs fall into the cluster with highest router probability.
13. Train end-to-end (encoder, router, heads) via Adam (lr=1e-4, β₁=0.9, β₂=0.999).

**Hyperparameters**: m = number of teachers × 5 (e.g., for 3 teachers, m=15). τ=0.5. λ=0.5.

**Pseudocode** is provided in Appendix A.

### (C) Why this design
We chose K-means over Gaussian Mixture Models because K-means yields hard cluster assignments that simplify head initialization and avoid probabilistic entanglement; the trade-off is that K-means assumes isotropic clusters, which may not capture elongated teacher manifolds, but this is mitigated by using a shared encoder later learned end-to-end and by augmenting features with policy divergence to separate clusters based on behavior. We selected the penultimate layer for representations because it balances abstraction and specificity—early layers are too low-level, logits are too task-specific. This design decision accepts the cost that the chosen layer may still contain task-irrelevant features; future work could automatically select the layer via relevance metrics. We trained the router via supervised learning on cluster assignments from the best teacher rather than from all teachers jointly, because the best teacher assignment encourages the router to identify the most reliable expert per state; the cost is that states where multiple teachers are equally good may be assigned to only one cluster, potentially losing complementary knowledge. We used a uniform average of teachers as target for distilling each head rather than a quality-weighted average, because weights are hard to calibrate and uniform averaging is simpler; the downside is that noisy teachers degrade the target, but we assume teachers are pre-trained to reasonable performance. Finally, we chose end-to-end fine-tuning over a frozen encoder+router because it allows the encoder to adapt to the router’s needs, improving routing accuracy; the cost is potential forgetting of the original clustering structure, which we mitigate by initializing heads close to cluster targets.

### (D) Why it measures what we claim
**Cluster assignment of teacher representations** measures **functional specialization** because teachers that produce similar hidden activations and policy divergences for a state are assumed to encode similar knowledge about that state’s action-value function; this assumption fails when two teachers coincidentally produce similar features but have diverging policies due to representational similarity independent of behavior—augmenting with policy divergence mitigates this failure. **Router probability rⱼ(s)** measures the **relevance of knowledge atom j** to state s because it is trained to predict the teacher whose augmented feature is closest to cluster center cⱼ; this assumes that cluster-center proximity implies expertise dominance, which fails when the best teacher for a state is not the one with nearest cluster center (e.g., due to outlier representations). **Expert head hⱼ(φ(s))** measures the **behavior implied by knowledge atom j** because it is trained to match the aggregated teacher outputs for states where the cluster is selected; this assumes that all states assigned to the cluster share a common behavior pattern, which fails when a cluster contains states with conflicting teacher behaviors (e.g., due to insufficient cluster count), in which case the head learns the average of conflicting behaviors. The **KL divergence loss** between student mixture and aggregated teacher output measures **knowledge coverage** because it penalizes the student for deviating from the teachers’ combined expertise; this assumes the aggregated target (uniform average over teachers) correctly represents the union of teacher knowledge, which fails when the aggregation drowns out rare but important expert knowledge, causing the loss to measure only majority behavior instead of comprehensive coverage. We validate this proxy by correlating cluster assignments with teacher performance on states (see experiment).

## Contribution

(1) A new multi-teacher knowledge distillation framework for RL that decomposes teacher knowledge into disjoint atoms via joint clustering of hidden representations, enabling a structured mixture-of-experts student. (2) A principled initialization and routing scheme that aligns each expert head to a cluster of teacher expertise and trains the router to predict cluster membership, ensuring comprehensive coverage without redundancy. (3) An end-to-end training procedure that jointly optimizes encoder, router, and heads while preserving the clustering-derived structure.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|----------------------|
| Dataset | DM Control Suite (tasks: Walker, Quadruped, Humanoid) | Diverse tasks with teacher specialization potential |
| Primary metric | Normalized return averaged over episodes | Directly measures task performance |
| Baseline 1 | Uniform distillation (averaged teachers) | Tests if naive averaging ignores specialization |
| Baseline 2 | Single-teacher per state (nearest activation) | Tests routing based on raw teacher representations |
| Baseline 3 | Best teacher (oracle) | Upper bound for comparison |
| Baseline 4 | Clustering teacher outputs (action probabilities) instead of hidden representations | Isolates benefit of hidden-layer clustering |
| Ablation of ours | MACo without end-to-end fine-tuning | Isolates benefit of end-to-end learning |
| Additional analysis | Correlation between cluster assignment and teacher return on held-out states | Validates the proxy that cluster-center proximity implies expertise dominance |

### Why this setup validates the claim

This setup forms a falsifiable test by combining diverse tasks where teachers may specialize (DM Control), a direct performance metric (normalized return), and baselines that isolate key sub-claims. Uniform distillation tests whether naive averaging fails to capture complementary knowledge; single-teacher per state tests whether raw activation routing is sufficient; the output-clustering baseline tests whether hidden representations provide additional benefit beyond action probabilities; the best teacher oracle sets an upper bound. The ablation tests whether end-to-end fine-tuning is critical. The metric directly measures the ultimate goal—task success—so performance patterns reveal whether MACo's clustering and routing actually improve knowledge distillation. The correlation analysis quantifies the assumption that cluster membership corresponds to teacher expertise dominance. If MACo outperforms baselines specifically on tasks with clear teacher specialization, the central claim is supported; if gains are uniform or absent, the idea is falsified.

### Expected outcome and causal chain

**vs. Uniform distillation** — On a task where two teachers specialize (e.g., one excels at fast movement, another at precise control), uniform averaging produces a policy that performs poorly at both because it averages contradictory action distributions. MACo clusters states by teacher representations and policy divergences, so the router selects the expert head trained on similar states, enabling strong performance in both regimes. We expect a noticeable gap (e.g., 20–30% higher return) on such heterogeneous tasks but parity on homogeneous tasks where teachers agree.

**vs. Single-teacher per state** — On a state where the best teacher’s activation is an outlier (e.g., due to noise), single-teacher routing picks a suboptimal expert, degrading performance. MACo uses joint clustering to define atoms and initializes heads as cluster averages, smoothing noise and robustly capturing shared knowledge. We expect MACo to outperform by a smaller but consistent margin (5–10%) with lower variance.

**vs. Clustering teacher outputs** — Clustering action probabilities directly ignores the rich structure of hidden representations, which may miss subtle state similarities. MACo's hidden-layer clustering yields better separation, leading to more coherent atoms. We expect MACo to outperform by 5–15% on tasks where hidden representations capture task-relevant features beyond raw actions.

**vs. Ablation (w/o fine-tune)** — Without end-to-end fine-tuning, the encoder cannot adapt to the router’s needs, leading to misaligned embeddings and reduced routing accuracy. MACo’s full version allows the encoder to specialize for routing, improving head selection. We expect full MACo to outperform the ablation by roughly 10% average return.

### What would falsify this idea

If MACo’s gain over uniform distillation is uniformly small across all tasks (or absent), or if the ablation performs equally well, then the clustering and routing mechanism is not capturing meaningful specialization—contradicting the central claim. Additionally, if the correlation between cluster assignment and teacher return is weak (e.g., Spearman ρ < 0.3), the proxy assumption is not validated.

## References

1. Knowledge distillation and dataset distillation of large language models: emerging trends, challenges, and future directions
2. DiffLM: Controllable Synthetic Data Generation via Diffusion Language Models
3. A Comprehensive Survey of Small Language Models in the Era of Large Language Models: Techniques, Enhancements, Applications, Collaboration with LLMs, and Trustworthiness
4. TAGCOS: Task-agnostic Gradient Clustered Coreset Selection for Instruction Tuning Data
5. Controlled Text Generation via Language Model Arithmetic
6. Mixed-Type Tabular Data Synthesis with Score-based Diffusion in Latent Space
7. Data Diversity Matters for Robust Instruction Tuning
8. MoDS: Model-oriented Data Selection for Instruction Tuning
