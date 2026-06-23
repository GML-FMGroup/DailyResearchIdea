# Logit Factorization Distillation: Training MoE Students from Teacher Outputs Alone

## Motivation

Existing distillation methods for reinforcement learning, such as PADD, rely on access to teacher internal states (e.g., hidden activations) under the assumption that output logits alone are insufficient for effective knowledge transfer. This limitation prevents distillation from black-box teachers or in privacy-sensitive scenarios. The root cause is the belief that output logits lack the structural information needed to guide a student mixture-of-experts (MoE) policy. We identify that many RL policies exhibit low-rank structure in their logit space, which can be exploited to recover expert prototypes without any internal signals.

## Key Insight

By factorizing the teacher's output logit matrix into non-negative expert prototypes and state-specific mixture weights, we can recover the teacher's modular structure solely from observable outputs, enabling structurally faithful student imitation.

## Method

(A) **What it is**: Logit Factorization Distillation (LFD) trains a student MoE policy to match a teacher policy by first decomposing the teacher's output logits (over a set of states) into a set of expert prototypes and per-state routing weights via non-negative matrix factorization (NMF). The student then learns to reproduce both the prototypes and the routing, without ever accessing teacher internals.

(B) **How it works**:
```pseudocode
Input: teacher policy π_T (black-box), student MoE π_S with N experts, trajectories D = {s}
Hyperparameters: N (number of experts, e.g., 4), λ (distillation weight, e.g., 1.0), α (policy gradient weight, e.g., 0.1)

1. Collect logits: for each state s in D, compute teacher logits l_T(s) = softmax^{-1}(π_T(a|s))? Actually use softmax outputs for non-negativity.
   Let p_T(s) = softmax(logits) be the action probability vector.
2. Build matrix P ∈ [0,1]^{|D| × |A|} where rows are p_T(s).
3. Apply non-negative matrix factorization: P ≈ W * H, where W ∈ ℝ^{|D| × N} (weight per state per expert) and H ∈ ℝ^{N × |A|} (expert prototypes), with W ≥ 0, H ≥ 0.
   Use multiplicative update with KL divergence objective.
4. For each state s, compute NMF-based expert assignment w(s) = W_{s,:} (normalized to sum to 1) and expert prototypes h_j = H_{j,:} (also normalized).
5. Train student MoE:
   - Student has router r(s) outputting distribution over experts, and per-expert heads e_j(s) outputting action logits.
   - Distillation loss L_dist = Σ_s KL( p_T(s) || Σ_j r_j(s) * softmax(e_j(s)) ), but also encourage r(s) to match w(s) and e_j(s) to produce softmax over actions similar to h_j.
     Specifically: L_router = Σ_s KL( w(s) || r(s) ), L_expert = Σ_j Σ_s KL( h_j || softmax(e_j(s)) ).
   - Combined distillation: L_kd = L_dist + λ (L_router + L_expert).
6. Add policy gradient loss L_pg (e.g., PPO) on task reward.
7. Total loss: L = L_kd + α L_pg.
8. Update student by gradient descent.
```
Hyperparameters: N (expert count) set via cross-validation; λ and α trade off imitation vs. task performance.

(C) **Why this design**: We chose NMF over other matrix factorization (e.g., PCA) because NMF yields non-negative components that can be interpreted as action probability distributions, directly aligning with the student's per-expert heads. This interpretability comes at the cost of requiring non-negative inputs, so we operate on softmax probabilities rather than raw logits, which may lose some ranking information but ensures consistency. We opted to separate the distillation into three losses (output, router, expert) rather than a single KL on the mixture, because matching only the final mixture may allow the student to learn different internal representations that do not capture the teacher's modular structure. The cost is additional hyperparameters and optimization complexity, but it enforces structural fidelity. Finally, we include a policy gradient term to maintain task performance when the distillation signal is imperfect; this trades off some distillation focus for robustness to factorization errors.

(D) **Why it measures what we claim**: The NMF decomposition of the teacher's output probabilities into expert prototypes and state weights measures the teacher's modular structure because it assumes that each state's action distribution is a convex combination of a small set of base distributions (experts). The assumption that the teacher policy has such a low-rank structure is what enables recovering experts from outputs alone; this assumption fails when the teacher is highly stochastic or multimodal without reuse of components, in which case the reconstruction error is high and the distillation signal may be misleading. The student's router matching to w(s) measures its ability to replicate the teacher's state-conditioned expert selection, because w(s) is inferred from the teacher's output distribution. The expert head matching to h_j measures the student's ability to learn the teacher's action prototypes. Together, these operationalize structurally faithful distillation because they force the student to imitate the teacher's decomposition, not just the final distribution.

## Contribution

(1) A novel distillation framework (LFD) that trains a student MoE policy to match a teacher policy using only output logits, via non-negative matrix factorization of the teacher's action probability matrix. (2) The demonstration that the factorization can recover meaningful expert prototypes that correspond to behavioral modes, enabling structured knowledge transfer without teacher internals. (3) A design principle for black-box distillation in RL: output logits alone are sufficient when the teacher policy exhibits low-rank structure in its action distribution.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | MuJoCo continuous control tasks | Tests high-dimensional action spaces |
| Primary metric | Average episodic return | Direct measure of task performance |
| Baseline 1 | Standard KD (output KL only) | Tests benefit of structural matching |
| Baseline 2 | MoE from scratch (PG only) | Tests value of distillation signal |
| Ablation of ours | LFD w/o router matching | Isolates effect of router constraint |

### Why this setup validates the claim
This combination of dataset, baselines, and metric forms a falsifiable test of the claim that Logit Factorization Distillation (LFD) enables structurally faithful imitation by recovering the teacher's modular structure via NMF. MuJoCo tasks provide continuous control with state-dependent action distributions, where expert reuse is plausible. Standard KD tests whether the student benefits from enforcing router and expert matching beyond just matching the output mixture. The MoE-from-scratch baseline isolates whether distillation adds any value over pure reinforcement learning. The ablation (removing router matching) tests whether the router constraint is critical for performance. The primary metric, average return, directly measures task success; if LFD outperforms both baselines and the ablation specifically on states where the teacher exhibits modular behavior (identified via reconstruction error), the central claim is supported. Conversely, if gains are uniform across all states, the claim of structural fidelity is undermined.

### Expected outcome and causal chain

**vs. Standard KD (output KL only)** — On a state where the teacher policy mixes two distinct skill patterns (e.g., a running gait that combines forward thrust and balance), standard KD forces the student's mixture to match the teacher's output distribution but allows internal representations to diverge. This can lead to the student learning brittle, entangled features that fail to generalize to novel situations (e.g., a sudden change in terrain). Our method, by explicitly matching the router weights to the NMF-inferred w(s) and expert heads to prototypes h_j, forces the student to learn the same decomposition. This structural alignment improves robustness, yielding a noticeable gap in return on states with high teacher reconstruction error (indicating modularity), while performance is similar on homogeneous states.

**vs. MoE from scratch (PG only)** — On a complex task like Humanoid, where exploration is difficult, the MoE trained only with policy gradients may converge to a suboptimal policy that uses only a subset of experts (e.g., one expert dominates), failing to utilize the full capacity. Teacher distillation provides a rich curriculum of behavioral patterns, and LFD's factorization ensures the student's experts are initialized to cover diverse teacher prototypes. Thus, the student achieves higher sample efficiency and final performance, especially in early training. We expect LFD to achieve higher return with fewer environment interactions, as measured by area under the learning curve.

**vs. LFD w/o router matching** — On a state where the teacher uses different experts for different action dimensions (e.g., left leg vs. right leg), the ablation that only matches output and expert prototypes may learn a router that mixes experts in a way that does not align with the teacher's state-conditioned allocation. This leads to interference between experts and degraded performance on states requiring precise coordination. Our full method preserves the teacher's routing, resulting in lower variance among expert contributions and better fine-grained control. We expect a performance drop in the ablation specifically on tasks requiring high coordination (e.g., Walker2d vs. Hopper).

### What would falsify this idea
If the performance gain of LFD over standard KD is uniform across all states (i.e., not concentrated in states where the teacher's NMF reconstruction error is high), then the central claim that structural fidelity drives improvement is false.

## References

1. PADD: Path-Aligned Decompression Distillation for Non-Router Teacher to Guide MoE Student Learning
2. Knowledge distillation and dataset distillation of large language models: emerging trends, challenges, and future directions
3. DiffLM: Controllable Synthetic Data Generation via Diffusion Language Models
4. A Comprehensive Survey of Small Language Models in the Era of Large Language Models: Techniques, Enhancements, Applications, Collaboration with LLMs, and Trustworthiness
5. TAGCOS: Task-agnostic Gradient Clustered Coreset Selection for Instruction Tuning Data
6. Controlled Text Generation via Language Model Arithmetic
7. Mixed-Type Tabular Data Synthesis with Score-based Diffusion in Latent Space
8. Data Diversity Matters for Robust Instruction Tuning
