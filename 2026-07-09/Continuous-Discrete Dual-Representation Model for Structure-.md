# Continuous-Discrete Dual-Representation Model for Structure-Property Prediction via Bidirectional Latent Consistency

## Motivation

Current methods like SciReasoner rely on discretizing structural data into token sequences and autoregressive decoding, which discards continuous geometric information (e.g., precise coordinates, angles) that is critical for accurate property prediction. This loss is a structural consequence of treating structural representations as purely discrete, ignoring the underlying continuous manifold. The root cause is the assumption that token-level reasoning is sufficient—an assumption that persists across multiple branches (e.g., SciReasoner, Seq2Symm) and limits reasoning completeness.

## Key Insight

Continuous geometric features and discrete structural tokens are dual representations of the same object; enforcing bidirectional consistency between a neural ODE latent dynamics and an autoregressive token decoder preserves information that either alone discards.

## Method

**CDRM: Continuous-Discrete Dual-Representation Model**

(A) **What it is:** CDRM is a framework that couples a neural ODE (continuous latent dynamics) with an autoregressive token decoder for structure-property prediction. It inputs a structural sequence (e.g., residue tokens) and outputs a property label, while preserving continuous geometric information through a shared latent state.

(B) **How it works:**
```
Input: token sequence x = [x_1,...,x_T], property label y (training)
Output: predicted property ŷ

Encoder: E(x) → latent state h_0 (continuous vector)
Neural ODE: dh/dt = f_θ(h, t) over t ∈ [0, T] → h_1,...,h_T (sampled at integer t)
Decoder: At each step i, h_i is fed into a transformer layer to predict token x_i+1 (autoregressive)
  p(x_{i+1} | h_i, x_{1:i}) = softmax(W * h_i + b)
Bidirectional consistency losses:
  L_forward = Σ_i ||g_φ(h_i) - onehot(x_{i+1})||^2  (g_φ is a 2-layer MLP, hidden=256, GeLU)
  L_backward = Σ_i ||h'_i - h_i||^2  where h'_i = BackwardEncoder(h_T, x_{i+1:T})  (BackwardEncoder is a 2-layer MLP, hidden=256, GeLU; trained with cycle-consistency: L_cycle = Σ_i ||h_i - BackwardEncoder(h_T, x_{i+1:T})||^2)
Property prediction: ŷ = MLP(h_T) (2-layer MLP, hidden=256, GeLU)
Training: minimize L = L_pred(ŷ, y) + λ_forward * L_forward + λ_backward * L_cycle
Hyperparameters: λ_forward = 0.1, λ_backward = 0.1, ODE solver: Runge-Kutta 4 with step size 0.1.
```
Note: The inverse ODE from the original description is replaced by a learned backward encoder to ensure invertibility (see load-bearing assumption).

(C) **Why this design:** We chose neural ODE over a discrete RNN for continuous dynamics because it can model irregularly sampled geometric evolution and provides a continuous path that can be interpolated, accepting higher computational cost per step. We chose bidirectional consistency over a single forward loss because backward consistency forces the decoder to be invertible, preventing information loss in the discrete-to-continuous mapping; the trade-off is increased training complexity and potential for conflicting gradients. We chose a simple projection g_φ for forward consistency (rather than a learned attention) to enforce a direct, interpretable link between latent state and token, at the cost of lower expressivity. The ODE is integrated over fixed time steps aligned with token positions to ensure a unified representation; this sacrifices the ability to model non-uniform time intervals but simplifies the coupling. A crucial load-bearing assumption underlying L_backward is that the neural ODE dynamics are invertible and that the token-conditioned inverse ODE accurately recovers the forward latent trajectory. Since neural ODEs are not inherently invertible, we replace the inverse ODE with a separate learned backward encoder (2-layer MLP) trained with a cycle-consistency loss, which empirically ensures that the backward mapping is well-defined.

(D) **Why it measures what we claim:** The forward consistency loss L_forward measures the degree to which the continuous latent state h_i captures the next token's discrete information; specifically, it assumes that a linear projection of h_i should predict the next token, so low L_forward implies h_i is a sufficient representation of the discrete structure. This assumption fails when the projection is too simple to capture token dependencies (e.g., long-range correlations), in which case L_forward becomes a noisy proxy. The backward consistency loss L_cycle measures the invertibility of the discrete-to-continuous mapping; it assumes that the learned backward encoder can recover earlier latents from later ones and token conditioning, so low L_cycle indicates the continuous trajectory retains the full context. This assumption fails when the backward encoder capacity is insufficient or the dynamics are chaotic, causing L_cycle to reflect reconstruction error rather than true invertibility. The joint minimization ensures that the continuous latent is both predictive of tokens and recoverable from them, directly operationalizing the motivation of preserving continuous geometry while enabling discrete reasoning.

## Contribution

(1) A novel framework CDRM that couples neural ODE with autoregressive token generation via bidirectional consistency losses, enabling structure-property prediction without losing continuous geometric information. (2) An empirical finding that preserving continuous geometry via this dual-representation approach improves property prediction accuracy on tasks where geometric details are critical (e.g., protein function prediction), compared to purely discrete baselines like SciReasoner. (3) A new training objective that enforces invertibility between continuous and discrete representations, which can serve as a principle for other sequence-to-structure models.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|-----------------------|
| Dataset | EC number prediction on PDB chains | Widely used benchmark for structure-function |
| Primary metric | Macro F1 score | Balances precision/recall across classes |
| Baseline 1 | Sequence-only RNN | Tests need for continuous dynamics |
| Baseline 2 | Structure-based GNN | Tests discrete structure processing |
| Baseline 3 | Transformer with positional encoding | Strong baseline with attention |
| Ablation-of-ours | CDRM w/o backward consistency loss | Tests contribution of backward loss |

### Why this setup validates the claim

The chosen dataset (EC number prediction) requires integrating sequential residue information with geometric structure from the PDB, which matches CDRM's dual-representation design. The baselines isolate key subclaims: the sequence-only RNN tests whether continuous dynamics matter, the structure-based GNN tests whether discrete geometry suffices, and the transformer tests whether attention plus positional encoding can replace ODE-based trajectories. The ablation directly tests the backward consistency loss. The macro F1 metric is appropriate because it treats all enzyme classes equally, ensuring that improvements on rare classes are not overshadowed. This combination creates a falsifiable test: if CDRM outperforms baselines on subsets where geometry or sequential dependencies are critical, the claim holds; otherwise, the mechanism is unsubstantiated.

### Expected outcome and causal chain

**vs. Sequence-only RNN** — On a case where the active site geometry is essential (e.g., an enzyme with a deep cleft), the RNN fails to capture spatial constraints because it only sees token order, so it predicts a wrong EC class. CDRM instead uses the neural ODE to embed continuous geometric evolution from the sequence, so its latent state encodes shape. Thus, we expect a large gap (e.g., +0.15 F1) on subsets with known geometric determinants, but parity on sequence-only tasks.

**vs. Structure-based GNN** — On a case where long-range sequential dependencies matter (e.g., an enzyme with distant residues forming a binding pocket), the GNN processes local 3D neighborhoods but misses global sequential context, leading to misclassification. CDRM's autoregressive decoder conditions on entire token history via the transformer, integrating sequential logic. Therefore, we expect CDRM to outperform by at least +0.10 F1 on proteins with long-range interactions, but similar performance on compact globular domains.

**vs. Transformer with positional encoding** — On a case where the structure is irregularly sampled (e.g., loops with variable geometry), the transformer's fixed positional encoding cannot adapt to continuous deformation, causing attention to be misaligned. CDRM's ODE provides a continuous latent path that can be interpolated, so it accurately tracks geometry. We expect CDRM to show a noticeable gain (e.g., +0.08 F1) on proteins with flexible loops, but comparable accuracy on regular secondary structures.

**vs. Ablation (CDRM w/o backward loss)** — On a case where the reverse consistency is needed (e.g., a symmetric protein where discrete tokens must map invertibly to geometry), the ablation loses information during encoding, leading to poorer property prediction. The full CDRM enforces invertibility, so we expect a gap of ~0.05 F1 on symmetric proteins, with parity elsewhere.

### What would falsify this idea

If CDRM's gain over baselines is uniform across all test subsets rather than concentrated on instances with critical geometry or long-range dependencies, then the proposed continuous-discrete coupling is not actually addressing those failure modes, and the central claim that the dual representation is beneficial would be falsified.

## References

1. Accurate, Interdisciplinary and Transparent Structure-property Understanding with Deep Native Structural Reasoning
2. Training a Scientific Reasoning Model for Chemistry
3. Rapid and accurate prediction of protein homo-oligomer symmetry using Seq2Symm
4. RSGPT: a generative transformer model for retrosynthesis planning pre-trained on ten billion datapoints
5. Evolutionary-scale prediction of atomic level protein structure with a language model
6. Protein language models can capture protein quaternary state
7. O1 Replication Journey - Part 2: Surpassing O1-preview through Simple Distillation, Big Progress or Bitter Lesson?
8. Aviary: training language agents on challenging scientific tasks
