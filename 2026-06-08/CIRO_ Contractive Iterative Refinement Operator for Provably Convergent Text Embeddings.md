# CIRO: Contractive Iterative Refinement Operator for Provably Convergent Text Embeddings

## Motivation

Existing iterative embedding refinement methods, such as GIRCSE, improve embedding quality by generating sequences of soft tokens under a contrastive objective, but they lack termination guarantees and can incur unbounded computational cost. The root cause is that no structural constraint ensures each refinement step reduces distance to a fixed point. Without a contractive property, the process may oscillate or require many steps, making it unsuitable for deployment where bounded latency is critical. This reflects the broader field-level bottleneck of ensuring bounded computational cost and convergence stability.

## Key Insight

Each refinement step is a learned contraction mapping with respect to a jointly optimized metric, guaranteeing convergence to a unique fixed point within a logarithmically bounded number of iterations.

## Method

```python
# CIRO: Contractive Iterative Refinement Operator
# Input: initial embedding e0 (d-dim vector from a pretrained encoder), contrastive positive/negative pairs
# Output: refined embedding e*
# Hyperparameters: Lipschitz constant L=0.9, max_iter K=5, metric network M_phi (2-layer MLP, hidden=256, ReLU), number of layers in f_theta N=3

# Load-bearing assumption: The learned refinement network f_theta is a contraction mapping with Lipschitz constant L < 1 with respect to the learned metric M_phi.
# To ensure this, we apply spectral normalization to each layer of f_theta with per-layer bound L^{1/N}.

# Phase 1: Train contraction network f_theta (3-layer MLP, hidden=512, GeLU) and metric M_phi end-to-end
for each batch:
    sample positive pair (x, x+) and negatives x-
    e0 = encoder(x)   # e0 is frozen encoder output
    e = f_theta(e0)   # one refinement step
    e+ = encoder(x+)  # encoder output for positive
    loss = InfoNCE(e, e+, negatives, temperature=0.07)  # contrastive loss
    # Apply spectral normalization to f_theta's weight matrices to enforce Lip(f_theta) <= L
    for each W in f_theta.parameters():
        W.data = W.data * (L ** (1 / N)) / max(1, spectral_norm(W))

# After training, verify contractivity: compute Jacobian spectral norm via power iteration on a calibration set of 512 held-out examples; confirm mean < L

# Phase 2: Inference refinement
e = e0
for t in 1..K:
    e_new = f_theta(e)
    if ||e_new - e||_{M_phi} < eps=1e-3:   # convergence check under learned metric
        break
    e = e_new
return e
```

**Why this design:** We chose spectral normalization over gradient penalty or weight clipping because it enforces a hard upper bound on the Jacobian norm of each layer, providing a provable Lipschitz constant for the entire network. This trade-off reduces model capacity but guarantees contractivity, which is essential for convergence. Second, we jointly learn a metric network M_phi to measure distances, adapting the contraction condition to the contrastive objective. Using a fixed Euclidean distance would impose a weaker condition, potentially slowing convergence, but learning M_phi adds parameters and risk of overfitting. Third, we adopt a fixed iteration budget K based on the contraction rate and desired precision, avoiding while loops that could cause non-termination. Accepting that in worst-case we may stop before full convergence, we ensure computational cost is bounded. Finally, we apply spectral normalization only to the refinement network, not the encoder, to avoid interfering with pretrained representations.

**Why it measures what we claim:** The spectral norm of the Jacobian of f_theta measures contractivity because if max_e ||J_f(e)||_{M_phi} ≤ L < 1, then f_theta is a contraction mapping in the metric induced by M_phi; this assumption fails when spectral normalization is imprecise (e.g., due to numerical errors), in which case the fixed point may not be unique or convergence may not hold. The contrastive loss measures embedding quality by driving the refined embedding to be similar to positives and dissimilar to negatives; this assumes that the fixed point of the contraction aligns with the global optimum of the contrastive objective, which fails when the training data distribution does not cover the test domain, leading a fixed point that is semantically meaningless. The convergence criterion using the learned metric distance operationalizes the notion of approximation error; it assumes that M_phi is a valid norm consistent with the contrastive objective, which fails if M_phi is not learned robustly, then the convergence check may trigger prematurely or too late.

## Contribution

(1) Introduces CIRO, a learned iterative text embedding refinement operator with provable convergence guarantees enforced by spectral normalization and a learnable metric. (2) Demonstrates that coupling a contractive mapping with a contrastive objective enables bounded-cost refinement without sacrificing embedding quality, providing a design principle for stable iterative representation learning. (3) Provides a theoretical guarantee that the fixed point is attained within O(log(1/ε)) steps under perfect enforcement of spectral normalization, where ε is the desired precision.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|-----------------------|
| Dataset | MTEB benchmark (STS, clustering, classification) | Diverse tasks to measure embedding quality. |
| Primary metric | Average task score across MTEB | Overall embedding quality measure. |
| Baseline | SimCSE (unsupervised) | Standard contrastive embedding baseline. |
| Baseline | Non-contractive iterative refinement with vanilla MLP (Lip=2.0) | Tests necessity of contractivity for convergence. |
| Baseline | Iterative refinement with Euclidean metric (no M_phi) | Tests benefit of learned metric for contraction. |
| Ablation | CIRO w/o spectral normalization | Isolates effect of Lipschitz constraint. |
| Latency metric | Average inference time per query (ms) on NVIDIA A100 | Measures bounded latency. |

### Why this setup validates the claim

This setup directly tests the core claim that contractive refinement improves contrastive embeddings through guaranteed convergence to a semantically meaningful fixed point. Using MTEB provides a comprehensive evaluation across diverse tasks. Comparing against SimCSE establishes whether refinement adds value over a single-pass encoder. Including a non-contractive baseline (vanilla MLP with Lip=2.0) tests whether contractivity is necessary; if it succeeds, our assumption is falsified. The Euclidean metric ablation tests whether the learned metric adapts contraction to the contrastive objective. The primary metric (average score) captures overall quality; if our method only improves on a subset, the uniform average might dilute gains, but we will analyze per-task to detect the predicted pattern. We also measure latency to demonstrate bounded computational cost, a key motivation.

### Expected outcome and causal chain

**vs. SimCSE** — On a case where the initial embedding is noisy (e.g., ambiguous query with multiple plausible meanings), SimCSE outputs a fixed embedding that may be close to an unrelated negative because it lacks iterative correction. Our method instead iteratively contracts toward positive neighborhoods under the learned metric, progressively reducing ambiguity. Thus, we expect a noticeable gap on ambiguous examples (e.g., +10% on STS hard subsets) but near parity on clean examples.

**vs. Non-contractive iterative refinement** — On a case where the initial embedding is far from the positive manifold (e.g., out-of-domain sample), a non-contractive MLP (Lipschitz=2.0) may amplify errors and oscillate, actually degrading the embedding. Our contractive network guarantees monotonic improvement towards the fixed point. Consequently, we expect non-contractive refinement to sometimes hurt performance (negative delta) while ours always improves, resulting in a gap on difficult samples.

**vs. Iterative refinement with Euclidean metric** — On a case where the positive and negative manifolds are highly nonlinear, the Euclidean metric may not capture the contrastive geometry, causing the contraction to be misaligned. The learned metric M_phi adapts to the contrastive objective, leading to more efficient convergence. We expect larger gains on tasks with complex semantic boundaries (e.g., clustering).

### What would falsify this idea

If non-contractive refinement (Lip=2.0) outperforms CIRO on average, then contractivity is not necessary for iterative embedding improvement. If CIRO's latency is not bounded (e.g., average < 10 ms) due to early stopping variance, the promise of bounded cost is broken. If the fixed point of f_theta does not correspond to improved contrastive scores on held-out data (e.g., score drops after convergence), the assumption linking contraction to embedding quality is false.

## References

1. Let LLMs Speak Embedding Languages: Generative Text Embeddings via Iterative Contrastive Refinement
2. Reasoning-Based Refinement of Unsupervised Text Clusters with LLMs
3. Uncovering Latent Arguments in Social Media Messaging by Employing LLMs-in-the-Loop Strategy
4. Contextualized Topic Coherence Metrics
5. Discovering Latent Themes in Social Media Messaging: A Machine-in-the-Loop Approach Integrating LLMs
6. Topic Coherence Metrics: How Sensitive Are They?
7. The Evolution of Topic Modeling
8. Challenging BIG-Bench Tasks and Whether Chain-of-Thought Can Solve Them
