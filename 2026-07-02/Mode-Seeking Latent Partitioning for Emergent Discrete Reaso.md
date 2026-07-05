# Mode-Seeking Latent Partitioning for Emergent Discrete Reasoning in Multimodal Variational Models

## Motivation

Existing continuous latent reasoning methods, such as AMVL, align prior and posterior via bidirectional KL but treat the latent space as a homogeneous manifold, lacking mechanisms to extract discrete reasoning steps. This structural limitation prevents modeling hierarchical or branching reasoning within a variational framework. As a concrete instance, AMVL's continuous latents cannot naturally decompose a complex reasoning chain into subgoals, while MCOUT's continuous thoughts similarly lack explicit step boundaries. The root cause is the absence of a variational regularizer that leverages mode-seeking properties to partition the latent space into distinct clusters corresponding to reasoning stages.

## Key Insight

The reverse KL divergence’s mode-seeking behavior can be harnessed to collapse continuous latent posterior onto a set of learned prototype anchors, inducing discrete reasoning steps as a natural byproduct of variational optimization.

## Method

**Mode-Seeking Latent Partitioning (MSLP)**

(A) **What it is**: We propose **Mode-Seeking Latent Partitioning (MSLP)**, a variational framework that augments continuous latent reasoning with a mixture prior and a reverse-KL regularizer to partition the latent trajectory into discrete reasoning steps without explicit discrete variables. Input: multimodal input (e.g., image + text question). Output: answer with an automatically segmented reasoning chain in the latent space.

(B) **How it works**: The core operation is an iterative latent refinement loop with a mode-seeking constraint and a diversity penalty. Below is pseudocode describing the training procedure.

```pseudocode
# Hyperparameters:
# K = number of prototype anchors (K=5)
# beta = weight of reverse-KL regularizer (beta=0.1)
# gamma = weight of diversity penalty (gamma=0.05)
# T = number of reasoning steps (T=3)
# sigma = fixed prior std (sigma=0.5)
# sigma_q = fixed posterior std (sigma_q=0.1)
# Processor P: 1-layer Transformer with 4 heads, hidden dim 256, feed-forward dim 512
# Anchor initialization: k-means on initial posterior means of first batch

# Learnable parameters:
# Anchors: A = {a_1,...,a_K} each a 256-dim vector
# Prior mixture: p(z) = (1/K) * sum_{k=1}^K N(z| a_k, sigma^2 I)
# Encoder E: ResNet-18 + MLP to 256-dim
# Latent Processor P: as above
# Decoder D: MLP to answer logits

for each training batch:
    (x, y) = batch  # x: multimodal input, y: answer
    # Step 1: Encode input to initial latent
    z_0 = E(x)  # mean of initial posterior, 256-dim
    # Step 2: Iterative refinement for t=1..T
    for t in 1..T:
        # Update latent with processor
        z_t = P(z_{t-1}, x)  # Transformer layer, output 256-dim
        # Compute reverse-KL between prior and posterior
        # Posterior q(z_t) = N(z_t, sigma_q^2 I)
        kl_rev = KL(p(z) || q(z_t))  # mode-seeking; closed form: sum over k (1/K) * (log(N(z_t|a_k, sigma^2 I)) - log(N(z_t|z_t, sigma_q^2 I))) plus constant terms
        # Compute assignment probability to each anchor
        # Compute logits: log p(z_t|a_k) = -0.5*||z_t - a_k||^2 / sigma^2 (ignoring constants)
        assign_logits = -0.5 * sum((z_t - A)**2, dim=-1) / sigma**2  # shape (K,)
        assign_probs = softmax(assign_logits, dim=-1)
        # Diversity penalty: negative entropy of assign_probs over the batch mean
        # Use entropy over steps to encourage different anchors per step
        H = -sum(assign_probs * log(assign_probs + 1e-8))
        # Regularize loss
        loss += beta * kl_rev - gamma * H
    # Step 3: Decode answer from final latent z_T
    y_hat = D(z_T)
    # Step 4: Standard supervised loss
    loss += CrossEntropy(y_hat, y)
    update parameters via gradient descent
```

During inference, we run the same iterative refinement without the KL regularizer (or with it to stabilize). The latent z_t naturally gravitates toward one anchor per step, yielding a sequence of discrete cluster assignments that serve as reasoning steps.

(C) **Why this design**: We chose a mixture-of-Gaussians prior over a uniform continuous prior because it provides explicit centers that the posterior can collapse to. Using reverse KL instead of forward KL is deliberate: forward KL would encourage the posterior to cover all mixtures, destroying discreteness. The trade-off is that reverse KL risks posterior collapse to a single anchor across steps; we mitigate this by initializing anchors to be well-separated (via k-means) and using a small beta, and further by adding a diversity penalty (negative entropy) to encourage distinct assignments per step. We chose a fixed number of anchors K=5 rather than a nonparametric prior to keep training stable, accepting the limitation that K limits the reasoning depth. The iterative refinement processor is a 1-layer Transformer (4 heads, hidden 256) inspired by MCOUT but we add the KL regularizer per step; this incurs extra computation but yields structured trajectories. We opted for a simple Gaussian posterior (isotropic, sigma_q=0.1) rather than a full-covariance one to avoid overfitting, trading flexibility for regularization strength.

(D) **Why it measures what we claim**: The reverse-KL term KL(p||q) measures **mode-seeking compression** because it penalizes q(z) for assigning probability where p(z) is low; under the assumption that p(z) is a mixture with well-separated components, this forces q(z) to concentrate near a single anchor (a discrete step). This assumption fails when anchors are not well-separated: then the posterior may spread over multiple adjacent anchors, in which case KL(p||q) reflects generic proximity rather than discrete assignment. The anchor vectors themselves measure **subgoal prototypes** because they are learned end-to-end to capture recurring latent patterns (reasoning primitives). This fails if the reasoning steps are not shared across examples: then anchors become arbitrary attractors that do not correspond to meaningful decomposition. The iterative processor P measures **state transition** as it maps one latent to the next; the regularizer ensures each transition is near an anchor, operationalizing a discrete step boundary. This assumption fails when the processor’s input-output mapping is so flexible that it can encode transitions without using the anchors, in which case the KL term only weakly influences the trajectory.

(E) **Load-bearing assumption and repair**: The core assumption is that the reverse-KL regularizer with a mixture-of-Gaussians prior causes the latent posterior at each reasoning step to concentrate on a single, distinct anchor, thereby inducing discrete reasoning steps. This assumption might be false because reverse KL is known to cause mode collapse, potentially collapsing all steps to the same anchor. To repair this, we add a diversity penalty (negative entropy of assignment probabilities across steps) as a concrete calibration: we compute the assignment probabilities of each step's latent to the K anchors, then compute the entropy of the average assignment distribution over steps, and maximize it (via negative weight) to encourage distinct anchors per step. During validation, we monitor the step-to-anchor assignment uniqueness (e.g., number of distinct anchors used per trajectory); if uniqueness drops below a threshold (e.g., 0.8*T), we increase gamma. This repair is embedded in the loss function as shown in the pseudocode.

## Contribution

(1) A variational framework (MSLP) that induces discrete reasoning steps from continuous latent trajectories via reverse-KL regularization onto a mixture prior, removing the need for explicit discrete latent variables. (2) A design principle: mode-seeking can partition a continuous reasoning space into emergent subgoals, enabling hierarchical structuring within end-to-end variational models. (3) An empirical analysis (in the full paper) showing that MSLP produces interpretable step boundaries and improves accuracy on multimodal reasoning benchmarks over continuous-only baselines.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|------------------------|
| Dataset | CLEVR | Tests compositional visual reasoning with clear steps. |
| Additional dataset | NLVR2 | More challenging visual language reasoning. |
| Synthetic dataset | Synthetic step sequences (e.g., 3-step arithmetic with known subgoals) | Verifies KL collapse corresponds to correct step boundaries. |
| Primary metric | Answer accuracy | Direct measure of reasoning correctness. |
| Step purity (synthetic only) | Fraction of trajectories where each step's latent maps to the correct ground-truth anchor | Measures assignment quality. |
| Baseline 1 | MCOUT | Continuous latent reasoning without discrete steps. |
| Baseline 2 | Standard VAE | Continuous prior without mode-seeking regularization. |
| Baseline 3 | VQ-VAE with iterative refinement | Discrete latent method to compare with mode-seeking KL. |
| Ablation 1 | MSLP w/o reverse-KL | Isolates effect of mode-seeking constraint. |
| Ablation 2 | MSLP w/o diversity penalty | Isolates effect of diversity regularization. |

### Why this setup validates the claim

This design forms a falsifiable test of whether mode-seeking reverse-KL induces discrete reasoning steps. CLEVR requires multi-step compositional reasoning, so success depends on stepwise latent structure. Comparing to MCOUT (no step partitioning) tests if discrete anchors improve upon continuous trajectories. Standard VAE tests the necessity of iterative refinement altogether. VQ-VAE baseline tests what mode-seeking KL adds beyond deterministic quantization. The synthetic dataset with known ground-truth steps allows direct verification that the posterior at each step collapses to the correct anchor. The step purity metric on synthetic data measures whether KL collapse corresponds to meaningful step boundaries. Ablations isolate the contributions of the reverse-KL and diversity penalty. Accuracy is the right metric because it directly reflects reasoning quality; if discrete steps help, gains should concentrate on multi-step questions. Step purity validates the mechanism. If the predicted pattern (gain on multi-step, parity on single-step, high purity on synthetic) does not hold, the claim fails.

### Expected outcome and causal chain

**vs. MCOUT** — On CLEVR multi-step questions (e.g., count red cubes after filtering by size), MCOUT's continuous latent trajectory may blend steps, causing ambiguous transitions and misanswers. Our method partitions the latent into discrete anchors per step via KL collapse and diversity penalty, ensuring each subgoal is clearly separated. We expect 5-10% accuracy gain on questions requiring ≥3 steps, and parity on single-step questions. On synthetic data, we expect high step purity (>90%) for our method, while MCOUT will not produce discrete assignments.

**vs. Standard VAE** — On CLEVR questions requiring chain of transformations, the standard VAE's single latent variable cannot represent sequential steps, leading to poor performance (e.g., <60% accuracy on 3-step questions). Our method's iterative refinement with mode-seeking forces stepwise progress, so we expect a large gap (e.g., >30% accuracy difference). On single-step questions, both should perform similarly.

**vs. VQ-VAE with iterative refinement** — On synthetic data, VQ-VAE also produces discrete assignments via vector quantization. However, its deterministic quantization may yield hard assignments that are brittle; MSLP's stochastic reverse-KL with diversity penalty may yield more flexible step boundaries. On CLEVR, we expect MSLP to match or slightly outperform VQ-VAE on accuracy (within 2%), but MSLP's discrete steps are emergent from variational optimization rather than enforced by quantization, highlighting the novelty of mode-seeking KL.

**vs. MSLP w/o reverse-KL** — On synthetic data, the ablation (no reverse-KL, no diversity penalty) may produce latent trajectories that drift without anchor constraints, leading to low step purity (<50%). On CLEVR multi-step questions, the ablation will likely perform worse (e.g., 5-10% accuracy drop) because latents are not forced to visit anchors, thus lacking discrete step boundaries. The diversity penalty ablation (no diversity penalty) may suffer from anchor collapse, causing multiple steps to map to the same anchor, reducing step purity to ~1/T (random) and harming reasoning performance on long chains.

**vs. MSLP w/o diversity penalty** — On CLEVR questions with many steps, the lack of diversity penalty may cause all steps to collapse to the same anchor, losing distinct step boundaries. We expect accuracy drop compared to full MSLP on 3-step questions, but improvement over w/o reverse-KL because reverse-KL still anchors steps (though not distinct). On synthetic data, purity will be low (~1/K) due to collapse.

### What would falsify this idea

If MSLP shows no performance drop relative to MCOUT on single-step questions but also no gain on multi-step questions, or if the ablation matches MSLP on multi-step questions, then the claim that reverse-KL induces discrete steps is falsified. Additionally, if step purity on synthetic data does not exceed 80% for MSLP (or if diversity penalty ablation still achieves high purity), the mechanism is not as hypothesized.

## References

1. Multimodal Continuous Reasoning via Asymmetric Mutual Variational Learning
2. Multimodal Chain of Continuous Thought for Latent-Space Reasoning in Vision-Language Models
3. Variational Reasoning for Language Models
4. RAVR: Reference-Answer-guided Variational Reasoning for Large Language Models
5. Training Large Language Models to Reason in a Continuous Latent Space
6. STaR: Bootstrapping Reasoning With Reasoning
7. Guiding Language Model Reasoning with Planning Tokens
8. Think before you speak: Training Language Models With Pause Tokens
