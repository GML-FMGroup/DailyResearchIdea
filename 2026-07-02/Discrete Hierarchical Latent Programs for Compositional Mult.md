# Discrete Hierarchical Latent Programs for Compositional Multimodal Reasoning with Cross-Modal Tree Alignment

## Motivation

Multimodal variational models such as AMVL (Asymmetric Mutual Variational Learning) assume that continuous latent trajectories can capture all reasoning steps, but this lacks structural inductive bias for discrete compositional operations, leading to poor generalization on tasks requiring explicit multi-step logic. Specifically, AMVL's continuous latent space cannot represent hierarchical task decomposition or conditional branching, causing it to fail on compositional reasoning where the same reasoning structure must be shared across modalities.

## Key Insight

Reasoning inherently involves discrete compositional operations, and representing them as tree-structured latent variables allows explicit modeling of reasoning steps while cross-modal tree alignment ensures the same structure is shared across modalities, directly overcoming the structural weakness of continuous latent spaces.

## Method

**DHLP: Discrete Hierarchical Latent Program**

(A) **What it is**: DHLP is a variational framework that introduces a latent grammar with tree-structured variables to represent reasoning steps. It takes multimodal inputs (e.g., image and text) and outputs a tree-structured latent program that is shared across modalities via cross-modal tree alignment.

(B) **How it works** (pseudocode):
```python
# Training procedure for one batch
for each multimodal pair (x_v, x_t) with answer y:
    # Step 1: Encode each modality to initial features
    h_v = Encoder_v(x_v)   # image encoder: ResNet-18, output dim 512
    h_t = Encoder_t(x_t)   # text encoder: BERT-base, output dim 768

    # Step 2: Generate latent parse tree for each modality using neural PCFG
    # Grammar G: nonterminals N = {S, COMPARE, SELECT, COUNT, AND}, terminals T = {obj, attr, rel, num, bool}
    # Generative process: sample production rules top-down using learned parameters (rule probabilities from MLP)
    tree_v = sample_tree(G, h_v)   # shape: tree with node embeddings (dim 256)
    tree_t = sample_tree(G, h_t)

    # Step 3: Compute tree-level representations (root embedding via recursive composition using TreeLSTM, hidden dim 256)
    z_v = tree_root_embedding(tree_v)  # dim 256
    z_t = tree_root_embedding(tree_t)  # dim 256

    # Step 4: Cross-modal tree alignment via tree kernel similarity
    # Tree kernel K(tree_v, tree_t) measures subtree similarity using subtree kernel (STK) with decay factor 0.4
    L_align = -log( K(tree_v, tree_t) )   # regularization (weight λ=0.1)
    # Monitor alignment quality: every 500 iterations, compute normalized tree edit distance on a validation batch of 32 examples; target < 0.3

    # Step 5: Variational objective (ELBO)
    # Reconstruction of answer from either modality's root embedding (additive combination)
    p_y_given_z = Decoder(z_v + z_t)   # MLP: 256 -> 128 -> num_classes
    L_recon = -log p(y | z_v + z_t)

    # Variational posterior: q(tree | x_v, x_t) is parameterized by separate encoders for each modality, approximated with Gumbel-Softmax (temperature τ=1.0, annealed to 0.1)
    # Prior: p(tree) is uniform over trees up to depth 5 under given grammar
    L_KL = KL_divergence( q(tree|h_v,h_t) || p(tree) )   # computed as expected log ratio using Monte Carlo with 1 sample

    # Full objective: L = L_recon + λ_align * L_align + β * L_KL   (β=0.01)
    loss = L_recon + 0.1 * L_align + 0.01 * L_KL
    backpropagate(loss)
```

(C) **Why this design**: We chose a probabilistic context-free grammar (PCFG) over a general sequential generator because PCFG naturally enforces hierarchical structure and compositionality; the cost is that the grammar must be predefined or learned separately, which may miss domain-specific operations. Cross-modal alignment via tree kernel (subtree similarity) was selected over exact tree edit distance because it is differentiable and computationally efficient, though it may not capture global structural consistency as precisely. We opted to combine representations from both modalities before decoding (z_v + z_t) rather than decoding separately and ensembling, because joint decoding encourages sharing of complementary information; the trade-off is that it may blur modality-specific details when one modality is more reliable. The hyperparameter λ=0.1 balances alignment strength: too high forces unnatural tree structures, too low loses cross-modal consistency. We chose β=0.01 for the KL term to prevent posterior collapse while allowing some regularization, a common practice in VAE literature.

(D) **Why it measures what we claim**: The tree-sampling procedure (Step 2) operationalizes **compositional reasoning** as a discrete sequence of production rules applied top-down; this captures reasoning steps explicitly because each production corresponds to an atomic operation (e.g., COMPARE, SELECT), and the tree structure encodes their hierarchical composition. The tree kernel K(tree_v, tree_t) with decay factor 0.4 measures **subtree overlap**; we assume that identical subtree patterns correspond to identical reasoning operations. This assumption fails when modalities use different operations for the same subtask (e.g., image uses spatial compare while text uses attribute compare), in which case L_align forces false similarity and may degrade performance. To verify alignment quality, we monitor normalized tree edit distance on a validation set (target < 0.3). The reconstruction loss L_recon measures **task accuracy** as the negative log-likelihood of the correct answer given the combined latent tree, ensuring that the program is directly tied to the reasoning outcome. The KL divergence measures **posterior regularization** relative to a uniform prior over trees up to depth 5, acting as a complexity penalty; this assumes that simpler programs are more generalizable, but when the true reasoning requires many steps, the uniform prior may overly penalize complex trees.

## Contribution

(1) A novel variational framework (DHLP) that introduces hierarchical discrete latent programs with tree-structured variables for multimodal reasoning, directly addressing the lack of compositional inductive bias in continuous latent models. (2) A cross-modal tree alignment objective that enforces structural consistency of reasoning steps across modalities, improving robustness and interpretability. (3) Empirical demonstration that modeling discrete reasoning steps via latent programs outperforms continuous latent representations on compositional reasoning benchmarks (e.g., multimodal arithmetic, step-by-step visual QA).

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|-----------------------|
| Primary dataset | CLEVR | Compositional VQA with explicit reasoning steps. |
| Secondary dataset | NLVR2 (hard subset: 1000 examples) | Realistic multimodal reasoning with compositional hard examples. |
| Primary metric | Accuracy | Measures exact answer correctness. |
| Secondary metric | Normalized tree edit distance (validation) | Verifies alignment quality (target < 0.3). |
| Baseline 1 | AMVL | Continuous latent reasoning without discrete structure. |
| Baseline 2 | MCoT | Chain-of-thought in latent space. |
| Ablation | DHLP w/o alignment | Removes cross-modal tree alignment loss. |

### Why this setup validates the claim

CLEVR requires explicit multi-step reasoning with compositional operations, directly testing the claim that discrete hierarchical programs improve structure. Comparing against AMVL (continuous latent) and MCoT (continuous chain) isolates the benefit of discrete trees. The ablation tests the necessity of cross-modal alignment. Accuracy on CLEVR's compositional questions (e.g., counting, comparison) is a direct measure of reasoning correctness, making the setup a falsifiable test: if DHLP outperforms baselines on complex queries, the claim is supported; if not, the mechanism is flawed. Secondary NLVR2 evaluation assesses generalization to more realistic data. Additionally, computing normalized tree edit distance on a validation set provides a direct check of whether tree kernel similarity indeed reflects structural alignment.

### Expected outcome and causal chain

**vs. AMVL** — On a case requiring multi-step attribute comparison (e.g., "are there the same number of blue cubes as red spheres?"), AMVL encodes continuous latent variables that may mix attributes, leading to incorrect reasoning because it lacks explicit tree structure to separate steps. Our method samples a parse tree with nodes like COMPARE and SELECT, disentangling operations, so we expect higher accuracy on compositional questions, especially those requiring >3 steps, with a noticeable gap (e.g., at least 12% improvement on CLEVR queries requiring ≥4 reasoning steps; on NLVR2, we expect at least 8% improvement).

**vs. MCoT** — On a case where visual and textual cues require different reasoning orders (e.g., image requires left-to-right scanning while text describes properties), MCoT generates a chain that is modality-specific, causing misalignment. Our method aligns trees via kernel, enforcing structural consistency, so we expect better performance on cross-modal reasoning tasks where order matters, with observed gain on subsets requiring alignment (e.g., >5% accuracy improvement on NLVR2 examples where modality reasoning order differs).

**vs. ablation (DHLP w/o alignment)** — On a case where the same reasoning structure must be applied across modalities (e.g., comparing sizes in image and text), the ablation lacks alignment loss, so trees may diverge, leading to inconsistent latent programs and lower accuracy. Our full method enforces alignment, so we expect the ablation to perform worse on cross-modal examples (e.g., >8% drop on NLVR2 hard subset), confirming the importance of structural consistency.

### What would falsify this idea

If DHLP performs comparably to AMVL and MCoT on compositional questions, or if the ablation (w/o alignment) matches the full method on cross-modal examples, then the central claims of discrete hierarchical structure and alignment are not supported. Additionally, if the normalized tree edit distance on validation exceeds 0.3, it suggests that tree kernel similarity does not guarantee structural equivalence, undermining the alignment mechanism.

## References

1. Multimodal Continuous Reasoning via Asymmetric Mutual Variational Learning
2. Multimodal Chain of Continuous Thought for Latent-Space Reasoning in Vision-Language Models
3. Variational Reasoning for Language Models
4. RAVR: Reference-Answer-guided Variational Reasoning for Large Language Models
5. Training Large Language Models to Reason in a Continuous Latent Space
6. STaR: Bootstrapping Reasoning With Reasoning
7. Guiding Language Model Reasoning with Planning Tokens
8. Think before you speak: Training Language Models With Pause Tokens
