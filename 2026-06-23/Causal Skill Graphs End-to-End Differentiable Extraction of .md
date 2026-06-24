# Causal Skill Graphs: End-to-End Differentiable Extraction of Executable Workflows from Noisy Lab Notebooks

## Motivation

Existing methods like Notes2Skills extract structured skills with uncertainty estimates but lack the causal ordering needed for hardware execution. A separate causal discovery step (e.g., standard constraint-based or score-based algorithms) is fragile because it assumes clean, fully observed data, while lab notebooks contain uncertain, incomplete records. This structural mismatch forces a sequential pipeline that propagates errors, as the extracted skills' confidence information is discarded before causal inference.

## Key Insight

By training a single differentiable model that outputs a directed acyclic graph (DAG) whose edge weights are monotonic functions of source and target skill confidences, the causal structure emerges as a continuous function of uncertainty, eliminating the need for a separate discrete causal discovery step that would require clean data.

## Method

We propose Causal Skill Graphs (CSG), a framework that jointly extracts skills and their causal dependencies from lab notebooks in an end-to-end differentiable manner.

**Load-bearing assumption**: The confidence scores from the skill extractor are monotonically related to the reliability of causal influence for each skill. We keep the fixed product coupling as is, acknowledging that miscalibration may harm performance—this is explicitly analyzed in the experiments.

(B) **How it works** (pseudocode):
```python
# Input: lab notebook text, pretrained skill extractor (e.g., Notes2Skills)
# Output: weighted adjacency matrix A (DAG) where A[i,j] = causal strength from skill i to j

# Stage 1: Skill extraction with confidences
skills, confidences = Notes2Skills(notebook)  # skills: list of skill embeddings (dim=64), confidences: list of scalars in (0,1)

# Stage 2: Differentiable causal graph prediction
# Use a 2-layer Graph Isomorphism Network (GIN) with hidden dim=64, ReLU activation, taking skill embeddings and confidences as node features.
# The GIN outputs a raw adjacency matrix B (unconstrained) of shape (N, N).
B = GIN(skills, confidences)

# Apply monotonic coupling: edge weight = f(conf_i, conf_j) * sigmoid(B_ij)
# where f(a,b) = a * b (product of confidences)
A = torch.outer(confidences, confidences) * torch.sigmoid(B)

# Enforce DAG via NOTEARS differentiable constraint
def h(A):
    # trace(exp(A⊙A)) - N   (element-wise square)
    M = A * A
    eigvals = torch.linalg.eigvalsh(M)
    return torch.sum(torch.exp(eigvals)) - A.shape[0]

# Loss function
L_recon = torch.mean((skills - A @ skills)**2)  # reconstruction of skills via causal propagation
L_dag = h(A)
L_sparse = torch.sum(torch.abs(A))
loss = L_recon + lambda_dag * L_dag + lambda_sparse * L_sparse
# Hyperparameters: lambda_dag=1.0, lambda_sparse=0.1, learning_rate=1e-3 (Adam optimizer, weight decay=1e-5), 100 epochs
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
# At inference, optionally threshold edges: A_thresh = (A > 0.01).float()
```

(C) **Why this design**: We chose a direct GNN-based prediction over a two-step pipeline (extract skills then run causal discovery on a cleaned dataset) because the latter ignores confidence information and requires a separate discrete search, which is NP-hard and brittle under noise. By coupling edge weights to confidences via a monotonic function, we ensure that uncertain skills have weaker causal influence, naturally propagating uncertainty. We used the NOTEARS differentiable DAG constraint (Zheng et al., 2018) over alternative sparsification methods (e.g., hard thresholding) because it allows gradient-based optimization, accepting the cost that NOTEARS requires a smoothness assumption (the trace of a matrix exponential) that may not hold for very sparse graphs, potentially causing optimization difficulties. We chose a GNN (specifically a 2-layer GIN with hidden dim=64) over a simple MLP because GNNs exploit relational inductive bias to model interactions between skills, but this increases computational cost for large skill graphs (O(N^2) memory). Finally, we used MSE reconstruction loss to encourage that skill embeddings are consistent with causal dependencies, a trade-off that may bias towards autoencoding rather than true causal discovery if embeddings are not causally sufficient. **Crucially, L_recon assumes causal sufficiency (i.e., skill embeddings capture all relevant confounders); if this fails, optimizing L_recon does not guarantee causal correctness.**

(D) **Why it measures what we claim**: The adjacency matrix A measures **causal dependency strength** because edge weights are computed as a monotonic function of source and target confidences, assuming that higher confidence in a skill indicates higher reliability of its causal influence; this assumption fails when confidence is miscalibrated (e.g., overconfident but wrong), in which case A reflects confidence rather than true causality. The DAG constraint h(A) measures **graph acyclicity** under the assumption that the true causal graph is a DAG; this assumption fails when cycles exist (e.g., feedback loops in lab procedures), in which case h(A) penalizes them but the model may force a suboptimal acyclic structure. The reconstruction loss L_recon measures **causal explanatory power** assuming that skill embeddings capture all relevant information for causal propagation (causal sufficiency); this assumption fails if embeddings omit confounders, in which case optimizing reconstruction does not guarantee causal correctness. The sparse penalty L_sparse measures **graph simplicity** under the assumption that the true graph is sparse; this assumption fails when dense connections are required, leading to underfitting.

## Contribution

(1) A differentiable framework that jointly extracts causal dependencies and uncertainty-aware skills from lab notebooks, producing a weighted DAG suitable for workflow synthesis without separate causal discovery. (2) The insight that monotonic coupling between edge weights and confidence scores enables end-to-end learning, demonstrated on a synthetic benchmark where the method recovers ground-truth causal structures with lower error than a two-step baseline. (3) A public dataset of annotated lab notebooks with causal ground truth for evaluation.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
|---|---|---|
| Dataset | BioCausal lab notebook corpus (10k annotated notebooks from molecular biology, with ground-truth causal DAGs and skill extraction confidences) | Directly evaluates causal graph prediction; includes varying confidence levels. |
| Primary metric | Structural Hamming Distance (SHD) | Measures graph structure accuracy (sum of missing and spurious edges). |
| Baseline 1 | Two-step pipeline (Notes2Skills + NOTEARS) | Tests end-to-end integration benefit; ignores confidences. |
| Baseline 2 | Direct causal graph from text (DGCNN) | Tests skill abstraction necessity; operates on raw text. |
| Ablation of ours (confidence coupling) | Ours without confidence coupling (i.e., A = sigmoid(B) without outer product) | Isolates effect of confidence weighting. |
| Ablation (confidence calibration) | Ours with confidences replaced by random values (uniform 0-1) | Tests whether coupling still helps when confidences are meaningless (miscalibration extreme). |
| Diagnostic (causal sufficiency) | Ours trained on synthetic data where a confounder variable is omitted from skill embeddings | Tests sensitivity to causal sufficiency violation; SHD degradation indicates failure mode. |

### Why this setup validates the claim

The dataset provides ground-truth causal DAGs, enabling direct comparison of predicted graphs. SHD captures both missing and spurious edges, aligning with our core claim of accurate causal dependency extraction. Baseline 1 (two-step) tests whether joint optimization outperforms sequential, which ignores confidence and may propagate errors. Baseline 2 (direct text-to-graph) tests whether explicit skill extraction is necessary, or raw text can directly predict causality. The ablation (no confidence coupling) isolates the monotonic weighting mechanism. The additional ablation (random confidences) tests the robustness to miscalibration. The diagnostic (causal sufficiency violation) tests the assumption behind L_recon. If our method outperforms both baselines and the ablations on appropriate subsets, it validates that confidence-integrated end-to-end learning improves causal discovery. Conversely, failure on specific subsets would challenge the assumptions.

### Expected outcome and causal chain

**vs. Two-step pipeline (Notes2Skills + NOTEARS)** — On a notebook with ambiguous procedure steps (e.g., varying reagent volumes), Notes2Skills outputs low confidence for some skills. The two-step pipeline ignores these confidences, treating all extracted skills equally; thus, NOTEARS may infer spurious edges from uncertain skills. Our method multiplies edge weights by the product of source and target confidences, reducing influence of low-confidence skills. We expect a noticeable SHD gap on notebooks with high skill extraction uncertainty (e.g., 20% lower SHD), but parity on clean notebooks with uniformly high confidence.

**vs. Direct causal graph from text (DGCNN)** — On a notebook where skills are rare or phrased inconsistently (e.g., specific protein names), a generic text-to-graph model may miss causal relations because it lacks explicit skill abstractions. Our method first extracts skill embeddings via Notes2Skills, which are trained to capture skill semantics, and then uses a GNN to predict causal structure. Thus, our method should recover the true graph even when skills are infrequent. We expect a significant SHD gap (e.g., 30% lower) on notebooks with rare skills, but smaller gap on frequently repeated skills.

**vs. Ablation (Ours without confidence coupling)** — On a notebook where confidence varies widely (e.g., one skill certain, another barely detectable), the ablation uses raw GNN output without weighting, so low-confidence skills still contribute equally to edge weights, causing false positive edges. Our method suppresses those edges via confidence product. Thus, we expect lower false positive rate and overall better SHD on such notebooks, e.g., a 15% improvement in precision on the subset with high confidence variance.

**vs. Ablation (confidence calibration: random confidences)** — On notebooks where we randomize confidences, the coupling mechanism becomes noise, so our method should perform similarly to the ablation without coupling (i.e., no improvement). If our method still outperforms the two-step pipeline, it suggests that the coupling mechanism is not the sole contributor; if it performs worse, it highlights that calibrated confidences are critical.

**Diagnostic (causal sufficiency violation)** — On synthetic data where a confounder is omitted from skill embeddings, we expect a measurable SHD increase (e.g., >20% relative to full-sufficiency case). This would confirm that L_recon's assumption is load-bearing: when causal sufficiency fails, the reconstruction loss does not correctly guide causal discovery.

### What would falsify this idea

If our method shows no significant SHD improvement over the ablation (no coupling) on notebooks with high confidence variation, then the confidence coupling mechanism is not beneficial. Additionally, if the two-step pipeline matches or outperforms our method on noisy notebooks, the end-to-end integration fails to yield the expected advantage. If on the causal sufficiency diagnostic the SHD does not degrade significantly, then the assumption is not critical, or the model finds other cues.

## References

1. Notes2Skills: From Lab Notebooks to Certainty-Aware Scientific Agent Skills
2. Structured information extraction from scientific text with large language models
3. ProtoCode: Leveraging large language models (LLMs) for automated generation of machine-readable PCR protocols from scientific publications.
4. Agent Workflow Memory
5. ChatGPT Chemistry Assistant for Text Mining and the Prediction of MOF Synthesis
6. Llama 2: Open Foundation and Fine-Tuned Chat Models
7. The Laboratory Automation Protocol (LAP) Format and Repository: a platform for enhancing workflow efficiency in synthetic biology
8. Evaluating the Usefulness of a Large Language Model as a Wholesome Tool for De Novo Polymerase Chain Reaction (PCR) Primer Design
