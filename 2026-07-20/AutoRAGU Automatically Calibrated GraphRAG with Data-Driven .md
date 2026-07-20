# AutoRAGU: Automatically Calibrated GraphRAG with Data-Driven Hyperparameter Selection

## Motivation

Existing GraphRAG methods like RAGU require manual tuning of hyperparameters for graph construction (e.g., DBSCAN epsilon, Leiden resolution), which are dataset-dependent and lack robustness. The root cause is that these hyperparameters are fixed across datasets, but entity density and semantic distinctiveness vary widely, forcing practitioners to hand-tune per dataset, hindering generalization. This limitation is explicitly inherited from the ancestor RAGU method, where DBSCAN epsilon and community-detection resolution are set manually.

## Key Insight

Optimal graph construction hyperparameters are a function of the local density and semantic heterogeneity of the entity embedding space, which can be estimated from the extracted entity set prior to graph building through curvature and silhouette-based heuristics.

## Method

### AutoRAGU: Automatically Calibrated GraphRAG with Data-Driven Hyperparameter Selection

**Method:** AutoRAGU extends RAGU with a Data-Driven Calibrator (DDC) that takes the raw entity set (with embeddings) and relation set from the two-stage extraction and outputs domain-adaptive hyperparameters (epsilon, min_samples for DBSCAN; gamma resolution for Leiden), removing manual tuning.

**Pipeline (pseudocode):**
```pseudocode
# AutoRAGU pipeline
1. Two-stage extraction (same as RAGU): extract typed entities and relations.
2. Compute entity embedding matrix E from a frozen sentence transformer (all-MiniLM-L6-v2).
3. Data-Driven Calibrator (DDC):
   a. Compute pairwise cosine distances D = 1 - cos_sim(E) → upper triangle.
   b. Estimate noise level: silhouette score of k-means with k=2 on E (using sklearn.KMeans, n_init=10, random_state=42).
      if silhouette < 0.3: set minPts = 5 else minPts = 3.
   c. Compute epsilon using k-distance graph: for each entity, distance to its (k=minPts)-th nearest neighbor; sort distances.
      Choose epsilon at elbow via maximum curvature: compute second derivative via central differences, set threshold = 0.1 * max(second derivative), select first point where curvature > threshold; if none, use median distance.
   d. Run DBSCAN with (epsilon, minPts) to get entity clusters.
   e. For relation graph, compute edge weights (e.g., co-occurrence frequency).
   f. Compute resolution gamma = 1 - (average within-cluster edge weight / average all edge weight).
4. Run DBSCAN deduplication with calibrated epsilon, minPts.
5. Run LLM summarization (as in RAGU).
6. Run Leiden community detection with calibrated gamma.
7. Build retrieval index.
```

(C) **Why this design** — We chose a heuristic elbow-point method for epsilon estimation over a learned regression model because it requires no training data and generalizes across domains, accepting the cost that it may be brittle for multi-density distributions where the elbow is ambiguous. The silhouette-based minPts adjustment was selected over a fixed default because it adapts to dataset noise level, at the risk of being misled by non-spherical clusters that inflate silhouette scores. The gamma derivation from edge-weight contrast was preferred over a learned predictor because it is computationally efficient (O(E) for edge list E) and domain-agnostic, though it may not capture nuanced community structures that require higher-order statistics (e.g., motif counts). Finally, we used a frozen sentence transformer for entity embeddings rather than fine-tuned ones to avoid additional training, accepting possibly suboptimal embeddings for domain-specific entities. These decisions trade off optimality for generality and simplicity, ensuring the calibrator works out-of-the-box on new datasets.

(D) **Why it measures what we claim** — The elbow-point of the k-distance graph (step 3c) measures the *density threshold* at which entities transition from core to outlier, which approximates the *optimal DBSCAN epsilon* under the assumption that the embedding space has roughly uniform density within clusters; this assumption fails when clusters have varying densities (the elbow becomes ambiguous), and in that case the maximum-curvature heuristic selects a conservative epsilon that may under-split clusters. The silhouette-based minPts adjustment (step 3b) measures *cluster coherence* via a two-cluster partition, operationalizing the *noise level* of the dataset under the assumption that the silhouette score of a k-means(2) partition reflects overall cluster tendency; this assumption fails for highly non-spherical clusters (e.g., chain-like), where silhouettes are inflated, leading to underestimation of minPts and over-fragmentation of entities. The gamma heuristic (step 3f) measures the *edge strength contrast* between within-community and all edges, which operationalizes the *desired community granularity* under the assumption that communities correspond to dense subgraphs with high internal edge weights; this assumption fails for graphs with uniform edge weights, where gamma defaults to 0.5, producing no preference and leaving resolution to Leiden's default.

**Implementation details:** DDC implemented in ~150 lines of Python. Expected runtime per dataset (10k entities) is 2 seconds for k-distance computation + 0.5 sec for elbow detection on a single CPU core.

## Contribution

(1) A data-driven hyperparameter calibrator (DDC) that replaces manual tuning in the RAGU pipeline, using only entity embeddings and relation edges to set DBSCAN and Leiden parameters. (2) Empirical demonstration that AutoRAGU achieves comparable or better evidence recall across multiple datasets (e.g., Medical, Wiki, Legal) without per-dataset hyperparameter tuning, reducing sensitivity. (3) Open-source release of the calibration module as an extension to RAGU.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
|------|--------|-----------|
| Dataset | 2WikiMultihopQA + SyntheticDensity (synthetic with known optimal epsilon=0.3, minPts=4, gamma=0.6) | Multi-hop QA tests graph retrieval; synthetic validates calibration recovery |
| Primary metric | Answer F1 (on 2WikiMultihopQA) + Clustering F1 (Adjusted Rand Index on synthetic) | End-to-end answer quality + clustering accuracy |
| Baseline 1 | RAGU (manual tuning: epsilon=0.5, minPts=5, gamma=1.0) | Direct predecessor with fixed params |
| Baseline 2 | Standard RAG | No graph structure, retrieval only |
| Baseline 3 | GraphRAG-Local (fixed resolution=1.0) | GraphRAG with local communities |
| Baseline 4 | RAGU + Bayesian optimization (BO over epsilon, minPts, gamma, 50 iterations) | Automated hyperparameter search baseline |
| Baseline 5 | RAGU + grid search (GS: epsilon in [0.1,0.9], minPts in [3,7], gamma in [0.5,1.5]) | Exhaustive search baseline |
| Ablation of ours | AutoRAGU w/o DDC (fixed hyperparams: epsilon=0.5, minPts=5, gamma=1.0) | Isolates DDC contribution |

### Why this setup validates the claim

This experimental design forms a falsifiable test of AutoRAGU's central claim – that data-driven calibration of DBSCAN and Leiden hyperparameters improves graph construction and downstream QA. By including SyntheticDensity with known optimal hyperparameters, we directly test whether the DDC recovers those values, isolating the novelty of calibration. The comparison against BO and GS baselines quantifies the practical significance of our heuristic approach relative to more expensive search methods. The 2WikiMultihopQA dataset focuses on multi-hop reasoning, where graph quality critically affects performance. The ablation with fixed hyperparameters measures the DDC contribution, while baselines RAGU and GraphRAG-Local provide direct and structural baselines. Using both clustering accuracy (on synthetic) and answer F1 (on real data) ensures that calibration success is not confounded by downstream components.

### Expected outcome and causal chain

**vs. RAGU (manual tuning)** — On 2WikiMultihopQA, we expect AutoRAGU to outperform RAGU by 5-10 F1 points on multi-hop questions, with gains concentrated on questions requiring entity deduplication. On SyntheticDensity, AutoRAGU should recover the true hyperparameters closely (ARI > 0.9), while RAGU's fixed params yield lower ARI (≈0.6). The causal chain: DCC adapts to dataset density, improving entity grouping and relation linking.

**vs. Standard RAG** — On multi-hop questions (2WikiMultihopQA), AutoRAGU is expected to beat Standard RAG by >15 F1 points due to its ability to traverse entity relations. On single-hop questions, the gap is minimal. DDC's calibration does not directly affect this comparison beyond enabling the graph structure.

**vs. GraphRAG-Local** — On datasets with variable community structure, AutoRAGU's adaptive gamma yields 3-5 F1 gain over fixed-resolution GraphRAG-Local. On uniform datasets, parity is expected. SyntheticDensity validates that gamma heuristic aligns with optimal community partition.

**vs. BO/GS baselines** — AutoRAGU should achieve at least 90% of the performance of BO/GS on 2WikiMultihopQA (within 2 F1 points), while being orders of magnitude faster (2.5 sec vs >1000 sec for BO). On SyntheticDensity, DDC should match or beat BO/GS in recovering ground truth, as its design is tailored to density-based clustering.

**Ablation (w/o DDC)** — AutoRAGU must outperform its fixed-parameter ablation on all datasets, with larger gaps on datasets where default parameters are suboptimal (e.g., SyntheticDensity).

### What would falsify this idea

If AutoRAGU fails to outperform the ablation on any dataset, or if its gains are uniform across question types rather than concentrated where density/noise variation occurs, then the central claim that data-driven calibration is beneficial would be falsified. On SyntheticDensity, failure to recover optimal hyperparameters (ARI < 0.7) would directly invalidate the DDC heuristic.

## References

1. RAGU: A Multi-Step GraphRAG Engine with a Compact Domain-Adapted LLM
2. From Local to Global: A Graph RAG Approach to Query-Focused Summarization
3. Knowledge Graph Prompting for Multi-Document Question Answering
4. Grape: Knowledge Graph Enhanced Passage Reader for Open-domain Question Answering
5. Visconde: Multi-document QA with GPT-3 and Neural Reranking
