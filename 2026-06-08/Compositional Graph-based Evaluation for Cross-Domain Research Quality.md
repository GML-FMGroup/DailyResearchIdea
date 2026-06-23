# Compositional Graph-based Evaluation for Cross-Domain Research Quality

## Motivation

Current benchmarks such as AI Idea Bench rely on domain-specific rubrics and reference-based semantic similarities that cannot transfer across scientific domains, leading to isolated benchmark ecosystems. The root cause is treating evaluation as matching content rather than assessing structural reasoning, which fails to capture the underlying conceptual dependencies that generalize across fields. For instance, AI Idea Bench evaluates idea quality by comparing to ground-truth content, which is inherently domain-bound.

## Key Insight

By decomposing research outputs into concept dependency subgraphs and scoring them with a graph isomorphism-invariant function, the evaluation becomes dependent only on relational structure, not domain vocabulary, enabling cross-domain transfer without retraining.

## Method

**A) What it is:** CGE (Compositional Graph-based Evaluation) is a two-stage pipeline that extracts a concept dependency graph from a research output and assigns a quality score by aggregating scores of its connected subgraphs via a domain-invariant Weisfeiler-Lehman kernel. This method operates under the explicit assumption that structural similarity (measured by WL kernel) correlates with expert-rated idea quality across domains, independent of domain-specific vocabulary.

**B) How it works:**
```pseudocode
Input: Research output text T, optional domain-specific knowledge graph K
1. Extract concept set V and relation set E using an LLM (e.g., GPT-4, gpt-4-0613, temperature=0.0) with prompt: "Identify key scientific concepts and their dependencies in the following text. Represent as list of (concept1, relation, concept2) triples."
2. Build directed graph G = (V, E). Handle coreference and synonym mapping using K if available, else use LLM to resolve; for synonym mapping, prompt: "Are the following concepts referring to the same scientific entity? Answer yes/no." Merge nodes if yes.
3. Decompose G into its biconnected components (subgraphs) S_1, ..., S_m using Hopcroft-Tarjan algorithm (O(|V|+|E|)).
4. For each subgraph S_i, compute a score via Weisfeiler-Lehman subtree kernel (h=3) that maps S_i to a real number based on its isomorphism class. The kernel's base distribution is learned from a precompiled corpus of 10,000 high-quality research graphs from diverse domains (physics, computer science, biomedicine; 3,333 from each) extracted from top-venue papers (e.g., Nature, NeurIPS, Physical Review Letters). The kernel uses a 3-dimensional feature map and sum aggregation.
5. Aggregate: final_score = mean of top-⌈m/2⌉ subgraph scores (to focus on strongest components). Hyperparameter: k = ⌈m/2⌉, in case m=0 set score=0.
Output: Quality score ∈ ℝ.
```
**C) Why this design:** We chose graph decomposition into biconnected components over a single graph embedding because it isolates local structural units that correspond to separable idea components, enhancing interpretability; the trade-off is that large graphs produce many subgraphs, increasing computational cost. We selected the Weisfeiler-Lehman (WL) kernel over a GNN with isomorphism constraints because the WL kernel guarantees invariance to graph structure without requiring domain-specific training data, whereas GNNs need retraining or fine-tuning across domains; the trade-off is that WL is less expressive than state-of-the-art GNNs for capturing high-order patterns. We opted for LLM-based extraction over rule-based parsing because LLMs generalize across scientific discourse without manual feature engineering; the trade-off is cost, latency, and potential hallucination. We chose mean-of-top-k aggregation over sum or max because it rewards ideas with multiple strong components while ignoring weak ones, balancing coverage and selectivity; the trade-off is that it may discard globally coherent but uniformly moderate ideas.

**D) Why it measures what we claim:** The WL kernel score on subgraph S_i measures structural soundness/complexity (the motivation-level concept of idea quality) because it assigns equal scores to isomorphic graphs, assuming research quality is a function of how concepts are interrelated rather than their specific labels; this assumption fails when a subgraph is structurally trivial but contains a groundbreaking concept, in which case the metric underestimates novelty. The biconnected decomposition measures modularity of the idea, assuming that high-quality research has well-separated conceptual submodules; this assumption fails when ideas are inherently intertwined across multiple biconnected components, leading to fragmented evaluation. The top-k mean aggregation measures the impact of the strongest conceptual chunks, assuming that a few excellent components compensate for weak ones; this assumption fails in domains where synergy across all modules is critical (e.g., tightly-integrated theoretical frameworks), in which case the metric overestimates overall quality by ignoring weak links.

## Contribution

(1) A novel compositional evaluation framework that decomposes research outputs into concept dependency subgraphs and scores them with a domain-invariant graph kernel. (2) The design principle that research evaluation can be made cross-domain transferable by leveraging graph isomorphism invariance. (3) A demonstration that the metric correlates with human judgment across multiple scientific domains (included in experimental section).

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | AI Idea Bench 2025 (1,000 ideas), plus human-annotated graphs for 200 ideas | Standardized LLM-generated ideas with ground-truth graphs for extraction accuracy |
| Primary metric | Spearman corr. with expert ratings (5-point scale, 3 experts per idea) | Measures alignment with human quality judgment |
| Baseline 1 | LLM-as-Judge (direct prompt: "Rate the quality of this research idea on a scale 1-5") | Tests need for structural graph extraction |
| Baseline 2 | Full-graph WL kernel (no decomposition) | Tests benefit of biconnected decomposition |
| Baseline 3 | Mean-of-all subgraphs aggregation | Tests top-k aggregation focus on strong components |
| Baseline 4 | CGE with WL kernel replaced by 2-layer GNN (hidden=128) trained on precompiled corpus | Tests benefit of invariance: GNN must generalize across domains without retraining |
| Ablation 1 | CGE w/o coreference resolution | Tests impact of synonym mapping |
| Ablation 2 | CGE with uniform kernel base distribution (no precompiled corpus) | Tests reliance on precompiled corpus |
| Ablation 3 | CGE on synthetic graphs with controlled structural properties (10,000 graphs, known quality scores from simulated expert ratings) | Tests calibration of kernel score with quality in a controlled setting |

### Why this setup validates the claim

This design tests the central hypothesis that structured graph decomposition and aggregation improve evaluation accuracy over holistic scoring. LLM-as-Judge checks whether pure LLM semantic understanding can match our graph-based reasoning; if CGE outperforms, it confirms that concept dependency structure matters. The full-graph baseline isolates the contribution of decomposition into biconnected components—if CGE wins, it shows that local structural units correlate better with quality. The mean-of-all aggregation baseline tests whether focusing on strong components (top-k) is beneficial versus averaging all. Including the GNN baseline tests the invariance assumption: if CGE (with WL) outperforms a GNN that must be trained and cannot transfer without retraining, it supports the domain-invariant claim. Ablations on coreference resolution and the precompiled corpus control for preprocessing and the learned base distribution. The synthetic graph ablation directly tests whether the kernel score correlates with quality under controlled conditions, bypassing extraction noise. The Spearman correlation with expert ratings directly targets the claim that our scores align with human judgment, making any deviation a falsifying signal.

### Expected outcome and causal chain

**vs. LLM-as-Judge** — On a case where an idea contains sound concepts but is poorly organized (e.g., disconnected claims), a direct LLM may produce a moderate-to-high score because it captures semantic quality but misses structural flaws. Our method instead detects weak connectivity through low subgraph scores after decomposition, leading to a lower overall score. We expect a notable gap on structurally flawed but semantically rich ideas, with our metric correlating higher with experts who penalize poor organization.

**vs. Full-graph WL kernel (no decomp.)** — On a case where an idea has one strong conceptual module embedded in noise, the full-graph kernel may average over all edges and produce a mediocre score. Our method isolates the strong biconnected component and gives it high weight via top-k aggregation, so we expect a higher quality estimate. The observable signal is a divergence on ideas with highly variable component quality, with our method showing stronger correlation with expert ratings.

**vs. Mean-of-all subgraphs aggregation** — On a case where an idea has many weak or trivial subgraphs (e.g., many single-concept components), mean aggregation pulls down the score. Our top-k mean ignores the weakest half, so it yields a higher score that matches expert focus on the most coherent parts. We expect our method to outperform specifically on ideas with many disconnected fragments, where experts allocate attention only to the best chunks.

**vs. GNN baseline** — On a cross-domain transfer task (e.g., training on physics graphs, testing on biology), the GNN must generalize without domain-specific retraining; we expect the WL kernel to achieve higher Spearman correlation because it is domain-invariant by design. The observable signal is a larger performance drop for GNN when moving across domains.

### What would falsify this idea

If CGE's Spearman correlation is not significantly higher than the best baseline across all subsets, or if its advantage is uniform across all idea types rather than concentrated on structurally fragmented ideas, the central claim that decomposition and top-k aggregation improve evaluation is falsified. Additionally, if the synthetic graph ablation shows no correlation between WL kernel scores and known quality scores, the fundamental assumption that structural similarity corresponds to quality is falsified.

## References

1. AI Idea Bench 2025: AI Research Idea Generation Benchmark
2. The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery
3. Hypothesis Generation with Large Language Models
4. Large Language Models for Automated Open-domain Scientific Hypotheses Discovery
5. Large Language Models are Zero Shot Hypothesis Proposers
6. SciMON: Scientific Inspiration Machines Optimized for Novelty
