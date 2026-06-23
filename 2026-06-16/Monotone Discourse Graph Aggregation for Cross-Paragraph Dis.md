# Monotone Discourse Graph Aggregation for Cross-Paragraph Disinformation Detection

## Motivation

Existing multi-agent disinformation detection systems, such as the MCP-Orchestrated system, evaluate only on short texts and treat each sentence independently, failing to aggregate evidence across paragraphs. This structural limitation prevents them from detecting long-range manipulation strategies that exploit cross-paragraph discourse cues, such as gradual introduction of false premises or subtle contradictions across sections.

## Key Insight

Monotone propagation over discourse graphs ensures that evidence reinforcement is consistent with discourse relations, preventing contradictory or unsupported claims from dominating the verdict.

## Method

**(A) What it is:** DISC-MONO (Discourse Monotone Aggregation) is a multi-agent framework that constructs a discourse graph from a long-form article, initializes node scores using specialized agents, and propagates scores monotonically along discourse edges to produce a final veracity score. Input: long-form article. Output: article-level veracity score.

**(B) How it works:**
```python
# Phase 1: Discourse Graph Construction
# Using PDTB-style parser (e.g., from Penn Discourse Treebank) with discourse relations: Elaboration, Contrast, Cause, Temporal, etc.
G = discourse_parse(article)  # nodes=paragraphs/sentences, edges=discourse relations (e.g., Elaboration, Contrast)

# Phase 2: Initial Agent Scores
agents = [ClaimDecompositionAgent, EvidenceRetrievalAgent, ConsistencyAgent]
# Each agent is a pre-trained LLM (e.g., GPT-3.5-turbo) fine-tuned for its task.
# Misclassification rates are estimated on a held-out calibration set of 512 articles.
for node in G.nodes:
    node.init_score = weighted_average([agent(node.text) for agent in agents], weights=[1/misclassification_rate])

# Phase 3: Monotone Propagation (synchronous updates until convergence)
def monotone_transfer(relation_type, score):
    if relation_type == 'Elaboration': return min(1, 1.2*score)  # slight increase
    elif relation_type == 'Contrast': return 0.5*score  # attenuation
    else: return score  # identity for others

while not converged:
    new_scores = {}
    for node in G.nodes:
        incoming = [monotone_transfer(edge.type, G.nodes[edge.source].score) for edge in G.in_edges(node)]
        new_scores[node] = max(incoming) if incoming else node.init_score
    # Update scores
    for node in G.nodes:
        node.score = new_scores[node]
    converged = all(abs(new_scores[n] - node.score) < 1e-6 for n in G.nodes)

# Phase 4: Article-Level Aggregation
article_score = sum(node.score * node.centrality for node in G.nodes) / sum(node.centrality)
return 'Real' if article_score > 0.5 else 'Fake'
```
Hyperparameters (1.2 for Elaboration, 0.5 for Contrast) are chosen via pilot experiments on a validation set of 100 articles (varying factors in {1.1,1.2,1.3} and {0.4,0.5,0.6} maximizing F1). The discourse parser is the publicly available PDTB-style parser (https://github.com/...), which outputs relation types and edge weights (not used).

**(C) Why this design:** We chose monotone transfer functions over additive propagation because monotonicity ensures that scores only increase or decrease along discourse relations in a consistent direction, mimicking the idea that elaboration supports truth and contrast weakens it; this structural property guarantees convergence without requiring training. We chose discourse-specific factors (e.g., 1.2 for Elaboration, 0.5 for Contrast) over learned weights because they are interpretable and data-efficient, at the cost of being less adaptive to domain-specific discourse patterns. We chose the max aggregation over incoming edges (instead of sum or average) because it prioritizes the strongest evidence, reflecting the intuition that a single strong supporting source can outweigh multiple weak contradictors; this trade-off sacrifices sensitivity to weak cumulative evidence for robustness against noise from irrelevant edges. Unlike the MCP-Orchestrated system which treats each sentence independently, our method explicitly models cross-paragraph relationships to capture long-range disinformation tactics.

**(D) Why it measures what we claim:** The computational quantity `node.score` after propagation measures "discourse-supported veracity" because we assume that if a sentence is consistently elaborated by other sentences (via Elaboration edges), its veracity should be reinforced; this assumption fails when the elaboration itself is deceptive (e.g., false supporting details), in which case the score reflects the deceptive support strength. The `monotone_transfer` function per relation type measures "typical evidentiary impact of that discourse relation" because we assume that relation types have consistent effects (e.g., Contrast usually weakens evidence); this assumption fails when the discourse parser misclassifies the relation, in which case the transfer function misapplies. Specifically, `monotone_transfer('Elaboration', score) = min(1, 1.2*score)` measures 'support strength' under the assumption that Elaboration relations always increase the likelihood of truth; this assumption fails when Elaboration provides false supporting details, in which case the score reflects the strength of deceptive support. Similarly, `monotone_transfer('Contrast', score) = 0.5*score` measures 'discounting effect' under the assumption that Contrast always weakens truth; failure occurs when Contrast is used to dismiss true evidence. The initial `node.init_score` from agents measures "local reliability of evidence from that paragraph" because we assume each agent's misclassification rate is a good proxy for its accuracy on this input; this assumption fails on out-of-distribution texts, in which case the weighted average retains the agent's bias.

## Contribution

(1) A novel framework (DISC-MONO) that extends multi-agent disinformation detection to long-form articles by incorporating monotone discourse graph propagation, enabling cross-paragraph evidence aggregation. (2) A design principle: discourse-aware monotone aggregation using relation-specific transfer functions preserves interpretability and guarantees convergence, while avoiding reliance on learned gating mechanisms. (3) A parameterized monotone transfer function schema that can be tuned for different discourse parsers and disinformation styles.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|-----------------------|
| Dataset | FakeNewsNet (politifact + buzzfeed subsets) and LIAR | Two diverse datasets; long and short texts. |
| Primary metric | F1-score | Balances precision and recall for imbalanced classes. |
| Baseline 1 | MCP-Orchestrated multi-agent system | Each sentence independent; no discourse links. |
| Baseline 2 | Single-pass LLM (ChatGPT) | Direct prompting, no decomposition or propagation. |
| Baseline 3 | BERT-based classifier | Contextual embeddings without explicit discourse. |
| Baseline 4 | GNN with learnable edge weights | Learns propagation weights; tests monotonicity contribution. |
| Ablation 1 | DISC-MONO w/o propagation | Only initial agent scores, no monotone transfer. |
| Ablation 2 | DISC-MONO w/ fixed random discourse labels | Tests sensitivity to parser errors. |

### Why this setup validates the claim

The central claim is that modeling discourse relations (Elaboration, Contrast) and applying monotone propagation improves veracity detection over methods that ignore such structure. The datasets FakeNewsNet (long-form articles with multiple paragraphs) and LIAR (short statements) provide diverse lengths and domains, enabling evaluation of cross-paragraph discourse modeling and generalization. The primary metric F1 appropriately measures performance on imbalanced classes. Baselines test distinct deficiencies: MCP-Orchestrated lacks cross-sentence aggregation; single-pass LLM lacks structured decomposition; BERT lacks explicit discourse; GNN with learnable weights isolates the benefit of monotonicity. Ablation 1 isolates the contribution of propagation; Ablation 2 tests robustness to parser errors. If DISC-MONO outperforms on articles with clear discourse patterns and the gain correlates with discourse density, the claim is supported.

### Expected outcome and causal chain

**vs. MCP-Orchestrated** — On an article where a false claim is reinforced by elaborative sentences, MCP-Orchestrated treats each sentence independently and may miss the cumulative support, leading to false positive. Our method aggregates discourse support via Elaboration edges, so such reinforcement increases veracity score correctly. We expect a noticeable gap on articles with strong intra-document elaboration, but parity on independent claims.

**vs. Single-pass LLM** — When a disinformation article uses subtle contradictions to hide falsehoods, an LLM may overlook the contrast structure and be misled by superficial coherence of individual sentences. Our method detects Contrast edges and attenuates scores, so contradictory evidence reduces veracity. We expect our gain concentrated on articles containing explicit contrary relations, e.g., "However," "But."

**vs. BERT-based classifier** — BERT's attention captures local context but lacks explicit discourse parsing, so on articles where falsehood is supported by a chain of elaborations spanning paragraphs, BERT may fail to propagate support across long distances. Our method uses discourse graph and monotone transfer to propagate evidence along edges. We expect larger improvement on articles with long-range discourse dependencies vs. short texts.

**vs. GNN with learnable edge weights** — If monotonicity is the key, our method should achieve comparable or better F1 while being simpler and more interpretable. On LIAR (short texts), the GNN may overfit to spurious patterns; our method's fixed transfer functions may be more robust. On FakeNewsNet, we expect our method to be competitive, especially when discourse relations are clear.

**vs. Ablation (w/o propagation)** — On an article where a true claim is initially scored low by agents due to noise, the ablation cannot recover; our propagation uses Elaboration from other nodes to increase its score. Hence our method should outperform ablation specifically on articles where initial scores are unreliable but discourse structure provides supporting evidence. We anticipate ablation's F1 lower on multi-paragraph articles.

### What would falsify this idea

If our method's performance gain over baselines is uncorrelated with the density of discourse relations (Elaboration and Contrast) in the articles, then the discourse modeling is not the source of improvement. Specifically, if the gain is uniform across all articles regardless of discourse structure, the claim is falsified. Additionally, if on LIAR (short texts) our method performs no better than BERT, then the discourse graph does not provide benefit for short texts, suggesting the method is specific to long-form articles.

## References

1. MCP-Orchestrated Multi-Agent System for Automated Disinformation Detection
2. Large Language Model Agentic Approach to Fact Checking and Fake News Detection
3. Stylometric Fake News Detection Based on Natural Language Processing Using Named Entity Recognition: In-Domain and Cross-Domain Analysis
4. Fake News Classification Based on Content Level Features
5. A Comparative Study of Machine Learning and Deep Learning Techniques for Fake News Detection
6. Context-Based Fake News Detection Model Relying on Deep Learning Models
7. Towards LLM-based Fact Verification on News Claims with a Hierarchical Step-by-Step Prompting Method
8. Encouraging Divergent Thinking in Large Language Models through Multi-Agent Debate
