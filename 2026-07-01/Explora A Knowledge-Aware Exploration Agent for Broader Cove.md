# Explora: A Knowledge-Aware Exploration Agent for Broader Coverage in Text-Rich Image Generation Data Construction

## Motivation

DataEvolver's iterative expansion remains confined to the seed distribution because the Generator synthesizes samples only in regions already hinted by the initial candidate pool, as evidenced by the Critic's feedback remaining within the same conceptual bounds. Adding an exploration agent that deliberately seeks underrepresented scenarios is necessary, but naive random perturbation risks generating irrelevant or redundant prompts. To break bounded search, we need a principled mechanism that identifies true coverage gaps and generates semantically novel prompts targeting those gaps.

## Key Insight

Coverage gaps can be operationalized as concept-level undersampling in the candidate pool, enabling a knowledge graph to systematically generate prompts that combine missing concepts with existing ones in semantically novel configurations, ensuring each new prompt is guaranteed to be distinct from current coverage.

## Method

### (A) What it is
We propose Explora, an exploration agent that takes the Critic's structured feedback and a domain-specific knowledge graph as input, and outputs a set of new prompts designed to fill underrepresented concept combinations. These prompts are appended to the candidate pool for the next round of DataEvolver.

### (B) How it works
```python
def explora(critic_feedback, knowledge_graph, candidate_pool, threshold=0.7, temp=1.0):
    gaps = parse_gaps(critic_feedback)  # e.g., ['digital screens', 'curved text']
    new_prompts = []
    for gap in gaps:
        # 1. Retrieve concepts from knowledge_graph related to gap (neighbors within depth=1)
        related = knowledge_graph.get_neighbors(gap, depth=1)
        # 2. Compute coverage score for each related concept: count of prompts containing that concept in candidate_pool
        for concept in related:
            coverage_score = sum(1 for p in candidate_pool if concept in p)
            if coverage_score < 2:  # undercovered threshold
                # 3. Generate a prompt by combining concept with gap using a pre-trained LLM (e.g., GPT-3)
                prompt = llm_generate(f"Generate an image description with text including '{concept}' in a '{gap}' style", temperature=temp)
                # 4. Filter by cosine similarity to existing prompts; threshold=0.7 (computed via sentence-BERT)
                if not any(cos_sim(prompt, ex) > threshold for ex in candidate_pool):
                    new_prompts.append(prompt)
    return new_prompts
```
Hyperparameters: threshold=0.7, temp=1.0, coverage threshold=2.

### (C) Why this design
We chose a domain-specific knowledge graph (constructed from TextCaps captions via co-occurrence mining with PMI > 0.5) over ConceptNet because it provides structured semantic relationships directly relevant to text-rich images, reducing noise from irrelevant commonsense concepts. We use an LLM for prompt generation (rather than fixed templates) to produce natural and diverse descriptions, accepting the computational cost and potential hallucination. The similarity filter (cosine > 0.7) ensures non-redundancy but may discard useful prompts that are semantically close to existing ones, a trade-off we accept to maintain diversity. We set coverage threshold to 2 to avoid over-exploration of already common concepts, but this may miss subtle underrepresentation that requires higher threshold. The design avoids a controller module: Explora directly contributes prompts to the candidate pool, not routing decision.

### (D) Why it measures what we claim
Coverage gap score (candidate pool count < 2) measures 'underrepresented scenario' under the assumption that the candidate pool is representative of the domain distribution; failure mode: for inherently rare concepts, low count reflects domain sparsity, not a gap. Knowledge graph neighbor retrieval measures 'missing concept combination' under the assumption that co-occurring concepts in the graph correspond to real-world co-occurrences in text-rich images; failure mode: the graph may be incomplete or the gap may be visual (e.g., font style) rather than conceptual, leading to irrelevant combinations. The similarity filter ensures 'semantic distinctness' under the assumption that cosine similarity > 0.7 on sentence-BERT embeddings correlates with prompt redundancy; failure mode: two prompts may describe the same scene with different wording (low cosine but same semantics), allowing redundant additions.

## Contribution

(1) A novel Exploration Agent (Explora) that systematically generates out-of-distribution prompts by combining knowledge graph concept retrieval with Critic feedback, augmenting the DataEvolver pipeline to achieve coverage beyond the seed distribution.
(2) An empirical demonstration that adding Explora improves dataset coverage closure, measured by a novel coverage metric that tracks the proportion of concept combinations present in the final dataset relative to a held-out test set.
(3) Analysis of the trade-off between exploration and exploitation in multi-agent data construction, providing design guidelines for the coverage threshold and similarity filter.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|------------------------|
| Dataset | TextCaps | Real images with rich text annotations. |
| Primary metric | FID | Measures overall image quality and diversity. |
| Secondary metric | Text Recognition Accuracy (TRA) | Quantifies readability of generated text. |
| Secondary metric | Scene Understanding Accuracy (SUA) | Measures visual concept coverage via classification. |
| Baseline | Static pipeline | Standard non-adaptive data collection. |
| Baseline | Random perturber | Tests knowledge graph structure necessity. |
| Baseline | No exploration agent | Isolates effect of Explora component. |
| Baseline | Random thesaurus concepts | Replaces structured graph with random WordNet synonyms. |
| Ablation | No knowledge graph | Replaces structured graph with random words. |
| Ablation | No similarity filter | Removes redundancy check to assess its necessity. |

### Why this setup validates the claim

The choice of TextCaps ensures diversity of text-rich scenes, enabling evaluation of concept coverage. The static baseline represents non-adaptive data collection, highlighting the need for exploration. The random perturber baseline tests whether structured knowledge graph reasoning is beneficial over random word variation. The no-exploration agent baseline isolates the additive value of the Explora component. The new baseline with random thesaurus concepts (e.g., WordNet synonyms) tests whether the domain-specific graph structure is necessary over any lexical resource. The ablated no-knowledge-graph condition directly assesses the contribution of the graph. FID captures overall image quality and diversity; TRA and SUA provide downstream validation. Synthetic data experiments (e.g., constructing candidate pools with known missing concept combinations) will quantitatively verify that Explora identifies and fills those gaps. This combination of baselines and metrics provides a falsifiable test of the claim that structured exploration enhances coverage of missing concept combinations.

### Expected outcome and causal chain

**vs. Static pipeline** — On a case where a concept combination like "curved text on digital screen" is rare in the static dataset, the static baseline cannot generate diverse instances because it is limited to pre-existing data. Our method actively generates prompts for that gap using knowledge graph neighbors (e.g., "curved" linked to "digital screen"), producing more instances. We expect a noticeable FID improvement on images containing rare concept combinations, but parity on common combinations. Additionally, TRA and SUA should improve on rare-concept subsets.

**vs. Random perturber** — On a case like the gap "digital screen", a random perturber might generate "text on digital screen" but miss related concepts like "curved" because it randomly perturbs words without structure. Our method retrieves "curved" as a neighbor in the domain-specific knowledge graph and generates a specific prompt, systematically covering the gap. Thus we expect lower FID on structured concept combinations, while the random perturber may show similar FID on unstructured ones. TRA and SUA will also favor Explora on structured concepts.

**vs. No exploration agent** — Without Explora, DataEvolver only evolves based on existing prompts and critic feedback, so it may not introduce new out-of-distribution combinations. Our method actively adds new prompts for detected gaps, leading to a larger diversity of generated images. We expect overall FID improvement, especially on subsets with previously low concept coverage. Secondary metrics should also show gains on those subsets.

**vs. Random thesaurus concepts** — On a gap like "digital screen", a random thesaurus (WordNet) might suggest "monitor" or "display"—terms that are semantically similar but may not capture visual co-occurrence patterns (e.g., "curved" is not a synonym of "screen"). The domain-specific graph, mined from captions, will suggest co-occurring concepts like "curved" or "touchscreen", leading to more relevant prompt generation. Hence, we expect lower FID and higher TRA/SUA for the domain-specific graph compared to the random thesaurus.

**vs. No knowledge graph** — On a case where a concept gap exists, the ablation (no knowledge graph) uses LLM to generate prompts for the gap without related concepts, potentially generating irrelevant or overly general prompts (e.g., "text on digital screen" without specifying shape). Our method with knowledge graph retrieves related concepts (e.g., "curved"), producing more specific and diverse prompts. We expect lower FID for our full method compared to the ablation on that gap.

### What would falsify this idea

If our method shows no FID improvement on the subset of images with rare concept combinations compared to baselines, or if the gain is uniform across all concept frequencies rather than concentrated on rare combinations, the central claim that structured exploration targets underrepresented combinations would be falsified. Additionally, if downstream metrics (TRA, SUA) do not improve on rare-concept subsets, the practical significance of coverage enhancement is undermined.

## References

1. DataEvolver: Self-Evolving Multi-Agent Data Construction for Text-Rich Image Generation
2. AnyText: Multilingual Visual Text Generation And Editing
3. AgentInstruct: Toward Generative Teaching with Agentic Flows
4. AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation
5. MetaMath: Bootstrap Your Own Mathematical Questions for Large Language Models
6. OCR-VQGAN: Taming Text-within-Image Generation
7. Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding
8. Character-Aware Models Improve Visual Text Rendering
