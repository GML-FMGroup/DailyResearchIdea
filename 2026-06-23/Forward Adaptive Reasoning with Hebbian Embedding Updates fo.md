# Forward Adaptive Reasoning with Hebbian Embedding Updates for Dynamic Knowledge Integration

## Motivation

Existing multi-agent systems like DelveAgent dynamically decompose tasks and reflect on outputs, but their parametric knowledge remains static throughout inference. This structural bottleneck prevents agents from autonomously correcting or enriching their knowledge based on verification feedback, limiting generalization to novel domains. The root cause is the reliance on fixed pre-trained parameters, which cannot be adjusted without costly backpropagation through the entire model.

## Key Insight

Verification feedback in inference provides a local error signal that, when paired with a Hebbian update rule, can adjust knowledge embeddings in a forward-only manner without backpropagation, leveraging the natural co-occurrence between context and correctness.

## Method

### (A) What it is
FAR-Heb is a method that augments a pre-trained LLM-based agent with a set of trainable knowledge embeddings updated during inference via a Hebbian-like rule triggered by a verifier module. Input: a user query. Output: final answer and updated embeddings.

### (B) How it works
```python
# Hyperparameters: η=0.1, τ=0.3, d=512
# Initialize: knowledge embeddings E = {e_1,...,e_K} (random, L2-normalized)
# Verifier V: small MLP taking LLM hidden state h, outputting p(correct)

for reasoning step t:
    # Agent generates step s_t and context hidden state h_t (last layer of LLM)
    # Select most relevant embedding by attention: i = argmax dot(h_t, e_j) for all j
    # Agent uses e_i in its reasoning (e.g., concatenation)
    # After generation, V outputs p_t = V(h_t)
    # If p_t < 0.5:  # step likely incorrect
        # Hebbian update: e_i = normalize(e_i + η * (1 - p_t) * (h_t - e_i))
    # Else if p_t > 0.7:  # step likely correct, reinforce
        # e_i = normalize(e_i + η * p_t * (h_t - e_i))
```

### (C) Why this design
We chose Hebbian updates over backpropagation because backpropagation would require differentiating through the LLM's autoregressive generation, which is computationally prohibitive during inference; Hebbian updates are local (only the embedding and hidden state), accepting that updates are noisier and may not achieve global optimality. We update only the single most active embedding per step (selected by attention dot product) rather than all embeddings, maintaining sparsity and preventing interference; the trade-off is that relevant knowledge not selected remains static, potentially missing correction opportunities. The update uses a continuous weighting by verifier confidence (1-p_t or p_t) rather than a binary gate, allowing graded updates that reduce instability; the downside is that the verifier's miscalibration can cause inappropriate update magnitudes. The verifier is trained independently on a static dataset of step correctness, avoiding co-adaptation with the agent, but this means it may degrade under distribution shift from frequent embedding updates. Finally, we set thresholds (p_t < 0.5 for negative update, p_t > 0.7 for positive) to avoid updating on ambiguous steps; this reduces noise but risks missing updates for borderline cases.

### (D) Why it measures what we claim
The update magnitude (1-p_t) or p_t measures "verification confidence" because it quantifies how strongly the verifier believes the step is incorrect or correct; this assumption fails when the verifier is miscalibrated (e.g., overconfident on errors), in which case the update magnitude reflects the verifier's error rather than true correctness. The context hidden state h_t measures "local reasoning context" because it is the LLM's internal representation of the current step's semantics and preceding information; this operationalizes the concept of "knowledge integration" because updating e_i towards h_t aligns the knowledge with the context that triggered verification; this assumption fails when h_t contains confounding information unrelated to the knowledge, causing the embedding to drift towards noise. The selection of e_i via attention dot product measures "knowledge relevance" because it picks the embedding most similar to the current reasoning context; this operationalizes "targeted knowledge update" because only the most used knowledge is updated, avoiding interference; this assumption fails when attention picks a superficially similar but semantically irrelevant embedding (e.g., due to embedding collapse), leading to updates on wrong knowledge. The update rule e_i → normalize(e_i + η * (p_correct - 0.5) * (h_t - e_i)) measures "proportional adaptation" because the step size scales with both verifier certainty and the discrepancy between context and embedding; this ensures that updates are larger when the verifier is confident about correctness or when the context diverges from the embedding; this assumption fails when the linear combination is a poor approximation of the true gradient of the verifier's score with respect to the embedding, in which case the update may not improve future verification scores.

## Contribution

(1) A novel inference-time knowledge update mechanism, FAR-Heb, that enables agents to adjust parametric knowledge embeddings via local Hebbian updates triggered by verification feedback, without backpropagation. (2) The design principle that verification confidence and context hidden states can serve as local learning signals for embedding adaptation, providing a new paradigm for dynamic knowledge integration in reasoning loops. (3) An open-source implementation and benchmark of FAR-Heb integrated into the DelveAgent framework, demonstrating improved adaptation on scientific reasoning tasks.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | PhySciBench | Tests scientific reasoning with step-level correctness. |
| Primary metric | Final answer accuracy | Measures overall task success. |
| Baseline 1 | Plain LLM agent | No external knowledge augmentation. |
| Baseline 2 | RAG agent with static docs | Fixed knowledge retrieval without adaptation. |
| Baseline 3 | Agent with static embeddings | Same as ours but embeddings are fixed. |
| Ablation of ours | Ours without verifier | Hebbian updates ignore verification confidence. |

### Why this setup validates the claim

This combination of dataset, baselines, and metric forms a falsifiable test of the central claim that Hebbian updates guided by a verifier improve reasoning by adapting knowledge embeddings. PhySciBench provides complex scientific tasks requiring multi-step reasoning, where step correctness can be assessed. Comparing against static knowledge baselines (RAG, static embeddings) isolates the benefit of adaptive updates. The ablation without verifier tests whether the graded confidence weighting is crucial for stability. Final answer accuracy directly measures whether the adapted embeddings lead to better outcomes. If our method outperforms all static alternatives and the ablation fails, the claim is supported; if not, the core hypothesis is falsified.

### Expected outcome and causal chain

**vs. Plain LLM agent** — On a case requiring domain-specific knowledge (e.g., thermodynamics), the plain LLM invents faulty principles because it has no external knowledge, leading to incorrect reasoning steps and low accuracy. Our method retrieves and adapts relevant embeddings from the knowledge base, grounding each step in verified facts. We expect a large accuracy gap (e.g., >20%) on knowledge-intensive subsets but smaller differences on commonsense questions.

**vs. RAG agent with static docs** — On a case where the correct knowledge depends on reasoning context (e.g., selecting the right chemical property for a specific reaction), RAG retrieves the same document regardless of step context, causing mismatch and errors. Our method selects the embedding most relevant to the current hidden state and updates it if needed, aligning knowledge with context. We expect our method to outperform on tasks requiring dynamic knowledge selection, with a noticeable advantage (e.g., 10-15% higher accuracy) on multi-step synthesis planning.

**vs. Agent with static embeddings** — On a case where the initial embeddings are suboptimal for certain reasoning patterns (e.g., a step requires a different representation than the initial one), static embeddings fail to adapt, leading to repeated mistakes. Our Hebbian updates gradually shift embeddings toward correct reasoning contexts, improving future steps. We expect our method to show monotonic improvement over the reasoning process, while static embeddings plateau. The final accuracy gap should be moderate (e.g., 5-10%) but concentrated on tasks where iterative refinement matters.

### What would falsify this idea

If our method performs no better than the static embeddings baseline, or if the ablation without verifier matches our full method, then the Hebbian update guided by verifier is not the source of improvement, falsifying the central claim.

## References

1. Deep Research in Physical Sciences: A Multi-Agent Framework and Comprehensive Benchmark
2. Autonomous chemical research with large language models
3. Augmenting large language models with chemistry tools
4. Training language models to follow instructions with human feedback
5. Measuring and Narrowing the Compositionality Gap in Language Models
