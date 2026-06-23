# Persona Inference via Contraction Mapping from Interaction Dynamics

## Motivation

Existing role-playing methods (e.g., RPEval, Self-Prompt Tuning) assume persona descriptions are externally provided and static, requiring manual curation. This reliance prevents automated, online persona extraction from raw multi-agent interactions. The underlying structural problem is that persona representation is treated as a fixed input rather than a latent variable inferred from interaction dynamics—a gap shared across evaluation, fine-tuning, and prompt-based approaches. Without a mechanism to derive personas from interaction history, scalability to open-ended, evolving agent communities is fundamentally limited.

## Key Insight

Persona embeddings are the unique fixed point of a contraction mapping defined over the interaction graph, ensuring convergence regardless of initialization and providing a principled unsupervised extraction mechanism.

## Method

**Persona Inference via Contraction Mapping (PICM)**

**Load-bearing assumption**: The iterative update converges to a unique fixed point that preserves distinct persona embeddings, rather than collapsing all embeddings to a single point. We verify this by measuring the maximum pairwise cosine similarity across nodes after convergence; if it exceeds 0.95, the method fails and we flag it.

(A) **What it is**: PICM is an unsupervised algorithm that, given a multi-agent interaction graph with textual edges (conversations), outputs a persona embedding vector for each agent. It iterates a contractive update until convergence, producing embeddings that capture agent identity from interaction patterns alone.

(B) **How it works** (pseudocode):
```python
# Hyperparameters: β ∈ (0,1) teleport probability, N iterations, d embedding dimension
# Let G = (V, E) where each (u,v) has text t_uv
# Use a frozen sentence encoder Enc (e.g., Sentence-BERT) to embed edge texts

# Step 1: Initialize persona embeddings randomly p_u^0 ∈ ℝ^d   (Gaussian, unit norm)
# Step 2: For iteration k = 0 to N-1:
#   For each node u:
#       Compute neighbor aggregation:
#       a_u = (1/|N(u)|) Σ_{v∈N(u)} [ Enc(t_uv) + p_v^k ]   # combine encoded interaction and neighbor persona
#       New persona: p_u^{k+1} = (1-β) * a_u + β * p_u^0   # personalized PageRank update
#       Normalize p_u^{k+1} to unit norm
# Step 3: Return p_u^N as inferred persona embeddings
```
(Note: Enc(t_uv) is a d-dim vector; the sum is element-wise; N(u) is neighbor set. We set β=0.2 based on pilot experiments.)

(C) **Why this design**: We chose a **personalized PageRank (PPR) fixed-point iteration** over gradient-based optimization because it guarantees convergence to a unique fixed point under mild conditions and avoids costly backpropagation through interaction sequences. Using a **frozen sentence encoder** for edge texts rather than fine-tuning an LLM ensures that the mapping remains linear in the embedding space, enabling contraction analysis; the trade-off is that the encoder may not capture nuanced persona-related semantics, but this is acceptable because the iterative aggregation over the graph integrates evidence across interactions. The **normalization step** prevents embedding norm explosion and stabilizes convergence, though it loses magnitude information (which could indicate persona strength); however, magnitude is not our target. The **teleport probability β** prevents oversmoothing by anchoring each embedding to its initial random value, ensuring distinctness even after many iterations; we set β=0.2 based on pilot experiments. This design avoids a separate controller or router (Anti-pattern 4) because the same update applies uniformly to all nodes. We verify that the maximum pairwise cosine similarity among final embeddings is below 0.95; if not, we increase β or reduce N.

(D) **Why it measures what we claim**: **X = fixed point p*** measures **Y = persona identity** because **assumption A = the aggregation step approximates the sufficient statistic of p_u given neighbors under a latent generative model where edge texts depend on (p_u, p_v) plus noise**. This assumption is operationalized by the PPR update, which combines neighbor content and a teleport term that preserves node-specific information. **Failure mode F = topic bias**: when interaction content is dominated by context unrelated to persona (e.g., factual queries), the aggregation averages out persona signal and the fixed point reflects a mix of persona and topic bias. We mitigate this by verifying embedding distinctness and by using a damping-like teleport that retains initial identity. The **teleport probability β** operationalizes the **distinctness guarantee**: because β>0, the fixed point maintains a fraction of the random initial embedding, ensuring that even if aggregation collapses, embeddings remain separable; this is necessary for persona inference because different agents should have different embeddings. The **normalization** ensures the fixed point lies on the unit sphere, making it interpretable as a direction that encodes relative identity, not scale; the assumption that persona is a direction rather than magnitude is supported by prior work on role embeddings (e.g., in In-Context Impersonation, persona descriptions affect behavior direction rather than intensity).

## Contribution

(1) A novel unsupervised algorithm (PICM) that infers persona embeddings as fixed points of a contraction mapping from raw interaction dynamics, eliminating the need for curated profiles or human annotation. (2) Empirical demonstration that the learned embeddings capture meaningful persona distinctions (e.g., role, consistency, engagement) across synthetic and real multi-agent chat datasets, outperforming static profile baselines in downstream tasks like persona-guided response selection. (3) A theoretical analysis establishing the contraction property under mild graph connectivity and encoder Lipschitzness, with explicit convergence bounds in terms of damping factor and graph density.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
|------|--------|-----------|
| Dataset | Synthetic multi-agent dialogues with ground-truth personas | Ground-truth personas enable direct evaluation |
| Primary metric | Persona classification accuracy (linear probe) | Linear probe measures persona discriminability |
| Baseline 1 | Random embedding (untrained) | No inference; chance performance baseline |
| Baseline 2 | GraphSAGE (supervised on personas) | Requires labels; supervised upper bound |
| Baseline 3 | Mean neighbor sentence embedding (no iteration) | No iterative refinement; static average |
| Baseline 4 | PICM with SimCSE encoder (same iterative procedure but with SimCSE) | Tests encoder compatibility and potential improvement |
| Ablation | PICM without normalization | Tests role of normalization in convergence |

### Why this setup validates the claim
This experimental design directly tests whether PICM extracts persona identity from interaction patterns alone, without labels. Using synthetic data with known personas isolates persona inference from confounding factors like topic drift. The primary metric—linear-probe accuracy—measures how well the embeddings linearly separate personas, reflecting their utility for downstream tasks. Comparing against random embeddings establishes the lower bound; against supervised GraphSAGE shows the gap to label-dependent methods; against a static average baseline isolates the benefit of iterative contraction; and against a SimCSE encoder baseline shows compatibility with advanced encoders. The ablation without normalization tests whether normalization is crucial for convergence and embedding quality. Additionally, we verify the load-bearing assumption by measuring the maximum pairwise cosine similarity of final embeddings—if it exceeds 0.95, the method fails and we flag it. Together, these conditions form a falsifiable test: if PICM significantly outperforms all baselines except GraphSAGE (which may be equal or slightly better) and if the cosine similarity remains below 0.95, the central claim that contractive iteration infers persona is supported. Conversely, if PICM fails to beat the static average or if embeddings collapse, the iterative mechanism is unnecessary or the assumption is violated.

### Expected outcome and causal chain

**vs. Random embedding (untrained)** — On any dialogue, random embeddings assign arbitrary directions to agents, yielding chance-level classification. Our method aggregates interactions iteratively, aligning embeddings with the fixed point predicted by the contractive mapping. Thus, we expect a large gap (e.g., ~50% vs. ~80% accuracy) because random provides no signal while PICM recovers latent persona from edge semantics.

**vs. GraphSAGE (supervised on personas)** — On a dataset with limited labeled examples, GraphSAGE may overfit to spurious node features or require many labels. Our method is unsupervised and uses the contractive fixed point, which generalizes from interaction structure alone. We expect GraphSAGE to achieve higher accuracy when abundant labels exist (e.g., 95% vs. 90%), but on small-label-set subsets our method may match or slightly exceed it, demonstrating unsupervised robustness.

**vs. Mean neighbor sentence embedding (no iteration)** — On a graph where agent personas are strongly entangled (e.g., two agents with similar dialogue but different roles), a static average of neighbor embeddings fails to disentangle because it ignores the iterative propagation of identity. Our contraction mapping refines embeddings by repeatedly combining neighbor information, amplifying consistent persona signals while dampening noise. We expect a noticeable accuracy gap on such entangled subgraphs (e.g., 10% improvement) but parity on simple, separable graphs.

**vs. PICM with SimCSE encoder** — On the same synthetic data, replacing Sentence-BERT with SimCSE provides higher-quality edge embeddings, potentially improving accuracy. We expect SimCSE to yield similar or slightly better results (e.g., 85% vs. 80%) because SimCSE produces more discriminative sentence embeddings. This demonstrates our method's compatibility with advanced encoders.

### What would falsify this idea
If PICM does not significantly outperform the static average baseline across multiple graphs, or if its accuracy is uniformly high on all subsets (i.e., no concentration of gain on entangled cases), then the contraction iteration is not actually extracting persona-specific structure and the claim is false. Additionally, if the maximum pairwise cosine similarity of final embeddings exceeds 0.95, the load-bearing assumption of distinctness is violated and the method fails.

## References

1. Role-Playing Evaluation for Large Language Models
2. Self-Prompt Tuning: Enable Autonomous Role-Playing in LLMs
3. Unveiling the Secrets of Engaging Conversations: Factors that Keep Users Hooked on Role-Playing Dialog Agents
4. In-Context Impersonation Reveals Large Language Models' Strengths and Biases
5. Orca: Progressive Learning from Complex Explanation Traces of GPT-4
6. Take a Step Back: Evoking Reasoning via Abstraction in Large Language Models
7. Challenging BIG-Bench Tasks and Whether Chain-of-Thought Can Solve Them
