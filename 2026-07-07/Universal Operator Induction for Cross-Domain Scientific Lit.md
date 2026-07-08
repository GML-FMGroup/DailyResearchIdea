# Universal Operator Induction for Cross-Domain Scientific Literature Search via Semantic Embeddings

## Motivation

Existing workflow induction methods (e.g., PaperPilot) assume access to domain-specific search APIs and structured metadata, preventing transferability across scientific domains. The root cause is that operators are tied to domain-specific tool interfaces and vocabularies, requiring manual adaptation for each new domain.

## Key Insight

The semantic embedding space (e.g., SPECTER) is sufficiently structured that operators (similarity search, expansion, filtering) can be defined as geometric transformations that preserve their semantics across domains, enabling zero-shot transfer.

## Method

### (A) What it is
UniFlow (Universal Workflow Induction) learns a set of domain-agnostic operators on semantic embeddings and induces executable search workflows across scientific domains without requiring domain-specific APIs. Input: a query (natural language) and an optional anchor paper. Output: a sequence of operator calls that retrieve and rank papers.

### (B) How it works
```pseudocode
# Training phase
Input: Source domain papers S, target domain papers T (unlabeled)
Embedder: SPECTER2 (frozen, 768-dim) -> e(p) for paper p

# Step 1: Define operator prototypes
ops = {similarity_search, citation_expansion, keyword_filter, rerank_by_relevance}
# Each operator is a neural module with parameters theta_op:
#   - 2-layer MLP, hidden=256, ReLU activation
#   - maps embedding state s (768-dim) to new set of embeddings (same dim)

# Step 2: Pre-train on source domain with supervised workflow demonstrations
for each (query, workflow_dag) in source_demonstrations:
    state = embed(query)
    for step in workflow_dag:
        op = ops[step.type]
        state = op(state, step.arguments)  # e.g., similarity_search(state, k=10)
    optimize theta_op to maximize nDCG of retrieved papers against ground truth
# Optimizer: Adam, lr=1e-4, batch_size=32

# Step 3: Adversarial domain adaptation
DomainDiscriminator: 2-layer MLP (hidden=256, ReLU) that predicts domain label from embedding state
for each batch from source and target:
    state_source = forward_pass(source_papers)
    state_target = forward_pass(target_papers)
    # Train discriminator to maximize domain classification accuracy
    # Train operator parameters to minimize discriminator accuracy (adversarial loss)
    # Combined loss = warmup * nDCG_loss + (1-warmup) * adversarial_loss
    warmup linearly decays from 1 to 0.5 over first 10k steps (i.e., after 10k steps, weight = 0.5)

# Verification of load-bearing assumption:
# After training, evaluate nDCG on a held-out calibration set of 512 target-domain queries.
# If nDCG drops more than 5% relative to source performance, we flag assumption violation.

# Inference phase
Input: query q, optional anchor paper a
state = embed(q or a)
workflow = []
for t in 1..max_steps (max_steps=10):
    action = BeamSearchSelectionOverOperators(state, beam_width=5)  # or greedy
    state = op[action](state)
    workflow.append(action)
Output: ranked papers from final state
```

### (C) Why this design
We chose to operate in a fixed semantic embedding space (SPECTER2) rather than learning domain-specific embeddings because it provides a common geometric reference that is pretrained over 47M papers across domains; the trade-off is that fine-grained domain knowledge may be lost when two domains use different jargon for the same concept. We chose adversarial domain adaptation over supervised fine-tuning on each target domain because it requires no annotated workflows in the target domain; the cost is that operators may become overly generic and miss domain-specific effective strategies (e.g., using conference filters in CS). We chose a sequential operator sequence (rather than a DAG) to simplify autoregressive generation with an LLM controller; this limits parallelism but avoids the complexity of predicting arbitrary DAG structures and makes training via standard sequence prediction feasible.

### (D) Why it measures what we claim
Embedding similarity (cosine distance in SPECTER2 space) measures cross-domain semantic relatedness because SPECTER2 is trained on a large multi-domain corpus with a contrastive objective that aligns citation graph proximity; this assumption fails when two domains use distinct terminology for identical concepts (e.g., 'neoplasm' vs 'tumor'), in which case similarity may reflect surface form rather than true relatedness. The adversarial domain discriminator measures domain invariance of operator outputs because it penalizes representations that contain domain-specific information; this assumption fails when effective retrieval requires domain-specific cues (e.g., recognizing that 'ML' in target domain refers to machine learning rather than maximum likelihood), in which case the operator may become insensitive to crucial distinctions. Missing equivalence: adversarial loss minimizes domain discrepancy as measured by discriminator accuracy, which operationalizes 'operator invariance'. The linking assumption is that lower discrepancy directly causes higher nDCG on the target domain. This assumption fails when domain-invariant representations lose discriminative power for retrieval (e.g., when domain-specific cues are essential).

## Contribution

(1) A novel framework (UniFlow) that learns domain-agnostic operators in semantic embedding space via adversarial adaptation, enabling zero-shot cross-domain scientific literature search. (2) Empirical demonstration that UniFlow operators transfer across computer science, biology, and physics without retraining, achieving comparable nDCG to domain-specific baselines. (3) A new cross-domain benchmark (XPaperSearch) covering three scientific domains to evaluate workflow induction transferability.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | UniFlowEval (3 domain pairs: CS→biomed, CS→physics, physics→chemistry) | Tests cross-domain transfer across multiple pairs |
| Primary metric | nDCG@10 | Sensitive to ranking quality |
| Baseline 1 | SPECTER2 cosine similarity | Checks if workflow needed |
| Baseline 2 | Source-only workflow (no adapt) | Tests adaptation necessity |
| Baseline 3 | GPT-4 agent with search tool | Tests LLM reasoning baseline |
| Ablation | UniFlow w/o adversarial loss | Isolates adversarial effect |

### Why this setup validates the claim
This design forms a falsifiable test of cross-domain workflow induction. The SPECTER baseline isolates the benefit of learned operator sequences over simple similarity. Source-only directly tests whether adversarial adaptation improves target performance. Comparing against GPT-4 evaluates whether controlled operator execution outperforms generic LLM tool use. The ablation identifies the adversarial loss's contribution. Using nDCG@10 ensures sensitivity to ranking improvements from domain-invariant operators. Together, these comparisons decompose the central claim into component hypotheses, each with a clear baseline falsifier. We also verify the load-bearing assumption (adversarial adaptation preserves retrieval effectiveness) by holding out a calibration set of 512 target-domain queries per domain pair; if nDCG drops more than 5% relative to source-only, the assumption is considered violated. Training requires 100 GPU hours on a single V100 GPU.

### Expected outcome and causal chain

**vs. SPECTER2 cosine similarity** — On a target query where surface terminology differs (e.g., 'myocardial infarction' vs 'heart attack'), SPECTER retrieves papers with lexical matches but irrelevant content. Our method uses citation expansion and keyword filter to incorporate relational cues, so we expect a noticeable gain on such domain-vocabulary queries, with roughly +15% nDCG@10.

**vs. Source-only workflow (no adaptation)** — On a target query requiring field-specific operator weights (e.g., conference filters matter in CS but not in biomed), source-only applies the same strategy and fails. Our adversarial loss pushes operators to ignore domain cues, yielding more robust performance. Expect a consistent +10% nDCG@10 across all target queries, with larger gaps on queries with high domain shift.

**vs. GPT-4 agent with search tool** — On a complex multi-step query (e.g., 'find papers that apply graph neural networks to drug discovery'), GPT-4 may hallucinate steps or misuse APIs. Our method executes a learned operator chain that is grounded in training demonstrations, resulting in more reliable retrieval. Expect a significant improvement (+20% nDCG@10) on queries requiring multiple operators.

### What would falsify this idea
If UniFlow performs no better than source-only on target domain across all three domain pairs, the adversarial adaptation fails to induce useful cross-domain operators, directly invalidating the central claim of domain-agnostic workflow induction.

## References

1. Multi-Turn Agentic Scientific Literature Search via Workflow Induction
2. PaSa: An LLM Agent for Comprehensive Academic Paper Search
3. ChatCite: LLM Agent with Human Workflow Guidance for Comparative Literature Summary
4. SPAR: Scholar Paper Retrieval with LLM-based Agents for Enhanced Academic Search
5. Language agents achieve superhuman synthesis of scientific knowledge
6. Target-aware Abstractive Related Work Generation with Contrastive Learning
7. G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment
8. LitLLM: A Toolkit for Scientific Literature Review
