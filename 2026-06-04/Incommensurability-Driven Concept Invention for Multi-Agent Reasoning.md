# Incommensurability-Driven Concept Invention for Multi-Agent Reasoning

## Motivation

Existing multi-agent reasoning systems, such as the multi-agent model for mathematical discovery (Discovering mathematical concepts) and coordination benchmarks (AgentsNet), assume a fixed problem domain and communication protocol. This structural limitation prevents agents from recognizing when their shared concepts are insufficient to resolve conflicts in justifications, leading to stagnation or failure on open-ended problems. For instance, AgentsNet shows performance degradation as network size grows because agents cannot adaptively expand their coordination vocabulary, while the mathematical discovery system relies on a static data distribution that precludes invention of new mathematical concepts.

## Key Insight

The structural divergence between agents' justification graphs—measured by the syntactic and semantic distance of the axioms and inference steps they employ—provides a reliable, emergent signal that the current ontology lacks a bridging concept, enabling self-triggered concept invention without external supervision.

## Method

### (A) What it is
IDCI (Incommensurability-Driven Concept Invention) is a multi-agent framework where LLM-based agents collaboratively solve problems. Input: a set of agents (N=5), a shared ontology of concepts (axioms, definitions, lemmas), and a problem statement. Output: a solution and an expanded ontology including newly invented concepts.

### (B) How it works

**Key assumption:** Structural divergence between justification DAGs reliably indicates a missing bridging concept, provided the justifications are internally consistent. We verify internal consistency before computing divergence.

```python
def IDCI(agents, ontology, problem, max_iter=10):
    for _ in range(max_iter):
        # Phase 1: Parallel reasoning with self-consistency filter
        paths = []
        for agent in agents:
            samples = [agent.reason(problem, ontology) for _ in range(3)]  # K=3
            # internal consistency: proportion of sample pairs agreeing on conclusion
            consistent = [s for s in samples if internal_consistency(s, samples) > 0.8]
            if consistent:
                paths.append(consistent[0])  # pick most consistent path
        if len(paths) < 2:
            continue  # insufficient consistent paths
        
        # Phase 2: Justification alignment
        conflicts = []
        for path_i, path_j in combinations(paths, 2):
            if not align(path_i, path_j, ontology):
                conflicts.append((path_i, path_j))
        
        if not conflicts:
            # Agreement reached
            return paths[0].conclusion, ontology
        
        # Phase 3: Incommensurability detection
        incommensurate = []
        for p_i, p_j in conflicts:
            d = structural_divergence(p_i, p_j, ontology)  # distance > threshold
            if d > tau:  # hyperparameter tau=0.7 (swept later)
                incommensurate.append((p_i, p_j))
        
        if not incommensurate:
            # Conflicts are resolvable within current ontology; retry
            continue
        
        # Phase 4: Concept invention with retry on failure
        new_concept = None
        for attempt in range(3):  # retry up to 3 times
            new_concept = propose_concept(incommensurate, ontology, large_language_model="GPT-4")
            if new_concept and validate_concept(new_concept, ontology):
                break
        if new_concept:
            # Test on held-out examples before adoption
            if test_on_heldout(new_concept, problem):
                ontology.add(new_concept)
        
    return failure, ontology
```

The key subroutines:
- `align`: checks if two justification paths can be reconciled by a sequence of ontology-compliant transformations (unification modulo the ontology).
- `structural_divergence`: computes a normalized edit distance between the directed acyclic graphs of justifications, where nodes are concept applications and edges are inference dependencies. Nodes with high structural overlap contribute negative distance, while nodes using axioms outside the other path’s support set add positive distance.
- `propose_concept`: prompts an LLM with the conflicting paths, the current ontology, and a request: “Propose a new concept C that, when added to the ontology, explains why both justifications are valid.” The LLM returns a textual definition and a set of axioms for C, which are parsed and added. If parsing fails, we retry with a more specific prompt.
- `internal_consistency`: for a sample and a list of samples from the same agent, returns the proportion of other samples that reach the same conclusion (string equality of final statement).
- `validate_concept`: checks syntactic consistency: the new concept's axioms must not contradict existing axioms (checked via a simple syntactic check: no axiom negates another).
- `test_on_heldout`: holds out 20% of the problem’s proof steps (randomly selected before reasoning); verifies that using the new concept reduces verification error (number of steps that cannot be derived) on held-out steps. If error decreases by at least 1 step, concept is adopted.

### (C) Why this design
We chose a decentralized detection mechanism (each agent pair independently checks alignment) rather than a central arbiter because the arbiter itself would need an ontology to compare justifications—creating a regress. By using pairwise structural divergence, we avoid a single point of failure and allow multiple incommensurability signals to trigger invention. This comes at the cost of O(n²) comparison overhead per iteration, which we accept because agent counts are small (≤20) in typical multi-agent reasoning setups. We chose edit distance on justification DAGs over semantic similarity of natural language, because DAGs reflect the logical structure that underpins incommensurability; natural language similarity would conflate paraphrasing with actual agreement. The trade-off is that DAGs require parsing formal proofs, which limits applicability to domains with formal or semi-formal reasoning (e.g., mathematics, software verification). Finally, we opted for a generative LLM to propose new concepts rather than a fixed library of templates, because the space of missing concepts is open-ended. The cost is occasional nonsense proposals, which we mitigate by requiring the proposed concept to be consistent with the existing ontology before addition.

### (D) Why it measures what we claim
The **structural divergence** between justification DAGs measures the **incommensurability** of the agents' reasoning because we assume that two justifications that cannot be aligned via ontology-preserving transformations indicate a missing bridging concept. This assumption fails when agents make random or erroneous reasoning steps (e.g., hallucinated axioms) that are logically far apart but not due to missing concepts; in that case, divergence reflects error rather than genuine ontological gap. To mitigate this, we first filter paths by internal consistency (before alignment), reducing false positives from noisy agents. The **alignment check** measures **resolvability within the current ontology** because we assume that all valid reasoning on the problem can be represented within that ontology. When this assumption fails (e.g., the problem requires concepts beyond the current domain), alignment falsely reports conflict even when agents agree on the missing concept. The **concept proposal** from the LLM measures **adequacy of the invention** only if the LLM’s output is consistent with the existing ontology and the conflict structure; we enforce syntactic consistency but cannot guarantee semantic relevance. Therefore, to prevent false positives, we require that invented concepts be tested on held-out examples before full adoption.

## Contribution

(1) A self-triggered mechanism for multi-agent systems that detects ontological insufficiency via pairwise structural divergence of justification graphs, without requiring a central controller. (2) A concept invention procedure that leverages existing reasoning conflicts as input to an LLM to generate new concepts that bridge incompatibilities, enabling ontology expansion in a closed loop. (3) A design principle: that incommensurability in justifications is a sufficient and computable signal for when an ontology needs expansion, formalized through the alignment check and divergence metric.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset 1 | Formal theorem proving dataset (miniF2F subset) | Problems requiring bridging concepts |
| Dataset 2 | Scientific hypothesis generation dataset (compilation of 50 problems from research paper abstracts with conflicting conclusions) | Demonstrates broader significance |
| Primary metric | Success rate of reaching agreement with a valid new concept | Measures ability to resolve incommensurability |
| Baseline 1 | Single-agent reasoning (no collaboration) | Tests need for multi-agent |
| Baseline 2 | Multi-agent voting (no concept invention) | Tests need for invention |
| Baseline 3 | Multi-agent debate (no concept invention) | Tests need for incommensurability detection |
| Ablation 1 | IDCI without structural divergence detection (random concept invention triggered on any conflict) | Isolates effect of incommensurability detection |
| Ablation 2 | IDCI with random concept invention (replace proposed concept with random concept from a fixed pool) | Further isolates detection mechanism |
| Hyperparameter sensitivity | Sweep tau ∈ {0.5, 0.6, 0.7, 0.8} on a validation set of 20 problems | Tests sensitivity to divergence threshold |
| Empirical grounding | Compute correlation between structural divergence and human-annotated incommensurability on held-out set of 100 problems with expert annotations | Provides groundedness of the key measurement |
| Cost estimate | ~$200 (GPT-4 API: each agent reasoning call ~500 tokens, each invention call ~1000 tokens, 5 agents × 10 iterations worst-case, plus retries) | Ensures feasibility |

### Why this setup validates the claim

The dataset contains problems where two valid reasoning paths are structurally divergent, requiring a new bridging concept to reconcile them. This directly tests the central claim that incommensurability-driven invention improves multi-agent reasoning. Single-agent baseline fails because it cannot access diverse perspectives. Voting and debate baselines fail because they lack invention: voting collapses disagreements without resolution, debate may converge on a consensus that ignores the missing concept. Our ablation removes incommensurability detection, so inventions are triggered randomly, not based on structural divergence. The metric—success rate of reaching agreement with a valid new concept—directly measures the ability to detect and bridge incommensurability. If IDCI outperforms baselines and the ablation, the claim is supported; if not, the mechanism is not responsible.

### Expected outcome and causal chain

**vs. Single-agent reasoning** — On a problem where two different proof paths exist but no single agent can see both, the single-agent produces an incomplete or incorrect proof because it lacks alternative viewpoints. Our method instead uses multiple agents, each tracing one path, then detects the structural divergence and invents a concept unifying both, so we expect a significant gap on problems requiring multiple perspectives, near parity on simple problems.

**vs. Multi-agent voting** — On a problem where agents disagree because they use conceptually different axioms, voting leads to deadlock or selects the majority path, missing the needed invention. Our method detects that the disagreements are incommensurable and invents a new concept, enabling unanimous agreement. We expect IDCI to solve problems where voting fails (high incommensurability), and achieve similar success where voting already works.

**vs. Multi-agent debate** — On a problem where agents converge on a wrong consensus due to persuasive but flawed reasoning, debate reinforces the error without introducing new concepts. Our method pauses debate when incommensurability is high, forcing invention that corrects the flaw. We expect IDCI to outperform debate on problems requiring conceptual innovation, with debate failing more often on those cases.

### What would falsify this idea

If IDCI's success rate is equal across problems with and without structural divergence, then incommensurability detection is not driving performance. Also, if the ablation matches IDCI, then concept invention is not the key mechanism.

## References

1. Discovering mathematical concepts through a multi-agent system
2. Seed-Prover: Deep and Broad Reasoning for Automated Theorem Proving
3. DeepSeek-Prover-V1.5: Harnessing Proof Assistant Feedback for Reinforcement Learning and Monte-Carlo Tree Search
4. Solving olympiad geometry without human demonstrations
5. Few-shot Learning with Retrieval Augmented Language Models
6. Draft, Sketch, and Prove: Guiding Formal Theorem Provers with Informal Proofs
7. HyperTree Proof Search for Neural Theorem Proving
8. AgentsNet: Coordination and Collaborative Reasoning in Multi-Agent LLMs
