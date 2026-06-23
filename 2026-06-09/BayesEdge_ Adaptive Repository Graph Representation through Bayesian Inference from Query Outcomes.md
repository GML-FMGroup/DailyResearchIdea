# BayesEdge: Adaptive Repository Graph Representation through Bayesian Inference from Query Outcomes

## Motivation

LARGER relies on a static repository graph where edges represent structural dependencies (imports, calls, inheritance). As the codebase evolves, these edges become stale, causing retrieved subgraphs to omit or misrepresent relevant code. LARGER has no mechanism to detect or correct such degradation without full graph recomputation, which is expensive. This structural gap—static completeness without dynamic validation—also affects RANGER and DraCo, which assume the extracted graph remains correct despite ongoing code changes.

## Key Insight

The agent's own query outcomes (success/failure of traversing an edge to retrieve useful context) provide a natural, free signal of edge validity, enabling a closed-loop Bayesian update that adapts the graph at negligible cost.

## Method

### Algorithm: BayesEdge - Lightweight Bayesian Validity Model
# Input: Static repo graph G = (V, E) with initial edge probabilities P(e)=0.5 (uniform prior)
# Parameter: threshold τ (tuned on a validation set of 500 queries from SWE-bench-verified, initial value 0.3)
# Parameter: Bayesian update factor α = 1.0 (controls prior strength)
# Assumption: Query success implies that the static edge e is valid (correctly reflects current code dependencies).

while agent runs queries Q:
    for each query q:
        (1) LARGER retrieves subgraph S_q from G using current edge probabilities
        (2) Agent executes task using S_q; record outcome o_q ∈ {success, failure}
        (3) For each edge e ∈ S_q:
            if o_q == success:
                s_e += 1
            else:
                f_e += 1
            # Beta-Bernoulli conjugate: update posterior mean
            P(e) = (s_e + α) / (s_e + f_e + 2*α)
        (4) For each edge e where P(e) < τ:
            # Re-extract using static analysis (e.g., pyright) on code units incident to e
            # If edge exists in current code, keep it and reset P(e)=0.5; otherwise remove e
            re_extract(e)

**Why this design**
We chose a Beta-Bernoulli conjugate model because it provides closed-form posterior updates without storing full query histories, accepting the cost that it assumes i.i.d. query outcomes per edge, which may be violated by correlated queries. We set a uniform prior (α=1) to avoid biasing toward existing edges, at the cost of requiring more observations to converge for rarely-traversed edges. The threshold τ triggers re-extraction only when confidence drops below a tunable level, balancing responsiveness against unnecessary recomputation; τ is tuned on a validation set of 500 queries from SWE-bench-verified, accepting that a single value may be suboptimal for repositories with different evolution patterns. We deliberately avoided learned thresholds because they would introduce dependency on task-specific data and reduce generality.

**Why it measures what we claim**
The computational quantity P(e) measures edge validity (the probability that e correctly reflects current code dependencies) because we assume that query success indicates the edge was useful and thus likely correct; this assumption fails when a success occurs despite a stale edge (e.g., irrelevant code happens to produce a correct answer), in which case P(e) overestimates true validity. The threshold τ measures staleness detection sensitivity because we assume edges with P(e) < τ are likely outdated; this assumption fails when noisy query outcomes (e.g., transient failures) drive P(e) below τ, triggering unnecessary re-extraction. Resetting P(e) to 0.5 after re-extraction measures the assumption that re-extraction restores accuracy to the prior level, which fails if the static analysis itself is incomplete or the code has changed in ways the analysis cannot capture.

## Contribution

(1) A lightweight Bayesian update mechanism that adapts repository graph edge probabilities online using agent query success/failure signals, eliminating the need for full graph recomputation. (2) A targeted re-extraction policy triggered by probability thresholds, focusing computational cost on edges most likely stale. (3) Demonstration that the agent's own interaction history provides a free supervisory signal for graph maintenance, without additional labeling or external validation.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|----------------------|
| Dataset | SWE-bench-verified + PyTorch (high-commit-frequency subset) | Tests real-world repo bug fixes plus frequent changes. |
| Primary metric | Task success rate (pass@1) | End-to-end agent performance. |
| Baseline 1 | LARGER (static lexical graph) | Tests static graph retrieval. |
| Baseline 2 | LocAgent (no memory) | Tests agent without edge updating. |
| Baseline 3 | RANGER (static graph retrieval) | Tests alternate static graph method. |
| Ablation-of-ours | BayesEdge w/o re-extraction | Isolates effect of Bayesian updates. |
| Additional ablation | Correlation measurement | Manual annotation of 100 edges to measure correlation between query success and ground-truth edge correctness. |
| Synthetic experiment | Vary percentage of stale edges (0%, 20%, 40%, 60%) | Measure P(e) vs ground truth under controlled staleness. |

### Why this setup validates the claim
This setup forms a falsifiable test of the central claim that Bayesian updating of edge probabilities improves agent performance on repository tasks. LARGER and RANGER rely on static graph retrieval, testing whether our adaptive approach outperforms static baselines. LocAgent without memory tests whether edge probability learning matters over time. The ablation removes the Bayesian update mechanism (but retains initial probabilities and re-extraction) to isolate the contribution of the posterior updates. The additional ablation directly tests the load-bearing assumption by manually annotating a random sample of 100 edges from SWE-bench-verified to compute the correlation between query success (over 5 queries) and actual edge correctness (determined by inspecting the repository at query time). If correlation is low, the Bayesian update may not capture true validity. The synthetic experiment controls staleness by artificially introducing outdated edges in a small repository and measuring how well P(e) tracks ground-truth validity. Task success rate directly measures end-to-end utility of the retrieved subgraph. If our method consistently outperforms all baselines, especially on PyTorch (high commit frequency) and in the synthetic staleness scenarios, the claim is supported; if gains are uniform or negative, the idea is falsified.

### Expected outcome and causal chain

**vs. LARGER** — On a repository that underwent a major refactoring (e.g., renaming a module), LARGER retrieves stale edges from its initial static graph, causing the agent to use incorrect dependencies and fail the task. Our method, after a few query failures on those edges, lowers their probability below τ and triggers re-extraction, updating the graph to reflect the new structure. Hence we expect a noticeable gap on tasks requiring updated dependencies, but parity on stable repositories.

**vs. LocAgent** — On a long-running agent session with many queries about the same codebase, LocAgent repeatedly attempts to use the same stale edge (e.g., an outdated function call) because it has no memory of past failures. Our method accumulates beta parameters: after successive failures, P(e) drops below τ, triggering re-extraction and fixing the edge. Thus over multiple queries our method's success rate should improve while LocAgent's remains flat, leading to a growing gap with query count.

**vs. RANGER** — On a task requiring knowledge of recently added dependencies (e.g., a new library import), RANGER's static graph does not include the new edge, so the agent fails to retrieve relevant code. Our method's uniform prior initially assigns 0.5 probability, but after a few query failures the edge is re-extracted (or if never present, remains low probability), but more importantly, if the edge should be present, re-extraction adds it. Thus we expect our method to succeed where RANGER fails on tasks involving recent changes, while performing similarly on stable dependencies.

**Additional correlation ablation** — We expect a statistically significant positive correlation (ρ > 0.3) between query success rate and manual edge correctness labels. If correlation is near zero or negative, the load-bearing assumption is violated and the method's effectiveness is undermined.

**Synthetic experiment** — In a controlled environment with 0% stale edges, P(e) should remain high (≈0.8) after 10 queries. With 60% stale edges, average P(e) should drop below 0.3 after 5 queries per edge, triggering re-extraction. If P(e) does not reflect staleness level, the Bayesian update is not capturing validity.

### What would falsify this idea
If our method shows no greater advantage on PyTorch (high commit frequency) compared to stable repositories, then the Bayesian updates are not capturing validity. Additionally, if the ablation (without Bayesian updates) achieves similar performance to the full method, then the posterior updates are unnecessary and the idea is falsified. The correlation ablation showing low correlation between query success and edge correctness would also falsify the core assumption.

## References

1. LARGER: Lexically Anchored Repository Graph Exploration and Retrieval
2. Improving Code Localization with Repository Memory
3. RANGER - Repository-Level Agent for Graph-Enhanced Retrieval
4. SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering
5. OpenHands: An Open Platform for AI Software Developers as Generalist Agents
6. Dataflow-Guided Retrieval Augmentation for Repository-Level Code Completion
7. GraphCoder: Enhancing Repository-Level Code Completion via Code Context Graph-based Retrieval and Language Model
8. CATCODER: Repository-Level Code Generation with Relevant Code and Type Context
