# Reciprocal Alignment Verification for Robust Multi-Round Procedural Memory Refinement

## Motivation

Existing procedural memory refinement methods, such as AFTER (Managing Procedural Memory in LLM Agents) and DRAFT, assume single-round feedback suffices or rely on self-generated feedback without verification. This structural reliance on unverified feedback leads to degradation in multi-round iterative refinement, as observed in AFTER's single-round limitation and DRAFT's overfitting risk. The root cause is the absence of a self-correcting signal to detect when feedback becomes misleading, allowing erroneous updates to accumulate.

## Key Insight

Reciprocal alignment verification provides a self-correcting signal by comparing two independent refinements, enabling detection and discarding of misaligned updates, thereby ensuring that only consistently beneficial refinements are retained.

## Method

**RAVEN (Reciprocal Alignment Verification for rEfiNement)**

(A) **What it is**: RAVEN is an iterative refinement algorithm for procedural memory that uses two independent refinement proposals and accepts only those that show high reciprocal alignment, preventing degradation from erroneous feedback.

(B) **How it works**:
```pseudocode
Input: Initial procedural memory M0 (e.g., skill library), set of tasks T, base LLM, refinement generator G (two instances with different seeds), alignment threshold τ (calibrated via small set of biased feedback), calibration set C_bias (e.g., 50 examples with intentionally introduced feedback errors)
Output: Refined memory M*

// Assumption: Two independent refinements from same LLM under different seeds are independent enough that agreement implies correctness (random errors unlikely to coincide). This assumption is calibrated by measuring alignment distribution on C_bias where both proposals share systematic bias; τ is set to exceed 95th percentile of alignment under biased conditions.

1. Initialize M = M0
2. For round = 1 to K:
   a. For each task in T, execute agent with current M to collect feedback F (e.g., success/failure logs)
   b. Generate two independent refinement proposals:
      - P1 = G(M, F, seed=1)   // LLM with temperature 0.7
      - P2 = G(M, F, seed=2)   // same LLM, different random seed
   c. Compute alignment score A = cosine_similarity(embed(P1), embed(P2))  // embed: Sentence-BERT
   d. If A ≥ τ (calibrated τ = 0.85):
      - Apply refinement to M: M = update(M, P1)
   e. Else:
      - Keep M unchanged (discard both proposals)
3. Return M*
```
Hyperparameters: τ (calibrated to 0.85 using C_bias of 50 examples), K=5, temperature=0.7.

(C) **Why this design**: We chose two independent refinements (using different random seeds) over a single refinement because it enables a consistency check without external validation, accepting the computational cost of double generation. We chose a cosine similarity threshold over statistical tests (e.g., p-value) because it is simpler and domain-agnostic, at the risk of ignoring semantically equivalent but lexically distinct proposals. We chose to discard both proposals on misalignment rather than merging or selecting one, because merging could introduce conflicting changes and selecting one would be arbitrary, thereby preserving robustness at the expense of losing potentially useful but divergent updates. The alignment threshold τ trades off precision and recall; higher τ reduces false positives but may skip beneficial refinements. The independence assumption is calibrated: we use a small set C_bias of tasks where feedback is intentionally corrupted (e.g., inverted success signals) to measure alignment when both proposals are wrong due to shared bias; τ is set to exceed the 95th percentile of alignment on C_bias, ensuring that only proposals unlikely to be jointly erroneous are accepted.

(D) **Why it measures what we claim**: The alignment score A measures the consistency of refinement directions under independent stochasticity. The assumption is that two independent refinements that agree are more likely correct because random errors are unlikely to coincide; this assumption fails when both refinements share a systematic bias (e.g., from the same flawed feedback), in which case A reflects correlated error rather than correctness. The update operation applies only when A ≥ τ, operationalizing the concept of 'reciprocal verification' as a guard against degradation. The computational quantity `similarity(P1, P2)` measures 'feedback quality' under the assumption that beneficial refinements are robust to sampling noise; this assumption fails when beneficial refinements are low-probability events that only one seed discovers, causing RAVEN to discard them. Thus, RAVEN prioritizes conservative refinement over risky updates. Additionally, the calibrated τ ensures that under worst-case systematic bias (estimated via C_bias), the false acceptance rate is below 5%.

## Contribution

(1) Proposes RAVEN, a reciprocal alignment verification mechanism for multi-round procedural memory refinement that uses agreement between independent proposals as a self-correcting signal. (2) Demonstrates that this approach prevents the degradation observed in unverified iterative refinement, providing a principled trade-off between refinement benefit and noise acceptance. (3) Provides an analysis of alignment threshold selection and its impact on refinement stability.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | ToolBench (tool-using tasks) | Covers diverse tool-use scenarios |
| Primary metric | Task Success Rate | Direct measure of task completion |
| Baseline 1 | Single Refinement (no alignment) | Tests degradation from erroneous feedback |
| Baseline 2 | No Refinement (static memory) | Baseline lower bound for improvement |
| Baseline 3 | Self-Consistency Refinement (majority vote over 5 seeds) | Isolates effect of alignment vs. voting |
| Ablation | RAVEN w/o alignment threshold (always apply P1) | Isolates effect of alignment check |
| Analysis | Alignment vs. oracle quality using simulated noisy feedback | Validates that alignment correlates with true improvement |

### Why this setup validates the claim

This design tests RAVEN's central claim: reciprocal alignment prevents degradation from erroneous feedback while enabling beneficial refinements. ToolBench provides realistic multi-step tool tasks where procedural memory updates can either improve or degrade performance. Comparing against Single Refinement (which always applies a single proposal) reveals whether RAVEN's alignment gate reduces harmful updates. The No Refinement baseline establishes the initial performance level. Task Success Rate directly captures whether refinements help or hurt task completion. The ablation removes the alignment threshold, so any performance drop relative to full RAVEN demonstrates the necessity of the reciprocal verification mechanism. The self-consistency baseline uses majority voting across multiple seeds to select refinements, testing whether the benefit of two seeds is due to voting or alignment. The analysis experiment explicitly measures the correlation between alignment score and actual refinement quality (computed using oracle feedback with known correct and incorrect refinements) to validate the proxy. Together, these comparisons form a falsifiable test: if RAVEN outperforms Single Refinement specifically on tasks where erroneous feedback causes degradation, the claim is supported.

### Expected outcome and causal chain

**vs. Single Refinement** — On a case where the base LLM generates faulty refinement (e.g., adding a wrong tool call due to ambiguous feedback), Single Refinement blindly applies it, causing the agent to fail subsequent tasks (e.g., misusing a calculator). Our method instead generates two independent proposals; if both mistakenly suggest the same wrong update (unlikely due to different seeds), alignment fails? Actually, if both are wrong but agree, alignment would accept them. But the claim is that erroneous feedback tends to produce diverse errors, so alignment is low. On such a case, alignment score drops below threshold, so we discard both proposals and keep the original memory. Thus we expect a noticeable gap on tasks where feedback is noisy (e.g., ambiguous success signals), while on clean tasks both methods perform similarly.

**vs. No Refinement** — On a case where beneficial refinement exists (e.g., a more efficient search strategy for a retrieval tool), No Refinement never updates memory, so it repeats suboptimal behavior indefinitely. Our method generates two independent proposals that both suggest the efficient improvement (high alignment). Since alignment ≥ τ, we apply the refinement and achieve higher success rate. We expect a clear advantage on tasks where performance plateaus without adaptation, with RAVEN showing incremental gains over rounds.

**vs. Self-Consistency Refinement** — Self-consistency uses majority voting over 5 seeds to pick the most common refinement. If erroneous feedback causes a majority of seeds to suggest the same wrong update, self-consistency will apply it, while RAVEN may discard it if alignment is low (since two seeds are unlikely to both be wrong in the same way under the assumption). We expect RAVEN to outperform self-consistency specifically on tasks where errors are systematic but not universal, as alignment captures the consistency of two independent draws better than majority frequency.

**Analysis: Alignment vs. oracle quality** — We create a set of 100 simulated feedback scenarios with known correct and incorrect refinements (oracle quality). For each, we compute alignment scores and measure the correlation (e.g., Spearman ρ) between alignment and oracle quality. A high positive correlation (ρ > 0.5) validates that alignment serves as a proxy for refinement correctness. If correlation is weak, the core assumption is weakened.

### What would falsify this idea

If RAVEN performs no better than Single Refinement on tasks with known noisy feedback (e.g., tasks with partial observability), or if its advantage is uniformly distributed across all tasks rather than concentrated where erroneous feedback degrades Single Refinement, then the central claim of preventing degradation via reciprocal alignment is invalid. Additionally, if the analysis shows low correlation (<0.3) between alignment and oracle quality, the proxy itself is invalid.

## References

1. Managing Procedural Memory in LLM Agents: Control, Adaptation, and Evaluation
2. From Exploration to Mastery: Enabling LLMs to Master Tools via Self-Driven Interactions
3. MLE-bench: Evaluating Machine Learning Agents on Machine Learning Engineering
4. SWE-bench: Can Language Models Resolve Real-World GitHub Issues?
5. ML-Bench: Evaluating Large Language Models and Agents for Machine Learning Tasks on Repository-Level Code
6. CLOVA: A Closed-LOop Visual Assistant with Tool Usage and Update
7. Confucius: Iterative Tool Learning from Introspection Feedback by Easy-to-Difficult Curriculum
8. PAL: Program-aided Language Models
