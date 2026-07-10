# Skill-Decomposed Performance Estimation for Sample-Efficient Embodied Agent Architecture Search

## Motivation

Existing automated architecture search for embodied agents (e.g., AgentCanvas/KDLoop) relies on many costly full environment rollouts to evaluate each candidate architecture. This is fundamentally sample-inefficient because it treats the architecture as a monolithic black box, ignoring its natural decomposition into modular skills (perception, planning, control). The structural root cause is that these methods require full task-level feedback for every evaluation, which is prohibitively expensive in real-world or sparse-reward settings.

## Key Insight

Because embodied agent architectures are structured as typed graphs of modular skills with well-defined interfaces, each skill’s proficiency can be estimated via cheap, isolated diagnostic tasks, and the overall architecture performance can be predicted as a learned composition of these skill-level scores, bypassing the need for full rollouts.

## Method

## Skill-Decomposed Performance Estimation for Sample-Efficient Embodied Agent Architecture Search

### (A) What it is
SDAS (Skill-Decomposed Architecture Search) is a sample-efficient method that estimates a candidate architecture's performance by running cheap diagnostic tasks on each modular skill component and on composite skill pairs, then uses a regression model to map skill scores to overall performance, drastically reducing the need for expensive full environment evaluations. **Load-bearing assumption**: The performance of each skill module (and composite of skills) on simplified, isolated (or paired) diagnostic tasks is a reliable and monotonic predictor of its contribution to the overall architecture performance in the full task. To verify this, we perform a calibration check on a held-out set of architectures.

### (B) How it works

```pseudocode
Input: Candidate architectures A (typed graphs per AgentCanvas), 
       small set S of full-rollout evaluations {(a_i, full_perf_i)},
       diagnostic task library T = {t_skill for each skill type} ∪ {t_pair for each connected skill type pair}.
Output: Best architecture with estimated performance f_hat.

Phase 1: Build regression model from diagnostics to full performance
1. For each training architecture a_i in S:
   For each skill module m in a_i:
      Execute diagnostic task t_m on module m in isolation (e.g., for perception: detect objects in simplified synthetic scenes, 10 steps).
      Record diagnostic score d_im (e.g., accuracy, latency).
   For each connected skill pair (m1,m2) in a_i:
      Execute composite diagnostic task t_pair on modules m1,m2 jointly (e.g., perception+planning: navigate to a red cube in clutter-free environment, 5 steps).
      Record composite diagnostic score c_im1m2.
   Let D_i = [d_i1, d_i2, ..., d_iK, c_i1, c_i2, ..., c_iL] be the concatenated skill score vector (K = #skill types, L = #connected pairs).
   Let f_i = full_perf_i from full rollout.
2. Train regression model R (GradientBoostingRegressor, max_depth=3) on pairs (D_i, f_i).
3. **Calibration**: On a held-out set of 5 architectures (not in S), compute predicted performance using R and actual full performance. If Pearson correlation r < 0.8, increase all diagnostic episode lengths by 50% and retrain R.

Phase 2: Efficient architecture search
4. For each candidate architecture a in search space (not in S):
   Compute its diagnostic score vector D via quick isolated and composite diagnostic tasks.
   Estimate full performance f_hat = R(D).
5. Use f_hat as the acquisition score for search (e.g., Bayesian optimization or evolutionary selection).
6. (Optional) Validate top-k candidates with full rollouts.
```
Hyperparameters: |S| = 50 architectures; regressor: GradientBoostingRegressor; diagnostic tasks: simplified episodes (e.g., 10 steps for isolated, 5 steps for composite); calibration set: 5 architectures; correlation threshold: 0.8.

### (C) Why this design
We chose a learned regression over a fixed weighted sum because skill interactions are non-linear and context-dependent (e.g., a perfect planner cannot compensate for a blind perception module); the trade-off is requiring a small set of full rollouts for training, but this set is orders of magnitude smaller than the full search budget (e.g., 50 vs. 1000 candidates). We designed diagnostic tasks as simplified versions of the full task (e.g., clutter-free scenes, shorter horizons) rather than arbitrary probes because they must retain the core skill challenge while being cheap; the trade-off is a potential distribution shift where diagnostic performance may not perfectly transfer. We opted to evaluate modules in isolation and in composite pairs because isolated evaluation is faster, but composite pairs capture key cross-module interactions; the trade-off is missing higher-order interactions (e.g., perception+planning+control), which we attempt to capture via the regression model's ability to learn correlations from the training architectures. Compared to prior works like AgentCanvas that perform full rollouts for every candidate, SDAS reduces total feedback cost by over 90% in expectation while maintaining search quality.

### (D) Why it measures what we claim
The diagnostic score for each skill module (or composite pair) measures the skill's isolated (or paired) proficiency under controlled, simplified conditions. The assumption is that proficiency on these simplified tasks monotonically relates to proficiency in the full task; this assumption fails when the full task introduces domain shifts or emergent difficulty not present in diagnostics (e.g., real-world sensor noise), in which case the diagnostic score reflects only the simplified scenario's demands. The regression model's learned mapping from diagnostic vector to full performance measures the compositional rule that combines skill scores into a system-level prediction. The assumption is that the functional mapping is consistent across architectures; this assumption fails when there are emergent interactions (e.g., a planning module that relies on precise perception but the composite diagnostic does not stress that precision), in which case the predicted performance reflects only the additive contributions captured by the diagnostics. The calibration step checks the monotonicity assumption by measuring correlation on held-out architectures; if correlation is low, we increase diagnostic difficulty to better match the full task distribution. The number of diagnostic episodes (cheap) versus full rollouts (expensive) operationalizes sample-efficiency: by using many cheap diagnostics and few expensive rollouts, we reduce total feedback cost proportionally to the cost ratio, under the assumption that diagnostic tasks are at least 10× cheaper per unit of information gained.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Embodied tasks from AgentCanvas (e.g., VLN) and long-horizon manipulation (block stacking, 10 blocks) | Covers diverse skills and sample-efficiency critical tasks. |
| Primary metric | Success rate in full environment | Directly measures task completion quality. |
| Baseline: Full-rollout search | Full-rollout search (AgentCanvas) | Upper bound on cost, not sample-efficient. |
| Baseline: Hand-designed architectures | Hand-designed architectures (domain experts) | Lower bound on performance without search. |
| Baseline: Fixed weighted sum | Fixed weighted sum of diagnostics | Tests need for learned regression. |
| Baseline: Weight-sharing proxy | Weight-sharing NAS (e.g., DARTS-like proxy) | Tests against zero-cost proxy approaches. |
| Baseline: Zero-shot proxy | Zero-shot proxy (FLOPs, gradient norm) | Tests against simple complexity measures. |
| Ablation-of-ours | SDAS with fixed weighted sum (no regression) | Isolates regression contribution to sample efficiency. |

### Why this setup validates the claim
This combination of dataset, baselines, and metric forms a falsifiable test of the central claim that SDAS reduces sample cost without sacrificing search quality. By comparing to full-rollout search, we test whether SDAS maintains performance while using 90% fewer full evaluations. The hand-designed baseline tests if automated search finds better architectures than human intuition. The fixed weighted sum baseline directly tests the necessity of the learned regression for capturing non-linear skill interactions, as predicted in our design rationale. The ablation (fixed weights) further isolates the regression's contribution. Adding weight-sharing and zero-shot proxies positions SDAS within the broader NAS literature, highlighting that our method leverages domain-specific skill decomposition rather than generic weight-sharing or complexity-based proxies. The long-horizon manipulation dataset stresses sample efficiency because full rollouts are extremely costly (e.g., 1000 steps). The success rate metric is the primary measure of task completion, directly reflecting the system-level performance we aim to approximate, making it sensitive to both false positives (overestimation) and false negatives (underestimation) in our predictions. If SDAS outperforms all baselines without requiring high cost, the claim is validated; otherwise, the diagnostic or regression assumptions must be revised.

### Expected outcome and causal chain

**vs. Full-rollout search** — On a case where the search space contains 1000 candidate architectures, the baseline must execute 1000 full rollouts (e.g., each 100 steps in a complex manipulation task), costing prohibitive time and compute. Our method instead uses only 50 full rollouts to train the regression, then predicts performance for all 1000 via cheap diagnostics (each 10 steps for isolated, 5 for composite), so we expect a >90% reduction in total feedback cost while maintaining a success rate within 5% of the full-rollout optimum (since the regression captures the mapping from diagnostics to performance with minimal bias).

**vs. Hand-designed architectures** — On a case requiring novel skill coordination (e.g., an embodied agent must simultaneously navigate and manipulate a deformable object in clutter), a human expert may default to a conventional pipeline (e.g., separate perception, planning, and low-level control) that fails due to unforeseen interaction delays. Our search explores diverse typed graph architectures and selects one where perception and planning are tightly coupled (e.g., shared attention mechanisms), leading to higher success in that scenario. Thus we expect SDAS to outperform the best hand-designed architecture by a noticeable margin (e.g., 10-20% absolute success rate) on tasks where conventional modularity is suboptimal.

**vs. Fixed weighted sum of diagnostics** — On a case where perception noise severely impacts planning (e.g., partial observability in a visual Q&A task), a fixed weighted sum (e.g., 0.5*perception + 0.5*planning) might overestimate performance because it assumes independence: a perfect planner with poor perception still scores middle. Our regression model, trained on diverse architectures from Phase 1, learns that low perception scores severely degrade final success even if planning is perfect, so it predicts lower scores for such imbalanced architectures. Consequently, during search, SDAS avoids those architectures and selects ones with balanced skills. We expect SDAS to show a clear gap over the fixed sum on subsets where skill imbalance dominates (e.g., at least 15% higher success on architectures with low perception but high planning).

**vs. Weight-sharing proxy** — On a task where weight-sharing proxies (e.g., gradient matching) are known to correlate poorly with actual performance (e.g., due to architecture-dependent optimization), SDAS's diagnostic-based estimation provides a more targeted proxy. We expect SDAS to discover significantly better architectures (e.g., 10% absolute success rate higher) in such tasks, especially for long-horizon manipulation where the weight-sharing proxy may be misleading.

**vs. Zero-shot proxy** — Zero-shot proxies such as FLOPs or parameter count fail to capture skill-level interactions (e.g., a large perception module with a tiny planning module may have low FLOPs but poor task performance). SDAS's diagnostic scores explicitly measure each skill's proficiency, leading to more reliable performance estimation. We expect SDAS to outperform zero-shot proxies by at least a 15% absolute success rate on tasks where skill imbalance is critical.

**vs. Ablation (SDAS with fixed weighted sum)** — Same as vs. fixed weighted sum: on the same imbalance case, the ablation fails to correct for interaction, leading to poor search outcomes, while full SDAS excels. Thus the ablation suffers a performance drop exactly where non-linearity is present, confirming the regression's role.

### What would falsify this idea
If SDAS with regression performs no better than the fixed weighted sum baseline across all task types (i.e., the improvement is not concentrated on skills-interaction cases), then the assumption of non-linear skill interactions is unfounded, and the regression is unnecessary — the central claim of sample efficiency through learned composition is false. Additionally, if the calibration check consistently fails (low correlation) even after increasing diagnostic episode length, it would indicate that the diagnostic tasks are fundamentally unrepresentative, violating the load-bearing assumption.

## References

1. Automating the Design of Embodied Agent Architectures
2. Multi-agent Architecture Search via Agentic Supernet
3. EvoAgentX: An Automated Framework for Evolving Agentic Workflows
4. Symbolic Learning Enables Self-Evolving Agents
5. MapCoder: Multi-Agent Code Generation for Competitive Problem Solving
6. Adaptive In-conversation Team Building for Language Model Agents
7. Language Agent Tree Search Unifies Reasoning Acting and Planning in Language Models
8. LLM Performance Predictors are good initializers for Architecture Search
