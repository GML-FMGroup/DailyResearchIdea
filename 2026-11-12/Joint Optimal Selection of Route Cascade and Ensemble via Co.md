# Joint Optimal Selection of Route, Cascade, and Ensemble via Constrained Dynamic Programming

## Motivation

Existing LLM inference frameworks treat routing, cascading, and ensembling as separate decision problems: PISA-E (Ensemble Routing for Prompt-Adaptive Cost-Aware LLM Inference) selects ensembles using a surrogate for ensemble success, while cascading and routing are optimized independently, so the unified formulation from routing and cascading is not shown to extend to ensembles. This separable-stages assumption structurally prevents joint optimality, because the choice of ensemble composition and the decision to cascade are coupled through the shared cost budget and the prompt's difficulty; optimizing them independently yields suboptimal cost-quality trade-offs.

## Key Insight

Because the cost-quality trade-off across route, cascade, and ensemble actions is additive over a directed acyclic graph of model invocations, the globally optimal joint policy can be computed exactly by dynamic programming on that DAG, without any surrogate for ensemble success.

## Method

**What it is.** JOINT (Joint Optimal Inference Navigation and Termination) takes as input a prompt, a set of available LLMs with known per-token costs, and a cost budget, and outputs a decision graph specifying which model to invoke, in what order, and whether to ensemble their outputs, achieving the minimum expected cost subject to a quality constraint (or maximum expected quality subject to a cost constraint).

**How it works.**

```pseudocode
Input: prompt x, models M={m1..mK} (K=5 in main experiments, K=8 in scalability experiment),
       per-model cost c_k, quality predictor q_k(x) (calibrated via isotonic regression on held-out set),
       budget B, ensemble quality model Q_ens(S, x) for subset S (|S|<=3),
       cascade transition model P(correct | m_k, x)
Output: optimal decision graph G*

1. Build DAG: nodes = (subset S of models already invoked, current state).
   Edges = actions: invoke model m not in S (cost c_m, update S), or stop and output
   (if |S|=1, single-model; if |S|>1, ensemble via Q_ens(S,x)).
   Cascade edges: after invoking m, with probability 1-P(correct|m,x) proceed to next node.
2. For each node, compute value function V(S) = min over actions of [cost(action) + expected future cost]
   subject to expected quality >= target, using backward induction on the DAG (topological order by |S|).
   Ensemble quality Q_ens(S,x) and cascade correctness P(correct|m,x) are estimated from a held-out
   calibration set of size N_cal (N_cal=512 in main experiments, swept over {128, 256, 512, 1024, 2048}
   in calibration-size sensitivity experiment).
   To make the DP robust to estimation error, we replace point estimates with conservative
   lower-confidence-bound (LCB) estimates: Q_ens_LCB(S,x) = Q_ens_hat(S,x) - beta * sigma_Q(S,x),
   P_LCB(correct|m,x) = P_hat(correct|m,x) - beta * sigma_P(m,x), where sigma_Q and sigma_P are
   standard errors from the calibration set (binomial standard error for P, bootstrap standard error
   over 100 resamples for Q_ens), and beta=1.0 (one standard error) in main experiments.
   The DP then optimizes a risk-averse objective: minimize expected cost subject to
   Q_ens_LCB(S,x) >= target (or maximize Q_ens_LCB subject to cost <= B).
   We also compute a coverage diagnostic: for each test prompt x, the distance to the nearest
   calibration cluster centroid (in the same feature space used for q_k) is computed; if this
   distance exceeds the 95th percentile of calibration-to-calibration distances, the prompt is
   flagged as out-of-coverage, and the DP falls back to the point estimate (beta=0) for that prompt
   while logging the flag.
3. Extract optimal policy: at each node, choose action minimizing cost subject to quality constraint
   using the LCB estimates.
4. Return decision graph G* (a DAG of model invocations and ensemble decisions).
```

**Why this design.** We chose a DAG-based dynamic program over a separate router-cascader-ensembler pipeline because the cost and quality of an ensemble depend on which models have already been invoked in a cascade, so the decisions are not separable; the DP captures this coupling exactly, accepting the cost that the state space grows with the number of models (mitigated by limiting ensemble size to 3). We chose a calibrated quality predictor q_k(x) over an uncalibrated surrogate because the DP's optimality depends on the predicted qualities being unbiased; we calibrate on a held-out set using isotonic regression, accepting the cost of a small calibration set. We chose to model cascade transitions with a per-model correctness probability P(correct|m,x) estimated from the same calibration set, rather than assuming independence, because ensemble quality Q_ens depends on the correlation of errors across models; we estimate the joint error distribution from calibration data, accepting the cost of estimating pairwise correlations. We chose LCB estimates and a risk-averse objective over point estimates because the load-bearing assumption is that Q_ens and P(correct|m,x) are estimated accurately enough for the DP's Bellman value function to be close to the true optimal expected cost; this assumption is stated explicitly: the estimates are unbiased and low-variance for the prompt distribution at test time. To guard against violation of this assumption (especially under distribution shift), we use conservative LCB estimates with beta=1.0 and a coverage diagnostic that flags out-of-coverage prompts. This design is not a variant of PISA-E: PISA-E uses a surrogate for ensemble success and selects ensembles separately from cascading, whereas JOINT solves the joint problem exactly via DP and uses calibrated LCB estimates directly, with no surrogate.

**Why this measures what we claim.** The DP's value function V(S) measures the minimum expected cost to achieve the quality target from state S because the additive cost structure and the Markov property of cascade transitions make the Bellman equation exact; this assumption fails when the quality predictor q_k(x) is biased for out-of-distribution prompts, in which case V(S) reflects the biased estimate rather than true cost. The ensemble quality Q_ens(S,x) measures the expected quality of ensembling models in S because we estimate it from held-out data with the same prompt distribution; this assumption fails when the calibration set does not cover the prompt's cluster, in which case Q_ens reflects the calibration set's average rather than the prompt-specific quality. The cascade transition P(correct|m,x) measures the probability that model m answers correctly because we estimate it from held-out data with the same prompt distribution; this assumption fails when the model's correctness is correlated with prompt difficulty in a way not captured by x, in which case the transition probability is miscalibrated. The LCB estimates Q_ens_LCB and P_LCB measure conservative lower bounds on the true ensemble quality and correctness probability because they subtract beta times the standard error from the point estimates; this assumption fails when the standard errors are underestimated (e.g., due to model misspecification), in which case the LCB may not be a true lower bound and the risk-averse objective may still be optimistic.

## Contribution

['A joint decision framework (JOINT) that unifies routing, cascading, and ensembling into a single cost-aware optimization over a DAG, solved exactly by dynamic programming without surrogate ensemble models.', 'A calibration procedure that estimates per-model quality and pairwise error correlations from a small held-out set, enabling the DP to use unbiased estimates of ensemble quality.', 'An empirical demonstration that joint optimization yields lower cost for a given quality target than separable routing-cascading-ensembling pipelines across multiple LLM routing benchmarks.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Four LLM routing benchmarks (in-dist + OOD) | Matches PISA-E comparison, exposes OOD bias |
| Primary metric | Macro-average cost-performance hypervolume | Jointly rewards cheap and high-quality routing |
| Baseline | PISA | Tests whether separate cascade+routing loses coupling |
| Baseline | PISA-E | Tests whether surrogate-based ensemble selection loses accuracy |
| Baseline | Fixed ensemble (size 3) | Tests whether adaptivity beats static ensembling |
| Ablation-of-ours | JOINT without calibrated q_k (uncalibrated) | Isolates value of calibration in the DP |
| Ablation-of-ours | JOINT with point estimates (beta=0) | Isolates value of LCB risk-averse objective |
| Ablation-of-ours | JOINT with LCB but no coverage diagnostic | Isolates value of coverage flag |
| Scalability experiment | K=8 models, exact DP vs. beam-search approximation (beam width=10) | Tests practical relevance at larger model pools |
| Calibration-size sensitivity | N_cal ∈ {128, 256, 512, 1024, 2048} | Measures DP-optimality gap vs. oracle |
| Oracle baseline | DP with true Q_ens and P (estimated from full test set) | Upper bound on achievable hypervolume |

### Why this setup validates the claim

The central claim is that coupling cascade, routing, and ensembling inside one exact DP yields lower expected cost at fixed quality than separable pipelines, and that this advantage hinges on calibrated per-model estimates. PISA tests the separability sub-claim: it routes and cascades but does not jointly optimize ensemble membership with cascade state, so any hypervolume gap between JOINT and PISA isolates the value of the coupling. PISA-E tests the surrogate sub-claim: it uses a surrogate for ensemble success, so a gap between JOINT and PISA-E isolates the value of direct calibrated estimation over a surrogate. The fixed-ensemble baseline tests whether adaptivity itself matters, guarding against the possibility that any ensemble is doing the work. The uncalibrated ablation is the diagnostic lever: if the DP's advantage came from exact optimization rather than from calibration, removing calibration should not collapse the gain. The point-estimate ablation (beta=0) tests whether the LCB risk-averse objective is necessary to preserve the gain under estimation error; if the LCB version outperforms the point-estimate version, it supports the load-bearing assumption's repair. The coverage-diagnostic ablation tests whether flagging out-of-coverage prompts and falling back to point estimates helps or hurts, isolating the value of the diagnostic. The scalability experiment with K=8 and beam-search approximation tests whether joint optimality is achievable at realistic model-pool sizes and reports the hypervolume gap between exact DP and the approximation. The calibration-size sensitivity experiment measures how the DP-optimality gap (vs. the oracle with true Q_ens and P) varies with calibration set size, providing an empirical estimation-error bound under which the DP remains near-optimal. Macro-average cost-performance hypervolume is the right metric because it directly encodes the cost-quality tradeoff the DP optimizes; a method that is merely better on quality or merely cheaper would not show a hypervolume shift. Evaluating both in-distribution and OOD forces the failure mode the design predicts (biased q_k under distribution shift) to surface.

### Expected outcome and causal chain

**vs. PISA** — On a case where a cheap model is likely correct alone but a moderately costly second model would substantially raise ensemble quality, PISA commits to a route or cascade before deciding ensemble membership, so it either over-spends on a cascade step it did not need or under-ensembles. Our method's DP sees the joint state (which models invoked, remaining budget) and chooses the ensemble-or-continue action that minimizes expected cost subject to quality, so we expect a visible hypervolume gap concentrated on prompts where single-model correctness is borderline (roughly the middle difficulty band), with near-parity on trivially easy and hopelessly hard prompts.

**vs. PISA-E** — On a case where two models have correlated errors on a prompt cluster, the surrogate in PISA-E predicts ensemble success from features that do not capture that correlation, so it selects an ensemble whose true quality is below target and the cost-quality point is misestimated. Our method estimates the joint error distribution from held-out data, so Q_ens reflects the actual correlation and the DP avoids the redundant ensemble, so we expect a gap concentrated on prompt clusters with high pairwise error correlation and parity on clusters where models err independently.

**vs. Fixed ensemble (size 3)** — On an easy prompt where a single small model answers correctly, the fixed ensemble pays three model costs for marginal quality gain, so its hypervolume is dragged down by overspending on the easy mass of prompts. Our DP terminates after one model when the quality constraint is already met, so we expect the gain to appear as a cost reduction on the easy subset with quality parity, and the fixed ensemble to match or slightly beat us only on the hardest subset where all three models are needed.

**vs. JOINT with point estimates (beta=0)** — On prompts where the calibration set is small or the prompt is near the coverage boundary, point estimates can be optimistic and lead the DP to choose an action that appears to meet the quality constraint but actually fails. Our LCB version subtracts one standard error, so it chooses more conservative actions (e.g., invokes an extra model or avoids a risky ensemble) that are more likely to meet the true quality constraint. We expect the LCB version to have higher hypervolume than the point-estimate version, with the gap concentrated on prompts with high estimation variance (e.g., rare clusters or small N_cal).

**vs. JOINT with LCB but no coverage diagnostic** — On OOD prompts that fall outside calibration coverage, the LCB estimates may still be biased because the standard errors are underestimated. The coverage diagnostic flags these prompts and falls back to point estimates, which may be less conservative but avoids over-penalizing. We expect the coverage diagnostic to improve hypervolume on OOD benchmarks, with the gain concentrated on prompts with large nearest-calibration-cluster distance.

**vs. Oracle (true Q_ens and P)** — The oracle represents the best achievable hypervolume with perfect knowledge. We expect JOINT with LCB to approach the oracle as N_cal increases, with the gap shrinking from N_cal=128 to N_cal=2048. The calibration-size sensitivity experiment will report this gap, providing an empirical estimation-error bound: we expect the gap to be within 5% of the oracle hypervolume for N_cal >= 512 on in-distribution prompts, and within 10% for OOD prompts.

**Scalability (K=8, exact vs. beam-search)** — Exact DP with K=8 and ensemble size <=3 has a state space of sum_{i=0}^{3} C(8,i) = 1+8+28+56 = 93 subsets, which is tractable. We expect exact DP to be feasible and to outperform beam-search approximation (beam width=10) by a small margin (e.g., <2% hypervolume gap), demonstrating that joint optimality is achievable at realistic model-pool sizes.

### What would falsify this idea

If JOINT's hypervolume gain over PISA and PISA-E is uniform across difficulty bands and error-correlation clusters rather than concentrated where the coupling and calibration failure modes predict, then the DP is not exploiting the structure we claim and the central argument is wrong. Likewise, if removing calibration (the ablation) leaves the gain essentially intact, the claimed role of calibrated estimates in the DP's optimality is falsified. Additionally, if the LCB risk-averse objective (beta=1.0) does not outperform the point-estimate version (beta=0) under small N_cal or OOD, then the load-bearing assumption's repair is ineffective and the DP's optimality claim is undermined.

## References

1. Ensemble Routing for Prompt-Adaptive Cost-Aware LLM Inference
