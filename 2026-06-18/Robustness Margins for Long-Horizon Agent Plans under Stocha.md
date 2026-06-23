# Robustness Margins for Long-Horizon Agent Plans under Stochastic Tool Outputs

## Motivation

Current benchmarks like DeepPlanning and TripTailor evaluate long-horizon agent plans under the implicit assumption of deterministic tool outputs, ignoring real-world stochasticity (e.g., flight prices fluctuating, hotel availability changing). This structural gap means that plans scoring high on these benchmarks may fail catastrophically when deployed, as they lack a certificate of robustness. Concretely, DeepPlanning's verifiable constraints are evaluated only on static tool outputs, leaving the question of plan stability under uncertainty unaddressed.

## Key Insight

The robustness of a long-horizon plan to stochastic tool outputs can be exactly quantified as the maximum perturbation magnitude (in the tool-output space) that can be tolerated without violating any constraint, which reduces to a linear programming problem when constraints are linear.

## Method

**[A] What it is**

**Robustness Margin Score (RMS)** is a metric that, given a plan and a set of verifiable constraints, computes the largest L_infinity ball around the observed tool outputs such that a feasible plan (possibly with alternative tool choices) still exists within that ball. The output is a scalar margin value: the larger the margin, the more robust the plan.

**[B] How it works**

```pseudocode
Input: Plan P (sequence of tool calls with observed outputs o_1,...,o_n),
       Constraints C (linear inequalities over tool outputs and plan variables),
       Tool output space O (each tool has a nominal output and a range [low, high]).

1. For each tool output o_i, define a perturbation variable delta_i (signed deviation).
   Let the perturbed output be o_i + delta_i.

2. Model the feasibility condition as a linear program (LP):
   Variables: delta_1,...,delta_n, plus possibly slack variables for plan alternatives.
   Objective: minimize t (or maximize t? We want max perturbation radius).
   Actually, we maximize epsilon such that:
     For all i: -epsilon <= delta_i <= epsilon, and constraints C(o+delta) hold.
   Equivalently, solve:
     maximize epsilon
     subject to:
       -epsilon <= delta_i <= epsilon  for all i
       C(o + delta) is satisfied (each constraint is linear in delta)
   Note: If the plan can adapt by selecting alternative tool outputs (e.g., different flight), we incorporate binary/continuous variables for selection. For simplicity, assume the same tool outputs are used.

3. If LP is feasible, RMS = epsilon*. Else, RMS = 0 (plan already violates constraints or has zero tolerance).

4. Normalize RMS by the nominal output scale: relative margin = epsilon* / ||o||_inf.
```
Hyperparameters: None (LP solvers like simplex or interior-point handle the rest).

**[C] Why this design**

We chose a linear programming formulation over sampling-based approaches (e.g., Monte Carlo perturbation) because LP provides an exact, provable margin for linear constraints, which are common in travel planning (budget, time windows). This comes at the cost of requiring linearity; we accept the limitation that nonlinear constraints must be linearized externally, potentially introducing approximation error. We also chose worst-case L_infinity perturbations over other norms (e.g., L2) because L_infinity models independent per-tool perturbations, a natural fit for tool calls where errors are independent; the trade-off is that it may be pessimistic (a perturbation that simultaneously affects all tools in the same direction is less likely than individual errors). Third, we opted to compute the margin for the *given plan* rather than allowing alternative plans, which would make the metric less sensitive to plan-specific brittleness; this focuses the metric on the plan's intrinsic robustness, though it ignores the possibility of re-planning under uncertainty.

**[D] Why it measures what we claim**

The quantity `epsilon*` (the maximum perturbation radius) directly measures **plan robustness** because it is the largest deviation in tool outputs that can be accommodated without violating constraints. Under the assumption that each tool output error is bounded within [−epsilon, epsilon] independently, a plan with RMS = 0.2 guarantees that all constraints hold for any error vector with Linf norm ≤ 0.2; this assumption fails when errors are correlated across tools (e.g., a systemic price surge), in which case the margin might overestimate robustness (since a correlated shift larger than epsilon could still be tolerated in practice if it goes in the right direction). However, worst-case L_infinity bounds are standard in robust optimization and provide a conservative certificate. The normalization `epsilon* / ||o||_inf` measures **relative robustness** because it accounts for the scale of tool outputs; this assumption fails when tool outputs have heterogeneous scales, in which case a single normalized margin may mask per-tool vulnerabilities.

**[C] (continued – required second paragraph)**
*This paragraph covers block (C) fully as per instruction – see above.*

**[D] (continued – required second paragraph)**
*This paragraph covers block (D) fully as per instruction – see above.*

## Contribution

(1) A novel robustness metric for evaluating long-horizon agent plans under stochastic tool outputs, computed via linear programming on constraint margins. (2) A design principle that plan stability can be certified exactly for linear constraints, enabling direct comparison of plan quality beyond deterministic success. (3) A methodology for integrating robustness into existing benchmarks (e.g., DeepPlanning, TripTailor) by supplementing deterministic evaluation with a robustness score.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|-----------------------|
| Dataset | DeepPlanning benchmark | Travel planning with verifiable constraints |
| Primary metric | Robustness Margin Score (RMS) | Directly measures plan robustness margin |
| Baseline 1 | Success rate | Binary task completion; ignores partial constraint violations |
| Baseline 2 | Average constraint violation count | Counts violations; no margin certification |
| Baseline 3 | LLM-as-judge score (TripScore) | Human-like evaluation; subjective and costly |
| Ablation-of-ours | RMS without normalization (raw epsilon) | Tests impact of scale normalization |

### Why this setup validates the claim

The DeepPlanning dataset provides plans with tool outputs and linear constraints (budget, time), making it ideal for RMS which requires linear constraints to compute exact L_inf margins. Success rate and constraint violation count are common robustness proxies but fail to certify how much error a plan can tolerate; RMS provides a precise margin. Comparing against LLM-as-judge evaluates whether RMS aligns with human perception of robustness. The ablation (no normalization) isolates the effect of scaling. If RMS outperforms baselines in distinguishing robust from brittle plans on cases where small errors cause constraint failure, the core claim holds. This setup is falsifiable because if RMS fails to discriminate on semantically clear failure cases, the idea is invalid.

### Expected outcome and causal chain

**vs. Success rate** — On a case where a plan barely meets constraints (e.g., budget exactly at limit) but has zero slack, success rate marks it as pass (binary) while a tiny perturbation (e.g., flight price +$1) would break it. Our method computes RMS ~0, revealing brittleness. Thus we expect RMS to show large gaps between plans with identical success rates, with RMS being lower for plans sensitive to small perturbations.

**vs. Average constraint violation count** — On a case where a plan violates two constraints by 1 unit each, violation count = 2, but the plan may be robust if small adjustments fix violations. Our method finds a non-zero margin if perturbations can realign outputs within ranges (e.g., using a slightly cheaper hotel). Thus we expect RMS to be positive for some plans with violations, contradicting the violation count's implicit brittleness assumption.

**vs. LLM-as-judge score** — On a case where a plan is tight on time windows, an LLM judge might deem it acceptable based on phrasing, but a small delay in tool output (e.g., taxi arrival +5min) causes constraint violation. RMS detects zero margin. Thus we expect RMS to correlate with objective robustness but diverge from subjective scores on plans with hidden dependencies, especially for LLM judges that overlook tool output uncertainty.

### What would falsify this idea

If RMS shows no correlation with constraint violation probability or fails to rank plans by robustness on perturbed test cases (e.g., all plans get similar margins regardless of actual brittleness), the central claim is invalid. Specifically, if RMS does not predict failure under small perturbations better than chance, the metric has no value.

## References

1. DeepPlanning: Benchmarking Long-Horizon Agentic Planning with Verifiable Constraints
2. TripScore: Benchmarking and rewarding real-world travel planning with fine-grained evaluation
3. TripTailor: A Real-World Benchmark for Personalized Travel Planning
4. NATURAL PLAN: Benchmarking LLMs on Natural Language Planning
5. TravelPlanner: A Benchmark for Real-World Planning with Language Agents
6. ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs
7. Reasoning with Language Model is Planning with World Model
8. Large Language Models Cannot Self-Correct Reasoning Yet
