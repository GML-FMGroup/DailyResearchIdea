# DualCheck: Autonomous Closed-Loop Learning for Operations Research Language Models via Primal-Dual Consistency Verification

## Motivation

Existing methods for training operations research (OR) language models rely on either external solvers for verification (e.g., OptiChat in 'Diagnosing infeasible optimization problems using large language models' uses a solver for IIS computation) or large human-annotated instruction datasets (e.g., 'Scaling Instruction-Finetuned Language Models'). This dependence prevents autonomous, scalable learning because it requires manual curation or solver integration. The structural problem is the absence of a self-verification signal that an LLM can compute without external computation or human labels.

## Key Insight

For convex linear optimization problems, the complementary slackness conditions of a primal-dual pair provide a necessary and sufficient optimality certificate that can be checked entirely within the LLM's outputs without any external solver.

## Method

### (A) What it is
DualCheck is a self-supervised fine-tuning framework that cycles through problem generation, primal and dual solution generation, verification via duality conditions, and filtering verified examples as training data. Its inputs are a base LLM and a template library for problem generation; its output is a fine-tuned LLM that can generate and self-verify OR solutions. **Load-bearing assumption**: The LLM-generated problem strings accurately encode a valid linear optimization problem in standard form with correct constraint coefficients and dimensions. To mitigate this, we insert a concrete validator: a formal parser that checks syntactic correctness (e.g., matching parentheses, defined variables) and semantic validity (e.g., dimension consistency of A, b, c) before any further processing. Problems that fail validation are discarded and not used in data generation.

### (B) How it works
```pseudocode
Algorithm: DualCheck
Input: Base LLM M, Problem template library T, Validator V (formal parser)
Output: Fine-tuned LLM M'

Phase 1: Data Generation
for step in 1..N:  # N = 10,000 for initial experiments, 100,000 for full
  t = sample(T)
  p = M(t, prompt="Generate a linear optimization problem in standard form")
  p_valid = V.parse_and_validate(p)  # Discards malformed problems; V is a deterministic parser that checks syntax (e.g., variable names, dimensions) and semantics (e.g., A is m×n, b is m×1, c is n×1)
  if not p_valid: continue
  x = M(p, prompt="Solve the primal problem: output a feasible primal solution vector")
  y = M(p, prompt="Solve the dual problem: output a feasible dual solution vector")
  feas_p = check_feasibility(p, x)          # primal constraints satisfied? (tolerance = 1e-6)
  feas_d = check_feasibility_dual(p, y)     # dual constraints satisfied? (tolerance = 1e-6)
  comp_slack = sum(x_i * (c_i - (A^T y)_i))  # complementary slackness (tolerance = 1e-6)
  v = feas_p + feas_d + comp_slack          # violation (0 iff optimal, within tolerance)
  if v == 0:
    add (p, x, y) to dataset D_verified

Phase 2: Fine-tuning
  Fine-tune M on D_verified using standard supervised fine-tuning with instruction templates: "Problem p: provide primal and dual solutions."
```
Hyperparameters: N=10,000 (initial) / 100,000 (final); violation threshold exact 0; primal/dual feasibility and complementary slackness checked with numerical tolerance of 1e-6 (we will also conduct sensitivity analysis with tolerances 1e-4, 1e-3).

### (C) Why this design
We chose to generate both primal and dual solutions from the same model (rather than using a dedicated solver for one side) because this forces the model to learn the duality mapping internally, accepting the cost that early iterations may produce infeasible pairs that are filtered out, reducing data efficiency. We used exact zero violation as the filter (instead of a slack threshold) to avoid incorrect positives from near-optimal but non-binding solutions, at the risk of discarding numerically acceptable optimal pairs. We adopted a template library for problem generation (rather than free-form generation) to ensure that problems are linearly convex and have known dual formulations, limiting generality but guaranteeing the validity of the verification signal. We deliberately avoided external solvers (unlike OptiChat) to maintain autonomy; this trades off the initial high-quality verification from solvers for the ability to scale without solver availability.

### (D) Why it measures what we claim
The violation v measures correctness of the solution pair because, for convex linear problems, strong duality holds and complementary slackness is necessary and sufficient for optimality; this assumption fails for non-convex or integer problems where a duality gap exists, in which case v measures only the violation of the dualized constraints rather than true optimality. The feasibility checks (primal and dual) measure whether the generated solutions lie in the feasible set, assuming that constraints are correctly specified in the problem string p; this assumption fails when the LLM generates a problem with inaccurate constraints, leading to false verification. The filtering step (v==0) operationalizes the concept of self-verification; it assumes exact arithmetic, which fails due to floating-point errors, causing some optimal solutions to be discarded. The parser V ensures that only syntactically and semantically valid problems are used, mitigating the main failure mode of malformed problems.

### (E) Generalization assumption (effectiveness)
The filtered training pairs represent all optimal solutions only if the template library covers the test distribution. The assumption is that the template library generates problems with similar constraint structure to the test set (e.g., Netlib). The failure mode is distribution shift: if test problems have different structure (e.g., network flow vs. portfolio optimization), the model may not generalize. To quantify this, we include experiments on out-of-template test problems from diverse OR domains.

## Contribution

(1) A self-verification mechanism for OR-LLM agents that uses primal-dual consistency as a training signal, eliminating the need for external solvers or human labels. (2) A closed-loop learning procedure where the LLM generates its own training data (problem instances and verified solutions) without manual curation. (3) The DualCheck framework that can be applied to any convex linear optimization problem class.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|-----------------------|
| Dataset | Held-out LP test problems (Netlib) | Real-world benchmark for generalization. |
| Primary metric | Solve rate (feasible + optimal) | Directly measures dual verification claim. |
| Baseline 1 | Standard SFT (no filter) | Tests value of verification filtering. |
| Baseline 2 | Solver-verified SFT | Tests autonomy vs. external solver. |
| Baseline 3 | Zero-shot prompting | Tests if fine-tuning is necessary. |
| Baseline 4 | Feasibility-only SFT (filter on feasibility, no complementary slackness) | Isolates novelty of complementary slackness. |
| Ablation of ours | No complementary slackness (same as Baseline 4) | Already covered. |
| Sensitivity | Variation of numerical tolerance (1e-4, 1e-3) | Quantifies impact of floating-point errors. |

Additionally, we use N=10,000 for initial experiments to test viability, and include out-of-template test sets (e.g., randomly selected problems from OR-Library) to assess distribution shift.

### Why this setup validates the claim
This setup tests the core claim that self-verification via duality conditions improves solution correctness. Using held-out real LP problems ensures the model generalizes beyond generated templates. Comparing to Standard SFT shows the benefit of filtering; Solver-verified SFT tests if the model can learn duality without an external oracle; Zero-shot baselines tests if fine-tuning itself adds value. The ablation removes complementary slackness to isolate its role. The feasibility-only baseline (Baseline 4) further isolates the contribution of complementary slackness beyond mere feasibility. Sensitivity analysis on tolerance quantifies how robust the verification signal is to numerical errors, addressing groundedness. Including out-of-template problems tests generalization under distribution shift, addressing the load-bearing assumption about coverage. Solve rate (feasibility + optimality) directly captures the target behavior: if DualCheck's internal verification is valid, it should outperform baselines, especially on problems requiring exact optimality.

### Expected outcome and causal chain

**vs. Standard SFT** — On a case where the primal and dual solutions are both feasible but not complementary slack, Standard SFT would keep the pair as training data, learning a spurious mapping. DualCheck filters it out, so the fine-tuned model learns only optimal pairs. We expect a noticeably higher solve rate for DualCheck, especially on large problems where infeasible pairs are common.

**vs. Solver-verified SFT** — On a case where no external solver is available (e.g., proprietary environment), Solver-verified SFT cannot generate data. DualCheck's self-verification works without any external tool, so it produces a fine-tuned model while Solver-based cannot. We expect DualCheck to achieve a solve rate above random on all test sets, whereas Solver-based SFT degrades to zero-shot in the absence of solver.

**vs. Zero-shot prompting** — On a case requiring precise dual solution, zero-shot often produces infeasible or non-optimal answers due to lack of training. DualCheck's fine-tuning on verified pairs teaches the model to produce correct complementary slackness. We expect a large gap: Zero-shot solve rate near 0%, DualCheck above 50% on typical problems.

**vs. Feasibility-only SFT** — On a case where feasibility is easy but optimality is hard (e.g., problems with many near-optimal feasible points), Feasibility-only SFT will include many suboptimal pairs, degrading performance. DualCheck's inclusion of complementary slackness should yield higher solve rates on problems requiring exact optimality. We expect DualCheck to outperform Feasibility-only SFT by at least 10 percentage points on Netlib problems.

### What would falsify this idea
If DualCheck's solve rate is not significantly higher than Standard SFT on problems requiring strict optimality, or if the gain is uniform across subsets instead of concentrated where complementary slackness is critical, then the verification filter is not effectively selecting optimal pairs. Additionally, if the sensitivity analysis shows that performance degrades sharply at higher tolerances (e.g., >1e-4), the verification signal is too fragile.

## References

1. Maybe Only 0.5% Data is Needed: A Preliminary Exploration of Low Training Data Instruction Tuning
2. Diagnosing infeasible optimization problems using large language models
3. DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model
4. JuMP 1.0: recent improvements to a modeling language for mathematical optimization
5. Scaling Instruction-Finetuned Language Models
