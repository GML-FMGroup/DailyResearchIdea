# Monotone Constraint-Potential Co-Evolution for Self-Improving Language Agents

## Motivation

Across the constraint-following and self-improving-agent literature, evaluation is treated as exogenous and terminal: AgentIF measures constraint violations against static human-annotated criteria and never feeds those signals back to adapt the verifier, while HGM optimizes a coding-agent policy against a fixed benchmark evaluator assumed to be exogenous. The structural gap is that violation signals are dead-end measurements — no mechanism jointly updates the policy and the constraint-verification criteria, and no condition prevents the degenerate fixpoint where the policy games a relaxed verifier. This recurs across both paper-trees because neither treats the evaluator as a co-evolving variable coupled to the policy it measures.

## Key Insight

If each constraint in the verifier carries a latent satisfaction potential shared by the policy and the constraint set, then the joint update becomes a monotone operator on a single potential, so the co-evolution converges to a non-degenerate fixed point where constraints are neither frozen nor relaxed into vacuity.

## Method

**What it is.** COPE (Constraint-Potential co-Evolution) takes an agent policy π, a constraint set C extracted from an instruction, and execution traces, and returns an updated policy and an updated constraint set whose joint fixed point is non-degenerate. Inputs: instruction I, constraint set C₀ (from an AgentIF-style extractor), execution rollout buffer. Outputs: π*, C*.

**How it works.**
```pseudocode
# Each constraint c has a latent potential phi_c in [0,1], initialized 0.5
# Potential of a trace = sum_c phi_c * 1[trace satisfies c] / sum_c phi_c
# Calibration: per constraint family f, a held-out human-verified set H_f of 200 (trace, c, label) triples
#   is used to compute judge-vs-human agreement a_f = fraction of H_f where judge s_c equals human label.
#   A constraint c in family f is 'calibrated' iff a_f >= 0.85; uncalibrated constraints cannot trigger relaxation.
# Independent check: for each constraint c, a programmatic/executable check p_c(tau) in {0,1} is defined
#   (e.g., regex for format, length counter, AST check for code) and is run alongside the LLM judge.
#   Relaxation requires agreement: s_c(tau) == p_c(tau) for all tau in the current rollout batch.
# Stationarity: within each iteration, the verifier criteria for all c are FROZEN (no relaxation/tightening)
#   during the measure phase (steps 1-3); relaxation/tightening (step 4) is applied only after phi is updated
#   and the operator is evaluated against the frozen verifier of that iteration.
repeat until ||phi - phi_prev||_1 < eps (eps=1e-3) or T=20 iters:
  # 1. Rollout policy against current constraints (verifier criteria frozen for this iteration)
  for each instruction I: sample K=8 traces tau ~ pi(.|I)
  # 2. Compute per-constraint satisfaction indicator s_c(tau) in {0,1}
  #    via the current verifier (LLM-judge + programmatic checks); also compute p_c(tau)
  # 3. Update potentials with a monotone operator:
  #    phi_c <- clip( phi_c + eta * (mean_tau s_c(tau) - phi_c) - lambda * relax_penalty_c , 0, 1)
  #    eta=0.1, lambda=0.05; relax_penalty_c = 1 if c was auto-relaxed this round else 0
  # 4. Constraint refinement (applied AFTER phi update, against frozen verifier of this iteration):
  #    a constraint c is RELAXED (verifier criterion loosened) ONLY if
  #      phi_c > 0.9 AND violation rate < 2% AND c is calibrated (a_f >= 0.85)
  #      AND judge-programmatic agreement holds on all traces in the batch;
  #    a constraint c is TIGHTENED (criterion made stricter) if phi_c < 0.2.
  # 5. Policy update: weighted SFT/DPO on traces, weight w(tau)=potential(tau)
  #    so high-potential traces dominate the gradient
  # 6. Recompute phi; assert monotone: potential(pi_{t+1}) >= potential(pi_t) - delta
end
return pi*, C* = argmax over iterates of mean potential
```

**Why this design.** We chose a shared scalar potential φ over independent per-constraint verifier tuning because independent tuning lets the policy and verifier drift apart (the policy games whichever constraint is cheapest to relax), accepting the cost that a single potential cannot represent constraint-specific difficulty. We chose to gate relaxation/tightening on φ crossing hard thresholds (0.9 / 0.2) rather than on a learned gate because a learned gate reintroduces the degenerate-solution risk that AgentIF's static verifier already exposes, accepting the cost that thresholds must be set per constraint family. We chose potential-weighted policy updates over uniform updates because uniform weighting lets easy constraints dominate the gradient and starves hard ones, accepting the cost that hard constraints receive little signal until their φ rises. A domain expert would not describe COPE as HGM-plus-AgentIF: HGM expands a tree of self-modifications against a fixed benchmark, whereas COPE never expands a tree — it updates the verifier and policy on the same potential surface, so the verifier is a co-evolving variable, not an exogenous evaluator.

**Why it measures what we claim.** The per-constraint potential φ_c measures the constraint's latent satisfaction level because we assume the verifier's satisfaction indicator s_c(τ) is a faithful (if noisy) readout of whether the trace genuinely satisfies c; this assumption fails when the LLM-judge verifier is itself biased (e.g., verbosity bias, as documented for LLM-as-a-judge), in which case φ_c reflects judge preference rather than true satisfaction. The potential-weighted policy gradient measures policy improvement on the coupled objective because we assume high-potential traces are causally responsible for constraint satisfaction rather than merely correlated with it; this fails when a trace satisfies constraints by luck (short traces trivially satisfy length constraints), in which case the weight reflects trace brevity rather than genuine compliance. The monotone relaxation gate measures non-degeneracy because we assume a constraint with φ_c > 0.9 and near-zero violation is genuinely saturated rather than gamed; this fails when the policy has learned a shortcut that satisfies the verifier but not the instruction, in which case relaxation removes a real constraint.

**(D) Potential-to-causal-credit equivalence.** The potential-weighted gradient measures causal credit for constraint satisfaction because we assume trace potential is not confounded by trace length/brevity or luck; this assumption fails when short traces trivially satisfy length constraints and dominate the weight, in which case the weight reflects trace brevity rather than genuine compliance. To test this, we add a length-stratified reweighting ablation (see experiment).

## Contribution

(1) COPE, a closed-loop co-evolution algorithm that couples policy improvement and constraint-verification refinement through a shared per-constraint satisfaction potential with a monotone update operator. (2) The design principle that verifier adaptation must be gated on potential saturation and violation-rate noise floors, not on raw violation counts, to avoid the degenerate fixed point where the policy relaxes its own constraints. (3) A reproducible protocol for measuring non-degeneracy of co-evolved verifiers (constraint-set drift, potential trajectory monotonicity) on AgentIF-style agentic instruction-following tasks.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | AgentIF constraint suite | Instruction-following with explicit per-constraint criteria |
| Primary metric | Constraint satisfaction rate + relaxed-constraint fraction | Jointly detects improvement vs. degenerate relaxation |
| Baseline | Static AgentIF verifier | Fixed verifier, no co-evolution |
| Baseline | HGM-style self-modification | Tree search against fixed benchmark |
| Baseline | Uniform-weight policy update | No potential weighting of traces |
| Baseline | Freeze-policy (verifier-only tuning) | Isolates necessity of joint co-evolution |
| Baseline | Freeze-verifier (policy-only tuning) | Isolates necessity of joint co-evolution |
| Ablation-of-ours | COPE with relaxation gate disabled | Isolates whether gating prevents degeneracy |
| Ablation-of-ours | COPE with length-stratified reweighting | Tests potential-to-causal-credit confound |
| Calibration set | 200 human-verified triples per constraint family | Enables judge-vs-human agreement check |

### Why this setup validates the claim

The central claim is that jointly evolving the constraint set and policy on a shared potential avoids the degenerate fixed points that arise when either is tuned alone. AgentIF supplies the right substrate because its constraints are explicit, per-instruction, and verifiable, so each constraint's satisfaction indicator is a clean readout rather than a holistic score. The static-verifier baseline tests the sub-claim that a frozen verifier lets the policy drift toward whatever the verifier rewards; if COPE's gain over it is absent, co-evolution buys nothing. The HGM-style baseline tests the sub-claim that tree expansion against a fixed benchmark is the wrong axis; if COPE does not beat it on hard-constraint subsets, the co-evolution framing is unnecessary. The uniform-weight baseline tests whether potential weighting actually routes gradient toward hard constraints. The freeze-policy and freeze-verifier baselines test whether joint co-evolution is necessary rather than incidental: if either alone matches COPE, the joint update is not the cause. The relaxation-disabled ablation is the diagnostic core: it isolates whether the monotone gate is what prevents the degenerate "relax everything" outcome. The length-stratified reweighting ablation tests the (D) sub-block: if COPE's advantage disappears when trace weights are stratified by length, the potential-to-causal-credit equivalence is confounded by brevity. The paired metric — satisfaction rate alongside relaxed-constraint fraction — is essential because satisfaction alone can be inflated by relaxation; the pair makes degeneracy visible as high satisfaction with high relaxation, which is exactly the failure the method claims to avoid. The calibration set (200 human-verified triples per constraint family) operationalizes judge faithfulness: we report judge-vs-human agreement per constraint and require agreement ≥0.85 before any relaxation, and we report the fraction of relaxations that survive the independent programmatic check.

### Expected outcome and causal chain

**vs. Static AgentIF verifier** — On a case where a hard constraint (e.g., a structural output requirement) coexists with many easy ones, the static verifier produces high aggregate satisfaction by the policy satisfying easy constraints and quietly failing the hard one, because its fixed criteria never re-weight pressure toward the hard constraint. Our method instead raises the hard constraint's φ through potential-weighted updates once easy constraints saturate, because the shared potential routes gradient toward unsatisfied mass, so we expect a noticeable gap on hard-constraint subsets but rough parity on easy-only subsets.

**vs. HGM-style self-modification** — On a case where the policy needs to revise its own behavior to meet a newly emphasized constraint, the HGM baseline produces repeated self-modifications that optimize benchmark score without touching the verifier, because its evaluator is exogenous and fixed, so it cannot discover that the constraint itself is the binding variable. Our method instead co-updates the verifier criterion and the policy on the same surface, because relaxation/tightening is gated on φ, so we expect faster saturation of hard constraints with fewer iterations and no score inflation from verifier drift.

**vs. Uniform-weight policy update** — On a case where one constraint is nearly saturated and another is far from it, the uniform baseline produces a gradient dominated by the easy constraint's abundant satisfying traces, because every trace contributes equally regardless of potential. Our method instead weights traces by their potential, because w(τ) concentrates signal on traces that satisfy high-φ constraints, so we expect the hard constraint's satisfaction to rise markedly while the easy constraint's stays flat — a divergence pattern absent under uniform weighting.

**vs. Freeze-policy (verifier-only tuning)** — On a case where the verifier criteria are relaxed without policy adaptation, the freeze-policy baseline produces high relaxed-constraint fraction with little satisfaction gain, because the policy cannot exploit the relaxed criteria without updates. Our method instead co-updates both, because the policy gradient follows the potential surface, so we expect higher satisfaction at lower relaxation fraction.

**vs. Freeze-verifier (policy-only tuning)** — On a case where the policy is updated against a fixed verifier, the freeze-verifier baseline produces saturation on easy constraints but stalls on hard ones, because the fixed criteria never re-weight pressure. Our method instead adapts the verifier via φ-gated relaxation/tightening, so we expect higher hard-constraint satisfaction without degenerate relaxation.

**vs. COPE with relaxation gate disabled** — On a case where the gate is disabled, the ablation produces high relaxed-constraint fraction and inflated satisfaction, because relaxation fires without φ corroboration. Our method instead gates relaxation on φ>0.9, calibration, and judge-programmatic agreement, so we expect lower relaxed-constraint fraction at comparable satisfaction.

**vs. COPE with length-stratified reweighting** — On a case where short traces trivially satisfy length constraints, the length-stratified ablation removes the brevity confound by reweighting traces within length strata. If COPE's advantage over uniform weighting disappears under stratification, the potential-to-causal-credit equivalence is confounded by brevity; we expect the advantage to persist if the weight reflects genuine compliance.

### What would falsify this idea

If the relaxation-disabled ablation matches full COPE on both satisfaction rate and relaxed-constraint fraction, the monotone gate is not doing the work and the degeneracy claim is wrong. If COPE's gains are uniform across easy and hard constraint subsets rather than concentrated on hard subsets, the potential-weighting mechanism is not the cause of improvement. If the length-stratified reweighting ablation eliminates COPE's advantage, the potential-to-causal-credit equivalence is confounded by trace brevity.

## References

1. AGENTIF: Benchmarking Instruction Following of Large Language Models in Agentic Scenarios
2. Huxley-Gödel Machine: Human-Level Coding Agent Development by an Approximation of the Optimal Self-Improving Machine
3. CFBench: A Comprehensive Constraints-Following Benchmark for LLMs
4. Judging LLM-as-a-judge with MT-Bench and Chatbot Arena
5. COLLIE: Systematic Construction of Constrained Text Generation Tasks
