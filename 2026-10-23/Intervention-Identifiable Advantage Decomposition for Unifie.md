# Intervention-Identifiable Advantage Decomposition for Unified LLM Routing

## Motivation

RouteWeaver's COMET-GRPO splits trajectory-level outcome reward into mode-selection and execution advantages using a hand-designed decomposition, and it requires externally assigned modes during progressive cold-start. This is a symptom of a structural problem that recurs across outcome-reward credit-assignment work (e.g., ReTool's outcome-only RL, Hybrid LLM's difficulty router): the decomposition is asserted rather than identified, so no one can verify that the two advantage signals recover the true causal contributions rather than a convenient correlated surrogate. When the decomposition is unfalsifiable, cold-start supervision becomes the only way to anchor the modes, which is exactly the assumption RouteWeaver must relax.

## Key Insight

If the learned decomposition is constrained to be recoverable by a known synthetic intervention on mode choice, then the two advantages are identified up to the intervention's causal effect, making outcome-only end-to-end learning of the decomposition possible without externally assigned modes.

## Method

## Advantage Decomposition by Interventional Identifiability (ADII)

**(A) What it is.** ADII replaces RouteWeaver's hand-designed COMET-GRPO advantage split with a learned two-head decomposition `(A_mode, A_exec)` produced by a small gating network over the routing trajectory, trained end-to-end on outcome-only rewards, and regularized by an *interventional consistency loss* computed on synthetically mode-intervened rollouts so that the learned split is identifiable rather than arbitrary. Inputs: routing trajectories (query, selected mode, model, execution trace, outcome reward). Outputs: per-trajectory mode-selection advantage and execution advantage, plus a scalar identifiability score used as a training diagnostic.

**(B) How it works.**

```pseudocode
# Phase 1: outcome-only rollouts (no external mode labels)
for each query q:
    tau = policy.rollout(q)              # tau = (m, model, exec_trace, r)
    r   = outcome_reward(tau)            # scalar, no mode supervision

# Phase 2: synthetic mode intervention (the identifiability anchor)
for each tau:
    m'   = sample_mode(q) != tau.m       # counterfactual mode
    tau' = policy.rollout(q, force_mode=m')
    r'   = outcome_reward(tau')
    Delta_mode(q) = r - r'               # interventional mode effect

# Phase 3: learned decomposition head
phi   = trajectory_encoder(tau)          # shared trunk: 2-layer MLP, hidden=256, GeLU
A_mode = g_mode(phi)                     # scalar head: linear(256->1)
A_exec = g_exec(phi)                     # scalar head: linear(256->1)

# Phase 4: losses (end-to-end, outcome-only)
L_policy  = GRPO(A_mode + A_exec, r)     # standard GRPO on summed advantage, clip=0.2
L_ident   = || A_mode(tau) - A_mode(tau') - Delta_mode(q) ||^2   # interventional consistency
L_exec    = || A_exec(tau) - (r - A_mode(tau)) ||^2               # execution absorbs residual
L_total   = L_policy + lambda_ident * L_ident + lambda_exec * L_exec
# lambda_ident = 0.5, lambda_exec = 0.1, GRPO clip = 0.2
```

**(C) Why this design.** We chose a *synthetic mode intervention* over a pure independence-head regularizer because mode selection and execution are operationally coupled in real routing data (the same query can succeed under multiple modes), so an independence assertion would be unidentifiable — the intervention instead supplies the causal contrast that pins down `A_mode`. We chose to make `A_exec` absorb the residual `r - A_mode` rather than train it against its own reward because the residual is the only quantity that is definitionally orthogonal to the interventional mode effect, which keeps the two heads from collapsing onto the same signal. We chose a shared trajectory encoder with two scalar heads over two separate encoders because the mode and execution signals must share the same sufficient statistic (the trajectory) to be comparable, accepting the cost that a shared trunk can leak information between heads — mitigated by the residual loss. This differs from VinePPO-style per-step Monte-Carlo value rollouts and from Process Reward Models: those estimate step values from prefix rewards, whereas ADII estimates a *causal contrast between two whole-trajectory outcomes under a forced mode*, which no prefix-value method produces. A domain expert would not describe ADII as a variant of COMET-GRPO because the decomposition is not hand-designed — it is defined by the intervention it must reproduce.

**(D) Why it measures what we claim.** `Delta_mode(q) = r - r'` measures the *causal contribution of mode selection* because the only thing that differs between `tau` and `tau'` is the forced mode, under the assumption that the policy's execution behavior is conditionally independent of mode given the query; this assumption fails when the policy's execution trace is itself mode-conditioned (e.g., agentic mode changes the number of tool calls), in which case `Delta_mode` reflects the joint mode-and-execution effect rather than mode alone. `A_mode(tau) - A_mode(tau')` measures the *learned mode-selection advantage* because it is trained to match `Delta_mode`, under the assumption that the encoder `phi` is invariant to the intervention mechanism; this fails when the encoder memorizes the forced-mode token, in which case the consistency loss is satisfied trivially and the identifiability score is inflated. `A_exec = r - A_mode` measures *execution advantage* because it is the residual outcome not attributable to mode choice, under the assumption that mode and execution contributions are additive; this fails when their interaction is non-additive (a mode that only helps with a specific execution style), in which case `A_exec` absorbs the interaction term and overstates execution credit. The reported identifiability score is the held-out `L_ident`, which measures *how well the learned decomposition recovers the interventional effect* because it is evaluated on queries whose interventions were not used in training; it fails to reflect true identifiability when the held-out queries are distributionally similar to training queries, in which case it measures generalization within a narrow query family rather than identifiability broadly.

## Contribution

(1) ADII, a learned advantage-decomposition framework for unified LLM routing that replaces COMET-GRPO's hand-designed split with a two-head gating network trained end-to-end on outcome-only rewards and anchored by a synthetic mode-intervention consistency loss. (2) An interventional identifiability score — the held-out consistency between learned mode-advantage differences and forced-mode outcome contrasts — that serves as a training diagnostic and removes the need for externally assigned modes during cold-start. (3) A residual-based execution-advantage estimator that absorbs the non-mode component of outcome reward, providing a concrete alternative to prefix-value Monte-Carlo credit assignment for trajectory-level routing rewards.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | RouteWeaver mixed-domain routing benchmark | Covers modes and execution traces needed for interventions |
| Primary metric | Accuracy on held-out queries | Directly measures routing quality from outcome-only rewards |
| Baseline | RouteWeaver (COMET-GRPO) | Tests whether learned split beats hand-designed advantage decomposition |
| Baseline | VinePPO-style per-step Monte-Carlo advantage | Tests whether whole-trajectory causal contrast beats prefix-value estimation |
| Baseline | Hybrid LLM static router | Tests whether mode-aware decomposition beats mode-agnostic routing |
| Ablation-of-ours | ADII without interventional consistency loss | Tests whether synthetic mode intervention is necessary for identifiability |

### Why this setup validates the claim

The central claim of ADII is that a learned, intervention-identifiable decomposition of mode-selection and execution advantages improves routing accuracy over hand-designed splits, prefix-value advantage estimators, and mode-agnostic routers. The RouteWeaver benchmark provides the mixed-domain queries and full routing trajectories (mode, model, execution trace, outcome) needed to construct synthetic mode interventions, so the evaluation directly tests whether ADII's causal contrast yields a better policy. RouteWeaver (COMET-GRPO) tests the sub-claim that learning the split beats a hand-designed one: if ADII's gain is real, it should appear where the hand-designed split misattributes credit. VinePPO-style per-step Monte-Carlo advantage tests the sub-claim that whole-trajectory interventional contrasts are more informative than prefix-value estimates: if ADII is correct, its advantage should be concentrated on queries where execution is mode-conditioned and prefix values are noisy. The Hybrid LLM static router tests the sub-claim that mode selection matters at all: if ADII's decomposition is meaningful, it should outperform a router that ignores mode. The ablation (ADII without the interventional consistency loss) isolates the identifiability mechanism: without the synthetic intervention, the two heads are unconstrained and should collapse or misattribute credit, so the ablation should show a clear drop precisely on queries where mode and execution are coupled. Accuracy is the right primary metric because the claim is about routing quality under outcome-only rewards; a decomposition that improves identifiability but not accuracy would not validate the idea. This combination forms a falsifiable test: each baseline and ablation targets a specific sub-claim, and the predicted pattern of gains (concentrated on coupled-mode queries, not uniform) distinguishes ADII's mechanism from generic improvement.

### Expected outcome and causal chain

**vs. RouteWeaver (COMET-GRPO)** — On a case where a query succeeds under multiple modes but the hand-designed COMET-GRPO split assigns most advantage to execution because the mode token is rare, the baseline produces a policy that under-explores alternative modes and overfits to the execution trace of the observed mode. Our method instead learns A_mode from the interventional contrast r − r' under a forced alternative mode, so the mode head receives a direct causal signal even when the observed mode is rare. We expect a noticeable accuracy gap on queries with multiple viable modes (where mode choice is consequential) but parity on queries with a single dominant mode, because the intervention provides no differential signal there.

**vs. VinePPO-style per-step Monte-Carlo advantage** — On a case where execution is long and mode-conditioned (e.g., agentic mode triggers many tool calls), the baseline produces noisy per-step advantages because prefix rewards are sparse and the Monte-Carlo estimates conflate mode-driven and execution-driven variance. Our method instead estimates a whole-trajectory causal contrast between two forced-mode rollouts, so the mode advantage is isolated from step-level noise. We expect a noticeable gap on long, mode-conditioned trajectories and parity on short single-round queries, where prefix-value estimation is already low-variance and the mode effect is small.

**vs. Hybrid LLM static router** — On a case where the same query can be answered correctly by a small model in single-round mode but requires a large model in agentic mode, the baseline produces a static model choice that ignores mode and either overspends on the large model or loses accuracy on hard queries. Our method instead learns a mode-selection advantage that captures the interaction between mode and model choice, so it routes to the cheaper successful mode when the interventional contrast is positive. We expect a clear accuracy–cost advantage on queries with heterogeneous mode–model tradeoffs, but parity on queries where a single model dominates regardless of mode.

### What would falsify this idea

If ADII's accuracy gain over RouteWeaver is uniform across all query subsets rather than concentrated on queries where mode and execution are coupled (multiple viable modes, mode-conditioned execution), then the interventional consistency loss is not doing the work claimed and the decomposition is not identifiable in the intended sense. Similarly, if the ablation without the interventional loss matches ADII's accuracy, the central mechanism is falsified.

## References

1. RouteWeaver: Weaving Mode Selection and Execution into Unified LLM Routing
2. Hybrid LLM: Cost-Efficient and Quality-Aware Query Routing
3. ReTool: Reinforcement Learning for Strategic Tool Use in LLMs
