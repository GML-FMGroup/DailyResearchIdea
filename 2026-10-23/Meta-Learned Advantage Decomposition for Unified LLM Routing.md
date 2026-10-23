# Meta-Learned Advantage Decomposition for Unified LLM Routing

## Motivation

Unified routing policies such as RouteWeaver decompose trajectory-level advantages into a mode-selection term and a within-mode execution term using a hand-designed rule, so the decomposition is never checked against the actual marginal contribution each decision makes to the final outcome. This leaves the optimality of the decomposition — whether the assigned credit equals the true counterfactual effect of a decision — as an unaddressed residual gap: when mode choice and execution interact (e.g., a multi-round trajectory that only pays off because a single-round attempt failed), the fixed rule misattributes credit and the policy gradient is biased. The root cause is structural: the decomposition is specified a priori rather than estimated from interventional evidence about each decision's effect.

## Key Insight

The true advantage decomposition is identified by the counterfactual marginal contribution of each decision, and this decomposition can be recovered as the fixed point of a bilevel objective in which an inner counterfactual-rollout estimator supplies the target and the outer decomposition parameters are fit to it, making the hand-designed credit rule a special case rather than an assumption.

## Method

**What it is.** We introduce **MAD** (Meta-learned Advantage Decomposition), a bilevel procedure that learns the parameters $\phi$ of RouteWeaver's advantage decomposition so that the decomposed per-decision advantages match counterfactually estimated marginal contributions. Inputs: routing trajectories $\tau$ from the unified routing policy $\pi_\theta$ (mode $m_t$ and model choice $a_t$ per step), a counterfactual rollout budget $K$, and a decomposition parameterization $g_\phi$ that maps (trajectory return, mode indicator, execution features) to a mode-advantage $\hat A^{\text{mode}}_t$, an execution-advantage $\hat A^{\text{exec}}_t$, and an explicit interaction term $\hat A^{\text{int}}_t$, with $\hat A^{\text{mode}}_t + \hat A^{\text{exec}}_t + \hat A^{\text{int}}_t = R(\tau) - V(s_t)$. Outputs: optimized $\phi^\star$, then a GRPO update of $\theta$ using $g_{\phi^\star}$.

**Load-bearing assumption (stated explicitly).** The method assumes that the mode decision and the execution decision admit an additive decomposition in which the mode's counterfactual marginal contribution can be isolated by holding the execution surface fixed, so that $MC^{\text{mode}}_t + MC^{\text{exec}}_t$ exactly partitions the total marginal contribution with no interaction term. This assumption fails when mode and execution are coupled through a shared hidden state (e.g., a multi-round mode that changes the model pool available later); in that case the residual $MC^{\text{exec}}_t$ absorbs the interaction and overstates execution credit. To make the assumption testable rather than hidden, we add an explicit interaction term $\hat A^{\text{int}}_t$ and estimate it from a factorial intervention, as detailed below.

**How it works.**

```pseudocode
# Outer loop: learn decomposition phi ; Inner: policy update on frozen phi
initialize phi (mode/exec/int mixing coefficients), V_psi, pi_theta
for iteration = 1..N:
    # --- inner: collect trajectories under current decomposition ---
    for b in 1..B:                      # B = 64 trajectories
        tau_b ~ pi_theta (route: pick mode m_t, then model a_t)
        store (s_t, m_t, a_t, R(tau_b))

    # --- counterfactual marginal contribution estimation (K=8 rollouts/decision) ---
    for each decision t in sampled subset S (|S| = 16 per trajectory):
        # intervene on the decision, hold the rest of prefix fixed
        for k in 1..K:
            tau'_{t,k} ~ pi_theta(. | s_t, do(decision = alternative))
        MC_t = mean_k R(tau'_{t,k}) - R(tau_b)     # marginal contribution of decision t
        # factorial intervention: do(mode), do(exec), do(mode,exec)
        MC^{mode}_t = mean_k R(tau' with do(mode=alt), exec held) - R(tau_b)
        MC^{exec}_t = mean_k R(tau' with do(exec=alt), mode held) - R(tau_b)
        MC^{joint}_t = mean_k R(tau' with do(mode=alt, exec=alt)) - R(tau_b)
        MC^{int}_t = MC^{joint}_t - MC^{mode}_t - MC^{exec}_t   # interaction term

    # --- outer: fit phi to match counterfactual targets ---
    L(phi) = sum_{t in S} [ (A^mode_phi(s_t,m_t) - MC^mode_t)^2
                          + (A^exec_phi(s_t,a_t) - MC^exec_t)^2
                          + (A^int_phi(s_t,m_t,a_t) - MC^int_t)^2 ]
    phi <- phi - eta * grad_phi L(phi)          # eta = 1e-3, Adam

    # --- policy update using learned decomposition ---
    for each t: A_t = g_phi(s_t, m_t, a_t)       # now a learned, counterfactually-anchored advantage
    theta <- GRPO_update(theta, A_t)
```

**Why this design.** We chose a bilevel formulation over a single joint loss because the decomposition target (counterfactual marginal contribution) must be estimated under the *current* policy and then held fixed while $\phi$ is fit — jointly optimizing $\phi$ and $\theta$ would let the decomposition drift toward whatever makes the policy loss small rather than toward the true marginal contribution, accepting the cost of two timescales and slower convergence. We chose K=8 counterfactual rollouts per sampled decision over full per-step enumeration because full enumeration is combinatorially expensive in multi-round/agentic modes, accepting higher variance in $MC_t$ that we control by sampling only 16 decisions per trajectory. We parameterized $g_\phi$ as a small linear mixing over RouteWeaver's mode/exec features plus an explicit interaction term $\hat A^{\text{int}}_t = \phi_{\text{int}}^\top [m_t \odot a_t]$ (elementwise product of mode and execution feature vectors) rather than a large free-form network because the linear form keeps the sum-to-return constraint $\hat A^{\text{mode}}+\hat A^{\text{exec}}+\hat A^{\text{int}}=R-V$ exactly satisfiable, accepting reduced expressivity. Unlike VinePPO (per-step Monte-Carlo value rollouts for PPO), which uses rollouts to replace the value function, MAD uses interventional rollouts as *supervision for the decomposition parameters themselves* while leaving the policy-gradient structure intact; a domain expert would not describe MAD as a VinePPO variant because VinePPO never learns a credit-assignment rule — it estimates values — whereas MAD's learned object is the split between mode, execution, and interaction credit.

**Why it measures what we claim.** The quantity $MC^{\text{mode}}_t$ measures the *causal marginal contribution of the mode decision* because the do-intervention holds the execution surface fixed and averages over downstream stochasticity, which equals the mode's counterfactual effect under the assumption that the only path from mode to return is through the execution it induces; this assumption fails when mode and execution are coupled through a shared hidden state (e.g., a multi-round mode that changes the model pool available later), in which case $MC^{\text{mode}}_t$ reflects the mode-plus-induced-execution effect rather than the isolated mode effect. The quantity $MC^{\text{exec}}_t$ measures execution credit because it is the residual of the total marginal contribution after removing the mode component, under the assumption that the two intervention surfaces partition all decision-relevant variation; this fails when an interaction term exists that belongs to neither surface, in which case the residual absorbs the interaction and overstates execution credit. The quantity $MC^{\text{int}}_t = MC^{\text{joint}}_t - MC^{\text{mode}}_t - MC^{\text{exec}}_t$ measures the *interaction effect* between mode and execution because it is the difference between the joint intervention and the sum of the two single interventions; this equals zero exactly when the additive decomposition holds, and its magnitude directly quantifies the violation of the load-bearing assumption. The outer loss $L(\phi)$ measures *decomposition optimality* because minimizing squared error to the counterfactual targets makes $g_\phi$ a projection of the true marginal contributions onto the constrained linear family, under the assumption that the linear family is rich enough to represent the true split; this fails when the true split is nonlinear in the features, in which case $\phi^\star$ yields the best linear approximation rather than the exact decomposition.

## Contribution

(1) MAD, a bilevel meta-learning procedure that replaces RouteWeaver's hand-designed advantage decomposition with parameters fit to counterfactually estimated marginal contributions, keeping the sum-to-return constraint exact. (2) A design principle: credit-assignment rules for unified routing should be treated as estimable objects anchored to interventional targets, not as fixed priors, and we show how to operationalize this without abandoning GRPO-style policy optimization. (3) A counterfactual-rollout estimator that separates mode credit from execution credit by partitioning the intervention surface, with a documented failure mode when the two are coupled.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | Mixed benchmark suite spanning math, coding, knowledge, multi-round agentic tasks | Forces both mode and execution decisions to matter |
| Primary metric | Accuracy at matched cost budget | Isolates credit quality from budget drift |
| Baseline | RouteWeaver (unified routing, single-loss) | Tests whether learned decomposition beats shared advantage |
| Baseline | VinePPO-style MC value rollouts | Tests whether decomposition beats value estimation |
| Baseline | Always-agentic / always-single-round | Tests whether routing itself is necessary |
| Ablation-of-ours | MAD with fixed uniform mode/exec split | Tests whether meta-learned decomposition is doing the work |
| Ablation-of-ours | MAD with interaction term forced to zero (two-term decomposition) | Tests whether the explicit interaction term is necessary to avoid misattribution |
| Diagnostic | Estimated interaction magnitude $|MC^{\text{int}}_t|$ per task subset | Quantifies violation of the additive assumption; reported alongside main results |

### Why this setup validates the claim

The central claim is that counterfactually-anchored decomposition of advantage into mode and execution credit produces better routing policies than treating the advantage as a single undifferentiated signal. RouteWeaver is the pivotal baseline because it uses the same unified policy and the same routing surface but assigns credit through a single joint loss; if MAD's bilevel decomposition is doing real work, the gap must appear here, not merely against always-agentic baselines that fail for unrelated reasons. The VinePPO-style baseline isolates the credit-assignment mechanism: it uses rollouts to estimate values rather than to supervise a decomposition, so any gain of MAD over it is attributable specifically to learning the split rather than to using rollouts at all. The uniform-split ablation is the sharpest test: it keeps every component of MAD except the meta-learned phi, so if the gain survives with a fixed split, the decomposition is not the cause. The two-term ablation (interaction forced to zero) directly tests the load-bearing assumption: if the interaction term is non-negligible, the two-term version will misattribute credit and underperform full MAD, and the diagnostic $|MC^{\text{int}}_t|$ will show where. Accuracy at matched cost is the right metric because a decomposition that improves credit assignment should raise quality per unit of execution spend; a raw accuracy metric would let MAD win simply by routing to expensive modes.

### Expected outcome and causal chain

**vs. RouteWeaver (unified routing, single-loss)** — On a case where a multi-round mode is chosen but a single execution step inside it carries most of the return, RouteWeaver's single advantage signal spreads credit uniformly across mode and execution decisions, so the policy reinforces the expensive mode even when the execution step was the productive one. MAD instead estimates the mode's counterfactual marginal contribution with the execution surface held fixed, learns that the mode contributed little, and routes the mode advantage toward zero while assigning credit to execution. We expect a noticeable accuracy-per-cost gap concentrated on mixed-mode tasks where mode and execution contributions diverge, with near-parity on tasks where one decision dominates.

**vs. VinePPO-style MC value rollouts** — On a case where two decisions in the same trajectory have similar value estimates but very different causal leverage on the return, the VinePPO-style baseline assigns advantages from per-step value rollouts, which measure expected return from a state rather than the marginal effect of the decision taken. It therefore reinforces a low-leverage decision and under-weights the high-leverage one. MAD's interventional rollouts directly target the marginal contribution of each decision, so the advantage ordering matches causal leverage. We expect MAD to show a modest but consistent edge on trajectories with many near-equal-value decisions, and parity on short single-decision trajectories where value and marginal contribution coincide.

**vs. Always-agentic / always-single-round** — On a case where a factual-recall query is cheap to answer in single-round but the always-agentic policy invokes multi-round execution, the baseline burns execution budget on a task whose marginal return from extra rounds is near zero, because it has no mode decision to make. MAD's mode advantage, supervised against the counterfactual return of switching modes, learns to push probability toward single-round on such queries while preserving agentic routing where the mode's marginal contribution is positive. We expect a clear cost reduction on recall-style subsets with flat accuracy, and preserved accuracy on math and coding subsets where agentic execution genuinely helps.

**vs. MAD with fixed uniform mode/exec split** — On a case where execution dominates the return, the uniform-split ablation assigns half the advantage to the mode decision, so the policy receives a systematically biased gradient that inflates mode probability independent of the mode's actual contribution. MAD's learned phi, fit to counterfactual targets, shrinks the mode advantage on these instances. We expect the ablation to drift toward over-selecting complex modes over training, producing a widening accuracy-per-cost gap relative to full MAD that grows with the fraction of execution-dominated trajectories in the batch.

**vs. MAD with interaction term forced to zero (two-term decomposition)** — On a task subset with known mode-exec coupling (e.g., multi-round agentic tasks where the model pool changes after the first round), the two-term decomposition absorbs the interaction into $MC^{\text{exec}}_t$, overstating execution credit and understating mode credit. Full MAD estimates $MC^{\text{int}}_t$ from the factorial intervention and assigns it explicitly, so the per-decision advantages match the true marginal contributions. We expect full MAD to outperform the two-term ablation on coupled subsets, with the gap scaling with the diagnostic $|MC^{\text{int}}_t|$; on uncoupled subsets (where $|MC^{\text{int}}_t| \approx 0$) the two should match.

### What would falsify this idea

If MAD's gain over RouteWeaver is uniform across all subsets rather than concentrated on tasks where mode and execution contributions diverge, the decomposition is not the causal driver and the claim fails. Likewise, if the uniform-split ablation matches full MAD, the meta-learned decomposition adds nothing and the central mechanism is refuted. Additionally, if the estimated interaction magnitude $|MC^{\text{int}}_t|$ is negligible on all subsets and the two-term ablation matches full MAD, then the load-bearing additive assumption holds and the interaction term is unnecessary; if $|MC^{\text{int}}_t|$ is large but full MAD does not outperform the two-term ablation, the interaction estimation is not effective.

## References

1. RouteWeaver: Weaving Mode Selection and Execution into Unified LLM Routing
2. Hybrid LLM: Cost-Efficient and Quality-Aware Query Routing
3. ReTool: Reinforcement Learning for Strategic Tool Use in LLMs
