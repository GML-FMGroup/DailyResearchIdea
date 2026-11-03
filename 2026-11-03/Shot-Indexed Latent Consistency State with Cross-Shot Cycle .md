# Shot-Indexed Latent Consistency State with Cross-Shot Cycle Consistency for Plan-Conditioned Video Generation

## Motivation

Planner-to-generator interfaces in multi-agent movie generation are strictly one-way: MovieAgent's hierarchical CoT planner emits a shot plan and the video model renders each shot independently, while DynaFlow's orchestration traces feed back only into subsequent planner actions, never into a persistent cross-shot state that constrains rendering. The structural root cause is that neither system carries a learned state variable that survives across shots and is simultaneously identifiable from the symbolic plan and the rendered pixels, so cross-shot consistency is enforced only by re-prompting rather than by a sufficient statistic that the planner can correct after observing evidence.

## Key Insight

A shot-indexed latent is a sufficient statistic for cross-shot consistency exactly when it is the unique fixed point of a cycle-consistency constraint requiring it to be both predictable from the symbolic plan and recoverable from the rendered pixels, which makes planner feedback a well-posed correction of the state rather than an arbitrary embedding.

## Method

**(A) What it is.** SILCS (Shot-Indexed Latent Consistency State) is a per-shot latent state $s_t \in \mathbb{R}^{d}$ with $d=512$ maintained across a multi-shot video. Inputs: the planner's symbolic shot plan $p_t$ (scene, camera, character IDs, action) and the previous state $s_{t-1}$. Outputs: a rendered shot $x_t = G(s_t, p_t)$ and an updated state $s_t$ that is re-estimated from the rendered pixels and re-projected onto the plan-predictable manifold before the next shot is generated.

**(B) How it works.**

```pseudocode
# Encoders / generators (all trained jointly)
E_plan(p_t) -> h_t            # symbolic plan encoder (LLM hidden state, 4096-d)
E_pix(x_t)  -> z_t            # video-latent encoder (per-shot pooled latent, 512-d)
P(h_t, s_{t-1}) -> s_t^plan   # plan-predictor: 2-layer MLP, hidden=1024, GeLU
R(s_t)      -> x_t            # renderer conditioned on state + plan
Q(z_t, h_t) -> s_t^pix        # pixel-recovery: 2-layer MLP, hidden=1024, GeLU

# Anti-collapse regularizers (VICReg-style) applied to s_t^plan and s_t^pix
L_var  = (1/d) * sum_j max(0, 1 - Var(s_t^plan_j)) + (1/d) * sum_j max(0, 1 - Var(s_t^pix_j))
L_cov  = (1/d) * sum_{i != j} [Cov(s_t^plan)_ij^2 + Cov(s_t^pix)_ij^2]

for t in 1..T:
    s_t^plan = P(E_plan(p_t), s_{t-1})          # forward prediction from plan
    x_t      = R(s_t^plan, p_t)                  # render shot t
    z_t      = E_pix(x_t)
    s_t^pix  = Q(z_t, E_plan(p_t))               # recover state from pixels
    # cross-shot cycle-consistency loss (per shot):
    L_cyc    = || s_t^plan - s_t^pix ||^2
    # plan-predictability loss: s_t^pix must be reachable from plan alone
    L_plan   = || P(E_plan(p_t), s_{t-1}) - s_t^pix ||^2
    # pixel-recoverability loss: s_t^plan must be recoverable from pixels
    L_pix    = || Q(E_pix(R(s_t^plan, p_t)), E_plan(p_t)) - s_t^plan ||^2
    # update state for next shot: re-estimate from evidence, then project onto plan manifold
    s_t      = Project(s_t^pix, manifold={s : P(E_plan(p_t), s_{t-1}) = s})
    # planner feedback: if normalized divergence > tau=0.15, planner revises p_{t+1}
    # normalized divergence: ||s_t^pix - s_t^plan|| / sqrt( (1/t) * sum_{k=1}^t ||s_k^pix - s_k^plan||^2 + eps )
    if L_cyc_normalized > tau: p_{t+1} = Planner.revise(p_{t+1}, s_t, s_t^plan)

L_total = L_recon + lambda_cyc * L_cyc + lambda_plan * L_plan + lambda_pix * L_pix + lambda_var * L_var + lambda_cov * L_cov
# lambda_cyc=1.0, lambda_plan=0.5, lambda_pix=0.5, lambda_var=1.0, lambda_cov=0.1, d=512, trained 200k steps, AdamW lr=1e-4
```

**Load-bearing assumption (explicit).** The plan-predictor $P$ and pixel-recovery $Q$ share a unique, non-degenerate fixed point, so that minimizing $\|s_t^{plan} - s_t^{pix}\|$ forces the latent to encode only plan-predictable and pixel-recoverable information rather than collapsing to a constant or trivially copying the plan. This assumption is enforced by the VICReg-style variance and covariance terms $L_{var}$ and $L_{cov}$ (Bardes et al., 2022) which penalize low variance and off-diagonal covariance in both $s_t^{plan}$ and $s_t^{pix}$, preventing constant-code collapse. The projection operator $Project(\cdot)$ is defined as a single gradient step: $s_t = s_t^{pix} - \eta \nabla_s \|P(E_{plan}(p_t), s_{t-1}) - s\|^2 |_{s=s_t^{pix}}$ with $\eta=0.1$, which is well-defined and converges locally under the assumption that the plan manifold is locally convex.

**(C) Why this design.** We chose a *cycle-consistency* constraint over a simple L2 penalty between adjacent shots because a plain smoothness prior (as in LoRA-BE's intra-shot temporal consistency) constrains pixels but does not force the state to be identifiable from both modalities, accepting the cost that cycle consistency is harder to optimize and can have degenerate fixed points that we break by projecting onto the plan manifold and adding VICReg-style variance/covariance regularizers. We chose a *learned state* over a retrieval scheme that matches shots to a bank of prior shots because retrieval cannot represent a state that the planner revises after observing evidence, accepting the cost that the state must be trained rather than looked up. We chose *planner feedback on state divergence* over a controller that routes between planner and renderer because the normalized divergence $\|s_t^{pix}-s_t^{plan}\| / \sigma_t$ (where $\sigma_t$ is the running RMS of past divergences) is a closed-form, interpretable signal that the planner can act on, accepting the cost that the planner must be able to consume a continuous state vector. A domain expert would not describe this as a variant of MovieAgent or DynaFlow because those systems have no state that survives across shots and is corrected by rendered evidence; SILCS's load-bearing object is the cycle-consistent latent, not the plan or the orchestration trace.

**(D) Why it measures what we claim.** The quantity $\|s_t^{pix} - s_t^{plan}\|$ measures *cross-shot consistency* because the assumption that the plan-predictor and pixel-recovery networks share the same fixed point makes the two estimates equal iff the state encodes only plan-predictable and pixel-recoverable information; this assumption fails when the renderer hallucinates content not in the plan (e.g., a character's clothing changes), in which case the divergence reflects renderer drift rather than planner inconsistency. The quantity $L_{plan}$ measures *plan-predictability* because the assumption that the plan encoder $E_{plan}$ is injective on shot plans makes minimizing $L_{plan}$ equivalent to forcing $s_t^{pix}$ into the plan's image; this assumption fails when two distinct plans map to the same hidden state (e.g., synonymous camera descriptions), in which case $L_{plan}$ reflects encoder collisions rather than genuine plan alignment. The quantity $L_{pix}$ measures *pixel-recoverability* because the assumption that $Q \circ E_{pix} \circ R$ is a contraction around $s_t^{plan}$ makes minimizing $L_{pix}$ equivalent to making the rendered pixels a sufficient statistic for the state; this assumption fails when the renderer is stochastic, in which case $L_{pix}$ reflects renderer variance rather than information loss. The projection step measures *planner-feedback correctness* because the assumption that the plan manifold is locally convex makes projection the unique nearest plan-consistent state; this assumption fails at shot boundaries with discontinuous camera cuts, in which case the projection reflects the nearest smooth approximation rather than the true plan-consistent state.

## Contribution

(1) SILCS, a shot-indexed latent consistency state trained with a cross-shot cycle-consistency objective that couples the planner's symbolic plan and the renderer's pixel latent into a single sufficient statistic for cross-shot consistency. (2) A planner-feedback loop in which the divergence between plan-predicted and pixel-recovered states is a closed-form correction signal that revises subsequent shot plans, replacing one-way conditioning with a bidirectional interface. (3) A projection operator onto the plan-predictable manifold that breaks degenerate cycle-consistency fixed points, with an empirical characterization of when cycle consistency reduces to renderer-drift detection versus genuine planner inconsistency.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Multi-shot narrative video benchmark with per-shot symbolic plans (e.g., MovieNet-style with 2-4 shot clips for proof-of-concept) | Only setting where plan and rendered pixels both exist per shot |
| Primary metric | Cross-shot character/scene consistency score (human + judge) | Directly targets the state that must survive across shots |
| Baseline | DynaFlow (Qwen3.5-27B, Forced mode) | Strongest orchestration baseline with no persistent latent state |
| Baseline | MovieAgent (CoT multi-agent planning) | Tests whether planning alone, without evidence-corrected state, suffices |
| Baseline | Retrieval-based shot matching | Tests whether a learned state beats lookup of prior shots |
| Baseline | CycleGAN-style plan<->pixel translation with a single shared latent | Tests whether shot-indexed recurrence and planner-revision loop are necessary |
| Ablation-of-ours | SILCS without cycle-consistency (L2 smoothness only) | Isolates whether cycle consistency, not just smoothing, carries the gain |
| Ablation-of-ours | SILCS without VICReg variance/covariance regularizers | Tests whether anti-collapse terms are necessary for non-degenerate state |
| Control | Renderer-drift measurement (independent of state) | Disambiguates divergence due to renderer hallucination vs planner inconsistency |

### Why this setup validates the claim

The central claim is that a cycle-consistent latent state, corrected by rendered evidence and projected onto the plan manifold, is what makes multi-shot output coherent — not the plan and not the orchestration trace. The dataset is the only one where each shot carries both a symbolic plan and rendered pixels, so the divergence signal is measurable per shot. DynaFlow tests the sub-claim that orchestration without a persistent state cannot carry information across shots; MovieAgent tests whether richer planning substitutes for an evidence-corrected state; retrieval tests whether the state must be learned rather than looked up. The CycleGAN-style shared-latent baseline tests whether the shot-indexed recurrence and planner-revision loop are necessary beyond a single shared latent. The ablation without cycle-consistency is the load-bearing control: if plain L2 smoothness between adjacent shots matches SILCS, then cycle consistency is doing no work and the mechanism story collapses. The ablation without VICReg regularizers tests whether the anti-collapse terms are necessary to avoid degenerate constant-code solutions. The renderer-drift control measures the divergence signal when the state is held fixed, providing a baseline for how much divergence is due to renderer hallucination alone. Consistency score is the right metric because it is exactly the property the state is supposed to guarantee; a metric like per-shot fidelity would be blind to cross-shot drift and could not distinguish the mechanisms.

### Expected outcome and causal chain

**vs. DynaFlow** — On a case where a character's clothing or a room layout must persist across a camera cut, DynaFlow produces drift because its orchestration trace carries no state that survives the shot boundary; each shot is planned fresh from the script. Our method instead re-estimates $s_t^{pix}$ from the rendered pixels and projects it onto the plan manifold, so the next shot's renderer is conditioned on evidence of what was actually drawn. We expect a noticeable consistency gap concentrated on multi-cut scenes with recurring characters, with near-parity on single-shot or single-character scenes.

**vs. MovieAgent** — On a case where the planner's symbolic description is under-specified (e.g., "the same street, later"), MovieAgent produces a plausible but different street because its CoT plan has no channel to receive what the renderer actually produced. Our method instead flags normalized divergence above $\tau=0.15$ and revises $p_{t+1}$ using the recovered state, closing the loop between plan and pixels. We expect MovieAgent to score well on script faithfulness but lag on cross-shot consistency, while SILCS narrows the consistency gap without losing faithfulness.

**vs. Retrieval-based shot matching** — On a case where the planner revises the plan mid-sequence after seeing evidence, retrieval produces a shot matched to a stale prior shot because its bank cannot represent a state that changes with new evidence. Our method instead maintains a continuous $s_t$ that the planner can revise, so the retrieved-style match is replaced by a state consistent with the revised plan. We expect SILCS to outperform retrieval specifically on sequences containing a planner revision, with parity on sequences with no revision.

**vs. CycleGAN-style shared-latent baseline** — On a case where the plan is revised after observing evidence, the shared-latent baseline cannot update its single latent per shot because it lacks a shot-indexed recurrence and planner-revision loop; it produces a shot consistent with the original plan but not with the revised one. Our method instead maintains a per-shot state $s_t$ that is re-estimated and projected, allowing the planner to correct the state. We expect SILCS to outperform the shared-latent baseline on sequences with planner revisions, with parity on sequences without revisions.

**vs. SILCS without VICReg regularizers** — On a case where the plan and pixels are highly correlated, the ablated model collapses to a constant state because the cycle-consistency loss alone admits degenerate solutions; the consistency score drops to near-random. Our method with VICReg maintains non-degenerate variance and covariance, preserving cross-shot consistency. We expect the ablation to show significantly lower consistency, confirming the necessity of anti-collapse terms.

**Renderer-drift control** — By measuring divergence when the state is fixed (no re-estimation), we isolate the component of $\|s_t^{pix} - s_t^{plan}\|$ due to renderer hallucination. We expect this control to show lower divergence than SILCS's adaptive state, confirming that SILCS's divergence signal primarily reflects planner inconsistency rather than renderer drift.

### What would falsify this idea

If SILCS's gain over DynaFlow, MovieAgent, and the CycleGAN-style baseline is uniform across all subsets rather than concentrated on multi-cut, recurring-character, and planner-revision cases, then the cross-shot state is not the causal mechanism and the claim is wrong. Likewise, if the no-cycle-consistency ablation matches SILCS on those same subsets, or if the no-VICReg ablation does not collapse, then cycle consistency and anti-collapse terms are doing no work and the design rationale fails.

## References

1. Automated Movie Generation via Multi-Agent CoT Planning
2. StoryAgent: Customized Storytelling Video Generation via Multi-Agent Collaboration
3. DynaFlow: Learning Free-Form Multi-Agent Orchestration
4. Why Do Multi-Agent LLM Systems Fail?
5. Challenges in Human-Agent Communication
6. AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation
7. The Rise and Potential of Large Language Model Based Agents: A Survey
