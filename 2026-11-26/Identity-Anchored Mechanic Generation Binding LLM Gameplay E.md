# Identity-Anchored Mechanic Generation: Binding LLM Gameplay Edits to a Persistent Visual Identity Subspace

## Motivation

Iterative LLM game-generation systems treat narrative/mechanic edits and visual asset production as two loosely coupled pipelines, so consistency under refinement is never established across modalities. Unbounded enforces character/environment visual consistency with a dynamic regional IP-Adapter but never conditions its LLM-generated mechanics on that visual identity, while Play2Evolve regenerates static and dynamic assets from video generation and code without any guarantee that gameplay-driven asset changes preserve the character's visual identity. Structurally this fails because the identity signal lives only inside the vision adapter's conditioning path and is invisible to the text-side edit loop, so no edit can be checked against it before assets are regenerated.

## Key Insight

A visual identity is representable as a low-dimensional subspace of the vision model's image-prompt embedding space, and any LLM-proposed mechanic can be projected into that same subspace via its rendered asset delta, so identity preservation becomes a measurable geometric residual that gates edits rather than a post-hoc perceptual judgment.

## Method

**What it is.** IAMG (Identity-Anchored Mechanic Generation) takes a running game state, a persistent identity embedding `e_id` for each character/environment, and an LLM-proposed edit (new mechanic or narrative beat) and returns either an accepted edit plus regenerated assets or a rejected edit with the identity residual that caused rejection. Identity is operationalized as a subspace `S = span(U_k)` of the regional IP-Adapter embedding space, estimated once per entity from its reference renders.

**How it works.**

```pseudocode
# Phase 0: identity subspace estimation (once per entity)
refs = render_reference_views(entity, n=8)          # existing IP-Adapter path
E    = ip_adapter_embed(refs)                        # [8, d]
U, s, _ = svd(E - mean(E))
U_k  = U[:, :k]            # k = 16, chosen by 95% variance
S    = {mean(E) + U_k z}   # affine identity subspace

# Phase 0b: supervised identity-direction calibration (new, but only refines S)
# Collect paired renders of the same entity under varied pose/background (n=32 per entity)
# and of different entities (n=32 per entity).
# Learn a linear projection W (d x k) that regresses out pose/context:
#   minimize || W^T (e_pose - e_ref) ||^2 + lambda * || W^T (e_other - e_ref) ||^2
#   with lambda = 0.1, solved in closed form via generalized eigenvalue problem.
# Replace U_k with W (columns orthonormalized) to form S_sup = {mean(E) + W z}.
# Calibrate tau_id on held-out legitimate new-view renders (n=16 per entity):
#   tau_id = 1.5 * max_{a in held-out} r_id(a)  (ensures new views pass)

# Phase 1: edit proposal (LLM, per iteration)
edit = LLM.propose(state, feedback_history)          # mechanic/narrative delta
assets_new = render(entity, edit)                    # IP-Adapter conditioned

# Phase 2: identity-preservation gate
for each rendered asset a in assets_new:
    e_a  = ip_adapter_embed(a)
    r_id = || (e_a - mean(E)) - W W^T (e_a - mean(E)) ||   # residual to S_sup
    if r_id > tau_id:  reject
# Phase 3: gameplay-consistency gate (only if Phase 2 passes)
delta_play = simulate(state, edit, horizon=20)       # deterministic rollout
if KL(p_after || p_before) > tau_play (=0.25):  reject
accept(edit, assets_new); state = delta_play.end
```

**Why this design.** We chose a fixed SVD-derived linear subspace over a learned identity classifier because the IP-Adapter embedding geometry is already the conditioning surface the vision model consumes, so a linear residual is directly interpretable and needs no extra training data, accepting the cost that strongly non-linear identity drift (e.g., style transfer) may project inside `S` and pass. We chose to project the *rendered asset delta* rather than the text edit embedding because text-to-visual mapping is many-to-one and a text-space check would be a weak proxy, accepting the cost of one extra render per candidate edit. We chose a deterministic 20-step rollout for the gameplay gate over a learned playfulness predictor (as in Play2Evolve's feedback loop) because a rollout gives a reproducible state-distribution shift, accepting that it under-covers long-horizon emergent mechanics. Unlike Unbounded, which applies the regional IP-Adapter as a generation conditioner with no acceptance test, and unlike Play2Evolve, whose playfulness feedback is a diagnostic signal that never touches the visual conditioning path, IAMG makes the identity subspace a *shared variable* that both the text edit loop and the vision regeneration must satisfy; a domain expert would not describe this as a variant of either because neither prior system has a rejection gate coupling the two modalities. The supervised calibration in Phase 0b is added solely to make the identity subspace robust to pose/background variation, as the original SVD over 8 views cannot separate identity from nuisance directions; this does not change the two-gate architecture.

**Why it measures what we claim.** The residual `r_id` measures *visual identity preservation* because the assumption is that the reference renders' IP-Adapter embeddings concentrate on a low-dimensional affine subspace whose orthogonal complement captures identity-irrelevant variation (pose, background); this assumption fails when a legitimate mechanic requires a new view angle outside the reference distribution, in which case `r_id` reflects viewpoint novelty rather than identity drift. The state-distribution KL measures *gameplay consistency* because the assumption is that a mechanic preserving the core loop induces a bounded shift in the 20-step reachable-state distribution; this assumption fails for mechanics whose effect is delayed beyond the horizon, in which case the KL reflects short-horizon stability rather than design coherence. The conjunction of the two gates measures *cross-modal consistency of the edit* because the assumption is that an edit is coherent iff it is simultaneously expressible in the identity subspace and bounded in state space; this assumption fails when an edit is visually and dynamically benign but narratively contradictory, in which case the gates reflect surface compatibility rather than semantic coherence.

## Contribution

(1) A cross-modal identity subspace and rejection gate that makes visual identity a first-class constraint on LLM-proposed gameplay/narrative edits, with a concrete residual-to-subspace test derived from the regional IP-Adapter embedding geometry. (2) The design principle that identity preservation is best operationalized as a geometric residual in the vision model's own conditioning space rather than a separate learned discriminator, together with the paired gameplay-shift gate that prevents the identity gate from being satisfied by trivially small edits. (3) An iterative-evolution protocol and diagnostic (per-iteration residual trajectories) for studying how identity drifts across successive LLM edits.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | GameCraft-Bench with reference-render character sets | Benchmark already tests generated gameplay; adds identity entities |
| Primary metric | Identity-preservation-accepted edits per playtest round | Directly measures the gated cross-modal claim |
| Baseline | Play2Evolve (playfulness feedback loop) | Tests whether identity gate adds over diagnostic-only feedback |
| Baseline | Unbounded (regional IP-Adapter, no gate) | Tests whether acceptance gating beats unconditional conditioning |
| Baseline | Game Initialization (single-shot) | Tests whether iterative gated edits beat no iteration |
| Ablation-of-ours | IAMG with both gates removed (accept-all) | Isolates whether gains come from the gates themselves |
| Ablation-of-ours | IAMG with direct embedding-distance gate (cosine threshold on IP-Adapter embeddings) | Tests whether subspace residual adds over simple distance |
| Ablation-of-ours | IAMG with unsupervised SVD subspace (no Phase 0b calibration) | Tests whether supervised calibration improves identity-sensitive separation |

### Why this setup validates the claim

The central claim is that coupling a visual identity subspace to a gameplay-state gate yields edits that are simultaneously identity-preserving and dynamically coherent, and that this coupling is what neither prior system has. GameCraft-Bench is the right substrate because it already scores generated games and contains the Play2Evolve/Game-Initialization baselines, so the comparison isolates the gate rather than the generator. The primary metric — accepted, identity-preserving edits surviving a playtest round — is the only one that directly detects the predicted effect: an edit that passes both gates should both look like the character and keep the loop stable, so accepted-edit yield rises only if the gates are doing real work. Unbounded tests the visual sub-claim (conditioning without acceptance), Play2Evolve tests the feedback sub-claim (diagnosis without touching conditioning), and the accept-all ablation tests whether the gates, not the LLM proposal loop, produce the gain. If the gates matter, the ablation must fall to roughly baseline levels. The direct embedding-distance ablation tests whether the subspace formulation is necessary: if a simple cosine threshold on IP-Adapter embeddings matches IAMG's identity-sensitive/identity-neutral separation, the subspace residual adds no value. The unsupervised SVD ablation tests whether the supervised calibration (Phase 0b) is needed to avoid rejecting legitimate new-pose mechanics.

### Expected outcome and causal chain

**vs. Unbounded** — On a case where the LLM proposes a mechanic requiring a new pose or prop for the character, Unbounded regenerates assets through the regional IP-Adapter with no acceptance test, so identity drift passes silently into the game and later frames compound it. Our method instead computes the residual of each new render to the entity's pre-estimated affine identity subspace and rejects edits whose residual exceeds the reference-derived threshold, so drift is caught before it enters state. We expect a noticeable gap on the identity-sensitive subset (new-pose mechanics) but rough parity on mechanics that reuse reference views.

**vs. Play2Evolve** — On a case where a proposed mechanic is visually fine but destabilizes the core loop (e.g., an economy tweak that collapses progression within a few steps), Play2Evolve's playfulness feedback diagnoses the problem only after the game has evolved, and its signal never touches the visual conditioning path, so the fix is a text-level patch that can silently break character identity. Our method instead runs a deterministic 20-step rollout and rejects the edit when the reachable-state KL exceeds threshold, before any asset is committed, keeping the visual and dynamic edits coupled. We expect fewer regressions per evolution round, concentrated on the mechanics with short-horizon state effects.

**vs. Game Initialization** — On a case where the first generated game is runnable but has a weak or incoherent mechanic, the single-shot baseline has no rejection path and must accept whatever the LLM emits, so quality is capped by first-draft variance. Our method instead iterates: each proposal is gated, rejected edits return an identity residual that steers the next LLM proposal, and only accepted edits advance state. We expect a steady rise in accepted-edit yield across rounds and a clear separation from the flat single-shot baseline, with the largest margin on entities whose reference renders are most distinctive.

**vs. IAMG with direct embedding-distance gate** — On a case where a new-pose mechanic produces a large but identity-preserving change in the IP-Adapter embedding, a direct cosine threshold would reject it because the embedding distance conflates pose with identity. Our subspace residual, calibrated on held-out new views, projects out pose variation, so it accepts the edit. We expect the direct-distance ablation to have lower accepted-edit yield on the identity-sensitive subset, demonstrating that the subspace formulation is necessary.

**vs. IAMG with unsupervised SVD subspace** — On a case where the reference renders contain limited pose variation, the unsupervised SVD subspace may not capture nuisance directions, causing legitimate new-pose mechanics to be rejected as identity drift. The supervised calibration (Phase 0b) explicitly regresses out pose/background, so the residual better reflects identity. We expect the unsupervised variant to have higher false rejection rate on new-pose mechanics, confirming the need for calibration.

### What would falsify this idea

If the accept-all ablation matches IAMG's accepted-edit yield and identity scores, the gates are not doing the claimed work and the coupling story is wrong. Likewise, if gains are uniform across identity-sensitive and identity-neutral subsets rather than concentrated where the predicted failure modes occur, the mechanism explanation fails even if the headline number rises. Additionally, if the direct embedding-distance ablation matches IAMG's separation, the subspace formulation is unnecessary.

## References

1. Playfulness as Feedback for Agentic Game Development
