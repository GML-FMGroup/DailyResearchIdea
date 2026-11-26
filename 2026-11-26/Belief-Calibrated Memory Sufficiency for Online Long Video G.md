# Belief-Calibrated Memory Sufficiency for Online Long Video Generation

## Motivation

Online long video generators must commit entity memory before the next prompt arrives, so memory selection is a decision under partial observability; Agentic LongLive handles this by training a small language model with a downstream-video-calibrated image reward, which optimizes remembered image quality rather than the probability that a future unseen prompt will reference an entity. Because that policy is never scored on whether the memory it kept actually sufficed for the next revealed prompt, the prompt-only memory agent and the render-grounded auditor remain uncalibrated to each other, and sufficiency — the property the system implicitly assumes — is never measured, only rewarded. The same residual gap recurs in any asynchronous memory pipeline that must act before future references are known.

## Key Insight

Memory sufficiency is a predictive quantity over the next prompt's entity-reference distribution, so it can be estimated as a belief and then calibrated by the auditor's delayed render residuals, making it a verifiable selection criterion rather than an unobserved assumption baked into an image reward.

## Method

**What it is.** We introduce BMS (Belief-calibrated Memory Sufficiency), a memory-selection head that replaces Agentic LongLive's reward-only policy. Inputs: the revealed prompt prefix, the current entity-memory store (text state + evolving images per entity), and a calibration buffer of past (selection, next-prompt, auditor-residual) triples. Outputs: a per-entity sufficiency score s(e) and a selected memory set M_t passed to the attention-sink prefiller.

**How it works.**

```pseudocode
# Phase 1: belief over future entity references
for each entity e in memory store:
    q_e = MLP_enc(text_state(e), image_embed(e))          # 2-layer MLP, hidden=256, GeLU; text_state is 768-d frozen text encoder output, image_embed is 512-d CLIP ViT-B/32
    b_e = sigmoid( w_q^T q_e + w_h^T h_t + b )              # P(e referenced in next prompt | prefix); w_q, w_h ∈ R^256, b scalar
    # h_t = GRU (hidden=256, 1 layer) over revealed prompt prefix tokens (frozen generator's text encoder)

# Phase 2: sufficiency score with calibrated residual offset
tau_e = render_residual_ema(e)                             # from auditor: EMA of |predicted - rendered| entity fidelity, decay=0.9
s_e = b_e * u_e - lambda * max(0, tau_e - tau_0)           # u_e = utility of keeping e (image quality proxy, CLIP similarity to entity text state)
# lambda=0.5, tau_0 = running median of tau over calibration buffer (initialized to 0.1)

# Phase 3: selection under budget
M_t = top_k(s_e, K=8)                                      # K entities per shot
prefill_attention_sink(M_t)                                # unchanged from Agentic LongLive

# Phase 4: delayed calibration (post-hoc, off critical path)
when prompt_{t+1} revealed:
    y_e = 1[e named in prompt_{t+1}]                       # realized reference
    loss_belief = BCE(b_e, y_e)                            # belief calibration
when shot_t rendered and auditor returns residual r_t:
    for e in M_t: update tau_e with EMA(r_t[e])            # decay=0.9
    push (b_e, y_e, r_t) into calibration buffer
    if |buffer| > N: retrain w_q, w_h by logistic regression on (b_e, y_e) pairs  # N=512, L2 reg=1e-3, 100 epochs
```

**Why this design.** We chose a belief head over the Agentic LongLive image-reward policy because the reward is computed on already-selected memory and therefore cannot score memory that was *not* kept, whereas a belief over future references is defined on every entity in the store; the cost is that the belief must be retrained online as prompts arrive. We chose an EMA residual offset from the auditor over a learned joint reward because the auditor's residual is a delayed but render-grounded signal that the reward-only policy in Agentic LongLive never uses for calibration; the cost is a one-shot lag before the offset stabilizes. We chose top-k selection over a threshold rule because the attention-sink prefiller has a hard capacity, so a budgeted argmax is the only rule whose output is guaranteed to fit; the cost is that entities just below the cutoff are dropped even when their belief is high. A domain expert would not describe this as a variant of Agentic LongLive: that system optimizes an image reward on remembered content, whereas BMS optimizes a predictive belief that is then corrected by the auditor, so the two systems consume different signals and target different quantities.

**Why it measures what we claim.** The belief b_e measures the probability that entity e is referenced in the next prompt because we assume the revealed prefix is a sufficient statistic for the reference process; this assumption fails when a later prompt introduces a brand-new entity never seen in the prefix, in which case b_e reflects prefix-conditional salience rather than true reference probability. The residual offset tau_e measures render-grounded memory insufficiency because we assume the auditor's per-entity residual is monotone in the fidelity loss suffered when e is omitted; this assumption fails when the auditor attributes a global artifact to the wrong entity, in which case tau_e reflects attribution noise rather than that entity's insufficiency. The selected set M_t measures memory sufficiency because we assume top-k under the calibrated score is a budget-feasible maximizer of expected reference coverage; this assumption fails under heavy entity co-reference where keeping one entity implies another, in which case M_t reflects a greedy marginal coverage rather than the joint sufficiency of the set.

**(D) Assumption audit.** The three proxy equivalences above are named assumptions A1–A3. A1: b_e is a calibrated probability of reference; tested by reliability diagram and ECE on held-out (b_e, y_e) pairs. A2: tau_e is monotone in omission-induced fidelity loss; tested by forced-omission interventions (drop e from M_t, measure fidelity drop) and Spearman correlation with tau_e. A3: top-k marginal coverage approximates joint sufficiency; tested by co-reference analysis comparing marginal vs. joint coverage on entity pairs.

## Contribution

['A belief-calibrated memory-sufficiency estimator (BMS) that scores every entity in an asynchronous memory store by its predicted future reference probability, corrected by a render-grounded residual offset from the delayed auditor.', "A post-hoc calibration loop that re-fits the belief head on realized next-prompt references and auditor residuals without blocking the live video stream, turning the auditor's delayed signal into a training target for memory selection.", 'A design principle: memory selection in online long video generation should be trained against future-reference prediction rather than against remembered-content image quality, because the two targets diverge exactly when the next prompt is unseen.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | LongStory-Online | Interactive long-story benchmark with on-the-fly prompts |
| Primary metric | Entity reference coverage on revealed prompts | Directly tests predictive sufficiency of selected memory |
| Secondary metric | Entity-consistency FVD (lower is better) | Links coverage to downstream video quality |
| Baseline | Agentic LongLive reward-only policy | Tests claim that reward signal cannot score unkept entities |
| Baseline | Recency heuristic (keep last-seen entities) | Tests whether belief beats naive temporal salience |
| Baseline | Uniform top-k (no belief, no residual) | Isolates belief and residual contributions |
| Ablation-of-ours | BMS without residual offset (λ=0) | Tests whether auditor calibration adds beyond belief |
| Ablation-of-ours | BMS with belief-only (no utility u_e) | Tests whether utility term contributes |

### Why this setup validates the claim

The central claim is that memory selection should be driven by a calibrated belief over future entity references, corrected by a delayed auditor residual, rather than by a reward computed on already-selected memory. LongStory-Online is the right testbed because its prompts are revealed incrementally, so a selector must commit to a budgeted set before the next reference is known — exactly the regime where the belief signal is defined on all entities while Agentic LongLive's reward is defined only on kept ones. Reference coverage on revealed prompts is the right primary metric because it directly operationalizes what the belief is supposed to predict; if coverage does not improve, the belief is not doing its job regardless of downstream visual quality. Entity-consistency FVD provides a downstream check that coverage gains translate to better video. The reward-only baseline tests the sub-claim that scoring unkept entities matters; the recency baseline tests whether the belief is more than a temporal prior; the uniform top-k baseline isolates whether either signal contributes; and the λ=0 ablation isolates the auditor-residual correction from the belief head itself. Together they form a falsifiable ladder: each baseline removes one mechanism, so any observed gain can be attributed to a specific component.

### Expected outcome and causal chain

**vs. Agentic LongLive reward-only policy** — On a case where an entity is introduced early, dropped from memory for several shots, then re-referenced, the baseline produces wrong behavior because its reward was computed only on the memory it kept, so it has no signal that the dropped entity will matter again. Our method instead assigns that entity a belief from the revealed prefix and a residual offset from the auditor, so it retains it before the re-reference arrives, and we expect a noticeable coverage gap concentrated on re-referenced-but-dropped entities with parity on entities referenced every shot. We also expect a corresponding drop in entity-consistency FVD.

**vs. Recency heuristic** — On a case where the story returns to a protagonist after a long detour through side characters, the baseline produces wrong behavior because recency evicts the protagonist in favor of recently named minor entities. Our method instead keeps the protagonist because the belief head sees prefix cues that precede a return and the residual offset flags fidelity drift on it, so we expect a gap on long-detour segments but parity on dense single-thread segments.

**vs. Uniform top-k (no belief, no residual)** — On a case where entity salience is uneven across the store, the baseline produces wrong behavior because it keeps a fixed arbitrary set and misses high-reference entities. Our method instead ranks by calibrated sufficiency, so we expect a moderate uniform gain across all subsets, which is the weakest signal of the three and mainly confirms the belief head is not noise.

**vs. BMS without residual offset (λ=0)** — On a case where the auditor detects a global artifact misattributed to a low-fidelity entity, the ablation produces wrong behavior because it trusts belief alone and over-keeps entities whose render quality has decayed. Our method instead down-weights them via the residual offset, so we expect the ablation to close most of the gap on clean segments but trail on segments with accumulated render drift, isolating the auditor's contribution.

**vs. BMS with belief-only (no utility u_e)** — On a case where image quality varies across entities, the baseline produces wrong behavior because it ignores the utility term and may select entities with high belief but low visual fidelity. Our method instead weights belief by utility, so we expect a small but consistent gain, mainly on entities with degraded images.

### What would falsify this idea

If the gain over the reward-only baseline is uniform across subsets rather than concentrated on re-referenced-but-dropped entities and long-detour segments, the belief head is not predicting future references and the central claim fails. Likewise, if the λ=0 ablation matches full BMS everywhere, the auditor-residual correction is doing no work and the design reduces to a belief-only selector.

## References

1. Agentic LongLive: Asynchronous Agentic Memory for Long Video Generation
