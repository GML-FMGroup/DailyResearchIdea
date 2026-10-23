# Bidirectional Grounding Certificates as an Observation-Modality Closure Gate for Zero-Shot VLA Policies

## Motivation

Vision-language-action policies such as π0.5 and RT-2 treat the observation interface as fixed — RGB frames plus a textual instruction — and assume the two streams are mutually aligned before committing actions; co-training over heterogeneous data (π0.5) and web-scale co-fine-tuning (RT-2) expand capacity and data coverage but never verify that the instruction's referents are grounded to specific visual evidence or that the scene's affordances satisfy the instruction's preconditions. Structurally, alignment is assumed rather than checked, so a policy can execute a confidently-wrong action when the language stream names an object absent from the scene or the visual scene violates the instruction's preconditions, and no mechanism inside the backbone can detect this because the two streams are only ever fused, never tested against each other. The bottleneck at this frontier is therefore interface integrity, not model capacity.

## Key Insight

A shared latent that must reconstruct each modality from the other — visual evidence from the instruction's referent slots and instruction tokens from the scene's affordance slots — makes mutual identifiability a measurable residual, so a low cross-reconstruction residual is a certificate that the two streams pick out the same scene entity, and this certificate is computable before any action is decoded because it depends only on the observation interface, not on action outcomes.

## Method

**What it is.** We introduce the Observation-Closure Certificate (OCC), a bidirectional grounding certificate computed from a shared latent inside a VLA/VLN policy. Inputs: a visual stream (patch tokens from the policy's vision encoder) and a textual instruction (token embeddings). Outputs: a scalar certificate score c ∈ [0,1] and, if c ≥ τ, the action decoded by the policy; if c < τ, action commitment is withheld and the policy emits a grounding-repair query (re-observation / clarification) instead of acting.

**How it works.**

```pseudocode
# Inputs: visual patch tokens V = {v_1..v_N}, instruction tokens T = {t_1..t_M}
# Hyperparameters: K referent slots = 8, L affordance slots = 8, τ = 0.6, λ = 0.5

1. Encode: H_v = VisionEncoder(V); H_t = TextEncoder(T)
2. Shared latent Z = CrossAttention(query=H_t, key=H_v, value=H_v)
   # Z has K referent slots + L affordance slots, each a d-dim vector
3. FORWARD grounding (instruction -> vision):
     For each referent slot z_k in Z:
        a_k = argmax_j  <z_k, H_v[j]>          # attend to visual patch
        p_k = softmax_j <z_k, H_v[j]>          # grounding distribution
     L_fwd = - sum_k log p_k[a_k]              # referent must land on visual evidence
4. BACKWARD grounding (vision -> instruction):
     For each affordance slot z_l in Z:
        q_l = softmax_m <z_l, H_t[m]>          # distribution over instruction tokens
     L_bwd = - sum_l log q_l[m_l*]             # affordance must be licensed by instruction
     where m_l* is the instruction token the slot claims as its precondition
5. Certificate: c = exp( -(L_fwd + λ * L_bwd) )
6. Gate: IF c >= τ: decode action a ~ Policy(Z, H_v); ELSE: emit repair query
```

**Supervision of m_l* (affordance precondition token).** The token index m_l* is not free: it is derived from the instruction parse. We run a lightweight dependency parser (spaCy en_core_web_sm) on the instruction string and extract the verb token and its direct-object noun token; for each affordance slot l we assign m_l* = the index of the verb token if the slot's role is 'action precondition' (e.g., 'open' in 'open the drawer') and the index of the object noun token if the slot's role is 'object precondition' (e.g., 'drawer'). Slot roles are fixed by position: slots 0–3 are action-precondition slots, slots 4–7 are object-precondition slots. This gives a deterministic m_l* per slot without a learned assignment head, so L_bwd is a genuine cross-reconstruction residual against an external parse rather than a self-consistent reconstruction of an arbitrary token. When the parser fails (e.g., imperative fragments), we fall back to m_l* = argmax_m <z_l, H_t[m]> and flag the sample; flagged samples are excluded from τ calibration.

**Training.** The certificate is trained jointly with the action objective: total loss = L_action + α·(L_fwd + λ·L_bwd) with α = 0.3, so the shared latent Z is shaped by both grounding directions and the action head sees a latent that already encodes mutual identifiability. We use AdamW, lr = 1e-4, batch size 64, 30k steps on 8×A100-80GB (≈ 96 GPU-hours).

**Why this design.** We chose a shared latent Z that must reconstruct each modality from the other over a separate post-hoc verifier bolted onto the policy because a bolted-on verifier can only check consistency of the final action against the observation, whereas the failure we target is upstream of the action; the cost is that Z must carry both referent and affordance structure, which slightly reduces the capacity available to the action head. We chose a bidirectional residual (forward referent grounding plus backward affordance licensing) over a single-direction grounding score because a one-directional check cannot detect the case where the instruction names an object that is present but whose preconditions the scene violates — the backward direction is what catches that — accepting the cost of a second decoder head. We chose a hard gate at τ = 0.6 over a soft penalty on the action logits because a soft penalty still commits an action, and the motivation requires closure before commitment; the cost is that a miscalibrated τ can withhold valid actions, which we mitigate by setting τ from a held-out split at the 90th percentile of the certificate on successful demonstrations. A domain expert would not describe OCC as a variant of PaliGemma-style VLM transfer or of RT-2-style action tokenization: those methods fuse the two streams into a single decoding surface and never compute a cross-reconstruction residual, so they have no quantity that can be thresholded before action commitment.

**Why it measures what we claim.** The forward grounding loss L_fwd measures whether the instruction's referents are grounded to specific visual evidence because it is the negative log-likelihood of the referent slot attending to its claimed visual patch under the assumption that a grounded referent has a peaked attention distribution over exactly one visual entity; this assumption fails when two visually similar entities are both present, in which case L_fwd reflects ambiguity rather than absence and the certificate is conservatively low rather than falsely high. The backward grounding loss L_bwd measures whether the scene's affordances are consistent with the instruction's preconditions because it is the negative log-likelihood of the affordance slot reconstructing the instruction token it claims as its precondition under the assumption that a licensed affordance has a peaked distribution over the instruction tokens that authorize it; this assumption fails when the instruction is underspecified (e.g., 'clean up'), in which case L_bwd reflects instruction vagueness and the repair query is the correct output. The certificate c = exp(-(L_fwd + λ L_bwd)) measures observation-modality closure because it is monotone in the joint cross-reconstruction residual under the assumption that mutual identifiability is equivalent to low cross-reconstruction error in the shared latent; this assumption fails when the shared latent collapses to a constant, in which case both losses are high and c is low, which is the safe failure direction for a gate. The gate threshold τ measures the commitment boundary because it is set from the empirical distribution of c on successful demonstrations under the assumption that demonstrations are grounded by construction; this assumption fails when demonstrations contain mislabeled instructions, in which case τ is set too low and the gate admits some ungrounded actions, which we bound by filtering demonstrations with c below the 50th percentile.

**Load-bearing assumption (stated explicitly).** The certificate c is a calibrated, thresholdable measure of whether the instruction's referents are actually grounded in the scene — i.e., that low cross-reconstruction NLL implies correct grounding rather than merely confident, self-consistent attention. This assumption is not automatic: cross-attention NLL measures the model's own attention confidence, and a slot can achieve low L_fwd by attending to any visually salient patch and low L_bwd by reconstructing any plausible instruction token, with no term tying either to the true referent. The dominant failure is confident-but-wrong grounding, which produces low loss and high c — the unsafe direction. We therefore add two concrete diagnostics that must pass before the certificate is trusted, and we report them alongside every result:

(D1) **Shuffled-pair control.** For each evaluation batch we compute c on the true (V, T) pairs and on shuffled pairs (V, T') where T' is the instruction from a different episode in the batch. If c is a genuine grounding measure, the distribution of c on shuffled pairs must be strictly lower than on true pairs; we report the mean c_true − c_shuffled gap and require it to be > 0.15 for the certificate to be considered valid. A collapsed or trivially aligned latent yields c_true ≈ c_shuffled and fails this control.

(D2) **Slot-entropy diagnostic.** For each referent slot we compute the entropy of p_k over visual patches, and for each affordance slot the entropy of q_l over instruction tokens. A collapsed latent produces near-uniform distributions (high entropy) yet can still yield low NLL if the argmax happens to land on a frequent patch/token; we therefore report mean slot entropy on true pairs and require it to be below 0.5·log(N) for referent slots and below 0.5·log(M) for affordance slots. Slots that exceed this threshold are flagged as degenerate and their contribution to c is zeroed (c is set to 0 for that sample), which makes the gate conservative rather than falsely permissive.

These diagnostics are computed at evaluation time only (no extra training), cost one forward pass per batch, and are reported in every table as 'c_true − c_shuffled' and 'mean slot entropy'.

## Contribution

(1) The Observation-Closure Certificate, a bidirectional grounding certificate computed from a shared latent that gates action commitment in zero-shot VLA/VLN policies without a separate verifier. (2) The design principle that mutual identifiability between visual and textual streams is operationalized as a symmetric cross-reconstruction residual whose two directions catch distinct failure modes — referent absence (forward) versus precondition violation (backward) — and that this residual is thresholdable before action decoding. (3) A grounding-repair query mechanism that converts a failed certificate into a re-observation or clarification action rather than a committed action, with the certificate threshold calibrated from demonstration statistics.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | π0.5 open-world home manipulation suite, restricted to 3 task families (pick-and-place, open/close, clean-up) with released checkpoints; 120 episodes per family, 40% ambiguous-referent, 30% precondition-violating, 30% unambiguous | Long-horizon tasks where grounding breaks in novel homes; reproducible subset |
| Primary metric | Ungrounded-commitment rate on ambiguous and precondition-violating scenes (fraction of episodes where the policy commits an action with c ≥ τ but the action is judged ungrounded by human annotator) | Directly detects the failure OCC gates against |
| Secondary metric | Task success rate after repair queries are executed (re-observation or clarification resolved by oracle) | Tests whether the gate improves real outcomes, not just abstains |
| Baseline | π0.5 (released checkpoint) | Tests whether upstream gating beats stronger action policy |
| Baseline | PaliGemma-style VLM-transfer policy (same vision/text encoders, fused decoding surface, no certificate) | Tests whether a fused decoding surface can self-detect |
| Baseline | Single-direction grounding verifier (forward L_fwd only, same gate τ) | Tests whether backward affordance licensing is necessary |
| Baseline | Cycle-consistency baseline (forward + backward reconstruction, no gate, action always committed) | Tests whether the gate, not just the residual, is load-bearing |
| Baseline | Learned post-hoc verifier (2-layer MLP on pooled Z, trained with binary grounding labels from the same demonstrations) | Tests whether OCC's residual is reducible to a supervised verifier |
| Ablation-of-ours | OCC with soft penalty (c multiplied into action logits) instead of hard gate | Tests whether closure-before-commitment matters |
| Ablation-of-ours | OCC without D1/D2 diagnostics (raw c) | Tests whether the diagnostics are necessary for calibration |

**τ calibration protocol (pre-registered).** τ is set to the 90th percentile of c on a held-out set of 512 successful demonstrations from the training distribution, computed once before evaluation and frozen. Demonstrations with c below the 50th percentile are filtered from the calibration set. The same τ is used for OCC, the single-direction verifier, and the soft-penalty ablation to ensure fair comparison.

### Why this setup validates the claim

The central claim is that ungrounded commitment arises upstream of the action, in the shared latent, and can only be caught by a bidirectional cross-reconstruction residual that is thresholded before commitment. The π0.5 suite supplies exactly the regime where this should bite: novel homes where referents are visually ambiguous and scene affordances violate instruction preconditions. The primary metric, ungrounded-commitment rate on ambiguous and precondition-violating scenes, is chosen because it is the quantity the gate directly controls; a method that merely improves task success without shifting this rate is not doing what OCC claims. The secondary metric, task success after repair queries, ensures the gate is not merely abstaining: a method that withholds action but never recovers is not useful. π0.5 tests whether OCC beats a stronger action policy despite a weaker head, isolating the value of the gate; the PaliGemma-style transfer policy tests whether a fused decoding surface can produce any thresholdable pre-commitment quantity; the single-direction verifier tests whether the backward affordance direction is load-bearing rather than decorative. The cycle-consistency baseline (no gate) and the learned post-hoc verifier test whether OCC's residual is reducible to either a gate-free residual or a supervised verifier, isolating what is genuinely new. The soft-penalty ablation tests whether the hard gate, not just the residual signal, produces the effect. The D1/D2 diagnostics (shuffled-pair gap > 0.15, slot entropy < 0.5·log(N)) are reported for every method; if OCC fails them, its certificate is not a valid grounding measure and the central claim is falsified. Together these form a falsifiable test: if OCC's advantage is uniform across subsets rather than concentrated on ambiguous and precondition-violating scenes, the causal story is wrong.

### Expected outcome and causal chain

**vs. π0.5** — On a case where a novel kitchen contains two near-identical mugs and the instruction says "put the mug in the sink," π0.5 commits to one mug because its action head fuses vision and language into a single decoding surface with no pre-commitment residual. Our method instead drives L_fwd high on the ambiguous referent slot, pushes c below τ, and emits a re-observation query. We expect a noticeable gap on the ambiguous-referent subset (OCC ungrounded-commitment rate ≤ 0.10 vs. π0.5 ≥ 0.35) but parity on unambiguous scenes, where π0.5's stronger policy should match or slightly beat us. On the secondary metric, we expect OCC to recover ≥ 60% of withheld episodes after one repair query, versus π0.5's 0% (it never withholds).

**vs. PaliGemma-style VLM-transfer policy** — On a case where the instruction says "open the drawer" but the drawer is already open, the fused transfer policy decodes an opening action because its single decoding surface never reconstructs the instruction token from scene affordances. Our method instead raises L_bwd, since the affordance slot cannot license "open" from an already-open drawer, and withholds commitment. We expect a concentrated gap on precondition-violating scenes (OCC ungrounded-commitment rate ≤ 0.15 vs. transfer policy ≥ 0.40) and near-parity elsewhere, with the gap widest where instruction preconditions are explicit (verb + object parse succeeds).

**vs. single-direction grounding verifier** — On a case where the instruction names a present object whose preconditions the scene violates, the forward-only verifier scores c high because the referent grounds cleanly, and it commits. Our method instead catches the violation through L_bwd and gates. We expect the forward-only verifier to match OCC on referent-absence scenes (both ≤ 0.12 ungrounded-commitment) but to lose most of its advantage on precondition-violation scenes (forward-only ≥ 0.35 vs. OCC ≤ 0.15), isolating the backward direction as the source of the gain.

**vs. cycle-consistency baseline (no gate)** — The cycle-consistency baseline computes the same L_fwd + λ L_bwd but always commits the action. We expect it to match OCC on the residual signal (similar c distribution) but to have ungrounded-commitment rate ≥ 0.30 on ambiguous scenes, showing that the gate, not just the residual, is load-bearing.

**vs. learned post-hoc verifier** — The post-hoc verifier is a 2-layer MLP (hidden=256, GeLU) on pooled Z trained with binary grounding labels from the same demonstrations. We expect it to underperform OCC on precondition-violation scenes (≥ 0.25 vs. OCC ≤ 0.15) because it lacks the explicit backward reconstruction term, and to be more sensitive to label noise in demonstrations. If it matches OCC, the cross-reconstruction residual is reducible to a supervised verifier and the novelty claim is weakened.

**vs. soft-penalty ablation** — The soft-penalty ablation multiplies action logits by c instead of hard-gating. We expect it to have ungrounded-commitment rate ≥ 0.25 on ambiguous scenes (vs. OCC ≤ 0.10) because it still commits an action, confirming that closure-before-commitment is the operative ingredient.

**D1/D2 diagnostics** — We expect OCC to show c_true − c_shuffled ≥ 0.20 and mean slot entropy ≤ 0.4·log(N) on true pairs, passing both controls. A collapsed latent would show c_true − c_shuffled ≈ 0 and entropy near log(N); we report these numbers for every method and every subset.

### What would falsify this idea

If OCC's improvement over the single-direction verifier is uniform across referent-absence and precondition-violation subsets, rather than concentrated on the latter, then the backward affordance direction is not doing the work we claim and the central mechanism is wrong. Likewise, if the soft-penalty ablation matches the hard gate on ungrounded-commitment rate, then closure-before-commitment is not the operative ingredient. Finally, if OCC fails the D1 shuffled-pair control (c_true − c_shuffled < 0.15) or the D2 slot-entropy control (mean entropy > 0.5·log(N)), then the certificate is measuring training-fit rather than grounding and the load-bearing assumption is false.

## References

1. π0.5: a Vision-Language-Action Model with Open-World Generalization
2. PaliGemma: A versatile 3B VLM for transfer
3. π0: A Vision-Language-Action Flow Model for General Robot Control
4. Visual Instruction Tuning
5. RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control
6. RT-1: Robotics Transformer for Real-World Control at Scale
7. Self-Instruct: Aligning Language Models with Self-Generated Instructions
8. Scaling Instruction-Finetuned Language Models
