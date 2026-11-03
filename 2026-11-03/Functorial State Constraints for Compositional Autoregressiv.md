# Functorial State Constraints for Compositional Autoregressive Video Generation

## Motivation

Existing interactive video generation systems (e.g., VRAG in Learning World Models for Interactive Video Generation, StreamEval's scripted harness) treat temporal state constraints as independent per-turn predicates: each request is checked or conditioned in isolation, with no composition operator. Consequently, satisfying each constraint individually does not guarantee that their conjunction holds across turns—a structural gap that neither better per-turn evaluators nor longer context windows can close, because the space of constraints lacks a closure property.

## Key Insight

If per-turn constraints are morphisms in a category whose objects are state-assertion bundles, then multi-turn satisfaction is exactly the categorical limit of the induced diagram, and training the generator's latent trajectory to be a functor from this category to the state manifold makes compositionality a structural invariant preserved by construction rather than a post-hoc check.

## Method

**What it is.** FSC-Video (Functorial State-Constraint Video generator) is an autoregressive video generator whose latent state trajectory is trained to be a functor from a user-authored constraint category C to the latent state manifold M. Inputs: per-turn constraint specifications (state assertions over named attributes, e.g., object presence, spatial relations, counts) and action/conditioning tokens. Outputs: generated video frames whose latent states satisfy the categorical limit of the constraint diagram.

**How it works.**

```pseudocode
# Phase 1: Build constraint category C from multi-turn script
C.objects = [B_0, B_1, ..., B_T]          # state-assertion bundles per turn
C.morphisms = {f_t: B_{t-1} -> B_t}      # constraint-induced transition maps
# Each B_t is a set of atomic assertions a = (attribute, value, scope)
# f_t is a partial map: assertions preserved by turn t's request
# Composition: f_{t+k} o ... o f_t is defined by assertion-set intersection

# Phase 2: Functor F: C -> M (latent state manifold)
# Encoder E maps each bundle B_t to a region R_t in M
# E is a learned bundle encoder: 2-layer MLP (hidden=512, GeLU) over the
# concatenation of frozen CLIP text embeddings of each atomic assertion
# string in B_t (CLIP ViT-L/14, 768-d per assertion, mean-pooled over
# assertions then projected to 256-d latent). E is trained jointly with
# an assertion classifier head (linear, 256->|A| logits per attribute,
# cross-entropy loss) on the same atomic schema, so that a probe decoding
# z recovers each atomic assertion above a held-out accuracy threshold
# (see Phase 2b).
# Transition model G (the video generator's latent dynamics) must satisfy:
#   G(E(B_{t-1}), action_t) ∈ R_t   for every morphism f_t
# Functoriality loss:
L_func = Σ_t  d( G(z_{t-1}, a_t),  E(B_t) )        # object preservation
       + Σ_{t,s} d( G(z_t, a_{t+1:s}), E(B_s) )      # composition preservation
# where d is latent-space distance (L2), z_t = G(z_{t-1}, a_t)

# Phase 2b: Encoder calibration gate (must pass before L_func/L_limit trusted)
# Train a linear probe on E's 256-d outputs to decode each atomic assertion
# (attribute, value) from z. Require held-out probe accuracy >= 0.90 per
# attribute on a 512-bundle calibration set drawn from the same schema.
# If any attribute falls below 0.90, E is retrained with that attribute
# upweighted (loss weight 2.0) until the gate passes; L_func and L_limit
# are not used in training until the gate passes.

# Phase 3: Limit loss (multi-turn conjunction)
# The categorical limit of the diagram {B_t} is the maximal bundle B* whose
# projections to each B_t commute with all f_t.
B* = limit({B_t}, {f_t})   # computed by assertion-set intersection + consistency check
L_limit = d( G(z_0, a_{1:T}), E(B*) )

# Phase 4: Training
L_total = L_diffusion + λ_func * L_func + λ_limit * L_limit
# λ_func = 0.1, λ_limit = 0.05 (fixed on a 50-script pilot before the
# full 300-script run; no further tuning on the test set)
# L_diffusion = standard denoising loss on frames
# G is the autoregressive latent transition (DiT backbone, 24 layers)
```

**Why this design.** We chose a categorical formulation over per-turn predicate checking because per-turn checking (as in StreamEval's scripted harness) has no composition operator, so multi-turn satisfaction is not implied; the cost is that the limit B* may be empty for inconsistent constraint sets, which we detect and flag rather than generate. We chose to enforce functoriality as a training loss on latent trajectories rather than as a post-hoc filter because post-hoc filtering cannot correct compounding drift (the irreducible error identified in VRAG); the cost is that the loss is only a soft constraint, so we add a projection step at inference that snaps z_t to the nearest point in R_t. We chose assertion-set intersection for the limit computation over a learned limit predictor because intersection is exact and interpretable; the cost is that it requires constraints to be expressed in a shared attribute vocabulary, which we enforce via a fixed schema. A domain expert would not describe this as a variant of retrieval-augmented generation (VRAG) because VRAG retrieves past video content to condition generation, whereas FSC-Video constructs a categorical diagram from constraint specifications and trains the generator's latent dynamics to be a functor from that diagram—a structural property (functoriality) that retrieval cannot provide.

**Why it measures what we claim.** The functoriality loss L_func measures compositionality of the latent trajectory because we assume that latent-space distance d is a valid proxy for assertion satisfaction (i.e., if z is close to E(B), then the video frame decoded from z satisfies B); this assumption fails when the encoder E collapses distinct assertions to nearby latents, in which case L_func reflects encoder geometry rather than constraint satisfaction. To make this assumption explicit and testable, we state it as: **Assumption A:** E is injective on the assertion space and d is monotone in assertion-violation count. **Failure mode F:** encoder collapse, where distinct assertions map to nearby latents. We add a diagnostic that estimates E's injectivity on the benchmark's assertion set by measuring the held-out linear-probe accuracy per attribute (Phase 2b gate) and reporting the pairwise distance between E(B) for bundles differing in exactly one atomic assertion; if the probe accuracy is below 0.90 or the mean pairwise distance is below 0.1, Assumption A is violated and L_func is not trusted. The limit loss L_limit measures multi-turn conjunction satisfaction because we assume that the categorical limit B* (computed by assertion-set intersection) exactly captures the conjunction of per-turn constraints; this assumption fails when constraints are inconsistent (empty limit), in which case L_limit is undefined and we fall back to flagging the script as unsatisfiable. The projection step at inference measures structural invariant preservation because we assume that snapping to the nearest point in R_t preserves functoriality; this assumption fails when R_t regions overlap in a way that violates composition, in which case the projection may break composition preservation—we detect this by checking the composition loss after projection and rejecting the rollout if it exceeds a threshold (composition loss > 0.5).

## Contribution

['A categorical framework that defines per-turn temporal state constraints as morphisms in a category of state-assertion bundles, with multi-turn satisfaction as the categorical limit—providing a closed composition operator absent from prior per-turn evaluators.', "A training procedure that enforces the generator's latent state trajectory to be a functor from the constraint category to the state manifold via a functoriality loss, making compositionality a structural invariant rather than a post-hoc check.", 'An inference-time projection step that snaps latent states to constraint regions, with a consistency check that flags unsatisfiable constraint sets.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | StreamEval 300-script multi-turn benchmark | Scripted multi-turn constraints with state assertions per turn |
| Dataset (secondary) | 100-script multi-turn benchmark with independently authored attribute vocabulary (e.g., colors, materials, relative sizes) | Tests generalization beyond one schema |
| Primary metric | Multi-turn constraint satisfaction rate (0–100) | Directly measures conjunction over turns, not per-frame fidelity |
| Secondary metric | Encoder calibration: held-out linear-probe accuracy per attribute (threshold 0.90) and mean pairwise distance between E(B) for bundles differing in one assertion | Tests Assumption A (injectivity) |
| Baseline | Naive autoregressive (extended context) | Tests whether context length alone solves composition |
| Baseline | VRAG (retrieval-augmented generation) | Tests whether retrieval substitutes for structural composition |
| Baseline | Per-turn predicate checking harness (StreamEval-style) | Tests whether post-hoc checking without composition suffices |
| Ablation-of-ours | FSC-Video w/o L_func and L_limit (diffusion only) | Isolates contribution of functoriality and limit losses |
| Ablation-of-ours | FSC-Video with frozen CLIP encoder (no learned E, no calibration gate) | Tests whether the learned, calibrated encoder is necessary |

### Why this setup validates the claim

The central claim is that enforcing functoriality and limit losses on latent trajectories yields multi-turn constraint satisfaction that per-turn checking and retrieval cannot provide. The StreamEval scripted benchmark is the right test bed because it supplies multi-turn constraints as state assertions, so the same scripts can be scored both by per-turn predicate checking and by full-diagram conjunction—the exact contrast the claim turns on. The extended-context baseline tests whether composition emerges from capacity alone; VRAG tests whether retrieval of past content substitutes for structural composition; the per-turn harness tests whether checking each turn independently implies multi-turn satisfaction. The diffusion-only ablation is the decisive control: if our gains come from the categorical losses rather than from the DiT backbone or diffusion loss, removing L_func and L_limit must collapse performance toward the baselines. The frozen-CLIP ablation tests the load-bearing assumption that the encoder E is injective on the assertion space (Assumption A): if the learned, calibrated encoder is necessary, this ablation should underperform FSC-Video, especially on composition-heavy scripts. The secondary 100-script benchmark with a different attribute vocabulary tests whether the structural gain generalizes beyond one schema. The primary metric is multi-turn satisfaction rate rather than frame quality because the claim is about conjunctions across turns; a method that improves frame fidelity but not conjunction would not support the thesis. Together these choices form a falsifiable test: the claim predicts a specific ordering—FSC-Video > ablation ≈ baselines—with the gap concentrated on scripts where constraints must compose across turns, and the calibration diagnostic must pass (probe accuracy ≥ 0.90) for the losses to be trusted.

### Expected outcome and causal chain

**vs. Naive autoregressive (extended context)** — On a script where an object introduced at turn 2 must persist while a spatial relation is added at turn 5, the extended-context baseline drifts: later frames drop the object or violate the relation because longer context does not impose a composition operator, only more conditioning signal. Our method instead carries z_t constrained to R_t and snaps to the nearest point in R_t at inference, so the turn-5 relation cannot overwrite the turn-2 assertion without violating functoriality. We expect a clear gap on scripts with long-range persistence and composition, with near-parity on single-turn scripts where drift cannot compound.

**vs. VRAG (retrieval-augmented generation)** — On a script where a constraint is stated once early and must hold through many turns, VRAG retrieves past frames or tokens as conditioning but has no mechanism that makes the retrieved content a compositional constraint; it can retrieve the right frame and still generate a frame that violates the assertion because retrieval conditions rather than constrains. Our method instead encodes the assertion bundle into E(B_t) and penalizes latent distance to it at every morphism, so the constraint is enforced structurally rather than suggested statistically. We expect VRAG to show high variance across turns—sometimes satisfying, sometimes not—while our method shows a stable satisfaction profile, with the largest gap on scripts with many composed morphisms.

**vs. Per-turn predicate checking harness (StreamEval-style)** — On a script where each turn is individually satisfiable but the conjunction is not (e.g., turn 3 requires a count of two and turn 6 requires a count of one without an intervening removal action), the per-turn harness scores each turn as satisfied while the multi-turn conjunction fails, because it has no composition operator and no limit computation. Our method instead computes B* by assertion-set intersection and either satisfies the limit or flags the script as unsatisfiable, so it never reports satisfaction on an inconsistent conjunction. We expect the harness to over-report satisfaction on precisely these inconsistent-conjunction scripts while our method reports flagged-unsatisfiable or genuine satisfaction, producing a gap concentrated on the inconsistency subset.

**vs. FSC-Video w/o L_func and L_limit (diffusion only)** — This ablation removes the categorical losses, leaving only the diffusion loss and the learned encoder E. If the categorical losses are the causal mechanism, this ablation should match the baselines on composition-heavy scripts and underperform FSC-Video by a margin concentrated on scripts with long-range persistence and many composed morphisms. We expect the gap to be near zero on single-turn scripts, confirming that the losses do not harm simple cases.

**vs. FSC-Video with frozen CLIP encoder (no learned E, no calibration gate)** — This ablation replaces the learned, calibrated encoder E with a frozen CLIP-style text encoder applied directly to assertion bundles, as in the original method. If Assumption A (injectivity) is load-bearing, this ablation should show degraded performance on scripts requiring fine compositional distinctions (e.g., spatial relations, counts), because CLIP collapses such distinctions. We expect the calibration diagnostic to fail (probe accuracy < 0.90) for this ablation, and its multi-turn satisfaction rate to be lower than FSC-Video, especially on the composition-heavy subset.

### What would falsify this idea

If FSC-Video's gain over the diffusion-only ablation is uniform across all script subsets—including single-turn scripts where no composition is required—rather than concentrated on scripts with long-range persistence, many composed morphisms, and inconsistent conjunctions, then the categorical losses are not doing the compositional work the claim attributes to them. Likewise, if the ablation matches FSC-Video on the composition-heavy subset, the functoriality and limit losses are not the causal mechanism. Additionally, if the encoder calibration diagnostic fails (probe accuracy < 0.90) yet FSC-Video still performs well, Assumption A is not necessary and the categorical machinery may be decoration.

## References

1. Learning World Models for Interactive Video Generation
2. Diffusion Models Are Real-Time Game Engines
3. SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis
4. GAIA-1: A Generative World Model for Autonomous Driving
5. MAGVIT: Masked Generative Video Transformer
6. Phenaki: Variable Length Video Generation From Open Domain Textual Description
7. StreamEval: Evaluating Interactive Streaming Video Generation Models
