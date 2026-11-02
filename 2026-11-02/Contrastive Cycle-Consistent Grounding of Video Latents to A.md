# Contrastive Cycle-Consistent Grounding of Video Latents to Actions without Action Labels

## Motivation

WorldBridge and similar video-to-action bridges (e.g., Motus) rely on supervised inverse-dynamics estimation, so the video-to-action mapping is only identifiable when action labels are present; without them, the structured bridge is underdetermined and cannot guarantee that distinct actions producing the same visual future are separated. This label dependence structurally limits transfer to out-of-distribution videos, because the bridge has no constraint forcing video-derived latents to align with action-derived latents in the absence of paired supervision.

## Key Insight

Enforcing a contrastive cycle-consistency between video-derived and action-derived latent codes through a shared, mutual-information-regularized latent space makes the video-to-action grounding mapping invertible and identifiable without action labels, because cycle consistency forces the two modalities to agree on a single latent code per transition while mutual-information maximization prevents collapse of that code.

## Method

**What it is:** CCG (Contrastive Cycle-Consistent Grounding) is a training objective and latent-space architecture that augments a WorldBridge-style video-action model with a label-free cycle-consistency and contrastive objective. Inputs are video-only clips (no actions) and action-labeled clips; outputs are a shared latent action code per transition and a cycle-consistent video-to-action grounding map. **Load-bearing assumption (stated explicitly):** The action-derived latent z_a retains information that distinguishes actions with identical or null visual deltas, so that mutual-information maximization between z_v and z_a can transfer that discriminability into the video-derived latent even though the video input contains no evidence of the difference. This assumption is testable: we measure I(z_v; z_a) on aliased pairs and expect it to be bounded by I(video; z_a), which is near zero for aliased pairs; the method's contribution is therefore restricted to the identifiable subspace (actions with distinct visual signatures), and we report null-effect/aliased cases as an explicit negative result with a bound on achievable I(z_v; z_a).

**How it works:**
```pseudocode
# Two encoders: E_v (video transition encoder) and E_a (action sequence encoder)
# Shared latent space Z; decoder D_v: Z -> predicted visual delta
# Grounding map G: Z_v -> Z_a (video-derived code to action-derived code)
# Encoders: E_v = 2-layer MLP (hidden=256, GeLU) on ResNet-18 frame features; E_a = 2-layer MLP (hidden=256, GeLU) on action sequence
# Decoders: D_v = 2-layer MLP (hidden=256, GeLU) -> visual delta; D_a = 2-layer MLP (hidden=256, GeLU) -> action
# G and G_inv: 2-layer MLPs (hidden=256, GeLU)
# MINE estimator: 2-layer MLP (hidden=128, GeLU) with gradient penalty (lambda_gp=10.0)

for each batch:
    # 1. Encode video-only clips (no action labels)
    z_v = E_v(video_clip)                # latent action code from visual transition
    # 2. Encode action-labeled clips
    z_a = E_a(action_seq)                # latent action code from actions
    # 3. Cycle consistency: G(z_v) must match z_a for paired clips,
    #    and G^{-1}(z_a) must match z_v; for unpaired clips, enforce
    #    cycle G^{-1}(G(z_v)) ≈ z_v and G(G^{-1}(z_a)) ≈ z_a
    z_a_hat = G(z_v)
    z_v_hat = G_inv(z_a)
    L_cycle = ||z_a_hat - z_a||^2 + ||z_v_hat - z_v||^2  # paired
             + ||G_inv(G(z_v)) - z_v||^2               # unpaired video-only
    # 4. Contrastive alignment: z_v and z_a from the same transition are positive pairs;
    #    z_v from different transitions are negatives. InfoNCE loss with temperature tau=0.1
    L_contrast = InfoNCE(z_v, z_a, tau=0.1)
    # 5. Mutual-information maximization between z_v and z_a (via MINE estimator)
    L_mi = -MINE(z_v, z_a)   # maximizes I(z_v; z_a)
    # 6. Reconstruction: D_v(z_v) predicts visual delta; D_a(z_a) predicts action
    L_recon = ||D_v(z_v) - delta_video||^2 + ||D_a(z_a) - action||^2
    # Total loss
    L = L_recon + lambda_cycle * L_cycle + lambda_contrast * L_contrast + lambda_mi * L_mi
    # lambda_cycle=1.0, lambda_contrast=0.5, lambda_mi=0.1
```

**Why this design:** We chose a contrastive InfoNCE loss over a simple L2 alignment because contrastive learning explicitly separates negative pairs (different transitions) while pulling positive pairs together, which is necessary to prevent the latent space from collapsing to a constant; the trade-off is that it requires careful negative sampling and a temperature hyperparameter. We chose a cycle-consistency constraint over a direct supervised mapping because cycle consistency enforces invertibility without requiring action labels for every video, accepting the cost that it introduces two additional mapping networks (G and G_inv) that must be trained jointly. We chose mutual-information maximization (MINE) over a variational autoencoder's KL term because MI directly measures the shared information between video and action latents, which is the quantity that guarantees identifiability, accepting the cost of a min-max optimization that can be unstable; we stabilize it with a gradient penalty. A domain expert would not describe this as a variant of WorldBridge because WorldBridge uses a supervised inverse-dynamics bridge, whereas our method introduces a label-free cycle-consistency and contrastive objective that operates on the latent space itself, not on the action output.

**Why it measures what we claim:** The contrastive InfoNCE loss measures the alignment between video-derived and action-derived latent codes because it assumes that transitions with the same underlying action should have similar latents; this assumption fails when two different actions produce identical visual deltas (e.g., pushing an object from opposite sides resulting in the same final frame), in which case the contrastive loss would incorrectly align them, reflecting visual similarity rather than action identity. The cycle-consistency loss measures invertibility of the grounding map because it assumes that the composition of G and G_inv is the identity on the latent space; this assumption fails when the latent space has a non-trivial null space (e.g., actions that have no visual effect), in which case the cycle loss would be minimized by collapsing those dimensions, reflecting degeneracy rather than invertibility. The mutual-information term measures the shared information between modalities because it assumes that the MINE estimator is consistent; this assumption fails when the estimator overfits to finite samples, in which case the MI estimate would be biased upward, reflecting overfitting rather than true dependence.

## Contribution

['A contrastive cycle-consistency objective that enforces the video-to-action grounding mapping to be invertible and identifiable without action labels, by aligning video-derived and action-derived latent codes through a shared latent space regularized by mutual-information maximization.', 'A training procedure that combines cycle consistency, contrastive alignment, and mutual-information maximization to learn a shared latent action space from unpaired video and action data, enabling label-free grounding.', 'An empirical demonstration that the proposed method improves out-of-distribution task transfer on RoboTwin 2.0 compared to supervised inverse-dynamics baselines, particularly when action labels are sparse.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|-----------------------|
| Dataset | RoboTwin 2.0 with held-out action-visual aliasing tasks | Contains transitions where visual delta is ambiguous across actions. |
| Primary metric | Success rate on aliased-action test subset | Directly measures disambiguation of visually identical transitions. |
| Baseline | WorldBridge (supervised inverse-dynamics bridge) | Tests whether label-free cycle-consistency beats supervised bridging. |
| Baseline | Motus (latent action world model) | Tests whether explicit identifiability constraint improves over implicit latent alignment. |
| Baseline | Video-only contrastive (no cycle/MI) | Isolates contribution of cycle-consistency and MI beyond contrastive alignment. |
| Ablation-of-ours | CCG without MI term (L_mi=0) | Tests whether mutual-information maximization is necessary for identifiability. |
| Diagnostic metric | Linear probe accuracy on aliased pairs | Measures latent separation directly, not just downstream success. |

### Why this setup validates the claim

The central claim is that label-free cycle-consistency plus contrastive alignment and MI maximization yields action latents that are identifiable even when visual deltas are ambiguous. The dataset choice is critical: RoboTwin 2.0 provides controlled manipulation tasks where we can construct a held-out subset of action-visual aliasing cases—transitions where two different actions produce nearly identical final frames (e.g., pushing an object from opposite sides to the same location). On this subset, any method that relies on visual similarity alone will fail to separate the action latents. WorldBridge tests the supervised-bridge alternative: if supervised inverse dynamics were sufficient, it would match our method on aliased cases; if our label-free constraint adds value, we expect a gap. Motus tests whether a state-of-the-art latent action world model already captures identifiability implicitly; a gap here would show that explicit cycle and MI constraints are necessary. The video-only contrastive baseline isolates the effect of cycle-consistency and MI: if our method outperforms it, the gain is attributable to those terms, not to contrastive alignment alone. The ablation (no MI) tests the specific role of mutual-information maximization: if removing it degrades performance on aliased cases but not on unambiguous ones, that confirms MI is the mechanism enforcing identifiability. The primary metric—success rate on the aliased-action test subset—is the right one because it directly measures the downstream consequence of correct action disambiguation; on unambiguous subsets all methods should perform similarly, so the metric is diagnostic precisely where the predicted failure mode occurs. The diagnostic linear probe on aliased pairs provides a direct measure of latent separation, testing whether the method separates aliased actions in latent space rather than only downstream success.

### Expected outcome and causal chain

**vs. WorldBridge** — On a case where two different actions (e.g., push left vs. push right) yield the same final object pose, WorldBridge's supervised inverse-dynamics bridge maps the visual delta to an action by minimizing output error, but because the visual input is identical for both actions, it collapses to the average or the more frequent action in training, producing systematically wrong actions on the minority class. Our method instead uses cycle-consistency to enforce that the video-derived latent code must be invertible to the action-derived code, and contrastive negatives from different transitions prevent the two actions from sharing a latent, so the aliased pair is separated in latent space even though their visual deltas are similar. We expect a noticeable gap on the aliased-action subset (our method substantially higher success) but near parity on unambiguous subsets, with the gap concentrated on the minority action of each aliased pair. The linear probe on aliased pairs should show higher accuracy for our method, confirming latent separation.

**vs. Motus** — On a case where an action has no visible effect (e.g., a small joint movement that does not move the object), Motus's latent action world model, trained to predict future frames, assigns near-zero latent change to that transition because the visual delta is null, so the latent code cannot distinguish that action from a no-op. Our method instead uses mutual-information maximization between video and action latents, which forces the action-derived latent to retain information about the action even when the visual delta is null; the cycle-consistency term then requires that the video-derived latent for that transition still map back to the correct action code, preventing collapse of null-effect dimensions. We expect our method to show a clear advantage on null-visual-effect actions (higher success rate) while matching Motus on actions with strong visual signatures. However, as stated in the load-bearing assumption, this advantage is bounded by I(video; z_a), which is near zero for null-effect actions; we therefore expect the MI term to increase action-side entropy but not inject information into z_v, and we report this as a negative result with a bound on achievable I(z_v; z_a).

**vs. Video-only contrastive** — On a case where two transitions have similar visual deltas but different actions, the video-only contrastive baseline pulls their video latents together because they are visually similar, and without a cycle or MI constraint there is no force to separate them, so the downstream action prediction confuses the two. Our method adds cycle-consistency, which requires the video latent to be invertible to the action latent, and MI maximization, which preserves action information in the shared latent; together these prevent the visually similar but action-different pair from collapsing. We expect the video-only baseline to perform well on unambiguous transitions but to degrade sharply on the aliased subset, while our method maintains high success on both, yielding a large interaction effect (method × subset aliasing). The linear probe on aliased pairs should show near-chance accuracy for the video-only baseline and higher accuracy for our method.

### What would falsify this idea

If our method's gain over WorldBridge and Motus is uniform across aliased and unambiguous subsets—i.e., no interaction between method and subset aliasing—then the central claim that cycle-consistency and MI specifically resolve visual-action ambiguity is wrong; the improvement would instead be due to generic capacity or training effects. Additionally, if the no-MI ablation matches full CCG on aliased cases, then mutual-information maximization is not the mechanism enforcing identifiability, falsifying the proposed causal chain. Furthermore, if the linear probe on aliased pairs shows near-chance accuracy for our method, the load-bearing assumption that z_a retains distinguishing information is false, and the method's contribution is restricted to the identifiable subspace.

## References

1. WorldBridge: Bridging Video and Action for Task Generalization in World Action Models
2. Motus: A Unified Latent Action World Model
