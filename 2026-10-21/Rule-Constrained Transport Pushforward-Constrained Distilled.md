# Rule-Constrained Transport: Pushforward-Constrained Distilled Game Renderers

## Motivation

Distilled game renderers such as Matrix-Game 4.0, Matrix-Game 2.0, and GameNGen condition a large video generator on action tokens or captions and then assume the rendered frames respect the engine's executable interaction rules; no mechanism enforces that the output distribution is a pushforward of the engine's transition kernel. This meta-gap recurs across branches D, E, and G: rule-adherence is asserted, not established, because conditioning on action-like signals provides no hard constraint on the latent trajectory. Consequently, a renderer can produce visually plausible frames that violate game rules (e.g., passing through walls, impossible health changes) while still scoring well on perceptual metrics.

## Key Insight

If the renderer's latent trajectory is constrained to lie in the image of the engine's executable transition relation, then every rendered frame sequence is the decoder image of an engine-valid state trajectory, so rule-adherence becomes a structural property of the output distribution rather than a hoped-for side effect of conditioning.

## Method

```pseudocode
# Rule-Constrained Transport (RCT)
Input: low-poly/high-fidelity video pairs D = {(x_t, y_t)}_{t=1..T}, engine E with state space S, action space A, transition relation R ⊆ S×A×S (executable), decoder D_θ: Z→X, encoder E_φ: X→Z, inverse model I_ω: Z→S
Output: trained decoder D_θ, latent dynamics f_ψ: Z×A→Z, and inverse model I_ω such that rendered frames = D_θ(z_t) and z_t ∈ Z_valid

# Load-bearing assumption (stated explicitly):
# The encoder E_φ is assumed to be surjective onto S and to preserve all action-relevant invariants,
# so that Euclidean projection Π_C in latent space corresponds to valid state projection in S.
# This assumption is not enforced by reconstruction loss alone; we therefore add a state-space
# projection path via an inverse model I_ω and a cycle-consistency loss (see step 4b).

1. Compile engine rules into a latent constraint set C = {z ∈ Z | ∃ s∈S: z = E_φ(s) and s satisfies E's invariants}
   using the engine's executable rule set (collision, resource, legality predicates) applied to a sampled state cover
   of size N_cover = 10^5 states (uniform grid over walkable regions plus 10^4 random legal states).
   For the initial implementation, restrict to a single engine with rules: (i) collision predicate (agent position
   must lie in walkable region W), (ii) resource counter invariant (resource ∈ [0, R_max]).
   Rule-to-projection compiler: for collision, project position coordinates onto W by clamping to nearest walkable
   cell; for resource, clamp counter to [0, R_max]. These projections are exact for the chosen engine.

2. Train encoder E_φ and decoder D_θ on aligned pairs with reconstruction loss
   L_rec = ||D_θ(E_φ(y_t)) - y_t||^2 + λ_lp ||D_θ(E_φ(x_t)) - y_t||^2 (λ_lp=1.0).

3. Train latent dynamics f_ψ(z_t, a_t) to predict z_{t+1} by minimizing
   L_dyn = ||f_ψ(E_φ(y_t), a_t) - E_φ(y_{t+1})||^2.

4. Enforce the pushforward constraint via a projection operator Π_C: Z→C derived from the engine's rule set:
   Π_C(z) = argmin_{z'∈C} ||z - z'||^2, implemented by projecting onto the feasible set defined by the engine's
   invariants (e.g., clamping positions to walkable regions, snapping resources to valid ranges).

4b. (State-space projection path) Train an inverse model I_ω: Z→S (2-layer MLP, hidden=256, GeLU) to predict
    engine state s from latent z, minimizing L_inv = ||I_ω(E_φ(y_t)) - s_t||^2 using ground-truth states from the
    engine logs. Then define the state-space projection Π_S(s) = argmin_{s'∈S} ||s - s'||^2 using the exact engine
    rules (clamp position to W, clamp resource to [0, R_max]). Enforce consistency between latent dynamics and
    state-space projection via cycle-consistency loss:
    L_cyc = ||E_φ(Π_S(I_ω(f_ψ(z_t, a_t)))) - Π_C(f_ψ(z_t, a_t))||^2.
    This loss encourages the latent projection Π_C to agree with the state-space projection mapped back through E_φ.

5. Train the renderer (decoder) with the rule-constrained transport loss:
   L_RCT = E[ ||D_θ(Π_C(f_ψ(z_t, a_t))) - y_{t+1}||^2 ] + β * L_rec + γ * L_cyc,
   with β=0.5, γ=0.1. Backpropagate through Π_C using a differentiable relaxation (soft projection with
   temperature τ=0.1).

6. At inference, given a starting frame y_0 and action sequence a_0..a_T, compute z_0 = E_φ(y_0),
   roll out z_{t+1} = Π_C(f_ψ(z_t, a_t)), and render y_{t+1} = D_θ(z_{t+1}).
```

**Why this design**
We chose to compile the engine's rules into a latent constraint set C and enforce it via a projection operator Π_C rather than conditioning the renderer on action tokens or captions, because conditioning provides only a soft correlation between actions and visuals while the engine's rules are hard executable constraints; the trade-off is that the projection may discard some visually plausible but rule-invalid latent states, potentially reducing diversity. We chose a differentiable relaxation of Π_C (soft projection) over a hard projection to allow gradient flow during training, accepting the cost that the relaxation introduces a temperature hyperparameter τ that must be tuned to balance constraint satisfaction and gradient fidelity. We chose to train the latent dynamics f_ψ jointly with the decoder rather than freezing the encoder, because the pushforward constraint requires that the latent trajectory be both engine-valid and decodable; the trade-off is increased training complexity and the risk of the encoder collapsing to trivial solutions, which we mitigate by the reconstruction loss L_rec. This design is not a variant of ControlNet-style conditioning (Zhang et al.) or Diffusion Forcing, because those methods treat rule-adherence as an emergent property of conditioning, whereas our method makes it a structural constraint on the latent space via an explicit projection derived from the engine's executable rules. To address the load-bearing assumption that E_φ preserves the engine's state-space geometry, we add an inverse model I_ω and a cycle-consistency loss L_cyc that ties the latent projection Π_C to the exact state-space projection Π_S, making the assumption explicit and providing a training signal that encourages the latent space to align with the engine's invariants.

**Why it measures what we claim**
The projection operator Π_C measures rule-adherence because we assume that the engine's executable rule set can be compiled into a latent constraint set C that exactly characterizes the set of engine-valid states; this assumption fails when the encoder E_φ is not surjective onto the engine's state space (e.g., unseen states are mapped outside C), in which case Π_C reflects the encoder's approximation rather than the true rule set. The latent dynamics f_ψ measures the engine's transition kernel because we assume that the latent transition z_{t+1} = f_ψ(z_t, a_t) is a sufficient statistic for the engine's state transition given action a_t; this assumption fails when the encoder loses action-relevant information (e.g., fine-grained collision geometry), in which case f_ψ reflects a lossy abstraction. The reconstruction loss L_rec measures visual fidelity because we assume that the decoder D_θ can invert the encoder for in-distribution frames; this assumption fails for out-of-distribution states reached during rollouts, in which case L_rec reflects the decoder's generalization gap rather than true fidelity. The cycle-consistency loss L_cyc measures the alignment between latent projection and state-space projection because we assume that the inverse model I_ω accurately recovers engine state from latent; this assumption fails when I_ω is inaccurate for out-of-distribution latents, in which case L_cyc reflects the inverse model's error rather than true alignment.

## Contribution

["A rule-constrained transport framework that formalizes rule-adherence of a distilled video renderer as a pushforward constraint on the latent trajectory space, enforced by a projection operator compiled from the engine's executable rule set.", 'A differentiable soft-projection mechanism that allows end-to-end training of the renderer under hard rule constraints, with a temperature-controlled relaxation that balances constraint satisfaction and gradient fidelity.', 'An empirical demonstration that pushforward-constrained renderers produce fewer rule violations than conditioning-based baselines on a suite of game environments, without requiring new engine instrumentation.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Aligned low-poly/high-fidelity engine video pairs with action logs from a single engine with collision + resource counter rules | Provides ground-truth transitions and executable rules |
| Primary metric | Rule-violation rate over long autoregressive rollouts (length T=1000) | Directly measures the structural claim, not visuals |
| Secondary metric | Constraint set calibration: false-positive and false-negative rate of C against ground-truth engine validity on 10^4 held-out states | Validates that C faithfully represents engine rules |
| Secondary metric | Soft-projection deviation: constraint satisfaction rate of soft projection (τ=0.1) vs. hard projection on 10^4 validation latents | Quantifies train-test mismatch |
| Baseline | Action-token conditioned video diffusion | Tests whether soft conditioning suffices for rule-adherence |
| Baseline | Diffusion Forcing | Tests whether temporal consistency alone yields valid states |
| Baseline | ControlNet-style structural conditioning | Tests whether visual structure conditioning replaces explicit constraints |
| Ablation-of-ours | RCT with hard projection (no soft relaxation) | Isolates gradient-flow contribution of the relaxation |
| Ablation-of-ours | RCT without cycle-consistency loss (γ=0) | Isolates contribution of state-space alignment |

### Why this setup validates the claim

The central claim is that rule-adherence must be a structural constraint on the latent space, not an emergent property of conditioning. The evaluation design makes this falsifiable by pitting our explicit projection against three baselines that each represent a different mechanism for achieving rule-adherence: action-token conditioning (soft correlation), Diffusion Forcing (temporal consistency without rule structure), and ControlNet-style structural conditioning (visual structure without executable constraints). If our claim is correct, these baselines should degrade specifically on long rollouts where rule-violations compound, while our method stays bounded. The primary metric — rule-violation rate over long autoregressive rollouts — is chosen because it directly measures the property we claim to enforce, rather than a proxy like visual fidelity that could improve for reasons unrelated to rule-adherence. The hard-projection ablation isolates whether the differentiable relaxation is doing real work or whether constraint satisfaction is trivially achievable; if the ablation matches our method, the soft-projection design choice is unmotivated. The calibration metrics directly test the load-bearing assumption that C faithfully represents engine rules: a high false-positive rate would mean C admits invalid states, and a high false-negative rate would mean C rejects valid states, both undermining the structural claim. The cycle-consistency ablation tests whether the state-space alignment path is necessary for the latent projection to correspond to valid state projection. Together, this combination forms a falsifiable test: the claim predicts a specific divergence pattern between our method and each baseline, localized to the failure mode each baseline's mechanism cannot address.

### Expected outcome and causal chain

**vs. Action-token conditioned video diffusion** — On a case where the action sequence drives the agent into a collision-prone corridor and then requires a resource pickup under a strict legality predicate, the baseline produces frames where the agent visually clips through geometry or picks up an out-of-range resource because its mechanism only learns a soft correlation between action tokens and pixels and has no representation of the engine's invariant set. Our method instead projects the predicted latent onto the compiled constraint set C after each transition, because the engine's collision and resource predicates are compiled into C and enforced by Π_C, so we expect a noticeable gap in rule-violation rate concentrated on long rollouts with dense legality predicates, with parity on short open-terrain rollouts.

**vs. Diffusion Forcing** — On a case where the rollout must maintain a persistent resource counter across many frames while the agent moves, the baseline produces frames where the counter drifts or resets because its mechanism enforces temporal consistency of appearance but has no notion of a state variable that must remain within a valid range. Our method instead carries the resource state in the latent and snaps it to the valid range via Π_C at every step, because the resource invariant is part of the compiled constraint set, so we expect the violation gap to grow with rollout length and to be largest on counter-persistence subsets while remaining near zero on appearance-only subsets.

**vs. ControlNet-style structural conditioning** — On a case where fine-grained collision geometry must be respected at a corner the agent rounds at speed, the baseline produces frames that respect the visual layout but violate the exact collision predicate because its mechanism conditions on structural maps that encode appearance-level geometry rather than executable collision rules. Our method instead enforces the collision predicate directly in latent space, because Π_C projects onto the feasible set defined by the engine's executable collision relation, so we expect our gain to be concentrated on tight-geometry subsets and to shrink to parity on open-layout subsets where structural conditioning is already sufficient.

### What would falsify this idea

If our rule-violation advantage over the action-token and Diffusion Forcing baselines is uniform across rollout lengths and subsets rather than concentrated where the predicted failure modes (long rollouts, dense legality predicates, tight geometry) occur, the structural-constraint explanation is wrong. Likewise, if the hard-projection ablation matches our method on rule-violation rate, the differentiable-relaxation design choice is not doing the work we claim. Additionally, if the calibration metrics show high false-positive or false-negative rates of C against ground-truth engine validity, the load-bearing assumption that C faithfully represents engine rules is falsified.

## References

1. Matrix-Game 4.0: Harnessing Video Models for Real-Time Playable Game Worlds
2. StreamDiT: Real-Time Streaming Text-to-Video Generation
3. FrameDiffuser: G-Buffer-Conditioned Diffusion for Neural Forward Frame Rendering
4. Matrix-Game 2.0: An Open-Source, Real-Time, and Streaming Interactive World Model
5. Adding Conditional Control to Text-to-Image Diffusion Models
6. Diffusion Forcing: Next-token Prediction Meets Full-Sequence Diffusion
7. Diffusion Models Are Real-Time Game Engines
8. From Slow Bidirectional to Fast Autoregressive Video Diffusion Models
