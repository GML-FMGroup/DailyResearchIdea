# Identifiable Latent-Context Estimation for Egocentric Agents under Partial Observability

## Motivation

Across egocentric-vision work, the first-person view is treated as a sufficient statistic for the agent's situation: HASARD assumes egocentric frames suffice for safe RL decisions, COMBO conditions its world model on partial egocentric observations without establishing what latent world state they determine, and EgoUseBench itself observes that first-person evidence from wearables 'may provide incomplete or ambiguous evidence of ongoing activities' yet evaluates agents without any formal account of whether the latent context is recoverable. The structural root cause is that no existing branch instantiates an explicit generative model of how the wearer's viewpoint, occlusion, and task-relevant state jointly produce the observation, so identifiability of the latent context is neither enforced nor testable, and closed-loop agents can silently hold inconsistent world states under ambiguous evidence.

## Key Insight

If the egocentric observation is modeled as a viewpoint-gated, occlusion-masked rendering of a persistent latent scene state, then the wearer's pose trajectory supplies the interventions that break the viewpoint–content symmetry, making the latent context identifiable up to a known group action rather than only up to the arbitrary posterior of a passive encoder.

## Method

**What it is.** We introduce VIGIL (Viewpoint-Interventional Generative Identifiable Latent-context estimator), a variational inference module that takes the egocentric frame stream and the agent's own action/pose history and outputs a posterior over a task-relevant latent context z_t plus a viewpoint variable v_t, together with an explicit identifiability certificate (the residual gauge group under which z_t is still ambiguous). VIGIL is dropped into a closed-loop observation-action kernel: at each step the agent acts on the posterior mean, then updates on the next frame.

**How it works.**

```pseudocode
# Generative model (per timestep t)
z_t   ~ p(z_t | z_{t-1}, a_{t-1})        # task-relevant latent scene state, dim 32
v_t   ~ p(v_t | v_{t-1}, a_{t-1})        # wearer viewpoint (pose), dim 7 (pos+quat)
m_t   = Render(z_t, v_t)                  # differentiable occlusion mask; Render is a 4-layer MLP with hidden=256, GeLU, outputting a 64x64x3 RGB image and a 64x64 binary mask via sigmoid
x_t   ~ p(x_t | m_t ⊙ z_t, v_t)          # egocentric frame likelihood: Gaussian with fixed σ=0.1 per pixel

# Inference (amortized, single encoder with two heads)
q(z_t, v_t | x_{≤t}, a_{<t}) = q_v(v_t | x_t, v_t-1, a_{t-1}) · q_z(z_t | x_t, v_t, z_{t-1})
# q_v: 2-layer MLP (hidden=128, ReLU) outputting mean and diagonal log-variance for 7-dim pose
# q_z: 2-layer MLP (hidden=128, ReLU) outputting mean and diagonal log-variance for 32-dim z

# Training: ELBO + intervention-consistency term
L = -ELBO(x_{≤t}) + λ · E[ || q_z(z_t | x_t, v_t, z_{t-1}) - q_z(z_t | x_t, v'_t, z_{t-1}) ||² ]
    # v'_t is a counterfactual viewpoint obtained by applying a known
    # action-induced transform to v_t; λ = 0.5

# Closed-loop kernel at deployment
for t in stream:
    z_t, v_t, G_t = infer(x_t, a_{t-1}, z_{t-1})   # G_t = residual gauge group
    if ||G_t|| > τ_gauge:                          # τ_gauge = 0.1
        request_active_probe()                    # one deliberate head-motion action
    a_t = policy(z_t)
```

The intervention-consistency term is the load-bearing piece: it forces the z-head to be invariant to viewpoint changes that the generative model says should not affect scene content, and the gauge group G_t is computed as the set of (z, v) transforms under which the ELBO is unchanged — a quantity the encoder cannot fake because it is derived from the likelihood, not from the encoder's own confidence. **Load-bearing assumption (stated explicitly):** the gauge norm ||G_t|| is a calibrated signal of true latent-context ambiguity only if the rendering model Render(z_t, v_t) is correctly specified and the action-induced viewpoint transform is exactly known. When the renderer is misspecified (e.g., on novel materials or lighting), ||G_t|| reflects renderer error rather than true ambiguity, and the certificate becomes self-referential. To make this assumption testable, we add a calibration step: on a held-out set of 512 episodes with ground-truth latent states (available in simulation), we compute the correlation between ||G_t|| and the true posterior diameter (measured by sampling z from the true posterior and taking the maximum pairwise distance in z-space). We report this correlation and use it to set τ_gauge such that the probe trigger fires when true ambiguity exceeds a threshold, rather than when renderer error is high. If the correlation is below 0.5, the certificate is not calibrated and the method's ambiguity signal is not trustworthy; we report this as a negative result.

**Why this design.** We chose an explicit generative rendering model over a discriminative encoder because only a likelihood-based model yields a gauge group that is a property of the data-generating process rather than of the network, accepting the cost that rendering must be differentiable and approximate. We chose to condition q_z on the inferred viewpoint v_t rather than marginalizing it out because the viewpoint is the variable that breaks the symmetry, accepting the cost that viewpoint errors propagate into z. We chose a closed-loop active-probe trigger over passive inference because ambiguity that persists in the gauge norm cannot be resolved by more frames from the same pose, accepting the cost of one extra action per ambiguous step. A domain expert would not describe VIGIL as a variant of COMBO's compositional world model: COMBO learns a generative model to *simulate* futures for planning, whereas VIGIL's generative model exists to *certify* what the current observation determines, and the gauge group has no analogue in COMBO.

**Why it measures what we claim.** The gauge-norm ||G_t|| measures latent-context identifiability because it equals the diameter of the set of z values that achieve equal likelihood under the generative model, under the assumption that the rendering model is correctly specified; this assumption fails when the renderer is misspecified on novel materials or lighting, in which case ||G_t|| reflects renderer error rather than true ambiguity. The intervention-consistency residual measures viewpoint–content disentanglement because it is zero exactly when q_z is invariant to action-induced viewpoint transforms, under the assumption that the known action transform is the only viewpoint change between paired frames; this fails under unmodeled wearer motion, in which case the residual reflects motion-prediction error. The active-probe trigger measures whether the agent's world state is *recoverable* rather than merely *confident*, because it fires on likelihood-derived ambiguity, not encoder variance; this fails when the probe action itself is occluded, in which case the trigger reflects probe feasibility rather than ambiguity. The calibration correlation (between ||G_t|| and true posterior diameter) measures whether the gauge norm tracks true ambiguity because it compares the likelihood-derived signal to a ground-truth ambiguity measure in simulation, under the assumption that the simulation provides accurate ground-truth latents; this fails when the simulation's latent representation is itself ambiguous, in which case the correlation reflects simulation artifacts rather than real-world identifiability.

## Contribution

(1) VIGIL, a variational latent-context estimator whose generative model factorizes egocentric observations into a viewpoint variable and a persistent scene state, producing a likelihood-derived gauge group that certifies what the current first-person evidence determines. (2) An intervention-consistency training objective that ties the scene-state posterior to action-induced viewpoint transforms, yielding a design principle: identifiability of egocentric latent context requires the agent's own actions to appear as interventions in the training signal. (3) A closed-loop kernel with an active-probe trigger that converts residual gauge ambiguity into a deliberate exploratory head motion, evaluated on EgoUseBench's Online Ego Use setting.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | EgoUseBench Online Ego Use split | Evolving first-person context forces ambiguity resolution over time |
| Dataset (calibration) | Synthetic egocentric episodes from Habitat with ground-truth latents (512 episodes) | Provides true latent states to calibrate gauge norm |
| Primary metric | Task success under viewpoint ambiguity | Directly tests whether recoverable context beats confident context |
| Secondary metric | Gauge-norm calibration: Pearson correlation between ||G_t|| and true posterior diameter | Tests whether gauge norm tracks true ambiguity |
| Baseline | Passive amortized VAE (no gauge) | Tests whether likelihood-derived ambiguity adds anything |
| Baseline | COMBO-style compositional world model | Tests whether simulation-for-planning matches certification-for-identifiability |
| Baseline | ECBench-style LVLM embodied reasoner | Tests whether frozen foundation-model cognition suffices |
| Ablation-of-ours | VIGIL without intervention-consistency term | Isolates the disentanglement term as load-bearing |

### Why this setup validates the claim

The central claim is that a likelihood-derived gauge group lets an embodied agent distinguish *recoverable* latent context from merely *confident* context, and that acting on this distinction (via active probing) improves closed-loop task success precisely where viewpoint ambiguity is high. EgoUseBench's Online Ego Use split is the right testbed because it supplies continuously arriving first-person observations whose interpretation must be revised as the wearer's viewpoint shifts — exactly the regime where a discriminative encoder's confidence is decoupled from actual recoverability. The synthetic Habitat episodes with ground-truth latents allow us to directly test the load-bearing assumption: we compute the true posterior diameter by sampling from the ground-truth generative model and compare it to ||G_t||, reporting the Pearson correlation. This calibration test is essential because the gauge norm is only meaningful if it tracks true ambiguity rather than renderer error. The passive-VAE baseline tests the sub-claim that encoder variance is insufficient: it has the same architecture minus the gauge computation, so any gap isolates the value of the likelihood-derived certificate. The COMBO-style baseline tests the sub-claim that generative modeling for *simulation* is not the same as generative modeling for *certification*: it also builds a world model from egocentric views, so a gap shows the gauge group — not generative modeling per se — is doing the work. The ECBench-style LVLM baseline tests whether a strong frozen embodied reasoner already solves the ambiguity, which would render the module unnecessary. The ablation removes the intervention-consistency term, testing whether viewpoint–content disentanglement is what makes the gauge group meaningful rather than noisy. Task success under viewpoint ambiguity is the correct primary metric because it is the downstream consequence the claim predicts; a proxy like ELBO or gauge-norm alone could improve while behavior does not.

### Expected outcome and causal chain

**vs. Passive amortized VAE (no gauge)** — On a case where the wearer turns past a briefly occluded object, the passive VAE produces a confident posterior over scene content that silently absorbs the viewpoint change into z, because its encoder variance is trained to fit frames, not to certify what the frame determines. Our method instead computes the gauge group from the rendering likelihood, detects that z remains ambiguous under the current pose, and fires an active probe, so we expect a noticeable success gap concentrated on high-viewpoint-change episodes and near-parity on static-view episodes.

**vs. COMBO-style compositional world model** — On a case where two scene layouts render nearly identically from the current viewpoint, the COMBO-style model produces a plausible simulated future for one layout and plans into it, because its world model is optimized for rollout fidelity, not for exposing which latents the observation fails to pin down. Our method instead reads the ambiguity directly off the likelihood's level set and probes before committing, so we expect our advantage to appear as fewer confidently-wrong action sequences on ambiguous-view subsets, with parity where the view is already informative.

**vs. ECBench-style LVLM embodied reasoner** — On a case where the query concerns an object the wearer has not yet fixated, the LVLM baseline produces a fluent but ungrounded answer from prior knowledge, because its embodied cognition is evaluated offline and it has no mechanism to request more evidence. Our method instead triggers a deliberate head-motion probe when the gauge norm exceeds threshold, so we expect gains that grow with the fraction of ambiguous steps and shrink toward parity on episodes where the first frame is already sufficient.

**vs. VIGIL without intervention-consistency term (ablation)** — On a case where action-induced viewpoint transforms are large, the ablated model produces a gauge group inflated by viewpoint–content entanglement, because without the consistency term the z-head drifts with pose and the likelihood level set reflects that drift. Our method instead keeps q_z invariant to known viewpoint transforms, so we expect the ablation to show a higher false-probe rate and lower success specifically on high-motion episodes, confirming the term is load-bearing rather than decorative.

### What would falsify this idea

If the gauge-norm calibration correlation on synthetic episodes is below 0.5, the gauge norm does not track true ambiguity and the certificate is not meaningful. If our gain over the passive VAE is uniform across low- and high-viewpoint-ambiguity subsets rather than concentrated where the predicted failure mode occurs, the gauge group is not measuring recoverability. If the ablation without the intervention-consistency term matches full VIGIL on high-motion episodes, the disentanglement term is not doing the causal work we claim.

## References

1. EgoUseBench: Benchmarking Computer-Use Agents under Evolving First-Person Contexts
