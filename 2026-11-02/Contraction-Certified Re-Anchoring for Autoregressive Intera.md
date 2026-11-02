# Contraction-Certified Re-Anchoring for Autoregressive Interactive Video Latents

## Motivation

Across both the interactive-video-generation lineage (Yan's KV-cache shift-window autoregressive denoising) and the fast-autoregressive-diffusion lineage (the causal DMD student with KV caching), the latent transition is an open-loop map from (past latents, action) to next latent, with no learned recoverability signal and no invariant tying drift to a correction operator; Yan's compressed 3D-VAE manifold and StreamEval's explicit state-preservation metrics supply the low-dimensional latent and the state-change signal, but neither branch detects when the trajectory has left the physically plausible manifold nor bounds how many steps it takes to re-enter. The structural consequence is that a single bad action-conditioned transition silently compounds — StreamEval's low Following/Edit-Success scores are the observable symptom of an uncertified open-loop drift, not a rendering-quality problem.

## Key Insight

If the action-conditioned latent transition is constrained to be a contraction on the compressed 3D-VAE manifold, then its unique fixed point defines an action-consistent invariant set, and the contraction residual measured against that fixed point is a calibrated recoverability certificate that bounds re-entry to a finite number of steps.

## Method

**What it is.** CARA (Contraction-Anchored Re-anchoring) augments an autoregressive interactive video denoiser with a per-action, empirically verified local contraction certificate on the latent transition and a lightweight residual head that predicts, from (latent, action), how far the current step is from the contraction fixed point; when the predicted residual exceeds a certified threshold, the loop projects the latent back onto the action-consistent invariant set before continuing. The load-bearing assumption is stated explicitly: **the action-conditioned latent transition map T(·, a) induced by the pretrained autoregressive denoiser is locally contractive (local modulus kappa_a < 1) on the compressed 3D-VAE latent manifold for the action families we certify; this is not assumed globally but measured per action on held-out clean trajectories, and actions with kappa_a >= 1 are flagged and handled by a damped fallback rather than the Banach projection.**

**How it works.**

```pseudocode
# Precompute (offline, once per action-conditioned 3D-VAE latent space)
#   anchor a(s) = action-embedding of scripted state-change signal from StreamEval-style probes
#   per-action local modulus kappa_a: estimate local Jacobian spectral radius of T(·,a)
#     on held-out clean trajectories via finite differences around 3 Picard iterates
#     (central differences, step h=1e-3 in latent units, 16 random probe directions per action)
#   certified set A_cert = { a : kappa_a < 1 - 1e-2 }   # margin 1e-2 for estimation noise
#   for a in A_cert: tau_cert(a) = kappa_a * r_prev + eps, eps = 1e-4
#   for a not in A_cert: flag as expansive; use damped step alpha_a = 0.5 / kappa_a

for each interaction step t:
    z_prev = latent at t-1                      # compressed 3D-VAE latent (Yan)
    a_t    = action token at t
    z_hat  = Denoiser(z_prev, a_t)              # KV-cache shift-window step
    # (1) contraction constraint on the transition map
    z_next = z_hat
    residual_hat = ResidualHead(z_prev, a_t)    # predicts ||T(z_next)-z_next||
    # (2) certificate check against fixed point of T
    fp = FixedPoint(T, a_t)                     # solve z = T(z,a_t) by 3 Picard iters
    r  = || z_next - fp ||                      # measured contraction residual
    if a_t in A_cert:
        if r > tau_cert(a_t):
            # (3) re-anchor: project onto action-consistent invariant set
            z_next = Project(z_next, fp, kappa_a)   # z_next <- fp + kappa_a*(z_next-fp)
            re_entry_steps += 1
    else:
        # expansive action: damped trust-region step, not Banach projection
        z_next = fp + alpha_a * (z_next - fp)       # alpha_a = 0.5 / kappa_a < 1
        re_entry_steps += 1
    z_prev = z_next
    emit frame = Decoder(z_next)

# ResidualHead training: contrastive probe
#   positive: (z_prev, a_t) from clean trajectory, label = measured r
#   negative: (z_prev, shuffle(a_t)) from action-shuffled counterfactual, label = r_shuffled
#   loss = L1(residual_hat, r) + margin * max(0, r_shuffled - r + delta)
#   margin = 0.1, delta = 0.05, batch size 64, Adam lr 1e-4, 10k steps
```

**Why this design.** We chose a *per-action, empirically verified local contraction certificate on the transition map* over a post-hoc discriminator because, where the measured local modulus kappa_a < 1, Banach's fixed-point theorem gives a unique invariant set per action, so the residual is a certificate rather than a heuristic score; we accept that the modulus must be measured per action family and that actions with kappa_a >= 1 are flagged and handled by a damped fallback rather than the Banach projection. We chose a *Picard-iteration fixed-point solve* over a learned anchor network because it keeps the anchor inside the same latent manifold the denoiser operates on, accepting three extra denoiser-equivalent evaluations per re-anchoring step. We chose a *contrastive residual head trained against action-shuffled counterfactuals* over a log-probability gate because log-prob calibration for recoverability is not established (Kadavath/Tian negative results), whereas the action-shuffle contrast directly probes whether the residual is action-conditioned; the cost is that the head is only as good as the shuffle distribution. A domain expert would not describe CARA as a variant of Toolformer-style retrieval or of a process-reward reranker, because the re-anchoring operator is a closed-form projection derived from the contraction fixed point, not a selection among candidates.

**Why it measures what we claim.** The contraction residual `r = ||z_next - fp||` measures *recoverability* because the per-action local contraction assumption (measured kappa_a < 1 on the action-conditioned transition) makes r an upper bound on the distance to the invariant set and hence on re-entry steps; this assumption fails when the measured local modulus is >= 1 for some action (e.g., fast camera pans), in which case r reflects distance to a spurious fixed point rather than recoverability, and the method falls back to a damped step. The residual head's contrastive margin measures *action-conditioning* because the action-shuffled counterfactual is a valid intervention on the action variable under the assumption that the action embedding is the only action-dependent input to the transition; this assumption fails when visual context leaks action information, in which case the margin reflects context sensitivity rather than action-conditioning. The re-entry step count measures *bounded re-entry* because each projection is a kappa_a-contraction toward fp, so the count is monotone in the initial residual under the assumption A that the Picard fixed point equals the true action-consistent invariant set and stays stable across the re-entry window; this assumption fails when the action changes mid-recovery or kappa_a >= 1, in which case the count reflects action-switching frequency or fallback damping rather than contraction quality.

## Contribution

(1) A contraction-anchored re-anchoring operator for autoregressive interactive video latent loops, with a per-step contraction constraint and a closed-form projection onto the action-consistent invariant set. (2) A contrastive residual head trained against action-shuffled counterfactuals that yields an action-conditioned recoverability certificate, together with the design principle that the certificate threshold should be tied to the fitted contraction modulus rather than tuned as a free hyperparameter. (3) A scripted diagnostic suite built on StreamEval-style state-change probes that reports re-entry step counts and residual trajectories alongside standard state-preservation metrics.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | StreamEval scripted interactive video suite | Scripted actions give ground-truth state changes for re-entry scoring. |
| Dataset (non-scripted) | Real user-driven action sequences (e.g., a held-out set of recorded interactive sessions) | Tests whether kappa_a < 1 holds for real actions. |
| Primary metric | Re-entry step count to action-consistent latent | Directly counts contraction recovery, the claimed mechanism. |
| Secondary metric | Fraction of action families with measured kappa_a < 1 | Conditions the certificate claim on a measured quantity. |
| Baseline | StreamEval best-performing autoregressive model | Tests whether contraction constraint beats unconstrained denoising. |
| Baseline | Discriminator-gated post-hoc filtering | Tests whether Banach certificate beats heuristic gating. |
| Baseline | Learned anchor network replacement | Tests whether Picard fixed-point solve beats learned anchoring. |
| Baseline | Banach projection on a random non-learned transition map | Isolates the action-conditioned contraction as the novel ingredient. |
| Ablation-of-ours | CARA without re-anchoring projection | Isolates the projection step from residual detection. |
| Ablation-of-ours | CARA with global fitted kappa (no per-action verification) | Tests whether per-action local verification matters. |

### Why this setup validates the claim

The central claim is that a contraction-anchored re-anchoring loop bounds re-entry steps to the action-consistent invariant set, and that this bound is what makes interactive rollouts recoverable. StreamEval's scripted suite is the right dataset because each case pairs an action token with a known state-change signal, so the invariant set per action is identifiable and re-entry steps can be counted without ambiguity. The non-scripted real user-driven dataset is included because the certificate claim is conditioned on a measured per-action local modulus kappa_a < 1; reporting the fraction of real action families that satisfy this condition tests whether the assumption holds outside scripted probes. The primary metric—re-entry step count—is the direct observable of the contraction mechanism: if CARA works, the count should be small and concentrated on cases where the unconstrained transition residual exceeds the certified threshold. The unconstrained autoregressive baseline tests the sub-claim that contraction is necessary; the discriminator-gated baseline tests the sub-claim that a certificate beats a heuristic score; the learned-anchor baseline tests the sub-claim that Picard iteration on the denoiser manifold beats a separate anchor network; the random-transition baseline tests the sub-claim that the gain comes from the action-conditioned contraction rather than from the projection operator itself. The ablation without projection isolates the projection operator, so any gain can be attributed to re-anchoring rather than to residual detection alone; the ablation with global fitted kappa tests whether per-action local verification is necessary. Together these form a falsifiable test: the claim predicts a specific ordering (CARA < ablation < discriminator-gated < unconstrained) on the subset where contraction fails, with parity elsewhere, and it predicts that CARA's gain vanishes on the random-transition baseline.

### Expected outcome and causal chain

**vs. StreamEval best-performing autoregressive model** — On a case where the action triggers a fast camera pan, the baseline produces latent drift because its transition map is expansive for that action and has no fixed-point constraint, so small denoiser errors compound across steps. Our method instead measures the local modulus kappa_a for that action; if kappa_a < 1 it projects the latent back onto the action-consistent invariant set whenever the measured residual exceeds the certified threshold, because the contraction modulus bounds the distance to the fixed point; if kappa_a >= 1 it applies the damped fallback step alpha_a = 0.5 / kappa_a. We expect a noticeable gap in re-entry step count on the fast-motion subset but parity on slow-motion cases where the baseline transition is already contractive.

**vs. Discriminator-gated post-hoc filtering** — On a case where the action is rare and the discriminator has not seen it, the baseline produces a heuristic gate score that is miscalibrated, so it either accepts drifted latents or rejects valid ones, and re-entry steps are not bounded because the gate has no fixed-point target. Our method instead uses the Banach residual as a certificate against the action-specific fixed point when the measured kappa_a < 1, so even for rare actions the projection direction is well-defined by the Picard solve; for actions with kappa_a >= 1 the damped fallback provides a bounded step. We expect a gap concentrated on rare-action cases, with the discriminator baseline showing high variance in re-entry counts across actions.

**vs. Learned anchor network replacement** — On a case where the action embedding is ambiguous, the baseline produces an anchor that lies off the denoiser's latent manifold, so projecting toward it introduces artifacts and re-entry steps do not decrease monotonically. Our method instead keeps the anchor inside the same latent manifold via three Picard iterations of the denoiser, so the projection direction is consistent with the transition map. We expect a smaller re-entry count and fewer decoder artifacts on ambiguous-action cases, with the gap shrinking on unambiguous actions where both anchors agree.

**vs. Banach projection on a random non-learned transition map** — On the same fast-motion cases, the random-transition baseline applies the identical projection operator but toward a fixed point of an arbitrary map, so the projection direction is uninformative and re-entry steps do not decrease. Our method instead uses the action-conditioned denoiser transition, whose measured local modulus kappa_a < 1 on certified actions, so the fixed point is the action-consistent invariant set. We expect CARA's gain over the unconstrained baseline to vanish for the random-transition baseline, isolating the action-conditioned contraction as the novel ingredient.

### What would falsify this idea

If the gain in re-entry step count is uniform across all subsets rather than concentrated on cases where the unconstrained transition residual exceeds the certified threshold, then the contraction constraint is not the causal mechanism and the central claim is wrong. Likewise, if the ablation without projection matches CARA's re-entry counts, the projection operator is doing no work and the certificate is decorative; and if the fraction of real action families with measured kappa_a < 1 is near zero, the certificate claim does not extend beyond the scripted suite.

## References

1. Yan: Foundational Interactive Video Generation
2. HunyuanVideo: A Systematic Framework For Large Video Generative Models
3. From Slow Bidirectional to Fast Autoregressive Video Diffusion Models
4. StreamEval: Evaluating Interactive Streaming Video Generation Models
