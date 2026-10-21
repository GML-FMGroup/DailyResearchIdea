# State-Render Functional Certification via Contractive Residual Jacobian Surrogates

## Motivation

Across the interactive-world-model tree (Matrix-Game 4.0, Matrix-Game 2.0, GameNGen, FrameDiffuser), the renderer is treated as the sole visual authority: conditioning signals (low-poly engine proxies, G-buffers, action tokens, captions) are assumed trustworthy and pixel output is never checked against the engine's authoritative world state. The structural problem is that state-render functional dependence is unformalized — Matrix-Game 4.0 trains a distilled causal renderer on aligned low-poly/high-fidelity pairs but has no per-frame mechanism proving the rendered frame is a function of the engine state rather than of memorized visual priors, so any state-render mismatch is displaced into unverifiable visual plausibility instead of being detected and bounded.

## Key Insight

A renderer trained so that its state-conditional Jacobian surrogate is contractive yields a residual that is a certified per-frame upper bound on pixel change under any declared state perturbation, converting the previously informal 'renderer follows the engine' assumption into an auditable invariant checkable without a full engine re-render.

## Method

**What it is.** We introduce SRFC (State-Render Functional Certification), a training-and-audit layer that wraps an autoregressive game renderer R_θ. Inputs: authoritative engine world state s_t ∈ ℝ^d, rendered frame x_t = R_θ(s_t, h_t) with history h_t. Outputs: the frame plus a scalar certified residual ρ_t and a per-frame certificate (ρ_t, δ, L̂) declaring that any state perturbation ‖Δs‖ ≤ δ induces pixel change ‖Δx‖ ≤ ρ_t, where L̂ is a learned state-conditional Jacobian surrogate. **Load-bearing assumption (stated explicitly):** the learned surrogate J_φ, trained on K=8 random probes within Ball(δ) and regularized toward L_target=1.5, yields a valid upper bound on the true directional sensitivity of R_θ over the entire declared ball at audit time, including at non-smooth state-sensitivity regions such as occlusion boundaries. This assumption is not guaranteed by construction; it is empirically checked and calibrated as described in Phase 1b and Phase 3.

**How it works.**

```pseudocode
# --- Phase 1: Jacobian surrogate training (per training pair) ---
for (s_t, x_t) in aligned_pairs:                      # low-poly/high-fid pairs
    for k in 1..K:                                    # K=8 random probes
        Δs_k ~ Ball(δ)                                # δ=0.02 in normalized state units
        x_pert_k = R_θ(s_t + Δs_k, h_t)
        J_k = J_φ(s_t, h_t)                           # surrogate: d x d matrix
        L_k = ||J_k @ Δs_k|| / ||Δs_k||               # directional Lipschitz estimate
    L_hat = max_k L_k
    residual_target = ||x_pert_k - x_t||_max          # observed max pixel delta
    loss_jac = mean_k ( smoothmax(||x_pert_k - x_t|| - L_hat*||Δs_k||, margin=0) )^2
    loss_contract = (L_hat - L_target)^2              # L_target = 1.5 (contractive band)
    update φ by ∇(loss_jac + λ_c loss_contract)       # λ_c = 0.1

# --- Phase 1b: empirical calibration of the bound (held-out pairs) ---
# Calibration set: 512 held-out aligned pairs (disjoint from training pairs).
# For each calibration pair, draw M=64 perturbations from Ball(δ) and record
# the observed max pixel delta D_m = ||R_θ(s_t+Δs_m,h_t) - R_θ(s_t,h_t)||_max.
# Compute the empirical quantile q_α of the ratio D_m / (L_hat(s_t,h_t)*δ + ε_t)
# at confidence level α=0.95, and set the calibration multiplier γ = max(1, q_α).
# The certified residual at audit time becomes ρ_t = γ * (L_hat(s_t,h_t)*δ + ε_t).
# This makes ρ_t an empirical high-probability bound (calibrated on held-out data)
# rather than a purely analytic one, while retaining J_φ as the cheap pre-filter.

# --- Phase 2: renderer contractive-residual objective ---
for batch of (s_t, h_t):
    x_t = R_θ(s_t, h_t)
    ρ_t = γ * (L_hat(s_t, h_t) * δ + ε_t)             # ε_t = learned slack head, ≥0
    loss_render = perceptual(x_t, x*_t) + λ_r * relu(||x_pert - x_t||_max - ρ_t)
    update θ, ε by ∇ loss_render                      # λ_r = 1.0

# --- Phase 3: per-frame audit (inference) ---
for each frame t:
    s_t, x_t, h_t = engine.step(), renderer(s_t, h_t), history
    ρ_t, L_hat = certify(s_t, h_t)                    # read surrogate + slack head + γ
    if ρ_t > τ_cert:                                  # τ_cert = 2× training median of ρ_t
        flag_frame(t, ρ_t, L_hat)                     # auditable certificate emitted
    emit (x_t, ρ_t, L_hat) to log
```

**Why this design.** We chose a *learned state-conditional Jacobian surrogate* J_φ over finite-difference re-rendering at audit time because the latter requires the full engine re-render the motivation explicitly wants to avoid; the cost is that J_φ must be trained on K=8 perturbations per pair, adding ~8× forward passes during training only. We chose a *contractive residual objective* (pushing L̂ toward L_target=1.5) over an unconstrained fit because an unconstrained surrogate can trivially inflate its output to cover any observed pixel delta, making the certificate vacuous; the cost is that the contractive band may under-cover rare high-sensitivity states, which we surface as audit flags rather than silently absorb. We chose a *learned slack head ε_t ≥ 0* over a fixed margin because renderer stochasticity (denoising noise) is state-dependent, but we constrain ε_t to be monotone in ‖s_t‖ so it cannot be used to hide state-render mismatch; the cost is a small extra parameter head. We chose a *held-out calibration multiplier γ* over a purely analytic bound because the surrogate's validity as an upper bound is an empirical question (the load-bearing assumption above); γ is computed once on 512 held-out pairs with M=64 probes each and inflates ρ_t so that the empirical coverage on calibration data reaches the 95% quantile, at the cost of a slightly looser bound. A domain expert would NOT describe this as ControlNet-style conditioning (Adding Conditional Control) because ControlNet learns a spatial conditioning map with no functional-dependence guarantee — it never certifies that the output is a bounded function of the condition — whereas SRFC's load-bearing object is the certified residual ρ_t, which ControlNet does not produce.

**Why it measures what we claim.** The certified residual ρ_t = γ·(L̂(s_t,h_t)·δ + ε_t) measures *state-render functional dependence* because the contractive surrogate assumption plus the calibration multiplier γ make L̂ an empirical upper bound on the directional derivative of R_θ w.r.t. s over Ball(δ) at the 95% confidence level on held-out data; this assumption fails when the renderer has non-smooth state sensitivity (e.g., occlusion boundaries where the Jacobian jumps), in which case ρ_t reflects local smooth-region sensitivity inflated by γ rather than a global analytic bound. The directional Lipschitz estimate L_k = ‖J_k Δs_k‖/‖Δs_k‖ measures *local state sensitivity* because the probe Δs_k is drawn from the declared norm ball and the ratio is the exact operator norm along that direction under the linearization assumption; this assumption fails when Δs_k is large enough that second-order terms dominate, in which case L_k reflects a secant slope rather than a derivative. The audit flag threshold τ_cert measures *certification failure* because the training-median residual is a calibrated scale under the assumption that training pairs are in-distribution; this assumption fails when the engine enters out-of-distribution states (novel game mechanics), in which case τ_cert reflects distribution shift rather than renderer infidelity. The slack head ε_t measures *renderer stochasticity* because it is trained to absorb only the residual variance not explained by L̂·δ under the assumption that stochasticity is state-conditional and smooth; this assumption fails at high-frequency state transitions, in which case ε_t absorbs genuine state-render mismatch.

## Contribution

(1) SRFC, a state-render functional certification layer that produces a per-frame certified residual ρ_t from a learned state-conditional Jacobian surrogate, making the renderer's functional dependence on engine world state auditable without a full engine re-render. (2) A contractive-residual training objective that turns the residual into a declared upper bound on pixel change within a norm ball rather than a heuristic, together with a learned state-conditional slack head that separates renderer stochasticity from state-render mismatch. (3) An audit protocol that emits per-frame certificates (ρ_t, δ, L̂) and flags frames whose residual exceeds a calibrated threshold, enabling downstream detection of state-render divergence in interactive world models.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|-----------------------|
| Dataset | Matrix-Game aligned low-poly/high-fid pairs | Provides ground-truth state-render pairs for training and audit |
| Primary metric | Certified residual coverage rate | Directly measures functional dependence guarantee |
| Baseline | Matrix-Game 4.0 (no certification) | Tests whether certified residuals detect failures |
| Baseline | StreamDiT (no state conditioning) | Tests whether state-render functional dependence is necessary |
| Baseline | FrameDiffuser (G-buffer conditioning) | Tests whether spatial conditioning suffices without functional guarantee |
| Ablation-of-ours | SRFC with fixed slack ε=0 | Tests whether learned slack head is necessary for coverage |
| Ablation-of-ours | SRFC without calibration multiplier (γ=1) | Tests whether held-out calibration is necessary for coverage |
| Ablation-of-ours | SRFC with K=1 probe (vs. K=8) | Tests sensitivity of bound to number of training probes |
| Ablation-of-ours | SRFC with δ=0.01 and δ=0.04 | Tests sensitivity of bound to declared perturbation radius |

### Why this setup validates the claim

This setup creates a falsifiable test of SRFC's central claim: that the certified residual ρ_t provides a valid upper bound on pixel change under state perturbations. The primary metric, certified residual coverage rate, directly operationalizes this claim by measuring the fraction of frames where the observed pixel change under a state perturbation falls within the certified bound. The Matrix-Game dataset is essential because it provides aligned low-poly (state) and high-fidelity (render) pairs, enabling both training of the Jacobian surrogate and ground-truth evaluation of the bound. The baselines each test a different sub-claim: Matrix-Game 4.0 lacks any certification mechanism, so it tests whether certification is necessary to detect state-render mismatch; StreamDiT lacks persistent state, so it tests whether state conditioning is required for functional dependence; FrameDiffuser uses G-buffer conditioning, so it tests whether spatial conditioning alone can provide a functional guarantee. The ablation with fixed slack ε=0 tests whether the learned slack head is necessary to achieve coverage, isolating the contribution of state-dependent stochasticity modeling. The ablation without calibration multiplier (γ=1) tests whether the held-out calibration step is necessary to achieve the claimed coverage, directly probing the load-bearing assumption that L̂ is a valid upper bound. The K=1 and δ-variation ablations test the sensitivity of the bound to the number of training probes and the declared perturbation radius, providing empirical evidence on how the surrogate's validity degrades with fewer probes or larger balls. If SRFC's coverage rate is significantly higher than all baselines and the ablations, the claim is supported; if not, the claim is falsified.

### Expected outcome and causal chain

**vs. Matrix-Game 4.0** — On a case where the engine state changes rapidly (e.g., a character moving behind an occluder), Matrix-Game 4.0 produces frames that may deviate from the true render without any warning because it lacks a certification mechanism to detect state-render mismatch. Our method instead computes a certified residual ρ_t that bounds the expected pixel change; when the actual change exceeds ρ_t, the frame is flagged. We expect SRFC to achieve high coverage (e.g., >90% of frames within bound) while Matrix-Game 4.0 has no notion of coverage, so the observable signal is a large gap in coverage rate (SRFC near 1.0 vs. baseline undefined/0).

**vs. StreamDiT** — On a case where the game state includes persistent objects (e.g., a door that remains open), StreamDiT generates frames without state conditioning, so it may produce inconsistent renders (e.g., door flickering) because it lacks a functional dependence on state. Our method conditions on state and certifies the residual, ensuring that small state changes produce bounded pixel changes. We expect SRFC to show high coverage on such persistent-state scenarios, while StreamDiT's coverage (if adapted) would be low, yielding a noticeable gap on subsets with persistent state but parity on stateless subsets.

**vs. FrameDiffuser** — On a case where the state perturbation is small but causes a large pixel change (e.g., a slight camera angle shift revealing a new object), FrameDiffuser's G-buffer conditioning may not capture the functional dependence, leading to uncertified large deviations because it learns a spatial map without a Lipschitz guarantee. Our method's Jacobian surrogate explicitly bounds the sensitivity, so it would flag such frames. We expect SRFC to maintain coverage on these high-sensitivity cases, while FrameDiffuser's coverage drops, resulting in a gap concentrated on high-sensitivity subsets.

### What would falsify this idea

If SRFC's certified residual coverage rate is uniform across all subsets (including those where baselines fail) rather than concentrated on subsets with high state sensitivity or persistent state, then the central claim that ρ_t measures state-render functional dependence is wrong. Additionally, if the ablation with fixed slack ε=0 achieves the same coverage as the full model, then the learned slack head is unnecessary, undermining the claim that state-dependent stochasticity is captured. Furthermore, if the ablation without calibration multiplier (γ=1) achieves the same coverage as the full model, then the held-out calibration step is unnecessary, and the load-bearing assumption that L̂ alone is a valid upper bound is not empirically supported.

## References

1. Matrix-Game 4.0: Harnessing Video Models for Real-Time Playable Game Worlds
2. StreamDiT: Real-Time Streaming Text-to-Video Generation
3. FrameDiffuser: G-Buffer-Conditioned Diffusion for Neural Forward Frame Rendering
4. Matrix-Game 2.0: An Open-Source, Real-Time, and Streaming Interactive World Model
5. Adding Conditional Control to Text-to-Image Diffusion Models
6. Diffusion Forcing: Next-token Prediction Meets Full-Sequence Diffusion
7. Diffusion Models Are Real-Time Game Engines
8. From Slow Bidirectional to Fast Autoregressive Video Diffusion Models
