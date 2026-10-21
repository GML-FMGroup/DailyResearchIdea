# State-Witness Chunk Consistency for Drift-Identifiable Autoregressive Game Renderers

## Motivation

Chunked autoregressive renderers such as Matrix-Game 4.0 and Matrix-Game 2.0 distill a short-horizon teacher into a causal student that emits fixed four-frame chunks with one or two denoising steps, and they implicitly assume that short-horizon distillation transfers to long horizons and that the conditioning signal stays reliable. Because no quantity ties the renderer's per-chunk output back to the engine's authoritative state transition, per-chunk drift is unobservable and compounds silently across chunks — exactly the failure mode FrameDiffuser acknowledges as autoregressive error accumulation but does not bound. The field's convergent gap is thus structural: fixed-chunk few-step generation has no identifiability anchor against ground-truth engine state, so long-horizon degradation cannot be detected or corrected at the point where it is injected.

## Key Insight

If the renderer is forced to emit a low-dimensional state-witness at every chunk boundary whose evolution is a deterministic function of the engine's authoritative transition, then per-chunk drift becomes an observable residual with a known zero target rather than an unobservable accumulation, because the witness channel is structurally tied to the engine state rather than learned freely from pixels.

## Method

**What it is.** Drift-Identifiable Chunk Consistency (DICC) augments a chunked causal renderer by requiring it to emit, at every four-frame chunk boundary, a low-dimensional state-witness `w_k ∈ R^d` (d=16: player position delta (3), velocity (3), facing angle (2), and a 4-bit interaction flag, all read directly from the lightweight engine). The renderer's conditioning for chunk `k+1` is tied to the witness residual `r_k = w_k - T(s_k, a_k)` where `T` is the engine's deterministic state transition and `s_k` the engine state at boundary `k`. Inputs: engine state `s_k`, action `a_k`, previous frames. Outputs: next four frames plus witness `w_k`, with residual `r_k` used online to correct conditioning.

**Load-bearing assumption (stated explicitly).** The method assumes that the emitted witness `w_hat` is a faithful readout of the renderer's internal latent state, so that the residual `r_hat = w_hat - T(phi(s_k), a_k)` measures the renderer's state drift rather than merely the accuracy of a supervised auxiliary head that can be regressed to zero while the visual/latent state drifts independently. This assumption is load-bearing because the entire drift-identifiability claim rests on `r_hat` being a probe of the renderer's internal state, not a copy of the input. It fails if the witness head can achieve near-zero witness error by reading the ground-truth-derived conditioning context (ctx_k contains past frames and engine state s_k) without the generative latent being state-correct. To enforce the assumption, the witness is made a bottleneck the frames must pass through: frames_hat are generated conditioned on `w_hat` (specifically, `w_hat` is concatenated to the conditioning vector at every denoising step), and a counterfactual consistency term is added at training time: perturb `w_hat` by a random vector `δ ~ N(0, 0.1^2 I)` and require `frames_hat(w_hat + δ)` to differ from `frames_hat(w_hat)` in a way that predicts the perturbation (measured by a small auxiliary predictor of `δ` from the frame difference, trained with MSE). Additionally, a held-out probe (linear regression) predicts `w` from the renderer's internal latent alone (the final hidden state of the denoising network) with no access to ctx_k's engine-state channel; the probe's R^2 is reported as a diagnostic. If the witness cannot be recovered from the latent without the engine-state input (probe R^2 < 0.5), the residual is not measuring renderer drift and the method must be redesigned around a latent-anchored state code.

**How it works.**
```pseudocode
# Training, per sampled chunk k of a recorded engine session
s_k, a_k, s_{k+1} = engine_log[k]          # authoritative transition
target_w = phi(s_{k+1})                     # phi: fixed state->witness map, d=16
frames_hat, w_hat, latent = renderer(ctx_k) # ctx_k = past frames + caption + memory ref
r_hat = w_hat - T(phi(s_k), a_k)            # witness residual, target = 0
loss_visual = L_diffusion(frames_hat, frames_k)          # base distillation loss
loss_witness = || w_hat - target_w ||^2                  # witness regression
loss_consistency = || r_hat ||^2 + lambda * D(ctx_{k+1}, cond(ctx_k, w_hat))
#   D = distance between actual next-chunk conditioning and the one implied by w_hat
#   D is implemented as L2 distance between conditioning vectors after a 2-layer MLP encoder (hidden=256, GeLU, output dim=128, LayerNorm on output).
#   lambda = 0.5, annealed to 0.1 over training
# Counterfactual consistency (enforces witness bottleneck):
delta = randn_like(w_hat) * 0.1
frames_hat_pert = renderer(ctx_k, w_hat + delta)  # same latent path, perturbed witness
loss_cf = MSE( predictor(frames_hat_pert - frames_hat), delta )  # predictor: 2-layer MLP (hidden=128, GeLU)
loss = loss_visual + alpha*loss_witness + beta*loss_consistency + eta*loss_cf   # alpha=1.0, beta=0.3, eta=0.2

# Inference, online at each chunk boundary
frames_hat, w_hat, latent = renderer(ctx_k)
r_hat = w_hat - T(phi(s_k), a_k)            # residual vs engine transition
ctx_{k+1} = cond(ctx_k, w_hat - gamma*r_hat)  # gamma=0.7, correct conditioning online
```

**Why this design.** We chose a *fixed* state-to-witness map `phi` read from the engine over a learned witness head because a learned head can absorb drift into its own parameters and destroy identifiability, accepting the cost that the witness dimension is capped at engine-observable quantities (d=16). We chose to tie the consistency loss to the *next chunk's conditioning* rather than only regressing the witness, because a witness that is accurate but ignored by the renderer gives no correction signal; the trade-off is that the conditioning-distance term `D` requires a differentiable conditioning encoder, which adds a small auxiliary module (2-layer MLP, hidden=256, output dim=128). We chose online residual correction with gain `gamma=0.7` over retraining or test-time optimization because the correction must be cheap enough for 60 FPS gameplay, accepting that a single scalar gain cannot fully absorb structured drift — the consistency loss is what keeps the residual small enough for the gain to suffice. We added the counterfactual consistency term and the latent probe to enforce that the witness is a bottleneck the frames must pass through, not a parallel head, because otherwise the residual could be zero while the latent drifts. A domain expert would not describe this as Diffusion Forcing or GameNGen with an extra head: those methods have no engine-authoritative transition to define a zero-target residual, so their error signal is purely visual and unidentifiable against ground truth.

**Why it measures what we claim.** The witness residual `r_hat = w_hat - T(phi(s_k), a_k)` measures *per-chunk drift against engine-authoritative state* because the assumption that `T∘phi` is a deterministic sufficient statistic of the engine transition makes a zero residual equivalent to the renderer's internal state matching the engine; this assumption fails when the engine transition is stochastic (e.g., physics with random seeds), in which case `r_hat` reflects engine nondeterminism rather than renderer drift. The consistency term `D(ctx_{k+1}, cond(ctx_k, w_hat))` measures whether the witness is *causally used* by the renderer because the assumption that the conditioning encoder is injective on the relevant state variables makes conditioning distance equivalent to state mismatch; this assumption fails when the caption dominates the conditioning, in which case `D` reflects caption variation rather than witness influence. The visual loss `L_diffusion` measures *perceptual fidelity* under the assumption that the distilled teacher's distribution is a valid target, which fails when the teacher itself drifts, in which case it reflects teacher bias. The counterfactual consistency loss `loss_cf` measures whether the witness is a *bottleneck* the frames must pass through because the assumption that perturbing `w_hat` produces a predictable change in `frames_hat` (via the auxiliary predictor) holds when the renderer uses `w_hat` to generate frames; this assumption fails when the renderer ignores `w_hat` (e.g., if `w_hat` is concatenated but the network learns to ignore it), in which case `loss_cf` reflects noise rather than causal use. The latent probe measures whether the witness is *recoverable from the internal latent alone* because the assumption that the latent encodes the state variables that `w_hat` summarizes holds when the probe achieves high R^2; this assumption fails when the latent does not encode those variables (e.g., if the witness head reads from ctx_k directly), in which case the probe R^2 is low and the residual is not a drift signal.

## Contribution

(1) A drift-identifiability invariant for chunked autoregressive video renderers: a state-witness channel structurally tied to the engine's deterministic transition, making per-chunk drift an observable zero-target residual. (2) A chunk-boundary consistency loss that couples the witness residual to the next chunk's conditioning, forcing online correction rather than compounding. (3) An online residual-correction rule at inference that uses the engine transition to adjust conditioning at each four-frame boundary without retraining.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Engine-logged gameplay sessions with authoritative state transitions (e.g., 10 hours of recorded sessions from a single open engine, split into 8h train / 1h val / 1h test) | Gives ground-truth `T∘phi` for zero-target residual |
| Primary metric | Per-chunk witness residual `||r_hat||` vs rollout horizon (measured every 4 frames, averaged over 100 rollouts of length 256 chunks) | Directly measures drift against engine state |
| Baseline | Matrix-Game 4.0 distilled chunked renderer | No engine-authoritative residual, visual-only error |
| Baseline | StreamDiT-style multistep distilled streaming generator | Real-time streaming without persistent state witness |
| Baseline | FrameDiffuser G-buffer-conditioned renderer | Strong per-frame conditioning, no chunk drift correction |
| Baseline | Direct state conditioning: renderer conditioned on engine state `s_k` without witness/residual machinery | Isolates what witness-residual adds beyond plain state conditioning |
| Ablation-of-ours | DICC with `gamma=0` (witness trained, correction disabled) | Isolates whether residual correction drives gains |
| Ablation-of-ours | DICC with counterfactual loss `eta=0` (witness bottleneck not enforced) | Tests whether bottleneck is necessary for residual to track latent drift |
| Ablation-of-ours | DICC with learned transition model `T_hat` substituted for `T` at inference (trained on engine logs) | Tests graceful degradation without live engine |
| Diagnostic | Conditioning-encoder state-recovery accuracy: linear probe from conditioning vector to `s_k` (R^2) | Tests injectivity assumption of `D` |
| Diagnostic | Latent probe: linear regression from renderer's final hidden state to `w` (R^2) | Tests whether witness is recoverable from latent alone |

### Why this setup validates the claim

The design turns the central claim — that a zero-target witness residual tied to engine transitions yields identifiable, correctable drift — into a measurable prediction: residual should stay bounded as rollout length grows, and visual fidelity should degrade more slowly than baselines. The engine-logged dataset is the linchpin, because only it supplies the authoritative `T∘phi` that makes `r_hat` a genuine drift signal rather than a self-referential one; baselines that lack this signal can only be scored on the same metric post-hoc, exposing whether their drift is larger. The `gamma=0` ablation is the causal test: if the witness is accurate but unused, residual stays low while visual fidelity matches the uncorrected baseline, proving the consistency term alone is insufficient. The direct state conditioning baseline isolates the contribution of the witness-residual formulation beyond simply conditioning on engine state. The counterfactual loss ablation tests whether the bottleneck is necessary for the residual to track latent drift. The `T_hat` ablation tests graceful degradation when authoritative state is unavailable. The conditioning-encoder state-recovery diagnostic tests the injectivity assumption of `D`, and the latent probe tests whether the witness is recoverable from the latent alone. The per-chunk residual metric is chosen because it localizes drift in time, so a real correction effect should appear as a widening residual gap concentrated at longer horizons, not as a uniform offset.

### Expected outcome and causal chain

**vs. Matrix-Game 4.0** — On a case where the player executes a sustained turn-and-move sequence over many chunks, the baseline produces slow positional and facing drift because its only error signal is visual, which cannot distinguish a plausible-looking frame from a state-correct one. Our method instead reads the engine transition at each boundary and corrects conditioning by `gamma*r_hat`, because the witness residual directly encodes the state mismatch. We expect a residual gap that stays near parity for the first few chunks but widens noticeably on long-horizon subsets, with visual fidelity following the same pattern.

**vs. StreamDiT** — On a case where a moving-buffer streaming generator must carry state across segment boundaries under multistep distillation, the baseline produces accumulated segment-boundary error because distillation compresses per-segment dynamics without any authoritative transition to anchor them. Our method instead ties each boundary's conditioning to the engine-implied witness, because the consistency loss forces the renderer to use `w_hat` rather than absorb drift silently. We expect the baseline's boundary errors to compound across segments while ours stay roughly flat, visible as a growing residual gap at later chunk indices.

**vs. FrameDiffuser** — On a case where G-buffer conditioning is strong per frame but the underlying state is unobserved, the baseline produces temporally consistent-looking frames that nonetheless diverge from true engine state because per-frame conditioning has no cross-chunk state anchor. Our method instead corrects the next chunk's conditioning using the residual against `T(phi(s_k), a_k)`, because the witness supplies exactly the missing state anchor. We expect parity on short clips where per-frame conditioning dominates, but a clear residual gap on longer rollouts where state accumulation matters.

**vs. Direct state conditioning** — On a case where the renderer is conditioned directly on engine state `s_k` without any witness or residual, the baseline can still drift because the conditioning signal is not tied to a zero-target residual and the renderer may ignore or mis-scale the state input. Our method instead uses the witness residual to correct conditioning online, because the residual provides a direct error signal against the engine transition. We expect our method to show lower residual and better visual fidelity on long horizons, with the gap widening as drift accumulates.

### What would falsify this idea

If the `gamma=0` ablation matches full DICC on residual and fidelity across all horizons, the residual correction is not doing the claimed work. If the residual gap versus baselines is uniform across short and long horizons rather than concentrated at long horizons, the effect is not drift correction but a generic offset. If the latent probe R^2 is below 0.5, the witness is not recoverable from the latent alone and the residual does not measure renderer drift.

## References

1. Matrix-Game 4.0: Harnessing Video Models for Real-Time Playable Game Worlds
2. StreamDiT: Real-Time Streaming Text-to-Video Generation
3. FrameDiffuser: G-Buffer-Conditioned Diffusion for Neural Forward Frame Rendering
4. Matrix-Game 2.0: An Open-Source, Real-Time, and Streaming Interactive World Model
5. Adding Conditional Control to Text-to-Image Diffusion Models
6. Diffusion Forcing: Next-token Prediction Meets Full-Sequence Diffusion
7. Diffusion Models Are Real-Time Game Engines
8. From Slow Bidirectional to Fast Autoregressive Video Diffusion Models
