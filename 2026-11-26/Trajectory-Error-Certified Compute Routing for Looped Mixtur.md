# Trajectory-Error-Certified Compute Routing for Looped Mixture-of-Experts Diffusion Transformers

## Motivation

Dynamic-compute diffusion transformers such as DyLoMo-DiT and DyDiT allocate loop passes, expert capacity, and spatial tokens using static proxies — the diffusion timestep index and spatial position — rather than any measurement of the current sample's own denoising trajectory state. Because the timestep index is shared by every sample in a batch, two samples at the same t can differ enormously in how far their predicted velocity is from the true score, yet they receive identical compute; the samples that most need extra capacity are silently under-computed while easy ones waste FLOPs. The root cause is structural: timestep is a schedule coordinate, not a sufficient statistic for per-sample residual denoising error, so no function of timestep alone can certify that the allocated compute is sufficient for a given trajectory.

## Key Insight

The one-step self-consistency residual of the velocity field along the actual ODE trajectory — the discrepancy between the model's predicted velocity and the velocity implied by its own denoised next state — is a per-sample, timestep-normalized sufficient statistic for residual denoising error, so a cheap predictor of this residual can gate compute with a monotone sufficiency certificate that timestep-indexed routing structurally cannot provide.

## Method

**What it is.** TERRA (Trajectory-Error-certified Routing) is a per-sample, per-step compute controller wrapped around a DyLoMo-DiT backbone. At each denoising step it takes the current latent x_t, the model's predicted velocity v_θ(x_t,t), and a scalar sufficiency predictor g_φ; it outputs a loop count L, an active expert subset, a retained-token mask, and a binary certificate flag indicating whether the chosen compute is certified sufficient for a per-sample error threshold τ.

**How it works.**

```pseudocode
# Precompute once per step, cheap: one extra forward through the FIRST loop block only
1. v0 = v_theta(x_t, t)                       # full teacher velocity, standard DiT forward
2. x_hat_0 = x_t - (1 - alpha_bar_t)^{1/2} * v0   # one-step denoised estimate
3. v1 = v_theta(x_hat_0, t_next)              # velocity at the implied next state (1 extra block-0 pass)
4. r_t = || v1 - v0 ||_2 / (sigma(t) + eps)   # timestep-normalized self-consistency residual
5. e_hat = g_phi(r_t, t, ||v0||, layer_norm_stats)   # scalar predicted residual error

# Sufficiency loop: escalate compute until certified
6. L = L_min (default 1); experts = top-1 router; tokens = full grid
7. while e_hat > tau and L < L_max (default 4):
8.     L += 1
9.     experts = top-k_router (k grows with e_hat: k = 1 + floor(e_hat/tau))
10.    tokens = keep top-p fraction (p = clamp(1 - e_hat/(2*tau), 0.5, 1.0)) by per-token residual norm
11.    run L loop passes with token mask and expert subset; recompute v0, r_t, e_hat
12. certified = (e_hat <= tau)
13. return x_{t_next}, L, experts, tokens, certified
```

Hyperparameters: τ (error threshold, tuned per dataset on a 1k-sample calibration split), L_min=1, L_max=4, eps=1e-4, g_φ is a 2-layer MLP (64 hidden units, GeLU activation) trained on 50k (r_t, t, ||v0||, layer-norm stats) → observed residual pairs collected offline from teacher rollouts.

**Assumption A (explicit).** The residual r_t = ||v1 − v0||/(σ(t)+ε) is a monotone, per-sample sufficient statistic for true residual denoising error, under the assumption that the velocity field is approximately Lipschitz with constant bounded by 1/σ(t). This assumption fails when the model is in a high-curvature region (e.g., near a mode boundary at low t), in which case r_t reflects local curvature rather than global residual error. We do not claim this holds universally; we measure it empirically (see experiment) and withhold the certificate when a high-curvature diagnostic fires.

**Why this design.** We chose a self-consistency residual computed from the model's own velocity field over an independently-trained uncertainty head because the residual is available at every step without labels and is causally tied to the quantity we care about (residual denoising error), accepting the cost of one extra block-0 forward pass per step. We chose a closed-loop escalation (while e_hat > τ) over a single-shot predictor because a one-shot predictor can be wrong in either direction, and the loop guarantees that any uncertified step receives strictly more compute — accepting the cost that hard samples may consume up to L_max passes. We chose to modulate experts and tokens jointly with loop count rather than independently because the three compute axes are coupled through the same residual signal; decoupling them (as DyDiT does for width and tokens) would reintroduce the timestep-indexed static allocation we are replacing. We chose a small MLP g_φ over a learned router on the full latent because g_φ must be differentiable and cheap enough to run per step, accepting that it is a coarse scalar rather than a full error map. A domain expert would not describe this as a variant of DyLoMo-DiT's timestep-indexed loop allocation, because the decision variable is a measured trajectory residual, not the schedule coordinate t; nor is it a variant of DyDiT's width/token scheduling, because certification is a per-sample closed-loop condition rather than a fixed function of t and position.

**Why it measures what we claim.** The quantity r_t = ||v1 − v0||/(σ(t)+ε) measures residual denoising error because the one-step self-consistency of a velocity field along its own ODE trajectory equals the local truncation error of the probability-flow ODE, under the assumption that the velocity field is approximately Lipschitz with constant bounded by 1/σ(t); this assumption fails when the model is in a high-curvature region (e.g., near a mode boundary at low t), in which case r_t reflects local curvature rather than global residual error. The predictor output e_hat measures per-sample difficulty because g_φ is trained on (r_t, t, ||v0||, layer-norm stats) → observed residual pairs from teacher rollouts, under the assumption that these four statistics are sufficient for the residual; this assumption fails when a sample's difficulty is driven by a token-localized structure not captured by layer-norm stats, in which case e_hat reflects average rather than worst-case difficulty. The certificate flag certified = (e_hat ≤ τ) measures compute sufficiency because escalation is monotone in e_hat and the loop terminates only when the predicted residual falls below τ, under the assumption that g_φ is calibrated so that e_hat ≤ τ implies true residual ≤ τ; this assumption fails when g_φ is overconfident on out-of-distribution t, in which case certified reflects predictor confidence rather than true sufficiency. The token mask (top-p by per-token residual norm) measures spatial compute necessity because per-token residual norm is the token's contribution to the global residual, under the assumption that residual norms are additive across tokens; this assumption fails when tokens interact strongly through attention, in which case the mask reflects marginal rather than joint necessity.

## Contribution

['A per-sample trajectory-error-certified compute controller (TERRA) for looped mixture-of-experts diffusion transformers, in which loop count, expert activation, and token retention are all driven by a measured self-consistency residual rather than by the diffusion timestep index.', 'A closed-loop sufficiency certificate that escalates compute until a lightweight predictor of the residual falls below a threshold, providing a per-sample guarantee that timestep-indexed routing cannot express.', 'An empirical design principle: that the self-consistency residual of the velocity field along the actual ODE trajectory is a usable per-step difficulty signal for compute allocation in diffusion transformers.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | ImageNet 256×256 class-conditional | Matches DyLoMo-DiT's FID benchmark |
| Primary metric | FID at matched average FLOPs | Directly tests compute-quality tradeoff |
| Baseline | DyLoMo-DiT (static loop/MoE schedule) | Tests per-sample residual routing vs. static allocation |
| Baseline | DyDiT (timestep/spatial width scheduling) | Tests closed-loop certification vs. open-loop schedule |
| Baseline | DiT-XL/2 (uniform compute) | Tests any adaptive gain over fixed compute |
| Ablation-of-ours | TERRA without certificate loop (one-shot e_hat) | Isolates closed-loop escalation contribution |
| Ablation-of-ours | TERRA with timestep-only predictor (g_φ replaced by a 2-layer MLP taking only t) | Tests whether residual carries information beyond t |
| Diagnostic | Calibration study: predicted e_hat vs. true measured residual on 5k held-out rollouts; reliability diagram; fraction of certified steps with true residual > τ; per-timestep Lipschitz constant estimate | Validates Assumption A and g_φ calibration |
| Diagnostic | High-curvature flag: steps where the local Lipschitz estimate exceeds 1/σ(t) by a factor of 2; report certificate withheld rate | Measures failure mode F |

### Why this setup validates the claim

The central claim is that a measured per-sample trajectory residual, not the timestep coordinate, is the right decision variable for allocating loop, expert, and token compute in a diffusion transformer. ImageNet 256×256 FID at matched average FLOPs is the correct metric because it forces every method to spend the same total budget, so any FID gap reflects where compute is placed rather than how much is spent. DyLoMo-DiT tests the first sub-claim: that residual-driven loop/expert allocation beats computation indexed by the schedule coordinate t; if DyLoMo-DiT matches TERRA at equal FLOPs, the residual is not a better decision variable. DyDiT tests the second sub-claim: that closed-loop certification beats open-loop width/token scheduling, since DyDiT's allocation is a fixed function of t and position and cannot react to a hard sample. DiT-XL/2 anchors the lower bound: any adaptive method must beat uniform compute to justify its overhead. The one-shot ablation is the decisive control: it removes the escalation loop while keeping the residual predictor, so if the ablation matches TERRA, the gain comes from the predictor alone and the closed-loop certificate contributes nothing, which would falsify the mechanism we claim. The timestep-only ablation isolates whether the residual signal itself, not just per-sample adaptivity, is the novel ingredient: if a predictor using only t matches TERRA, the residual carries no information beyond the schedule coordinate. The calibration study directly measures Assumption A and the g_φ calibration: we plot predicted e_hat against true measured residual on held-out rollouts, report reliability diagrams, and compute the fraction of certified steps whose true residual exceeds τ; we also estimate the Lipschitz constant per timestep band by finite differences along the ODE trajectory. The high-curvature diagnostic flags steps where the local Lipschitz estimate exceeds 1/σ(t) by a factor of 2, and we report the certificate withheld rate; this converts the failure mode F from hand-waving into a measured quantity. Together these rows form a falsifiable test: the residual-routing claim predicts a specific ordering (TERRA > DyLoMo-DiT ≈ DyDiT > DiT-XL/2) that is visible only if the residual, not t, carries the signal, and the calibration study provides the error bars on the certificate.

### Expected outcome and causal chain

**vs. DyLoMo-DiT** — On a case where a mid-timestep sample sits in a high-curvature region of the ODE but the schedule coordinate t is unremarkable, DyLoMo-DiT allocates its loop count and expert subset from t alone, so it spends the same compute on this hard sample as on an easy one and under-resolves the trajectory, producing a residual that propagates into visible artifacts. TERRA instead measures r_t from the one-step self-consistency of the velocity field, sees the elevated residual, and escalates loop count and expert breadth until e_hat ≤ τ, so we expect a clear FID gap concentrated on the high-residual subset and near-parity on the low-residual majority at matched average FLOPs.

**vs. DyDiT** — On a case where difficulty is spatially localized (a small region of the latent that the model has not yet resolved), DyDiT's Spatial-wise Dynamic Token strategy drops tokens by a fixed function of timestep and position, so it can prune exactly the tokens carrying the unresolved residual, and its Timestep-wise Dynamic Width shrinks capacity at a step that actually needs it, yielding blur in that region. TERRA's token mask keeps the top-p fraction by per-token residual norm and its escalation is triggered by the measured residual, so it retains the unresolved tokens and widens compute only when certified insufficient, producing a localized-fidelity gain on the spatially-hard subset with no FLOP increase on easy samples.

**vs. DiT-XL/2** — On a case where the sample is easy and the velocity field is nearly linear along the ODE trajectory, DiT-XL/2 still runs the full depth and width, spending compute that buys no quality, so its FID is set by its uniform budget and it cannot trade easy-sample savings for hard-sample investment. TERRA certifies sufficiency at L_min with top-1 routing and full tokens on these samples, banking the saved compute, and reallocates it to the hard subset via escalation, so we expect TERRA to match DiT-XL/2 FID at a substantially lower average FLOPs, with the gap in FLOPs concentrated on the easy majority and the FID parity holding there.

### What would falsify this idea

If TERRA's FID gain over DyLoMo-DiT and DyDiT is uniform across residual strata rather than concentrated on the high-r_t subset, the residual is not the causal decision variable and the central claim is wrong. Equally, if the one-shot ablation matches full TERRA at matched FLOPs, the closed-loop certificate contributes nothing and the mechanism reduces to a predictor, not a certified controller. Additionally, if the calibration study shows that the fraction of certified steps with true residual > τ exceeds 5% (i.e., g_φ is miscalibrated), the certificate flag is not valid and the method reduces to a heuristic predictor.

## References

1. DyLoMo-DiT: Dynamic Looped Mixture-of-Experts Diffusion Transformers
