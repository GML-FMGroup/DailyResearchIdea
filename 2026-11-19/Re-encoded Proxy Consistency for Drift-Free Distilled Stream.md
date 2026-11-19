# Re-encoded Proxy Consistency for Drift-Free Distilled Streaming Video Rendering

## Motivation

Distilled streaming renderers such as Invoker-Render in Code World Model 2.0 are trained with ProForcing to match a causal predictor step-by-step, but the proxy-to-pixel mapping itself is never re-measured on self-generated histories: the renderer can satisfy every step-wise target while the rendered frame silently implies a structural proxy that has drifted from the conditioning proxy. This is a meta-level failure recurring across controllable video generation, where control signals (optical flow in FloVD, layout/trajectory in IM-Zero, motion flows in AnimateAnything) are supplied as open-loop conditioning without any measurable invariant tying the rendered output back to the control state, so drift accumulates unobserved across rollout.

## Key Insight

If the structural proxy can be recovered from a rendered frame by the same proxy encoder used at conditioning time, then proxy-to-pixel drift becomes an observable, differentiable residual that a distillation loss can penalize directly on self-generated histories, converting open-loop conditioning into a closed-loop state-consistency invariant.

## Method

**What it is.** We introduce PCL (Proxy-Consistency Loop), a training-time closed-loop loss for distilled streaming renderers. Inputs: a four-step causal renderer R, a frozen proxy encoder E that maps a frame to a structural proxy, a conditioning proxy sequence, an appearance reference, and language. Output: a renderer whose rendered frames re-encode to the conditioning proxy along its own rollout history.

**How it works.**

```pseudocode
# Frozen: E (proxy encoder, shared with conditioning interface), causal predictor P
# Trainable: 4-step renderer R; ProForcing distillation loss L_distill
# Calibration set: 512 ground-truth frames with known proxies, used only to compute
#   the sensitivity bound s = median_{i,j} ||E(x_i) - E(x_j)||_1 / ||proxy_i - proxy_j||_1
#   and the collapse rate c = fraction of frame pairs with ||proxy_i - proxy_j||_1 > 0.1
#   but ||E(x_i) - E(x_j)||_1 < 0.01. If c > 0.2, E is replaced by a proxy decoder D
#   trained on rendered frames (see calibration note below).
for each training clip (proxy_1..T, appearance_ref, lang):
    history = []                       # self-generated frames, no teacher frames
    for t in 1..T:
        x_t = R(proxy_t, appearance_ref, lang, history[-k:])   # k=4 causal context
        # --- closed-loop proxy consistency ---
        proxy_hat_t = E(x_t)                                   # re-encode rendered frame
        L_pc += || proxy_hat_t - proxy_t ||_1                  # state-consistency residual
        # --- ProForcing distillation (unchanged target) ---
        L_distill += || x_t - P(proxy_t, history[-k:]) ||_2
        history.append(x_t.detach_for_proxy_only)              # gradients flow via L_pc only
    L = L_distill + lambda * L_pc        # lambda = 0.5, annealed to 0.1 after 20k steps
    update R with dL/dtheta
```

Key hyperparameters: lambda=0.5 initial, annealed to 0.1 at 20k steps; re-encoding applied at every rollout step (not only final); E frozen throughout; proxy residual computed in proxy space (not pixel space). Calibration: E is validated on a 512-frame calibration set by computing its sensitivity s and collapse rate c; if c > 0.2, E is replaced by a proxy decoder D (same architecture as E, trained to reconstruct proxies from rendered frames) and the residual becomes ||D(x_t) - proxy_t||_1. The calibration set is also used to report the residual's sensitivity to injected geometric perturbations of magnitude equal to the expected drift (0.05–0.2 in proxy space).

**Why this design.** We chose to re-encode *every* rollout step rather than only the final frame because drift compounds along the self-generated history that ProForcing explicitly constructs, so a single terminal residual would miss the early-step divergence that causes later compounding, accepting the cost of T forward passes through E per clip. We chose a frozen proxy encoder shared with the conditioning interface over a separately trained decoder because a shared encoder makes the residual directly comparable to the conditioning proxy (same coordinate system), accepting the cost that E's inductive bias bounds what drift is detectable; this choice is guarded by the calibration check (s and c) and falls back to a proxy decoder D when E is too many-to-one. We chose to detach the rendered frame from the distillation gradient path for the proxy loss only, so that L_pc cannot be trivially satisfied by collapsing the renderer toward E's reconstruction manifold, accepting the cost of a slightly weaker joint gradient. This is not a variant of ProForcing: ProForcing is an open-loop step-wise distillation target with no rendered-state measurement, whereas PCL adds a closed-loop invariant on the *implied* proxy state of the renderer's own outputs; a domain expert would not describe it as ProForcing with an extra loss because the loss changes what is measured (state consistency, not prediction accuracy).

**Why it measures what we claim.** The re-encoding residual ||E(R(proxy_t,...)) - proxy_t||_1 measures *proxy-to-pixel state consistency* because the assumption that E is a faithful inverse of the conditioning interface makes the recovered proxy equivalent to the proxy the renderer actually rendered; this assumption fails when E is many-to-one (distinct frames map to the same proxy), in which case the residual reflects only the equivalence-class of the rendered state, not its full geometry. The per-step accumulation of this residual measures *drift across self-generated history* because the assumption that each step's input history is the renderer's own output (not teacher frames) makes the residual a rollout-conditional quantity; this assumption fails when history is partially teacher-forced, in which case the metric reflects single-step error rather than compounding drift. The lambda-annealed weighting measures *the trade-off between pixel fidelity and state consistency* because the assumption that L_distill and L_pc are approximately separable in the renderer's parameter space makes the weighted sum a meaningful Pareto point; this assumption fails when the two losses conflict on the same parameters, in which case lambda selects a compromise rather than a joint optimum. The calibration check (s and c) operationalizes the faithfulness assumption: if c > 0.2, E is replaced by D, and the residual then measures state consistency under D's reconstruction error, which is bounded by the calibration set.

## Contribution

(1) A closed-loop proxy-consistency loss (PCL) that re-encodes rendered frames with the conditioning proxy encoder and penalizes the recovered-vs-conditioning proxy residual along self-generated rollout histories during ProForcing-style distillation. (2) A design principle: state-consistency invariants on the *implied* control state of rendered outputs are measurable and trainable without ground-truth proxies at inference, because the encoder is shared with the conditioning interface. (3) An ablation protocol separating per-step from terminal re-encoding and frozen from jointly-trained encoders, isolating which component carries the drift-reduction effect.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Multi-granularity proxy-video corpus (proxy + appearance + language) | Matches conditioning interface with aligned structural proxies. |
| Primary metric | Per-step proxy re-encoding residual over long rollouts | Directly measures state drift along self-generated history. |
| Baseline | ProForcing open-loop distillation | Tests whether closed-loop state measurement is needed. |
| Baseline | Teacher-forced proxy-consistency loss | Tests whether rollout-conditional residual beats single-step. |
| Baseline | Pixel-space terminal-frame residual | Tests whether per-step proxy-space residual beats terminal pixel check. |
| Baseline | Cycle-consistency loss on non-distilled bidirectional model | Tests whether rollout-conditional per-step proxy-space formulation is novel. |
| Baseline | Perceptual/feature-space cycle loss | Tests whether proxy-space residual beats generic feature-space cycle. |
| Ablation-of-ours | PCL with lambda fixed at 0.5, no annealing | Tests whether annealing matters for fidelity/state trade-off. |
| Ablation-of-ours | PCL with re-encoding every m-th step (m=2,4) | Tests whether subsampling preserves drift signal. |
| Ablation-of-ours | PCL with E replaced by proxy decoder D (if c>0.2) | Tests whether calibration fallback changes conclusions. |
| Diagnostic | Correlation of residual with ground-truth proxy distance | Validates residual measures true geometric drift. |

### Why this setup validates the claim

The central claim is that adding a closed-loop, per-step, proxy-space state-consistency residual to a ProForcing-distilled streaming renderer reduces compounding drift along self-generated history without sacrificing pixel fidelity. The ProForcing baseline isolates the sub-claim that open-loop step-wise distillation alone cannot detect rendered-state divergence: if PCL beats it, the gain must come from measuring the renderer's own implied proxy state. The teacher-forced proxy-consistency baseline isolates the sub-claim that the residual must be computed on the renderer's own rollout history rather than on teacher frames; if PCL beats it, the gain is attributable to rollout-conditional measurement rather than merely adding a proxy loss. The terminal-frame pixel-residual baseline isolates the sub-claim that per-step proxy-space measurement beats a single late check. The cycle-consistency baseline on a non-distilled bidirectional model isolates whether the mechanism's novelty is in the rollout-conditional, per-step, proxy-space formulation rather than in re-encoding per se; if PCL beats it, the gain is not from generic cycle-consistency. The perceptual/feature-space cycle loss baseline isolates whether proxy-space residual is necessary versus a generic feature-space cycle. The lambda-annealing ablation isolates whether annealing is doing real work or whether any fixed weighting suffices. The subsampling ablation (m=2,4) tests whether re-encoding every step is necessary or whether every m-th step preserves the drift signal, addressing compute feasibility. The calibration fallback ablation (E vs. D) tests whether the choice of encoder affects conclusions. The primary metric, per-step proxy re-encoding residual accumulated over long rollouts, is the right detector because the predicted effect is drift that compounds with rollout length; a metric that only looks at final frames or at pixel error would be blind to exactly the early-step divergence the mechanism targets. The diagnostic correlating the residual with ground-truth proxy distance (computed from the true rendered geometry) validates that the residual measures true geometric drift rather than an arbitrary equivalence class. Together these form a falsifiable design: each baseline removes one ingredient of the claimed mechanism, so a gain concentrated on long-rollout subsets and absent on short ones would confirm the causal story, while a uniform gain would not.

### Expected outcome and causal chain

**vs. ProForcing open-loop distillation** — On a case where the renderer must roll out many steps from its own history under a coarse proxy, the baseline produces slowly compounding geometric drift because its distillation target never inspects the rendered state, so early small errors are unpenalized and feed forward. Our method instead re-encodes each rendered frame and penalizes deviation from the conditioning proxy at every step, so early divergence is corrected before it compounds. We expect a noticeable residual gap that grows with rollout length, with near-parity on short clips and a widening gap on long ones.

**vs. Teacher-forced proxy-consistency loss** — On a case where the renderer's own output at step t differs from the teacher frame, the baseline produces a residual that reflects single-step error against teacher context rather than the drift the renderer actually accumulates, because its history is teacher-forced. Our method instead conditions each step on the renderer's own detached history, so the residual is rollout-conditional and captures compounding error. We expect the gap to be concentrated on clips with large self-generated deviation, with parity on clips where teacher and rollout histories coincide.

**vs. Pixel-space terminal-frame residual** — On a case where drift begins early but the final frame happens to land near a plausible appearance, the baseline produces a small terminal residual and thus misses the divergence, because a single late pixel-space check cannot localize early-step error. Our method instead measures proxy-space state consistency at every step, so early divergence is detected even when the terminal frame looks acceptable. We expect a clear gap on early-drift instances and near-parity on instances whose drift is genuinely terminal.

**vs. Cycle-consistency loss on non-distilled bidirectional model** — On a case where the bidirectional model is trained with a re-encoding loss but without distillation or rollout-conditional history, the baseline may reduce single-step cycle error but cannot address compounding drift because it never conditions on self-generated histories. Our method instead applies the residual on the renderer's own rollout history, so it directly penalizes compounding error. We expect PCL to outperform on long rollouts, with parity on short clips, confirming novelty is in the rollout-conditional per-step formulation.

**vs. Perceptual/feature-space cycle loss** — On a case where the feature-space cycle loss is minimized by collapsing to the encoder's null space, the baseline may show low cycle error but high geometric drift. Our method instead uses proxy-space residual, which is calibrated to distinguish proxies differing by the drift magnitude of interest (via s and c). We expect PCL to have higher correlation with ground-truth proxy distance and lower actual drift, especially on instances where feature-space cycle loss is low but drift is high.

### What would falsify this idea

If PCL's gain over ProForcing is uniform across rollout lengths and subsets rather than concentrated on long rollouts and early-drift instances, then the improvement is not caused by closed-loop drift correction and the central claim is wrong. Likewise, if the teacher-forced and terminal-frame baselines match PCL on long rollouts, the claimed necessity of per-step rollout-conditional proxy measurement is falsified. Additionally, if the calibration check shows c > 0.2 and the proxy decoder D fails to improve correlation with ground-truth drift, then the residual does not measure state consistency and the method's measurement claim is falsified.

## References

1. Code World Model 2.0: Towards Infinite Controllable Video Generation
