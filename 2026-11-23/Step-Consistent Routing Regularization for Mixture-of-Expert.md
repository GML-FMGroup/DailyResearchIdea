# Step-Consistent Routing Regularization for Mixture-of-Experts Diffusion Language Models

## Motivation

Clover-35B-A3B-Preview (E1) reports strong agent metrics for a MoE diffusion language model but evaluates its router only through aggregate downstream scores, leaving unresolved whether expert specialization stays stable across the denoising trajectory and across block positions. The router is inherited implicitly from autoregressive MoE practice, where each token is routed once; in block diffusion the same token position is re-routed at every denoising step under a shifting context (recurrent memory plus partially denoised bidirectional context), so a per-token static router assumption is structurally violated. This meta-level problem recurs across MoE work in the Idea Cards — OLMoE (E3) analyzes routing specialization only at the token level, and MoE-X (E4/E6) redesigns routing for interpretability — none of which provides any notion of routing stability along a generative trajectory.

## Key Insight

In block diffusion the router's input distribution is a controlled sequence — the same token position is re-encoded at every denoising step with monotonically increasing context certainty — so routing flips that are not explained by context change are a measurable, regularizable signal of expert-allocation noise rather than a property that must be assumed away.

## Method

**What it is.** We introduce **STRIDE** (Step-Trajectory Routing-Instability Diagnostic and regularizer), a training-time auxiliary loss plus a diagnostic suite attached to a Clover-style MoE block-diffusion backbone. Inputs: for each denoising step $t$ and token position $i$ within a block, the router logits $z_{i,t}\in\mathbb{R}^{E}$ and the selected top-$k$ expert set $S_{i,t}$. Outputs: (i) a scalar regularization term $\mathcal{L}_{\text{STRIDE}}$ added to the denoising loss, and (ii) per-layer, per-block-position, per-step routing-stability statistics (assignment entropy, specialization drift, flip rate) logged during training and evaluation.

**Load-bearing assumption (stated explicitly).** STRIDE assumes that in a Clover-style block-diffusion MoE, step-to-step top-$k$ routing flips for a fixed token position are attributable to router noise rather than to genuine, informative changes in the partially denoised context, so penalizing them with a sigmoid-weighted set-overlap loss improves downstream agent quality rather than suppressing legitimate context-driven expert reallocation. This assumption is unverified by published work; the method is designed to test it, and the context-change-normalized flip target below is the concrete calibration that makes the regularizer measure unexplained instability rather than raw flip rate.

**How it works.**

```pseudocode
# per training micro-batch, per MoE layer l
for each block b in batch:
    for t in 1..T_denoise:                      # denoising steps, T_denoise = block_len
        z[:, t] = router(h[:, t])               # h = hidden state after GDN memory + in-block attn
        S[:, t] = topk(z[:, t], k=8)            # k=8, matching Clover's 3B/35B sparsity
        c[:, t] = max_e softmax(z[:, t])[e]     # per-position denoising certainty proxy
    # --- context-change-normalized flip penalty (step-to-step) ---
    for t in 2..T_denoise:
        flip[i,t] = 1 - |S[i,t] ∩ S[i,t-1]| / k
        delta[i,t] = |c[i,t] - c[i,t-1]|        # measured context-change magnitude
        flip_norm[i,t] = flip[i,t] * exp(-delta[i,t] / sigma)   # sigma = 0.1
    L_flip = mean_{i,t} flip_norm[i,t] * w(t)   # w(t) = 1/(1+exp(-(t - T/2)/tau)), tau=2
    # --- entropy floor (per-position assignment sharpness) ---
    p[:, t] = softmax(z[:, t] / T_route)        # T_route = 1.0
    H[:, t] = -sum_e p*log p
    L_ent = mean_{i,t} relu(H[i,t] - H_max)     # H_max = 0.5 * log(E)
    # --- specialization drift (expert usage across steps) ---
    U[e, t] = fraction of positions routed to expert e at step t
    L_drift = mean_e  sum_{t=2..T} |U[e,t] - U[e,t-1]|
    L_STRIDE = lambda_f * L_flip + lambda_e * L_ent + lambda_d * L_drift
    # lambda_f=0.01, lambda_e=0.005, lambda_d=0.002 (tuned on held-out agent val split)
    loss = L_denoise + L_STRIDE
```

Diagnostics (no gradient): per layer, record flip rate vs. step index, flip rate vs. block position, and expert-usage entropy; these are reported as STRIDE curves. Additionally, log the per-step certainty delta $\delta_{i,t}$ and the normalized flip $\text{flip\_norm}_{i,t}$ so that the fraction of flips explained by context change can be read off directly.

**Why this design.** We penalize step-to-step flips rather than per-token routing entropy alone because the diffusion trajectory gives a natural counterfactual (the same position under a slightly more denoised context), which per-token entropy cannot see; the cost is that early denoising steps legitimately flip as context becomes informative, which we absorb with the sigmoid weight $w(t)$ that down-weights the first steps and with the context-change normalization $\exp(-\delta/\sigma)$ that suppresses flips accompanied by large certainty updates. We use a hard top-$k$ set-overlap flip measure rather than a KL between soft routing distributions because Clover's dispatch is hard top-$k$ (E1), so soft-distribution KL would not track the actual expert set that processes the token; the cost is a non-differentiable statistic, which we handle by detaching the set and back-propagating only through the logit-margin surrogate $z_{i,t}[e]-z_{i,t}[e']$ for flipped pairs. We add the specialization-drift term on expert usage rather than only per-position flips because a router can be position-stable while still collapsing onto a few experts across steps; the cost is a global statistic that is coarser than per-position signals. A domain expert would not describe this as a variant of OLMoE's routing-property analysis (E3) or MoE-X's sparsity-prioritized router (E4/E6): those operate on a single forward pass per token, whereas STRIDE's load-bearing quantity is defined only along the denoising trajectory, which does not exist in an autoregressive MoE.

**Why it measures what we claim.** The context-change-normalized flip rate measures routing instability under context refinement because we assume that between adjacent denoising steps the informative content of the block changes only marginally (the model's own denoising schedule is smooth), so any flip not accompanied by a large certainty delta is attributable to router noise rather than to genuine new evidence; this assumption fails when the block-diffusion schedule makes large single-step jumps in token certainty, in which case the raw flip rate reflects schedule sharpness rather than router instability — the $\exp(-\delta/\sigma)$ factor is the concrete calibration that discounts such jumps, and the schedule-perturbation control in the experiment directly tests whether residual flips track router noise or schedule sharpness. The per-position assignment entropy measures expert-allocation sharpness because we assume the router logits are the sufficient statistic for expert relevance at that position; this fails when two experts are functionally redundant (as OLMoE's high-specialization finding suggests they are not, E3), in which case high entropy reflects benign redundancy rather than indecision. The expert-usage drift measures specialization stability across the trajectory because we assume usage frequency is a proxy for specialization; this fails when an expert is legitimately step-specialized (e.g., a 'late-denoising' expert), in which case drift reflects intended temporal division of labor rather than instability — which is why we report drift separately rather than only penalizing it.

## Contribution

(1) STRIDE, a diffusion-trajectory routing-stability regularizer and diagnostic suite for MoE diffusion language models, comprising a step-flip penalty with a sigmoid step-weight, a per-position entropy floor, and an expert-usage drift term, all defined over the denoising trajectory rather than a single forward pass. (2) A design principle: routing stability in block diffusion should be measured as a function of denoising step and block position jointly, because the router's input distribution is non-stationary along the trajectory in a way that autoregressive MoE routing analyses do not model. (3) An open diagnostic protocol (STRIDE curves: flip rate vs. step, flip rate vs. block position, expert-usage entropy) applicable to any block-diffusion MoE backbone.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | Internal agent workload suite (SWE-bench Verified + Terminal-Bench 2.1 style tasks) | Matches Clover's reported agent regime and long block-denoising trajectories |
| Dataset (open backbone) | Open block-diffusion MoE: a sub-1B model trained from scratch on the same agent-style data (or an upcycled OLMoE-style backbone adapted to block diffusion) | Tests whether STRIDE generalizes beyond Clover and scales with block length and denoising step count |
| Primary metric | Task success rate vs. denoising-step flip-rate curve | Jointly tests quality gain and predicted instability mechanism |
| Secondary metric | Context-change-normalized flip rate (flip_norm) and per-step certainty delta δ | Separates router noise from schedule sharpness; validates the load-bearing assumption |
| Baseline | Clover-35B-A3B-Preview (unmodified) | Isolates effect of STRIDE loss on same backbone |
| Baseline | Clover + per-token router entropy penalty | Tests trajectory-specific claim against static entropy regularizer |
| Baseline | Clover + soft-KL step-consistency loss | Tests hard top-k set-overlap against soft-distribution surrogate |
| Baseline | Clover + EMA-of-routing / router-logit-smoothing loss | Tests against a concrete temporal-consistency baseline from MoE practice; per-token static regularizers are invariant to step permutation, STRIDE is not |
| Ablation-of-ours | STRIDE minus L_drift (flip + entropy only) | Isolates contribution of global specialization-drift term |
| Ablation-of-ours | STRIDE with fixed sigmoid weight only (no context-change normalization) | Isolates the calibration term that addresses the load-bearing assumption |
| Control | Schedule-perturbation experiment: vary denoising schedule sharpness (e.g., number of steps, step-size distribution) and measure whether flip rate tracks schedule sharpness or router noise | Directly tests the assumption that residual flips are noise-dominated |

### Why this setup validates the claim

The central claim is that routing instability along the denoising trajectory — not per-token router indecision — is a distinct, penalizable failure mode that costs agent-task quality on a Clover-style MoE block-diffusion backbone. The unmodified Clover baseline tests whether STRIDE moves quality at all on the same architecture and data, so any gain cannot be attributed to backbone or scale. The per-token entropy baseline is the critical discriminator: it shares our entropy term but lacks any step-to-step signal, so if STRIDE's advantage comes from trajectory structure, the gap must appear here. The soft-KL step-consistency baseline tests our hard top-k set-overlap argument against the natural differentiable alternative, so a tie would undercut our design rationale. The EMA-of-routing / router-logit-smoothing baseline tests whether a generic temporal-consistency regularizer from MoE practice can capture the same effect; because such regularizers are invariant to step permutation while STRIDE is not, a STRIDE advantage here isolates the trajectory-indexed signal. The no-drift ablation isolates whether the global usage statistic adds anything beyond per-position flips, and the no-normalization ablation isolates the calibration term that makes the regularizer measure unexplained instability. The open-backbone evaluation tests whether the effect generalizes beyond Clover and scales with block length and denoising step count, establishing a general principle rather than a single-model tuning result. The schedule-perturbation control directly tests the load-bearing assumption: if flip rate tracks schedule sharpness rather than router noise, the context-change normalization should absorb the difference and STRIDE's quality gain should persist; if not, the mechanism claim is undermined. Task success rate is the right primary metric because routing instability is only interesting if it degrades downstream agent behavior; the flip-rate curve and the normalized flip rate are the mechanism readouts that should move before and alongside quality. Together these form a falsifiable test: STRIDE must beat all four routing-aware comparators on the same backbone, with the gap concentrated where normalized flip rates are highest and robust to schedule perturbation.

### Expected outcome and causal chain

**vs. Clover-35B-A3B-Preview (unmodified)** — On a long-horizon agent task where the block-diffusion schedule forces many denoising steps over a code-edit block, the unmodified router flips top-k expert sets between adjacent steps as in-block attention sharpens, so the same token is processed by inconsistent expert mixtures and the drafted edit drifts. Our method instead penalizes those flips through the context-change-normalized set-overlap term, keeping expert assignment coherent across the trajectory, so we expect a modest but consistent success-rate gain on multi-step agent tasks with parity on short single-block tasks where few flips occur.

**vs. Clover + per-token router entropy penalty** — On a case where each step's routing is individually confident but the selected expert set changes between steps, the entropy baseline sees low per-token entropy and applies no corrective gradient, so instability persists and the block's denoising trajectory oscillates. Our method instead measures set overlap across adjacent steps, which is nonzero precisely in this regime, so it applies gradient where the entropy baseline is blind; we expect STRIDE to show lower normalized flip rates and a quality edge concentrated on high-flip-rate task subsets, with near-parity where flips are already rare.

**vs. Clover + soft-KL step-consistency loss** — On a case where two experts receive similar soft probabilities but only one enters the hard top-k set, the KL baseline treats the routing as stable and does not penalize the actual dispatch change, so the token is processed by a different expert than the previous step despite an apparently small distribution shift. Our method instead penalizes the hard set-overlap directly, matching Clover's hard top-k dispatch, so we expect STRIDE to catch flips the KL baseline misses and to show a quality gap on subsets where soft-KL reports low divergence yet hard flips are frequent.

**vs. Clover + EMA-of-routing / router-logit-smoothing loss** — On a case where the denoising trajectory revisits a token position with a genuinely different context, an EMA or logit-smoothing regularizer treats the step index as exchangeable and cannot distinguish a noise-driven flip from a context-driven reallocation, so it either over-penalizes informative flips or under-penalizes noise depending on the smoothing constant. Our method instead conditions the flip penalty on the measured certainty delta and the step index, so it is not invariant to step permutation; we expect STRIDE to show a larger quality gain than the EMA baseline on subsets with high normalized flip rates and to avoid the EMA baseline's degradation on subsets where legitimate mid-trajectory reallocation is frequent.

**vs. STRIDE minus L_drift (flip + entropy only)** — On a case where the router is position-stable but collapses onto a few experts across steps, the no-drift ablation has no term to penalize the global usage shift, so expert specialization drifts while per-position flips remain low. Our method instead adds the specialization-drift term, so we expect the full STRIDE to show lower expert-usage drift and a small but measurable quality edge on tasks that require diverse expert usage, with parity where usage is already stable.

**vs. STRIDE with fixed sigmoid weight only (no context-change normalization)** — On a case where the denoising schedule makes a large single-step jump in token certainty, the fixed-weight variant penalizes the accompanying informative flip, suppressing legitimate context-driven expert reallocation. Our method instead discounts flips accompanied by large certainty deltas, so we expect the normalized STRIDE to show higher quality on schedule-perturbed tasks and lower normalized flip rates without reducing raw flip rates on informative transitions.

### What would falsify this idea

If STRIDE's quality gain over the unmodified Clover baseline is uniform across task subsets rather than concentrated on subsets with high measured normalized step-to-step flip rates, or if the schedule-perturbation control shows that residual flips track schedule sharpness rather than router noise, the trajectory-instability mechanism is not what drives the improvement and the central claim is wrong. Equally fatal: if the per-token entropy baseline or the EMA-of-routing baseline matches STRIDE on both normalized flip-rate reduction and task success, then the denoising-trajectory framing adds nothing beyond static router sharpening or generic temporal smoothing.

## References

1. Clover-35B-A3B-Preview Technical Report
