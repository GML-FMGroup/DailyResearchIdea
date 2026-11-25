# Kernel-Closure Certificates for Token-Rewired Attention: Verifying LoomFormer Circuits in DeltaFormer's Kernel State-Space Class

## Motivation

The field now routinely asserts that architectural modifications unlock expressivity beyond the TC0 ceiling that DeltaFormer (E1/E8) characterizes for fixed-kernel state-space transformers, but it does so without any invariant that survives token-dependent rewiring. LoomFormer (E6) is the sharpest instance: its per-token matrix memory is updated by a gated linear-attention fixed-point recurrence, and its vertical-attention rewiring of q/k/v is asserted to remain expressive only empirically, because no one has checked whether the rewired kernel at each token still lies in DeltaFormer's kernel-function class — DeltaFormer assumes one shared kernel, so a token-varying kernel is literally outside its hypotheses. The root cause is structural: closure under per-token rewiring is a property of the update operator, not of any single forward pass, so benchmark gains cannot establish it and a certificate must be attached to the recurrence itself.

## Key Insight

Closure is a property of the fixed-point operator family, not of any forward pass: if the per-token memory update is a contraction in a norm compatible with DeltaFormer's kernel parametrization and the rewiring map is Lipschitz with constant below the contraction margin, then every rewired kernel K_t provably remains in the kernel-function state-space class, so the certificate is a computable per-token scalar rather than an empirical claim.

## Method

**What it is.** Kernel-Closure Certifier (KCC) is an inference-time and training-time monitor attached to a LoomFormer block. Inputs: at each token t, the per-token matrix memory M_t ∈ R^{d×d}, the gated linear-attention update operator G_t (gate vector g_t, normalization denominator z_t), and the rewiring map R_t that produces (q_t,k_t,v_t) from the matrix stream. Outputs: a per-token closure margin ρ_t ∈ R and a binary certificate c_t ∈ {0,1}; a block is *certified* when c_t = 1 for all t in a window of length W=64.

**How it works.**

```pseudocode
# Per LoomFormer block, per token t. Constants: eps=1e-6, tau_rho=0.0, W=64, beta=0.9 (EMA on Lipschitz estimate), n_probe=8 (random probe directions per token), z_min=1e-3 (denominator floor for analytic bound)

function KCC_STEP(M_prev, g_t, z_t, R_t, M_t, K_prev):
    # 1. Contraction audit of the gated linear-attention fixed point
    #    G_t(M) = diag(g_t) M + (1 - z_t) * outer(k_t, v_t)  is affine; its linear part is diag(g_t)
    s_t      = max_i |g_t[i]|                      # spectral norm of diag(g_t), exact
    L_G_t    = s_t                                  # contraction factor of G_t

    # 2. Certified upper bound on the Lipschitz constant of the rewiring map R_t
    #    (a) Analytic bound for vertical-attention + normalization composition:
    #        L_R_analytic = (||W_q||_2 + ||W_k||_2 + ||W_v||_2) / max(z_t, z_min)
    #        where W_q, W_k, W_v are the projection weight matrices of R_t.
    #    (b) Empirical multi-direction estimate via power iteration on the Jacobian J_R_t:
    #        Initialize v_0 ~ N(0, I) (unit norm). For i = 1..n_probe:
    #            u_i = J_R_t v_{i-1}   (computed by finite difference: (R_t(M_t + h v_{i-1}) - R_t(M_t)) / h, h=1e-4)
    #            v_i = u_i / ||u_i||_2
    #        L_R_emp = ||u_{n_probe}||_2 * (1 + 1/sqrt(n_probe))   # high-probability correction factor
    #    (c) Conservative bound: L_R_t = min(L_R_analytic, L_R_emp)
    L_R_t    = min(L_R_analytic, L_R_emp)
    L_R_hat  = beta * K_prev.L_R_hat + (1 - beta) * L_R_t   # EMA, carried across tokens

    # 3. Fixed-point stability condition
    #    Require the composite map F_t = R_t o G_t to be a contraction on the memory space
    rho_t    = 1.0 - L_R_hat * L_G_t                # closure margin

    # 4. Bounded spectral norm of the per-token memory (DeltaFormer kernel validity)
    sigma_t  = ||M_t||_2
    bounded  = sigma_t <= KAPPA_MAX                 # KAPPA_MAX set from init: 4 * ||M_0||_2

    c_t      = 1 if (rho_t > tau_rho and bounded) else 0
    return rho_t, c_t, L_R_hat

function CERTIFY_WINDOW(certificates):
    return all(c_t == 1 for t in last W tokens)
```

At training time, a hinge penalty `max(0, tau_rho - rho_t)` is added to the loss with weight 0.01, and the gate vector g_t is reparametrized as `g_t = sigmoid(a_t) * (1 - delta)` with `delta = 0.05`, which hard-caps `s_t ≤ 1 - delta` so `L_G_t` can never reach 1 and the contraction margin is structurally available. `KAPPA_MAX` is recomputed once per epoch from the 99th percentile of observed `sigma_t` on a held-out batch, not from the training batch, to prevent the bound from being gamed.

**Why this design.** We chose to *measure* `L_R_t` by a certified upper bound combining an analytic bound (which explicitly floors the normalization denominator at `z_min=1e-3` to avoid blow-up) and a multi-direction power-iteration estimate with a high-probability correction factor, rather than a single-direction finite difference, because a single-direction estimate is only a lower bound on the spectral norm and can miss violations along orthogonal directions; the analytic bound provides a conservative upper bound, and taking the minimum ensures the certificate is conservative rather than optimistic. We chose a hard reparametrization `g_t = sigmoid(a_t)(1-delta)` over a soft penalty on `s_t` because the contraction margin must be *structurally* guaranteed for the certificate to mean anything; a soft penalty lets a single token violate closure and voids the whole window, accepting the cost of a fixed 5% gate-capacity loss. We chose an EMA (`beta=0.9`) for `L_R_hat` over the instantaneous `L_R_t` because single-token estimates are dominated by quantization noise, accepting the cost that the certificate lags a genuine Lipschitz spike by ~10 tokens. A domain expert would not describe this as "DeltaFormer + LoomFormer": DeltaFormer supplies a *fixed* kernel class and LoomFormer supplies a *token-varying* rewiring, and the contribution is the contraction-margin invariant `rho_t` that neither paper defines and that no composition of the two produces, since DeltaFormer's hypotheses exclude token-varying kernels by construction.

**Why it measures what we claim.** `rho_t = 1 - L_R_hat * L_G_t` measures *closure of the rewired kernel in DeltaFormer's kernel-function class* because the named assumption is that the composite memory update `F_t = R_t o G_t` is differentiable and its local Lipschitz constant is dominated by `L_R_hat * L_G_t` (a product bound valid when `G_t` is affine with linear part `diag(g_t)` and `R_t` is applied after); this assumption fails when `R_t`'s Jacobian is ill-conditioned along directions orthogonal to the probe directions, in which case `rho_t` reflects only the sampled directions and a true violation can pass uncertified. `sigma_t = ||M_t||_2` measures *validity as a DeltaFormer kernel* because the named assumption is that DeltaFormer's kernel class is norm-bounded by `KAPPA_MAX` inherited from initialization; this assumption fails when training drifts `M_t` beyond the init scale, in which case `bounded` reflects initialization hygiene rather than a property of the rewiring. `c_t` measures *per-token closure* because the assumption is that a window of `W=64` consecutive certified tokens is representative of the sequence's operator regime; this fails under distribution shift mid-sequence, in which case the window certificate reflects the pre-shift regime.

## Contribution

(1) A per-token closure certificate (KCC) consisting of a contraction-margin scalar, a measured Lipschitz estimate of the rewiring map, and a memory-norm bound, with a gate reparametrization that makes the contraction margin structurally available. (2) A design principle: token-varying kernels remain inside a fixed-kernel state-space class iff the composite memory-update operator is a contraction with margin, which converts an empirical expressivity claim into a monitorable training signal. (3) An open-source certificate monitor plus a certified-vs-uncertified LoomFormer checkpoint pair for downstream auditing.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
|------|--------|-----------|
| Dataset | Long-context state-tracking suite (256K) | Stresses token-varying rewiring and kernel closure |
| Primary metric | Certified-window rate vs. downstream accuracy gap | Ties certificate to actual correctness, not just margin |
| Baseline | LoomFormer (no KCC) | Tests whether monitoring alone predicts failures |
| Baseline | DeltaFormer (fixed kernel) | Tests token-varying rewiring advantage under closure |
| Baseline | Jamba hybrid Transformer-Mamba | Tests expressivity gain beyond hybrid state-space |
| Ablation-of-ours | KCC with soft gate penalty, no hard cap | Isolates structural contraction guarantee |
| Ablation-of-ours | KCC with single-direction finite-difference Lipschitz estimate (original) | Tests whether the certified multi-direction bound is necessary for soundness |

### Why this setup validates the claim

The central claim is that the contraction margin `rho_t` is a *causal* predictor of when a LoomFormer block's rewired kernel leaves DeltaFormer's kernel-function class, and that structurally guaranteeing `rho_t > 0` recovers performance exactly on the inputs where that exit would have occurred. This design makes that falsifiable by pairing a metric that couples the certificate to behavior (certified-window rate *and* the accuracy gap restricted to uncertified windows) with baselines that each isolate one sub-claim. LoomFormer-without-KCC tests whether monitoring is merely descriptive or genuinely predictive: if uncertified windows are no less accurate than certified ones, `rho_t` measures nothing. DeltaFormer tests whether token-varying rewiring, once closure is guaranteed, buys expressivity over a fixed kernel class. Jamba tests whether the gain is specific to rewired-kernel closure rather than generic state-space capacity. The soft-penalty ablation is the decisive one: if a soft gate penalty matches the hard cap, then the *structural* guarantee was never load-bearing and the mechanism story is wrong. The single-direction ablation tests whether the certified multi-direction bound is necessary for soundness: if the single-direction estimate yields the same certified-window rate and accuracy gap, then the additional computation is unnecessary; if it produces more false certificates (certified windows with high error), the multi-direction bound is load-bearing. Downstream accuracy concentrated on the predicted failure subsets—not uniform gains—is the only pattern consistent with the causal chain.

### Expected outcome and causal chain

**vs. LoomFormer (no KCC)** — On a long-context entity-tracking instance where the rewiring map's local Lipschitz constant spikes mid-sequence, unmonitored LoomFormer produces a gate vector whose spectral norm drifts toward 1, so the composite memory update stops contracting and stale matrix-memory states persist; the baseline has no signal that this happened. Our method flags `c_t = 0` at the spike (EMA lag ~10 tokens) and the hinge penalty keeps `s_t ≤ 1-delta`, so closure is restored. We expect a pronounced accuracy gap on uncertified-window tokens and near-parity on certified ones—i.e., the certificate partitions errors rather than uniformly improving them.

**vs. DeltaFormer (fixed kernel)** — On a Python-code-evaluation-style state-tracking case requiring the kernel to change per token, DeltaFormer's fixed kernel class cannot represent the token-varying update, producing systematic errors on exactly the tokens where the required kernel differs from the fixed one. Our method's rewiring map `R_t` supplies token-varying kernels while `rho_t` certifies each stays within a contracting regime, so the expressivity is used without losing stability. We expect a moderate but consistent gain on state-tracking subsets and parity on standard language-modeling perplexity.

**vs. Jamba hybrid Transformer-Mamba** — On a 256K-context retrieval instance where attention sinks and hybrid layer interleaving mask a local contraction violation, Jamba's fixed interleaving schedule cannot react per token, so errors cluster where the Mamba layers' state update and the Transformer layers disagree. Our method monitors every token's closure margin and certifies windows, so disagreements surface as `c_t = 0` before they propagate. We expect a visible gap on the long-context retrieval subset with parity on short-context benchmarks.

### What would falsify this idea

If the accuracy gain from KCC is uniform across certified and uncertified windows—rather than concentrated on the uncertified-window subset the certificate is supposed to flag—then `rho_t` is not measuring closure and the central claim fails. Equivalently, if the soft-penalty ablation matches the hard gate cap, the structural contraction guarantee is not load-bearing.

## References

1. Rewirable Circuits in Transformers with Loom Attention
