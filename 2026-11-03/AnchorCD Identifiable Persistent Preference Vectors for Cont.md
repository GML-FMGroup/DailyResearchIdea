# AnchorCD: Identifiable Persistent Preference Vectors for Contrastive Personalized Decoding

## Motivation

Contrastive personalized decoding (as in "Personalized LLM decoding via contrasting personal preference", 2025) extracts a preference signal only from the in-context examples present at inference time, so the effective preference vector is re-estimated from scratch whenever the context changes; under distribution shift the estimated direction is unidentifiable up to the span of whatever the current context happens to mention, and the contrastive logit correction drifts arbitrarily. The field's trajectory toward persistent personal agents (as framed conceptually in "Avatar Intelligence", which explicitly leaves identity-continuous personalization as an unsolved research challenge) makes this structural defect the binding bottleneck: a persistent agent needs a stable preference representation across tasks, yet no existing decoding method provides an anchor that survives context replacement.

## Key Insight

If the user preference vector is treated as a slowly-varying latent state in a linear-Gaussian state-space model whose observation operator is the current context's contrastive feature span, then the posterior is identifiable exactly when that span is full-rank over the preference subspace, and regularizing the inference-time contrastive signal toward the posterior mean converts an underdetermined per-context estimate into a constrained one whose drift is bounded by the Kalman gain rather than by context noise.

## Method

**What it is.** AnchorCD (Anchored Contrastive Decoding) maintains a persistent user preference vector `p_t ∈ R^d` (d = hidden dim of the contrastive layer) as the mean of a latent state in a linear-Gaussian state-space model. Inputs: a stream of interaction contexts `c_1..c_t` (each context = a prompt plus a positive/negative response pair or preference elicitation), the frozen base LLM `M`, and a contrastive head. Output: at each decoding step, a logit correction `Δ = α · (p_t · h(x))` applied to `M`'s logits, where `h(x)` is the residual-stream activation at a fixed layer.

**Load-bearing assumption (explicit).** The contrastive feature `f_t = mean(h_L(y_pref)) - mean(h_L(y_dispref))` is a noisy linear observation of a single, context-invariant latent user preference vector `p`, i.e. `f_t = H_t p + s_t`, where `s_t` is a per-context nuisance offset with zero mean and covariance `S = s * I_d` (s = 0.1). The Kalman update estimates `p` only from the component of `f_t` that is consistent across contexts; the nuisance `s_t` is averaged out by the observation noise model. This assumption fails when `s_t` has a component outside the span of `p` that is correlated across contexts (e.g., task-specific directions), in which case `p_t` reflects a task-averaged direction rather than a stable user trait.

**How it works.**

```pseudocode
# Persistent state: p_t (mean), Sigma_t (covariance), both initialized at t=0
p_0 = 0 ; Sigma_0 = sigma0^2 * I_d          # sigma0 = 1.0
Q  = q * I_d                                  # process noise, q = 1e-4 (slow drift)
R  = r * I_d                                  # observation noise, r = 0.05
S  = s * I_d                                  # nuisance covariance, s = 0.1

# --- Per-context anchoring update (runs once per new context) ---
for each new context c_t:
    # 1. Contrastive feature extraction: mean activation difference between
    #    preferred and dispreferred response at layer L
    f_t = mean(h_L(y_pref)) - mean(h_L(y_dispref))     # f_t in R^d
    # 2. Observation operator: the context's feature span (rank-k basis)
    H_t = orthonormal_basis({f_t})  union  cached_span(c_t)   # k x d, k <= d
    # 3. Kalman predict
    p_pred = p_t-1 ; Sigma_pred = Sigma_t-1 + Q
    # 4. Gate: only update if context is informative (rank of H_t >= r_min)
    if rank(H_t) >= r_min (=2):
        # 5. Kalman update with observation y_t = f_t and total observation noise R_t = R + S
        R_t = R + S
        K_t = Sigma_pred @ H_t.T @ inv(H_t @ Sigma_pred @ H_t.T + R_t)
        p_t = p_pred + K_t @ (f_t - H_t @ p_pred)
        Sigma_t = (I - K_t @ H_t) @ Sigma_pred
    else:
        p_t = p_pred ; Sigma_t = Sigma_pred   # no anchor update; keep prior

# --- Inference-time anchored contrastive decoding ---
for each decoding step s:
    h_s = h_L(prefix)                              # current activation
    # Raw in-context contrastive signal (the 2025 baseline)
    delta_raw = alpha * (f_t - f_neg) . h_s        # f_neg = mean dispreferred feat
    # Anchored correction: pull delta_raw toward persistent p_t
    delta_anchor = alpha * (p_t . h_s)
    # Directional posterior-variance weight: trust anchor less when h_s is uncertain
    num = h_s @ Sigma_t @ h_s
    den = num + h_s @ R @ h_s
    lam = num / den                                # in [0,1], high when anchor uncertain along h_s
    delta = (1 - lam) * delta_anchor + lam * delta_raw
    logits = M.logits(prefix) + delta
```

Hyperparameters: `alpha` (contrastive strength, 0.5), `q` (process noise, 1e-4), `r` (observation noise, 0.05), `s` (nuisance variance, 0.1), `r_min` (minimum context rank for update, 2), `L` (contrastive layer, mid-depth).

**Why this design.** We chose a Kalman state-space model over a rolling-average or EMA of per-context preference vectors because the EMA has no notion of which context directions are informative and will average in noise from low-rank contexts, whereas the Kalman gain `K_t` automatically down-weights observations whose feature span `H_t` does not cover the current uncertainty `Sigma_pred`; the cost is that we must commit to a linear-Gaussian observation model, which may underfit genuinely nonlinear preference structure. We chose a rank-gated update (`r_min`) over always updating because a context whose contrastive features are rank-1 cannot identify more than one preference direction and would collapse `Sigma_t` along unobserved axes; the cost is that some genuinely informative low-rank contexts are ignored. We chose to blend the anchor and raw signal with the directional posterior-variance weight `lam = (h_s^T Sigma_t h_s) / (h_s^T Sigma_t h_s + h_s^T R h_s)` over a fixed mixing weight because `lam` is a closed-form function of the posterior uncertainty along the current activation direction `h_s` and therefore adapts automatically as evidence accumulates, whereas a fixed weight would require per-user tuning; the cost is that `lam` is only as trustworthy as the Gaussian noise assumptions. A domain expert would not describe AnchorCD as a variant of the 2025 contrastive-personalization baseline, because that baseline has no persistent state at all — it recomputes the preference direction from the current context every time — and would not describe it as a variant of retrieval-based personalization, because the anchor is a posterior over a latent variable, not a nearest-neighbor lookup.

**Why this measures what we claim.** The posterior mean `p_t` measures the persistent user preference direction because the linear-Gaussian state-space assumption makes the posterior mean the minimum-mean-squared-error estimator of the latent preference under the observation model `f_t = H_t p + s_t` with nuisance `s_t` averaged out; this assumption fails when the true preference evolves nonlinearly or when `s_t` is contaminated by task-specific rather than preference-specific features, in which case `p_t` reflects a task-averaged direction rather than a stable user trait. The Kalman gain `K_t` measures the identifiability of the preference direction from context `c_t` because under the Gaussian model `K_t` is proportional to the reduction in posterior variance along the observed subspace, so a full-rank `H_t` yields a gain that updates all directions while a rank-deficient `H_t` leaves orthogonal directions at their prior; this assumption fails when `H_t`'s rows are correlated with noise directions, in which case `K_t` overstates identifiability. The directional weight `lam = (h_s^T Sigma_t h_s) / (h_s^T Sigma_t h_s + h_s^T R h_s)` measures trust in the persistent anchor versus the current context along the current activation direction `h_s` because it is the ratio of anchor uncertainty to total observation uncertainty projected onto `h_s`, so it is high (turn toward raw context) exactly when the anchor is poorly identified along `h_s`; this assumption fails when `Sigma_t` is strongly anisotropic and `h_s` does not align with the dominant eigenvector, in which case `lam` reflects miscalibrated directional confidence rather than genuine trust.

## Contribution

(1) AnchorCD, a contrastive decoding mechanism that maintains a persistent user preference vector as the posterior mean of a linear-Gaussian state-space model, with a rank-gated Kalman update that provably leaves unidentifiable directions at their prior rather than injecting context noise. (2) A design principle: anchoring the inference-time contrastive signal toward the posterior mean with a Kalman-gain-derived blending weight bounds per-context drift by the posterior variance rather than by context noise, which we validate by measuring preference-vector drift across context switches. (3) An open-source implementation of the anchored decoding loop compatible with any HuggingFace causal LM, plus a diagnostic that reports the rank of each context's contrastive feature span so practitioners can detect when the anchor is being updated from uninformative contexts.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Long-horizon multi-session preference dialogue suite | Tests persistence across drifting contexts, not single-shot. |
| Dataset (controlled) | Synthetic rank-controlled context generator (varying rank of H_t and low-rank fraction) | Pre-registered complement to real suite; isolates rank effects. |
| Primary metric | Held-out preference-direction alignment on late sessions | Directly probes whether anchor captures stable trait. |
| Secondary metric | Directional calibration error: |lam - empirical trust| on held-out steps | Tests whether lam reflects actual anchor reliability. |
| Baseline | In-context contrastive personalization (2025 baseline) | No persistent state; recomputes direction per context. |
| Baseline | EMA rolling-average preference vector | Persistent but rank-blind; averages low-rank noise. |
| Baseline | Retrieval-based nearest-neighbor personalization | Persistent via lookup, not latent posterior. |
| Baseline | Memory-augmented prompting (retrieval of past preference summaries injected into context) | Tests whether latent-state anchor is the right locus. |
| Ablation-of-ours | AnchorCD with rank gate disabled (always update) | Isolates contribution of rank-gated Kalman update. |
| Ablation-of-ours | AnchorCD with scalar trace-based lam (original) | Isolates contribution of directional posterior-variance weight. |

### Why this setup validates the claim

The central claim is that a Kalman posterior over a latent preference direction, updated only on informative (rank-sufficient) contexts, yields a more stable and better-identified preference estimate than either stateless recomputation or rank-blind averaging. The multi-session dialogue suite is the right dataset because it forces contexts to arrive sequentially with varying informativeness, so persistence and gating both matter; a single-session benchmark would let the stateless baseline tie by construction. The synthetic rank-controlled generator provides a pre-registered, fully controlled complement where the rank of H_t and the low-rank fraction are varied systematically, so causal claims about rank gating can be tested without depending on the availability of a long-horizon real dataset. The late-session alignment metric is chosen because the predicted effect is cumulative: the anchor should separate from baselines only after enough informative contexts have accumulated, so measuring early sessions would be insensitive to the mechanism. The memory-augmented prompting baseline tests whether the latent-state anchor is the right locus of intervention rather than one of several equivalent fixes. The directional calibration error metric tests whether the new directional weight lam correctly reflects anchor reliability along h_s. The in-context contrastive baseline tests the sub-claim that persistence is necessary; the EMA baseline tests the sub-claim that rank-awareness is necessary; the retrieval baseline tests whether a latent posterior beats lookup; the disabled-gate ablation tests whether the rank gate itself (not just persistence) drives the gain; the scalar-lam ablation tests whether directional weighting improves over trace-based weighting. Together these form a falsifiable design: if the anchor's advantage is real, it must appear specifically on late sessions and specifically against rank-blind and stateless baselines, and the directional weight must show lower calibration error than the scalar weight.

### Expected outcome and causal chain

**vs. In-context contrastive personalization (2025 baseline)** — On a case where session 8 arrives with a sparse rank-1 context whose contrastive feature points off the true preference axis, the stateless baseline recomputes the decoding direction from that context alone and produces a preference signal that flips toward the context-specific artifact. Our method instead blends toward the persistent anchor `p_t`, whose posterior is already tight along the true axis, because the Kalman gain down-weights the rank-deficient observation. We expect a noticeable alignment gap on late sparse-context sessions, with near-parity on early sessions before the anchor has accumulated evidence.

**vs. EMA rolling-average preference vector** — On a case where several consecutive contexts are low-rank and noisy, the EMA baseline averages those feature vectors into its running mean, dragging the preference direction off-axis because it has no notion of which directions are identifiable. Our method instead leaves orthogonal directions at their prior via the rank gate and updates only the observed subspace, so `p_t` stays on-axis. We expect the EMA to degrade progressively with the proportion of low-rank contexts while our method stays roughly flat, producing a widening gap as the low-rank fraction increases.

**vs. Retrieval-based nearest-neighbor personalization** — On a case where the current context superficially resembles an old session whose preference was task-specific rather than user-stable, the retrieval baseline pulls the nearest stored context and injects a task-averaged direction. Our method instead conditions on the posterior over a single latent trait, so task-specific features that are inconsistent across sessions are averaged out by the observation noise model. We expect a gap concentrated on sessions with high lexical overlap to stale, task-specific contexts, with parity on sessions whose nearest neighbor is genuinely preference-relevant.

**vs. Memory-augmented prompting (retrieval of past preference summaries injected into context)** — On a case where the current context is sparse but past summaries contain relevant preference information, the memory-augmented baseline must rely on the LLM to integrate retrieved text into its context, which can be noisy and limited by context window. Our method instead maintains a compact latent state that is updated online and used directly in decoding, avoiding the need to re-parse retrieved summaries at each step. We expect our method to outperform memory-augmented prompting on late-session alignment, especially when the number of past sessions exceeds the context window, because the latent anchor compresses information more efficiently.

**vs. AnchorCD with rank gate disabled** — On a case where a rank-1 context is followed by a rank-1 context in an orthogonal direction, the disabled-gate ablation updates on both, collapsing posterior variance along unobserved axes and over-trusting noisy observations. Our method instead skips the under-informative update, preserving calibrated uncertainty so the blending weight `lam` stays appropriately high. We expect the ablation to show overconfident drift on low-rank-heavy sequences, visible as worse late-session alignment and a miscalibrated `lam` trajectory relative to our gated version.

**vs. AnchorCD with scalar trace-based lam (original)** — On a case where the posterior covariance `Sigma_t` is strongly anisotropic and the current activation `h_s` aligns with a low-variance direction, the scalar `lam = trace(Sigma_t)/(trace(Sigma_t)+trace(R))` overestimates anchor uncertainty because it averages over all directions, leading to over-reliance on the raw context. Our method instead uses the directional weight `lam = (h_s^T Sigma_t h_s)/(h_s^T Sigma_t h_s + h_s^T R h_s)`, which correctly reflects uncertainty along `h_s`. We expect the scalar-lam ablation to show higher directional calibration error and worse alignment on steps where `h_s` is misaligned with the dominant eigenvector of `Sigma_t`, while our directional weight maintains lower calibration error and better alignment.

### What would falsify this idea

If the anchor's advantage over the stateless and EMA baselines is uniform across early and late sessions and across low-rank and full-rank context subsets, rather than concentrated on late sessions with informative-context accumulation and on low-rank-heavy sequences, then the persistent-state and rank-gating mechanisms are not doing the causal work claimed. A second falsifier: if the disabled-gate ablation matches the gated version on low-rank-heavy sequences, the rank gate is not the source of robustness. A third falsifier: if the directional weight lam does not reduce calibration error relative to the scalar trace-based lam, then the directional posterior-variance operationalization is not an improvement.

## References

1. Avatar Intelligence: Amplifying Human Agency Through Persistent Personal Agents
