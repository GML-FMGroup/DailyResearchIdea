# Monotone Certificate Residuals for Certified Visual Routing in Diffusion VLMs

## Motivation

Existing step-wise visual routing methods recompute a step-local proxy at every denoising step with no invariant linking the certified set at step t to the set at t+1. LOWN (Look Only When Needed) certifies a token as image-free sufficient whenever the image-free and last image-conditioned predictions agree, but this agreement is re-evaluated from scratch each step, so a token certified at t can silently become image-dependent at t+1 without any proof obligation firing. The same structural gap recurs across the paper-tree: Elastic-Cache reuses KV blocks by a drift test and SlowFast/KLASS unmask tokens by a step-local stability signal, all treating a score from one step as valid at later steps without nesting or a checkable re-opening condition, so certification is non-monotone under belief refinement and cached reuse is unjustified. The root cause is that no cross-step quantity is defined whose monotonicity would make the certified set provably nested.

## Key Insight

The certificate residual — the gap between the image-free and image-conditioned predictive distributions for a token — is non-increasing across denoising steps under a contraction condition on the masked-diffusion transition kernel, so the certified set is nested and any re-opening is forced by a measurable entropy increase rather than by a fresh heuristic score.

## Method

**What it is.** We introduce MACR (Monotone-certified Adaptive Certificate Routing), an inference-time routing rule for diffusion VLMs. Its inputs are, at each denoising step t, the per-token image-free predictive distribution p_t^0 (computed by the same network with the visual prefix masked out) and the last image-conditioned distribution q_s^I cached from the most recent step s at which the image was actually processed. Its outputs are a certified set C_t of tokens whose cached image-conditioned prediction may be reused, a set R_t of re-opened tokens that require a fresh image-conditioned forward pass, and the per-step unmask count |C_t|.

**How it works.**

```pseudocode
# per denoising step t, for each masked token i
# state: cached q_s^I[i], cached residual r_s[i] = D(p_s^0[i], q_s^I[i]), certified flag
# D = total variation distance (L1/2), NOT symmetric KL, because TV is non-increasing
# under any stochastic map applied to both arguments (data-processing inequality).

function step(t, masked_tokens, cached):
    p0 = network(masked_tokens, image=None)          # image-free pass, always cheap
    C_t, R_t = set(), set()
    for i in masked_tokens:
        r_t[i] = D(p0[i], cached.q_s^I[i])            # certificate residual, D = TV
        # re-opening condition: entropy of sharpened belief rises above cached entropy
        H_t[i] = H( sharpen(p0[i], tau=0.7) )
        reopen = (H_t[i] > cached.H_s[i] + delta) or (r_t[i] > cached.r_s[i] + eps)
        if reopen:
            R_t.add(i)                               # proof obligation fires
        else:
            C_t.add(i)                               # certified: reuse q_s^I[i]
    if R_t:
        q_t^I = network(masked_tokens, image=visual_prefix)   # one image-conditioned pass
        for i in R_t: cached.q_s^I[i] = q_t^I[i]; cached.H_s[i] = H_t[i]; cached.r_s[i] = r_t[i]
    # unmask the |C_t| most-confident certified tokens (adaptive step size)
    unmask(top_k(C_t, k=|C_t|))
    return C_t, R_t

# per-token, per-step contraction certificate (checked for each token i at step t):
#   let alpha_t[i] be the actual mask/keep probability used by the sampler at step t for token i
#   (available from the unmask schedule; for LLaDA-V, alpha_t[i] = 1 - confidence_i / sum_j confidence_j).
#   The applied stochastic masking map M_t[i] satisfies, for any two distributions u,v on the token vocabulary,
#       ||M_t[i](u) - M_t[i](v)||_1 <= (1 - alpha_t[i]) * ||u - v||_1
#   because M_t[i] is a convex combination of the identity and a constant (mask) distribution with weight alpha_t[i].
#   We therefore set the per-token contraction factor to (1 - alpha_t[i]) and check:
#       r_t[i] <= (1 - alpha_t[i]) * r_s[i] + eps_tol,   with eps_tol = 1e-3 (numerical slack).
#   If this holds for token i, the residual is non-increasing for that token and C_t is nested in C_{t+1} for that token.
#   If it fails, token i is forced into R_t (re-opened) regardless of the entropy condition.

# Assumption A (stated explicitly): the cached image-conditioned belief q_s^I has not drifted since step s
# beyond what the same per-token contraction bound allows; i.e., the staleness residual ||q_s^I - q_t^I||_1
# is bounded by (1 - (1 - alpha_t[i])) * ||p_s^0 - q_s^I||_1. When this fails (image-conditioned belief drifts
# faster than the contraction bound), r_t reflects staleness rather than image-independence, and the per-token
# check above will fail, forcing a re-open. This is the failure mode F.
```

The residual D is total variation distance (half the L1 distance) between the image-free and cached image-conditioned distributions; we use TV rather than symmetric KL because TV is provably non-increasing under any stochastic map applied to both arguments (data-processing inequality), whereas KL is not contracted by a stochastic map in general. The per-token contraction check uses the actual mask/keep probability alpha_t[i] from the sampler's unmask schedule, which for LLaDA-V is the confidence-based normalization alpha_t[i] = 1 - confidence_i / sum_j confidence_j; the closed-form L1 contraction factor of the absorbing-mask kernel is (1 - alpha_t[i]). Hyperparameters: sharpening temperature tau=0.7, re-opening margins delta=0.05 nats and eps=0.02 nats, numerical slack eps_tol=1e-3.

**Why this design.** We chose a residual defined as total variation distance between image-free and cached image-conditioned distributions rather than LOWN's binary agreement flag because agreement discards the magnitude that monotonicity needs, and TV (unlike symmetric KL) is provably non-increasing under the stochastic masking map actually applied, accepting the cost of one extra distribution comparison per token per step. We chose a per-token cached entropy as the re-opening trigger rather than re-running the full image-conditioned pass each step because the entropy rise is exactly the quantity the per-token contraction condition bounds, accepting the cost that a stale cached entropy can delay a re-open by one step. We chose adaptive unmask count |C_t| rather than a fixed schedule because nesting makes the certified set a principled step-size, accepting the cost that early steps with few certified tokens unmask slowly. A domain expert would not describe MACR as a variant of LOWN: LOWN re-derives a fresh agreement score each step with no cross-step obligation, whereas MACR's decision is a monotone certificate whose re-opening is a proof obligation tied to a per-token, per-step kernel contraction condition, so the certified set is nested by construction rather than by luck.

**Why it measures what we claim.** The residual r_t[i]=D(p_t^0[i], q_s^I[i]) measures image-free sufficiency of the cached image-conditioned prediction because total variation distance between the two distributions is zero exactly when the image does not change the token's predictive distribution, under the assumption A that the cached q_s^I is a faithful image-conditioned belief (staleness bounded by the same contraction); this assumption fails when the image-conditioned belief has drifted since step s faster than the contraction bound (failure mode F), in which case r_t reflects staleness rather than image-independence and the per-token contraction check forces a re-open. The sharpened-belief entropy H_t[i] measures belief refinement because entropy of a sharpened distribution decreases monotonically as the model becomes more certain, under the assumption that sharpening preserves the ranking of the underlying belief; this assumption fails when the image-free belief is confidently wrong, in which case H_t reflects false certainty. The per-token contraction check r_t[i] <= (1 - alpha_t[i]) * r_s[i] + eps_tol measures the nesting property because the L1 contraction of the absorbing-mask kernel with actual mask probability alpha_t[i] is exactly the condition under which the TV residual is non-increasing, under the assumption that the sampler uses the absorbing-mask kernel whose per-token operator norm is (1 - alpha_t[i]); this assumption fails for non-masked (uniform-noise) diffusion, in which case the check reflects a different kernel and nesting is not certified.

## Contribution

(1) MACR, a per-token certificate residual with a checkable contraction condition and an entropy-rise re-opening rule that makes the certified visual-routing set provably nested across denoising steps. (2) A design principle: adaptive unmask count driven by the nested certified set, replacing step-local agreement heuristics with a cross-step proof obligation. (3) An inference-time routing recipe requiring no training and no extra image-conditioned forward passes beyond those already used by LOWN-style baselines.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | LLaDA-V multimodal reasoning benchmarks (image-dependent and image-independent subsets) | Separates cases where the image matters from where it does not. |
| Primary metric | Image-conditioned forward-pass count at matched accuracy | Directly measures the compute the certificate claims to save. |
| Secondary metric | Fraction of tokens/steps satisfying the per-token contraction check r_t[i] <= (1 - alpha_t[i]) * r_s[i] + eps_tol | Tests whether the contraction condition empirically holds; if it rarely holds, the significance collapses to a heuristic. |
| Baseline | LLaDA-V full image processing every step | Upper bound on accuracy with no routing. |
| Baseline | LOWN binary-agreement routing | Tests whether magnitude-aware residuals beat agreement flags. |
| Baseline | Fixed-period KV/image refresh | Tests whether adaptive re-opening beats a static schedule. |
| Ablation-of-ours | MACR with per-token contraction check disabled (no nesting guarantee) | Isolates the certificate from the routing heuristic. |
| Ablation-of-ours | MACR with symmetric KL instead of TV as residual D | Tests whether the choice of divergence (TV vs KL) affects the empirical contraction rate and pass-count savings. |

### Why this setup validates the claim

The central claim is that a monotone certificate lets us skip image-conditioned passes on tokens where the image is provably irrelevant, without losing accuracy, and that the savings concentrate exactly where image-independence holds. The full-processing baseline fixes the accuracy ceiling, so any accuracy drop is attributable to routing rather than the model. LOWN tests the sub-claim that a magnitude-bearing residual is necessary: if LOWN matches MACR, the binary flag suffices and our residual is unjustified overhead. The fixed-period refresh baseline tests the sub-claim that re-opening must be driven by a per-token proof obligation rather than a global schedule; if a static period matches us, adaptivity is unnecessary. The disabled-contraction ablation tests whether nesting itself carries the benefit or whether the entropy/re-opening heuristic alone does. The secondary metric (fraction of tokens/steps satisfying the per-token contraction check) directly tests the load-bearing assumption: if the check holds on a large fraction of tokens, the nesting guarantee is empirically meaningful; if it rarely holds, MACR reduces to a heuristic and the significance claim is weakened. The TV-vs-KL ablation tests whether the choice of divergence matters for the empirical contraction rate, since TV is the divergence the per-token check actually bounds. Forward-pass count at matched accuracy is the right metric because the method's entire value proposition is reduced image-conditioned compute at equal predictive quality; a speed metric alone would hide accuracy regressions, and accuracy alone would hide the compute story.

### Expected outcome and causal chain

**vs. LLaDA-V full image processing every step** — On a case where most masked tokens are image-independent (e.g. syntactic or format tokens), the baseline pays a full image-conditioned forward pass at every denoising step because it has no mechanism to certify irrelevance. Our method instead certifies those tokens via a near-zero TV residual and reuses the cached image-conditioned distribution, re-opening only the small set whose residual or entropy exceeds the margins or whose per-token contraction check fails. We expect near-parity in accuracy with a clear reduction in image-conditioned passes, with the gap in pass count widening on image-independent subsets and shrinking toward parity on heavily image-dependent subsets. We further expect the per-token contraction check to hold on a large fraction (target >80%) of tokens on image-independent subsets, confirming the nesting guarantee is empirically active.

**vs. LOWN binary-agreement routing** — On a case where the image changes a token's distribution only slightly but the binary flag flips, LOWN re-opens that token and, worse, cannot distinguish a marginal flip from a large drift, so it either over-reopens or accepts stale beliefs. Our method instead compares the TV magnitude against the cached residual, so marginal drifts stay certified while genuine drift triggers re-opening. We expect MACR to match LOWN's accuracy while issuing fewer image-conditioned passes on the marginal-drift subset, and to beat LOWN's accuracy on the large-drift subset where LOWN's flag under-reacts.

**vs. Fixed-period KV/image refresh** — On a case where image relevance is temporally clustered (many tokens become image-dependent at once, then few), a fixed period either refreshes too often during the quiet phase or too rarely during the burst, producing wasted passes and stale predictions respectively. Our method instead lets the certified set shrink exactly when residuals rise, so re-opening tracks the burst. We expect a fixed-period scheme to show either higher pass counts at equal accuracy or lower accuracy at equal pass counts, while MACR sits on the better frontier, with the separation largest on benchmarks whose image-dependence is bursty.

**vs. MACR with per-token contraction check disabled** — If disabling the contraction check leaves accuracy and pass counts unchanged, the monotone-certificate machinery is doing no work beyond the entropy heuristic. We expect the disabled-check variant to show slightly higher pass counts or slightly lower accuracy on the bursty subset, confirming the per-token check contributes to routing quality.

**vs. MACR with symmetric KL instead of TV** — If symmetric KL yields the same empirical contraction rate and pass-count savings as TV, the choice of divergence is immaterial. We expect TV to yield a higher fraction of tokens satisfying the per-token contraction check (since TV is the divergence the check bounds), and thus slightly better pass-count savings at equal accuracy.

### What would falsify this idea

If MACR's pass-count savings are uniform across image-dependent and image-independent subsets rather than concentrated where the image-free and cached distributions nearly coincide, the certificate is not tracking image sufficiency and the central claim is wrong. Likewise, if the per-token contraction check holds on fewer than 50% of tokens/steps on LLaDA-V, the nesting guarantee is empirically vacuous and the method reduces to a heuristic; and if disabling the contraction check leaves accuracy and pass counts unchanged, the monotone-certificate machinery is doing no work beyond the entropy heuristic.

## References

1. Look Only When Needed: Certificate-Guided Visual Routing for Efficient Diffusion VLMs
2. Attention Is All You Need for KV Cache in Diffusion LLMs
3. Accelerating Diffusion Large Language Models with SlowFast Sampling: The Three Golden Principles
4. KLASS: KL-Guided Fast Inference in Masked Diffusion Models
5. Simplified and Generalized Masked Diffusion for Discrete Data
6. Your Absorbing Discrete Diffusion Secretly Models the Conditional Distributions of Clean Data
7. Simple and Effective Masked Diffusion Language Models
8. Simple Guidance Mechanisms for Discrete Diffusion Models
