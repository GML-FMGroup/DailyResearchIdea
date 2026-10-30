# Fixed-Point Finality: Identifiability-Tested Token Commitment for Iterative Parallel Denoising

## Motivation

Iterative parallel denoisers such as Dream 7B and LLaDA-V commit a per-step token budget from an exogenous schedule, while LOWN and KLASS replace the fixed schedule with proxies — agreement between an image-free and an image-conditioned pass, or low KL between successive refinement steps — that certify *stability*, not *finality*. The structural gap is that no existing criterion distinguishes a token that is genuinely final from one that is merely currently confident: agreement and KL drift are both functions of two forward passes that can be small while the token's conditional distribution still moves under a different, equally admissible mask continuation, so the budget remains a heuristic count rather than the output of an identifiability test. This is the same defect that recurs across the LOWN, KLASS, and SlowFast paper-trees, and it is why none of them can derive the budget from first principles.

## Key Insight

Because the denoising operator is deterministic given the mask schedule, a token is genuinely final exactly when its conditional distribution is a fixed point of the operator restricted to the unmasked set — a property testable by bracketing the fixed-point residual of two perturbed-mask continuations, which is an invariance condition rather than a calibrated score, so the per-step budget becomes the cardinality of the invariant set instead of a schedule input.

## Method

**What it is.** FPF (Fixed-Point Finality) is an inference-time token-commitment rule for masked-diffusion VLMs. Its inputs are the current masked sequence, the visual context, and two perturbed-mask forward passes; its output is the set of tokens certified final this step, whose cardinality *is* the unmask budget.

**Load-bearing assumption (stated explicitly).** FPF certifies a token as final when its conditional distribution is invariant under a two-sided bracket of admissible mask continuations, where the bracket is constructed deterministically from the top-k highest-attention and top-k lowest-attention unmasked tokens rather than from random draws. The assumption is: *a small two-sided L1 residual under this deterministic bracket is a sufficient certificate that the token's conditional is a fixed point of the denoising operator restricted to the unmasked set.* This assumption fails when both bracket continuations lie in the same basin and miss a direction along which the conditional would move; in that case the residual reflects local rather than global invariance and the token is committed prematurely. The method below makes this assumption testable via a basin-collapse diagnostic (reported in the experiment) and removes the uncalibrated random-draw parameters r and tau from the load-bearing path by replacing them with a deterministic attention-ranked bracket and a held-out residual quantile.

**How it works.**

```pseudocode
# state: sequence x_t with mask set M_t, image I, denoiser f_theta
# hyperparams: bracket size k (number of highest/lowest-attention unmasked tokens re-masked per side),
#              residual tolerance tau (set as the 90th percentile of residuals on a held-out calibration split of 512 examples),
#              bracket slack gamma = 0.5 * tau

1. p0 = f_theta(x_t, I)                      # baseline conditional over M_t
2. Compute attention scores a_j for each unmasked token j from the last denoising step.
   Rank unmasked tokens by a_j; let H = top-k highest-attention unmasked tokens,
   L = top-k lowest-attention unmasked tokens (k = 0.15 * |unmasked|, rounded up).
3. For each admissible continuation c in {c_plus, c_minus}:
       c_plus  = re-mask exactly the set H
       c_minus = re-mask exactly the set L
       p_c = f_theta(x_t with continuation c, I)
4. For each token i in M_t:
       R_plus[i]  = |p_c_plus[i]  - p0[i]|_1
       R_minus[i] = |p_c_minus[i] - p0[i]|_1
       # one-sided residual cannot certify a fixed point; require both sides small
       residual[i] = max(R_plus[i], R_minus[i])
5. final_set = { i in M_t : residual[i] <= tau }
       # bracket slack: tokens whose two residuals straddle tau are held, not committed
       straddle = { i : min(R_plus[i],R_minus[i]) <= tau < max(R_plus[i],R_minus[i]) }
       final_set = final_set \ straddle
6. unmask exactly |final_set| tokens (highest p0 confidence within final_set)
7. if |final_set| == 0: fall back to unmasking the single argmax-confidence token
```

The two continuations are now deterministic: c_plus re-masks the top-k highest-attention unmasked tokens and c_minus re-masks the top-k lowest-attention unmasked tokens, with k = 0.15·|unmasked| rounded up. This replaces the previous random r-fraction draw with a fixed, reproducible perturbation set that brackets the attention-ranked extremes of the unmasked set. Tolerance tau is no longer a fixed 1e-2; it is set to the 90th percentile of the residual distribution on a held-out calibration split of 512 examples, and bracket slack gamma = 0.5·tau. The calibration split is drawn from the same benchmark distribution but disjoint from the evaluation instances.

**Why this design.** We chose a two-sided bracket over LOWN's single image-free/image-conditioned agreement because agreement is a difference between two *different conditioning inputs*, which conflates finality with visual dependence, whereas a two-sided residual under mask perturbation isolates invariance of the denoising operator itself, accepting the cost of one extra forward pass beyond LOWN's two. We chose to re-mask *already unmasked* tokens rather than to perturb the mask set M_t, because the fixed-point condition is defined on the unmasked set and perturbing M_t would change the operator being tested, accepting the cost that the bracket cannot probe tokens that are still masked. We chose to *drop* straddling tokens rather than commit them, because a one-sided-small residual is exactly the KLASS failure mode (low drift in one direction with a large residual in the other), accepting a smaller budget per step in exchange for a set that is certified rather than merely stable. We replaced random re-masking with a deterministic attention-ranked bracket (top-k high vs. top-k low) so that the perturbation set is reproducible and spans the extremal attention regions of the unmasked set, removing the uncalibrated random-draw parameter r from the load-bearing path; we set tau by a held-out residual quantile rather than a fixed 1e-2 so that the tolerance is scaled to the model's residual distribution. A domain expert would not describe this as a variant of LOWN or KLASS because neither computes a residual under an *admissible mask continuation* — both compare two predictions on the *same* mask — so the object being tested (operator invariance) is different from the object they test (prediction agreement / prediction drift).

**Why it measures what we claim.** The quantity `residual[i] = max(R_plus, R_minus)` measures *finality* (invariance of the token's conditional to admissible continuations) because the denoising operator is deterministic given the mask schedule, so a small two-sided residual under the deterministic attention-ranked bracket is taken as evidence that the token's conditional is a fixed point of the operator restricted to the unmasked set; this assumption fails when the two bracket continuations happen to lie in the same basin and both miss a direction along which the conditional would move, in which case the residual reflects *local* rather than *global* invariance and the token is committed prematurely. The bracket slack term measures *certifiability* because a token whose residuals straddle tau has not been shown invariant on both sides; this fails when tau is mis-scaled to the model's residual distribution, in which case the slack reflects tolerance choice rather than genuine ambiguity — which is why tau is now set by a held-out residual quantile rather than a fixed constant. The budget `|final_set|` measures the *identifiability-derived budget* because it is by construction the cardinality of the set passing the invariance test; this fails when the fallback fires on every step, in which case the budget degenerates to the single-token schedule it was meant to replace.

## Contribution

(1) A fixed-point invariance test for token finality in iterative parallel denoising, operationalized as a two-sided residual bracket over admissible mask continuations, that replaces agreement- and KL-based proxies with an identifiability condition. (2) A design principle — the per-step token budget is the cardinality of the invariance-passing set, with a straddle-exclusion rule that provably shrinks the committed set relative to one-sided stability criteria. (3) An inference-time procedure requiring only two perturbed-mask forward passes, matching LOWN's forward-pass budget without its image-free assumption.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | LLaDA-V multimodal reasoning benchmarks | Matches where image-dependence varies across instances |
| Primary metric | Accuracy at matched unmask-step budget | Isolates budget quality from raw speed |
| Secondary metric | Basin-collapse diagnostic: fraction of committed tokens with residual ≤ tau that later flip value in a subsequent step | Directly tests the load-bearing assumption |
| Baseline | LOWN | Single image-free/image-conditioned agreement rule |
| Baseline | KLASS | Single-direction prediction-drift stability rule |
| Baseline | Confidence-threshold unmasking | No invariance test, commits on confidence alone |
| Ablation-of-ours | FPF with one-sided residual only | Tests whether two-sided bracket is load-bearing |
| Ablation-of-ours | FPF with random r=0.15 re-masking (original) | Tests whether deterministic attention-ranked bracket matters |

### Why this setup validates the claim

The central claim is that finality is operator invariance under an admissible mask continuation, not prediction agreement across conditioning inputs (LOWN) or one-directional drift (KLASS). To falsify this, the design must hold the unmask-step budget fixed and ask whether the tokens FPF commits are more often correct than the tokens those rules commit at the same budget. LOWN tests the sub-claim that image-conditioned agreement conflates visual dependence with finality: if FPF beats LOWN specifically on instances where the image is weakly informative, the mechanism difference is doing the work. KLASS tests the sub-claim that one-sided stability is insufficient: FPF's straddle-dropping should pay off exactly where KLASS commits tokens with small drift in one direction and large residual in the other. The confidence-threshold baseline tests whether any invariance test is needed at all. The one-sided ablation tests whether the two-sided bracket is load-bearing versus a cosmetic extra forward pass. The random-bracket ablation tests whether the deterministic attention-ranked bracket (the smallest repair to the load-bearing assumption) is what makes the residual a certificate rather than a Monte-Carlo estimate. The basin-collapse diagnostic directly measures the failure mode named in the load-bearing assumption: it reports how often R_plus and R_minus are both small yet the token later flips, which is the empirical signature of both continuations missing a movement direction. Accuracy at matched budget is the right primary metric because it directly measures the quality of the committed set, which is the object the method claims to improve; the basin-collapse rate is the right secondary metric because it tests the assumption that the bracket certifies finality rather than merely stability.

### Expected outcome and causal chain

**vs. LOWN** — On a case where the image is weakly informative for a token (e.g. a function word whose conditional is driven by textual context), LOWN's two predictions agree closely because the image barely moves the distribution, so LOWN commits the token as final even though the denoising operator has not been shown invariant. Our method instead re-masks already-unmasked tokens and measures the residual under mask continuation, which is large for such tokens because their conditional is sensitive to the unmasked-set state, so FPF holds them. We expect a noticeable accuracy gap on the low-image-informativeness subset but near parity on the high-image-informativeness subset, where LOWN's agreement is a genuine signal.

**vs. KLASS** — On a case where a token's conditional drifts little under one perturbation but substantially under another (e.g. a token sitting near a basin boundary), KLASS observes small single-direction drift and commits, producing a token that later proves wrong and forces downstream corrections. Our method computes both R_plus and R_minus and drops tokens whose residuals straddle tau, so the straddling token is held rather than committed. We expect FPF to show fewer premature commits and a higher final accuracy at the same budget, with the gain concentrated on the straddle-prone subset identified by KLASS's own one-sided residuals.

**vs. Confidence-threshold unmasking** — On a case where the model is confidently wrong (a peaked but mislocated conditional), confidence-threshold unmasking commits the argmax token immediately because confidence is high, with no invariance check. Our method's residual is large for such tokens because the peaked mode is not a fixed point of the operator under mask continuation, so FPF excludes them from the final set. We expect a clear gap on the confidently-wrong subset, with parity elsewhere, indicating that the invariance test rather than confidence is selecting the committed tokens.

**vs. FPF with one-sided residual only (ablation)** — On the same straddle-prone cases, the one-sided ablation commits tokens whose single-direction residual is small, reproducing KLASS's failure path, while the full two-sided FPF holds them. We expect the ablation to recover most of KLASS's behavior and to lose most of FPF's gain, confirming that the two-sided bracket, not the extra forward pass per se, is what produces the improvement.

**vs. FPF with random r=0.15 re-masking (ablation)** — On the same evaluation instances, the random-bracket variant draws two independent 15% re-maskings, which can both land in the same attention region and miss the extremal directions that the deterministic top-k high/low bracket covers. We expect the random-bracket variant to show a higher basin-collapse rate (more committed tokens that later flip) and a lower accuracy at matched budget than the deterministic attention-ranked FPF, with the gap concentrated on tokens whose attention rank is extreme (top or bottom k). This confirms that the deterministic bracket, not merely the presence of two perturbations, is what makes the residual a certificate.

**Basin-collapse diagnostic (all methods)** — For each committed token with residual ≤ tau, we record whether its value changes in any subsequent denoising step before the sequence is fully unmasked. We report the basin-collapse rate per method and per subset. We expect FPF's deterministic bracket to yield a lower basin-collapse rate than the random-bracket variant and than LOWN/KLASS, and we expect the rate to be highest on the straddle-prone subset, which is where the load-bearing assumption is most stressed.

### What would falsify this idea

If FPF's accuracy gain over LOWN and KLASS is uniform across subsets rather than concentrated on the low-image-informativeness and straddle-prone subsets where the mechanism predicts a difference, then the invariance-under-continuation account is wrong and the gain is coming from something other than the claimed mechanism. Likewise, if the one-sided ablation matches full FPF, the two-sided bracket is not load-bearing and the central design claim fails; and if the deterministic attention-ranked bracket does not reduce the basin-collapse rate relative to the random-bracket variant, then the load-bearing assumption that the bracket certifies finality rather than merely stability is empirically unsupported.

## References

1. Look Only When Needed: Certificate-Guided Visual Routing for Efficient Diffusion VLMs
2. Attention Is All You Need for KV Cache in Diffusion LLMs
3. Accelerating Diffusion Large Language Models with SlowFast Sampling: The Three Golden Principles
4. KLASS: KL-Guided Fast Inference in Masked Diffusion Models
5. Simplified and Generalized Masked Diffusion for Discrete Data
6. Your Absorbing Discrete Diffusion Secretly Models the Conditional Distributions of Clean Data
7. Simple and Effective Masked Diffusion Language Models
8. Simple Guidance Mechanisms for Discrete Diffusion Models
