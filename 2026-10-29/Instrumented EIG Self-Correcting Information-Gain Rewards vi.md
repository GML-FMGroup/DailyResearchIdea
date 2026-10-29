# Instrumented EIG: Self-Correcting Information-Gain Rewards via Policy-Induced Likelihood-Ratio Tests

## Motivation

Across amortized Bayesian experimental design, the expected information gain (EIG) reward is computed as a plug-in functional of a user-supplied generative model, which every branch treats as an oracle. Amortizing Sequential Experimental Design in Reasoning LLMs and Step-DAD both optimize this plug-in reward directly; when the likelihood is misspecified the reward is biased in a direction that depends on the query distribution, so the policy systematically optimizes a distorted objective with no mechanism to detect or compensate the distortion. The structural gap is that no existing method uses the policy's own query distribution as an instrument to identify the reward bias from the same data used for training.

## Key Insight

Because the policy's query distribution provides exogenous variation in the design space that the user-supplied likelihood cannot explain, the direction and magnitude of the EIG bias become identifiable from the policy's own held-out queries, allowing a low-dimensional correction field to be learned jointly with the policy without any oracle likelihood.

## Method

**What it is.** Instrumented EIG (IEIG) is a training loop that jointly learns an amortized design policy π_θ and a low-dimensional bias-correction field c_φ. Inputs: a user-supplied (possibly misspecified) generative model p̃(y|h,d), a policy π_θ, and a stream of policy-induced held-out queries; outputs: a debiased EIG reward and a corrected policy.

**Load-bearing assumption (stated explicitly).** The instrument is valid only if the query distribution used to fit c_φ is exogenous to the likelihood misspecification — i.e., the sampling probabilities of (h,d) are known and do not depend on the learned reward EIG̃. Because a policy trained on EIG̃ will, by construction, drift toward designs that exploit the misspecification, we do NOT fit c_φ on raw policy rollouts. Instead, the correction field is fit on a randomized probe pool with known sampling probabilities, and importance weights 1/π_probe(d|h) are applied in the residual regression. This enforces exogeneity by design rather than assuming it.

**How it works.**

```pseudocode
initialize π_θ (transformer over history h, 6 layers, d_model=256, 8 heads),
           c_φ (2-layer MLP over sufficient stats s(h,d), hidden=64, GeLU),
           q_ψ (3-layer MLP discriminator over [embed(h), embed(d), y], hidden=128, LeakyReLU)

# Pre-policy calibration pool (fixed, never refit on policy rollouts)
build calibration pool C_cal of 2048 (h,d) pairs:
    h ~ prior over histories (uniform over task instances)
    d ~ Uniform(design space)  # known sampling probability π_probe(d|h) = 1/|D|
compute EIG̃_cal(h,d) via nested Monte Carlo (M=64 inner samples)
fit initial c_φ on C_cal by least squares of EIG̃_cal - EIG_ref_cal on s(h,d),
    where EIG_ref_cal is estimated by held-out likelihood-ratio contrastive estimator
    using q_ψ pretrained on C_cal vs. p̃ samples (see step 3)

for each training round r:
    # 1. Randomized probe pool (the instrument, exogeneity enforced by design)
    sample N=512 held-out histories h ~ buffer of recent rollouts
    for each h: draw K=8 designs d ~ π_probe(·|h) = ε·Uniform(D) + (1-ε)·π_θ(·|h), ε=0.2
        record known sampling probability π_probe(d|h)
    # 2. Plug-in EIG under user likelihood
    for each (h,d): estimate EIG̃(h,d) via nested Monte Carlo (M=64 inner samples)
    # 3. Policy-induced likelihood-ratio test (discriminator trained on probe pool)
    for each (h,d): compute log r(h,d) = log p̃(y|h,d) - log q_ψ(y|h,d)
        where q_ψ is trained to separate (h,d,y) from p̃ vs. from the probe pool
        (ε-mixture), i.e., binary cross-entropy on 4096 samples per round
    # 4. Bias signal: weighted regression of residual EIG̃ - EIG_ref on s(h,d)
    #    EIG_ref estimated by held-out likelihood-ratio contrastive estimator
    #    using q_ψ as the reference (exogenous variation from probe pool)
    fit c_φ(s(h,d)) to residual via weighted least squares with weights 1/π_probe(d|h)
        over the probe pool; update φ every 4 rounds
    # 5. Corrected reward and policy update
    R(h,d) = EIG̃(h,d) - c_φ(s(h,d))
    update θ via policy gradient on R with advantage baseline (REINFORCE, lr=3e-4, batch=32 histories)
    update ψ on discriminator loss (Adam, lr=1e-3) every round
```

**Why this design.** We chose a randomized ε-mixture probe distribution π_probe with known sampling probabilities over a purely policy-induced pool because the policy's query distribution is precisely the exogenous variation the user likelihood cannot explain, but a policy trained on the biased reward will collapse onto designs that exploit the misspecification, violating exogeneity; the ε-mixture and importance weights enforce exogeneity by design, accepting the cost that ε=0.2 reduces the fraction of on-policy designs and adds variance from importance weighting. We chose a low-dimensional correction field c_φ over sufficient statistics s(h,d) (design entropy, history length, posterior spread) rather than a full neural correction over raw (h,d) because a low-dimensional field is identifiable from the probe pool without an oracle, accepting the cost that residual bias orthogonal to s(h,d) is not corrected. We chose to update φ every 4 rounds rather than every step because the residual regression needs a stable probe pool to avoid chasing noise, accepting slower adaptation to policy drift. A domain expert would not describe this as a variant of Step-DAD: Step-DAD refines the policy on observed data but never constructs an instrument to identify reward bias, and its test-time updates assume the likelihood is correct.

**Why it measures what we claim.** The discriminator log-ratio log r(h,d) measures the direction of EIG bias because under the assumption that the probe distribution π_probe is exogenous to the likelihood misspecification (enforced by ε-mixture and known sampling probabilities), the residual EIG̃ - EIG_ref is a linear functional of log r; this assumption fails when the probe pool is not used (i.e., pure policy rollouts), in which case log r reflects policy-induced selection rather than exogenous bias. The correction field c_φ(s(h,d)) measures the magnitude of EIG bias because under the assumption that bias is a smooth function of the sufficient statistics s(h,d) (design entropy, history length, posterior spread), the weighted least-squares fit over the probe pool recovers the bias conditional mean; this assumption fails when bias depends on higher-order interactions not captured by s, in which case c_φ reflects only the projected bias. The held-out EIG_ref measures the debiased target because under the assumption that the contrastive estimator with q_ψ as reference is consistent for the true EIG, the residual is zero at the true reward; this assumption fails when q_ψ is under-trained, in which case EIG_ref reflects the discriminator's approximation error rather than the true EIG.

## Contribution

(1) A self-consistent EIG reward calibration loop that uses the policy's own query distribution as an instrument to identify and correct likelihood-misspecification bias in the amortized design reward, with a low-dimensional correction field trained jointly with the policy. (2) A policy-induced likelihood-ratio test that operationalizes the exogenous variation of the query distribution as an identifiable bias signal, removing the need for an oracle likelihood during training. (3) An empirical demonstration on Twenty Questions and a misspecified clinical dialogue benchmark that the corrected reward reduces the gap between plug-in EIG and true information gain under controlled likelihood perturbations.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Twenty Questions with held-out categories | Sequential EIG with known misspecification structure |
| Primary metric | Realized information gain per query | Directly measures debiased EIG reward quality |
| Baseline | Amortized DAD policy | Tests whether instrumentation beats plain amortization |
| Baseline | Step-DAD | Tests whether instrument identifies bias Step-DAD ignores |
| Baseline | Greedy nested-MC EIG | Tests whether bias correction beats oracle-free plug-in |
| Ablation-of-ours | IEIG without correction field (c_φ≡0) | Isolates contribution of the bias-correction field |
| Ablation-of-ours | IEIG with pure policy-induced pool (ε=0, no importance weights) | Tests whether exogeneity enforcement is necessary |
| Ablation-of-ours | IEIG with fixed calibration pool only (no refit on probe pool) | Tests whether refitting on probe pool adds value |

### Why this setup validates the claim

The central claim is that a policy-induced discriminator instrument plus a low-dimensional correction field recovers the portion of EIG bias that plain amortization cannot see. Twenty Questions is the right testbed because the user likelihood can be deliberately misspecified (e.g., a miscalibrated answer model), producing bias whose direction depends on which designs the policy visits — exactly the regime where the instrument has exogenous variation to exploit. The primary metric, realized information gain per query on held-out categories, is the correct detector because it measures the true reward rather than the plug-in surrogate; if the correction works, realized gain should rise even while plug-in EIG̃ stays flat. Amortized DAD tests the sub-claim that instrumentation beats naive amortization; Step-DAD tests the sub-claim that a constructed instrument identifies bias a semi-amortized method assumes away; greedy nested-MC tests whether correction beats an unbiased-but-expensive plug-in estimate. The ablation c_φ≡0 is the falsification lever: if removing the correction field changes nothing, the instrument alone is doing no work and the mechanism claim collapses. The additional ablation with pure policy-induced pool (ε=0) tests the load-bearing exogeneity assumption: if performance degrades without the ε-mixture and importance weights, the exogeneity enforcement is necessary; if it matches full IEIG, the assumption may hold empirically in this testbed.

### Expected outcome and causal chain

**vs. Amortized DAD** — On a case where the misspecified likelihood systematically over-rewards high-entropy designs, DAD's policy collapses onto those designs and its plug-in reward keeps rising while realized gain stalls, because DAD optimizes the biased surrogate with no signal separating bias from true EIG. Our method instead fits c_φ on the residual against the discriminator-referenced EIG_ref using the randomized probe pool, subtracting the entropy-correlated bias before the policy gradient, so we expect DAD to show a widening gap between plug-in and realized gain while IEIG shows a shrinking one, with the largest divergence on high-entropy query subsets.

**vs. Step-DAD** — On a case where the policy drifts toward designs that exploit misspecification mid-training, Step-DAD refines on observed data under the assumption the likelihood is correct, so its test-time updates reinforce the biased designs and its realized gain plateaus below the debiased target. Our method instead retrains q_ψ each round on the ε-mixture probe pool to track the drifting policy and regresses the residual with importance weights, so the correction follows the drift; we expect Step-DAD to plateau early on misspecified tasks while IEIG continues improving, with parity on well-specified tasks where no bias exists to correct.

**vs. Greedy nested-MC EIG** — On a case requiring many sequential queries, greedy nested-MC pays high per-query cost and, under misspecification, still optimizes the biased plug-in objective, so its realized gain is limited by both budget and bias. Our method amortizes the policy and corrects the reward, so we expect greedy to match IEIG on small budgets but fall behind as horizon grows, with the gap concentrated on later queries where accumulated bias compounds.

### What would falsify this idea

If the gain over Step-DAD and DAD is uniform across well-specified and misspecified subsets rather than concentrated where the predicted bias mechanism operates, the correction field is not doing what we claim. Likewise, if the c_φ≡0 ablation matches full IEIG, the instrument-plus-correction mechanism is not the source of any improvement; and if the ε=0 ablation matches full IEIG, the exogeneity enforcement is unnecessary in this testbed.

## References

1. Amortizing Sequential Experimental Design in Reasoning LLMs
2. Shoot First, Ask Questions Later? Building Rational Agents that Explore and Act Like People
3. Step-DAD: Semi-Amortized Policy-Based Bayesian Experimental Design
4. Amortized Bayesian Experimental Design for Decision-Making
5. PASOA- PArticle baSed Bayesian Optimal Adaptive design
