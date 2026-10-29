# Reward-Likelihood Identifiability Correction for Amortized Bayesian Experimental Design

## Motivation

Amortized sequential BED policies such as those in 'Amortizing Sequential Experimental Design in Reasoning LLMs' are trained against a fixed generative model and evaluated with no feedback channel, so when the deployment likelihood differs from the training model the policy's expected-information-gain scores are silently biased and no mechanism exists to revise them. The meta-gap recurs across the tree: Step-DAD requires gradient updates at test time (infeasible for an LLM policy and non-identifiable from a handful of observations), while the LLM amortization work and the decision-aware TNDP framework both assume deployment model = training model. The root cause is structural: the reward is a plug-in EIG against a generative model that is never checked against live outcomes, so contradicting observations have no path back into the reward.

## Key Insight

Under an exchangeability assumption on calibration queries, the discrepancy between the training generative model and the live likelihood reduces to a low-dimensional parameter that is identifiable from a small number of calibration outcomes, and the policy's own predicted outcome distribution can be reweighted in closed form by the resulting likelihood ratio — so no retraining and no gradient updates are needed.

## Method

**What it is.** RLIC (Reward-Likelihood Identifiability Correction) takes an offline-trained EIG policy π and a small set of K calibration queries with observed outcomes, estimates a scalar (or low-dimensional) discrepancy parameter θ between the training generative model p₀ and the live likelihood, and returns corrected EIG scores for every candidate design the policy proposes. Inputs: policy π, calibration set {(q_k, o_k)}_{k=1..K}, training generative model p₀, discrepancy family {p_θ}. Output: reweighted EIG estimates used to select the next design; the policy weights are frozen.

**Load-bearing assumption (stated explicitly).** The correction is exact only if the policy's outcome head p0(o|d,h) is a faithful estimate of the training generative model's predictive distribution p₀(o|d,h) on the designs the policy proposes, and the true misspecification lies in span(f). We make this assumption explicit and operationalize it with a held-out calibration split: we split the K calibration queries into K_fit (used for θ̂) and K_val (held out), and compute the held-out log-likelihood gap Δ = (1/|K_val|) Σ_{k∈K_val} [log p0(o_k|q_k) − log p_θ̂(o_k|q_k)]. We report Δ as a gating diagnostic; if Δ exceeds a pre-registered threshold δ (set to 0.1 nats per query), we flag the correction as unreliable and fall back to the uncorrected EIG for that episode. This check cannot detect head error on designs proposed after calibration, so we additionally restrict the correction to designs d for which the head's predictive distribution has been verified against p₀ on held-out calibration queries (i.e., designs in the support of the calibration design distribution).

**How it works.**

```pseudocode
# Phase 1: calibration (offline policy frozen)
# Pre-registered: K = 20, f = [log p0(o|q), log p0(o|q)^2, 1{o in top-3 p0 outcomes}], dim(θ)=3
for k in 1..K:
    q_k  <- calibration query (fixed, pre-registered)
    o_k  <- live outcome
# Discrepancy family: p_theta(o|q) = p0(o|q) * exp(theta^T f(q,o)) / Z(theta,q)
# f = low-dim sufficient statistic (e.g., log-likelihood ratio features)
theta_hat <- argmax_theta  sum_{k in K_fit} log p_theta(o_k | q_k)      # MLE, dim(theta)=3 << K_fit=15
# Held-out diagnostic: compute Delta on K_val (size 5)
Delta <- (1/|K_val|) sum_{k in K_val} [log p0(o_k|q_k) - log p_theta_hat(o_k|q_k)]
if Delta > 0.1: flag unreliable; use uncorrected EIG for this episode

# Phase 2: closed-form EIG reweighting (no retraining)
for each candidate design d proposed by pi:
    # policy's own predicted outcome distribution under p0
    p0(o | d, h)  <- pi's outcome head conditioned on history h
    # likelihood-ratio reweighting
    w(o) <- exp(theta_hat^T f(d, o)) / Z(theta_hat, d)
    EIG_corrected(d) <- E_{p0(o|d,h)}[ w(o) * log( p_theta(o|d,h) / p_theta(o|h) ) ]
select d* <- argmax_d EIG_corrected(d)
```

**Why this design.** We chose a low-dimensional exponential-family discrepancy p_θ over full online fine-tuning because the calibration set has K=20 observations while a full policy update has millions of parameters, so only a low-dimensional θ (dim=3) is identifiable from K outcomes; the cost is that θ cannot capture arbitrary misspecification, only the directions spanned by f. We chose closed-form likelihood-ratio reweighting of the policy's own predicted distribution over retraining or importance-sampling the policy because the amortized policy's outcome head already supplies p₀(o|d,h), so no new model is needed; the cost is that the correction is exact only when the policy's outcome head is a faithful estimate of p₀, which we check with a held-out calibration split (Δ ≤ 0.1 nats) and by restricting to designs in the calibration support. We chose a fixed pre-registered calibration set over adaptive calibration because adaptive query selection would confound the discrepancy estimate with the policy's own design choices; the cost is that K must be spent on queries that may be less informative than policy-selected ones. A domain expert would not describe this as Step-DAD: Step-DAD performs stochastic gradient ascent on the policy network at every step, whereas RLIC never touches the policy weights and instead corrects the EIG reward through a closed-form likelihood ratio whose parameter is identifiable from a handful of outcomes.

**Why it measures what we claim.** The MLE θ̂ measures the discrepancy between the training generative model and the live likelihood because the exchangeability assumption (calibration queries are drawn from the same design distribution the policy will face) makes the calibration likelihood equal the deployment likelihood; this assumption fails when the deployment design distribution shifts after calibration, in which case θ̂ reflects a stale discrepancy. The sufficient statistic f(d,o) measures the direction of misspecification that matters for EIG because f is chosen to span the log-likelihood-ratio space between p₀ and p_θ; this assumption fails when the true misspecification lies outside span(f), in which case the correction is a projection and under-corrects. The reweighted EIG_corrected(d) measures the true information gain under the live likelihood because the likelihood ratio w(o) is the exact Radon–Nikodym derivative between p_θ and p₀ under the exponential-family form; this assumption fails when p₀ assigns near-zero mass to an outcome that p_θ assigns positive mass, in which case the ratio is unstable and we clip w(o) at a fixed cap (w_max=10). The held-out diagnostic Δ measures the faithfulness of the policy's outcome head to p₀ because Δ is the average log-likelihood gap on held-out calibration queries; this assumption fails when the head is miscalibrated off-distribution, in which case Δ exceeds 0.1 nats and we flag the correction as unreliable.

## Contribution

(1) A deployment-time correction mechanism, RLIC, that pairs a frozen amortized EIG policy with a low-dimensional identifiable discrepancy parameter estimated from a small calibration set and propagates it into EIG scores via closed-form likelihood-ratio reweighting, requiring no retraining or gradient updates. (2) A design principle: the discrepancy parameter must be low-dimensional relative to the calibration budget for identifiability, and the sufficient statistic must span the misspecification directions relevant to EIG, which we operationalize as an explicit choice of exponential-family features. (3) A calibration protocol with a fixed pre-registered query set and a held-out split to check the faithfulness of the policy's outcome head, which makes the correction auditable.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Twenty Questions with held-out category shift | Tests EIG under deployment misspecification |
| Primary metric | Expected information gain per query | Directly measures the quantity RLIC corrects |
| Baseline | Frozen amortized EIG policy (DAD-style) | Isolates benefit of correction without retraining |
| Baseline | Step-DAD semi-amortized online update | Tests whether closed-form beats gradient adaptation |
| Baseline | Inference-time Monte Carlo BED | Tests whether amortization plus correction suffices |
| Baseline | Naive full importance-sampling correction | Isolates benefit of low-dimensional θ vs full IS |
| Ablation-of-ours | RLIC with θ fixed to zero | Confirms gain comes from calibration, not reweighting |
| Ablation-of-ours | RLIC with head replaced by explicit p₀ | Tests sensitivity to head faithfulness |
| Diagnostic | Held-out calibration log-likelihood gap Δ | Measures head faithfulness to p₀ |

### Why this setup validates the claim

The central claim is that a low-dimensional likelihood-ratio correction, estimated from a handful of calibration outcomes (K=20), recovers most of the EIG lost to model misspecification without touching policy weights. The frozen DAD-style policy tests the null hypothesis: if correction adds nothing, RLIC matches it exactly. Step-DAD tests the competing mechanism: it adapts via gradient ascent on millions of parameters, so if RLIC closes the same gap with K=20 outcomes, the low-dimensional-identifiability argument is vindicated; if Step-DAD dominates everywhere, that argument fails. Monte Carlo BED tests whether amortization is even the bottleneck. The naive full importance-sampling correction baseline tests whether the low-dimensional exponential-family form is necessary or whether a full nonparametric correction would suffice; if RLIC matches it, the low-dimensional restriction is not a limitation. The θ=0 ablation is the decisive control: any gain must vanish when the discrepancy estimate is disabled, otherwise the reweighting itself, not calibration, is doing the work. The head-replacement ablation tests the load-bearing assumption: if replacing the policy's outcome head with an explicit p₀ model changes results substantially, the head faithfulness assumption is critical. The held-out diagnostic Δ is reported for every episode to gate the correction: episodes with Δ > 0.1 nats are flagged and the correction is disabled, so the reported EIG gain is conditioned on reliable head faithfulness. Expected information gain per query is the right metric because it is exactly the quantity EIG_corrected estimates, so a correction that measures what it claims must move this metric, and a spurious one will not.

### Expected outcome and causal chain

**vs. Frozen amortized EIG policy (DAD-style)** — On a case where deployment categories shift the outcome likelihood away from p₀, the baseline keeps proposing questions that are informative under the stale training model, so it wastes queries on designs whose answers it already predicts well. Our method instead reweights each candidate's EIG by the calibrated likelihood ratio, up-weighting designs whose outcomes are unexpectedly informative under the live likelihood, because θ̂ captures the shift direction from K=20 calibration outcomes. We expect a clear EIG gap concentrated on shifted-category episodes, with near-parity on in-distribution episodes.

**vs. Step-DAD semi-amortized online update** — On a case where the shift is real but low-dimensional, Step-DAD spends its gradient steps chasing high-variance policy updates from few live outcomes, producing noisy design choices early in deployment. Our method instead estimates a single identifiable θ (dim=3) and applies an exact closed-form reweighting, because the exponential-family form makes the Radon–Nikodym derivative analytic. We expect RLIC to match Step-DAD's asymptotic EIG while reaching it with far fewer live outcomes (K=20 vs. hundreds of gradient steps), and to be less variable across episodes during the early deployment window.

**vs. Inference-time Monte Carlo BED** — On a case requiring many candidate evaluations, Monte Carlo BED estimates EIG by sampling outcomes per design, so its estimates are noisy and its query selection thrashes between near-tied designs. Our method instead reuses the policy's outcome head as p₀ and applies a deterministic reweighting, because the correction is closed-form given θ̂. We expect RLIC to show lower variance in selected-design EIG and comparable mean, with the advantage growing as the candidate pool grows.

**vs. Naive full importance-sampling correction** — On a case where the true misspecification is high-dimensional, full importance-sampling correction uses all calibration outcomes to reweight without a low-dimensional parameterization, but it suffers from high variance and overfitting with K=20. Our method instead projects the correction onto a 3-dimensional sufficient statistic f, because the exponential-family form reduces variance at the cost of potential under-correction. We expect RLIC to have lower variance and comparable or better mean EIG than full IS when K is small, and full IS to catch up only when K is large (e.g., K>100).

### What would falsify this idea

If RLIC's gain over the frozen policy is uniform across shifted and in-distribution episodes rather than concentrated where the calibrated discrepancy is largest, the correction is not tracking misspecification and the central claim is wrong. Equally fatal: if the θ=0 ablation reproduces the full method's gain, the benefit comes from reweighting mechanics rather than from identifiable calibration. Additionally, if the head-replacement ablation shows that replacing the policy's outcome head with an explicit p₀ model eliminates the gain, the load-bearing assumption of head faithfulness is violated and the method's correction is not reliable.

## References

1. Amortizing Sequential Experimental Design in Reasoning LLMs
2. Shoot First, Ask Questions Later? Building Rational Agents that Explore and Act Like People
3. Step-DAD: Semi-Amortized Policy-Based Bayesian Experimental Design
4. Amortized Bayesian Experimental Design for Decision-Making
5. PASOA- PArticle baSed Bayesian Optimal Adaptive design
