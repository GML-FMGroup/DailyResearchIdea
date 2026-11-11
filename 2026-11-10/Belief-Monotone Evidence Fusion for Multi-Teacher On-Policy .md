# Belief-Monotone Evidence Fusion for Multi-Teacher On-Policy Distillation

## Motivation

Multi-teacher distillation for agentic visual reasoning (e.g., Reframe's OG-MOPD) fuses teacher guidance by normalizing each teacher's evidence-aware reward independently, so the fused signal is not a function of the student's evolving posterior over latent visual facts — removing or repairing a visual observation perturbs inputs but never changes the fusion rule itself. Structurally, this means the distillation target can become less informative exactly when the student's belief becomes more accurate, violating the monotone-consistency property that on-policy distillation implicitly assumes when it trusts teacher evidence. The root cause is that independent per-teacher normalization treats teacher evidence as conditionally independent given the input, discarding the joint likelihood structure that observation history actually provides.

## Key Insight

If the fused teacher guidance is constructed as a potential-based shaping term whose value is monotone in the log-likelihood of the student's observation history, then any belief refinement strictly cannot decrease the fused advantage, so the fusion inherits the monotonicity of a Bayesian posterior update rather than the arbitrariness of per-teacher normalization.

## Method

**What it is.** BEMF (Belief-Monotone Evidence Fusion) is a distillation fusion rule that replaces independent per-teacher normalization in OG-MOPD-style multi-teacher on-policy distillation. Inputs: a rollout with observation history $o_{1:t}$, per-teacher evidence-aware rewards $r_k$ from $K$ specialist teachers, and the student's current belief. Outputs: a single fused advantage $A^{\text{fuse}}_t$ used for the policy-gradient update.

**Load-bearing assumption (stated explicitly).** BEMF assumes that the closed-form categorical Bayes posterior $b_t[f^*]$ over an enumerable latent-fact vocabulary $F$, computed from tool-returned observations, is a calibrated sufficient statistic whose log-posterior increments $dLL_t$ track genuine belief refinement. This assumption is load-bearing because $\Phi_t = \beta LL_t$ is used as a state-only potential and the monotone floor constrains $A^{\text{fuse}}_t$ via $dLL_t$. It fails when (i) the true fact is outside $F$ (vocabulary-coverage drift), (ii) the posterior is small-sample noisy, or (iii) teacher posteriors are miscalibrated (Guo et al. 2017). We therefore add a calibration step (Phase 0) and a vocabulary-coverage diagnostic, and we replace the history-dependent floor with a potential-consistent monotone link (Phase 3).

**How it works.**

```pseudocode
# Phase 0: Per-teacher calibration of b_t[f*] (Guo et al. 2017)
# Fit on a held-out calibration set of 512 rollouts with ground-truth fact labels
for each teacher k:
    fit temperature T_k by minimizing NLL of b_t^{(k)}[f*] on calibration set
    b_t^{(k)}[f*] <- sigmoid( logit(b_t^{(k)}[f*]) / T_k )   # calibrated posterior mass
# Report vocabulary-coverage diagnostic: fraction of calibration facts in F

# Phase 1: Belief-state estimator (per rollout)
initialize b_0 = uniform over latent visual facts F
for t = 1..T:
    # observation o_t is tool-returned image/video crop
    b_t = BayesUpdate(b_{t-1}, o_t)          # closed-form categorical posterior
    LL_t = log b_t[f*]                       # log-likelihood proxy = log posterior mass on realized fact
    dLL_t = LL_t - LL_{t-1}                  # belief refinement increment

# Phase 2: Joint evidence-weighted fusion (replaces per-teacher norm)
for each teacher k:
    w_k = softmax_k( log b_t^{(k)}[f*] )     # teacher k's calibrated posterior mass on realized fact
    # joint weight = product of per-teacher likelihoods, NOT independent norm
    W = prod_k ( b_t^{(k)}[f*] )^{1/K}
    r_fuse = sum_k w_k * r_k

# Phase 3: Monotone consistency via potential-based shaping with monotone link
Phi_t = beta * LL_t                            # potential = beta * log-likelihood
# potential-based shaping form (Ng et al. 1999)
A_fuse_t = r_fuse + (gamma * Phi_{t+1} - Phi_t)
# enforce monotonicity through a monotone link on the potential, not a history-dependent floor
# replace hard floor with: A_fuse_t <- A_fuse_t + eta * max(0, dLL_t)
# this keeps Phi_t state-only and preserves the optimal policy class
A_fuse_t = A_fuse_t + eta * max(0, dLL_t)      # monotone link, eta = 1.0

# Phase 4: policy gradient
loss = -sum_t log pi(a_t | o_{1:t}) * A_fuse_t
```

Hyperparameters: $\beta = 0.5$ (potential scale), $\gamma = 0.99$ (discount), $\eta = 1.0$ (monotone link slope), $K$ = number of specialist teachers (2 in Reframe: spatial, temporal), calibration set size = 512 rollouts.

**Why this design.** We chose a closed-form categorical Bayes update over a learned belief head because the latent visual facts in agentic visual reasoning (object presence, spatial relation, temporal order) are discrete and enumerable per observation, so the posterior is exactly computable and needs no approximation; the cost is that the fact vocabulary must be specified per benchmark. We chose a product-of-likelihoods joint weight $W$ over independent per-teacher softmax normalization because independent normalization is precisely the structural defect in Reframe's OG-MOPD — it treats teacher evidence as conditionally independent given the input, which discards the shared observation history; the cost is that a single miscalibrated teacher can dominate the product, which we mitigate by the $1/K$ geometric mean and per-teacher temperature calibration. We chose potential-based shaping $\gamma\Phi_{t+1}-\Phi_t$ over a direct reward bonus because potential-based shaping is the unique form that preserves the optimal policy class (Ng et al. 1999), so monotone belief refinement cannot create a spurious incentive; the cost is that the potential must be a function of state only, which forces us to define belief over observable facts rather than hidden ones. We replaced the history-dependent monotone floor with a monotone link $\eta \max(0, dLL_t)$ added to the shaped advantage, because the floor violated the state-only requirement of Ng et al. 1999; the link keeps the shaping potential-consistent while still enforcing that $A^{\text{fuse}}_t$ is non-decreasing in $dLL_t$. A domain expert would NOT describe BEMF as a variant of Reframe's OG-MOPD, because OG-MOPD normalizes each teacher's evidence independently and repairs observations, whereas BEMF couples teachers through a shared joint likelihood and constrains the fused advantage to be monotone in the student's own posterior — a coupling OG-MOPD does not produce.

**Why this measures what we claim.** The computational quantity `dLL_t` (change in log posterior mass on the realized fact) measures *belief refinement* because the Bayes-update assumption makes posterior mass a sufficient statistic for the student's information about the latent fact; this assumption fails when the fact vocabulary omits the true fact, in which case `dLL_t` reflects vocabulary-coverage drift rather than genuine refinement. The joint weight `W = prod_k (b_t^{(k)}[f*])^{1/K}` measures *joint teacher evidence* because the conditional-independence-given-history assumption makes the product equal to the joint likelihood of the observation under all teachers; this assumption fails when teachers share a correlated visual encoder, in which case `W` overcounts redundant evidence. The potential `Phi_t = beta * LL_t` measures *the value of the student's current belief* because potential-based shaping theory guarantees that a state-only potential preserves the optimal policy class; this assumption fails when the belief is non-Markovian (history-dependent beyond $b_t$), in which case `Phi_t` reflects a truncated sufficient statistic. The monotone link `A_fuse_t = A_fuse_t + eta * max(0, dLL_t)` measures *monotone consistency* because the link is a state-only function of `dLL_t` that is non-decreasing in `dLL_t`; this assumption fails when `dLL_t` is noisy (small-sample posterior), in which case the link reflects noise propagation rather than consistency.

## Contribution

(1) A joint evidence-weighted fusion rule that replaces independent per-teacher normalization with a product-of-posteriors geometric mean coupled through a shared observation history, yielding a fused advantage that is a function of the student's belief rather than of each teacher's marginal. (2) A monotone consistency constraint implemented as potential-based shaping on the log-posterior, which structurally forbids the fused advantage from decreasing when the student's belief refines — a property independent per-teacher normalization cannot express. (3) A closed-form categorical belief-state estimator over latent visual facts that makes the fusion rule exactly computable per rollout without a learned approximation head.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | Reframe's 14 image/video benchmarks | Same suite isolates fusion-rule effect |
| Primary metric | Accuracy on belief-refinement-sensitive subsets | Directly tracks monotone posterior gain |
| Baseline | Reframe OG-MOPD | Tests independent-normalization defect |
| Baseline | Mix-RL | Tests multi-teacher fusion without belief coupling |
| Baseline | Uniform-mean teacher fusion | Tests whether weighting scheme matters at all |
| Ablation-of-ours | BEMF minus monotone link | Isolates potential-shaping contribution |
| Ablation-of-ours | BEMF minus per-teacher calibration | Tests calibration assumption |
| Ablation-of-ours | BEMF with independent per-teacher normalization (no joint W) | Tests joint-likelihood coupling |

### Why this setup validates the claim

The central claim is that coupling teachers through a shared joint likelihood and constraining the fused advantage to be monotone in the student's own posterior yields gains that independent per-teacher normalization cannot. The Reframe OG-MOPD baseline is the sharpest test because it shares the exact same teachers, rollouts, and observation-repair machinery, so any gap is attributable to the fusion rule rather than to teacher quality or data. Mix-RL tests the weaker sub-claim that belief-aware weighting at all beats naive multi-teacher mixing. Uniform-mean fusion tests whether the product-of-likelihood weight W carries information beyond simply averaging teachers. The ablation of the monotone link tests the shaping sub-claim separately from the weighting sub-claim, so a reader can see which mechanism drives the effect. The calibration ablation tests whether the load-bearing calibration assumption is necessary for the monotone link to help. The independent-normalization ablation tests whether the joint W is doing work beyond per-teacher softmax. The primary metric is restricted to belief-refinement-sensitive subsets because that is precisely where dLL_t is non-trivial; on subsets where the latent fact is trivially observable at t=1, no fusion rule should differ, and including those would dilute the falsifiable signal. We also report a vocabulary-coverage diagnostic (fraction of calibration facts in F) to show how much of the budget is spent on vocabulary engineering versus the fusion comparison.

### Expected outcome and causal chain

**vs. Reframe OG-MOPD** — On a case where the spatial teacher's posterior on the realized fact is sharp but the temporal teacher's is diffuse, OG-MOPD normalizes each teacher's evidence independently and then averages, so the diffuse teacher contributes nearly as much as the sharp one and the fused advantage fails to track the student's actual belief refinement. Our method instead weights teachers by their calibrated posterior mass on the realized fact through the joint product W, so the sharp teacher dominates and the fused advantage rises with dLL_t. We expect a noticeable accuracy gap concentrated on multi-teacher-conflict subsets, with parity on subsets where both teachers agree.

**vs. Mix-RL** — On a case where a rollout requires chaining two observations to resolve a spatial relation, Mix-RL mixes teacher rewards without any belief-state coupling, so the advantage assigned to the resolving action is diluted across the chain and the policy learns a delayed, weak credit assignment. Our method instead propagates belief refinement through the potential Phi_t = beta*LL_t, so the action that resolves the fact receives a shaped advantage proportional to its dLL_t. We expect a gap that grows with rollout length and is near-zero on single-observation cases.

**vs. Uniform-mean teacher fusion** — On a case where one teacher is miscalibrated (overconfident on a wrong fact), uniform averaging lets that teacher's error pull the fused advantage away from the student's true posterior trajectory, producing updates that move the policy opposite to belief refinement. Our method's geometric-mean joint weight W and monotone link suppress the miscalibrated teacher and refuse updates where dLL_t > 0 but A_fuse would drop. We expect uniform-mean to show accuracy degradation specifically on miscalibration-heavy subsets and BEMF to hold parity there while still gaining on conflict subsets.

**vs. BEMF minus monotone link** — On a case where dLL_t is large and positive (strong belief refinement), removing the monotone link means the shaped advantage can decrease if r_fuse is negative, so the policy update moves opposite to belief refinement. Our method adds eta*max(0, dLL_t) to the advantage, ensuring A_fuse_t is non-decreasing in dLL_t. We expect the ablation to underperform on belief-refinement-sensitive subsets, with parity elsewhere.

**vs. BEMF minus per-teacher calibration** — On a case where a teacher is overconfident (miscalibrated), uncalibrated b_t[f*] yields noisy dLL_t and a distorted W, so the monotone link and joint weight propagate miscalibration. Our method calibrates each teacher's posterior via temperature scaling, making dLL_t a calibrated log-likelihood increment. We expect the calibration ablation to show degraded accuracy on miscalibration-heavy subsets, confirming the load-bearing assumption.

**vs. BEMF with independent per-teacher normalization (no joint W)** — On a case where teachers disagree, independent normalization treats evidence as conditionally independent given the input, discarding shared history, so the fused advantage fails to reflect joint likelihood. Our method uses the product-of-likelihoods W, coupling teachers through shared history. We expect the independent-normalization ablation to lose the multi-teacher-conflict advantage, isolating the joint-coupling contribution.

### What would falsify this idea

If BEMF's gain over OG-MOPD is uniform across all subsets rather than concentrated on the belief-refinement-sensitive and multi-teacher-conflict subsets, the coupling mechanism is not doing the work and the claim is wrong. Likewise, if removing the monotone link or per-teacher calibration changes nothing on subsets where dLL_t is large but noisy, the shaping and calibration sub-claims are unsupported.

## References

1. Reframe: Observation-Aligned Multi-Teacher On-Policy Distillation for Agentic Visual Reasoning
