# Cross-Platform Behavioral Equivalence Regularization for Distilled User Simulators

## Motivation

UserArena certifies its on-policy trajectory distillation (OPTD) simulator with a single-platform production A/B test, so trajectory-level fidelity is established only relative to one e-commerce catalog, one reward-shaping regime, and one user population; the shared assumption across this paper-tree and its neighbors (KuaiSim, SUBER) is that one platform's real-trajectory corpus plus one A/B test suffices to certify a simulator. This is a structural failure, not a data-volume failure: the distillation objective has no term that ties the simulated action distribution to human behavior under catalog or reward-shaping shift, so any fidelity it reports is platform-relative and cannot transfer. Because the simulator is the evaluation substrate for downstream recommender agents, this platform-relativity silently contaminates every long-horizon benchmark result that reuses it.

## Key Insight

Behavioral equivalence across platforms is a property of the *invariant* part of the action-conditioned transition kernel — the mapping from (user intent state, catalog context) to click/purchase action — which is shared across platforms, while the marginal action frequencies are platform-specific nuisance; a domain-adversarial objective on the distilled simulator's intent-state representation, coupled with a trajectory-level equivalence regularizer, isolates the shared kernel so the simulated action distribution stays stable under catalog and reward-shaping shifts.

## Method

**What it is.** We introduce **XPIRE** (Cross-Platform Invariant Regularized Emulation), a training objective for an on-policy trajectory-distilled user simulator. Inputs: a source-platform real-trajectory corpus with an OPTD teacher (UserArena), a target-platform unlabeled interaction corpus, a catalog/reward-shaping shift operator, and a small labeled intent probe (512 examples) used to verify that the shift operator preserves intent. Output: a simulator policy whose click/purchase action distribution is stable across platforms.

**How it works.**

```pseudocode
XPIRE(student S, teacher T, source corpus D_s, target corpus D_t, catalog shift op C, reward-shaping op R, lambda_adv=0.1, lambda_eq=0.5, lambda_mis=0.5, T_rounds=10):
  # 0. Intent-preservation calibration for C (one-time, before training)
  #    Use a held-out labeled intent probe P of 512 (context, intent) pairs from D_s.
  #    For each (x, y) in P, apply C to the catalog context: x' = C(x).
  #    Compute teacher intent posterior drift: delta_intent = mean KL( T.intent_target(x) || T.intent_target(x') ).
  #    If delta_intent > tau (tau=0.05 nats), reject C and re-parameterize C (e.g., reduce shift magnitude) until delta_intent <= tau.
  #    Store the calibrated C and the probe P for use in L_mis below.

  for round in 1..T_rounds:
    # 1. On-policy trajectory distillation (OPTD, unchanged)
    tau_s ~ S.rollout(contexts from D_s)          # student generates long-horizon sessions
    L_distill = sum_t KL( T.intent_target(tau_s_t) || S.action_logits(tau_s_t) )

    # 2. Cross-platform behavioral equivalence regularizer (intent-matched)
    #    For each context x in D_s, sample a matched intent label y ~ T.intent_target(x).
    #    Construct C-shifted context x' = C(x) and target context x_t from D_t with the same intent y (via propensity matching on teacher intent posterior).
    tau_s_shift ~ S.rollout( {x'} )   # same intents, shifted catalog
    tau_t       ~ S.rollout( {x_t} )   # target-platform contexts, matched intent
    # action-distribution equivalence on the *purchase/click* head only:
    L_eq = MMD_k( S.action_dist(tau_s_shift, head=click_purchase),
                  S.action_dist(tau_t, head=click_purchase) )   # kernel MMD, RBF kernel, bandwidth = median pairwise distance

    # 3. Explicit intent-mismatch term (estimated and subtracted)
    #    For the same matched pairs (x', x_t), compute the teacher's intent posterior divergence:
    L_mis = mean KL( T.intent_target(x') || T.intent_target(x_t) )
    #    This term is subtracted from L_eq to isolate kernel mismatch:
    L_eq_kernel = L_eq - lambda_mis * L_mis

    # 4. Domain-adversarial invariance on the intent-state encoder
    h_s = S.encode_intent(tau_s) ; h_t = S.encode_intent(tau_t)
    L_adv = - sum log D_phi(h_s) - sum log(1 - D_phi(h_t))   # D_phi: 2-layer MLP discriminator, hidden=256, ReLU
    L_inv = lambda_adv * L_adv

    S.update( L_distill + lambda_eq * L_eq_kernel + L_inv )
    D_phi.update( -L_adv )   # alternating, gradient-reversal equivalent
  return S
```

**Why this design.** We chose an *explicit* MMD equivalence penalty on the click/purchase head over a purely adversarial domain classifier because the adversarial signal on a high-capacity encoder is known to collapse to marginal-frequency matching (the failure mode of DANN-style training on recommendation data), accepting the cost that MMD requires a kernel bandwidth choice and is O(n^2) per batch. We chose to apply the adversarial term to the *intent-state encoder* rather than to the raw action logits because the intent state is the sufficient statistic for the shared kernel, accepting the cost that the encoder must be identifiable enough to separate intent from catalog nuisance — which is why we also feed the catalog shift operator C as a controlled intervention rather than relying on naturally occurring platform differences. We chose alternating gradient-reversal over a single joint loss because the discriminator must be trained to optimality for the invariance signal to be informative, accepting the cost of a second optimizer state and a hyperparameter (lambda_adv=0.1) that trades invariance strength against distillation fidelity. A domain expert would not describe XPIRE as a variant of DANN or of CORAL because those methods align *marginal* feature distributions; XPIRE aligns the *action-conditional* kernel under a *controlled catalog intervention* C, which is the object that determines simulator fidelity, not the marginal intent distribution. To make the intent-preserving assumption explicit: we assume that C is an intervention on catalog context that holds user intent fixed, and we verify this assumption on a 512-example labeled probe by requiring teacher intent-posterior drift ≤ 0.05 nats; if the drift exceeds this threshold, C is re-parameterized (e.g., by reducing the magnitude of price-tier substitution) until the check passes. The equivalence target is therefore the action-conditional distribution p(action | intent, catalog) estimated on matched intents, not the marginal action distribution under C.

**Why it measures what we claim.** The MMD term `L_eq` measures cross-platform behavioral equivalence because the assumption that the click/purchase action kernel is a function of intent state and catalog context, and that C holds intent fixed while varying catalog (verified by the intent-preservation probe with drift ≤ 0.05 nats), makes equality of action distributions under C equivalent to equality of the kernel; this assumption fails when C also perturbs intent (e.g., catalog shift changes what users want), in which case `L_eq` reflects residual intent mismatch rather than kernel mismatch — which is why we explicitly estimate and subtract the intent-mismatch term `L_mis` (teacher KL between matched pairs) to form `L_eq_kernel`. The adversarial term `L_adv` on `h = S.encode_intent` measures invariance of the intent representation because the assumption that platform identity is *not* predictive of intent given the shared kernel makes an optimal discriminator's loss a sufficient statistic for platform leakage; this assumption fails when the target corpus has systematically different intent priors (population shift), in which case `L_adv` reflects prior mismatch rather than kernel mismatch. The distillation term `L_distill` measures long-horizon fidelity because the assumption that the OPTD teacher's trajectory-level intent targets are platform-agnostic makes KL on student rollouts a valid fidelity signal; this assumption fails when the teacher was fit on source-platform reward shaping, in which case `L_distill` reflects source-platform reward bias. Each component is therefore a named proxy with a stated failure mode, not an unstated equivalence.

## Contribution

(1) XPIRE, a cross-platform invariant training objective for on-policy trajectory-distilled user simulators that couples a controlled catalog-shift intervention, an action-kernel MMD regularizer, and an intent-encoder domain-adversarial term. (2) The design principle that simulator fidelity certification requires an *invariant* action-conditional kernel rather than marginal action-frequency matching, operationalized by the controlled intervention C. (3) A multi-platform A/B validation protocol that replaces UserArena's single-platform A/B certification with a held-out-platform transfer suite.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Source UserArena corpus + target-platform corpus with catalog shift, plus a 512-example labeled intent probe for C calibration | Provides controlled intervention C with verified intent preservation |
| Primary metric | Click/purchase action-distribution MMD on held-out target intents, after subtracting the estimated intent-mismatch term L_mis | Directly measures the kernel equality we claim |
| Baseline | OPTD (unregularized on-policy distillation) | Tests whether cross-platform regularization is needed at all |
| Baseline | DANN-style domain-adversarial alignment on action logits | Tests marginal-adversarial alternative to action-conditional MMD |
| Baseline | CORAL feature-alignment on intent encoder | Tests second-order marginal alignment alternative |
| Ablation-of-ours | XPIRE without controlled catalog shift operator C | Tests whether the intervention, not natural shift, drives equivalence |
| Ablation-of-ours | XPIRE without the intent-mismatch subtraction term L_mis | Tests whether explicit intent-mismatch correction is necessary |
| Diagnostic | Intent-posterior drift under C measured on the 512-example probe, and correlation between drift and residual L_eq gap | Tests the C-holds-intent-fixed assumption empirically |

### Why this setup validates the claim

The central claim is that aligning the action-conditional click/purchase kernel under a controlled catalog intervention C yields a simulator whose purchase behavior transfers across platforms, and that marginal-feature alignment methods cannot achieve this. The dataset pairing is what makes the claim testable: source trajectories give the OPTD teacher's intent targets, while the target corpus plus C gives matched-intent, shifted-catalog conditions, so any residual action-distribution gap is attributable to kernel mismatch rather than intent mismatch. The 512-example labeled intent probe provides a direct check that C preserves intent: we measure teacher intent-posterior drift and require it to be ≤ 0.05 nats; if the drift exceeds this threshold, C is re-parameterized until the check passes. Additionally, we explicitly estimate and subtract the intent-mismatch term L_mis (teacher KL between matched pairs) from L_eq, so the primary metric L_eq_kernel isolates kernel mismatch even if C induces small residual intent drift. OPTD tests the sub-claim that distillation alone leaves a cross-platform gap; the DANN-style and CORAL baselines test the sub-claim that marginal alignment is insufficient because it matches frequencies rather than the action-conditional kernel. The ablation removing C tests whether the controlled intervention is doing the causal work or whether naturally occurring platform differences suffice. The ablation removing L_mis tests whether explicit intent-mismatch correction is necessary. The primary metric — MMD on the click/purchase head evaluated on held-out target intents, after subtracting L_mis — is the right detector because the claim is precisely equality of that kernel; a metric on marginal intent features or overall reward would be insensitive to the predicted failure. If XPIRE's advantage is real, it should appear as a metric gap concentrated exactly where C-induced catalog shift is largest and vanish where C is identity, which is the falsifiable signature the design is built to expose. The diagnostic correlating intent-posterior drift with residual L_eq gap further tests whether the kernel-mismatch interpretation is confounded by intent drift.

### Expected outcome and causal chain

**vs. OPTD** — On a case where the target platform's catalog presents substitute items at different price tiers, unregularized OPTD produces a simulator that reproduces source-platform purchase frequencies and over-clicks the source-favored tier because its distillation loss never sees the shifted catalog's action consequences. Our method instead rolls out the student under C and penalizes MMD between shifted-source and target click/purchase distributions, forcing the kernel to condition on catalog context rather than memorize source frequencies. We expect OPTD to show a large MMD gap on high-shift target intents and XPIRE to close most of it, with near-parity on low-shift intents.

**vs. DANN-style domain-adversarial alignment** — On a case where the target corpus has a different item-popularity marginal but the same intent-conditioned purchase kernel, the DANN baseline produces a simulator that matches marginal click frequencies by collapsing the intent encoder toward popularity statistics, because its discriminator only sees marginal feature distributions and cannot distinguish kernel equality from marginal matching. Our method instead applies MMD on the action-conditional click/purchase head under C, so marginal popularity differences are absorbed while the conditional kernel is aligned. We expect DANN to look competitive on marginal click-rate metrics but to show a clear MMD gap on the conditional head, especially on rare-intent subsets where marginal matching is least informative.

**vs. CORAL feature-alignment** — On a case where catalog shift changes the second-order statistics of item features without changing intent-conditioned preferences, CORAL produces a simulator whose intent representation is decorrelated from catalog nuisance but whose purchase head still drifts, because second-order feature matching constrains the encoder, not the action kernel. Our method instead leaves the encoder free beyond the adversarial term and directly regularizes the click/purchase head via MMD under C. We expect CORAL to reduce encoder-level distribution distance while leaving a persistent action-head MMD gap on high-shift subsets, whereas XPIRE should show the reverse pattern: modest encoder alignment but markedly lower action-head MMD.

### What would falsify this idea

If XPIRE's MMD advantage over OPTD is uniform across target intents rather than concentrated on the high-catalog-shift subset where C is largest, or if the ablation removing C performs identically to full XPIRE, or if the intent-posterior drift under C exceeds 0.05 nats and correlates strongly with the residual L_eq gap (indicating that L_eq reflects intent mismatch rather than kernel mismatch), then the controlled-intervention mechanism is not what produces the gain and the central claim is wrong.

## References

1. UserArena: Benchmarking Interactive User Simulation for Long-Horizon Shopping Agents
2. SUBER: An RL Environment with Simulated Human Behavior for Recommender Systems
3. Do LLMs Understand User Preferences? Evaluating LLMs On User Rating Prediction
4. KuaiSim: A Comprehensive Simulator for Recommender Systems
5. Generative Agents: Interactive Simulacra of Human Behavior
