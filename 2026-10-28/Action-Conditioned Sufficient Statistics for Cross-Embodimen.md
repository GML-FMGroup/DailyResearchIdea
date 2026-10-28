# Action-Conditioned Sufficient Statistics for Cross-Embodiment Physical Tokens

## Motivation

Existing VLA models such as PDRVLA train physical tokens by predicting future RGB, depth, and optical-flow features under a single embodiment and simulator, so the token space silently absorbs embodiment-specific kinematics and simulator-specific appearance/contact statistics. This nuisance is structural: because the supervision target is a fixed feature distribution rather than the action-conditioned transition operator, no amount of data augmentation or independence penalty can remove it—the token is not forced to be a sufficient statistic for dynamics, and thus it remains free to encode the nuisance.

## Key Insight

A token that is a sufficient statistic for the action-conditioned transition operator under a learned embodiment-conditional reparameterization is forced to quotient out embodiment and simulator nuisance, because any residual nuisance would violate sufficiency for the dynamics rather than merely correlate with the label.

## Method

**What it is.** We introduce ACSS (Action-Conditioned Sufficient Statistics), a training framework for the Physical Dynamics Module (PDM) in PDRVLA-style VLAs. The PDM maps vision-language features to a physical token `z`. ACSS trains `z` so that it is a sufficient statistic for the action-conditioned next-state transition, while being uninformative about the embodiment/simulator label—but only after a learned reparameterization absorbs nuisance factors. **Load-bearing assumption (stated explicitly):** the action-conditioned transition operator is shared across embodiments/simulators after the learned reparameterization `u = g_phi(z, e)`, so that sufficiency for this operator forces `u` to quotient out embodiment/simulator nuisance rather than encoding embodiment-specific dynamics. This assumption fails when the transition operator itself is embodiment-dependent (e.g., different contact dynamics), in which case the likelihood reflects embodiment-specific dynamics rather than shared dynamics. To make the assumption testable, we factor the transition into a shared dynamics operator and an embodiment-specific kinematic/contact operator, and assert sufficiency/invariance only on the shared factor: the transition head is embodiment-conditional, `p_psi(o_{t+1} | u, a_t, e)`, and sufficiency is required for the *residual* dynamics after conditioning on `e`. The adversarial head is evaluated on the `e`-conditional residual `r = u - E[u | e]` (computed via a learned linear probe `h_xi(e)` trained to predict `u` from `e`), not on `u` directly.

**How it works.**

```pseudocode
Input: batch of transitions (o_t, a_t, o_{t+1}) from multiple embodiments/simulators; embodiment label e, simulator label s.
PDM: z = f_theta(o_t, a_t)  # physical token; f_theta is a 2-layer MLP (hidden=256, GeLU) on concatenated [vision-language feature; action embedding]
Reparameterization: u = g_phi(z, e)  # embodiment-conditional; g_phi is a 2-layer MLP (hidden=256, GeLU) taking [z; one-hot(e)]
Embodiment-conditional transition head: p_psi(o_{t+1} | u, a_t, e)  # predicts action-conditioned next state; p_psi is a 3-layer MLP (hidden=512, GeLU) taking [u; a_t; one-hot(e)]
Residual probe: h_xi(e) -> u_hat  # linear probe trained to predict u from e; residual r = u - u_hat
Adversarial head: c_omega(r) -> (e, s)  # tries to recover embodiment/simulator from the e-conditional residual

Phase 1 (sufficiency): train f_theta, g_phi, p_psi to minimize negative log-likelihood of o_{t+1} given u, a_t, e: L_suff = -log p_psi(o_{t+1} | u, a_t, e).
Phase 2 (invariance by identifiability): freeze g_phi; train c_omega to predict (e, s) from r with cross-entropy loss L_adv; simultaneously train g_phi to maximize c_omega's loss (gradient reversal).
Joint objective: L = L_suff + lambda * L_adv, with lambda=0.1, alternating updates every 2 steps.
```

**Why this design.** We chose to train the reparameterization `g_phi(z, e)` jointly with the sufficiency objective rather than adding an independence penalty on `z` directly, because the nuisance factors are entangled with the dynamics in the raw token—only by letting `g_phi` condition on the embodiment label can the sufficiency loss drive nuisance into the reparameterization's null space, accepting the cost that `g_phi` must be learned and may overfit to seen embodiments. We chose an adversarial head on the `e`-conditional residual `r` (post-reparameterization and post-conditioning) rather than on `z` or on `u` directly, because the invariance must be a consequence of identifiability: if `r` is a sufficient statistic for the residual transition, then by the data-processing inequality it cannot contain extra information about `e` beyond what the transition requires, so the adversarial loss on `r` is a diagnostic of sufficiency, not an independent constraint. We chose a gradient-reversal layer over alternating GAN-style training because it gives a single unified gradient path through `g_phi`, accepting the cost that the adversarial signal may be weaker. This design differs from a standard disentanglement head (e.g., a β-VAE or a domain-adversarial network) because the invariance is not asserted statistically—it follows from the sufficiency of `r` for the residual transition operator, which is a structural identifiability condition, not a correlation penalty.

**Why it measures what we claim.** The negative log-likelihood of `o_{t+1}` given `u`, `a_t`, and `e` measures sufficiency of `u` for the action-conditioned transition because the assumption that the transition operator is invariant across embodiments after reparameterization makes the likelihood equivalent to a test of whether `u` captures all dynamics-relevant information; this assumption fails when the transition operator itself is embodiment-dependent (e.g., different contact dynamics), in which case the likelihood reflects embodiment-specific dynamics rather than shared dynamics. The adversarial loss on the residual `r` measures invariance because the assumption that a sufficient statistic cannot contain extra label information (data-processing inequality) makes the adversarial accuracy a direct test of sufficiency; this assumption fails when the transition head is underfit, in which case the adversarial loss reflects optimization noise rather than residual nuisance. To ground this in a measured quantity, we compute the mutual-information decomposition `I(o_{t+1}; u | a_t) = I(o_{t+1}; u | a_t, e) + I(o_{t+1}; e | u, a_t)` and use a probe to estimate the second term, verifying it is near zero.

## Contribution

['A training framework (ACSS) that enforces physical tokens to be sufficient statistics for the action-conditioned transition operator under a learned embodiment-conditional reparameterization, with invariance to embodiment/simulator labels emerging as an identifiability consequence rather than an independence penalty.', 'A joint sufficiency-adversarial objective with gradient reversal that operationalizes the identifiability condition, and a diagnostic showing that adversarial accuracy on the reparameterized token is a proxy for sufficiency rather than an independent constraint.', 'An empirical demonstration on multi-embodiment and multi-simulator data that tokens trained with ACSS transfer across embodiments and simulators without fine-tuning, and that the reparameterization absorbs nuisance factors that would otherwise contaminate the token space.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Multi-embodiment robot manipulation suite (LIBERO + RoboTwin 2.0 + SimplerEnv) | Spans distinct embodiments and simulators with shared task structure. |
| Primary metric | Action-conditioned next-state prediction error on held-out embodiment/simulator | Directly probes sufficiency of the physical token. |
| Secondary metric | Task success rate of zero-shot policy transfer to held-out embodiment | Measures downstream impact beyond prediction error. |
| Diagnostic metric | Estimated mutual information I(e; u) via probe | Quantifies residual label information in u. |
| Baseline | PDRVLA without ACSS reparameterization | Tests whether sufficiency alone yields invariance. |
| Baseline | Domain-adversarial feature learning (DANN-style) | Tests whether statistical invariance suffices without sufficiency. |
| Baseline | β-VAE disentanglement head | Tests whether generic disentanglement substitutes for structural identifiability. |
| Ablation-of-ours | ACSS with adversarial head on z (pre-reparameterization) | Isolates the role of post-reparameterization invariance. |
| Ablation-of-ours | ACSS with adversarial head on u (not residual r) | Tests whether conditioning on e is necessary for invariance. |

### Why this setup validates the claim

The central claim is that a physical token made a sufficient statistic for the action-conditioned transition, after a learned embodiment-conditional reparameterization, will transfer across embodiments and simulators without asserting statistical independence on the raw token. The dataset choice matters because the claim is only meaningful when the same dynamics appear under different embodiment labels: LIBERO, RoboTwin 2.0, and SimplerEnv provide overlapping task structure but distinct contact and actuation characteristics, so a held-out embodiment/simulator split forces the token to carry shared dynamics rather than label-correlated shortcuts. The primary metric—next-state prediction error on the held-out embodiment—is the right detector because the sufficiency claim predicts a specific pattern: error should stay low on held-out embodiments where the transition operator is shared, and rise only where the operator is genuinely embodiment-dependent. The secondary metric of task success rate on zero-shot policy transfer directly measures whether the learned token is useful for control, not just prediction. The diagnostic mutual-information probe I(e; u) provides a quantitative check that residual label information is near zero, grounding the invariance claim. The DANN-style baseline tests whether statistical invariance without sufficiency is enough; if it matches us, sufficiency is doing no work. The β-VAE baseline tests whether generic disentanglement substitutes for structural identifiability. The pre-reparameterization adversarial ablation tests whether the invariance must emerge post-`g_phi` rather than on `z`. The post-reparameterization but pre-conditioning ablation tests whether the e-conditional residual is necessary. Together these form a falsifiable lattice: our claim predicts a specific ordering of outcomes, not merely that we win.

### Expected outcome and causal chain

**vs. PDRVLA without ACSS reparameterization** — On a case where a held-out embodiment has different contact stiffness but identical task semantics, the baseline produces high next-state error because its physical token entangles embodiment-specific dynamics with shared dynamics and has no mechanism to separate them. Our method instead routes the token through `g_phi(z, e)` and conditions the transition head on `e`, so the sufficiency loss drives shared dynamics into `u` while nuisance is absorbed into the reparameterization's null space, so we expect a noticeable error gap concentrated on held-out-embodiment subsets with contact mismatch, and near-parity on subsets where contact dynamics coincide. Additionally, we expect our method to achieve higher zero-shot task success rate on the held-out embodiment, confirming downstream utility.

**vs. Domain-adversarial feature learning (DANN-style)** — On a case where the adversarial head successfully removes embodiment information from the token but the token was never forced to be sufficient for the transition, the baseline produces a token that is invariant but under-informative, yielding elevated next-state error on all subsets because the invariance constraint stripped dynamics-relevant signal along with nuisance. Our method instead ties invariance to sufficiency via the data-processing argument, so the adversarial loss on the residual `r` acts as a diagnostic rather than a competing constraint, and we expect our error to be uniformly lower than DANN's, with the gap widening on tasks requiring fine contact prediction. The mutual-information probe should show near-zero I(e; u) for both, but only our method maintains low prediction error.

**vs. β-VAE disentanglement head** — On a case where the β-VAE pressure collapses a dynamics-relevant latent dimension because it correlates with embodiment label, the baseline produces a token that loses action-conditioned predictive power on that dimension, showing error spikes on tasks whose dynamics depend on the collapsed factor. Our method instead lets `g_phi` condition on the embodiment label so the sufficiency loss can retain that dimension in `u` while pushing label information into the reparameterization, so we expect our error to be stable across tasks that load on the collapsed factor, with the baseline diverging specifically on those tasks. The MI probe should confirm that our method retains dynamics-relevant information while achieving low I(e; u).

**vs. ACSS with adversarial head on z (pre-reparameterization)** — On a case where the raw token `z` contains embodiment-specific nuisance, the ablation fails to remove it because the adversarial head on `z` competes with the sufficiency loss, leading to either high error or high I(e; z). Our method, by placing the adversarial head on the residual `r`, allows the sufficiency loss to drive nuisance into the reparameterization's null space, so we expect lower error and lower I(e; u) than the ablation.

**vs. ACSS with adversarial head on u (not residual r)** — On a case where the transition operator is embodiment-dependent, the ablation on `u` may still retain embodiment information because `u` is allowed to depend on `e` through the transition head. Our method, by evaluating the adversarial head on the e-conditional residual `r`, explicitly removes the e-conditional mean, so we expect lower I(e; r) and better transfer to held-out embodiments.

### What would falsify this idea

If our gain over the DANN-style baseline is uniform across all subsets rather than concentrated on held-out embodiments with shared transition operators, then sufficiency is not doing the causal work we claim and the result is explained by generic regularization. Likewise, if the pre-reparameterization adversarial ablation matches our method, then the post-`g_phi` placement—the core structural claim—is not responsible for the invariance.

## References

1. PDRVLA: A Vision-Language-Action Model with Physical Dynamics Representations
2. BC-Z: Zero-Shot Task Generalization with Robotic Imitation Learning
3. Wan: Open and Advanced Large-Scale Video Generative Models
