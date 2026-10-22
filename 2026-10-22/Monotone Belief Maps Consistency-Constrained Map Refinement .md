# Monotone Belief Maps: Consistency-Constrained Map Refinement for Active Camera Control

## Motivation

Active object-level mapping requires an agent to repeatedly revise its spatial belief as new views arrive, yet no existing system constrains how successive map estimates relate to one another. SpatialScout evaluates a single final map per scene, so an agent can pass through arbitrarily contradictory intermediate revisions — an object center placed at one location and then relocated across the room — without any signal that the belief trajectory is incoherent; even the strongest model scores only 41.71% mean overlap, and part of that deficit plausibly comes from unconstrained revision rather than from poor perception alone. The structural gap is that belief refinement is supervised only at the endpoint, so the field has no way to tell whether a map is converging or oscillating, and BLINK-style episodic benchmarks provide no belief-state machinery at all.

## Key Insight

If each new observation is required to refine the map through an evidence-monotone update — never silently overwriting prior evidence but only re-weighting it under an explicit contradiction cost — then the per-step belief revision becomes a measurable, ordered quantity whose total variation reveals whether the agent's spatial belief is converging or thrashing.

## Method

## MBM: Monotone Belief Maps

**(A) What it is.** MBM wraps SpatialScout's camera-control loop with an explicit per-object belief state (a Gaussian mixture over world-frame object centers with per-component evidence counts) and trains the camera policy with two auxiliary signals: a *refinement-consistency loss* that penalizes non-monotone changes in a proper, evidence-conditioned belief score, and a *contradiction penalty* that charges the agent whenever a new component is added that is incompatible with an existing one. Inputs: RGB observation, camera pose/calibration, object inventory. Outputs: camera action, updated belief map, and a scalar *belief coherence score* per step.

**(B) How it works.**

```pseudocode
# per object o in inventory, maintain belief b_o = {(mu_k, Sigma_k, w_k, n_k)}
# w_k: mixture weight, n_k: evidence count (pseudo-count of supporting views)
# held-out buffer D_o: last 16 (mu, Sigma) proposals NOT used in the current update

for step t in 1..T:
    obs, pose = environment.step(a_{t-1})
    # 1. Evidence extraction: VLM proposes candidate centers with view-ray uncertainty
    cands = vlm_propose_centers(obs, pose, inventory)      # each has mu, Sigma, conf c
    # 2. Monotone merge: assign each candidate to nearest existing component
    for (mu, Sigma, c) in cands:
        k* = argmin_k mahalanobis(mu, b_o[k])
        if mahalanobis(mu, b_o[k*]) < tau_assoc (=3.0):
            # Bayesian update with evidence count, NOT overwrite
            n_k* += 1
            w_k* = n_k* / sum_j n_j
            b_o[k*] = kalman_update(b_o[k*], mu, Sigma)
            contradiction = 0
            new_component = False
        else:
            # new component: charge contradiction against ALL existing components
            contradiction = sum_k w_k * exp(-mahalanobis(mu, b_o[k])**2 / 2)
            b_o.append((mu, Sigma, c, 1))
            new_component = True
        # push the pre-update proposal into the held-out buffer (FIFO, max 16)
        D_o.append((mu, Sigma))
    # 3. Refinement-consistency: proper scoring signal on held-out proposals
    #    NLL_t = mean over D_o of -log sum_k w_k N(mu | mu_k, Sigma_k + Sigma)
    NLL_t = heldout_nll(b_o, D_o)
    delta_NLL = NLL_t - NLL_{t-1}
    # gate: only penalize non-monotone NLL when the update was an association
    # (new_component == False); legitimate new-component corrections are exempt
    L_mono = relu(delta_NLL - eps) * (1 - new_component)   # eps = 0.02
    # 4. Contradiction penalty: any new component with high contradiction is charged
    L_contra = lambda_c (=0.5) * contradiction
    # 5. Belief coherence score (used for early-stop and for logging)
    coherence_t = 1 - (L_mono + L_contra) / (1 + |NLL_t|)
    # 6. Policy loss: task loss (overlap score) + auxiliaries + temporal smoothing
    L_smooth = lambda_s (=0.1) * ||b_o - b_o^{t-1}||^2   # penalize large jumps
    L = L_task + lambda_m (=0.3) * L_mono + L_contra + L_smooth
    a_t = policy(obs, b, coherence_t)
```

**(C) Why this design.** We chose a Bayesian mixture with explicit evidence counts over a single point estimate because SpatialScout's endpoint-only evaluation lets a point estimate jump freely between views, and an evidence count is the minimal sufficient statistic for making refinement monotone — accepting the cost that the mixture grows with views and must be pruned (we cap at 8 components per object, merging lowest-weight pairs). We chose a Mahalanobis association gate over nearest-neighbor assignment because the gate encodes view-ray uncertainty geometrically, so a far-away candidate from a well-conditioned view is not spuriously merged with a nearby-but-uncertain one — accepting the cost that a mis-calibrated Sigma produces false associations. We chose a soft contradiction penalty over a hard rejection rule because hard rejection would discard genuine corrections (an object previously mislocalized), while the soft penalty lets the policy learn when a contradiction is worth paying for — accepting the cost that the penalty weight lambda_c must be tuned and is scene-dependent. We chose to add a temporal smoothing term over the raw belief because the monotonicity loss alone permits large but entropy-preserving jumps — accepting the cost that smoothing can slow legitimate corrections. We replaced raw mixture entropy with the held-out negative log-likelihood (a strictly proper scoring rule) because mixture entropy is not monotone under Bayesian updating — adding a component strictly increases mixture entropy even when the new evidence is correct, and pruning can arbitrarily raise or lower it — so an entropy-based monotonicity loss would penalize exactly the genuine corrections we want; the NLL is monotone by construction under informative observations and is gated to exempt new-component corrections. A domain expert would not describe this as a variant of SpatialScout fine-tuning or of BLINK-style episodic evaluation: neither maintains a per-object evidence-counted belief state, and neither trains on a per-step coherence signal.

**(D) Why it measures what we claim.** `L_mono = relu(delta_NLL - eps) * (1 - new_component)` measures **belief refinement monotonicity** because the assumption that the held-out negative log-likelihood of proposals under the belief is a strictly proper scoring rule — so that informative observations decrease it in expectation — makes NLL a sufficient statistic for refinement; this assumption fails when the VLM proposes spurious components with high confidence, in which case `L_mono` reflects proposal noise rather than map incoherence, which is why we gate it to association updates and pair it with a calibration check (see experiment). `contradiction` measures **belief contradiction** because the assumption that Mahalanobis distance under the view-ray covariance is a valid likelihood ratio for two observations of the same object makes the exponential kernel a calibrated contradiction probability; this assumption fails under camera-pose drift, in which case the penalty reflects pose error rather than genuine map conflict. `coherence_t` measures **convergence of the spatial belief** because the assumption that a converging belief trajectory has bounded total variation in NLL and contradiction makes the normalized sum a monotone proxy for convergence; this assumption fails when the agent intentionally explores a new region (NLL legitimately rises), in which case coherence_t dips for exploration rather than for incoherence — which is why we use it for logging and early-stop rather than as a training gate.

**(D-sub) Explicit calibration assumption and failure mode.** The equivalence `L_mono` measures refinement monotonicity rests on assumption A: the VLM proposal distribution (mu, Sigma, conf c) is calibrated so that entropy decrease (or NLL decrease) reflects informative evidence rather than confidence inflation. The failure mode F is a miscalibrated VLM that produces NLL-decreasing but wrong updates (e.g., high-confidence spurious centers that happen to fit the current mixture). We therefore add a calibration check on the proposal confidences: on a held-out set of 512 same-object proposal pairs, we test whether the Mahalanobis distances under the proposed Sigma follow a chi-square distribution with 3 degrees of freedom (the dimension of the object center); we report the empirical false-association rate at tau_assoc=3.0 and use it to set the gate threshold. If the calibration check fails (false-association rate > 5%), we do not claim that `L_mono` measures refinement monotonicity; instead we report that it measures proposal consistency under the given VLM.

## Contribution

(1) A per-object evidence-counted Gaussian-mixture belief state with a monotone Bayesian merge rule and an explicit contradiction penalty, giving active-mapping agents a per-step belief trajectory rather than a single endpoint map. (2) A refinement-consistency training objective combining entropy-monotonicity, contradiction, and temporal-smoothing terms that is trained jointly with the camera policy, plus a per-step belief coherence score that exposes whether a VLM agent's map is converging or oscillating. (3) An analysis of which SpatialScout failures are attributable to unconstrained revision versus perception error, obtained by comparing coherence trajectories across frontier VLMs.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | SpatialScout 80-scene, 3,140-object benchmark | Matches the task and object-inventory setting we target. |
| Primary metric | Mean overlap score | Directly measures center-estimation accuracy under bounded interaction. |
| Secondary metric | Total variation of belief over steps (TV) and its correlation with final overlap across all 80 scenes | Tests whether per-step coherence predicts endpoint accuracy. |
| Baseline | SpatialScout (Claude Opus 5.5) | Strongest endpoint-only VLM explorer; tests the value of explicit belief state. |
| Baseline | SpatialScout + point-estimate Kalman filter | Isolates whether monotone evidence counts, not filtering alone, drive gains. |
| Baseline | BLINK-style episodic re-planning | Tests whether per-step coherence beats re-initialized episode planning. |
| Ablation-of-ours | MBM without L_mono and L_contra | Isolates the auxiliary belief signals from the mixture representation. |
| Ablation-of-ours | MBM with raw mixture-entropy L_mono (original formulation) | Tests whether the proper scoring rule (NLL) is necessary. |
| Calibration check | Chi-square test on Mahalanobis distances of 512 held-out same-object proposal pairs; report false-association rate at tau_assoc=3.0 | Grounds the association gate's validity. |

### Why this setup validates the claim

The central claim is that an evidence-counted per-object belief state plus monotone-refinement and contradiction auxiliary signals produce more stable, more accurate object-center estimates than endpoint-only exploration. SpatialScout (Claude Opus 5.5) is the right primary baseline because it is the strongest frontier-VLM explorer under the same 80-scene, 3,140-object, bounded-budget protocol; any gain over it is a gain attributable to belief maintenance rather than to raw perception. The point-estimate Kalman baseline tests a weaker sub-claim: if filtering alone closed the gap, our mixture and auxiliary losses would be unnecessary, so a gap over this baseline directly supports the evidence-count and monotonicity design. The BLINK-style re-planning baseline tests whether per-step coherence carries information beyond episodic re-initialization. The ablation (MBM without L_mono and L_contra) is the decisive internal control: it holds the mixture architecture fixed while removing the two signals, so any residual gain over the ablation must come from those signals, not from the representation. The additional ablation with raw mixture-entropy L_mono directly tests the load-bearing assumption: if the proper scoring rule (NLL) is necessary, the entropy variant should underperform full MBM. The calibration check on 512 held-out same-object proposal pairs verifies that the Mahalanobis gate is valid (chi-square with 3 df) and reports the empirical false-association rate at tau_assoc=3.0; if the rate exceeds 5%, we do not claim that L_mono measures refinement monotonicity. The secondary metric (total variation of belief over steps and its correlation with final overlap across all 80 scenes) tests whether per-step coherence predicts endpoint accuracy, providing a reusable diagnostic. Mean overlap score remains the primary metric because it is the benchmark's native endpoint measure and is sensitive to exactly the failure mode we predict (center drift from non-monotone refinement), while per-step coherence is logged as a diagnostic and used for early-stop.

### Expected outcome and causal chain

**vs. SpatialScout (Claude Opus 5.5)** — On a case where the agent revisits an object from a second viewpoint and the new VLM proposal is offset from the first, SpatialScout overwrites its point estimate or averages naively because it has no evidence count and no association gate, so the center drifts toward whichever view was seen last. Our method instead assigns the candidate through a Mahalanobis gate, increments n_k, and Kalman-updates the component, so the estimate moves only in proportion to accumulated evidence. We expect a noticeable overlap-score gap concentrated on objects observed from multiple, widely separated viewpoints, with near-parity on objects seen from a single frontal view.

**vs. SpatialScout + point-estimate Kalman filter** — On a case where the VLM proposes a spurious high-confidence center far from the true object, the point-estimate filter either accepts it (large jump) or rejects it without recording why, because a single Gaussian cannot represent an alternative hypothesis and has no contradiction term. Our method instead adds a new mixture component, charges a soft contradiction proportional to its Mahalanobis overlap with existing components, and lets the policy learn to seek disambiguating views. We expect a modest but consistent gain over the point-estimate filter on scenes with reflective or cluttered objects, and parity on clean single-view scenes.

**vs. BLINK-style episodic re-planning** — On a case where the agent's belief is mid-refinement and the episode budget forces a re-plan, the BLINK-style baseline re-initializes from the current observation, discarding accumulated evidence, because its unit of reasoning is the episode rather than the per-object belief trajectory. Our method instead carries the mixture and its evidence counts across re-plans and uses coherence_t to early-stop only when the belief has converged. We expect a gap that grows with the number of re-plan events per scene, i.e., larger on scenes requiring more interaction, and near-zero on short episodes.

**vs. MBM without L_mono and L_contra** — On a case where the agent receives a sequence of informative views, the ablation has no signal to discourage non-monotone belief updates, so the mixture can oscillate or drift even though the representation is identical. Our method adds the gated NLL-based monotonicity loss and the contradiction penalty, which should reduce total variation of belief and improve final overlap. We expect the ablation to match full MBM on single-view objects but to underperform on multi-view objects and on scenes with re-plan events.

**vs. MBM with raw mixture-entropy L_mono** — On a case where a genuine correction requires adding a new far-away component, the entropy-based loss penalizes the entropy increase and suppresses the correction, whereas the NLL-based loss is gated to exempt new-component updates. We expect the entropy variant to underperform full MBM on objects that are initially mislocalized and later corrected, and to show higher false-association rates if the gate is mis-calibrated.

### What would falsify this idea

If MBM's gain over SpatialScout is uniform across single-view and multi-view objects, and the ablation without L_mono and L_contra matches full MBM, then the evidence-counted mixture and auxiliary signals are not doing the work we claim and the central mechanism is wrong. A second falsifier: if the calibration check shows that Mahalanobis distances under the proposed Sigma do not follow a chi-square distribution (false-association rate > 5% at tau_assoc=3.0), then the association gate is invalid and L_mono cannot be claimed to measure refinement monotonicity.

## References

1. SpatialScout: Can VLM Agents Actively Explore and Map 3D Scenes?
