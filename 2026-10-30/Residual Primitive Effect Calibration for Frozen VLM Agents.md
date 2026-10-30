# Residual Primitive Effect Calibration for Frozen VLM Agents

## Motivation

Frozen VLM agents like the minimal-interface controller of Embodied Agents Take Control delegate all action selection and reward judgment to a static model, which cannot absorb execution feedback; as a result, drift, collisions, and reward mis-ranking accumulate because no estimator maps the agent's discrete primitive vocabulary to observed physical effects. This is a structural bottleneck across the field: π0 and ACT show that learned action heads or imitation policies can absorb feedback, but only by reinstating heavy trained modules that violate the minimal-interface spirit, while MapGPT and LangNav add map or language scaffolds without closing the execution loop. The missing piece is neither a static reward model (Agent-RewardBench shows such predictions do not calibrate) nor a full world model (too heavy), but a lightweight, online-trained calibration layer that learns per-primitive affine corrections in a low-dimensional effect space from the controller's own rollouts.

## Key Insight

Because the VLM's action interface is a small, discrete primitive vocabulary, the effect of each primitive can be captured by a per-primitive affine residual operator whose parameters are identified online from the controller's own rollouts without any human labels or heavy action head, making execution feedback absorbable at minimal interface cost.

## Method

**What it is** — RAPEC (Residual Affine Primitive Effect Calibration) is an online-trained estimator that, for each primitive in the frozen VLM's vocabulary, predicts an affine correction from the current low-dimensional effect state to the next effect state. Its predicted effect signature for each candidate primitive is appended to the VLM's prompt, and its residuals are used to re-rank the VLM's primitive proposals, closing the execution loop without any trained action head, map, or diffusion policy.

**Load-bearing assumption (stated explicitly)** — The per-primitive affine map A_p, b_p learned online from the controller's own rollouts is identifiable and predictive enough that its residual, used as a re-ranking penalty, tracks true physical deviation (collision/drift risk) rather than merely the distance of the predicted next state from the current state. This assumption is load-bearing because the entire re-ranking correction depends on the residual being a valid signal of physical deviation; if it is not, the method reduces to the frozen VLM baseline. We make this assumption explicit and add a concrete calibration/verification story below (goal-referenced potential, decoupled fitting, confidence gating, and an identifiability check) so that the assumption is testable rather than hidden.

**How it works** — We choose a pseudocode form because the mechanism is a tight online loop with clear data dependencies.

```pseudocode
# Inputs per step: frozen VLM V, primitive vocabulary P (|P| small, e.g., 12),
# low-dim effect state s_t in R^d (d=8: end-effector displacement 3, collision flag 1,
# object-relation change 4), goal effect state s_goal (provided by the task; e.g.,
# target displacement and zero collision flag).
# Hyperparameters: learning rate eta=0.05, ridge lambda=0.01, effect horizon H=1,
# residual penalty weight gamma=0.5, exploration rate epsilon=0.1,
# confidence threshold tau=0.1 (trace of ridge posterior covariance).

Initialize per-primitive affine maps A_p in R^{d x d}, b_p in R^d for all p in P
  (A_p = I, b_p = 0).
Initialize per-primitive ridge posterior covariance matrices Sigma_p = lambda * I (d x d).

for each control step t:
    # 1. VLM proposes primitive candidates with scores
    candidates = V.propose(observation o_t, prompt augmented with effect signatures)
    # 2. Predict effect signature for each candidate
    for p in candidates:
        s_hat_p = A_p @ s_t + b_p
        signature_p = compress(s_hat_p)   # e.g., top-2 dimensions + collision flag
    # 3. Re-rank candidates by VLM score minus goal-referenced residual penalty,
    #    gated by per-primitive confidence
    for p in candidates:
        # Goal-referenced potential-based penalty (Ng et al. 1999):
        # Phi(s) = -||s - s_goal||_2, so penalty = ||s_hat_p - s_goal||_2 - ||s_t - s_goal||_2
        # (the difference form is potential-based and policy-invariant).
        potential_current = -||s_t - s_goal||_2
        potential_pred = -||s_hat_p - s_goal||_2
        residual_p = potential_pred - potential_current   # negative when predicted state is closer to goal
        # Confidence gate: if trace(Sigma_p) > tau, the affine map is not calibrated;
        # set penalty to zero (no spurious zero residual from A_p=I, b_p=0).
        if trace(Sigma_p) > tau:
            residual_p = 0.0
        rank_score_p = VLM_score(p) - gamma * residual_p   # gamma=0.5
    # Epsilon-greedy exploration to decouple data collection from selection:
    # with probability epsilon, select a uniformly random primitive from candidates;
    # otherwise select p* = argmax rank_score_p.
    if uniform(0,1) < epsilon:
        p* = random_choice(candidates)
    else:
        p* = argmax rank_score_p
    # 4. Execute p*, observe next effect state s_{t+1}
    execute(p*)
    # 5. Online update of the selected primitive's affine map (ridge/RLS update)
    #    using all executed primitives (including exploratory ones) to avoid selection bias.
    error = s_{t+1} - (A_{p*} @ s_t + b_{p*})
    # RLS update with forgetting factor lambda (ridge):
    K = Sigma_{p*} @ s_t / (lambda + s_t^T @ Sigma_{p*} @ s_t)
    A_{p*} <- A_{p*} + K @ error^T
    b_{p*} <- b_{p*} + K * error
    Sigma_{p*} <- (Sigma_{p*} - K @ s_t^T @ Sigma_{p*}) / lambda
    # 6. Update VLM prompt with observed effect signature for next step
    prompt = augment(prompt, observed_signature(s_{t+1}))
```

**Why this design** — We chose a per-primitive affine residual operator over a full world model or a neural dynamics model because the primitive vocabulary is small and discrete, so a per-primitive affine map is extremely sample-efficient (d^2 + d parameters per primitive) and can be trained online from the controller's own rollouts without any human labels, accepting the cost that it cannot capture nonlinear effects beyond the low-dimensional effect space. We chose to augment the VLM prompt with predicted effect signatures rather than fine-tuning the VLM because the VLM is frozen and we must not reinstate a trained action head, accepting the cost that the VLM may ignore the signatures if its attention is dominated by visual tokens. We chose to re-rank the VLM's primitive proposals by subtracting a goal-referenced residual penalty rather than replacing the VLM's scores entirely because the VLM's semantic grounding remains valuable and we only want to correct for physical effects the VLM cannot see, accepting the cost that the residual penalty weight gamma must be tuned. We chose online ridge updates (eta, lambda) with epsilon-greedy exploration rather than batch retraining because the controller must adapt within a single episode, accepting the cost that early estimates are noisy; the epsilon-greedy exploration decouples data collection from selection so that A_p is fitted on all executed primitives, not only those selected by the residual, avoiding selection bias. We chose a confidence gate on the ridge posterior covariance so that uncalibrated primitives (A_p=I, b_p=0) contribute zero penalty instead of a spurious zero residual. Unlike VinePPO (per-step Monte-Carlo value rollouts) or a process reward model, RAPEC does not estimate value or reward at all; it estimates a physical effect operator over a discrete primitive vocabulary, which is a different object that those methods do not produce.

**Why it measures what we claim** — The computational quantity `s_hat_p = A_p @ s_t + b_p` measures the predicted effect signature of primitive p because we assume the effect of each primitive is an affine function of the current low-dimensional effect state; this assumption fails when the primitive's effect depends on unobserved variables (e.g., object mass, friction), in which case the predicted signature reflects only the observable linear component. The residual `||s_hat_p - s_goal||_2 - ||s_t - s_goal||_2` measures the expected physical deviation (drift or collision risk) of primitive p relative to the goal because we assume the effect state is centered on the desired trajectory and the goal state s_goal is known; this assumption fails when the effect state is not centered on the desired trajectory, in which case the residual reflects distance from the current state rather than from the goal. The online update error `s_{t+1} - (A_{p*} @ s_t + b_{p*})` measures the calibration gap of the selected primitive because we assume the observed next effect state is a noisy but unbiased sample of the true effect; this assumption fails when the observation is delayed or occluded, in which case the error reflects observation noise rather than model miscalibration. The re-ranking score `VLM_score(p) - gamma * residual_p` measures the combined semantic and physical suitability of primitive p because we assume the VLM score and the residual penalty are on comparable scales after gamma tuning; this assumption fails when the VLM's scores are highly peaked, in which case the re-ranking reflects the VLM's prior more than the physical correction. The confidence gate `trace(Sigma_p) > tau` measures whether the affine map for primitive p is calibrated because we assume the ridge posterior covariance trace reflects parameter uncertainty; this assumption fails when the covariance is poorly estimated due to few samples, in which case the gate may incorrectly suppress a calibrated primitive or allow an uncalibrated one.

## Contribution

["A residual affine primitive effect (RAPEC) estimator that learns per-primitive affine corrections in a low-dimensional effect space online from the controller's own rollouts, providing a lightweight calibration layer between a frozen VLM's discrete primitive vocabulary and observed physical effects without any trained action head, map, or diffusion policy.", "A prompt augmentation and re-ranking scheme that injects the estimator's predicted effect signatures into the frozen VLM's prompt and uses the estimator's residuals to re-rank the VLM's primitive proposals, closing the execution loop at minimal interface cost.", 'An empirical demonstration that online residual calibration reduces drift, collisions, and reward mis-ranking in vision-and-language navigation and embodied question-answering compared to the uncalibrated minimal-interface agent of Embodied Agents Take Control, without reinstating heavy trained modules.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | R2R + REVERIE zero-shot | Discrete primitives, low-dim effect state, VLN grounding |
| Primary metric | Success rate on collision/drift-heavy episodes | Directly sensitive to physical correction |
| Baseline | Frontier VLM agent (frozen, no calibration) | Tests whether online calibration adds anything |
| Baseline | MapGPT-style map-guided planner | Tests whether explicit maps substitute for effect calibration |
| Baseline | VinePPO-style rollout re-ranking | Tests whether value rollouts beat effect operators |
| Baseline | Offline batch-ridge per-primitive affine dynamics model | Tests whether online update beats offline batch fitting |
| Baseline | Small MLP dynamics head (2-layer, hidden=64) trained online | Tests whether affine operator is sufficient vs nonlinear |
| Ablation-of-ours | RAPEC without residual re-ranking (signature-only) | Isolates the re-ranking contribution |
| Ablation-of-ours | RAPEC with raw state-deviation residual (no goal reference) | Tests the goal-referenced potential fix |
| Ablation-of-ours | RAPEC without confidence gate | Tests the spurious zero-residual issue |
| Ablation-of-ours | RAPEC without epsilon-greedy exploration | Tests selection bias in online fitting |

### Why this setup validates the claim

The central claim is that a per-primitive affine effect operator, learned online and injected as a goal-referenced residual re-ranking signal, corrects physical execution errors that a frozen VLM alone cannot see. R2R and REVERIE provide exactly the stress case: a small discrete action vocabulary (move-forward, turn, etc.) whose effects are affine-ish in a low-dimensional state but whose executed outcomes drift and collide under VLM semantic priors. The frozen-VLM baseline tests the sub-claim that semantic grounding alone is insufficient; the map-guided baseline tests that explicit spatial memory is a different object from effect calibration; the rollout re-ranking baseline tests that value estimation is not the missing ingredient. The offline batch-ridge baseline tests whether the online update is necessary versus a static affine model trained on a batch of rollouts; the MLP dynamics head tests whether the affine operator is sufficient versus a nonlinear model. The signature-only ablation is the falsification lever: if the residual penalty does nothing, it will match the full method. The raw state-deviation ablation tests whether the goal-referenced potential is necessary to avoid the sign error; the no-confidence-gate ablation tests whether the gate prevents spurious zero residuals; the no-exploration ablation tests whether selection bias degrades online fitting. Success rate on collision- and drift-heavy episodes is the right metric because it is where the predicted effect (fewer physical failures) should concentrate; a metric like overall success would dilute the signal. This combination forms a falsifiable test: the method must beat all three baselines specifically on the physically hard subset, and the ablations must lose specifically there.

### Expected outcome and causal chain

**vs. Frontier VLM agent (frozen, no calibration)** — On a case where the VLM proposes "move forward" toward a visually plausible but physically blocked doorway, the baseline executes the primitive and collides because its semantic prior is dominated by visual tokens and it has no signal that this primitive's effect in this state is a collision. Our method instead predicts the collision flag in the effect signature for that primitive, penalizes the goal-referenced residual, and re-ranks toward a lateral primitive because the affine map was calibrated online from prior collisions. We expect a noticeable success-rate gap concentrated on collision-heavy episodes, with near-parity on open-corridor episodes.

**vs. MapGPT-style map-guided planner** — On a case where the map says the target is reachable but a primitive's executed displacement drifts due to an unmodeled surface, the baseline replans on the map but keeps selecting the same drifting primitive because the map encodes topology, not per-primitive effect. Our method instead attributes the drift to that primitive's affine map, updates A_p and b_p online, and re-ranks away from the drifting primitive within the same episode because the goal-referenced residual penalty reflects the calibrated effect. We expect a gap on drift-heavy episodes that grows with episode length, with parity on short clean episodes.

**vs. VinePPO-style rollout re-ranking** — On a case where Monte-Carlo rollouts are noisy because the episode horizon is long and the VLM's proposals are all semantically reasonable, the baseline's per-step value estimates have high variance and re-rank inconsistently because it estimates reward-to-go rather than physical effect. Our method instead re-ranks on a low-variance affine effect residual that is directly observed each step, so the correction is stable. We expect our method to show lower variance in primitive selection and a modest but consistent success gain on long episodes, with parity on short ones.

**vs. Offline batch-ridge per-primitive affine dynamics model** — On a case where the environment's surface properties change mid-episode (e.g., a slippery patch), the offline batch-ridge baseline uses a static affine map trained on a fixed batch of rollouts and cannot adapt, so its residual penalty is stale and mis-ranks primitives after the change. Our method updates A_p and b_p online with each executed primitive, so it adapts within the episode. We expect our method to outperform the offline baseline on episodes with mid-episode changes, with parity on stationary episodes.

**vs. Small MLP dynamics head (2-layer, hidden=64) trained online** — On a case where the primitive effects are strongly nonlinear (e.g., contact-rich collisions), the MLP baseline may capture nonlinearities but requires more samples to train online and may overfit early. Our method's affine operator is more sample-efficient but may underfit nonlinear effects. We expect the MLP to match or slightly exceed our method on highly nonlinear collision episodes if given enough data, but our method to be more stable early in the episode due to fewer parameters; overall, we expect comparable performance, with our method winning on sample efficiency and the MLP winning on nonlinearity if data is abundant.

### What would falsify this idea

If the full method's gain over the frozen-VLM baseline is uniform across collision-heavy, drift-heavy, and clean subsets rather than concentrated on the physically hard subsets, or if the signature-only ablation matches the full method, or if the raw state-deviation ablation matches the goal-referenced version, then the residual re-ranking is not doing the work and the central claim is wrong.

## References

1. Embodied Agents Take Control: Minimal-Interface Zero-Shot Agents Rival Industrial-Scale Policies in Vision-and-Language Navigation
2. π0: A Vision-Language-Action Flow Model for General Robot Control
3. MapGPT: Map-Guided Prompting with Adaptive Path Planning for Vision-and-Language Navigation
4. Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware
5. BridgeData V2: A Dataset for Robot Learning at Scale
6. Visual Instruction Tuning
7. LangNav: Language as a Perceptual Representation for Navigation
8. March in Chat: Interactive Prompting for Remote Embodied Referring Expression
