# Certified Command Generation for Vision-Language-Action Models via a Jointly Learned Viability Barrier

## Motivation

Across VLA systems such as π0.5 and π0, the high-level semantic subtask head and the low-level action controller are trained and defined independently, so a generated subtask is committed without any shared representation against which its induced action sequence can be checked for dynamic feasibility; feasibility is only discovered when the controller fails at execution time. This is a forest-level structural gap, not a local one: the same decoupling appears in navigation planners that emit stage transitions and defer feasibility to a downstream controller. The root cause is that no object in the command representation encodes the controller's constraint set, so the map from generated command to feasible action is not decidable at generation time.

## Key Insight

If the command head is trained to emit a scalar margin over a barrier function whose sublevel set is simultaneously trained to contain exactly the low-level controller's certified viable action set, then the sign of that margin is a decision procedure for command executability, because the barrier and the command head share the same learned constraint representation rather than being composed post hoc.

## Method

**What it is.** We introduce CVC (Certified Viability Command head), a modification of the π0.5-style high-level command head that, for each generated semantic subtask, additionally emits a scalar certificate margin m(s, c) = min_{t=0..H-1} [B_θ(s_{t+1}, c) − B_θ(s_t, c) + α B_θ(s_t, c)], where B_θ is a learned control-barrier-style function over state s and command c, α ∈ (0,1) is a fixed decay rate (α=0.1), and the min is taken over the H-step controller rollout from (s, c). The command is committed only if m > 0; otherwise the head re-samples with a margin penalty. This margin is the minimum slack of the discrete-time barrier decrease condition B_θ(s_{t+1}, c) − B_θ(s_t, c) ≥ −α B_θ(s_t, c) along the rollout, so sign(m) certifies forward invariance of the sublevel set {B_θ < κ} over the rollout horizon H, not short-horizon terminal success.

**How it works.**

```pseudocode
# Training (joint, end-to-end with the VLA backbone)
# Hyperparameters: H=20, α=0.1, κ=0.5, δ=0.5, λ1=1.0, λ2=1.0, λ3=1.0
for each batch of (observation o, language l, subtask c*, action chunk a*):
    c_hat = command_head(o, l)                 # semantic subtask
    a_hat = low_level_controller(o, c_hat)     # frozen controller
    # Rollout: execute a_hat for H=20 steps, record states s_0=o, s_1,...,s_H
    # Barrier decrease condition (discrete-time CBF):
    #   For each t in 0..H-1, require B_θ(s_{t+1}, c_hat) - B_θ(s_t, c_hat) >= -α B_θ(s_t, c_hat)
    #   Define slack_t = B_θ(s_{t+1}, c_hat) - B_θ(s_t, c_hat) + α B_θ(s_t, c_hat)
    #   m = min_t slack_t
    # Barrier regression loss (encourage B_θ to be small on viable states, large on violating states):
    #   B_target = 1 if rollout from (s, c_hat) stays in viable set V for H steps (no violation/termination)
    #   B_target = 0 otherwise
    L_barrier = BCE(B_θ(o, c_hat), B_target)
    # Certificate margin, trained contrastively:
    #   positive: c_hat == c* and rollout viable -> push m > +δ
    #   negative: c_hat != c* or rollout infeasible -> push m < -δ
    L_cert = max(0, δ - m_pos) + max(0, δ + m_neg)
    # Viability-kernel consistency: sublevel set {B_θ < κ} must match
    # empirical viable set from H-step rollouts
    L_cons = || 1[B_θ < κ] - 1[rollout viable] ||_1
    L = L_vla + λ1 * L_barrier + λ2 * L_cert + λ3 * L_cons

# Inference
c_hat = command_head(o, l)
# Compute m by rolling out controller for H=20 steps from (o, c_hat) and evaluating B_θ along the trajectory
m = min_{t=0..H-1} [B_θ(s_{t+1}, c_hat) - B_θ(s_t, c_hat) + α B_θ(s_t, c_hat)]
if m > 0:
    commit c_hat to controller
else:
    c_hat = command_head(o, l, margin_penalty=+1.0)  # re-sample once
```

**Why this design.** We chose a control-barrier-style scalar function over a classifier over discrete subtask labels because the controller's feasible set is a continuous region in action space, and a scalar sublevel set gives a decision boundary that is differentiable with respect to the command embedding, accepting the cost that B_θ must be trained with rollout-based targets rather than pure supervised labels. We chose to freeze the low-level controller during certificate training rather than co-train it, because a moving constraint set would make the barrier target non-stationary and invalidate the contrastive margin, accepting the cost that the certificate is tied to a specific controller checkpoint. We chose a contrastive margin loss with a fixed δ over a log-probability gate, because log-probs are not calibrated for feasibility and a margin over a barrier function has a direct geometric interpretation as distance to the feasible-set boundary, accepting the cost that δ must be tuned per task family. A domain expert would not describe this as a post-hoc verifier or as retrieval over subtask templates, because the barrier is trained jointly with the command head and its sublevel set is the controller's viability kernel, not an external database.

**Why this measures what we claim.** The computational quantity B_θ(s, c) measures the motivation-level concept of command executability because we assume that the H-step rollout viability of the controller from (s, c) is a sufficient statistic for dynamic feasibility; this assumption fails when the controller is stochastic or when H is shorter than the task horizon, in which case B_θ reflects short-horizon viability rather than full-horizon feasibility. The contrastive margin m measures the signed distance from the command to the feasible-set boundary because we assume the barrier sublevel set {B_θ < κ} coincides with the empirical viable set; this assumption fails when the barrier is under-trained on rare commands, in which case m reflects training-data density rather than feasibility. The consistency loss L_cons measures the alignment between the learned barrier and the rollout-derived viability kernel because we assume the H-step rollout is an unbiased estimator of the true viability kernel; this assumption fails when the rollout policy differs from the deployment controller, in which case L_cons reflects the gap between training and deployment controllers. The barrier decrease condition (slack_t ≥ 0) measures forward invariance of the sublevel set over the rollout horizon because we assume the discrete-time CBF condition with decay α is enforced along the rollout; this assumption fails when the barrier is under-trained on rare commands, in which case the minimum slack m reflects training-data density rather than invariance.

## Contribution

['A command-head architecture for VLA models that emits a scalar certificate margin over a jointly learned barrier function, making command executability decidable at generation time rather than at execution time.', "A training scheme that couples the barrier's sublevel set to the low-level controller's empirical viability kernel via rollout-based targets and a contrastive margin loss, producing a certificate whose sign is a decision procedure for feasibility.", 'A design principle: the certificate must be part of the command representation (trained end-to-end with the command head), not a post-hoc verifier, because a post-hoc verifier cannot influence the command distribution at generation time.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|-----------------------|
| Dataset | π0.5 open-world kitchen/bedroom manipulation | Tests long-horizon dexterous generalization in unseen homes |
| Primary metric | Task success rate | Directly measures command executability and viability |
| Baseline | π0.5 command head | Isolates benefit of certificate gating over standard head |
| Baseline | π0.5 + post-hoc feasibility classifier | Tests whether joint barrier training beats external verifier |
| Baseline | π0.5 + log-probability gate | Tests whether barrier margin beats uncalibrated confidence |
| Baseline | π0.5 + learned CBF trained with HJ-reachability-style Bellman backups | Tests whether contrastive margin yields tighter viability-kernel approximation |
| Ablation-of-ours | CVC without L_cons | Tests necessity of viability-kernel consistency loss |
| Ablation-of-ours | CVC without barrier decrease condition (using m = B_θ − κ) | Tests necessity of discrete-time CBF condition for certification |

### Why this setup validates the claim

The central claim is that a jointly trained control-barrier certificate on the command head improves command executability in open-world manipulation by filtering infeasible subtasks before they reach the controller. The π0.5 command head baseline isolates the contribution of the certificate itself: any gain over it must come from the barrier margin. The post-hoc feasibility classifier baseline tests the sub-claim that joint training with rollout-based barrier targets is superior to an external verifier trained after the fact, because the latter lacks gradient coupling to the command embedding. The log-probability gate baseline tests the sub-claim that a calibrated geometric margin over a barrier function outperforms an uncalibrated confidence score. The learned CBF with HJ-reachability-style Bellman backups baseline tests whether the contrastive margin loss yields a measurably tighter viability-kernel approximation (e.g., lower false-positive rate on boundary commands) than the standard CBF objective. The ablation without L_cons tests whether the consistency loss is what aligns the learned sublevel set with the empirical viability kernel; if removing it does not hurt, the barrier is not measuring feasibility as claimed. The ablation without the barrier decrease condition tests whether the discrete-time CBF condition is necessary for the certificate to reflect forward invariance; if removing it does not hurt, the margin is not measuring invariance as claimed. Task success rate is the right primary metric because the predicted effect is a reduction in infeasible command commitments that manifest as failed rollouts; a metric like command accuracy or margin calibration could improve without any downstream success gain, and success rate would then falsify the causal chain.

### Expected outcome and causal chain

**vs. π0.5 command head** — On a case where the observation suggests a subtask that is semantically plausible but dynamically infeasible for the current controller checkpoint (e.g., "open the high cabinet" when the arm is near joint limit), the baseline commits the command because it only maximizes semantic likelihood, producing a rollout that terminates early or collides. Our method instead computes m < 0 (the barrier decrease condition is violated along the rollout) for that command and re-samples with a margin penalty, yielding a nearby feasible subtask such as "open the lower cabinet," so we expect a noticeable success-rate gap concentrated on infeasibility-triggering episodes but near parity on episodes where all plausible subtasks are feasible.

**vs. π0.5 + post-hoc feasibility classifier** — On a case where the feasible set boundary is continuous in command embedding space (e.g., a reach slightly beyond the controller's workspace), the post-hoc classifier produces a hard label with no gradient coupling to the command head, so it cannot steer re-sampling toward the feasible side and often rejects all top candidates or accepts a boundary-straddling command. Our method instead uses the differentiable barrier margin to penalize and re-sample in the command embedding, so we expect a moderate success gain on boundary cases and a lower re-sampling rejection rate, while both methods match on clearly feasible or clearly infeasible commands.

**vs. π0.5 + log-probability gate** — On a case where the command head is confidently wrong (high log-prob for an infeasible subtask due to language prior), the log-probability gate commits because confidence is not calibrated to feasibility, producing a failed rollout. Our method instead gates on a margin trained against rollout viability targets, so the barrier assigns a negative margin to that command despite high likelihood, and we expect the gap to appear specifically on high-confidence infeasible commands, with the log-prob gate showing near-zero improvement there while our method recovers a substantial fraction of those episodes.

**vs. π0.5 + learned CBF trained with HJ-reachability-style Bellman backups** — On a case where the feasible set boundary is continuous and the command is near the boundary, the Bellman-backup CBF may overfit to the discrete backup targets and produce a looser approximation of the viability kernel, leading to more false positives (accepting infeasible commands) or false negatives (rejecting feasible commands). Our method instead uses the contrastive margin loss with the discrete-time CBF condition, which directly penalizes boundary violations, so we expect a lower false-positive rate on boundary commands and a higher success rate on episodes that require precise feasibility discrimination, while both methods match on clearly feasible or clearly infeasible commands.

### What would falsify this idea

If our gain over the π0.5 command head is uniform across all episodes rather than concentrated on episodes where the predicted infeasibility trigger occurs, or if the ablation without L_cons or without the barrier decrease condition matches our full method, then the barrier margin is not measuring controller feasibility or forward invariance and the central claim is wrong.

## References

1. π0.5: a Vision-Language-Action Model with Open-World Generalization
2. PaliGemma: A versatile 3B VLM for transfer
3. π0: A Vision-Language-Action Flow Model for General Robot Control
4. Visual Instruction Tuning
5. RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control
6. RT-1: Robotics Transformer for Real-World Control at Scale
7. Self-Instruct: Aligning Language Models with Self-Generated Instructions
8. Scaling Instruction-Finetuned Language Models
