# Forward-Inverse Consistency as Intrinsic Reward for Autonomous Robotic Manipulation

## Motivation

Existing methods like VLAC and human-in-the-loop RL (e.g., 'Precise and Dexterous Robotic Manipulation via Human-in-the-Loop RL') reduce but do not eliminate human involvement because they rely on human trajectories for training or human corrections during deployment. The root cause is the assumption that human oversight is necessary for robustness and safety. This structural assumption prevents full autonomy in robotic RL.

## Key Insight

Consistency between forward and inverse dynamics predictions provides a self-supervised signal that is naturally aligned with task success in goal-directed manipulation because any deviation from optimal behavior causes inconsistency in the cause-effect chain.

## Method

(A) **What it is**: We introduce the Forward-Inverse Consistency Critic (FICC), a method that uses two learned dynamics models—a forward model predicting the next observation from the current observation and action, and an inverse model predicting the action from the current and next observation—to compute an intrinsic reward. The discrepancy between the predicted and actual next observation (forward error) and between the taken action and the action inferred from the observed transition (inverse error) together form a consistency-based reward. When consistency is low, the robot triggers a self-correction by re-sampling an action via the inverse model and re-executing. **Load-bearing assumption:** The consistency error (forward prediction error + inverse prediction error) is a reliable indicator of task-relevant failure, and the inverse model's prediction can be used for effective self-correction if the error exceeds a threshold. To mitigate issues of non-uniqueness and aleatoric uncertainty, we use ensembles to estimate epistemic uncertainty and only trigger correction when model uncertainty is low.

(B) **How it works**: Pseudocode for one training iteration:

```python
# Hyperparameters
τ = 0.5  # consistency threshold (mse)
δ = 0.1  # correction trigger threshold (mean mse)
σ²_thresh = 0.01  # ensemble variance threshold
N = 5  # ensemble size

# Initialize models (ensembles)
F = [ForwardModel() for _ in range(N)]  # each: 2-layer MLP, hidden=256, GeLU activation
I = [InverseModel() for _ in range(N)]  # each: 2-layer MLP, hidden=256, GeLU activation
π = Policy()  # o_t -> a_t
B = ReplayBuffer()

while not converged:
  o = env.reset()
  for t in range(T):
    a = π(o)
    o_next = env.step(a)
    
    # Ensemble predictions
    o_pred = [Fk(o, a) for Fk in F]
    a_pred = [Ik(o, o_next) for Ik in I]
    
    forward_errs = [mse(o_next, op) for op in o_pred]
    inverse_errs = [mse(a, ap) for ap in a_pred]
    
    mean_forward = mean(forward_errs)
    mean_inverse = mean(inverse_errs)
    var_forward = var(forward_errs)
    var_inverse = var(inverse_errs)
    
    r = - (mean_forward + mean_inverse)  # intrinsic reward
    
    # Correction only if mean error high AND epistemic uncertainty low
    if (mean_forward > δ or mean_inverse > δ) and (var_forward < σ²_thresh and var_inverse < σ²_thresh):
      # self-correction: use best guess from inverse model (averaged over ensemble)
      a_corrected = mean([Ik(o, mean(o_pred)) for Ik in I])
      o_corrected = env.step(a_corrected)
      B.add((o, a_corrected, o_corrected, r))
    else:
      B.add((o, a, o_next, r))
    
    # update models
    sample = B.sample(batch_size)
    for Fk, Ik in zip(F, I):
      update Fk via MSE(sample.o_pred_k, sample.o_next)
      update Ik via MSE(sample.a_pred_k, sample.a)
    update π via SAC using rewards from buffer
```

(C) **Why this design**: We chose forward-inverse consistency over alternatives like contrastive learning or handcrafted rewards because it directly measures the internal coherence of the robot's action consequences. Three design decisions: (1) We use separate forward and inverse models rather than a single joint model because predicting both directions enforces a bidirectional consistency that is more robust to noise; the trade-off is increased computational cost but the benefits of a stronger training signal. (2) We apply correction by re-sampling actions via the inverse model on the predicted next state rather than simply discarding the transition, because this allows the robot to attempt to align its actual state with its expected trajectory, accepting the risk that the inverse model's prediction may be imperfect but leveraging it as a more context-aware correction than random exploration. (3) We use a threshold-based trigger for correction instead of continuous smoothing, because discrete correction actions are easier to integrate with RL and evaluate, though they introduce non-smooth dynamics. To make the trigger robust, we use model ensembles: correction is only executed when the mean consistency error exceeds δ and the ensemble variance (epistemic uncertainty) is low, thus avoiding corrections due to model uncertainty rather than true inconsistency. This design avoids human demonstrations or human-in-the-loop corrections, directly addressing the meta-gap of full autonomy.

(D) **Why it measures what we claim**: The forward prediction error (mse between actual and predicted next observation) measures the extent to which the robot's action produced an unexpected outcome, which is a proxy for task-relevant failure under the assumption that the world dynamics are deterministic and the models are well-trained; this assumption fails when the dynamics are stochastic or the models are undertrained, in which case the error reflects aleatoric uncertainty rather than true failure. The inverse prediction error (mse between taken action and action inferred from the observed transition) measures the consistency between the action taken and the action that would have been inferred, under the assumption that the inverse mapping is approximately unique; this assumption fails when multiple actions lead to the same next state (redundancy), in which case a high error may occur for a correct action. Together, these two errors operationalize the concept of 'consistency' that the motivation identifies as the key to autonomous reward: high consistency implies the robot's internal model of its own behavior is coherent, which correlates with successful task progression because failures typically disrupt the cause-effect chain. The correction trigger uses the same quantities to detect when the robot's expectation deviates enough to warrant intervention, but it also uses ensemble variance to ensure that correction is only applied when the models are confident, mitigating the failure modes above.

## Contribution

(1) A novel intrinsic reward mechanism (FICC) that uses forward-inverse model consistency to autonomously generate reward signals without any human input. (2) A self-correction loop that leverages inverse model predictions to automatically correct anomalous state transitions during rollouts, enabling full autonomy in robotic RL. (3) Design principle that bidirectional consistency serves as a dense, task-agnostic reward signal for manipulation.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|------------------------|
| Dataset | Simulated manipulation (MetaWorld, DMControl) | Diverse tasks with continuous state/action spaces. |
| Primary metric | Success rate on unseen test episodes | Direct measure of task completion. |
| Baseline 1 | RL with handcrafted reward | Standard engineering baseline; lower bound. |
| Baseline 2 | Behavior Cloning from demonstrations | Pure imitation; tests adaptation. |
| Baseline 3 | RL with only forward prediction error | Isolates benefit of inverse term. |
| Baseline 4 | RL with only inverse prediction error | Isolates benefit of forward term. |
| Ablation of ours | FICC without self-correction | Tests necessity of correction step. |

### Why this setup validates the claim
This combination of dataset, baselines, and metric directly tests the central claim that forward-inverse consistency and self-correction improve autonomous RL. The simulated manipulation suite provides a controlled environment with diverse dynamics, allowing generalization assessment. Handcrafted reward RL serves as a lower bound showing gains over manual engineering. Behavior cloning tests whether our method surpasses pure imitation. The forward-only baseline isolates the contribution of the inverse consistency term, and the inverse-only baseline isolates the forward term. The ablation of self-correction quantifies the benefit of action re-sampling. Success rate is the ultimate measure of task completion, making it ideal to detect predicted improvements. If our method excels where others fail due to poor reward or lack of adaptation, the hypothesis is confirmed. Additionally, we will test under increasing stochasticity (Gaussian noise on actions/observations) and analyze failure cases where consistency is high but task fails (e.g., local optima), to directly assess the load-bearing assumption.

### Expected outcome and causal chain

**vs. RL with handcrafted reward** — On a task with complex contact dynamics (e.g., peg insertion), a handcrafted reward may only reward final success, providing sparse feedback. The baseline struggles due to reward sparsity, often converging to suboptimal solutions. Our method generates dense intrinsic rewards from consistency, guiding exploration even without task reward. Thus, we expect a significant gap: baseline success ≤50%, ours ≥80% on such tasks.

**vs. Behavior Cloning** — On a task with novel object poses never seen in demonstrations (e.g., pick-and-place with rotated box), BC fails because it lacks ability to recover from errors. Our method uses RL to explore and self-correct via the inverse model. We expect BC to plateau at demonstration-level success (~60%) while ours generalizes, reaching >85% on varied initial poses.

**vs. RL with only forward prediction error** — On a task with action redundancy (multiple actions lead to same next state, e.g., pushing), forward error alone cannot distinguish correct vs. incorrect actions, leading to policy oscillation. Our method penalizes inverse inconsistency, stabilizing action selection. We expect the forward-only baseline to show slower convergence and ~15% lower final success rate compared to ours.

**vs. RL with only inverse prediction error** — On a task with high perceptual aliasing (different states look similar, e.g., visual occlusion), inverse error alone may be low while forward error is high, causing the policy to miss failures. Our joint signal addresses both. We expect the inverse-only baseline to plateau at ~65% success, while ours reaches >85%.

**Ablation: FICC without self-correction** — On a task where the policy occasionally takes erroneous actions (e.g., due to offline model inaccuracies), the baseline only uses consistency reward but cannot correct the current trajectory. Our full method triggers action re-sampling, providing immediate correction. We expect performance drop of 10-20% without correction, especially in high-error scenarios.

**Effect of stochasticity:** We will test on environments with increasing noise (Gaussian noise on actions and observations) and analyze performance degradation. We expect the method to maintain high success rate up to moderate noise levels, but degrade under high noise due to model uncertainty. This analysis directly tests the robustness of the load-bearing assumption.

**Failure case analysis:** We will manually inspect episodes where consistency is high but task fails (e.g., local optima) to identify limitations. This will inform future improvements.

### What would falsify this idea
If our method's success rate is not significantly higher than the forward-only and inverse-only baselines, or if the ablation without self-correction performs similarly to the full method, then the claimed benefits of joint consistency and correction are not supported. Additionally, if under moderate stochasticity the method degrades sharply, the assumption that consistency indicates failure may be invalid.

## References

1. A Vision-Language-Action-Critic Model for Robotic Real-World Reinforcement Learning
2. Precise and Dexterous Robotic Manipulation via Human-in-the-Loop Reinforcement Learning
3. π0: A Vision-Language-Action Flow Model for General Robot Control
4. Visual Instruction Tuning
5. RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control
6. RT-1: Robotics Transformer for Real-World Control at Scale
7. LAION-5B: An open large-scale dataset for training next generation image-text models
8. Self-Instruct: Aligning Language Models with Self-Generated Instructions
