# Self-Consistency Certification: Gating On-Policy Self-Supervision with Learned Forward Models

## Motivation

Existing on-policy self-supervision methods for language agents, such as the OPID framework (see Idea Cards), derive dense supervision from the student's own trajectories without external verification. They lack a structural property to guarantee that these trajectories contain informative feedback, especially when rollouts are low-quality or truncated, leading to unreliable supervision and suboptimal learning. This meta-gap persists across multiple approaches (e.g., GKD, SCoRe) that rely on self-generated data without formalizing when self-supervision is valid.

## Key Insight

The forward model's prediction error on a student trajectory provides a domain-agnostic indicator of trajectory self-consistency, enabling a certification that gates self-supervision to regions where the forward model is accurate, thereby ensuring self-supervision signal validity.

## Method

### (A) What it is
Self-Consistency Certification (SCC) is a framework that introduces a learned forward model to gate the use of on-policy self-supervision. It takes as input a student policy, a forward dynamics model, and environment interactions; it outputs a gated self-supervision loss and a regularization term that constrain the student to regions where the forward model is accurate.

**Load-bearing assumption**: The forward model's prediction error on a trajectory is a reliable indicator of that trajectory's self-consistency and the validity of self-supervision derived from it. This assumption is verified during training by periodically checking the correlation between prediction error and trajectory success on a calibration set.

### (B) How it works
```python
# Hyperparameters
threshold_error = 0.1  # maximum average prediction error for certification, tuned on calibration set
alpha = 0.01           # weight for regularization loss
calibration_set_size = 100  # number of trajectories to tune threshold
forward_model_hidden = 128  # hidden dimension of forward model MLP (2 layers, GeLU activation)
forward_model_lr = 1e-3
forward_model_buffer_size = 10000

# Initial calibration: collect trajectories from initial student, compute errors, tune threshold to maximize correlation with success
calib_trajs = student.rollout(env, n=calibration_set_size)
threshold_error = tune_threshold_on_calibration(calib_trajs, forward_model)

for iteration in range(num_iterations):
    # 1. Collect on-policy trajectories using current student policy
    trajectories = student.rollout(env)
    
    # 2. For each trajectory, compute forward model prediction errors
    for traj in trajectories:
        avg_error = 0
        for (state, action, next_state) in traj:
            pred_next = forward_model(state, action)
            avg_error += MSE(pred_next, next_state)
        avg_error /= len(traj)
        
        # 3. Certify trajectory if prediction error is below threshold
        if avg_error < threshold_error:
            # Generate self-supervision target (e.g., hindsight skill as in OPID)
            target = generate_skill_target(traj)
            supervise_loss = CE(student(state_seq), target)
        else:
            supervise_loss = 0.0
        
        # 4. Regularize student to stay in accurate regions
        reg_loss = alpha * avg_error
        total_loss = supervise_loss + reg_loss
        
        # 5. Update student policy via gradient descent on total_loss
        student.update(total_loss)
    
    # 6. Periodically update forward model on all trajectories (or a buffer)
    forward_model.train_on_batch(trajectories)  # minimizes prediction MSE
```

### (C) Why this design
We chose a learned forward model over simpler alternatives (e.g., outcome-based value functions) because it provides step-level dynamics consistency rather than final outcome proxy, enabling fine-grained certification even for partial trajectories. The threshold gating (binary) was chosen over continuous weighting because having a hard cutoff ensures self-supervision is only applied when the trajectory is well-predicted, avoiding noise accumulation from borderline cases; the cost is that some informative trajectories might be falsely rejected if the forward model is poor. Regularizing the student policy with the prediction error (reg_loss) is a design choice to actively push the student toward regions where the forward model is accurate, creating a co-adaptation loop. Alternative regularization, such as KL divergence to a reference policy, would not exploit the forward model's feedback. Finally, we update the forward model jointly with the student (off-policy from a buffer) to prevent it from overfitting to the student's current distribution, which could lead to overconfident gating. The trade-off is computational cost: the forward model must be trained concurrently, adding overhead.

### (D) Why it measures what we claim
The average forward model prediction error on a trajectory measures the **self-consistency** of that trajectory under the learned dynamics because it quantifies how well the student's actions and transitions match the predictable patterns seen during training; this assumes that the forward model has been trained on adequate data from the same task distribution. **Explicit proxy assumption**: low prediction error (X) is used as a proxy for trajectory utility (Y) under the assumption that the forward model captures relevant task dynamics. This assumption fails when the forward model is accurate but the self-supervision target is still incorrect (e.g., due to spurious correlations or when the forward model fits noise in the training data). In such cases, low prediction error does not guarantee a useful self-supervision signal. The regularization term (reg_loss) partially mitigates this by encouraging the student to stay in regions where the forward model is also accurate, thereby reducing the chance of relying on spurious correlations. Another failure mode occurs when the forward model sees out-of-distribution states (e.g., from a novel subtask), in which case high error reflects novelty rather than inconsistency, potentially causing the method to incorrectly reject informative trajectories. The gating threshold operationalizes the **validity** of self-supervision: trajectories with low error are deemed reliable because the dynamics are well-characterized, so self-supervision from them (e.g., hindsight skills) is likely consistent with the true environment. The regularization term (reg_loss) operationalizes the **student's adherence** to reliable regions: minimizing it pushes the student toward actions that keep the forward model accurate, maintaining the gating's effectiveness. The failure mode of this regularization is that it may reduce exploration of uncertain but potentially beneficial regions, a cost we accept.

## Contribution

(1) Introduction of Self-Consistency Certification (SCC), a framework that uses a learned forward model to gate on-policy self-supervision, ensuring validity through dynamics consistency. (2) A co-adaptation regularization that constrains the student policy to regions where the forward model is accurate, preserving the reliability of the gating mechanism. (3) Empirical demonstration across multiple language agent tasks (e.g., interactive QA, web navigation) that SCC improves sample efficiency and final task success rates compared to OPID and other on-policy self-supervision baselines.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | WebShop | Multi-step agentic language navigation |
| Primary metric | Success rate | Directly measures task completion |
| Baseline 1 | Outcome-based PPO | No self-supervision from trajectories |
| Baseline 2 | On-policy self-distillation without skills | Uses all trajectories naively |
| Baseline 3 | Off-policy skill distillation | External skill buffer mismatches dynamics |
| Ablation-of-ours | SCC w/o reg_loss | Isolates effect of regularization term |
| Additional Ablation | SCC with uncertainty-based gating (ensemble variance) | Compares against alternative gating mechanism |

**Computational budget**: Forward model is a 2-layer MLP (hidden=128, GeLU) trained with Adam (lr=1e-3) on a replay buffer of 10k transitions. Total compute ~50 GPU hours on a single A100. Threshold tuning on calibration set of 100 trajectories takes <0.5 GPU hours.

### Why this setup validates the claim
The combination of WebShop, success rate, and chosen baselines provides a falsifiable test of SCC's central claim: gating self-supervision via forward model prediction error yields more reliable learning. WebShop demands multi-step reasoning with partial feedback, where dynamics consistency is crucial. Outcome-based PPO isolates whether value-based self-supervision suffices; on-policy self-distillation without skills tests if naive trajectory use helps; off-policy skill distillation examines reliance on external data. The ablation of regularization tests whether the co-adaptation loop is necessary. The additional ablation with uncertainty-based gating tests the novelty of using forward model prediction error specifically (as opposed to ensemble uncertainty). Success rate is the appropriate metric because it reflects end-to-end performance, directly capturing the effect of more reliable self-supervision on task outcomes. If SCC outperforms baselines and the ablation underperforms, the claim is supported; if not, the idea is falsified.

### Expected outcome and causal chain

**vs. Outcome-based PPO** — On a case where an early subgoal is achieved but the final task fails (e.g., clicking a correct link but later getting lost), the baseline overestimates state values due to early reward, leading to suboptimal policy that repeats early successes. Our method gates out trajectories with high forward model error (indicating unreliable dynamics), so it avoids learning from partial successes where the rest of the transition is unpredictable. We thus expect a noticeable gap (e.g., 5-10% higher success rate) on tasks with long horizons or stochastic transitions, but parity on simple one-step tasks.

**vs. On-policy self-distillation without skills** — On a case where random exploration produces trajectories with inconsistent actions (e.g., typing unrelated queries), the baseline naively distills these sequences, injecting noise into the student. SCC's certification rejects trajectories with high average prediction error, filtering out such noise. Therefore, we expect SCC to maintain stable success rate (within 2% of initial) while the baseline degrades by 5-8% over training as it accumulates errors.

**vs. Off-policy skill distillation** — On a case where the external skill buffer contains skills for different environments (e.g., Wikipedia QA skills transferred to WebShop), the baseline suffers from distribution mismatch, often failing to execute correct actions. SCC uses only on-policy trajectories and conditionally applies self-supervision ensuring dynamics consistency, adapting to the current environment. Thus, we expect SCC to outperform by 8-12% on tasks that require environment-specific knowledge, with smaller gaps on generic tasks.

**vs. SCC with uncertainty-based gating** — On a case where the forward model's ensemble uncertainty is high due to distribution shift but prediction error is low (e.g., on a novel but predictable transition), the alternative gating would incorrectly reject the trajectory. SCC's prediction-error gating would correctly accept it, leading to better use of informative trajectories. We expect SCC to outperform by 2-4% on such tasks.

### What would falsify this idea
If SCC's success rate is not significantly higher than outcome-based PPO on long-horizon tasks, or if the improvement is uniform across all task types (rather than concentrated where forward model error is high), then the claim that gating via forward model improves learning is invalid.

## References

1. OPID: On-Policy Skill Distillation for Agentic Reinforcement Learning
2. On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes
3. Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning
4. Large Language Models Are Reasoning Teachers
5. The Llama 3 Herd of Models
6. Qwen2.5 Technical Report
7. Grounding by Trying: LLMs with Reinforcement Learning-Enhanced Retrieval
8. Training Language Models to Self-Correct via Reinforcement Learning
