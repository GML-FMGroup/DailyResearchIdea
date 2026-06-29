# Meta-Discovery of Velocity Fields for Unsupervised Capability Emergence in Flow-Matching Models

## Motivation

DanceOPD assumes capability fields are predefined (e.g., T2I, editing), limiting the model to known tasks and preventing discovery of novel velocity patterns from unlabeled data. Without a mechanism to identify new fields automatically, the model cannot adapt to unseen data distributions or task structures. This structural limitation arises because DanceOPD inherits the assumption that capabilities are evaluation categories rather than learnable structures from the GPT-4o study.

## Key Insight

By optimizing a meta-learning objective that measures the predictive performance of a learned capability head on self-generated rollouts, the model can discover latent velocity field structures that generalize across diverse unlabeled data.

## Method

(A) **What it is**: MCD is a meta-learning framework that trains a capability discovery head θ (a 2-layer MLP with hidden size 256, ReLU activation, output size K with softmax) to predict which velocity field (capability) to apply at each flow state, using self-supervised objectives on unlabeled data. Input: flow state s_t; output: a capability index k ∈ {1,...,K} (K grows dynamically: if average entropy of output over a batch falls below 0.1 for 10 consecutive meta-epochs, K ← K + 1).

(B) **How it works (pseudocode)**

```python
# Predefine K0=4 initial capabilities (e.g., from DanceOPD: T2I, editing, style transfer, inpainting) or random fields.
Initialize student model s with velocity fields {v_k}.
Initialize discovery head θ with random weights (Glorot uniform).

for meta_epoch in range(100):  # M=100
    # Sample unlabeled data batch D={x_0, x_1} with batch size 64
    for (x_0, x_1) in D:
        # Outer loop: candidate field discovery
        # Rollout student from x_0 to x_1 using its current fields (on-policy), N=10 steps, dt=1/N
        trajectory = []
        state = x_0
        for t in linspace(0,1,N):
            k = θ(state)  # predict capability (softmax, sample or argmax? use argmax during outer loop for stability)
            v = v_k(state)  # get velocity from student
            state = state + v * dt
            trajectory.append((state, k))
        
        # Compute meta-objective: MSE between predicted velocity and pseudo-target
        # pseudo-target v_target = (next_state - state)/dt from the rollout (finite difference)
        loss_meta = mean(||v_k(s) - v_target||^2) over all transitions
        
        # Add entropy bonus to encourage diverse field selection
        entropy = -mean(sum over k of p_k * log(p_k)) with p_k = softmax(θ(state))
        loss_meta_total = loss_meta - 0.01 * entropy
        
        # Update discovery head θ using Adam with lr_outer=1e-4
        θ -= lr_outer * grad(loss_meta_total, θ)
        
        # Inner loop: update student using predicted fields (similar to DanceOPD)
        # For each sample, predict capability using θ (with argmax) and compute student loss
        k = argmax(θ(state))  # or sample? use argmax for stability
        v_target = v_k_target_computed_from_teacher_or_single_step?  # use teacher model's predefined velocity for that capability (as in DanceOPD) or compute via conditional flow matching from teacher
        student_loss = MSE(v_k(student_state), v_target)
        update student parameters via SGD with lr_inner=1e-4
    
    # Diagnostic check every 10 meta-epochs:
    if meta_epoch % 10 == 0 and meta_epoch > 0:
        # Compute alignment between pseudo-target and teacher-based conditional flow matching loss on a held-out calibration set of 1000 pairs
        alignment = cosine_similarity(pseudo_target, teacher_target) averaged over calibration set
        if alignment < 0.9:
            lr_outer *= 0.5
            reinitialize θ with smaller variance (scale 0.1)
```

**Hyperparameters**: K0=4; M=100; N=10; lr_outer=1e-4; lr_inner=1e-4; batch_size=64; optimizer=Adam for θ, SGD for student; entropy_weight=0.01; warm-start: first 10 epochs use predefined fields from DanceOPD (no θ updates).

(C) **Why this design**: We chose a meta-learning outer loop over clustering or heuristic field assignment because it directly optimizes the discovery head for downstream task performance, accepting higher training cost and potential instability. We use on-policy rollouts to align with DanceOPD's core mechanism, but this introduces high variance early in training; we mitigate by warm-starting with predefined fields for the first 10% of meta-epochs. We select a separate discovery head rather than a unified architecture to maintain modularity and allow incremental addition of new capabilities without retraining the entire student. The trade-off is that the head may overfit to frequent patterns, missing rare ones; we address this by adding an entropy bonus to encourage diverse field selection.

(D) **Why it measures what we claim**: The meta-objective `loss_meta` (mean squared error between predicted velocity and pseudo-target) measures the utility of the predicted field for the student's learning because, under the assumption that the velocity field that best predicts the next state on the student's own trajectory will also reduce the student's overall flow-matching error on future data. This assumption fails when the pseudo-target is noisy due to discrete time steps or when the student's current fields are poor, causing `loss_meta` to reflect artifacts of bad rollouts rather than true field utility. Specifically, `loss_meta` measures field utility assuming the pseudo-target (next_state - state)/dt approximates the true optimal velocity for that capability under the student's dynamics; this assumption fails when time discretization error is large (dt too large) or the student fields are far from optimal (early training). To diagnose failure, we compute alignment with teacher-based conditional flow matching loss on a calibration set every 10 meta-epochs; if alignment drops below cosine similarity 0.9, we reduce outer loop learning rate and reinitialize θ. Additionally, the capability discovery head's output `θ(state)` measures the relevance of each field at a given state because it is trained to minimize `loss_meta`, which ties field selection to improvement in the student's dynamics; this assumption fails when multiple fields yield similar improvements, in which case the head may arbitrarily split the state space among them.

## Contribution

(1) A meta-learning framework for unsupervised capability discovery in flow-matching models, enabling emergence of new velocity field patterns from unlabeled data. (2) An algorithm that trains a discovery head to predict capability indices by optimizing a meta-objective tied to the student's flow-matching performance. (3) Demonstration of emergent capability composition on synthetic datasets with structured velocity fields.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
|------|--------|-----------|
| Dataset (initial) | CIFAR-10 | Small-scale, quick debugging; validate mechanism before scaling |
| Dataset (main) | ImageNet-1K | Large-scale, diverse images |
| Synthetic dataset | Custom dataset with 10 hidden velocity fields (e.g., rotation, scaling, brightness, etc.) | Test discovery of unknown fields; ground truth available |
| Primary metric | FID | Standard for generation quality |
| Baseline 1 | DanceOPD | Direct on-policy distillation |
| Baseline 2 | Standard KD | Off-policy teacher-student |
| Baseline 3 | MultiDiffusion | Multi-field fusion method |
| Ablation 1 | MCD w/o meta-loop (θ fixed to random) | Tests necessity of discovery head |
| Ablation 2 | MCD with randomly initialized fields (K0=4 random velocities) | Tests whether learned fields are better than random |

### Why this setup validates the claim

The combination of baselines directly isolates MCD's core mechanism: state-dependent field selection via meta-learning. DanceOPD uses predefined fields, testing whether learning fields helps; Standard KD lacks on-policy rollouts and field selection, testing their combined benefit; MultiDiffusion uses multiple fields but without learned selection, testing the selection versus fusion. The ablations (w/o meta-loop and with random fields) test whether the meta-loop and learned fields are essential. The synthetic dataset provides a controlled setting where the true velocity field structure is known, allowing direct measurement of discovery accuracy (e.g., field assignment accuracy >85%). Initial validation on CIFAR-10 ensures the method is feasible before scaling to ImageNet-1K. FID as a metric captures global generation quality, which should improve if field selection better matches data modes. Any significant FID improvement over all baselines would support the claim, while failure would indicate the mechanism is ineffective.

### Expected outcome and causal chain

**vs. DanceOPD** — On a case where a predefined field (e.g., smooth motion) is poor for fine textures, DanceOPD applies a fixed velocity, producing blurry results. Our method discovers a specialized field for that region via meta-learning, yielding sharper details. We expect a noticeable FID gap (e.g., 2–3 points lower) on texture-rich subsets like animals or fabrics, with parity on smooth regions like skies.

**vs. Standard KD** — On a case where teacher-student capacity mismatch is high (e.g., large teacher), off-policy KD ignores student's own trajectory, causing error accumulation in regions where student struggles. Our method uses on-policy rollouts and adapts fields to student's weaknesses, reducing error accumulation. We expect a clear FID advantage (e.g., 3–5 points lower) overall, especially on complex scenes where mismatch is typical.

**vs. MultiDiffusion** — On a case requiring composition of multiple distinct objects (e.g., a dog next to a car), MultiDiffusion fuses fields uniformly, causing boundary artifacts or inconsistent styles. Our method selects appropriate fields per state, leading to coherent boundaries and style consistency. We expect lower FID (e.g., 2–4 points) on compositional subsets and comparable performance on single-object images.

**Synthetic dataset** — On the custom synthetic dataset with 10 known hidden fields, we expect MCD to recover the ground-truth field assignments with >85% accuracy (measured by overlap of assigned states per field). This confirms that the meta-objective is effective at discovering latent field structures.

### What would falsify this idea

If the FID improvement over baselines is uniform across all data subsets (including those where field selection should be irrelevant, e.g., uniform textures), then the claim that state-dependent capability discovery drives gains would be falsified. Additionally, if MCD fails to recover hidden fields on the synthetic dataset (accuracy < random baseline), the core mechanism is invalid.

## References

1. DanceOPD: On-Policy Generative Field Distillation
2. On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes
3. MultiDiffusion: Fusing Diffusion Paths for Controlled Image Generation
4. Stochastic Interpolants: A Unifying Framework for Flows and Diffusions
5. Large Language Models Are Reasoning Teachers
6. SpaText: Spatio-Textual Representation for Controllable Image Generation
7. Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow
8. Building Normalizing Flows with Stochastic Interpolants
