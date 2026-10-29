# Inverse Dynamics Adapter with Motion-Geometry Regularization for Embodiment Grounding

## Motivation

S² (Sensorimotor Alignment) provides a training-free metric correlating representation-motion alignment with policy success, but it is purely correlational: it cannot adapt a frozen video diffusion prior's motion features to a new robot's action space, leaving the alignment-to-action grounding unestablished. Vidar showed that a masked inverse dynamics adapter can learn such a mapping, yet S² dropped it, so the field lacks a mechanism to test whether alignment is causally sufficient for action grounding rather than merely predictive.

## Key Insight

A trainable inverse dynamics adapter, constrained by the S² motion-geometry alignment loss, forces the frozen video prior's features to become a sufficient statistic for the target robot's actions, making alignment a causal precondition for action grounding rather than a correlate.

## Method

**What it is** — We introduce the Inverse Dynamics Adapter (IDA), a lightweight trainable module that maps frozen video diffusion prior motion features to target-embodiment actions. IDA is trained jointly with a motion-geometry alignment loss (S²) that regularizes the adapted features to preserve the prior's motion structure.

**Load-bearing assumption (stated explicitly)** — IDA assumes the frozen video diffusion prior's features already contain a motion-geometry structure that is embodiment-agnostic and recoverable by a thin adapter, so that a 2-layer MLP plus an S² correlation regularizer can map them to a new robot's actions without the prior having been trained on that embodiment's visual/action domain. This is the transferability assumption that the 'frozen prior + linear probe' baseline is designed to test; prior work on frozen visual representations for control (R3M/VC-1 linear-probe evaluations) reports frozen features as predictive but not sufficient for fine-grained action regression across embodiments. We therefore do not assert sufficiency from the training loss; we operationalize it via a held-out per-embodiment calibration probe (step 7 below) whose residual is reported as the operational definition of 'sufficient statistic'.

**How it works** —
```pseudocode
Input: frozen video prior V, target robot dataset D = {(o_t, a_t, o_{t+1})}
Output: policy π_θ(a_t | o_t) = IDA_φ( V(o_t) )

1. Extract motion features: f_t = V(o_t)  # frozen, no gradients; f_t ∈ R^d (d = prior feature dim, e.g., 768)
2. IDA forward: â_t = MLP_φ(f_t)  # 2-layer MLP, hidden 256, ReLU, output dim = action dim
3. Action loss: L_act = MSE(â_t, a_t)
4. Alignment loss: compute S² on a batch of f_t and corresponding motion descriptors m_t (e.g., end-effector velocities); L_align = -corr( sim(f_i, f_j), sim(m_i, m_j) )
   # sim(·,·) = cosine similarity; corr = Pearson correlation over all pairs in the batch (batch size 256)
5. Total loss: L = L_act + λ * L_align, λ = 0.1
6. Update φ only; V frozen.
7. Per-embodiment calibration probe (operational sufficiency check): after training, fit a ridge regression readout a_t ≈ W · g_t + b on a held-out calibration set of 512 examples, where g_t = MLP_φ(f_t) are the adapted features; report the calibration residual (MSE) as the operational definition of 'sufficient statistic' for the target embodiment. This residual is reported alongside L_act, not used for training.
```

**Why this design** — We chose a trainable MLP adapter over fine-tuning the video prior because fine-tuning would destroy the motion geometry that S² relies on, accepting the cost that the adapter has limited capacity to correct systematic prior errors. We chose to keep the S² alignment loss as a regularizer rather than a hard constraint because a hard constraint could conflict with the action loss when the prior's motion geometry is imperfect for the target embodiment, accepting that the regularizer may be insufficient to guarantee alignment. We chose a simple MSE action loss over a generative action head (e.g., diffusion policy) because the goal is to test causal sufficiency of alignment, not to maximize task performance; a generative head would confound the alignment signal with action multimodality. We chose to freeze the video prior entirely to isolate the adapter's contribution, accepting that the prior may not be optimally suited to the robot's visual domain. A domain expert would not describe this as a variant of Vidar's masked inverse dynamics adapter because Vidar used masking as a self-supervised objective, whereas our adapter is trained with an explicit motion-geometry alignment loss that directly operationalizes S²'s metric as a training signal, creating a causal test that Vidar did not provide.

**Why it measures what we claim** — The action loss L_act measures action grounding because minimizing MSE between predicted and true actions ensures the adapted features are predictive of the robot's actions; this assumes the action space is fully observable and deterministic given the features, which fails when actions are multimodal, in which case L_act reflects average action rather than a calibrated policy. The alignment loss L_align measures motion consistency because it maximizes the correlation between representation similarity and motion descriptor similarity; this assumes motion descriptors (e.g., end-effector velocities) capture the relevant physical motion geometry, which fails when the descriptors are noisy or incomplete, in which case L_align reflects descriptor noise rather than true motion structure. The joint training measures causal sufficiency of alignment because the adapter is forced to use aligned features to predict actions; this assumes that if alignment is causally sufficient, then the adapter with L_align will achieve lower action loss than without, which fails if the adapter can learn to ignore misaligned features, in which case the comparison would not isolate alignment's causal role. The calibration residual (step 7) measures sufficiency because a low residual on held-out examples indicates the adapted features linearly encode the target actions; this assumes the calibration set is representative of the deployment distribution, which fails under distribution shift, in which case the residual reflects calibration-set bias rather than true sufficiency.

## Contribution

['A trainable inverse dynamics adapter (IDA) that maps frozen video diffusion prior features to target-embodiment actions, jointly optimized with an S²-based motion-geometry alignment loss.', 'A causal test of whether alignment is sufficient for action grounding by comparing policies with and without the alignment loss, isolating the effect of alignment from other factors.', 'An empirical demonstration that alignment loss improves action prediction on a new robot embodiment, providing evidence that alignment is causally sufficient for grounding.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | DROID robot manipulation episodes | Diverse real-world manipulation with paired observations and actions |
| Primary metric | Closed-loop task success rate | Ground-truth behavioral signal, not proxy correlation |
| Secondary metric | Calibration residual (ridge readout MSE on 512 held-out examples) | Operational definition of feature sufficiency |
| Baseline | Frozen prior + linear action probe | Tests whether adapter capacity matters vs. trivial readout |
| Baseline | Fine-tuned video prior + action head | Tests whether freezing is necessary for alignment |
| Baseline | IDA without S² alignment loss | Isolates the causal contribution of alignment |
| Ablation-of-ours | IDA with S² but shuffled motion descriptors | Tests whether alignment signal is genuine vs. incidental |

### Why this setup validates the claim

The central claim is that an explicit motion-geometry alignment loss (S²) applied to a frozen video prior's features is causally sufficient to produce usable target-embodiment actions, not merely correlated with them. This requires three contrasts. The linear-probe baseline tests whether the adapter's nonlinearity is doing real work or whether the frozen prior already linearly encodes actions — if linear matches IDA, the adapter is decoration. The fine-tuned-prior baseline tests the design's core bet: that fine-tuning destroys the motion geometry S² depends on, so a fine-tuned model should lose the alignment benefit even if its raw action loss looks competitive. The no-alignment baseline is the direct causal test — same architecture, same data, same capacity, only the S² term differs — so any gap is attributable to alignment itself. The shuffled-descriptor ablation guards against the possibility that L_align is a generic regularizer whose specific geometry content is irrelevant. Closed-loop success rate is the right metric because offline action MSE can improve while behavior degrades (average-seeking under multimodality); only rollouts reveal whether aligned features yield a policy that actually acts correctly. The calibration residual provides the operational definition of 'sufficient statistic': a low held-out ridge-readout MSE on adapted features indicates the features linearly encode target actions, directly testing the load-bearing assumption rather than asserting it from training loss. This combination makes the claim falsifiable: if alignment is causally sufficient, the gap should appear specifically where the prior's motion geometry is preserved and be absent where it is destroyed, and the calibration residual should track the success-rate gap.

### Expected outcome and causal chain

**vs. Frozen prior + linear action probe** — On cases requiring nonlinear mapping from prior features to embodiment-specific joint commands (e.g., gripper articulation coupled to arm configuration), the linear probe produces systematically biased actions because a single linear map cannot untangle the prior's entangled motion features into the target action manifold. Our method instead learns a nonlinear adapter that reshapes the frozen features while S² keeps their motion ordering intact, so we expect a clear success-rate gap on articulation-heavy tasks but near-parity on simple reaching tasks where the mapping is approximately linear. The calibration residual should be lower for IDA than for the linear probe on articulation-heavy tasks, confirming that the adapter's nonlinearity improves feature sufficiency.

**vs. Fine-tuned video prior + action head** — On episodes where the prior's motion structure would otherwise transfer (smooth end-effector trajectories with consistent velocity profiles), the fine-tuned baseline produces erratic or overfit action sequences because gradient updates on the action loss overwrite the motion geometry that S² would have exploited, leaving features that fit the training action distribution but misrepresent motion. Our method instead freezes the prior and only adapts a thin MLP under the alignment regularizer, so we expect the fine-tuned baseline to look competitive on in-distribution training-like episodes but to degrade sharply on held-out motion patterns, while IDA holds its success rate. The calibration residual for the fine-tuned baseline should be low on training-like episodes but high on held-out motion patterns, reflecting the destroyed motion geometry.

**vs. IDA without S² alignment loss** — On cases where the prior's motion features are misaligned with the target embodiment (e.g., a different arm's velocity scale or workspace geometry), the no-alignment adapter produces actions that fit the training MSE but drift under distribution shift because nothing constrains the adapted features to preserve motion structure, so the MLP can latch onto spurious visual correlates. Our method instead uses L_align to keep representation similarity tracking motion-descriptor similarity, so we expect the no-alignment variant to match IDA on the training distribution but show a noticeable success-rate drop on shifted initial conditions and novel object configurations, with the gap concentrated on episodes whose motion descriptors diverge most from the training mean. The calibration residual for the no-alignment variant should be higher on shifted conditions, indicating that without L_align the adapted features are less sufficient for actions under distribution shift.

### What would falsify this idea

If the S²-alignment gain is uniform across all task subsets — including those where the prior's motion geometry is demonstrably destroyed or where the shuffled-descriptor ablation performs identically — then L_align is acting as a generic regularizer rather than operationalizing motion-geometry alignment, and the causal-sufficiency claim is wrong. Equivalently, if the fine-tuned-prior baseline matches or beats IDA on held-out motion patterns while the no-alignment ablation matches IDA on shifted conditions, or if the calibration residual does not track the success-rate gap, the predicted failure path is absent and the design's core bet is refuted.

## References

1. Sensorimotor Alignment: A Platonic Proxy for Embodied Policy Success
2. RoboArena: Distributed Real-World Evaluation of Generalist Robot Policies
3. Robot Learning as an Empirical Science: Best Practices for Policy Evaluation
4. π0: A Vision-Language-Action Flow Model for General Robot Control
5. BridgeData V2: A Dataset for Robot Learning at Scale
6. Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware
7. RT-1: Robotics Transformer for Real-World Control at Scale
