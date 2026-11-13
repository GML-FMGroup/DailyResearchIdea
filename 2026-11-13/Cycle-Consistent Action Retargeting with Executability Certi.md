# Cycle-Consistent Action Retargeting with Executability Certification for In-the-Wild Demonstrations

## Motivation

Wild-teaching pipelines and demonstration-scale datasets such as Ropedia Xperience 2 assume that unconstrained egocentric human motion can be faithfully retargeted into whole-body robot trajectories and admitted directly into training data, yet no mechanism verifies that a retargeted action is even executable within the target environment's action-primitive and affordance set. The structural root cause is the absence of a bidirectional mapping: existing pipelines only carry actions forward from human space to robot space, so there is no way to test whether the mapped action corresponds to the recorded demonstration or whether it lies inside the simulator's executable set. This makes 'demonstration-space to environment-action-space closure' an unverified assumption that silently corrupts every downstream policy trained on the pair.

## Key Insight

A retargeting function that is certified by round-trip agreement (robot-space action mapped back to a human-space action that reproduces the original demonstration) AND by membership in the environment's executable affordance set is exactly the condition under which a wild demonstration and its simulated counterpart are informationally equivalent, so the pair can be admitted without further human inspection.

## Method

**What it is.** We introduce CARC (Cycle-certified Action Retargeting and Calibration), a module placed between in-the-wild human demonstrations and a converted interactive environment. Its inputs are a segmented human demonstration trajectory $\tau_h$ (hand/body pose stream plus object-centric contact events) and the target environment's action-primitive set $\mathcal{A}_e$ with its affordance predicate $\mathrm{Exec}: \mathcal{A}_e \times \mathcal{S} \to \{0,1\}$. Its outputs are a retargeted robot trajectory $\tau_r$ and a per-demonstration admission decision with a calibration certificate.

**How it works.**

```pseudocode
Input: human demo tau_h = [(p_t, c_t)]_{t=1..T}, env E with primitives A_e and Exec(.)
Hyperparams: lambda_cyc = 1.0, lambda_exec = 0.5, lambda_task = 1.0, K = 8 (perturbation samples), tau_admit = 0.85, beta = 0.1 (IB strength), tau_contrast = 0.07

# Phase 1: forward retargeter (learned)
f_theta: human pose/contact -> A_e      # 2-layer MLP, hidden=256, GeLU, input dim = 3*21 (hand joints) + 3 (contact) = 66, output dim = |A_e|
for t in 1..T:
    a_t = f_theta(p_t, c_t)             # robot-space action primitive + params

# Phase 2: inverse retargeter (learned, separate head)
g_phi: A_e -> human pose/contact
    p_hat_t = g_phi(a_t)                # 2-layer MLP, hidden=256, GeLU, input dim = |A_e|, output dim = 66

# Phase 3: task-semantics cycle-consistency loss (training signal)
# Instead of raw pose reconstruction, recover task-relevant features:
#   - object-centric contact events c_t (binary vector of length 5: grasp, push, lift, rotate, release)
#   - end-effector trajectory in object-relative coordinates e_t (3D position + 3D orientation)
# Define task feature extractor h: (p_t, c_t) -> (c_t, e_t) where e_t = p_t[hand] - p_t[object] (object pose from demo).
# g_phi outputs (c_hat_t, e_hat_t).
L_cyc = mean_t [ BCE(c_hat_t, c_t) + ||e_hat_t - e_t||^2 ]

# Phase 4: executability loss (training signal)
for t in 1..T:
    s_t = rollout_state(E, a_{1:t-1})   # simulator state before step t
    L_exec += (1 - Exec(a_t, s_t))^2

# Phase 5: information-bottleneck / sufficiency constraint (training signal)
# Penalize f_theta for discarding task-relevant content: maximize mutual information between a_t and task features (c_t, e_t).
# Use InfoNCE with temperature tau_contrast = 0.07 between a_t and (c_t, e_t) over a batch of 256 frames.
L_task = - InfoNCE( a_t, (c_t, e_t); tau=tau_contrast )

# Phase 6: train jointly
theta, phi <- minimize L_cyc + lambda_exec * L_exec + lambda_task * L_task + beta * ||theta||^2

# Phase 7: per-demonstration calibration certificate
for each demo d:
    cyc_err_d = mean_t [ BCE(c_hat_t, c_t) + ||e_hat_t - e_t|| ]   # task-semantics cycle error
    exec_rate_d = mean_t Exec(f_theta(p_t,c_t), s_t)
    # robustness: perturb human pose by K=8 samples (Gaussian noise sigma=0.01 rad on joint angles), check exec stability
    stab_d = mean_k mean_t Exec(f_theta(p_t + eps_k, c_t), s_t)
    score_d = w1*(1-cyc_err_d) + w2*exec_rate_d + w3*stab_d   # w1=0.4, w2=0.4, w3=0.2
    admit d iff score_d >= tau_admit
```

**Why this design.** We chose a *joint* cycle-consistency + executability objective over training the forward retargeter alone with supervised pose-matching because supervised matching cannot detect when the mapped action lies outside $\mathcal{A}_e$ — the environment's affordance predicate is the only signal that knows executability, so we make it a differentiable training term rather than a post-hoc filter, accepting that rollout-based Exec evaluation is expensive and requires a fast simulator. We chose a *separate inverse head* $g_\phi$ over a shared encoder because a shared encoder would let the inverse head trivially invert the forward head's own representation (collapsing the cycle loss to near-zero without constraining the forward map), while a separate head forces the forward map to be information-preserving about the human pose, accepting the cost of doubled parameters and slower convergence. We chose *per-demonstration admission with a perturbation-stability term* over a global threshold on cycle error because a single demo can have low cycle error yet be brittle to small pose noise (indicating the retargeter is memorizing rather than generalizing), accepting the cost of K=8 extra simulator rollouts per demo. We chose a *task-semantics cycle* (recovering contact events and object-relative end-effector trajectory) over raw pose reconstruction because raw pose cycle error is dominated by irreducible coarseness when robot primitives are coarser than human dexterity (e.g., two-finger gripper vs. five-finger pinch), and thus carries no signal about retargeting correctness; the task-semantics cycle focuses on what the demonstration is trying to achieve. We added an *information-bottleneck / sufficiency constraint* (InfoNCE between $a_t$ and task features) to explicitly penalize $f_\theta$ for discarding task-relevant content, addressing the cycle-consistency degeneracy where $g_\phi$ can invert $f_\theta$'s lossy encoding without preserving task semantics. This differs from Ropedia Xperience 2's one-directional human-to-robot retargeting: a domain expert would not describe CARC as a variant of that pipeline because Ropedia has no inverse map, no executability term, and no admission certificate — its retargeting is a fixed kinematic solver, not a learned bidirectional function certified against the environment's affordance set. It also differs from standard cycle-consistent imitation (e.g., CycleGAN-style domain translation) because the cycle here is grounded in a *physical executability predicate* rather than a distributional discriminator, and the cycle operates on task-semantics features rather than raw poses.

**Why this measures what we claim.** The task-semantics cycle error $\mathrm{BCE}(\hat{c}_t, c_t) + \|\hat{e}_t - e_t\|$ measures *demonstration-space to environment-action-space closure* because it assumes the inverse head $g_\phi$ can faithfully decode the robot action's task-relevant meaning (contact events and object-relative end-effector trajectory) — i.e., that the task-relevant human features and the robot action manifold are locally homeomorphic under $f_\theta$; this assumption fails when the robot's action primitives are strictly coarser than human dexterity (e.g., a two-finger gripper cannot represent a five-finger pinch), in which case the cycle error reflects irreducible representational coarseness rather than a retargeting failure. The executability rate $\mathrm{Exec}(f_\theta(p_t,c_t), s_t)$ measures *simulated affordance membership* because it assumes the simulator's affordance predicate is a sound over-approximation of physical executability — i.e., anything the simulator admits is physically realizable; this assumption fails when the simulator's contact model is more permissive than reality (e.g., allows interpenetration), in which case the executability rate reflects simulator optimism rather than true physical validity. The perturbation stability $\mathrm{stab}_d$ measures *retargeting generalization* because it assumes small pose perturbations should not change executability if the forward map has learned a smooth, non-memorizing function; this assumption fails when the demonstration itself sits near an affordance boundary (e.g., a grasp at the edge of the gripper's reach), in which case stability reflects boundary proximity rather than memorization. The admission score combines all three precisely so that no single proxy can admit a pair on its own.

## Contribution

["A bidirectional action-retargeting module (CARC) whose forward map is trained jointly with an inverse map under cycle consistency and a differentiable executability term grounded in the environment's affordance predicate, producing a per-demonstration calibration certificate rather than a one-directional kinematic conversion.", 'The design principle that demonstration admission should be gated by a three-term certificate (cycle error, executability rate, perturbation stability) rather than any single proxy, with each term measuring a distinct failure mode of the human-to-simulator mapping.', 'A reusable admission-scoring procedure and certificate format that can be layered onto existing wild-teaching and demonstration-scaling pipelines (e.g., Ropedia Xperience 2) without modifying their capture or simulation infrastructure.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Xperience 2 egocentric demos, scene-converted | In-the-wild human demos with paired executable environments |
| Primary metric | Admission precision on held-out demos | Directly tests whether certificate admits only executable retargetings |
| Secondary metric | Downstream policy success rate (trained on admitted demos) | Shows certificate value propagates to learning |
| Baseline | Ropedia fixed kinematic retargeter | One-directional solver with no executability or admission check |
| Baseline | Supervised pose-matching forward retargeter | Matches pose without knowing affordance set membership |
| Baseline | CycleGAN-style unidirectional cycle translation | Cycle without physical executability grounding |
| Ablation-of-ours | CARC minus executability term | Isolates whether Exec term drives the predicted gain |
| Ablation-of-ours | CARC minus task-semantics cycle (raw pose cycle) | Isolates whether task-semantics cycle improves over raw pose cycle |
| Ablation-of-ours | CARC minus information-bottleneck term | Isolates whether sufficiency constraint prevents information loss |

### Why this setup validates the claim

The central claim is that grounding the retargeting cycle in the environment's executability predicate, plus a perturbation-stability certificate, admits demonstrations that actually execute while rejecting brittle or infeasible ones. The fixed kinematic baseline tests the sub-claim that a learned bidirectional map is needed at all: if it matches CARC, the inverse head and Exec term are doing nothing. The supervised pose-matching baseline tests the sub-claim that pose matching alone cannot detect affordance violations. The CycleGAN-style baseline tests whether the gain comes specifically from grounding the cycle in Exec rather than from cycle-consistency as such. The ablation removing the Exec term isolates the executability signal. The ablation replacing the task-semantics cycle with raw pose cycle tests whether focusing on task-relevant features (contact events, object-relative end-effector trajectory) is necessary to avoid the cycle-consistency degeneracy where cycle error is dominated by irreducible coarseness. The ablation removing the information-bottleneck term tests whether the sufficiency constraint is needed to prevent $f_\theta$ from discarding task-relevant content. Admission precision on held-out demos is the right primary metric because it directly measures the certificate's decision quality — the thing CARC claims to produce — rather than downstream reward, which would confound retargeting with policy learning. The secondary metric of downstream policy success rate (training a policy on admitted demos and evaluating in simulation) directly tests whether the certificate's value propagates to learning, as suggested by the reviewer.

### Expected outcome and causal chain

**vs. Ropedia fixed kinematic retargeter** — On a case where a human demo uses a five-finger pinch on a thin object, the baseline produces a kinematically valid but non-executable gripper command because its solver has no affordance predicate to consult. Our method instead learns a forward map trained against Exec, so infeasible primitives are penalized during training, and the certificate flags the residual mismatch. We expect a noticeable gap in admission precision concentrated on contact-rich, fine-dexterity subsets, with parity on coarse whole-hand grasps.

**vs. Supervised pose-matching forward retargeter** — On a case where the human pose is well matched but the mapped action lies just outside the reachable affordance set, the baseline produces a confident but infeasible retargeting because pose loss is blind to environment constraints. Our method instead folds Exec into the training objective, so the forward map is pushed toward feasible primitives, and the stability term catches boundary-adjacent brittleness. We expect the gap to concentrate on demos near affordance boundaries, with parity on comfortably interior poses.

**vs. CycleGAN-style unidirectional cycle translation** — On a case where the discriminator is satisfied but the action is physically unrealizable, the baseline produces a low cycle error with high executability failure because its cycle is distributional, not physical. Our method instead grounds the cycle in Exec, so cycle closure and executability are jointly optimized. We expect the baseline to show near-zero cycle error yet markedly lower admission precision, while CARC's precision gap widens precisely where simulator-optimism cases occur.

**vs. CARC minus executability term** — On a case where the action is infeasible but cycle error is low, the ablation admits the demo because it lacks the Exec penalty. Our method rejects it. We expect the ablation to have lower admission precision, especially on contact-rich subsets, confirming the Exec term's necessity.

**vs. CARC minus task-semantics cycle (raw pose cycle)** — On a case where the robot gripper is coarser than human hand (e.g., two-finger vs. five-finger), the raw pose cycle error is high due to irreducible coarseness, leading to false rejection of feasible demos. Our task-semantics cycle focuses on contact events and object-relative trajectory, reducing coarseness sensitivity. We expect the ablation to have lower admission recall (more false negatives) on fine-dexterity demos, while our method maintains higher precision and recall.

**vs. CARC minus information-bottleneck term** — On a case where $f_\theta$ discards task-relevant details (e.g., contact timing), the cycle loss can still be low because $g_\phi$ inverts the lossy encoding. The ablation thus admits demos with poor task preservation. Our method penalizes information loss via InfoNCE, leading to better task preservation and higher downstream policy success. We expect the ablation to show lower policy success rate despite similar admission precision, revealing the sufficiency constraint's value.

### What would falsify this idea

If CARC's admission-precision gain over the supervised pose-matching baseline were uniform across all demo subsets rather than concentrated on contact-rich and affordance-boundary subsets, the claim that executability grounding is the causal driver would be wrong. Likewise, if the executability-ablation matched full CARC, the Exec term would be doing no work and the central mechanism claim would fail. Additionally, if the task-semantics cycle ablation matched full CARC, the task-semantics cycle would be unnecessary, and if the information-bottleneck ablation matched full CARC, the sufficiency constraint would be doing no work.

## References

1. Ropedia Xperience 2: A Gold Standard for Scaling Physical AI Environments and Behaviors
