# Jacobian-Matched Latent Action World Models: Identifying the Action That Causes the Transition

## Motivation

Both DreamGen and Contrastive World Models decouple the dynamics model from the action model — DreamGen recovers pseudo-actions post-hoc from a frozen video generator with no consistency to the true robot transition kernel, and Contrastive World Models learns action-conditioned latents under a mutual-information bound that admits many latent/action decompositions fitting the same future frames. The structural consequence is that in neither tree is the action variable constrained to be the unique cause of the predicted transition: any of several action codes can explain the same observation, so the recovered action is merely predictive rather than identifiable. This is the meta-level gap the field now hits as generative and contrastive backbones mature without a dynamical-consistency certificate tying action to transition.

## Key Insight

If the recovered pseudo-action is pushed through the SAME latent dynamics operator as the true robot action and that operator's action-conditioned Jacobian is required to equal the empirical transition Jacobian, the action-to-transition map becomes identifiable up to a known group action because the transition kernel's action-sensitivity is a structural invariant that no alternative decomposition can preserve.

## Method

**What it is.** JAM-WM (Jacobian-Matched World Model) learns a video world model whose latent action code `a_t` is tied to the true robot action `u_t` through one shared identifiable latent dynamics operator `F_θ`, with a dynamical-consistency loss that matches the action-conditioned Jacobian of `F_θ` to the empirical transition Jacobian estimated from real robot rollouts. The load-bearing assumption, stated explicitly: the empirical transition Jacobian `J_true(u_t) = dE_φ(o_{t+1})/du_t`, estimated by finite differences through the learned encoder `E_φ`, is a well-defined, non-degenerate, and encoder-invariant signature of the true action-to-transition map, so that matching `F_θ`'s action-Jacobian to it uniquely pins the pseudo-action to the true action up to `O(d)`. To make this assumption testable rather than assumed, `E_φ` is anchored during Phase 2 (frozen after Phase 1, or regularized by an encoder-rigidity penalty) and `J_true` is computed in observation space as `dT(o_t,u_t)/du_t` via finite differences on real rollouts, then pushed through the fixed encoder Jacobian `dE_φ/do_t`; the Procrustes alignment is solved per-state (or locally conditioned) to match the state-dependence of the encoder Jacobian, not as a single global `R`.

**How it works.**

```pseudocode
Inputs: video frames x_{0:T}, real robot transitions (o_t, u_t, o_{t+1}) from a small teleop set (N_teleop = 512 transitions)
Encoder E_φ: frame -> latent z_t (ResNet-18, latent dim d_z = 64)
Shared dynamics operator F_θ: (z_t, a_t) -> z_{t+1}, with a_t in R^d (d = 7, robot arm DoF); F_θ is a 3-layer MLP, hidden=256, GeLU
Pseudo-action encoder Q_ψ: (z_t, z_{t+1}) -> a_t   # recovers latent action from video alone; 2-layer MLP, hidden=256, GeLU

Phase 1 - Generative world model (DreamGen-style backbone):
  for each video clip:
    z_t = E_φ(x_t)
    a_t = Q_ψ(z_t, z_{t+1})                 # pseudo-action from future latent
    z_hat_{t+1} = F_θ(z_t, a_t)            # shared operator predicts transition
    L_gen = ||z_hat_{t+1} - sg(E_φ(x_{t+1}))||^2   # sg = stop-gradient target
  # After Phase 1, freeze E_φ (or apply encoder-rigidity penalty λ_rig * ||dE_φ/do_t||_F^2 with λ_rig=1e-3)

Phase 2 - Dynamical-consistency constraint (the certificate):
  # Estimate empirical transition Jacobian in OBSERVATION space from REAL robot data
  for each real transition (o_t, u_t, o_{t+1}):
    J_obs(u_t) = dT(o_t, u_t) / d u_t   # finite-difference over local u-neighborhood, step ε=1e-3
    J_true(u_t) = J_obs(u_t) @ (dE_φ/do_t)|_{o_{t+1}}   # push through fixed encoder Jacobian
  # Action-conditioned Jacobian of the SHARED operator, evaluated at the pseudo-action
  J_pred(a_t) = d F_θ(z_t, a) / d a |_{a=a_t}
  L_dyn = || J_pred(a_t) - J_true(u_t) R_t ||_F^2          # Frobenius norm, matched up to a per-state group action G_t
  # Group action: a_t and u_t identified up to a state-dependent orthogonal transform R_t in G=O(d),
  # so match after solving min_{R_t} || J_pred(a_t) - J_true(u_t) R_t ||_F^2 via Procrustes per state

Phase 3 - Joint training:
  L = L_gen + λ_dyn * L_dyn + λ_reg * ||a_t||^2 + λ_rig * ||dE_φ/do_t||_F^2,  λ_dyn=1.0, λ_reg=1e-4, λ_rig=1e-3
  Update E_φ, F_θ, Q_ψ jointly; Procrustes R_t re-solved every 100 steps (per-state, not global)
```

**Why this design.** We chose a *single shared* dynamics operator `F_θ` for both pseudo-actions and true robot actions over two separate heads because a shared operator is what makes the Jacobian match meaningful — if the two action streams passed through different operators, matching their Jacobians would be vacuous, accepting the cost that `F_θ` must simultaneously fit video-derived latents and robot transitions, which can slow convergence. We chose to match the action-conditioned *Jacobian* rather than the raw transition because the Jacobian isolates action-sensitivity from scene content, accepting the cost of a finite-difference estimate that is noisy for small ε. We chose a fixed orthogonal group `G=O(d)` solved by Procrustes over a learned alignment because a learned alignment would reintroduce the non-identifiability we are trying to remove, accepting the restriction that identification holds only up to rotation/reflection. A domain expert would not describe this as a variant of Contrastive World Models: CWM maximizes a mutual-information bound between states and future patches, whereas JAM-WM imposes a differential-geometric equality between two Jacobians through a shared operator — a constraint CWM's MI objective cannot express, since MI is invariant to any invertible reparameterization of the action code.

**Why it measures what we claim.** The Frobenius distance `||J_pred(a_t) - J_true(u_t) R_t||_F^2` measures *dynamical consistency of the recovered action* because the assumption that the true transition kernel is locally differentiable in `u` with non-degenerate action-Jacobian makes the Jacobian a complete local signature of action-sensitivity; this assumption fails when the robot is near a kinematic singularity or when `u` has redundant dimensions, in which case the metric reflects finite-difference noise rather than a genuine consistency gap. The per-state Procrustes residual `min_{R_t}` measures *identifiability up to the group action G* because the assumption that any two action codes explaining the same transitions differ by an element of `O(d)` (a standard linear-identifiability condition on the operator's action argument) makes the residual zero exactly when the codes are group-equivalent; this assumption fails when the operator is nonlinear in `a` beyond the local neighborhood, in which case the residual reflects curvature rather than non-identifiability. The pseudo-action `a_t = Q_ψ(z_t, z_{t+1})` measures *the cause of the transition* because the assumption that `z_{t+1}` is generated from `(z_t, a_t)` through `F_θ` makes `Q_ψ` an inverse of `F_θ`; this assumption fails when the encoder `E_φ` collapses distinct transitions to the same latent, in which case `a_t` reflects encoder aliasing rather than the causal action.

## Contribution

(1) A video world model training framework, JAM-WM, that couples a post-hoc pseudo-action encoder to the true robot transition kernel through a single shared latent dynamics operator and a Jacobian-matching dynamical-consistency loss, giving an action-to-transition map identifiable up to a fixed orthogonal group action. (2) A Procrustes-based group-action alignment procedure that converts the Jacobian-matching objective into a well-posed optimization over the action code, and a finite-difference estimator of the empirical transition Jacobian from a small teleop set. (3) A design principle: action identifiability in video world models follows from matching differential action-sensitivity through a shared operator rather than from tightening a contrastive mutual-information bound, which is invariant to action reparameterization.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | Single-task teleop rollouts (N_teleop = 512 transitions) + unseen-behavior videos (N_video = 10,000 frames) | Small real-action set anchors Jacobian; video tests generalization |
| Primary metric | Action-recovery error after per-state Procrustes alignment (mean squared error between recovered `a_t` and `u_t R_t`) | Directly measures identifiability up to G |
| Baseline | DreamGen pseudo-action pipeline | Tests whether Jacobian matching beats generative-only recovery |
| Baseline | Contrastive World Model (CWM) | Tests whether differential constraint beats MI objective |
| Baseline | DreamerV3 latent-action head | Tests whether shared operator beats reconstruction-driven codes |
| Ablation-of-ours | JAM-WM without L_dyn (λ_dyn=0) | Isolates the dynamical-consistency certificate |
| Ablation-of-ours | JAM-WM with global (not per-state) Procrustes R | Tests whether state-dependent alignment is necessary |
| Ablation-of-ours | JAM-WM with frozen E_φ vs. joint training | Tests the encoder-rigidity assumption |

### Why this setup validates the claim

The central claim is that matching the action-conditioned Jacobian of a *shared* operator to the empirical transition Jacobian yields pseudo-actions that are identifiable up to `O(d)` and causally faithful, whereas generative-only or MI-based recovery leaves the action code free to absorb scene content. The dataset choice is what makes this falsifiable: real teleop transitions supply the ground-truth Jacobian `J_true(u_t)` against which recovered pseudo-actions can be scored, while the video-only behaviors supply cases where no real action exists, so any recovery must come from the shared operator rather than memorized action labels. The primary metric — action-recovery error after per-state Procrustes alignment — is the right detector because it is exactly the quantity the identifiability argument predicts should shrink: if `L_dyn` truly pins `a_t` to `u_t` up to `O(d)`, the residual after the optimal rotation should be small and should stay small on held-out behaviors, not just on the teleop distribution. Each baseline isolates one mechanism gap: DreamGen tests whether generative pseudo-action recovery alone suffices, CWM tests whether an MI bound can substitute for a differential equality, and DreamerV3's latent-action head tests whether reconstruction-driven codes carry action-sensitivity at all. The ablation `λ_dyn=0` is the sharpest control: it is the same architecture with the certificate removed, so any gain of full JAM-WM over it is attributable to the Jacobian match and nothing else. The additional ablations (global vs. per-state Procrustes, frozen vs. joint encoder) directly test the load-bearing assumption that `J_true` is encoder-invariant and that the required alignment is state-dependent. Together these form a falsifiable test: the claim survives only if the full method beats all three baselines and the ablation *specifically* on the subsets where action-sensitivity is the bottleneck, and if the per-state Procrustes residual is lower than the global-Procrustes residual on held-out behaviors.

### Expected outcome and causal chain

**vs. DreamGen pseudo-action pipeline** — On a case where the video contains a distractor moving independently of the robot (e.g., a sliding object or lighting change), DreamGen's pseudo-action encoder, trained only through generative reconstruction, produces an action code that partially encodes the distractor motion because reconstruction rewards explaining all pixels. Our method instead suppresses this because `L_dyn` forces the action-conditioned Jacobian of `F_θ` to match the real transition Jacobian, which is blind to scene content by construction. We expect a noticeable gap in action-recovery error on distractor-heavy clips but near parity on clean clips — the signature of a content-invariance gain rather than a general capacity gain.

**vs. Contrastive World Model (CWM)** — On a case where two distinct robot actions produce visually similar next frames (e.g., small translations near a symmetric object), CWM's MI objective is satisfied by any invertible reparameterization of the action code, so it can assign codes that are mutually informative yet not linearly aligned with `u_t`; the recovered action then drifts under Procrustes. Our method instead anchors the code through the shared operator's differential signature, so the per-state Procrustes residual stays bounded because identification holds up to `O(d)` by construction. We expect CWM to show a heavy tail of large recovery errors on these ambiguous transitions while JAM-WM shows a tight distribution, with the gap concentrated on the ambiguous subset.

**vs. DreamerV3 latent-action head** — On a case where the robot is near a kinematic singularity (e.g., fully extended arm), the true action-Jacobian becomes near-degenerate, and DreamerV3's reconstruction-driven latent action, which never sees a differential constraint, produces codes whose sensitivity does not track the true sensitivity — it either saturates or becomes noisy. Our method instead inherits the finite-difference estimate `J_true`, so even though the estimate is noisy at small `ε`, the shared operator is pushed toward the correct local sensitivity and the pseudo-action remains monotone in `u_t`. We expect DreamerV3 to show a sharp error spike on singularity-adjacent transitions and JAM-WM to show a milder, bounded increase — parity away from singularities, divergence at them.

**vs. JAM-WM without L_dyn (ablation)** — On a case where the teleop set is small relative to video diversity, the ablation's pseudo-action encoder is free to fit video transitions with codes that are internally consistent but not aligned to `u_t`, since nothing ties `a_t` to real actions. Our method instead uses the Jacobian match as the only bridge between the two action streams, so even sparse real transitions constrain the code. We expect the ablation to match full JAM-WM on in-distribution teleop clips but degrade substantially on held-out behaviors, isolating the certificate as the source of generalization.

**vs. JAM-WM with global Procrustes R (ablation)** — On a case where the encoder Jacobian `dE_φ/do_t` varies significantly across states (e.g., different object poses), a single global rotation cannot align `J_pred` and `J_true` for all states, so the residual remains high even if the action code is correct. Our method uses per-state Procrustes, which should reduce the residual on these states. We expect the global-R ablation to show higher action-recovery error on states with high encoder-Jacobian variance, while matching full JAM-WM on states where the encoder Jacobian is nearly constant.

**vs. JAM-WM with joint (unfrozen) E_φ (ablation)** — On a case where the encoder is allowed to co-adapt with `L_dyn`, it can reshape its latent geometry to make both `J_pred` and `J_true` small and mutually consistent regardless of whether `a_t` tracks `u_t`, as warned by the load-bearing assumption. Our method freezes `E_φ` after Phase 1 (or applies the rigidity penalty), preventing this degenerate solution. We expect the joint-training ablation to show lower `L_dyn` but higher action-recovery error on held-out behaviors, revealing that the loss was gamed rather than satisfied.

### What would falsify this idea

If full JAM-WM's advantage over the `λ_dyn=0` ablation and over CWM is uniform across all subsets — including distractor-free, non-singular, unambiguous transitions — rather than concentrated on the subsets where the predicted mechanism gap occurs, then the Jacobian match is not doing the causal work claimed and the central identifiability argument is wrong. A second falsifier: if the per-state Procrustes residual does not shrink relative to a randomly rotated baseline on held-out behaviors, or if the frozen-encoder variant does not outperform the joint-training variant on action-recovery error, the identification-up-to-`O(d)` claim fails.

## References

1. DreamGen: Unlocking Generalization in Robot Learning through Video World Models
2. HunyuanVideo: A Systematic Framework For Large Video Generative Models
3. Moto: Latent Motion Token as the Bridging Language for Learning Robot Manipulation from Videos
4. Contrastive World Models
5. Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture
6. Mastering diverse control tasks through world models
7. Context Autoencoder for Self-supervised Representation Learning
8. Masked Siamese Networks for Label-Efficient Learning
