# Causal Action-Consistent Latent Dynamics: Enforcing Intervention Structure for Robot Planning

## Motivation

Existing joint video-action models like DiT4DiT learn latent features that are sufficient for action prediction but may capture spurious correlations, lacking causal necessity: actions can be predicted without isolating action-dependent video features. This structural coupling prevents reliable iterative planning because future-state simulation requires a causal model where interventions on actions produce corresponding latent changes. Without causal decomposition, planning by simulating future latents risks propagating confounders rather than true action effects.

## Key Insight

By modeling actions as low-rank interventions on a continuous-time latent ODE and enforcing bi-directional consistency between forward dynamics and inverse dynamics, we isolate action-necessary latent dimensions, making video features causally necessary for action prediction.

## Method

(A) **What it is**: CACLD (Causal Action-Consistent Latent Dynamics) is a latent dynamics model that learns a causally factored representation where actions are low-rank additive interventions on a controlled ODE, with a soft consistency constraint ensuring forward-predicted latents and inverse-predicted actions form a closed loop. Input: observation history o_{1:t}, actions a_{1:t-1}. Output: causally disentangled latent trajectory z_{1:T} and action-conditioned future latents.

(B) **How it works** (pseudocode):
```python
# Given encoder E that maps observations o_t to latent z_t (dim D)
# Learnable ODE function f(z, t) parameterized by neural network
# Learnable low-rank intervention matrix B (dim D x K, K << D) and map g: action -> intervention magnitude
# Forward dynamics: 
#   dz/dt = f(z, t) + B @ g(a_t)   # action as low-rank perturbation
# Solve ODE forward to get predicted z_{t+1} from z_t and a_t
# Inverse dynamics:
#   a_hat_t = h(z_{t+1}, z_t)   # predict action from consecutive latents
#   Consistency loss: L_cons = MSE(a_t, a_hat_t)
#   Reconstruction loss: L_rec = MSE(o_t, decoder(z_t))
#   Regularization: L_rank = ||B||_2,1 to enforce low-rank
#   Total loss: L = L_rec + λ1 L_cons + λ2 L_rank
#   Hyperparams: λ1=0.1, λ2=0.01, D=128, K=4
```
The ODE is integrated using an Euler solver with step size 0.1.

(C) **Why this design**: We chose a continuous-time ODE over discrete state-space models because it naturally enforces smooth temporal consistency and allows variable-step planning, though it incurs higher simulation cost. The low-rank intervention assumption (K=4) forces action effects to lie in a low-dimensional subspace, which isolates action-necessary features from nuisance variations; the trade-off is that complex actions may require higher rank, so we monitor reconstruction quality to adapt K. Using a shared encoder for forward and inverse dynamics ensures representation consistency, but we accept potential gradient conflict between the two objectives by weighting λ1=0.1 to prioritize forward modeling, since inverse consistency is a regularizer not the primary task. Finally, we opted for a soft consistency loss rather than hard alignment to avoid degenerate solutions where the model ignores action information; the penalty approach allows the model to retain flexibility while encouraging causal necessity.

(D) **Why it measures what we claim**: The consistency loss L_cons measures causal necessity of video features for action prediction because it enforces that the latent dynamics, when forward-simulated, must contain enough action information to recover the original action; the assumption is that if actioning was redundant (i.e., determinable from non-causal parts of z), the inverse model would predict the action regardless of the intervention structure. This assumption fails when the inverse model learns a shortcut from z_{t+1} and z_t that bypasses the action intervention—e.g., using static background features that correlate with action—in which case L_cons would be low even without causal necessity. To mitigate this, we additionally measure intervention rank: the nuclear norm of B indicates how many independent action channels are required; a low rank K=4 provides a capacity constraint that prevents the model from encoding action effects in many latent dimensions, thus proxy-measuring causal isolation. The assumption here is that genuine action-causal features lie in a low-dimensional subspace, which holds for typical robotic tasks where actions affect only a few degrees of freedom; it fails in highly complex manipulations, where the model may need higher rank, causing the method to over-constrain.

## Contribution

(1) A causally structured latent dynamics model (CACLD) that treats actions as low-rank interventions on a continuous-time ODE, enabling principled separation of action-dependent and action-independent latent features. (2) A bi-directional consistency training objective that enforces inverse-action predictability from forward latents, providing a soft constraint for causal necessity without requiring explicit causal annotations. (3) A practical framework for iterative planning in robotics that uses the causally decomposed latent representation to simulate action-interventions reliably, building on joint video-action pretraining.

## Experiment

{
  "experiment_md": "### Evaluation Setup\n\n| Role | Choice | Rationale (≤12 words) |\n|------|--------|----------------------|\n| Dataset | RoboNet | Large-scale robot video with diverse actions |\n| Primary metric | Task success rate | Directly measures action-effectiveness |\n| Baseline 1 | RSSM (Dreamer) | Standard latent dynamics without causal factorization |\n| Baseline 2 | Forward-only model | No inverse consistency loss |\n| Baseline 3 | High-rank model (K=128) | No low-rank intervention constraint |\n| Ablation | w/o inverse consistency | Removes soft consistency loss |\n\n### Why this setup validates the claim\nThis combination forms a falsifiable test of our central claim that low-rank additive interventions with inverse consistency enable causally necessary features. The dataset (RoboNet) provides diverse robotic scenes where actions typically affect only a few degrees of freedom, making the low-rank assumption plausible. Task success rate directly measures whether the learned latents support effective action execution. The baselines isolate our contributions: RSSM tests the value of continuous-time dynamics and causal factorization; forward-only tests whether inverse consistency adds benefit beyond forward prediction; high-rank model tests whether low-rank constraint is necessary for disentanglement. The ablation directly measures the impact of the consistency loss, showing that gains come from the closed-loop objective, not just from the ODE architecture. If our model outperforms RSSM but the high-rank model matches it, then low-rank is not key; if forward-only matches, then inverse consistency is redundant; if ablation matches full model, then consistency loss is irrelevant. This setup thus allows disambiguating which component drives improvements.\n\n### Expected outcome and causal chain\n\n**vs. RSSM** — On a scene with a static background and a moving arm grasping an object, the RSSM latent might encode background textures since it lacks causal constraints, leading to jittery action prediction due to irrelevant variations. Our method forces action effects into a low-dimensional subspace (K=4), so the latent only tracks arm+object dynamics. Thus we expect a 10-15% higher success rate on tasks with distractors, but similar performance on simple scenes.\n\n**vs. Forward-only model** — On an occluded object where the arm disappears temporarily, the forward-only model predicts latents forward but may accumulate drift because actions are not checked for consistency. Our inverse consistency loss ensures that when we predict forward from z_t with a_t, the resulting z_{t+1} must allow a_hat_t to match a_t, making the representation action-aware even under occlusion. Thus we expect a 5-10% success rate gain on long-horizon tasks with occlusions.\n\n**vs. High-rank model (K=128)** — On a task where only one joint moves (e.g., rotating a knob), the high-rank model spreads action effects across many latent dimensions, allowing background correlations to sneak in. Our low-rank constraint (K=4) isolates the action to few dims, yielding more robust inverse prediction. We expect our model to maintain >90% success on simple tasks while the high-rank model may drop to 80% due to overfitting to spurious correlations.\n\n### What would falsify this idea\nIf our model’s gains over forward-only are uniform across all tasks regardless of occlusions or distractor presence, then the inverse consistency loss is not actually enforcing causal necessity but merely improving dynamics in general, falsifying the core claim."

## References

1. DiT4DiT: Jointly Modeling Video Dynamics and Actions for Generalizable Robot Control
2. World Simulation with Video Foundation Models for Physical AI
