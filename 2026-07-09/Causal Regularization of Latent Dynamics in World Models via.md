# Causal Regularization of Latent Dynamics in World Models via iKCE Penalty

## Motivation

World models trained with behavior cloning (e.g., DreamerV3) often produce kinematic imagination: imagined rollouts follow simple extrapolations and fail to capture action-dependent dynamics, especially under distribution shift. This is diagnosed by the iKCE metric in 'Imagined Rollouts are Kinematic, Not Dynamic', which shows that the model's imagined states align closely with a constant-velocity baseline even when true dynamics change. The root cause is that the maximum-likelihood objective does not explicitly enforce that transitions are causally linked to actions; it only reconstructs observations.

## Key Insight

Penalizing deviation from a kinematic baseline forces the world model to learn transitions that require action information, because any residual not explained by constant-velocity extrapolation must be induced by the action.

## Method

We propose Causal Latent Dynamics Regularization (CLDR), a regularizer added to the world model training loss. The world model follows DreamerV3 architecture: an encoder (CNN, 4 layers, 32, 64, 128, 256 channels, ReLU, stride 2), a recurrent state-space model (RSSM) with deterministic GRU (hidden=512) and stochastic latent (32-dim diagonal Gaussian), and a decoder (transposed CNN mirroring encoder). Training uses Adam (lr=1e-4) with gradient clipping (norm=100). The regularizer is applied to the dynamics model's predicted latent states during imagined rollouts, not to posterior latents, to prevent encoder gaming.

### Pseudocode (modified during each world model update step):
```python
for batch in replay_buffer:
    # Encode observations to latent states (posterior)
    s_posterior_prev = encoder(o_{t-1})   # posterior latent at t-1
    s_posterior_cur = encoder(o_t)        # posterior latent at t (for reconstruction loss only)
    s_posterior_prev2 = encoder(o_{t-2})  # posterior latent at t-2
    # Obtain the dynamics model's predicted latent state using the transition model
    # transition: s_pred = f(s_prev, a_prev), where s_prev is the posterior from t-1
    s_pred_cur = transition(s_posterior_prev, a_{t-1})  # predicted latent at t
    # Compute kinematic extrapolation (constant velocity) from predicted states
    # For the prediction baseline, we use the predicted state from t-2? Actually, we need a kinematic baseline for the predicted state at t.
    # We compute s_kin = s_pred_prev + (s_pred_prev - s_pred_prev2) where s_pred_prev = f(s_posterior_prev2, a_{t-2}) and s_pred_prev2 = f(s_posterior_prev3, a_{t-3})? That requires three timesteps.
    # Simpler: use posterior latents for kinematic baseline (as they represent true states), but apply penalty on predicted states.
    # Revised: Compute kinematic baseline from posterior latents (unchanged): s_kin = s_posterior_prev + (s_posterior_prev - s_posterior_prev2)
    # Then compute iKCE on predicted state vs kinematic baseline:
    L_iKCE = mean(||s_pred_cur - s_kin||^2) over timesteps where t>=2
    # Standard reconstruction loss: L_recon = -log p(o_t | s_posterior_cur) + KL divergence
    L = L_recon + λ * L_iKCE   # λ = 0.1 (default, tuned via grid search {0.01, 0.05, 0.1, 0.5, 1.0})
    # Backpropagate through encoder, transition, decoder to update θ
```

(C) **Why this design**: We apply the iKCE penalty to the dynamics model's predicted latent states (from the transition model) rather than to the posterior latents because posterior latents can be trivially adjusted by the encoder to satisfy the kinematic baseline without affecting the transition model's action dependence—a representation collapse issue identified in causal representation learning (Schölkopf et al., 2021). By penalizing predicted states, we directly force the transition model to produce state transitions that deviate from constant-velocity kinematics, ensuring the penalty cannot be gamed. We chose constant velocity as the kinematic null because it is parameter-free and directly captures the 'kinematic' signature identified in the diagnostic paper; we set λ=0.1 to balance regularization strength. Using the posterior latents for the kinematic baseline (as they represent ground-truth states) provides a stable reference.

(D) **Why it measures what we claim**: The computational quantity L_iKCE measures the deviation of the predicted latent transitions from a kinematic baseline (constant velocity). This deviation operationalizes the concept of 'causal necessity of actions' because if the world model ignored actions and simply interpolated, its predicted states would align with the kinematic baseline, yielding low iKCE. By penalizing low iKCE, we force the model to produce transitions that cannot be explained by action-ignorant kinematics, i.e., that must be causally influenced by actions. **Linking assumption A**: any deviation from constant velocity requires action information. This assumption fails when the true dynamics include constant-velocity motions not caused by actions (e.g., drift), in which case the penalty would incorrectly push transitions away from a correct physical model. However, in typical RL environments (e.g., DMC, Atari), such constant-velocity motions are rare; the diagnostic paper shows good world models have high iKCE, indicating that constant-velocity is a suitable null. We test the robustness of this assumption in the ablation with constant acceleration null.

## Contribution

(1) A novel regularization method, CLDR, that integrates iKCE as a training objective to enforce causal latent dynamics in world models. (2) An empirical finding that adding a kinematic null penalty reduces the kinematic-dynamic gap and improves robustness of imagined rollouts under distribution shift. (3) A practical design principle: using a simple physics-based prior (constant velocity) can effectively guide world model learning toward action-conditional representations.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
|------|--------|-----------|
| Dataset | DMC walker-walk, DMC cheetah-run, Atari (Pong, SpaceInvaders) | Where kinematic failure identified; Atari adds visual complexity |
| Primary metric | Imagination reward (mean over 5 seeds, 1000-step rollouts), iKCE of model rollouts | Measures rollout quality and action sensitivity |
| Baseline 1 | DreamerV3 (no iKCE) | Standard world model baseline |
| Baseline 2 | Ours with obs-space iKCE | Tests latent vs observation penalty |
| Baseline 3 | Ours with learned null model (linear predictor trained on real transitions) | Tests constant velocity assumption |
| Baseline 4 | Inverse dynamics prediction (predict action from latent states) | Alternative causal regularization (compare iKCE vs. action prediction) |
| Baseline 5 | Action prediction (predict action from latent states) | Another causal regularization (distinct from iKCE) |
| Ablation | Ours with constant acceleration null | Tests kinematic baseline choice |
| Analysis | Intervention test: swap actions in rollouts, measure change in iKCE | Assesses action sensitivity |

### Why this setup validates the claim
By selecting DMC walker-walk, a domain where previous work diagnosed kinematic failure in world models, we directly test our method's ability to mitigate this issue. The standard DreamerV3 baseline establishes the default performance without regularization. Comparing to obs-space iKCE isolates the benefit of applying the penalty in latent space. The learned null model baseline tests whether the simplicity of constant velocity is crucial. The ablation with constant acceleration further probes robustness of the kinematic assumption. Imagination reward as the primary metric directly measures downstream policy performance, which is the ultimate goal. Adding inverse dynamics and action prediction baselines provides comparison against alternative causal regularization methods. Including Atari environments tests generality across visual domains. This combination provides a falsifiable test: if our method fails to improve imagination reward over DreamerV3 on long-horizon rollouts, the central claim that iKCE regularization yields better action-awareness is disproven.

### Expected outcome and causal chain

**vs. DreamerV3 (no iKCE)** — On a long-horizon rollout with repetitive actions (e.g., walker walking), the baseline's world model produces kinematic transitions because it learns to ignore actions when reconstruction is easy. Our method applies iKCE penalty on predicted states, forcing transitions to deviate from constant velocity, so the model must attend to actions. We expect our method to achieve noticeably higher imagination reward (e.g., 20-30% improvement on DMC walker-walk, 10-20% on Atari) on long-horizon tasks, but similar performance on short tasks where kinematic failure is less pronounced.

**vs. Ours with obs-space iKCE** — When observations contain irrelevant details (e.g., background textures), applying iKCE in observation space penalizes those details, distorting dynamics to match kinematics. Our method applies it in latent space, where the representation is compact and focused on task-relevant factors. Thus we expect our latent version to substantially outperform the obs-space version on visually complex environments (e.g., Atari), with a clear gap in imagination reward (e.g., 15-25% higher).

**vs. Ours with learned null model** — A learned null model (linear predictor trained on real state transitions) could overfit to training data and produce a kinematic baseline that already captures action influence, making the penalty ineffective. Our constant velocity null is simpler and ensures that any action-ignoring transitions are penalized. We expect our constant velocity version to yield higher imagination reward on held-out dynamics (e.g., 10-20% better), while the learned null version may show smaller or inconsistent gains.

**vs. Inverse dynamics prediction (Baseline 4)** — Inverse dynamics regularizes by predicting actions from latent states, which encourages the latent to encode action-relevant information but does not directly penalize kinematic transitions. iKCE regularization directly targets the kinematic signature, so we expect our method to produce more action-sensitive rollouts and higher long-horizon imagination reward.

**vs. Action prediction (Baseline 5)** — Similar to inverse dynamics, action prediction encourages action-informative latents but lacks the explicit kinematic null. We expect our method to outperform on long-horizon tasks where kinematic failure is pronounced.

### What would falsify this idea
If our method does not outperform DreamerV3 on long-horizon tasks (imagination reward similar or worse) across multiple environments, then the claim that iKCE regularization improves action-awareness is false. Additionally, if the improvement is uniform across all tasks regardless of horizon length, it would suggest the gain does not stem from mitigating kinematic failure. If the intervention tests show that swapping actions does not change iKCE in our model's rollouts, it would indicate the penalty did not induce action sensitivity.

## References

1. Imagined Rollouts are Kinematic, Not Dynamic: A Diagnosis of Long-Horizon World-Model Failure
2. Mastering Diverse Domains through World Models
3. Transformers are Sample Efficient World Models
4. Deep Hierarchical Planning from Pixels
