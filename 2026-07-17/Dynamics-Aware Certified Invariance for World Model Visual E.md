# Dynamics-Aware Certified Invariance for World Model Visual Encoders

## Motivation

Existing world models (e.g., BadWAM) rely on pre-trained visual encoders that are not invariant to task-irrelevant perceptual perturbations, allowing adversarial attacks to alter latent states and cause incorrect action selection. While RT-2 and π0 treat the visual encoder as fixed, they assume (often incorrectly) that features are linearly separable for actions, leaving a structural vulnerability: the encoder's representation space is not aligned with the world model's dynamics, so small perturbations can change latent states without affecting predicted futures. We need a certified defense that uses the world model's own dynamics to define task-relevant invariance, ensuring that perturbations that do not change dynamics also do not change actions.

## Key Insight

By jointly bounding the Lipschitz constants of the encoder and the dynamics model, we can certify that any perturbation within a certain radius cannot change the downstream action, because the encoder's output change is bounded and the dynamics model's sensitivity to that change is also bounded.

## Method

(A) **What it is**: DACE (Dynamics-Aware Certified Encoder) is a training and certification framework that ensures a world model's visual encoder is provably robust to perturbations of a given radius, by jointly bounding the Lipschitz constants of the encoder and the dynamics model.

(B) **How it works**:
```
# Hyperparameters: eps (perturbation radius), L_target (target Lipschitz for encoder), lambda (regularization weight)

1. Initialize encoder f_θ with spectral normalization (SN) to enforce Lipschitz constant L_f ≤ L_target.
2. Initialize dynamics model g_φ with SN to enforce Lipschitz constant L_g.
3. For each training batch (obs, action, next_obs):
   a. Compute latent: z = f_θ(obs)
   b. Add random noise δ to obs within radius eps: obs_adv = obs + δ, with ||δ|| ≤ eps
   c. Compute latent_adv = f_θ(obs_adv)
   d. Enforce invariance via regularization: L_inv = max(0, ||latent_adv - z|| - L_target * eps)
   e. Compute dynamics prediction: pred_z_next = g_φ(z) and pred_z_adv = g_φ(latent_adv)
   f. Enforce that dynamics output change is small: L_dyn = max(0, ||pred_z_adv - pred_z_next|| - L_g * L_target * eps)
   g. Action prediction loss: L_act = action_prediction_loss(z, action) (standard)
   h. Total loss = L_act + lambda * (L_inv + L_dyn)
   i. Update θ and φ with gradient descent, applying SN after each step.
4. Certification: For a given input obs, the certified radius is r = action_bound / (L_g * L_f), where action_bound is the minimum change in action that would flip the decision.
```

(C) **Why this design**: We chose spectral normalization over gradient penalty because it provides a hard bound on Lipschitz constant rather than a soft penalty, accepting the computational cost of per-layer normalization but ensuring the certified radius is valid. We enforce invariance on both latent and dynamics output rather than only on latent, because the downstream dynamics amplifies encoder changes; without dual enforcement, the guaranteed bound would be too loose. We use a max(0, ...) hinge loss for invariance rather than MSE, because MSE would penalize all changes even within the allowed bound, potentially harming clean accuracy; the hinge loss only activates when the change exceeds the Lipschitz-predicted bound. We train with random noise perturbations rather than adversarial perturbations because adversarial training would require solving an inner maximization per step, greatly increasing training time; random noise suffices to produce a worst-case bound under Lipschitz assumption, as long as the bound is certified via the Lipschitz product. We set L_target as a hyperparameter to trade off expressiveness vs robustness; a lower L_target gives a better certified radius but may reduce clean accuracy.

(D) **Why it measures what we claim**: The computational quantity L_inv = max(0, ||latent_adv - z|| - L_target * eps) measures the degree to which perturbations cause latent change beyond the worst-case bound implied by the encoder's Lipschitz constant; this is a proxy for task-irrelevant perturbation invariance because we assume that if latent change is within the Lipschitz bound, then the perturbation is irrelevant to the downstream task; this assumption fails when the encoder's Lipschitz constant is overestimated (i.e., the true Lipschitz is higher than L_target due to SN approximation), in which case L_inv may under-penalize actual invariance violations. The quantity L_dyn = max(0, ||pred_z_adv - pred_z_next|| - L_g * L_target * eps) measures the extent to which latent changes cascade into dynamics prediction changes beyond the product bound; this is a more direct measure of task-relevant invariance because the dynamics model transforms latent changes into action-affecting changes; the assumption is that dynamics model's Lipschitz constant L_g captures the worst-case amplification, which fails when the dynamics model is not globally Lipschitz (e.g., chaotic dynamics) or when L_g is improperly bounded, in which case L_dyn might not bound the actual effect on actions. The certification formula r = action_bound / (L_g * L_f) provides a certified radius for perturbations that cannot change the action, under the assumption that the action function is also Lipschitz with constant 1 (or known); this assumption fails if the action head's Lipschitz is large or unknown, requiring joint certification.

## Contribution

(1) A certified invariance training framework (DACE) that jointly bounds the Lipschitz constants of a world model's visual encoder and dynamics model to ensure robustness to task-irrelevant perturbations. (2) A principled bound linking perturbation radius, encoder sensitivity, and dynamics model sensitivity, enabling instance-wise certification of action invariance. (3) Empirical verification that DACE improves robustness against adversarial attacks without sacrificing clean accuracy, demonstrating that dynamics-aware certification outperforms encoder-only defenses.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|-----------------------|
| Dataset | DMControl (Walker, Cheetah) | Standard continuous control with visual inputs. |
| Primary metric | Certified robust reward | Measures provable task performance under attack. |
| Baseline 1 | Vanilla world model (Dreamer) | No defense, shows attack vulnerability. |
| Baseline 2 | Adversarially trained world model | Empirical defense, no certification. |
| Baseline 3 | Encoder-only spectral normalization | Ablation missing dynamics Lipschitz. |
| Ablation of ours | DACE with MSE invariance loss | Test hinge loss design choice. |

### Why this setup validates the claim

The combination of DMControl with pixel observations and certified robust reward provides a direct falsifiable test: if DACE’s certified Lipschitz bounds hold, its robust reward should degrade only beyond the certified radius, while baselines fail earlier. Vanilla world model shows that standard training is vulnerable to small perturbations, establishing the need for robustness. Adversarial training demonstrates that empirical defenses do not provide guarantees; comparing to DACE isolates the benefit of certification. Encoder-only SN tests the necessity of dual (encoder + dynamics) Lipschitz enforcement: if its performance drops significantly, it confirms that dynamics amplification is a real threat. The ablation with MSE invariance loss tests the hinge loss design; if it performs worse, the hinge loss is justified. Thus, the setup can confirm or refute each sub-claim of the method.

### Expected outcome and causal chain

**vs. Vanilla world model** — On a perturbed observation where the pixel shift is 2 pixels, the vanilla world model produces a latent encoding far from the clean one, causing the dynamics model to predict an incorrect next latent, leading to a suboptimal action and reward drop. Our method produces a latent within the Lipschitz bound, so dynamics predictions remain accurate, and the action is unchanged. We expect the vanilla baseline to show a sharp reward decline even at extremely small perturbations (e.g., <0.01 norm), while DACE maintains near-perfect reward up to the certified radius.

**vs. Adversarially trained world model** — On an observation perturbed by a PGD attack of radius 0.05, the adversarially trained model may maintain reward because it saw similar attacks during training, but on a different perturbation type (e.g., random noise or FGSM), its robustness fails due to overfitting. Our method’s Lipschitz-based certification guarantees robustness to all bounded perturbations, not just those seen during training. We expect adversarial training to show non-uniform robustness (high under PGD, low under other attacks), whereas DACE shows consistent reward up to the certified radius.

**vs. Encoder-only spectral normalization** — On a perturbation of radius 0.05, the encoder-only baseline bounds the latent change, but the dynamics model amplifies this change into a large error in predicted next latent, causing action deviation. Our method enforces both encoder and dynamics Lipschitz, so the product bound ensures the latent change is harmless. We expect encoder-only to perform well at small radii (<0.02) but degrade at larger radii, while DACE maintains robust reward up to the full certified radius (e.g., 0.05).

### What would falsify this idea

If DACE’s robust reward is not significantly higher than the encoder-only SN baseline at radii beyond 0.02, then the dual enforcement of dynamics Lipschitz is unnecessary, falsifying the central claim. Alternatively, if DACE’s clean reward is drastically lower than all baselines, the method sacrifices utility for robustness, which would invalidate the practical claim.

## References

1. BadWAM: When World-Action Models Dream Right but Act Wrong
2. RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control
3. π0: A Vision-Language-Action Flow Model for General Robot Control
4. VLMPC: Vision-Language Model Predictive Control for Robotic Manipulation
5. VIMA: General Robot Manipulation with Multimodal Prompts
6. Visual Instruction Tuning
7. GPT-4V(ision) for Robotics: Multimodal Task Planning From Human Demonstration
8. Video Language Planning
