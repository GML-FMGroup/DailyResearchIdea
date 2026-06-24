# CoRe: Adversarial Coverage Training for Causally Sufficient Failure Detection in Robotics

## Motivation

Existing failure detection methods for robotic manipulation, such as Foresight, rely on deviation from a fixed learned latent representation without verifying that this representation captures all possible failure causes. This structural gap means failures that do not manifest as noticeable deviations in the chosen latent space are systematically missed, leading to undetected failures. The root cause is that these methods treat the latent representation as given and do not enforce a sufficiency condition linking failures to latent changes.

## Key Insight

Adversarial coverage training forces the latent representation to be a sufficient statistic for failure by searching for failures that evade the current detector while keeping the latent unchanged, ensuring that any failure within the generator's capacity must cause a detectable latent deviation.

## Method

### (A) What it is
**CoRe (Coverage Representation)** is a training framework that augments a world model-based failure detector (e.g., Foresight) with an adversarial objective to enforce the causal sufficiency of the latent representation for failure detection. The system takes as input a set of successful trajectories and a simulator for generating failures, and outputs an encoder, predictor, and detector that provably (under assumptions) flag failures when they arise.

### (B) How it works
```
Notation:
  E : encoder (2-layer MLP, hidden=256, GeLU activation, output latent dim=32)
  P : predictor (LSTM with hidden size 64, 2 layers)
  D : deviation detector (L2 norm ||z_t - P(z_{t-1})||_2)
  G : adversarial generator (same architecture as E + predictor, but with added noise layer)
  S : set of successful trajectories (at least 10k steps total)
  Sim : simulator that can roll out actions and detect task success/failure

Hyperparameters:
  epsilon : maximum L2 perturbation per action, set to 0.1
  delta   : maximum allowed latent similarity (L2) between s and s', set to 0.05
  lambda_adv : weight for adversarial loss, set to 1.0
  coverage_threshold : minimum number of distinct failure clusters (by k-means with k=10) to consider coverage sufficient, set to 5

Assumption (load-bearing):
  The adversarial generator G, constrained by epsilon and delta, can produce all plausible near-success failure modes that the robot might encounter, thereby ensuring the detector is trained to detect them. This assumption is validated after training by measuring the diversity of generated failures (see Coverage Verification).

Training Loop:
1. For each epoch:
   a. Sample a batch (batch size 64) of success trajectories s from S.
   b. For each s, generate an adversarial failure s' = G(s) by adding bounded noise to actions (||a - a'|| ≤ epsilon) such that:
        - Sim(s') yields failure (e.g., task fails)
        - ||E(s) - E(s')|| ≤ delta  (keep latents close)
        - D(s') is minimized (i.e., detector thinks it's success)
   c. Update E, P, D to minimize:
        L_det = D(s') + D(f)   (detect adversarial and real failures f, sampled 1:1 ratio)
        L_pred = MSE prediction loss on s (next latent prediction)
        Total loss = L_pred + lambda_adv * L_det
   d. Update G to maximize D(s') (i.e., make adversarial failures look normal) while keeping Sim(s')=failure and latent constraint (using PPO with clip=0.2, learning rate 1e-4).
   e. (Optional) Use a separate set of real failures f from Sim for coverage (sample 64 failures per epoch).
   f. Coverage Verification: After each epoch, generate 1000 candidate failures from G and cluster them in state space (k-means with k=10). If number of clusters with >5 members is below coverage_threshold, increase epsilon by 0.01 (max 0.5) or add random perturbations (uniform noise in [-0.01, 0.01]) to actions.
2. After training, calibrate a threshold tau using conformal prediction on deviation scores from D over a held-out success set of 512 examples, as in Foresight (miscoverage rate 0.1).
```

### (C) Why this design
We chose an adversarial generator over random augmentation because random perturbations are unlikely to produce failures that specifically challenge the detector's blind spots, whereas an adversary actively searches for the worst-case failures, ensuring coverage of plausible failure modes that are close in latent space. The latent similarity constraint (||z - z'|| ≤ delta) is critical: without it, the adversary could produce drastically different latents that are trivially detected, failing to test sufficiency. The trade-off is that delta must be chosen carefully—too large (e.g., >0.1) and the adversary becomes unrealistic; too small and coverage may be too narrow. We also separate the generator G from the detector D, rather than using GAN-style joint training, to avoid mode collapse and maintain training stability. This design accepts the computational cost of an extra network and simulator queries (about 2x training time relative to Foresight) for the benefit of targeted coverage. Finally, we adopt conformal prediction for threshold calibration (as in Foresight) because it provides a distribution-free false positive guarantee, complementing the coverage guarantee from adversarial training.

### (D) Why it measures what we claim
The deviation score D(z_t, P(z_{t-1})) measures "failure presence" under the assumption that failures cause predictable changes in latent dynamics; this assumption fails when failures are not reflected in the latent space (the very problem we address). The adversarial objective max_G D(s') measures "coverage incompleteness" because if the generator can produce a failure s' with low D(s'), then the current representation is not sufficient—the detector fails to flag a real failure. By minimizing this objective, we force the representation to be a sufficient statistic for failure: any failure that can be generated within the bounded divergence must increase the deviation score. The latent similarity constraint ||z - z'|| ≤ delta operationalizes "causal sufficiency"—it ensures that the adversary must find failures that are close in representation space, testing the condition that all failures cause latent changes. This assumption fails when the generator cannot produce all plausible failure modes (e.g., hardware failures outside simulator), in which case coverage is only guaranteed for generator-accessible failures, not all real-world failures. The Coverage Verification step (clustering generated failures) provides an empirical check on this assumption: if generated failures are not diverse, the assumption may be invalid and the method's guarantees weakened.

## Contribution

(1) A novel adversarial training framework, CoRe, that enforces causal sufficiency of latent representations for failure detection in robotic manipulation. (2) The design principle that explicitly searching for undetected failures via an adversary provides a coverage guarantee complementary to conventional deviation-based detection. (3) A theoretical argument linking adversarial coverage to sufficiency, with practical hyperparameter guidelines for the latent similarity budget.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Simulated long-horizon manipulation (LIBERO-Long) | Realistic failure modes; open-source benchmark |
| Primary metric | AUROC | Threshold-independent; measures ranking quality |
| Baseline 1 | Foresight (world model) | Detects failures via latent deviation; no adversarial coverage |
| Baseline 2 | Uncertainty-aware detector (MC Dropout) | Captures epistemic uncertainty; fails on systematic failures |
| Baseline 3 | Task-specific binary classifier (trained on failures) | Requires failure data; not robust to distribution shift |
| Baseline 4 | Information Bottleneck (IB) detector (train encoder with IB loss to enforce sufficiency) | Highlights advantage of adversarial coverage over regularization approach |
| Ablation-of-ours | CoRe without adversarial generator (random augmentation) | Tests necessity of adversarial targeting |
| Ablation-of-ours | CoRe with varying delta (0.01, 0.05, 0.1, 0.2) | Tests trade-off between coverage and false positives |

### Why this setup validates the claim

The dataset (LIBERO-Long) provides a controlled environment with diverse failure modes, enabling fair comparison. Foresight directly tests the sufficiency of latent representation—our core claim. Uncertainty-aware detector isolates the role of epistemic uncertainty versus our causal sufficiency approach. Task-specific classifier queries the value of not needing failure data. The added Information Bottleneck (IB) baseline uses a different regularization to enforce latent sufficiency (by minimizing mutual information between latent and input), isolating the specific benefit of adversarial coverage over other sufficiency-promoting methods. AUROC is chosen because it evaluates detection ranking independently of threshold, aligning with our goal of distinguishing failures from successes. The ablation (random augmentation) isolates the adversarial generator's contribution; the delta sweep operationalizes the coverage-false positive trade-off. This combination forms a falsifiable test: we predict improved AUROC specifically on near-miss failures, not uniformly. Additionally, we measure generator coverage by reporting the number of distinct failure clusters (k-means, k=10) produced by G over 1000 samples, to empirically validate the load-bearing assumption.

### Expected outcome and causal chain

**vs. Foresight** — On a case where a failure occurs with latent trajectory similar to a successful one (e.g., gripper slightly off but still visually similar), Foresight's low deviation score fails to flag it because its detector never learned to distinguish such near-miss failures. Our CoRe forces the adversary to generate exactly such failures, pushing the detector to assign high deviation to them. We expect a noticeable AUROC gap on these near-miss failures (e.g., +0.15) but parity on obvious failures (e.g., dropping object).

**vs. Uncertainty-aware detector** — On a case where a systematic failure (e.g., robot arm stalls due to friction) produces low aleatoric uncertainty (policy confident but wrong), MC Dropout misses it because uncertainty is low. Our method's adversary actively seeks failures that look normal in latent space, so the detector must raise alarm even when dynamics are predictably off. We expect our method to outperform (e.g., AUROC +0.10) on such systematic, low-uncertainty failures, while performing similarly on stochastic ones.

**vs. Task-specific binary classifier** — On a case where a novel failure mode appears (e.g., new object weight not in training), the classifier trained on limited failure examples fails due to distribution shift. Our method, using only successful trajectories and a simulator, can detect any deviation from learned dynamics. We expect competitive AUROC on in-distribution failures but a clear edge (e.g., +0.20) on out-of-distribution failures, especially those close to success in latent space.

**vs. Information Bottleneck (IB)** — The IB detector also enforces sufficiency but via a variational bound that requires careful tuning of beta. On near-miss failures, IB may not force failures to be close in latent space; adversarial coverage explicitly trains on such examples. We expect CoRe to achieve higher AUROC on near-miss failures (e.g., +0.08) while performing similarly on obvious failures. The delta ablation will show that intermediate delta (e.g., 0.05) yields best AUROC balance.

### What would falsify this idea

If our AUROC improvement over Foresight is uniform across all failure types (not concentrated on near-miss failures predicted by causal sufficiency) or if the ablation (random augmentation) performs similarly to the full method, then our claim that adversarial targeting of latent-close failures is necessary would be falsified. Additionally, if the Information Bottleneck baseline matches or exceeds CoRe on near-miss failures, the specific advantage of adversarial coverage would be unsupported. Finally, if coverage verification reveals that G produces fewer than 3 distinct failure clusters (out of 10), the load-bearing assumption on generator diversity is invalidated.

## References

1. Foresight: Failure Detection for Long-Horizon Robotic Manipulation with Action-Conditioned World Model Latents
2. Can We Detect Failures Without Failure Data? Uncertainty-Aware Runtime Failure Detection for Imitation Learning Policies
3. SAFE: Multitask Failure Detection for Vision-Language-Action Models
4. Revisiting Feature Prediction for Learning Visual Representations from Video
5. AHA: A Vision-Language-Model for Detecting and Reasoning Over Failures in Robotic Manipulation
6. Unpacking Failure Modes of Generative Policies: Runtime Monitoring of Consistency and Progress
7. Hiera: A Hierarchical Vision Transformer without the Bells-and-Whistles
8. VideoMAE V2: Scaling Video Masked Autoencoders with Dual Masking
