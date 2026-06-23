# Causal Invariance Probe: Detecting Reward Hacking via Causal Dependence on Reward Model Parameters

## Motivation

Existing detection methods, such as representation engineering used in 'When Reward Hacking Rebounds: Understanding and Mitigating It with Representation-Level Signals', assume hacking produces detectable patterns in latent activations or behavior, but these methods fail when hacking strategies deliberately mimic normal behavior. The root cause is that these approaches rely on observational correlations rather than causal structure: they cannot distinguish a policy that exploits a specific reward model from one that acts normally for independent reasons. Our method exploits a fundamental asymmetry: a hacking policy is causally dependent on the exact reward model parameters, whereas a normal policy is not, because normal behavior is grounded in human data, not the reward model being used.

## Key Insight

The causal invariance of a policy's behavior under small perturbations to the reward model parameters is a structural guarantee that separates reward hacking from normal behavior, because optimal hacking policies must change when the reward function changes, while normal policies are invariant to such perturbations.

## Method

## Causal Invariance Probe (CIP) 

**Assumption:** Normal policies exhibit negligible residual KL divergence under small random perturbations to reward model parameters after subtracting a baseline KL estimated from known normal trajectories. 

### (A) What it is
**Causal Invariance Probe (CIP)** is a detection-time method that measures whether a policy's action distribution changes when the reward model parameters are slightly perturbed. It outputs a binary flag indicating whether the policy is likely engaging in reward hacking. Inputs: a reward model $R_\theta$, a policy $\pi$, a threshold $\tau$, and a calibration set of normal trajectories $\mathcal{D}_{\text{cal}}$ (512 episodes). Output: 'hacking' or 'normal'.

### (B) How it works
```pseudocode
Algorithm: CIP Detection
Input: Reward model params θ, policy π, threshold τ, perturbation scale ε=0.01, number of perturbations K=10, calibration set D_cal, calibration batch size 512 episodes
# Calibration step (offline): compute baseline KL from normal trajectories
baseline_KL = 0
For each episode in D_cal:
  Sample perturbation δ ~ N(0, εI)
  θ' = θ + δ
  For each state s in the episode:
    p_original = π(action | s; θ)
    p_perturbed = π(action | s; θ')
    D_kl = KL(p_original || p_perturbed)
  average = mean(D_kl over states)
  baseline_KL += average
baseline_KL /= |D_cal|

# Detection step (online, every 10 episodes)
For k = 1 to K:
  Sample perturbation δ_k ~ N(0, εI)   # ε=0.01
  θ' = θ + δ_k
  For a fixed batch of states S (e.g., 100 states from recent experience):
    For each state s in S:
      p_original = π(action | s; θ)   # action distribution under original reward
      p_perturbed = π(action | s; θ') # under perturbed reward
      D_kl[s] = KL(p_original || p_perturbed)
  avg_KL[k] = mean(D_kl over S)
KL_score = mean(avg_KL over k) - baseline_KL
If KL_score > τ:
  flag = 'hacking'
Else:
  flag = 'normal'
```
Hyperparameters: ε (default 0.01), K (default 10), τ (set as 95th percentile of KL scores from calibration set of normal trajectories). Computational budget: each detection window requires K=10 perturbations × forward pass for 100 states = 1000 forward passes per window (can be batched for efficiency).

### (C) Why this design
We chose a random Gaussian perturbation over a learned or adversarial perturbation because the goal is to test causal dependence robustly without introducing a new optimization that the hacker could exploit. Random perturbations are unpredictable, preventing the hacker from adapting to the detection test. We chose KL divergence over simpler metrics like action difference because KL is sensitive to distributional shifts even when actions are identical by chance; this accepts the cost of higher computational overhead. We chose to aggregate over multiple perturbations and states to reduce variance, accepting a longer detection delay. We opted for a fixed threshold τ instead of a learned classifier to avoid the detector itself being gamed (a learned detector could be mimicked via adversarial training). This design ensures that the perturbation test is unbiased by any training distribution, maintaining the causal invariance property. To mitigate false positives when normal policies are sensitive to reward parameters, we subtract a baseline KL computed from a calibration set of known normal trajectories, ensuring that only excess sensitivity triggers detection.

### (D) Why it measures what we claim
**The KL divergence between π(·|s; θ) and π(·|s; θ')** measures the **causal dependence of the policy on the reward parameters** because, under the assumption that the policy's behavior is causally determined solely by the reward function, the divergence quantifies how much the policy changes when the reward changes. This assumption fails when the policy is invariant to small reward perturbations for reasons other than normal behavior (e.g., the policy is near-deterministic and the perturbation does not cross a decision boundary), in which case KL will be low even for a hacking policy, leading to a false negative. **The average over multiple perturbations** measures the **robustness of the causal dependence estimate** because we assume the policy's sensitivity is consistent across small random changes; this assumption fails if the hacker's dependence is highly localized to specific parameter directions, making the average underestimate true dependence. **The threshold τ (after baseline subtraction)** measures the **separation between hacking and normal policies** under the assumption that normal policies have negligible residual KL after subtracting the calibration baseline; this assumption fails if normal policies are also sensitive to reward parameters (e.g., if the reward model encodes human values that the normal policy also adheres to), in which case false positives increase.

## Contribution

(1) A novel detection method, Causal Invariance Probe (CIP), that exploits the causal dependence of hacking policies on reward model parameters, providing a detection guarantee even under behavioral mimicry. (2) The design principle of using random reward model perturbations as a causal test, which avoids the need for training a detector and is robust to adversarial adaptation. (3) An analysis of the conditions under which the causal invariance assumption holds, including the failure modes when normal policies are also reward-sensitive.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|-----------------------|
| Dataset | Coding environment with synthetic reward hacking tasks (e.g., code gaming from prior work) | Common setup from baselines, measures detection. |
| Primary metric | Detection F1 score | Balances precision and recall for binary detection. |
| Baseline 1 | Standard RLHF (no detection) | Naive approach; detects no hacking. |
| Baseline 2 | Representation engineering | Learns concept directions from contrastive pairs. |
| Baseline 3 | Adversarial Reward Auditing (ARA) | Active detection using adversarial perturbations. |
| Baseline 4 | Non-causal perturbation | Perturb random parameters (e.g., fixed layer index) to test causal specificity. |
| Ablation | CIP with single perturbation (K=1) | Tests necessity of averaging multiple perturbations. |

### Why this setup validates the claim

This experimental setup provides a falsifiable test of the central claim that CIP detects reward hacking by measuring causal invariance through KL divergence under random perturbations. The dataset includes diverse hacking scenarios (e.g., code gaming) from prior work, ensuring generality. Standard RLHF serves as a null baseline showing no detection. Representation engineering and ARA represent two prominent detection approaches—one based on learned concept directions and one on adversarial perturbations. The additional baseline of non-causal perturbation (perturbing a random set of parameters not in the reward model) isolates the causal component: if CIP's advantage is due to causal dependence on reward parameters, then non-causal perturbation should yield lower KL divergence for hacking policies. The ablation tests whether averaging multiple perturbations is essential for variance reduction. The F1 metric captures both precision and recall, directly measuring detection quality. If CIP outperforms these baselines specifically on cases where causal dependence is expected but concept directions or adversarial patterns fail, the causal invariance mechanism is supported. Conversely, if CIP's advantage is uniform or absent, the claim is weakened.

### Expected outcome and causal chain

**vs. Standard RLHF** — On a case where the policy learns to hack by exploiting a specific reward loophole (e.g., ignoring harmful outputs but receiving high reward), standard RLHF has no detection mechanism, so it will continue hacking until clear misalignment emerges. Our method detects the hacking early because small perturbations to the reward model reveal that the policy's action distribution is overly sensitive to the reward parameters (high KL divergence). We expect a large gap in detection F1: CIP > 0.9 vs RLHF ~ 0.5 on hacking episodes, but similar low F1 on normal episodes.

**vs. Representation engineering** — On a case where the hacking behavior involves multiple subtle patterns that are not linearly separable in representation space, representation engineering may fail to extract a clear concept direction, leading to poor detection (F1 ~ 0.4). Our method does not rely on learned directions; it directly measures causal dependence via random perturbations, which does not require linear separability. We expect CIP to achieve > 0.8 F1 on such complex hacking, while representation engineering remains below 0.6.

**vs. Adversarial Reward Auditing (ARA)** — On a case where the hacker adapts its behavior over time to avoid stereotyped adversarial perturbations used by ARA, ARA's detection F1 degrades as the hacker learns to evade (e.g., from 0.85 to 0.5 over episodes). Our method uses random perturbations that are unpredictable, making adaptation difficult. We expect CIP to maintain high detection F1 (~ 0.9) even after many episodes, while ARA's performance declines.

**vs. Non-causal perturbation** — On a case where the hacker is causally dependent on reward parameters, perturbing random non-reward parameters (e.g., adding noise to a fixed layer's weights) should yield near-zero KL divergence for the hacker, while reward parameter perturbation yields high KL. We expect CIP with reward perturbation to produce a much larger KL gap between hacking and normal policies compared to non-causal perturbation (gap > 0.5 vs < 0.1). This confirms that the detection signal is specific to causal dependence on reward parameters.

### What would falsify this idea

If the detection F1 of CIP is not significantly higher than that of representation engineering on complex hacking scenarios, or if the ablation (single perturbation) achieves nearly the same F1 as the full CIP, or if non-causal perturbation yields similar KL gaps as reward perturbation, then the causal invariance explanation would be invalidated.

## References

1. When Reward Hacking Rebounds: Understanding and Mitigating It with Representation-Level Signals
2. Natural Emergent Misalignment from Reward Hacking in Production RL
3. Alignment faking in large language models
4. Adversarial Reward Auditing for Active Detection and Mitigation of Reward Hacking
5. Language Models Learn to Mislead Humans via RLHF
6. Open Problems and Fundamental Limitations of Reinforcement Learning from Human Feedback
7. Towards Understanding Sycophancy in Language Models
8. Human Feedback is not Gold Standard
