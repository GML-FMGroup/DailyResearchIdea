# Gradient-Independent Adversarial Robustness for Vision-Language-Action Models via Dual-Consistency Invariant Latent Actions

## Motivation

Existing adversarial attacks on VLA models (e.g., BadWAM) exploit white-box gradient access to craft perturbations that disrupt action selection while preserving world-model predictions. This reliance on gradient access is a structural limitation across all branches of VLA research (RT-2, π0, VLMPC, VIMA). Without a method that establishes robustness without requiring gradient-based attack generation, VLA models remain vulnerable in scenarios where attackers lack model access. We propose to overcome this by learning latent action representations that are invariant to random visual perturbations through a dual-consistency framework, directly addressing the need for gradient-independent security.

## Key Insight

Invariance to random perturbations, enforced via contrastive and temporal consistency, forces the latent action representation to encode only the causal structure of the task (which is stable under small visual changes) while discarding perturbation-sensitive spurious correlations, thereby providing robustness without gradient access.

## Method

### (A) What it is
We propose **Dual-Consistency Invariant Latent Actions (DCILA)**, a self-supervised framework that jointly trains an action encoder and a world model to produce latent action representations that are invariant to black-box adversarial perturbations (specifically SimBA), achieving adversarial robustness without white-box gradient access. Inputs: observation o, black-box perturbation from SimBA, VLA backbone (e.g., RT-2, π0). Outputs: robust action encoder E and world model W.

### (B) How it works (pseudocode)
```python
# Hyperparameters: τ=0.1 (contrastive temperature), λ_temp=1.0, λ_cont=0.5, SimBA query_budget=100, SimBA step_size=0.01
# Data: batch of observations {o_i}, each paired with SimBA perturbation p_i (generated without gradient access)
for o_i in batch:
    p_i = simba_perturb(o_i, query_budget=100, step_size=0.01)  # black-box perturbation
    o_i_pert = o_i + p_i
    z_i = E(o_i)                              # latent action representation (dim=64)
    z_i_pert = E(o_i_pert)
    a_i = D(z_i)                               # action from clean latent
    a_i_pert = D(z_i_pert)                     # action from perturbed latent
    # Contrastive consistency
    pos_sim = cosine_sim(z_i, z_i_pert)
    neg_sims = [cosine_sim(z_i, E(o_j)) for j != i]
    L_cont = -log( exp(pos_sim/τ) / (exp(pos_sim/τ) + sum_j exp(neg_sims/τ)) )
    # Temporal consistency
    o_next_i = W(o_i, a_i)
    o_next_i_pert = W(o_i_pert, a_i_pert)
    L_temp = MSE(o_next_i, o_next_i_pert)
    L_total = λ_cont * L_cont + λ_temp * L_temp
# Update E, D, W jointly (W is pretrained on clean data and fine-tuned)
```

### (C) Why this design
We chose contrastive consistency over a direct L2 loss between z and z_pert because a simple L2 pull would collapse all representations to a single point, losing discriminative information. Instead, contrastive loss preserves feature diversity while enforcing invariance locally. We used SimBA perturbations rather than Gaussian noise to directly target black-box adversarial robustness; the cost is that training requires additional queries (100 per sample) but maintains gradient independence. We opted for a world model consistency (future prediction match) over a direct action consistency because actions derived from the latent may be discrete or quantized in VLA models; predicting future observations provides a continuous supervisory signal that forces the latent to capture dynamics-relevant features. The trade-off: training the world model jointly increases computational cost, but ensures the temporal consistency loss adapts to the learned representation.

### (D) Why it measures what we claim
**Load-bearing assumption:** Enforcing latent representation invariance to SimBA perturbations (via contrastive and temporal consistency) is sufficient to achieve robustness against other black-box adversarial perturbations not restricted to the SimBA distribution. The contrastive loss measures **invariance to SimBA perturbations** because it assumes that the cosine similarity between clean and perturbed latents should be higher than between different observations; this assumption fails when two different observations have similar latents (e.g., multiple objects in the same pose), in which case the loss reflects discriminability rather than invariance. The temporal consistency loss L_temp measures **causal sufficiency of the latent action representation under perturbation** because it assumes that if the predicted future observations match, then the action latent contains the same dynamics-relevant information; this assumption fails when the world model has high capacity and learns to ignore action differences, in which case L_temp becomes trivially low and does not enforce invariance. Together, the dual losses ensure that the latent representation is both invariant to perturbations and sufficient for predicting future dynamics, operationalizing gradient-independent robustness. To verify the load-bearing assumption, we test against a different black-box attack (e.g., random search with squared loss) and measure the success gap.

## Contribution

(1) A self-supervised dual-consistency framework for training VLA models that achieves adversarial robustness without requiring gradient access for attack generation. (2) The insight that random perturbation invariance, when combined with temporal consistency, can substitute for adversarial training in world-model-based policies. (3) A training protocol that is compatible with existing VLA architectures (e.g., RT-2, π0) without architectural modification.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|---|---|---|
| Dataset | MetaWorld (10 tasks) | Common manipulation benchmark |
| Primary metric | Task success under black-box adversarial attack (SimBA, query budget 100) | Measures gradient-independent robustness |
| Secondary metric | Task success under adaptive black-box attack (random search, budget 500) | Tests resilience against stronger attacks |
| Baseline 1 | π0 (VLA generalist) | Strong non-robust baseline |
| Baseline 2 | VLMPC (MPC with language) | Strong planner without robustness |
| Baseline 3 | π0 + adversarial training with random Gaussian noise (std=0.05) | Ablates effect of random vs. SimBA training |
| Ablation 1 | DCILA w/o contrastive consistency | Isolates contrastive contribution |
| Ablation 2 | DCILA with frozen world model (no fine-tuning) | Assesses world model adaptation importance |
| Compute | ~100 GPU hours on single A100 (for all 10 tasks) | Feasibility estimate |

### Why this setup validates the claim
This combination forms a falsifiable test of our central claim—adversarial robustness without white-box gradient access—by pitting DCILA against strong baselines that are vulnerable to black-box attacks. Using MetaWorld with black-box attacks (SimBA and random search) directly evaluates robustness in the intended threat model. Comparing to π0 and VLMPC tests whether our invariance mechanism yields advantage over state-of-the-art generalist and planner policies. The inclusion of π0 trained with random noise isolates the effect of our SimBA-based invariance training. Ablations remove contrastive consistency and world model fine-tuning to verify their necessity. The secondary metric using an adaptive attack (random search) tests generalization to out-of-distribution black-box perturbations, addressing the load-bearing assumption.

### Expected outcome and causal chain
**vs. π0** — On a case where a small adversarial perturbation subtly alters object texture, π0’s direct action mapping from the perturbed visual input produces a wrong action because it has no invariance to pixel changes. Our method instead maps perturbed observations to similar latent actions via contrastive consistency, preserving correct behavior. We expect a noticeable gap on attacked tasks (π0: ~20% success, DCILA: ~70%) but parity on clean tasks.

**vs. VLMPC** — On a case where adversarial perturbation distorts the predicted future observation, VLMPC’s MPC based on inaccurate predictions fails. Our method’s temporal consistency loss ensures future predictions from perturbed latent match those from clean latent, so MPC plans correctly. We expect VLMPC success drops to ~10% under attack, while DCILA remains above 60%.

**vs. π0 + random noise adversarial training** — On a case where the adversarial perturbation is crafted (SimBA), π0 trained with random noise may still be vulnerable because random noise has different structure from SimBA. Our method, trained directly on SimBA, should show higher success (e.g., 70% vs. 50%).

### What would falsify this idea
If DCILA’s success rate under black-box attack is not significantly higher than all baselines, or if the gain is uniform across all attack magnitudes instead of concentrated where baselines fail, the invariance claim is false. Additionally, if performance under adaptive random search attack drops dramatically (e.g., below baseline level), the load-bearing assumption is invalidated.

## References

1. BadWAM: When World-Action Models Dream Right but Act Wrong
2. RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control
3. π0: A Vision-Language-Action Flow Model for General Robot Control
4. VLMPC: Vision-Language Model Predictive Control for Robotic Manipulation
5. VIMA: General Robot Manipulation with Multimodal Prompts
6. Visual Instruction Tuning
7. GPT-4V(ision) for Robotics: Multimodal Task Planning From Human Demonstration
8. Video Language Planning
