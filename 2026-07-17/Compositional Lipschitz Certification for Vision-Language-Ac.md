# Compositional Lipschitz Certification for Vision-Language-Action Models

## Motivation

BadWAM demonstrates that small visual perturbations can cause world-action models to produce catastrophic actions while preserving future predictions, exposing a critical vulnerability in VLA pipelines. Existing certified defenses typically require retraining the entire model end-to-end with Lipschitz constraints, which is costly and may degrade performance on complex tasks. There is a need for a modular approach that allows each component (vision encoder, language grounding, action policy) to be independently certified and then composed to yield a global robustness guarantee without full retraining.

## Key Insight

The Lipschitz constant of a composition of functions is bounded by the product of their individual Lipschitz constants, enabling a provable global sensitivity bound by ensuring each module is locally 1-Lipschitz.

## Method

### (A) What it is
CL-VLA (Certifiable Lipschitz Vision-Language-Action) is a modular framework that replaces each standard component of a VLA pipeline with a Lipschitz-bounded version using spectral normalization (with 5 power iterations per layer), and provides a certified upper bound on the change in output action for any bounded visual perturbation. Input: image I, language instruction L, state s. Output: action a.

### (B) How it works
```pseudocode
# Step 1: Enforce Lipschitz constraint on each component
f_v = LipschitzVisualEncoder(I)          # ResNet-18 with spectral norm on all Conv and Linear layers; L_v ≤ 1 (target)
f_l = LipschitzLanguageGrounding(L, z)   # tiny GPT (2 layers, hidden=256) with spectral norm on all linear projections; L_l ≤ 1 (target)
f_a = LipschitzActionPolicy(z, f_l, s)   # 2-layer MLP (hidden=128) with spectral norm and flow matching (15 denoising steps); L_a ≤ 1 (target)

# Step 2: Compute calibrated global bound
ε = user-specified perturbation bound (e.g., ℓ2 norm, default 0.05)
# Calibration: estimate empirical Lipschitz constants of each module via finite differences on 512 calibration samples
L_v_emp = 1.1 * L_v_target (adjustment factor from calibration)
L_l_emp = 1.1 * L_l_target
L_a_emp = 1.1 * L_a_target
global_L = L_v_emp * L_l_emp * L_a_emp
certified_change = global_L * ε

# Step 3: Certification rule
if certified_change ≤ threshold_T:       # T = 0.1 in action space (ℓ2 norm)
    accept prediction
else:
    reject or fallback
```
Training protocol: fine-tune each module independently using standard loss + spectral normalization penalty (λ=0.01). Vision encoder: fine-tuned for 10 epochs on clean CALVIN data, batch size 64, learning rate 1e-4, weight decay 0.01. Language module: fine-tuned for 5 epochs, same optimizer. Action policy: trained for 20 epochs with flow matching loss (no additional penalty beyond spectral norm).

### (C) Why this design
We chose spectral normalization over gradient penalty because it directly controls the operator norm at each layer and is computationally efficient during training (O(1) per layer via power iteration with 5 steps); the cost is that it may over-constrain capacity, reducing expressivity. We chose to enforce L=1 per module because it yields a simple product bound, avoiding the need to compute exact Lipschitz constants via SVD; the trade-off is that we may impose stricter constraints than necessary, possibly lowering task performance. However, due to non-Lipschitz operations (self-attention, layer norm) in vision and language modules, the true Lipschitz constant may exceed 1; we compensate by calibrating empirically. We chose to apply the constraint to all three modules (vision, language, action) rather than only the vision encoder because the language grounding and action policy can also amplify perturbations; the cost is increased training time and potential degradation on language tasks. This design intentionally sacrifices some task accuracy to gain a provable worst-case guarantee, which is necessary for safety-critical deployment.

### (D) Why it measures what we claim
The computational quantity `L_v` measures the sensitivity of visual features to pixel perturbations because spectral norm bounds the ratio of output change to input change under ℓ2 norm, assuming all activation functions are 1-Lipschitz (ReLU is 1-Lipschitz, fine); this assumption fails when the module contains non-Lipschitz operations like softmax or layer normalization, in which case `L_v` only reflects the sensitivity after spectral normalization but not the true worst-case. The product `L_v * L_l * L_a` measures the global sensitivity of the action output to visual input because the Lipschitz constant of composition is exactly the product, assuming each function is Lipschitz continuous in the ℓ2 sense; this assumption fails if any component has discontinuities (e.g., argmax in language decoding), but we bypass discrete tokens by working in the continuous embedding space. To account for residual errors, we calibrate empirical Lipschitz constants using finite differences on a calibration set and apply a 1.1 multiplicative margin. The certification rule `certified_change ≤ T` measures provable robustness under the calibrated bound; this assumption fails if the true Lipschitz constant exceed the calibrated estimate, in which case the guarantee is invalid.

## Contribution

(1) A modular certification framework for VLA pipelines that composes independently Lipschitz-constrained components to yield a global robustness guarantee without retraining the entire network. (2) A practical method to enforce Lipschitzness on vision, language, and action modules via spectral normalization with minimal performance loss. (3) A theoretical bound on the sensitivity of actions to visual perturbations, enabling rejection or fallback when the perturbation exceeds a safe threshold.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
|------|--------|-----------|
| Dataset | CALVIN (ABCD setup, 6000 training episodes) | Language-conditioned manipulation tasks. |
| Primary metric | Certified success rate at ε=0.05 ℓ2 | Measures robustness under bounded perturbation. |
| Secondary metrics | Robust accuracy at ε=0.01 and 0.1 ℓ2, clean success rate, average action deviation | Provide fuller picture of robustness-performance trade-off. |
| Baseline 1 | BadWAM (open-loop world model, unconstrained encoder) | Uncensored world-action model without certification. |
| Baseline 2 | π0 (base VLA without Lipschitz constraints) | State-of-the-art VLA flow model. |
| Baseline 3 | VLMPC (MPC with VLM vision, same backbone as CL-VLA but no Lipschitz) | MPC baseline with VLM vision. |
| Ablation | CL-VLA - no lang/action Lipschitz (only vision Lipschitz) | Isolates vision-only contribution. |

### Why this setup validates the claim

This combination forms a falsifiable test of the central claim that Lipschitz constraints on all VLA modules are necessary and sufficient for certified robustness. CALVIN provides diverse language-conditioned manipulation tasks with visual inputs, enabling perturbation injection via ℓ2-norm bounded noise or adversarial patches. The certified success rate directly measures whether our guarantee holds empirically under the specified bound. Baselines represent different families: BadWAM (uncertified world model), π0 (uncertified VLA), and VLMPC (MPC alternative). The ablation removes language and action Lipschitz constraints, testing their necessity. If our method outperforms baselines and the ablation fails, the causal role of full Lipschitz constraints is confirmed. Secondary metrics allow comparison of clean performance and granular robustness.

### Expected outcome and causal chain

**vs. BadWAM** — On a case where a small adversarial patch (ℓ2 norm ≤0.05) changes the appearance of an object, BadWAM uses its world model to hallucinate a different state, leading to incorrect action. Our method’s spectrally normalized vision encoder ensures the visual embedding change is bounded by L_v_emp * ε, and with calibration, the global bound is tight. Thus we expect CL-VLA to achieve certified success rate ≈80% at ε=0.05, while BadWAM drops to ≈30% (based on prior results). The causal chain: vision Lipschitz → feature variation bounded → action variation bounded.

**vs. π0** — On a case where a bounded ℓ2 perturbation (ε=0.05) shifts pixel values slightly, π0’s unconstrained vision encoder may produce large feature variation, cascading through language and action heads to cause a large action deviation. Our method’s product of calibrated Lipschitz constants (≈1.331) bounds the total action change to ≤0.066, which is below T=0.1, so certification holds. We expect π0 to have certified success rate ≈45% at ε=0.05 compared to CL-VLA’s ≈80%. The observed signal: a noticeable gap in certified success rate on perturbations near the bound, but parity on clean inputs (≈95% success for both).

**vs. VLMPC** — On a case where adversarial noise (ε=0.05) corrupts the VLM’s perception of object location, VLMPC plans a trajectory based on incorrect features, leading to failed execution. Our method’s Lipschitz language grounding and action policy prevent amplification of visual errors, so even if the vision encoder is slightly off, the action stays within safe bounds. We expect CL-VLA to achieve certified success rate ≈80% while VLMPC achieves ≈50% at the same perturbation. The larger margin is expected on tasks requiring precise positioning (e.g., pick-and-place within 2cm tolerance).

### What would falsify this idea

If CL-VLA’s certified success rate at ε=0.05 is not significantly higher (≥20 percentage points) than all baselines, or if the ablation (without language and action Lipschitz) performs similarly (within 5 points) to the full method, then the central claim that full Lipschitz constraints are necessary for certified robustness would be falsified. Additionally, if the clean success rate of CL-VLA drops below 80%, the trade-off may be unacceptable.

## References

1. BadWAM: When World-Action Models Dream Right but Act Wrong
2. RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control
3. π0: A Vision-Language-Action Flow Model for General Robot Control
4. VLMPC: Vision-Language Model Predictive Control for Robotic Manipulation
5. VIMA: General Robot Manipulation with Multimodal Prompts
6. Visual Instruction Tuning
7. GPT-4V(ision) for Robotics: Multimodal Task Planning From Human Demonstration
8. Video Language Planning
