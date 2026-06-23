# Geometric-Symbolic Latent Verification for Adaptive Replanning in Multi-Agent Robotics

## Motivation

Existing multi-agent orchestration frameworks, such as the Verified Multi-Agent Orchestration (VMAO) framework, rely on LLM-based verifiers that assume clean symbolic inputs, making them brittle when applied to robotic domains with high-dimensional, noisy sensor data. The root cause is a structural mismatch: VMAO's verification operates on symbolic subgoal completions, but robotic perception cannot be reliably mapped to those symbols without a learned grounding that is robust to noise. This fails because noisy sensor observations are not directly comparable to symbolic correctness, causing verification errors that cascade into costly replanning at the task level.

## Key Insight

If verification and geometric state are encoded in a shared latent space that is learned to be both discriminative for subgoal achievement and robust to sensor noise, then local geometric replanning can be triggered solely by latent-space discrepancy without needing explicit symbolic reconstruction.

## Method

**Self-name:** Geometric-Symbolic Latent Verification for Adaptive Replanning (GSLV-AR)
**Inputs:** a task decomposition DAG (subgoals), a set of motion primitives, online sensor observations. **Outputs:** a sequence of successful motion primitives or a local geometric replan.

**(B) How it works:**
```python
# Phase 1: Offline Joint Representation Learning (with sensor noise augmentation)
train VAE with two heads: 
  - Encoder: f_enc(o_t) -> z_t (latent, dim=64)
  - Decoder (symbolic): f_sym(z_t) -> logits over subgoal state {achieved, not_achieved}
  - Decoder (geometric): f_geo(z_t) -> reconstructed point cloud (chamfer loss)
  # Augment training data by adding Gaussian noise with σ=0.05 (fraction of bounding box diagonal) to input point clouds
Loss = L_sym (cross-entropy) + L_geo (chamfer) + β * KL(z_t || N(0,1))

# Phase 2: Online Execution
for each subgoal s in DAG (executed in parallel per agent):
  plan motion primitive p_s (e.g., joint-space trajectory)
  execute p_s and observe o_t
  z_t = f_enc(o_t)
  z_pred = f_enc(predict_obs(p_s))   # predicted latent from motion model
  if L2(z_t, z_pred) > τ (threshold=0.3, tuned on validation set with injected noise σ_val=0.05):
      # local geometric replan
      sample alternative motion primitives
      execute best according to lowest L2(z_t, z_pred)
  else:
      continue
```
Hyperparameters: latent dimension=64, β=0.1, τ=0.3 (tuned via validation with noise augmentation), noise σ=0.05 (during training).

**(C) Why this design:**
We chose a VAE over a deterministic autoencoder because the probabilistic latent space provides a smooth manifold where small sensor noise does not cause large jumps in latent coordinates, accepting the cost of slightly noisier reconstructions. We chose a two-headed decoder (symbolic and geometric) rather than a single decoder that jointly predicts both because it allows separate loss weighting and prevents geometric reconstruction errors from corrupting symbolic classification (trade-off: separate heads require more parameters but avoid gradient interference). We used L2 discrepancy in latent space as the verification signal instead of symbolic logits because the latent space directly encodes geometric state, enabling local replanning without needing to interpret symbolic correctness; this trades the ability to output a textual failure reason for speed and robustness to symbol grounding errors.

**(D) Why it measures what we claim:**
The L2 discrepancy between z_t and z_pred measures geometric execution success because we assume the VAE's latent space is a sufficient statistic for subgoal achievement and invertible up to sensor noise (i.e., the encoder is injective modulo noise). This assumption fails when sensor noise causes distribution shift (e.g., camera glitch) such that the encoder maps o_t to a different region of the latent space than expected, in which case the discrepancy reflects sensor abnormality rather than task failure. To mitigate this, we augment training data with Gaussian noise (σ=0.05) to improve robustness. The symbolic decoder's output serves as a fallback verification for the replanning trigger: when the latent discrepancy is low but the symbolic prediction gives low confidence, the system can request human intervention—this measures *confidence-calibrated symbolic verification* because we assume the decoder's softmax probability correlates with true subgoal achievement; this assumption fails when the training data lacks coverage of certain sensor conditions, leading to overconfident wrong predictions.

## Contribution

(1) A variational autoencoder architecture that jointly learns a latent space for both geometric state reconstruction and symbolic subgoal verification, enabling robust verification under noisy sensor inputs. (2) A local geometric replanning rule based on latent-space discrepancy, which avoids task-level replanning and reduces computational overhead. (3) Design principle that verification in robotic orchestration should operate in a learned latent space rather than on raw sensor or symbolic outputs, providing a bridge between LLM-based symbolic verification and continuous motion control.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | Multi-agent furniture assembly (simulated) | Challenging geometric coordination task. |
| Primary metric | Subgoal achievement rate | Directly measures multi-step task success. |
| Baseline | Single-agent LLM (VMAO) | No decomposition or verification baseline. |
| Baseline | Multi-agent without verification | Tests necessity of geometric verification. |
| Ablation | GSLV-AR w/o replanning | Ablates online reactive replanning. |

### Why this setup validates the claim

This experimental design forms a falsifiable test of the central claim that geometric-symbolic latent verification enables adaptive replanning for improved multi-agent orchestration. The furniture assembly dataset requires both task decomposition and precise geometric execution, directly exercising the proposed method's core components. The single-agent LLM baseline (VMAO) tests the necessity of multi-agent decomposition, while the multi-agent without verification baseline isolates the value of the online discrepancy check. The ablation, which removes the replanning trigger, quantifies the contribution of local replanning itself. Subgoal achievement rate captures the ultimate measure of task success, and it is sensitive to both symbolic failures (missing subgoals) and geometric failures (misaligned moves), making it the appropriate metric to detect the predicted pattern: gains concentrated in geometrically challenging or sensor-noisy subgoals.

### Expected outcome and causal chain

**vs. Single-agent LLM (VMAO)** — On a case where the task requires two agents to simultaneously lift a long beam, the single-agent baseline serializes the action, causing the beam to tilt and the subgoal to fail. The single-agent lacks a mechanism to coordinate parallel actions. Our method decomposes the DAG and executes motion primitives in parallel per agent, then verifies each subgoal's geometric state via latent discrepancy. Thus, on such multi-agent subgoals, we expect a large gap (e.g., 30–40% higher subgoal achievement) while on sequential subgoals performance is similar.

**vs. Multi-agent without verification** — On a case where a sensor glitch causes an agent to misgrasp a component, the naive multi-agent continues execution as planned, propagating the error and eventually failing the entire assembly. Our method detects the high latent discrepancy (L2 > 0.3) between observed and predicted latent, triggers a local geometric replan that samples alternative motion primitives, and corrects the grasp before proceeding. Therefore, in the presence of sensor noise or execution errors, we expect a significant improvement (e.g., 20–50% higher subgoal achievement) compared to the baseline, while in ideal conditions performance is comparable.

### What would falsify this idea

If our method shows no improvement over the multi-agent without verification baseline on subgoals with artificially injected sensor noise (e.g., 10% pose perturbation), or if the gain is uniform across all subgoals rather than concentrated on geometrically uncertainty-prone ones, then the central claim that latent discrepancy enables adaptive replanning is falsified.

## References

1. Verified Multi-Agent Orchestration: A Plan-Execute-Verify-Replan Framework for Complex Query Resolution
2. Tree of Thoughts: Deliberate Problem Solving with Large Language Models
3. AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation
4. ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs
5. Self-Instruct: Aligning Language Models with Self-Generated Instructions
6. Faithful Reasoning Using Large Language Models
7. MAOP: Multi-Agent Orchestration Platform for Robotic Applications
