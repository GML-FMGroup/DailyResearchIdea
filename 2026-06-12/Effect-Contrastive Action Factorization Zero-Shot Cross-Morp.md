# Effect-Contrastive Action Factorization: Zero-Shot Cross-Morphology Transfer in Vision-Language-Action Models

## Motivation

Current fine-tuning of VLA models, such as OpenVLA-OFT (Kim et al.), requires task-specific demonstrations for each robot embodiment, limiting scalability across diverse morphologies. The root cause is that action spaces are tightly coupled with embodiment, and no mechanism exists to generalize action effects across different robots. We need a factorization that separates what effect an action produces from how that effect is realized by a specific embodiment.

## Key Insight

The effect of an action on the environment (e.g., change in object position) is invariant to the robot's morphology, so a shared latent space encoding intended effects can be learned from multi-embodiment data and then mapped to embodiment-specific actions via a lightweight adapter.

## Method

We propose Effect-Contrastive Action Factorization (ECAF).

(A) **What it is**: ECAF is a two-stage framework that learns a universal effect latent space from multi-embodiment data and enables zero-shot cross-morphology transfer by fine-tuning only a small action adapter per new embodiment using random interaction.

(B) **How it works**:
```pseudocode
# Stage 1: Effect Latent Pretraining
Input: Dataset D = {(s_t, a_t, s_{t+1})} from multiple embodiments E1..En
Initialize effect encoder phi (MLP, hidden=256, output dim=64, GeLU activation) and next-state predictor psi (MLP, hidden=256, output dim=state_dim, GeLU activation)
Optimizer: Adam with lr=1e-4, weight_decay=1e-5, batch_size=256
For each batch:
  z_t = phi(concat(s_t, a_t))  # effect latent, dim=64
  pred_s_{t+1} = psi(z_t)
  loss_rec = MSE(pred_s_{t+1}, s_{t+1})
  # Contrastive regularizer: InfoNCE with temperature τ=0.07 over effect latents
  # Positive pairs: samples with cosine similarity of next state deltas > 0.9
  # Negative pairs: random samples from batch
  loss_cont = InfoNCE(z_i, z_j) over positive/negative pairs
  loss = loss_rec + lambda * loss_cont, with lambda=0.1
  
# Stage 2: Lightweight Adapter for New Embodiment
Given new embodiment E_new, collect T=100 random actions and record (s_t, a_t, s_{t+1}) pairs.
Train a small adapter network f_new (2-layer MLP, hidden=128, output dim=action_dim, ReLU activation) mapping z to a.
Freeze phi. Optimizer: Adam with lr=1e-3, no weight decay. Loss: MSE(f_new(z_t), a_t) + alpha * L1 with alpha=0.1. Train for 10 epochs, batch_size=64.
At test time, VLM (e.g., OpenVLA) is finetuned to output effect latent z directly instead of actions using a linear head (finetune for 1000 steps on 100 task demos per task, lr=1e-5). Then adapter maps z to embodiment-specific actions.
```

(C) **Why this design**: We chose an effect latent space over direct action prediction because effect is embodiment-invariant, enabling shared learning across morphologies. We use a contrastive regularizer (InfoNCE) on effect latents rather than purely predictive loss to enforce that actions producing similar next states are close in latent space, which improves adapter generalization. We train the adapter with only random actions (not task-specific demos) to minimize data collection effort, accepting the cost that the adapter may not capture complex motor synergies initially but can be refined with few task demonstrations if needed. We freeze the effect encoder during adapter training to prevent catastrophic forgetting and maintain the universal latent structure. The VLM is finetuned to output effect latents instead of actions, which is a minimal change that leverages the pretrained latent space. **Load-bearing assumption**: The effect latent z, learned from next-state prediction and contrastive alignment of state deltas, captures an embodiment-invariant representation of action outcomes, enabling the adapter to generalize across morphologies. **Additional assumptions**: We assume the next state captures all relevant task information (full observability) and that random actions cover the action space well enough to learn the adapter mapping.

(D) **Why it measures what we claim**: The effect latent z encodes the next-state prediction, and the contrastive loss measures effect consistency across embodiments because it assumes that similar next-state deltas correspond to similar effects; this assumption fails when the next state is not fully observed (e.g., partial observability), in which case z may reflect only observed changes. Next-state delta similarity is used as a proxy for effect equivalence; the underlying assumption is that the environment is deterministic and fully observed, so delta similarity implies same effect. Under stochastic transitions or partial observability, two actions with different effects can have similar deltas due to unobserved confounders, causing the contrastive loss to align unrelated latents. The adapter's action reconstruction loss measures how well a new embodiment can realize the universal effect latent; this assumes that the adapter can map z to actions given enough random data, which fails if the embodiment has uncorrelated joint redundancies not captured by random actions, in which case the adapter outputs a plausible but suboptimal action. The overall framework claims zero-shot cross-morphology transfer because the effect latent is learned from multiple embodiments and the adapter requires only random interaction; this assumes that the effect latent space is sufficiently rich to represent all tasks, which fails for tasks requiring precise force control not captured by state transitions.

## Contribution

(1) A novel two-stage framework, ECAF, that factorizes action representation into an embodiment-invariant effect latent and embodiment-specific adapter, enabling zero-shot cross-morphology transfer. (2) The finding that contrastive learning on effect latents from multi-embodiment data improves adapter generalization over purely predictive objectives. (3) An analysis showing that minimal random interaction (e.g., 100 steps) is sufficient to train an adapter for a new embodiment, reducing data collection cost dramatically.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
|------|--------|-----------|
| Dataset | MERB (multi-embodiment tasks) with deterministic simulation environments | Tests cross-morphology transfer under full observability assumption |
| Primary metric | Task success rate (%) | Direct measure of task completion |
| Baseline | OpenVLA (no adaptation) | Shows zero-shot fails without adaptation |
| Baseline | Full OpenVLA fine-tuning (on 100 task demos per task) | Upper bound of full retraining with comparable data |
| Baseline | Goal-conditioned BC (using goal image as input, with a ResNet encoder and MLP policy) | Standard cross-embodiment baseline that uses goal images instead of effect latents |
| Ablation of ours | ECAF w/o contrastive loss (only predictive loss) | Isolates contrastive regularizer effect |
| Resource estimates | Stage1: 2 GPU days (A100), Stage2: 1 GPU hour per embodiment | Concrete resource usage for reproducibility |

### Why this setup validates the claim

This experimental design directly tests the central claim that ECAF enables zero-shot cross-morphology transfer with minimal data. The MERB dataset includes multiple robot embodiments with varied action spaces (e.g., 6-DoF arm, 7-DoF arm, gripper variants) in deterministic simulation, making it ideal for evaluating transfer under the full observability assumption. The primary metric, task success rate, captures whether the method generalizes effectively. OpenVLA without adaptation shows the inherent difficulty of cross-embodiment transfer, while full fine-tuning provides an upper bound requiring extensive data. Goal-conditioned BC tests alternative cross-embodiment approaches that rely on goal images rather than effect latents. The ablation of the contrastive loss (removing the InfoNCE regularizer) isolates its contribution to the effect latent space's embodiment invariance. We also verify the load-bearing assumption by monitoring the alignment of effect latents across embodiments: if the contrastive loss effectively aligns latents for similar state deltas from different embodiments, the assumption holds. Success is detected if ECAF significantly outperforms the no-adaptation baseline and approaches full fine-tuning with far less data, especially on tasks with large morphology differences and contact-rich interactions.

### Expected outcome and causal chain

**vs. OpenVLA (no adaptation)** — On a case where the new embodiment has a very different action space (e.g., 6-DoF arm vs 7-DoF arm), OpenVLA outputs actions mismatched to the robot's kinematics, causing random failures because it directly predicts action values without adaptation. Our method instead first predicts an effect latent z, which is embodiment-invariant, then maps z to actions via the adapter trained on random interactions, so it produces coherent motions that achieve the intended effect. Hence we expect a large gap in success rate (e.g., 20% vs 70%) on embodiments with distinct action structures.

**vs. Full OpenVLA fine-tuning** — On a case where only 100 random actions are available for the new embodiment (no task-specific demos), full fine-tuning updates the entire model, risking catastrophic forgetting of visual features and producing degraded performance (e.g., 50% success). Our method freezes the effect encoder and trains only a small adapter, preserving the universal latent space and avoiding forgetting, resulting in higher success (e.g., 65%) and better data efficiency. We expect our method to be within 10% of full fine-tuning while using far less data (100 random actions vs 100 task demos), and to exceed it when limited data is available.

**vs. Goal-conditioned BC** — On a case where the task involves precise force control (e.g., peg insertion), goal-conditioned BC predicts actions conditioned on the goal image, but fails because goals do not capture the required force profile; it produces jerky motions that miss the hole. Our method uses effect latents that encode the next state (including force cues via position changes), so the adapter learns smoother, more accurate actions. We expect a significant success rate advantage (e.g., 75% vs 40%) on contact-rich tasks.

### What would falsify this idea

If ECAF's improvement over the no-adaptation baseline is uniform across all tasks and embodiments (rather than concentrated on those with large morphology differences or contact-rich interactions), then the claimed mechanism of effect invariant latent space is not driving the gain, and the benefit might stem from other factors like the adapter architecture or random action collection.

## References

1. Fine-Tuning Vision-Language-Action Models: Optimizing Speed and Success
