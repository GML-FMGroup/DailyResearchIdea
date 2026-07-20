# Autonomous Skill Induction via Self-Supervised Multimodal Exploration

## Motivation

Existing skill induction methods, such as RESOURCE2SKILL, require pre-existing curated multimodal resources (tutorial videos, code repositories, articles) that are costly to obtain and may not align with the target environment. This structural dependency prevents agents from acquiring skills for novel tasks or domains where no such resources exist. We address this by enabling the agent to autonomously generate its own aligned multimodal data through purposive exploration, eliminating the need for external curation.

## Key Insight

The environment's dynamics provide a natural source of causal structure that the agent can exploit to generate self-consistent multimodal data, aligning visual, textual, and action modalities without human annotation.

## Method

(A) **What it is**: ASIIE is a framework for zero-shot skill induction that takes a natural language task specification and outputs an executable policy. The agent explores the environment driven by an intrinsic reward combining task relevance and multimodal consistency, generating aligned video-text-code trajectories. These trajectories are then distilled into a skill policy via behavioral cloning.

(B) **How it works**:
```python
# Pseudocode for ASIIE Training Phase
def generate_skill(task_spec, env):
    # Initialize skill policy π_θ, multimodal alignment model M, intrinsic reward model R_int
    # Exploration buffer B
    for episode in range(N_explore):
        state = env.reset()
        trajectory = []
        intrinsic_rewards = []
        for t in range(T_max):
            action = π_θ.explore_action(state)  # epsilon-greedy with noise (ε=0.1)
            next_state = env.step(action)
            # Generate textual description using BLIP-2 captioning model
            text = M.generate_caption(state, action, next_state)
            # Compute intrinsic reward
            task_rel = cosine_similarity(CLIP_embed(text), CLIP_embed(task_spec))
            consis = M.alignment_score(state, text, action)  # from pretrained contrastive model
            r_int = λ1 * task_rel + λ2 * consis   # λ1=0.5, λ2=0.5
            trajectory.append((state, action, text, next_state))
            intrinsic_rewards.append(r_int)
            state = next_state
        # Store trajectory if average intrinsic reward > threshold τ=0.6
        if mean(intrinsic_rewards) > τ:
            B.append(trajectory)
    # Distill skill policy via behavioral cloning
    train π_θ on B with L_BC = L_action + α * L_text_consistency   # α=0.1
    # L_action: cross-entropy on actions; L_text_consistency: MSE between CLIP embedding of text and state-embedding
    return π_θ
```
(C) **Why this design**: We chose exploration driven by a composite intrinsic reward (task relevance + multimodal consistency) over pure random exploration because the reward structure guides the agent toward trajectories that are both useful for the task and self-consistent, accelerating skill acquisition; the trade-off is computational overhead for computing embeddings and alignment scores. We used a pretrained captioning model (BLIP-2) to generate text from visual states rather than training from scratch because it provides strong grounding without requiring task-specific data, though it may introduce domain shift; we mitigate this by fine-tuning on in-domain data if available. We opted for behavioral cloning distillation with an additional text consistency loss (L_text_consistency) rather than pure BC because enforcing alignment between the policy's actions and the generated textual descriptions improves multimodal grounding and transferability; the cost is increased training complexity and potential regularization. The threshold τ for storing trajectories avoids low-quality data that could harm BC, but may discard useful exploratory trajectories; we set τ dynamically based on running statistics to balance data quantity and quality.

(D) **Why it measures what we claim**: The intrinsic reward component task_rel measures task relevance because it uses cosine similarity between the generated text embedding and the task spec embedding under a CLIP space, under the assumption that CLIP embeddings capture semantic relatedness; this assumption fails when the task spec describes abstract concepts or rare actions, in which case the similarity reflects only surface-level word overlap. The consistency score consis from M measures multimodal alignment under the assumption that M has learned a joint embedding that correlates visual states, actions, and text; this assumption fails when environment dynamics deviate from pretraining distribution, causing the alignment score to reflect modality gap rather than actual consistency. The behavioral cloning loss L_BC measures skill policy fidelity under the assumption that the collected trajectories are optimal demonstrations; this assumption fails when exploration yields suboptimal trajectories, causing the policy to imitate noisy behaviors. The text consistency loss component measures grounding under the assumption that the generated text accurately describes relevant aspects of each transition; this assumption fails when the captioning model hallucinates or omits critical details, in which case the loss penalizes the policy for mismatches that may be irrelevant.

## Contribution

(1) A framework for zero-shot skill induction that generates its own aligned multimodal training data through self-supervised exploration, eliminating dependence on curated resources. (2) A composite intrinsic reward design that jointly optimizes for task relevance and multimodal consistency, enabling efficient autonomous data generation. (3) Empirical demonstration across multiple robotic manipulation domains showing that the induced skills match or exceed performance of those trained on human-created resources.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|----------------------|
| Dataset | Simulated robotic manipulation with language instructions (MetaWorld) | Diverse tasks with multimodal grounding |
| Primary metric | Task Success Rate | Directly measures skill execution |
| Baseline 1 | Random Exploration + BC | Isolates effect of intrinsic reward |
| Baseline 2 | Task Relevance Only (no consistency) | Tests importance of multimodal consistency |
| Baseline 3 | Human-Demonstration BC | Upper bound with optimal trajectories |
| Ablation-of-ours | ASIIE w/o text consistency loss | Isolates text consistency contribution |

### Why this setup validates the claim
This combination of dataset, baselines, and metric creates a falsifiable test of ASIIE's central claim: that composite intrinsic reward (task relevance + multimodal consistency) and text-consistency loss improve zero-shot skill induction from language. The dataset requires grounding language in visual and motor spaces, matching our method's multimodal focus. Task Success Rate directly reflects whether the induced policy executes the specified task. Random Exploration + BC tests whether the intrinsic reward adds value over naive exploration. Task Relevance Only tests whether multimodal consistency is necessary beyond task relevance. Human-Demonstration BC sets an upper bound, showing headroom. The ablation isolates the text consistency loss's contribution. If ASIIE outperforms all but the oracle baseline, with the ablation performing worse, the core mechanism is validated. Differences across tasks of varying multimodal alignment difficulty further pinpoint where our method helps.

### Expected outcome and causal chain

**vs. Random Exploration + BC** — On a task like "slide block to the left", random exploration often produces aimless movements with weak task relevance. The baseline collects many low-quality trajectories, so behavioral cloning yields a noisy policy that fails to consistently slide the block. Our method's intrinsic reward guides the agent toward task-relevant interactions (pushing the block) while penalizing inconsistent state-text-action pairs (e.g., pushing without caption coherence). Thus we expect a large success gap (e.g., 70% vs. 20%) on tasks requiring precise object manipulation, and smaller gaps on tasks with high reward density.

**vs. Task Relevance Only (no consistency)** — On a task like "pick up the red cube", the relevance-only agent may fixate on the color but produce actions that don't align with the visual state (e.g., closing gripper before contact). The baseline's trajectories can be semantically relevant but physically inconsistent. Our method's consistency term penalizes such mismatches, yielding smoother, more coherent trajectories. We expect a moderate advantage (e.g., 60% vs. 40%) on tasks that require tight action-state coupling, such as grasping or stacking.

**vs. Human-Demonstration BC** — On any task, human demonstrations are near-optimal, so the oracle baseline achieves near-perfect success (e.g., 95%). Our method may plateau lower (e.g., 80%) because exploration cannot fully cover the human-level trajectory space, especially for rare or complex motions. We expect our method to approach but not exceed this upper bound, with the gap smaller on simple tasks (e.g., 90% vs. 95%) and larger on multi-step tasks (e.g., 60% vs. 95%).

**vs. ASIIE w/o text consistency loss** — On a task with visual ambiguities (e.g., "move the blue mug" where multiple blue items exist), the ablation may produce policies that ignore subtle visual details because its loss only optimizes action imitation. Our method's text consistency loss forces the policy to align its state embeddings with generated captions, improving discrimination. We expect a clear gap (e.g., 75% vs. 55%) on tasks requiring fine-grained grounding, but parity on tasks with unambiguous state-action mappings.

### What would falsify this idea
If ASIIE's success rate is close to Random Exploration + BC across all tasks, or if the improvement over Task Relevance Only is uniform (not concentrated on tasks requiring multimodal alignment), the central claim that the composite intrinsic reward and text consistency loss drive skill acquisition would be invalid.

## References

1. RESOURCE2SKILL: Distilling Executable Agent Skills from Human-Created Multimodal Resources
