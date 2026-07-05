# OEMA: Online Episodic Memory Adaptation for Embodied Reasoning Models

## Motivation

Vesta and its ancestors assume a static pre-collected dataset suffices for all scenarios, but this fails when the model encounters novel situations requiring online adaptation. The root cause is the lack of an online memory writing mechanism that uses environment feedback as a training signal without weight updates. For example, Vesta's multimodal memory harness supports extended temporal reasoning but cannot write new memories from interaction feedback, limiting its ability to learn from successes and failures at test time.

## Key Insight

Environment feedback (success/failure) provides a natural credit assignment signal that can be used to selectively write compressed episodic memories, enabling test-time adaptation through non-parametric memory retrieval rather than parametric fine-tuning.

## Method

**Algorithm: Online Episodic Memory Adaptation (OEMA)**

(A) **What it is:** OEMA extends Vesta's multimodal memory harness with an online memory writing module that, upon task completion, compresses the interaction episode into a key-value memory entry using a success-triggered consolidation rule.

(B) **How it works:**
```python
# OEMA - Online Episodic Memory Adaptation
# Input: Vesta model (ViT-B/16 backbone, memory harness M_base with pre-filled episodic memory)
#        Current observation sequence o_t (RGB images 224x224), action sequence a_t (discrete or continuous), reward/feedback r_t (binary 0/1)
#        Threshold t_success for writing new memory
# Output: Updated memory M (including new episode if successful)

# Load-bearing assumption: Binary success/failure feedback is a reliable indicator of episode quality.
# Calibration: Prior to main experiments, run 100 random rollouts per task to estimate luck-based success rate.
#   Set t_success = max(1.0, 90th percentile of random success rate). For stochastic environments, t_success = 2.0 (require two consecutive successes) or use confidence threshold based on policy entropy (write if entropy < 0.5).

def oema_interaction(vesta_model, environment, task_goal):
    buffer = []  # working buffer of (obs, action, reward) tuples
    obs = environment.reset(task_goal)
    while not environment.done:
        # Vesta reasons with current memory M_base and retrieves relevant episodes
        action = vesta_model.select_action(obs, task_goal, memory=M_base)
        next_obs, reward, done = environment.step(action)
        buffer.append((obs, action, reward))
        obs = next_obs
    final_reward = reward if done else 0
    if final_reward >= t_success:
        # Compress buffer into a compact episode representation
        # Encode key from initial observation and task goal: key = concat(goal_embed, obs_embed) where goal_embed is from goal encoder (BERT-base, 768d) and obs_embed from spatial_grounder (ViT-B/16, 768d) -> final key dimension 768 with projection
        key = vesta_model.spatial_grounder.encode(task_goal, buffer[0][0])
        # Compress value by extracting salient states: use a Transformer encoder (2 layers, 4 heads, hidden 512) that processes the sequence of (obs, action) tokens (obs encoded as 768d, action as one-hot or embedding) and outputs a 512-dim value vector
        value = vesta_model.multimodal_fusion.compress(buffer)
        # Deduplicate: cosine similarity on keys (dim 768) with threshold 0.9
        existing_keys = M_base.get_keys()
        if all(cosine_similarity(key, ek) < 0.9 for ek in existing_keys):
            M_base.write(key, value)
    return final_reward
```
Hyperparameters: `t_success = 1.0` (or calibrated as above), deduplication threshold `= 0.9` cosine similarity. The key encoder and compressor add ~50M parameters and 2 GB GPU memory overhead.

(C) **Why this design:** We chose to write only successful episodes to avoid polluting memory with noisy failures, accepting the cost that the model may not learn from failures directly, but failures can still influence behavior indirectly by not being reinforced. We chose a key-value memory with key derived from initial observation and goal encoding because it enables efficient retrieval at test time by matching the current task context; the trade-off is that this requires a reliable encoder, but Vesta's spatial grounding provides that. We chose a deduplication step to prevent memory bloat from near-identical episodes, which maintains retrieval efficiency at the cost of potentially losing nuanced variations. We chose to compress the buffer using multimodal fusion rather than storing raw sequences to reduce memory footprint and force the model to extract causally relevant information, but this compression may discard fine-grained details that could be useful. Compared to prior work like VLM^2 which updates episodic memory via a consolidation process likely offline, our method uses explicit success feedback to trigger writing, a simpler and more direct adaptation signal aligned with the task objective.

(D) **Why it measures what we claim:** The writing threshold (t_success) operationalizes the motivation concept of "learning from successes" because we assume that environment feedback (final_reward) is a reliable indicator of task success; this assumption fails when the reward is sparse or noisy (e.g., only given at the end with no partial credit), in which case the metric reflects only terminal success, not intermediate progress. The deduplication similarity threshold (0.9) measures the concept of "novelty" in episodes because we assume that cosine similarity on the key encoding captures semantic similarity of tasks; this assumption fails when two different tasks share the same initial observation but require different strategies, in which case deduplication may incorrectly merge distinct experiences. The compression step (encode_key, compress_buffer) measures the concept of "causally relevant memory" because we assume that the multimodal fusion can extract the minimal sufficient state for future task success; this assumption fails when the causal structure is complex (e.g., long-term dependencies), in which case compression may discard temporally distant but crucial information.

## Contribution

(1) A simple yet effective online memory writing mechanism for embodied reasoning foundation models that uses environment feedback to update episodic memory at test time without weight updates. (2) A design principle: selective writing of successful episodes with key-based deduplication enables efficient test-time adaptation, avoiding catastrophic forgetting and maintaining inference speed. (3) Empirical demonstration (to be performed) that OEMA improves Vesta's performance on novel scenarios by leveraging its own past successes.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|-----------------------|
| Dataset | ObjectNav on Gibson v2 (200 scenes) & ALFRED kitchen tasks (100 episodes each) | Tests memory adaptation across diverse tasks with real-world complexity. |
| Primary metric | Task Success Rate | Direct measure of task completion; averaged over 5 seeds. |
| Baseline | Vesta (original) | Isolates OEMA's improvement over static memory. |
| Baseline | VLM^2 | Compares online vs offline consolidation (VLM^2 uses offline clustering). |
| Baseline | No memory (transformer context only) | Baseline for memory benefit. |
| Ablation-of-ours | OEMA without deduplication | Tests deduplication's role in memory quality and bloat prevention. |
| Ablation-of-ours | OEMA without compression (store full raw sequence) | Tests compression's effect on memory retrieval quality; memory size capped at 50 episodes. |

### Why this setup validates the claim

The proposed OEMA method claims that online, success-triggered episodic memory adaptation improves embodied reasoning. This experimental setup provides a falsifiable test by combining a dataset (ObjectNav & ALFRED) that requires both memory and adaptation (varying layouts, object locations), a primary metric (Success Rate) that directly measures task performance, and baselines that isolate each component of the claim. Vesta (original) tests whether adaptation itself yields benefit over static memory; VLM^2 tests whether online consolidation is superior to offline; and No memory tests the baseline necessity of external memory. The ablations test whether deduplication (to avoid bloat) and compression (to force causal extraction) are crucial for OEMA's performance. If OEMA fails to outperform Vesta, the central adaptation claim is unsupported. If OEMA matches VLM^2, online writing may not be advantageous. If the ablation without deduplication matches OEMA, deduplication is not vital; similarly for compression. The metric captures the predicted effect because success rate directly reflects whether the model can leverage past episodes to complete tasks in new instances. Expected compute: 200 GPU hours on a single A100 for full evaluation (including calibration and 5 seeds).

### Expected outcome and causal chain

**vs. Vesta (original)** — On a task where the initial strategy fails but the agent can later discover a successful alternative (e.g., navigating through a blocked doorway), Vesta with static memory repeats the same error because it cannot update its knowledge. OEMA writes the successful episode after the alternative path is found, so on subsequent trials it retrieves the correct plan. We expect OEMA to achieve 85% success rate vs Vesta 60% on multi-solution tasks (e.g., ALFRED with multiple possible action sequences), but parity (~90%) on single-fixed tasks (e.g., simple pick-and-place).

**vs. VLM^2** — On a task where immediate online feedback is available (e.g., stepwise rewards for object interactions), VLM^2's offline consolidation (clustering episodes after a session) delays memory updates until after the session, so it cannot adapt within the same task instance. OEMA writes immediately upon success, enabling intra-episode learning (e.g., adjusting to a new object behavior). We expect OEMA to reach 70% vs VLM^2 50% on tasks requiring rapid adaptation (e.g., ALFRED with changing object dynamics), with smaller differences (~80% vs 75%) on tasks where offline consolidation suffices.

**vs. No memory** — On a long-horizon task requiring recall of past observations (e.g., remembering the location of a key seen earlier in a 10-step episode), the No memory baseline suffers from limited transformer context window (2048 tokens) and fails. OEMA stores relevant episode data and can retrieve the critical state at decision time. We expect OEMA to achieve 80% vs No memory 30% on long tasks (≥15 steps), but similar performance (~90% both) on short tasks (≤5 steps) where context fits in the base model.

### What would falsify this idea
If OEMA's improvement over Vesta is uniform across all task types (e.g., gains of 10% on both short and long tasks) rather than concentrated on tasks requiring adaptation (e.g., multiple solution paths or long horizons), then the central claim that success-triggered writing drives improvement would be unsupported. Furthermore, if the ablation without compression matches or exceeds OEMA, the compression step is unnecessary and the method's efficiency benefit is lost.

## References

1. Vesta: A Generalist Embodied Reasoning Model
2. Qwen3-VL Technical Report
3. Vision-Language Memory for Spatial Reasoning
4. RoboSpatial: Teaching Spatial Understanding to 2D and 3D Vision-Language Models for Robotics
5. Intern VL: Scaling up Vision Foundation Models and Aligning for Generic Visual-Linguistic Tasks
6. VILA: On Pre-training for Visual Language Models
7. EmbodiedScan: A Holistic Multi-Modal 3D Perception Suite Towards Embodied AI
8. Scaling Instruction-Finetuned Language Models
