# Factorized Scene-Task Memory for Environment-Invariant Robot Skill Evolution

## Motivation

Existing lifelong memory systems for robotic agents, such as ABot-AgentOS's Universal Multi-modal Graph Memory, store monolithic representations that tightly couple task knowledge with specific environmental layouts. This coupling causes catastrophic forgetting when agents encounter new environments: learned manipulation skills fail because they are tied to scene-specific coordinates and object arrangements. The root cause is that task-relevant information (e.g., grasp poses relative to object frames) is not separated from scene-relevant information (e.g., object locations in the global map).

## Key Insight

Manipulation primitives are inherently invariant to global scene layout; by storing them in object-relative coordinates, they can be reused across different environments without retraining, while scene memory adapts to new object instances and layouts independently.

## Method

### Factorized Scene-Task Memory (FSTM)

We assume that object-relative skill parameters (e.g., grasp poses) are invariant across different object instances of the same category, provided that the object's functional parts (e.g., handle) are consistently oriented relative to the object frame. This assumption underlies the separation of task memory from scene memory.

**(A) What it is:** Factorized Scene-Task Memory (FSTM) is a dual-memory architecture for lifelong robotic agents. It takes RGB-D observations, robot joint states, and a task goal as input, and outputs an action sequence or skill primitive by separating memory into a Task Memory (TM) storing skill templates in object-relative coordinates, and a Scene Memory (SM) storing object instances and layouts.

**(B) How it works:**
```python
# Pseudocode for one task execution
class FSTM:
    def __init__(self):
        self.TM = TaskMemory()  # map: (object_category, skill_name) -> list of (params, success_count, variance)
        self.SM = SceneMemory()  # 3D scene graph: nodes = objects with category, pose, attributes
        self.variance_threshold = {'position': 0.05, 'rotation': 0.1}  # meters^2, rad^2

    def execute_task(self, observation, goal):
        # Step 1: Update Scene Memory
        objects = detect_objects(observation)  # using YOLOv5 with pretrained weights
        self.SM.update(objects)

        # Step 2: Retrieve skill templates from Task Memory
        required_skills = decompose_goal(goal)
        for skill in required_skills:
            obj_cat = skill.get_object_category()
            if obj_cat in self.TM and skill.name in self.TM[obj_cat]:
                # Retrieve the best parameter set (highest success count)
                entries = self.TM[obj_cat][skill.name]
                best_entry = max(entries, key=lambda e: e['success_count'])
                params = best_entry['params']
                variance = best_entry['variance']
                if variance > self.variance_threshold:
                    # Re-sample via exploration: add Gaussian noise
                    noise = {'position': np.random.normal(0, 0.02, 3),
                             'rotation': np.random.normal(0, 0.05, 3)}
                    params = {'position': params['position'] + noise['position'],
                              'rotation': params['rotation'] + noise['rotation']}
            else:
                params = default_skill_params(skill)  # heuristic: centroid + top grasp

            # Step 3: Ground to global actions using Scene Memory
            obj_instance = self.SM.get_instance(obj_cat)
            global_actions = transform(params, obj_instance.pose)

            # Step 4: Execute via low-level controller (Panda robot, impedance control)
            success = execute(global_actions)

            # Step 5: Update Task Memory with outcome
            if success:
                if obj_cat not in self.TM:
                    self.TM[obj_cat] = {}
                if skill.name not in self.TM[obj_cat]:
                    self.TM[obj_cat][skill.name] = []
                # Store this parameter set with success_count=1 and moving variance
                self.TM[obj_cat][skill.name].append({'params': params, 'success_count': 1, 'variance': 0.0})
                # Keep only last 10 successful entries to bound memory
                if len(self.TM[obj_cat][skill.name]) > 10:
                    self.TM[obj_cat][skill.name].pop(0)
                # Update variance estimate across all stored entries
                all_params = [e['params'] for e in self.TM[obj_cat][skill.name]]
                variance = compute_variance(all_params)  # across stored entries
                for e in self.TM[obj_cat][skill.name]:
                    e['variance'] = variance
```
Hyperparameters: `variance_threshold = {'position': 0.05, 'rotation': 0.1}`; exploration noise std = 0.02m, 0.05rad; max entries per skill = 10; object detector = YOLOv5 (pretrained on COCO).

**(C) Why this design:** We chose object-relative coordinate representation over world-coordinate storage because manipulation tasks like grasping are defined relative to object frames; world-coordinate storage would require re-learning for each new layout. The cost is that we need robust perception to estimate object poses accurately. We decoupled skill templates from scene instances rather than storing monolithic episodic memories, because episodic memories are environment-specific and cause interference when different environments have different object layouts. The trade-off is that we lose fine-grained temporal context (e.g., object movement history), but we argue that for skill evolution, the successful parameters generalize across scenes. We used weighted averaging across experiences for task memory refinement instead of online gradient-based learning, because we want to avoid catastrophic interference and ensure stability; the cost is slower adaptation to new tasks but prioritizes retention across environments.

**(D) Why it measures what we claim:** The computational quantity "success count per skill-object category pair" measures "cross-environment generalizability" because a high success count aggregated across environments indicates that the skill parameters transfer; this equivalence relies on the assumption that object-relative skill parameters are invariant across environments. This assumption fails when object affordances vary (e.g., a mug in one environment has a different handle orientation relative to the mug frame), in which case success count may reflect overfitting to a dominant environment rather than true invariance. To mitigate, we also store variance of successful parameters; high variance triggers a re-sampling of the skill from diverse experiences.

## Contribution

(1) A factorized memory architecture that separates task memory (object-relative skill templates) from scene memory (object instances and layouts) for lifelong robot learning. (2) A cross-environment aggregation rule for task memory that updates skill parameters using only successful outcomes and a moving average, preventing catastrophic forgetting. (3) An empirical demonstration on a suite of diverse robotic manipulation environments showing improved success rate and reduced interference compared to monolithic memory baselines.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | EmbodiedWorldBench (16 scenes, 200+ tasks) | Cross-environment test with varying object layouts |
| Primary metric | Success rate per skill-object pair | Directly measures generalizability across layouts |
| Baseline 1 | ABot-AgentOS | Stores monolithic episodic memories; world coordinates |
| Baseline 2 | Remember Me Refine Me | Passive accumulation; no scene-task separation |
| Baseline 3 | Evo-Memory | Self-evolving but no factorization |
| Ablation-of-ours | FSTM w/o Task Memory update (no variance trigger) | Removes cross-environment aggregation and adaptation |

### Why this setup validates the claim

EmbeddedWorldBench provides multiple distinct environments with varying object layouts, which is ideal for testing cross-environment generalizability—the central claim of FSTM. The primary metric, success rate per skill-object category pair, directly captures whether skill parameters learned in one scene successfully transfer to another for the same object category. ABot-AgentOS serves as a baseline representing systems that store memories in world coordinates, which should fail when layouts differ. Remember Me Refine Me tests passive accumulation without factorization, while Evo-Memory tests self-evolution without explicit scene-task separation. The ablation (no task memory update, including the variance trigger) isolates the benefit of cross-environment aggregation and adaptation, controlling for other architectural choices. This combination ensures that any observed advantage can be attributed to the factorized memory design and its update rule, with the metric providing a direct signal of the predicted effect. To test the invariance assumption explicitly, we will compute the distribution of successful grasp parameters across environments for each object category; if invariance holds, the variance should be low (below threshold), otherwise variance will be high and we will trigger re-sampling.

### Expected outcome and causal chain

**vs. ABot-AgentOS** — On a task where the robot must pick a mug placed at a new location on a table, ABot-AgentOS's world-coordinate memory would activate a stored grasp pose from a previous environment that no longer aligns with the current mug pose, causing grasp failure because it lacks object-relative invariance. Our method instead retrieves the skill template in mug-relative coordinates and transforms it using the current mug's detected pose, so the grasp succeeds regardless of table layout. We expect a noticeable gap on tasks with varied layouts (e.g., 20% higher success for FSTM) but near parity on tasks with identical layouts.

**vs. Remember Me Refine Me** — Consider a task requiring picking a cup after previously encountering a cup with a different handle orientation. Remember Me Refine Me would average all past cup configurations in its memory, leading to a degraded grasp that fails because the average might not fit the current cup. Our method stores separate skill templates per object category and refines them via weighted averaging of successful parameters, which retains the successful grasp while discarding failures, so the grasp remains robust. We expect FSTM to maintain high success (e.g., >80%) while the baseline degrades after diverse experiences (e.g., <50%).

**vs. Evo-Memory** — In a sequential task stream where the robot first picks a handle-less mug, then a mug with a handle, Evo-Memory would update its memory for 'mug grasp' based on the first task, causing the second to fail because it tries to grasp the handle-less way. Our Task Memory keeps separate entries for mug subcategories (if needed via object category discrimination) or uses object-relative parameters that adapt to the mug's geometry; the handle leads to a different relative grasp point that we sample from experience. We expect FSTM to show increasing success over tasks (e.g., from 70% to 90%) while Evo-Memory plateaus or drops.

### What would falsify this idea

If our method's success rate on varied-layout tasks is not significantly higher than on fixed-layout tasks, or if the ablation (no memory update) performs comparably in cross-environment tasks, then the central claim that object-relative skill generalization drives improvement would be falsified. Additionally, if the variance of successful parameters across environments remains high (above threshold) for all object categories, indicating that the invariance assumption is not supported, the method would require redesign.

## References

1. ABot-AgentOS: A General Robotic Agent OS with Lifelong Multi-modal Memory
2. Remember Me, Refine Me: A Dynamic Procedural Memory Framework for Experience-Driven Agent Evolution
3. Evo-Memory: Benchmarking LLM Agent Test-time Learning with Self-Evolving Memory
4. NavForesee: A Unified Vision-Language World Model for Hierarchical Planning and Dual-Horizon Navigation Prediction
5. StreamBench: Towards Benchmarking Continuous Improvement of Language Agents
6. LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory
7. ImagineNav: Prompting Vision-Language Models as Embodied Navigator through Scene Imagination
8. NaVILA: Legged Robot Vision-Language-Action Model for Navigation
