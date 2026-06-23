# Meta-Affordance: Zero-Shot Task Generalization for 3D Motion Affordance via Bilinear Decomposition

## Motivation

BridgeACT requires task-specific demonstration videos for each novel tool-use task because its affordance predictor treats the scene holistically, preventing reuse of shared tool and target dynamics across tasks. Prior zero-shot methods like UAD rely on static object pose and well-defined task structures, failing to generalize to the continuous 3D motion affordances needed for tool manipulation. The root cause is that affordance is not explicitly factorized into tool and target components, which are independently modifiable across tasks.

## Key Insight

The causal structure of tool-use affordance is bilinear in the tool and target latent spaces, enabling compositional generalization to unseen tasks by recombining learned components via a language-conditioned interaction matrix.

## Method

**Method: Meta-Affordance**

(A) **What it is**: Meta-Affordance is a meta-learning framework that, given RGB-D observations of a tool and target object and a language instruction, predicts a 3D motion affordance (a direction vector and orientation) in a zero-shot manner. It learns a bilinear decomposition of affordance into tool and target latent spaces, conditioned on task embeddings from language.

(B) **How it works** (pseudocode):
```python
# Meta-training phase (episodic)
for each episode (task T with tool_T, target_T, instruction_T):
    # Sample support set S = {(scene, affordance)}
    # Sample query set Q
    # Encode tool point cloud (from support) -> z_t = f_tool(obs_tool)
    # Encode target point cloud (from support) -> z_s = f_target(obs_target)
    # Encode instruction -> z_l = f_task(instruction)  # e.g., CLIP text embedding
    # Compute bilinear matrix: M = W_m * z_l + b_m  (W_m, b_m learnable)
    # Predict affordance: a_pred = z_t^T @ M @ z_s + bias
    # Compute loss: L = MSE(a_pred, a_gt) over support
    # Update all parameters (f_tool, f_target, f_task, W_m, b_m) via meta-optimizer
# Meta-testing (zero-shot): for new task, encode tool and target from initial observation and instruction, compute a_pred directly.
```
Hyperparameters: latent dimension d=128, learning rate 1e-4, batch size 4 tasks per episode.

(C) **Why this design**: We chose bilinear factorization over a concatenated MLP because the interaction between tool and target is inherently multiplicative (e.g., the angle of a hammer head to the target surface determines impact direction); concatenation would require the MLP to learn this structure from data, risking overfitting to spurious correlations. We chose meta-learning on a diverse set of tasks instead of joint training because joint training would treat each task independently, learning separate features for tool and target per task, whereas meta-learning encourages learning globally transferable latent spaces that work across tasks via the bilinear form. We chose to condition the bilinear matrix on language via a linear mapping rather than a separate decoder per task to maintain zero-shot capability: at test time, the instruction directly modulates the interaction without requiring task-specific fine-tuning. The trade-off is that the bilinear assumption may be restrictive if the true affordance is highly nonlinear, and meta-training requires a large corpus of diverse tasks to learn generalizable latent spaces.

(D) **Why it measures what we claim**: The bilinear product z_t^T * M * z_s measures the joint effect of tool and target on the predicted affordance under the assumption that affordance is a bilinear function of the tool and target latent representations. Specifically, the computational quantity M (the task-conditioned matrix) operationalizes the task-specific coupling between tool and target: for a given instruction, M captures how tool features should be multiplied with target features to produce the motion. This assumption holds when the physical interaction between tool and target is dominated by linear interactions in the chosen latent space (e.g., direction aligns with dominant tool and target axes). This assumption fails when the interaction is non-bilinear, e.g., when tool and target have fine-grained geometry that requires higher-order interactions; in that case, the predicted affordance may reflect only the dominant linear mode and miss subtle effects. The latent representations z_t and z_s themselves measure the identity of tool and target in a task-invariant way, learned via meta-learning to be reusable across tasks.

## Contribution

(1) A meta-learning framework that learns a bilinear decomposition of 3D motion affordance into tool and target latent spaces, conditioned on language task embeddings, enabling zero-shot generalization to unseen tasks without task-specific data. (2) A design principle that bilinear interaction is a suitable inductive bias for tool-use affordance prediction, supported by the compositional nature of tool-target physics. (3) A synthetic benchmark for evaluating zero-shot generalization of affordance models across diverse tools and targets.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | BridgeData + MetaWorld Tool Tasks | Covers diverse tools, targets, and tasks. |
| Primary metric | Success rate over 100 hold-out tasks | Direct measure of task completion. |
| Baseline 1 | TraceGen | Video-based world model, strong temporal reasoning. |
| Baseline 2 | UAD | Unsupervised affordance, pretrained on synthetic data. |
| Baseline 3 | BridgeACT | Imitation from human video, fine-tuned on robot. |
| Ablation | Ours w/o language conditioning | Tests role of task embedding. |

### Why this setup validates the claim
This setup tests the central claim that bilinear meta-learned affordances generalize zero-shot across tools and tasks. By evaluating on a diverse set of tool-use tasks (e.g., hammering, scooping) from BridgeData and MetaWorld, we stress-test generalization. The baselines represent key alternatives: TraceGen tests if temporal prediction can substitute for explicit affordance; UAD tests if unsupervised affordances capture tool-target interactions; BridgeACT tests if imitation from human video can bypass affordance reasoning altogether. The ablation isolates the contribution of language-conditioned bilinear interaction. The success rate metric directly measures whether the predicted affordance leads to successful task completion, which is the ultimate test of utility. If our method outperforms, it confirms that the bilinear form captures essential structure; if not, the assumption is falsified.

### Expected outcome and causal chain

**vs. TraceGen** — On a case where a novel hammer shape is used with a nail at an odd angle, TraceGen predicts pixel-level video frames that may hallucinate impossible motion due to lack of physical affordance reasoning, leading to robot actions that miss the nail. Our method directly predicts a 3D direction from tool and target latent features via language-conditioned bilinear mapping, so it correctly aligns hammer head to nail surface. We expect our success rate to be 20-30% higher than TraceGen on such unseen combinations, while they may match on familiar ones.

**vs. UAD** — On a task like "scoop rice from bowl" with a novel spoon-like tool, UAD predicts a generic affordance (e.g., upward pull) based on synthetic training, which fails because the scoop needs a specific concave motion. Our meta-learned bilinear representation captures the interaction between spoon shape and rice pile via the task-conditioned matrix, producing a scooping trajectory. We expect our method to exhibit a 15-25% success rate advantage on tool-intensive tasks, with parity on simple pushing tasks.

**vs. BridgeACT** — On a task requiring fine manipulation like "place key in lock" where human video demonstrations are sparse, BridgeACT's policy overfits to the few examples and fails on slight variations in key shape. Our zero-shot affordance predicts the insertion direction from the tool-token and target-token interaction without requiring robot demos. We expect ours to achieve 30-40% higher success on such precision tasks, but BridgeACT may exceed on tasks well-covered by human videos.

### What would falsify this idea
If our success rate is not significantly higher than UAD or TraceGen on tool-target interaction tasks (e.g., hammering, scooping) and improvements are uniform across all tasks, then the bilinear assumption and meta-learning do not capture affordance structure better than simpler alternatives.

## References

1. BridgeACT: Bridging Human Demonstrations to Robot Actions via Unified Tool-Target Affordances
2. TraceGen: World Modeling in 3D Trace Space Enables Learning from Cross-Embodiment Videos
3. UAD: Unsupervised Affordance Distillation for Generalization in Robotic Manipulation
4. MegaSaM: Accurate, Fast, and Robust Structure and Motion from Casual Dynamic Videos
5. Find any Part in 3D
6. RoboPoint: A Vision-Language Model for Spatial Affordance Prediction for Robotics
7. Metric3D: Towards Zero-shot Metric 3D Prediction from A Single Image
8. Deep geometry-aware camera self-calibration from video
