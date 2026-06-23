# Compositional Reward Estimation via Primitive Decomposition for Scalable Oversight of Superintelligent Agents

## Motivation

Existing reward modeling approaches, such as PPO-based RLHF with hybrid oversight (from 'PPO-based RLHF with Hybrid Oversight'), struggle to generalize to novel complex tasks because the reward model must extrapolate from seen states. For superintelligent tasks, the state space is vast and novel compositions of actions arise. Prior debate protocols (e.g., 'Debate Helps Supervise Unreliable Experts') provide scalable oversight but rely on adversarial setup and do not directly estimate rewards. The root cause is that reward identifiability—the property that human feedback uniquely determines the reward function—is lost when tasks are too complex for holistic human evaluation. We need a structural decomposition that re-establishes identifiability by confining human verification to simple, verifiable primitives.

## Key Insight

Because primitive actions are independently verifiable by humans and the composition function is learned from human-provided global rewards on a small set of composite tasks, the reward function is identifiable on all compositions that share the same primitive space and compositional structure.

## Method

**Compositional Reward Estimation via Primitive Decomposition for Scalable Oversight (CREPD)**

### Phase 1: Decomposition Specification
- Human engineers define a set P of primitive action types, each with a clear verification criterion (e.g., "move to coordinate (x,y)", "pick up object O"). Assumption: Each primitive action type is independently verifiable by humans (i.e., a human can reliably judge its correctness without context of the composite task). Assumption: The composition function f learned from a limited set of composite tasks generalizes to all novel compositions sharing the same primitive space and compositional structure.
- For each composite task T, automatically decompose its plan into a sequence of primitives p_1,...,p_K, each belonging to P.

### Phase 2: Primitive Reward Verification
- For each primitive p_i in a small held-out set of primitive instances (e.g., 1000 examples per primitive type), collect human binary feedback (good/bad) or preference pairs. Use these to train a primitive reward model r_prim(p) that outputs a scalar reward in [0,1] representing human-estimated value.
  - Architecture: 2-layer MLP, hidden sizes [64, 32], ReLU activation, output sigmoid.
  - Training: Binary cross-entropy loss, Adam optimizer, learning rate 1e-3, batch size 64, trained for 50 epochs.
- Train a primitive verifier V(p) that outputs a confidence score (e.g., probability that human would approve).
  - Architecture: 2-layer MLP, hidden size 64, output sigmoid (same as r_prim but trained with calibration: after training, apply temperature scaling with T=1.2 on a held-out calibration set of 512 examples).
  - Training: Binary cross-entropy loss, same optimizer and schedule as r_prim.

### Phase 3: Composition Function Learning
- Collect human global rewards for a set of composite tasks R_global(T) (e.g., pairwise preferences). Use 500 composite tasks covering diverse primitive combinations and orderings.
- Represent each composite task T by the multiset of primitive rewards (r_prim(p_1),...,r_prim(p_K)) and primitive confidences (V(p_1),...,V(p_K)).
- Learn a composition function f: ℝ^{2K} -> ℝ (global reward) by minimizing mean squared error between f(primitive rewards, confidences) and observed global rewards. Use a small transformer encoder with attention over primitives. Hyperparameters: 2 layers, 4 heads, hidden dim 64, learning rate 1e-4, weight decay 0.01, Adam optimizer, batch size 32, trained for 5000 steps.
- During deployment, periodically collect human global rewards on a small set of composite tasks (e.g., 10 per day) and perform online fine-tuning of the composition function using the same loss (MSE) to correct for distribution shift.

### Phase 4: Reward Estimation for New Tasks
- For a new task T', decompose into primitives, compute r_prim and V for each primitive using the trained models, then compute global reward = f(primitive rewards, confidences).

(C) **Why this design**: We chose a human-specified primitive decomposition over a learned decomposition because learned decomposition risks producing primitives that are not independently verifiable or are entangled (anti-pattern 1). The cost is that human engineers must define primitives for each domain, limiting generality but ensuring verifiability. We train a separate primitive reward model rather than a monolithic model because primitives have limited variation and are easier to model; this avoids overfitting to rare composite states. The composition function uses a transformer with attention to capture interactions between primitives (e.g., ordering, dependencies) rather than a simple sum or product, which would ignore sequential dependencies; the trade-off is increased sample complexity for learning the composition function. We include primitive verifier confidence as input to the composition function to indicate uncertainty in primitive rewards, allowing the composition to down-weight unreliable primitives.

(D) **Why it measures what we claim**: The primitive reward r_prim(p) measures the human-assigned value of that primitive action, assuming that human verifiability is feasible for each primitive type; this assumption fails when a primitive action is too subtle for humans to evaluate (e.g., a complex social inference), in which case r_prim reflects the human's biased or noisy judgment. The composition function f measures the structural rule by which primitive rewards combine into a global reward, assuming that the training composite tasks are representative of the compositional space; this assumption fails when novel compositions involve primitives in arrangements unseen during training (e.g., long-range dependencies), in which case f may extrapolate incorrectly. The primitive verifier confidence V(p) measures the reliability of the primitive reward estimate, assuming that the primitive reward model's confidence correlates with accuracy; this assumption fails when the model is overconfident on out-of-distribution primitives, in which case V reflects model uncertainty rather than true verifiability. We operationalize identifiability via global reward recovery error (L2 distance between predicted and true global rewards on a held-out set of 100 synthetic composite tasks with known rewards derived from the true reward function). This assumes that low recovery error implies correct identification; failure mode: composition function may overfit to training composites, which we mitigate by using held-out tasks with combinatorial variations unseen during training.

## Contribution

(1) A compositional reward estimation framework that decomposes superintelligent tasks into human-verifiable primitives, restoring reward identifiability for complex tasks. (2) A design principle that reward identifiability can be restored through primitive decomposition combined with a learned composition function that incorporates primitive confidence, enabling scalable oversight without holistic human feedback. (3) A method for training a composition function that uses a small transformer to aggregate primitive rewards and confidences, with explicit handling of uncertainty.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | SafeGridworld (10 primitive types, 1000 examples each for primitive training; 500 composite tasks for composition training; 100 synthetic composite test tasks with known true rewards) | Compositional tasks with safety primitives; allows reward recovery error. |
| Primary metric | Human preference alignment accuracy | Directly measures alignment with human values. |
| Secondary metric | Reward recovery error (L2) on 100 held-out synthetic composite tasks | Measures identifiability of true reward. |
| Baseline 1 | Standard RLHF (PPO with monolithic reward model trained on global human preferences) | Tests benefit of decomposition vs. monolithic reward. |
| Baseline 2 | Supervised fine-tuning (SFT) on demonstration data | Tests benefit of decomposition vs. pure imitation. |
| Baseline 3 | Naive sum-of-primitives reward (sum of primitive rewards without learned composition) | Tests importance of learned composition. |
| Ablation-of-ours | CREPD without verifier confidence (composition function uses only primitive rewards) | Isolates effect of confidence weighting. |

### Why this setup validates the claim
The SafeGridworld dataset provides tasks composed of primitive actions (e.g., move, pick, drop) with explicit safety constraints, enabling controlled evaluation of compositional reward estimation. By comparing against Standard RLHF, we test whether primitive decomposition improves reliability in rare composite tasks. SFT tests if explicit reward modeling outperforms pure imitation. The naive sum baseline isolates the added value of learning a composition function over simple aggregation. The ablation removes the verifier confidence input to test whether uncertainty weighting is essential. The primary metric, human preference alignment accuracy, directly assesses whether the learned rewards lead to choices that match human judgments, which is the core claim of CREPD. The secondary metric, reward recovery error on synthetic tasks with known true rewards, provides a direct measure of identifiability, linking the method's output to the underlying reward function. Together, these choices form a falsifiable test: if CREPD fails to outperform baselines precisely where the predicted failure modes occur (e.g., compositions with safety trade-offs), the claim is undermined.

### Expected outcome and causal chain

**vs. Standard RLHF** — On a task requiring a sequence where a safe primitive (e.g., pick up a red object) precedes an unsafe one (e.g., move near a hazard), the monolithic RLHF reward model may assign high global reward because it learned from training data that "pick red" correlates with overall reward, ignoring the unsafe move. CREPD instead evaluates each primitive individually: the unsafe move gets a low primitive reward from the verifier, and the composition function weights it negatively, leading to a low global reward. We expect a noticeable gap on tasks involving safety-critical primitives (e.g., >10% alignment difference; reward recovery error >0.15 lower) but parity on simple tasks.

**vs. Supervised fine-tuning (SFT)** — On a task where the optimal policy requires a sequence absent from the training demonstrations (e.g., picking a blue object then a red one, but training only showed red-blue), SFT generalizes poorly and may output an invalid or unsafe action. CREPD leverages the learned primitive rewards: each primitive action (pick blue, pick red) is still evaluated as good individually, and the composition function learns to combine them even if the order is novel, because it attends to the primitive identities and confidences. We expect CREPD to maintain alignment accuracy on novel compositions, while SFT drops sharply (e.g., 20% drop; reward recovery error >0.2 difference), visible on held-out task combinations.

**vs. Naive sum-of-primitives reward** — On a task where primitive ordering matters (e.g., first pick up a tool then use it, versus doing the reverse which is impossible), naive sum treats both sequences equally. CREPD's composition function learns from human preferences that only the correct order yields positive global reward, so it outputs a lower reward for the invalid order. We expect a gap on tasks where sequence dependencies exist (e.g., >15% accuracy difference; reward recovery error >0.1 lower), and parity on tasks where order does not matter.

### What would falsify this idea
If CREPD's alignment accuracy is no better than naive sum on tasks with sequential dependencies, or if the ablation without verifier confidence matches full CREPD across all task subsets, or if reward recovery error >0.3 on the synthetic test set for CREPD, then the central claim that primitive decomposition and learned composition improve reward estimation and ensure identifiability is falsified.

## References

1. PPO-based Reinforcement Learning with Human Feedback with Hybrid Oversight and Predictive Reward Evaluation for AGI
2. Super Co-alignment of Human and AI for Sustainable Symbiotic Society
3. On scalable oversight with weak LLMs judging strong LLMs
4. Autonomous Alignment with Human Value on Altruism through Considerate Self-imagination and Theory of Mind
5. Debate Helps Supervise Unreliable Experts
6. Scalable AI Safety via Doubly-Efficient Debate
7. Teaching language models to support answers with verified quotes
8. Self-critiquing models for assisting human evaluators
