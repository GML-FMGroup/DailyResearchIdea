# SOLAR: Online Symbolic Video Generation for Robust Chain-of-Frame Reasoning

## Motivation

OpenCoF's reliance on a static dataset of 17K samples from only 11 task families fundamentally limits generalization to unseen logical structures because the model learns dataset-specific correlations. The root cause is the static dataset assumption; training data does not cover the combinatorial space of possible reasoning chains. Our method overcomes this by replacing the fixed dataset with an online symbolic generator that produces videos with novel logical structures on demand.

## Key Insight

The combinatorial space of logical operations in a symbolic solver is enumerable and controllable, so online generation ensures the model never overfits to a finite sample of reasoning patterns.

## Method

(A) **What it is**: SOLAR is an online data generator that produces multi-step reasoning videos using a symbolic planner and a renderer, enabling the video generation model to be trained on an infinite stream of novel logical structures. Inputs: a set of primitive logical operations (e.g., spatial relations, object transformations) and a random seed for logical structure generation. Output: a video showing a valid reasoning chain from initial state to goal state.

(B) **How it works** (pseudocode):
```python
class SOLAR:
    def __init__(self, operations, max_steps=5, renderer):
        self.operations = operations  # list of symbolic actions
        self.max_steps = max_steps
        self.renderer = renderer

    def generate_video(self):
        # Step 1: Sample a random logical structure
        target_goal = self.sample_goal(self.operations)  # e.g., "block A is on top of block C"
        # Step 2: Use symbolic planner (STRIPS) to find a sequence of operations achieving goal from random initial state
        initial_state = self.random_initial_state()
        plan = self.strips_plan(initial_state, target_goal, self.operations, max_steps=self.max_steps)
        if plan is None: return None  # reject unsolvable
        # Step 3: Render each step as a video frame
        frames = []
        state = initial_state
        for op in plan:
            frame = self.renderer.render(state)  # e.g., an image
            frames.append(frame)
            state = self.apply_operation(state, op)
        # Step 4: Concatenate frames into video (with optional textual reasoning tokens)
        video = self.to_video(frames)
        # Step 5: Add ground-truth reasoning chain as textual label
        plan_text = self.plan_to_text(plan)
        return video, plan_text
```
Hyperparameters: max_steps (5), operations set (e.g., move, stack, unstack), renderer (e.g., PyBullet simulation).

(C) **Why this design**: We chose a STRIPS-style symbolic planner over an LLM-based planner because symbolic planners guarantee action feasibility and logical soundness, ensuring that generated videos are always valid reasoning chains (no hallucinations). This trade-off accepts that the logical structures are limited to the predefined operation set, but the set can be expanded. We chose random initial and goal states to maximize diversity, sacrificing coverage control for variability. We chose a physics-based renderer (e.g., PyBullet) over video retrieval because rendered videos are fully controllable and avoid dataset biases; the cost is higher computational cost per video. Finally, we include textual reasoning tokens (as in OpenCoF) to align with the existing architecture, accepting that the tokens are grounded in symbolic plans rather than learned.

(D) **Why it measures what we claim**: The online generator ensures exposure to novel logical structures because the symbolic planner can produce combinatorial variations of operation sequences and state combinations. The variable `plan` (sequence of operations) measures **logical structure novelty** because each new random seed yields a new composition of operations not seen in prior training; this assumption fails when the operation set is too small, in which case plans may repeat. The `max_steps` hyperparameter measures **reasoning depth** because longer chains require more steps; this assumption fails when the planner finds shortcuts, in which case steps don't correspond to independent reasoning steps. The `renderer` ensures **visual faithfulness** to the symbolic state because it directly translates each state into an image; this assumption fails when the renderer lacks visual diversity, causing the model to overfit to rendering style.

## Contribution

(1) A novel framework for online generation of logical reasoning videos using a symbolic planner and renderer, enabling training on unbounded logical structures. (2) A demonstration that replacing static datasets with online generation improves generalization to unseen logical structures, as measured by out-of-distribution performance. (3) A publicly available symbolic planner and renderer suite for generating diverse reasoning tasks.

## Experiment

### Evaluation Setup

| Role                | Choice                           | Rationale                              |
|---------------------|----------------------------------|----------------------------------------|
| Dataset             | CLEVRER                          | Standard benchmark for video reasoning.|
| Primary metric      | Task accuracy on multi-step Q&A   | Directly measures logical reasoning.   |
| Baseline 1          | Wan2.2-I2V-A14B (no CoF)         | Tests vanilla generation without training.|
| Baseline 2          | OpenCoF (static data)             | Tests reasoning fine-tuning on static data.|
| Ablation-of-ours    | SOLAR w/o reasoning tokens        | Isolates effect of reasoning tokens.   |

### Why this setup validates the claim
The experimental design directly tests the central claim that online generation of novel logical structures improves reasoning. By comparing against a model trained on static data (OpenCoF), we isolate the effect of exposure to an infinite stream of novel logical compositions. The baseline Wan2.2 without CoF fine-tuning tests whether any fine-tuning is needed. The ablation of reasoning tokens tests whether the symbolic grounding of tokens is crucial. The metric of task accuracy on the standard benchmark CLEVRER captures multi-step reasoning ability. If our hypothesis holds, we expect SOLAR to outperform both baselines, especially on examples requiring compositional reasoning steps not seen in static datasets. The combination of these baselines and the ablation provides a falsifiable test: if performance gains are not concentrated on novel compositions, or if the ablation matches the full method, the central claim is weakened.

### Expected outcome and causal chain

**vs. Wan2.2-I2V-A14B (no CoF)** — On a case requiring unseen multi-step reasoning (e.g., "after moving block A onto B, then stack C on A, what is the final position of C?"), the baseline produces a plausible but incorrect video because it lacks any reasoning mechanism to enforce logical consistency. Our method instead uses symbolic planning to guarantee valid chains, and the training on diverse structures teaches the model to attend to logical dependencies. Thus we expect a clear gap on novel reasoning tasks but parity on simple memorized patterns.

**vs. OpenCoF (static data)** — On a case with a novel combination of operations (e.g., rotate and then unstack, not seen in static training data), OpenCoF fails because its reasoning tokens are learned from a fixed set of plan structures, so it cannot generalize compositionally. Our method's online generator ensures exposure to arbitrary sequences, enabling the model to reason about new combinations. Thus we expect a noticeable gap on compositional reasoning subsets but similar performance on common patterns.

### What would falsify this idea
If SOLAR's performance gain is uniform across all reasoning tasks rather than concentrated on novel compositions, or if the reasoning token ablation matches full SOLAR, then the central claim of novel logical structure exposure being key would be false.

## References

1. OpenCoF: Learning to Reason Through Video Generation
