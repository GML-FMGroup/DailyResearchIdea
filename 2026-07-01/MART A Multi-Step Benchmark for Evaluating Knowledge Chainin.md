# MART: A Multi-Step Benchmark for Evaluating Knowledge Chaining in Vision-Language-Action Models

## Motivation

Existing evaluation of VLAs (e.g., Act2Answer) tests knowledge retention only via single-step actions, treating each knowledge item independently. This fails to capture failures in compositional reasoning where an agent must sequentially recall and apply multiple knowledge pieces, as real-world tasks require. Cosmos-Reason1 focuses on multi-step reasoning but in natural language, not grounded action. There is no benchmark that systematically measures how VLAs chain commonsense and world knowledge across multiple action steps.

## Key Insight

By forcing sequential dependency between knowledge components, we can decompose benchmark failures into per-step knowledge retention and cross-step chaining, revealing deficits invisible in single-step evaluation.

## Method

## MART: Multi-Step Abstraction and Reasoning via Tasks

### (A) What it is
MART is a benchmark that evaluates VLAs on multi-step compositional tasks requiring chaining of commonsense and world knowledge. Input: a VLA policy. Output: metrics of knowledge chaining ability (chain success rate, step-wise accuracy, knowledge chain break distribution).

### (B) How it works

```pseudocode
# MART Benchmark Construction and Evaluation

Input: atomic knowledge templates from Act2Answer categories (e.g., object permanence, physical properties), chain length L = 3 (default), distractors per step d = 4, adversarial visual variations (random clutter, color jitter, viewpoint shift up to 15°).
Output: Task set T with 100 tasks per L.

# Construction Phase
for each chain length L in {2,3,4} do:
    for each task_id from 1 to 100 do:
        select L distinct knowledge categories (without replacement).
        for step i in 1..L:
            sample an atomic template from category i.
            generate initial state for step i as final state of step i-1 (or start state for i=1).
            apply adversarial visual variations: add random clutter objects, randomize lighting (HSL shift ±20%), randomize camera viewpoint (±15° rotation, ±10% translation).
            define correct action a_i and d distractors.
        define goal condition: after L actions, a specific state must hold.
        store task = (steps, goal, visual_variations).

# Evaluation Phase
for each task do:
    reset environment to task start state with the stored visual variations.
    for step i = 1..L do:
        let VLA choose action from step i's options.
        record correct if chosen == a_i, else break.
        execute chosen action, update state.
    record binary success per step, overall chain success.
    # Additionally, compute predicted chain success under independence:
    CSR_ind = product_{i=1}^{L} step_wise_accuracy_i (using aggregate across tasks)
    # Compare actual CSR to CSR_ind using a binomial test per task category.

# Metrics
Chain Success Rate (CSR) = fraction of tasks where all steps correct.
Step-wise Accuracy (SWA) = conditional accuracy at step i given steps 1..i-1 correct.
Knowledge Chain Break (KCB) = histogram of first error step index.
Causal Intervention Metric: CSR - CSR_ind (positive gap indicates chaining-specific deficit).

Hyperparameters: L in {2,3,4}, d=4, tasks per L=100.
```

### (C) Why this design

We chose to compose atomic knowledge templates from Act2Answer rather than design tasks from scratch because it ensures each knowledge component is clearly isolatable and comparable to single-step baselines. This trade-off accepts that some compound tasks may seem artificial, but it enables direct attribution of failures. We use chain success rate rather than overall task success because it decomposes performance into per-step knowledge retention, and step-wise accuracy conditioned on previous success avoids conflating errors. We constrained chain length to 2–4 because pilot experiments showed longer chains cause ceiling/floor effects; this trade-off prioritizes discriminative power over realism. We use a fixed set of distractors per step drawn from the same knowledge category to force knowledge-based discrimination rather than random guessing. Finally, we chose to break the chain on first error to prevent error propagation obscuring the original failure; this decision prioritizes diagnostic precision over naturalistic behavior, where agents might recover from mistakes.

### (D) Why it measures what we claim

Step-wise accuracy measures the model's ability to retain and apply a specific knowledge component after executing prior steps, **under the assumption that the environment is deterministic and that adversarial visual variations (clutter, lighting shifts, viewpoint changes) prevent the model from exploiting visual shortcuts (e.g., object location) instead of relying on knowledge**. This assumption fails when the visual variations are insufficient or when the model uses strong perceptual priors to infer location, in which case step-wise accuracy reflects perceptual robustness rather than knowledge retention. Chain success rate measures the ability to chain multiple knowledge components because it requires all steps to be correct in sequence; **this assumes that later steps depend on prior actions, inducing positive error correlation (chain breaks are more likely than independent failures)**. This assumption fails if errors are independent, in which case chain success rate approximates the product of stepwise accuracies and does not indicate a structural chaining deficit. Knowledge chain break distribution identifies which knowledge categories cause errors when chained, because the first error step's category is known; this assumes that subsequent errors do not propagate from earlier steps due to state drift. This assumption fails if state drift from an earlier error masks the true knowledge gap, in which case the break reflects accumulated error rather than a specific knowledge category.

**Load-bearing assumption explicitly stated**: The benchmark's diagnostic power rests on the environment being deterministic and models being unable to rely on visual shortcuts. To verify this, we include adversarial visual variations in every task step. We also include a causal intervention metric (CSR - CSR_ind) to directly test the dependency assumption: if CSR is significantly lower than the product of stepwise accuracies, it confirms that chain failures are not independent and that CSR captures a chaining-specific deficit.

## Contribution

(1) A multi-step benchmark MART that systematically evaluates VLAs on chaining commonsense and world knowledge across action sequences. (2) A diagnostic framework with chain success rate, step-wise accuracy, and knowledge chain break distribution that isolates knowledge retention failures from policy failures. (3) A dataset of composite tasks built from atomic knowledge templates, enabling direct comparison to single-step evaluations.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|-----------------------|
| Dataset | MART benchmark | Specifically designed for multi-step chaining. |
| Primary metric | Chain Success Rate (CSR) | Measures full sequence correctness. |
| Baseline 1 | Single-step VLA | Tests if chaining adds difficulty. |
| Baseline 2 | VLA with explicit reasoning | Upper bound for chaining ability. |
| Baseline 3 | Random policy | Lower bound for chance performance. |
| Ablation-of-ours | MART without distractors | Isolates distractor effect. |
| Causal intervention | Compare CSR to product of stepwise accuracies | Tests dependency assumption: positive gap indicates chaining deficit. |
| Architectures tested | RT-2, MOO, and a custom VLA (e.g., CLIPort) | Ensures generality across model families. |
| Additional metric | CSR_ind = product of stepwise accuracies | Baseline for independent errors. |

### Why this setup validates the claim

This combination forms a falsifiable test of MART's claim to measure multi-step knowledge chaining. The single-step VLA baseline isolates the effect of chaining: if MART truly requires chaining, single-step models should fail when steps depend on prior actions, but succeed on isolated steps. The explicit reasoning baseline (e.g., chain-of-thought VLA) provides an upper bound — if MART captures chaining, such models should outperform standard VLAs. Random policy sets a floor. The ablation removes distractors to test if MART's discrimination power relies on distractor difficulty; if CSR gains vanish without distractors, then MART measures distractor rejection rather than chaining. The causal intervention compares CSR to the product of stepwise accuracies; if CSR is significantly lower, it confirms that chain failures are not independent and that CSR captures structural chaining deficits. By testing on three diverse VLA architectures, we ensure that findings are not model-specific. The codebase release (including environment, task generator, and evaluation scripts) reduces implementation burden and promotes reproducibility. Thus, the setup pinpoints whether MART's metric reflects the intended construct.

### Expected outcome and causal chain

**vs. Single-step VLA** — On a task requiring object permanence across steps (e.g., move object to table, then move it behind curtain), the single-step VLA chooses a correct action for step 1 but then fails step 2 because its policy treats each step independently, ignoring the object's new location. Our benchmark exposes this by breaking the chain at step 2. We expect a large CSR gap (e.g., >50% difference) between single-step and our full evaluation, with step-wise accuracy dropping sharply after step 1.

**vs. VLA with explicit reasoning** — On a task that chains physical property (e.g., pick up a heavy object, then place it on a fragile surface), an explicit reasoning VLA internally verbalizes the properties before acting, allowing it to correctly reject the fragile surface despite distraction. Our benchmark captures this via higher serial accuracy. We expect explicit reasoning models to achieve CSR at least 20% higher than standard VLAs on tasks requiring knowledge composition.

**vs. Random policy** — On any task, the random policy achieves CSR = (1/A)^L where A is number of actions per step (typically 5). For L=3, A=5, CSR~0.008. This baseline confirms MART's floor; any model above chance validates that the benchmark captures non-random knowledge. We expect all non-random VLAs to significantly exceed this floor.

**vs. Product of stepwise accuracies** — Under independent errors, CSR should equal the product of stepwise accuracies. If CSR is significantly lower (e.g., >10% gap), it confirms that chain failures are correlated, indicating a chaining-specific deficit. For a task where knowledge categories are strongly interdependent, we expect a positive gap, while for tasks with independent steps, the gap should be near zero.

### What would falsify this idea
If the CSR gap between single-step and chained models is negligible, or if explicit reasoning VLAs do not outperform standard ones on MART, then the benchmark fails to measure chaining. Also, if removing distractors raises CSR for all models uniformly, then MART measures only task complexity, not knowledge chaining. Additionally, if the causal intervention shows CSR ≈ product of stepwise accuracies across various tasks, that would falsify the claim that MART captures structural chaining failures. If adversarial visual variations do not affect CSR (i.e., models still perform well without them), then the load-bearing assumption is invalid.

## References

1. Does VLA Even Know the Basics? Measuring Commonsense and World Knowledge Retention in Vision-Language-Action Models
2. Cosmos-Reason1: From Physical Common Sense To Embodied Reasoning
3. The Llama 3 Herd of Models
4. HybridFlow: A Flexible and Efficient RLHF Framework
5. Expanding Performance Boundaries of Open-Source Multimodal Models with Model, Data, and Test-Time Scaling
6. Robotic Control via Embodied Chain-of-Thought Reasoning
7. NVLM: Open Frontier-Class Multimodal LLMs
8. BridgeData V2: A Dataset for Robot Learning at Scale
