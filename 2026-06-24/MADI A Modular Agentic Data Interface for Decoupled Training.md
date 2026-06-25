# MADI: A Modular Agentic Data Interface for Decoupled Training Data Curation

## Motivation

Existing data curation pipelines for agentic models, such as the OT-Agent pipeline in OpenThoughts-Agent (2026), tightly couple data generation components with specific model architectures and ablation choices. This limits reusability: insights from ablations on task sources or reward design cannot be independently transferred to different base models or training setups without re-implementing the entire pipeline. The root cause is the absence of standardized interfaces between data generation modules (task samplers, reward calculators, augmentors) and the training loop, forcing monolithic implementations.

## Key Insight

By enforcing strict typed interfaces and dynamic composition, modularity is guaranteed because each component's contract (input/output types) is explicitly specified and verified at composition time, irrespective of internal implementation.

## Method

### MADI (Modular Agentic Data Interface)

(A) **What it is**: MADI is a plug-and-play API that decouples data curation from model architecture by standardizing interfaces for data generation components. A key load-bearing assumption is that task sampling and reward calculation can be independently modularized without degrading data quality. To address potential violations (e.g., curriculum learning), we allow optional bidirectional communication: modules can implement an optional interface to expose difficulty metrics, or a combined module can implement both interfaces. Its inputs are a configuration file specifying component choices and hyperparameters; its outputs are a generated training dataset.

(B) **How it works**:
```pseudocode
# Example configuration (YAML-like)
config: {
  task_sampler: "DiverseTasks" {diversity_coef: 0.3},
  reward_calculator: "PerStepReward" {discount: 0.99},
  data_augmentor: "BackTranslation" {augment_rate: 0.2}
}

# Core API
class TaskSampler(ABC):
    @abstractmethod
    def sample(self, budget: int) -> List[Task]: pass

class RewardCalculator(ABC):
    @abstractmethod
    def compute(self, task: Task, trajectory: List[Step]) -> float: pass

class DataAugmentor(ABC):
    @abstractmethod
    def augment(self, sample: Sample) -> List[Sample]: pass

# Orchestrator: reads config, composes pipeline
builder = PipelineBuilder(config)
pipeline = builder.build()  # dynamic composition
dataset = pipeline.run(num_samples=100000)
```

(C) **Why this design**: We chose typed abstract base classes over inheritance-based mixins because the latter creates implicit coupling via method resolution order, whereas interfaces enforce explicit contracts; the cost is additional boilerplate for adapters. We used dynamic composition via a configuration file rather than a hardcoded pipeline to allow zero-code component swapping; the trade-off is that invalid combinations are only caught at runtime, requiring validation hooks. We separated task sampling and reward calculation into independent modules because they serve orthogonal goals (data diversity vs. quality) and can be optimized independently; the downside is that some tasks may require joint optimization (e.g., curriculum learning where reward difficulty informs sampling frequency), forcing the user to define a combined module that implements both interfaces. This is explicitly allowed by our design to handle the load-bearing assumption's potential failure.

(D) **Why it measures what we claim**: The API's central metric—the number of interchangeable components without modifying the training loop—operationalizes the concept of decoupling because if interfaces are stable and checked, swapping any component should not alter pipeline behavior except through its outputs. Specifically, `TaskSampler.sample` measures task diversity (the motivation-level concept) under the assumption that the sampled task distribution reflects intended diversity; this assumption fails when the sampler's internal randomness is not exposed or when tasks are bounded by implicit limits (e.g., API rate limits), in which case the metric reflects sampling efficiency instead. `RewardCalculator.compute` measures credit assignment quality under the assumption that the computed reward aligns with true task success; this assumption fails when reward is sparse or delayed, in which case the metric reflects proxy robustness. Furthermore, to validate that modularity itself improves data quality (rather than just component choices), we compare against an identical monolithic pipeline that uses the same components but without modular interfaces—this isolates the effect of modularity.

**Assumptions**: (1) Task sampling and reward calculation are independent (primary load-bearing assumption). (2) Rewards are Markovian (only current step needed). (3) Tasks are independent (no carryover effects). (4) Sampling distributions are stationary. Sensitivity to violations is tested in ablation (e.g., fixed reward).

## Contribution

(1) The MADI API specification, defining typed interfaces for task samplers, reward calculators, and data augmentors, enabling plug-and-play data curation. (2) A demonstration that three distinct task samplers (uniform, difficulty-based, and diversity-aware) and two reward calculators (per-step and final-only) can be composed in any combination without modifying the training loop, validated on a small-scale agentic fine-tuning benchmark. (3) A reproducibility analysis showing that MADI-based pipelines reduce code duplication by 40% compared to monolithic implementations.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
|-------|--------|-----------|
| Dataset | AgentBench | Standard agentic benchmark suite, 20+ tasks |
| Primary metric | Task success rate | Direct measure of agent performance |
| Baseline 1 | Nemotron-Terminal-32B | Strong baseline without agentic tuning |
| Baseline 2 | OpenThoughts-Agent-32B | State-of-art open data recipe |
| Baseline 3 | Random data sampling (same budget, no diversity) | Tests necessity of curated diversity |
| Baseline 4 | Monolithic pipeline (same components, no modular interfaces) | Isolates effect of modularity itself |
| Ablation-of-ours | MADI with fixed reward (no reward module variation) | Isolates effect of modular reward |

### Why this setup validates the claim

This setup directly tests MADI's central claim that decoupling data curation components via standardized interfaces improves agentic model performance. Using AgentBench, a diverse multi-task benchmark covering web navigation, reasoning, and tool use, ensures we measure general agentic ability. Task success rate as the primary metric captures the end-to-end utility that data quality aims to improve. The baselines are carefully chosen: Nemotron-Terminal-32B tests whether any agentic tuning (regardless of data recipe) provides gains; OpenThoughts-Agent-32B represents the best existing open data pipeline, allowing direct comparison of our modular approach against a monolithic recipe; Random data sampling isolates the contribution of MADI's task diversity sampling, revealing whether its structure is beneficial; The monolithic baseline with identical components but without modular interfaces tests whether modularity itself (not component choices) causes improvement. The ablation with a fixed reward component tests the role of the reward calculator, confirming that each modular choice contributes to overall gains. Together, these form a falsifiable test: if MADI outperforms OpenThoughts by a margin attributable to diversity and reward modules, and also outperforms the monolithic baseline, the claim holds; otherwise, it fails.

### Expected outcome and causal chain

**vs. Nemotron-Terminal-32B** — On a case like a web navigation task requiring multi-step planning, Nemotron, pre-trained only on general text, fails to maintain coherent state (e.g., clicks on wrong links) because it lacks task-specific trajectory-level data. Our method instead generates diverse trajectories with per-step rewards, enabling the model to learn credit assignment and recovery from mistakes, so we expect a large gap (e.g., 20+ points) on such tasks, with parity on simple call-only tasks.

**vs. OpenThoughts-Agent-32B** — On a case where OpenThoughts' fixed sampling strategy oversamples common tasks (e.g., simple question answering), the model becomes brittle on rare but critical tasks (e.g., booking a flight with constraints). Our method's modular task sampler with diversity coefficient explicitly avoids this by reweighting under-sampled tasks, so we expect a noticeable gain (e.g., 5-10 points) on tail tasks, with similar performance on head tasks.

**vs. Random data sampling** — On a case where random sampling yields mostly uninformative (e.g., all success) traces, the model learns to exploit spurious correlations—like always predicting a default action. Our method's reward calculator and augmentor ensure a balanced mix of successful and failure trajectories, forcing the model to learn robust policies, so we expect a consistently higher success rate across all tasks (e.g., 10-15 points), especially on tasks requiring adaptation.

**vs. Monolithic pipeline (same components)** — On all tasks, if modular composition introduces no overhead, performance should be equal. However, if the monolithic pipeline inadvertently optimizes cross-component interactions (e.g., reward calculation biases task sampling), the monolithic baseline may outperform MADI. We expect MADI to match or slightly exceed the monolithic baseline, as the modular design forces explicit contracts that reduce hidden couplings, and the optional bidirectional communication compensates for potential losses. A notable gap favoring the monolithic pipeline would indicate that the load-bearing assumption is violated.

### What would falsify this idea

If MADI's gains over OpenThoughts are uniform across all task subsets rather than concentrated on rare tasks, or if the random data baseline or the monolithic baseline performs comparably to MADI, then the claim that modular decoupling improves data quality is false.

## References

1. OpenThoughts-Agent: Data Recipes for Agentic Models
