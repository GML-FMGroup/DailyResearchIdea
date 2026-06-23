# Grammar-Guided Evolutionary Bootstrapping for Multi-Agent Self-Improvement

## Motivation

Existing self-improving multi-agent systems, such as SiriuS, require an initial set of successful reasoning trajectories to bootstrap an experience library, creating a cold-start problem. This dependence on pre-existing successes is structurally limiting because success itself is what the system aims to discover; in open-ended tasks, such seeds may be unavailable or costly to obtain. The root cause is that prior methods lack a mechanism to generate plausible candidate trajectories without prior examples.

## Key Insight

A probabilistic context-free grammar induced from unlabeled LLM completions of multi-agent dialogues captures latent structural patterns that make candidate trajectories both diverse and likely to be successful, enabling evolutionary search to bootstrap without any initial successes.

## Method

### (A) What it is
**Grammar-Augmented Evolutionary Bootstrapping (GAEB)** is a method that induces a Probabilistic Context-Free Grammar (PCFG) from unlabeled multi-agent dialogues, then uses evolution guided by that grammar to generate candidate reasoning trajectories, selecting those with high self-consistency across agents to build an experience library without requiring any pre-existing successful examples.
**Load-bearing assumption:** A PCFG induced from unlabeled multi-agent dialogues captures latent structural patterns that are predictive of successful reasoning trajectories. This assumption is tested in ablations and failure-mode analysis (Section D).
**Inputs:** A set of multiple LLM agents with generic task-agnostic prompts; a corpus of unlabeled multi-agent dialogues (e.g., task-relevant interactions, such as dialogues on similar problem types, rather than random conversations to avoid spurious productions).
**Outputs:** A library of successful reasoning trajectories ready for downstream fine-tuning.

### (B) How it works
```pseudocode
1. INDUCE GRAMMAR:
   - Collect N unlabeled multi-agent dialogues (e.g., 500 conversation transcripts from task-relevant interactions).
   - Parse each dialogue into a sequence of turn labels (e.g., "Agent A proposes", "Agent B critiques", "Agent A revises").
   - Apply a Bayesian grammar induction algorithm (e.g., the one from the Darwin Godel Machine's seed generation) to infer a PCFG over turn-label sequences. Hyperparameter: maximum number of non-terminals = 20.
2. INITIALIZE POPULATION:
   - Generate 100 candidate trajectories by sampling from the PCFG (probabilistic derivations). Each trajectory is a sequence of turn types.
3. EVALUATE FITNESS:
   - For each candidate trajectory, instruct the LLM agents to follow the turn sequence exactly in a reasoning task (e.g., math problem).
   - Record the final answers from each agent. Compute self-consistency score = pairwise agreement fraction among agents (e.g., if 3 agents agree out of 5, score = 0.6).
4. EVOLVE POPULATION:
   - Repeat for G generations (e.g., G=20):
     a. SELECTION: Tournament selection (size k=3) based on fitness.
     b. MUTATION: Replace a random non-terminal symbol in the trajectory with an alternative production from the grammar (probability 0.3).
     c. CROSSOVER: Swap subsequences between two parent trajectories (probability 0.5).
     d. EVALUATE new offspring via self-consistency.
     e. Replace lowest-fitness individuals with new offspring (elite count=10).
5. BUILD EXPERIENCE LIBRARY:
   - Take the top 10% of trajectories from the final population.
   - Convert each turn-type sequence into actual dialogue text by running the agents through that protocol and recording successful interactions (where agents reach a consensus). Store these successful dialogues in the library.
```
**Computational budget:** Approximate total LLM calls = (population size + offspring per generation) * generations * (number of agents). With population=100, offspring per generation=20 (after selection/crossover/mutation), generations=20, agents=5: ~120*20*5 = 12,000 calls. Using a 7B-parameter LLM, this requires ~1 day on 4 GPUs (A100).

### (C) Why this design
We chose PCFG induction over free-form generation (e.g., random or GPT-sampled sequences) because the grammar captures latent structural regularities—such as typical turn-taking patterns—that make trajectories plausible and effective for coordination; the trade-off is that grammar induction may miss rare but valuable structures not present in the unlabeled data. We selected self-consistency as the fitness signal over external human ratings or task-specific metrics because it is supervision-free and leverages the diversity of agents to signal correctness; the cost is that it assumes agent disagreement is likely for errors—an assumption that fails when all agents share the same base LLM, leading to correlated mistakes. We opted for evolutionary search with grammar-guided mutation/crossover rather than reinforcement learning because the action space (trajectory sequences) is high-dimensional and discrete, and evolution naturally explores diverse candidates; this comes at the computational cost of many LLM calls during evaluation. We chose tournament selection over rank-based to maintain steady selection pressure without requiring sorting of the entire population, accepting a slight increase in stochasticity.

### (D) Why it measures what we claim
- The induced PCFG's production probabilities measure "latent structural plausibility" because we assume that frequent turn patterns in unlabeled dialogues correspond to efficient communication protocols; this assumption fails when the unlabeled corpus contains only chaotic or unrelated conversations, in which case the grammar encodes noise rather than structure.
- Self-consistency across agents measures "agreement as a proxy for correctness" under the assumption that multiple agents are less likely to independently converge on the same incorrect answer by chance; this assumption fails when agents inherit similar biases from the underlying LLM (e.g., all produce the same hallucination), causing high consistency for wrong outputs.
- The evolutionary selection pressure (tournament size, mutation rate) measures the trade-off between exploration and exploitation because they control the balance of generating novel trajectories versus refining high-fitness ones; this assumption fails in very rugged fitness landscapes, where greedy selection may prematurely converge to a local optimum of consistent-but-erroneous trajectories.
- **PCFG probability vs. success likelihood:** PCFG probability measures frequency of turn sequences in the unlabeled corpus, not directly their success likelihood for reasoning tasks. The assumption is that frequent patterns in task-relevant dialogues are likely to be successful for reasoning. Failure mode: if the corpus contains only chaotic or unrelated dialogues, the grammar encodes noise and PCFG probability does not correlate with success. To mitigate, we use task-relevant dialogues and verify in ablation.

## Contribution

(1) A novel method for bootstrapping multi-agent self-improvement without any pre-existing successful examples, combining grammar induction from unlabeled dialogues with grammar-guided evolutionary search. (2) The identification of self-consistency across agents as a viable fitness signal for trajectory evolution, enabling supervision-free candidate evaluation. (3) A demonstration that a PCFG induced from generic dialogues can structure the search space of multi-agent reasoning trajectories, making evolution efficient.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset (primary) | Multi-step reasoning tasks (e.g., GSM8K) | Tests self-consistency on objective answers |
| Dataset (secondary) | Open-ended reasoning (e.g., MuSiQue) | Tests generalizability without objective answers |
| Primary metric | Self-consistency accuracy | Measures agreement proxy for correctness |
| Baseline 1 | SiriuS | Self-improvement without grammar induction |
| Baseline 2 | Darwin Godel Machine | Evolution with grammar but from scratch |
| Baseline 3 | Random trajectory generation | No grammar or evolutionary selection |
| Baseline 4 | GAEB with generic task-agnostic grammar (e.g., from human dialogues) | Isolates benefit of domain-specific induction |
| Ablation | GAEB without grammar induction | Isolates effect of grammar guidance |

### Why this setup validates the claim

This combination of dataset, baselines, and metric forms a falsifiable test of GAEB's central claim: that inducing a grammar from unlabeled dialogues and evolving trajectories under that grammar yields high-quality reasoning trajectories without external supervision. The multi-step reasoning dataset (e.g., GSM8K) provides objective answers, enabling self-consistency as a reliable proxy for correctness. The open-ended dataset (MuSiQue) tests whether the method generalizes when no single correct answer exists. SiriuS tests whether bootstrapped reasoning alone (without grammar) can achieve similar gains; if GAEB outperforms, it shows grammar adds value. The Darwin Godel Machine tests whether evolving from scratch (without a pre-induced grammar) is less efficient; GAEB should converge faster. Random trajectory generation serves as a lower bound, confirming that guided evolution is necessary. The generic grammar baseline controls for the source of the grammar; if GAEB with domain-specific grammar outperforms, it shows that task-relevant induction matters. The ablation (no grammar) isolates the grammar's contribution: if it performs similarly to full GAEB, the grammar is redundant; if worse, it confirms the grammar's role. The metric self-consistency accuracy directly captures the intended effect: higher agreement among agents on correct answers, which should translate to better performance when trajectories are used for fine-tuning.

### Expected outcome and causal chain

**vs. SiriuS** — On a case where agents initially disagree on a reasoning step (e.g., misinterpreting a math operation), SiriuS bootstraps by iterating on the same trajectory, potentially amplifying the error because it lacks structural diversity. Our method instead samples diverse trajectory structures from the grammar, allowing different turn patterns that may expose alternative reasoning paths, so we expect a noticeable gap on hard problems where initial disagreement is high but parity on simple ones.

**vs. Darwin Godel Machine** — On a case where the optimal turn sequence is rare (e.g., requires sequences with many back-and-forth critiques), the Darwin Godel Machine starts from random sequences and must discover that structure through evolution, requiring many generations. Our method seeds the population with grammar-sampled sequences that already mimic plausible dialogue patterns, so we expect GAEB to reach high self-consistency in fewer generations, observable as a faster convergence curve.

**vs. Random trajectory generation** — On any reasoning task, random trajectories produce chaotic interactions where agents fail to coordinate, leading to low self-consistency across the board. Our method's grammar-guided mutation and crossover maintain structural plausibility, so we expect GAEB to consistently achieve self-consistency scores above 0.6 while random generation hovers near 0.2.

**vs. GAEB with generic grammar** — On a task where domain-specific turn patterns matter (e.g., math problem involving step-by-step verification), the generic grammar may propose inefficient sequences. Our domain-specific grammar should yield higher self-consistency, especially on complex tasks; on simple tasks both may perform similarly.

### What would falsify this idea

If GAEB does not outperform all baselines on self-consistency accuracy, especially on hard problems where agent disagreement is high, then the central claim that grammar induction captures latent structural regularities is false. More specifically, if the ablation (no grammar) matches full GAEB, then grammar is unnecessary.

## References

1. Darwin Godel Machine: Open-Ended Evolution of Self-Improving Agents
2. Constitutional AI: Harmlessness from AI Feedback
3. LLM-POET: Evolving Complex Environments using Large Language Models
4. Evolution Gym: A Large-Scale Benchmark for Evolving Soft Robots
5. Quality-Diversity through AI Feedback
6. MarioGPT: Open-Ended Text2Level Generation through Large Language Models
7. Procedural content generation using neuroevolution and novelty search for diverse video game levels
8. SiriuS: Self-improving Multi-agent Systems via Bootstrapped Reasoning
