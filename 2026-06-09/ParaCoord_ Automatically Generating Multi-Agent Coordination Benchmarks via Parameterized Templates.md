# ParaCoord: Automatically Generating Multi-Agent Coordination Benchmarks via Parameterized Templates and LLM-in-the-Loop

## Motivation

Existing multi-agent coordination benchmarks are manually designed, limiting coverage of the coordination parameter space. BeTaL showed that parameterizing design choices and using an LLM to explore the space can generate benchmarks with target properties in tool-agent settings, but this methodology has not been adapted to multi-agent coordination where emergent behaviors depend on structural parameters like communication topology and task interdependence. Without automated generation, it is costly to systematically test coordination capabilities across varied conditions.

## Key Insight

The emergent coordination behavior in multi-agent systems is governed by a small set of structural parameters (communication graph, role constraints, task dependencies) that can be systematically varied to generate benchmarks that isolate specific coordination mechanisms, because these parameters directly control information flow and task decomposition that drive coordination.

## Method

### (A) What it is
ParaCoord is a framework that takes a parameterized template for multi-agent coordination scenarios and uses an LLM-in-the-loop to search over the parameter space, generating concrete benchmark instances that target specific coordination behaviors (e.g., role specialization, information diffusion). Input: a base template with placeholders for coordination variables; output: a set of benchmark instances with associated behavioral labels.

### (B) How it works (pseudocode)
```pseudocode
1. Define template T = (C, R, O) where:
   - C: communication topology (e.g., fully connected, ring, star, random with constraints)
   - R: role constraints (e.g., fixed roles, flexible roles, role swapping allowed)
   - O: task interdependence (e.g., independent, sequential, parallel with dependencies)
2. Initialize parameter space P = {C_i, R_j, O_k} with ranges and constraints.
3. For each desired coordination behavior B (e.g., role specialization):
   a. Prompt LLM (temperature=0.7): "Generate a combination of parameters (C, R, O) that would elicit behavior B in a team of 3-5 agents. The agents must use a fixed rule-based policy (no learning)."
   b. LLM returns a candidate parameter vector p.
   c. Instantiate template T with p to get a concrete scenario S.
   d. Simulate S for 100 steps with a baseline agent (fixed rule-based policy: each agent follows hand-coded rules for its role, if defined, else random action).
   e. Measure behavior B via a specific metric:
      - For role specialization: entropy of role assignments across agents = -Σ_i (p_i log p_i) normalized by log(num_agents), where p_i is proportion of agents assigned to role i based on action patterns. Lower entropy indicates specialization.
      - For information diffusion: average pairwise mutual information of agent beliefs (belief states are discrete vectors of size 10), computed across all agent pairs over simulation steps.
      - For coordination necessity: ratio of joint actions (actions where agents must act simultaneously to achieve a task) to total actions.
   f. If measured behavior matches target (within threshold 0.8 normalized for each metric), proceed to verification step; else return feedback to LLM: "The scenario did not produce enough role specialization. The communication topology may be too dense, consider a sparser graph."
   g. Verification: Run ablation where only the target parameter (e.g., R for role specialization) is changed to its opposite (e.g., fixed vs. flexible) while holding C and O fixed, and confirm behavior metric changes by at least 0.3. If verification passes, add S to benchmark set; else reject S and return feedback to LLM.
4. Repeat until at least 10 instances per behavior are generated, ensuring diversity (pairwise JS divergence between parameter vectors > 0.3).
```
Hyperparameters: temperature=0.7; simulation length=100 steps; behavior threshold=0.8 (normalized metric); diversity threshold=0.3 (Jensen-Shannon divergence on (C,R,O) using one-hot encoding of categorical values); max iterations per behavior=10; LLM: GPT-4o-mini; cost estimate: ~1000 LLM calls and ~10K simulation runs per behavior, costing ~$2 per behavior.

**Load-bearing assumption**: The three structural parameters (C, R, O) are sufficient to control emergent coordination behaviors in multi-agent systems, such that varying them can generate benchmarks that isolate specific coordination mechanisms. We verify this by running a control ablation (step 3g) for each accepted scenario.

### (C) Why this design
We chose an LLM-in-the-loop over random search because the parameter space is high-dimensional and the mapping from parameters to behavior is nuanced; the LLM can leverage common sense about coordination (e.g., that sparse communication encourages specialization) to propose plausible candidates. However, this introduces cost and potential hallucination. We mitigate cost by using a small, cheap LLM (e.g., GPT-4o-mini) and limiting iterations per behavior to 10. We chose to include simulation-based validation rather than relying solely on LLM reasoning because the LLM's predictions may not always hold; simulation provides ground-truth feedback. The trade-off is that simulation is expensive for large scenarios, so we restrict to small teams (3-5 agents). We chose to define the template abstractly over communication, roles, and tasks rather than more fine-grained parameters because these three are the primary determinants of coordination as evidenced by prior multi-agent literature; adding more parameters would increase search difficulty without clear benefit. We make the load-bearing assumption that these three parameters are sufficient, and we verify this through an ablation control for each generated scenario (step 3g).

### (D) Why it measures what we claim
The parameter search explicitly targets coordination behaviors by generating scenarios with specific parameter combinations that are hypothesized to elicit those behaviors. The computational quantity `entropy of role assignments` measures `role specialization` because lower entropy indicates that agents perform distinct roles; this assumes that roles are observable from action sequences. This assumption fails when agents use hidden strategies or when tasks are symmetric, in which case low entropy may instead reflect task homogeneity. The communication topology parameter directly influences information diffusion: sparser graphs reduce information flow, which `information diffusion` is measured by the average pairwise mutual information of agent beliefs; this assumes that belief changes are solely due to communication, ignoring shared observations. The task interdependence parameter controls how much agents must coordinate: sequential dependencies force synchronization, which `coordination necessity` is measured by the ratio of joint actions to total actions; this assumes that the baseline policy does not pre-emptively coordinate outside task needs. Each parameter is chosen to isolate a specific coordination mechanism, and the simulation validation ensures that the generated scenario actually produces the intended behavior within the assumed scope. Additionally, to ensure that the generated instances span diverse coordination challenges, we compute the pairwise Jensen-Shannon divergence between parameter vectors (one-hot encoded) of all instances within a behavior class, and require a minimum average divergence of 0.3. This diversity metric directly addresses the risk that high hit rate from a narrow region would not cover the coordination parameter space.

## Contribution

(1) ParaCoord, a framework for automatically generating multi-agent coordination benchmarks by parameterizing coordination variables and using an LLM-in-the-loop to search the parameter space, validated with a proof-of-concept on a simulated coordination environment. (2) A coordination-specific template language that captures communication topology, role constraints, and task interdependence, enabling targeted generation of scenarios that elicit role specialization and information diffusion. (3) An analysis of the mapping between coordination parameters and emergent behaviors, showing that the generated benchmarks expose failure modes in standard multi-agent RL algorithms.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|-----------------------|
| Dataset | Custom coordination templates (warehouse, search&rescue) | Covers diverse coordination types |
| Primary metric | Hit rate per behavior class | Measures target behavior elicitation |
| Coverage metric | Average pairwise JS divergence between parameter vectors | Ensures diverse parameter space exploration |
| Baseline 1 | Random search | Tests LLM efficiency over blind search |
| Baseline 2 | Manual design (human expert) | Compares to human curation effort |
| Baseline 3 | LLM-only (no simulation validation) | Isolates simulation feedback importance |
| Ablation | ParaCoord w/o simulation feedback | Identifies validation necessity |

### Why this setup validates the claim

This combination of dataset, baselines, metric, and ablation forms a falsifiable test of the central claim that ParaCoord reliably generates benchmark instances eliciting targeted coordination behaviors. The dataset covers two distinct coordination domains, ensuring generality. Comparing to random search tests whether LLM guidance improves efficiency over blind exploration. Comparing to manual design tests whether automated generation matches or surpasses human expertise, which is the practical alternative. The LLM-only baseline isolates the role of simulation validation, and the ablation removes it to directly measure its contribution. The hit rate metric directly quantifies whether generated instances actually produce the intended behavior, making the evaluation outcome interpretable as a success or failure of the core idea. The coverage metric ensures that high hit rate is not achieved by many similar instances from a narrow region; instead, the generated parameter vectors must be diverse, connecting hit rate to benchmark coverage of the coordination parameter space.

### Expected outcome and causal chain

**vs. Random search** — On a case where the target behavior is role specialization with 5 agents, random search may sample a fully connected communication graph, causing all agents to share information uniformly, thus failing to produce distinct roles because agents have no incentive to specialize. Our method instead uses the LLM to propose a sparser topology (e.g., ring) that forces information bottlenecks, and simulation confirms reduced entropy in role assignments. We expect a noticeable gap in hit rate: around 80% for ours vs. 20% for random search across behaviors, with larger differences on complex behaviors like sequential interdependence. Additionally, our method's coverage metric (average JS divergence) is expected to be higher than random search (e.g., 0.5 vs. 0.3) because the LLM avoids redundant parameter combinations.

**vs. Manual design (human expert)** — On a nuanced behavior like information diffusion under partial observability, manual designers may inadvertently create scenarios where agents share redundant observations or have too many communication channels, leading to high diffusion even when sparser conditions are desired, because human intuition struggles with the parameter space. Our method systematically iterates with LLM proposals and simulation feedback, tuning parameters until the desired diffusion level is achieved. We expect our method to achieve higher hit rate (e.g., 85% vs. 60%) and to require less human effort, though manual design may excel on simple, well-understood behaviors. We also expect our method to produce higher diversity (coverage metric) because the LLM explores a wider range of parameter combinations than a human expert might.

**vs. LLM-only (no simulation validation)** — On a scenario where the LLM proposes a star communication topology to promote role specialization, but simulation reveals that the central agent becomes a bottleneck and all agents converge to similar roles, the LLM-only method would accept this flawed instance, while ParaCoord detects the mismatch and rejects it, continuing search. Thus, we expect our method to have significantly fewer false positives (instances that do not elicit the target behavior) — e.g., a false positive rate <10% vs. >30% for LLM-only — while maintaining similar coverage of valid instances. The coverage metric may be similar between the two, but the hit rate advantage is decisive.

### What would falsify this idea

If ParaCoord's hit rate does not significantly exceed random search across multiple behavior types, or if the ablation without simulation validation performs equivalently to the full method, then the claims that LLM guidance and simulation feedback are essential would be falsified. Additionally, if manual design consistently matches or outperforms our method on all behaviors, the need for automated generation is unsubstantiated. If the coverage metric is low (e.g., average JS divergence <0.3) despite high hit rate, then the method fails to achieve diverse coverage, falsifying the claim that it covers the coordination parameter space.

## References

1. Automating Benchmark Design
2. τ-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains
3. LLM-POET: Evolving Complex Environments using Large Language Models
4. APIGen: Automated Pipeline for Generating Verifiable and Diverse Function-Calling Datasets
5. Automatic Generation of Benchmarks and Reliable LLM Judgment for Code Tasks
6. MarioGPT: Open-Ended Text2Level Generation through Large Language Models
7. Evolution Gym: A Large-Scale Benchmark for Evolving Soft Robots
8. AgentBench: Evaluating LLMs as Agents
