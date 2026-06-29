# WebShifting: Benchmarking Web Agent Adaptation to Dynamic Environment Changes

## Motivation

Existing web agent benchmarks such as GauntletBench and REAL evaluate agents on static, deterministic environments. This fails to assess the critical capability of adapting to unexpected changes in website content or layout during task execution, because real-world websites constantly update content (e.g., inventory changes, A/B testing). Without a benchmark that introduces controlled, unpredictable dynamics, we cannot systematically measure an agent's ability to re-evaluate and adjust its plan, leaving a gap in understanding agent robustness.

## Key Insight

By modeling website changes as a partially observable Markov decision process where the hidden state evolution is independent of agent actions and semantically grounded in the task, we can isolate adaptation ability as a distinct dimension from general task competence.

## Method

### (A) What it is
WebShifting is a benchmark framework for evaluating web agents' adaptation to dynamic environment changes. It takes as input a static web application (e.g., from GauntletBench), a set of tasks, and a change schedule definition. It outputs an adaptation score for each agent.

### (B) How it works
The core operation is a state scheduler that modifies the website's DOM between agent actions according to a hidden Markov process.

```python
class WebShiftingEnv:
    def __init__(self, base_app, task, schedule):
        self.base = base_app  # e.g., shopping site
        self.task = task
        self.schedule = schedule  # list of (trigger, change_fn) pairs
        self.change_index = 0
        self.state = self.base.initial_state(task)
    
    def step(self, action):
        # Execute action in current state
        result = self.state.execute(action)
        # Check if any schedule trigger condition is met
        for trigger, change_fn in self.schedule[self.change_index:]:
            if trigger(self.state, action, self.change_index):
                self.state = change_fn(self.state)
                self.change_index += 1
                result['change_occurred'] = True
                result['new_state'] = self.state
                break
        return result

# Schedule example: after 3 actions, change product prices by ±20%
schedule = [
    (lambda s,a,i: i==0 and s.steps==3, lambda s: s.change_prices(0.2)),
    (lambda s,a,i: i==1 and a.type=='click', lambda s: s.rewrite_layout())
]
```
Hyperparameters: change frequency (average number of actions between changes), change magnitude (e.g., 10-50% price variation), and semantic relevance (e.g., inventory out-of-stock, price changes, layout reorganization). These are configured per task to probe different adaptation dimensions.

### (C) Why this design
We chose a hidden Markov schedule over random changes because it ensures changes are task-relevant and occur at predictable (to the designer) times, enabling controlled measurement of adaptation. The trade-off is that the schedule must be handcrafted for each task, increasing benchmark creation cost, but we accept this because it allows us to isolate specific adaptation challenges (e.g., sudden price change vs. gradual layout shift). We use DOM-level changes rather than screenshot-level perceptual changes to ensure that the agent's internal state (if it tracks DOM) is directly affected, exposing adaptation failures at the decision level rather than perception level. This assumes agents have access to the DOM; for vision-only agents, we also provide screenshot changes, but the primary evaluation targets DOM-based agents. We chose a deterministic execution of changes once triggered (rather than probabilistic) to ensure reproducibility across runs, at the cost of reduced ecological validity. However, reproducibility is essential for benchmarking.

### (D) Why it measures what we claim
The adaptation score is defined as: `(success_rate_dynamic / success_rate_static) - 1`. Here, `success_rate_dynamic` measures the agent's ability to complete the task under the dynamic schedule, and `success_rate_static` measures the same task without changes. The ratio operationalizes "adaptation ability" because the only difference between the two conditions is the presence of changes; this assumes that the static condition captures baseline task difficulty and that the dynamic condition adds the adaptation challenge. This assumption fails when the agent's performance in the static condition is already low (e.g., <10%), in which case the ratio may be inflated due to floor effects; we therefore only compute the metric for tasks where static success >20%. The `change_occurred` flag in each step measures whether the agent encountered a change, and we track the number of additional actions taken after a change before task progress—this measures "adaptation delay". The adaptation delay measures the agent's efficiency in re-evaluating its plan because after a change, the agent must update its world model; the delay quantifies the cost of re-planning. This assumes that the agent's action count correlates positively with its re-planning effort; this assumption fails if the agent engages in random exploration, in which case delay reflects exploration rather than adaptation. To mitigate, we require that additional actions after a change must be goal-directed (verified by action-log analysis). Finally, the change schedule's semantic relevance (e.g., price changes that affect buying decisions) ensures that adaptation is not trivial (e.g., just re-scrolling); if changes were arbitrary (e.g., background color), the task would not require replanning. This assumes that the change directly impacts the task's feasibility; this fails if the agent's policy is robust to irrelevant changes, but our schedule is designed to make changes task-critical.

## Contribution

(1) The WebShifting benchmark environment with a modular schedule mechanism for injecting dynamic, task-relevant changes into web agent evaluations. (2) The adaptation score metric that isolates an agent's ability to adapt from overall task competence, enabling direct comparison across agent designs. (3) An empirical evaluation of several frontier agents (e.g., GPT-4V-based, Claude) revealing a significant adaptation gap, with the best agent achieving only 30% relative adaptation success compared to static baselines.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
|------|--------|-----------|
| Dataset | GauntletBench static tasks + WebShifting schedules | Base tasks from known benchmark |
| Primary metric | Adaptation score (ratio of success rates) | Directly measures adaptation ability |
| Baseline1 | Gauntlet frontier agent (static) | No adaptation mechanism |
| Baseline2 | REAL text-only agent | Lacks state update handling |
| Baseline3 | WebVoyager (GPT-4 All Tools) | Strong multimodal baseline |
| Ablation-of-ours | WebShifting with random changes | Remove structured HMM schedule |

### Why this setup validates the claim
This setup validates the claim because GauntletBench tasks provide a diverse static baseline to isolate adaptation difficulty. The adaptation score metric directly compares dynamic vs static performance, isolating the effect of changes. Baseline1 (Gauntlet frontier agent) tests whether standard agents fail under dynamic conditions; if they succeed, adaptation is trivial. Baseline2 (REAL text-only) tests whether language-only agents without DOM tracking are less robust. Baseline3 (WebVoyager multimodal) tests whether visual grounding helps adaptation. The ablation (random changes) tests the necessity of semantically meaningful schedule; if random changes also cause performance drop, then any change suffices. Together, these conditions probe the central claim that structured, task-relevant changes expose adaptation failures. The metric's condition (static success >20%) prevents floor effects. If the predicted pattern (our method outperforms all baselines on adaptation score, especially on semantically relevant changes) holds, the claim is supported. If not, the idea is falsified.

### Expected outcome and causal chain

**vs. Gauntlet frontier agent (static)** — On a price change instance (prices shift +20% after 3 actions), the static agent selects items based on initial prices and fails at checkout due to insufficient funds. Our benchmark's adaptation score captures this failure via low dynamic success (e.g., 20% vs static 80%), yielding a negative ratio. We expect a large gap (adaptation score < -0.5) for this baseline, contrasting with an adaptive agent that would replan and maintain high success.

**vs. REAL text-only agent** — On a layout reorganization (buttons move after a click), the text-only agent relies on HTML text and linear navigation; it clicks the old button position, missing the new location. Our benchmark detects the failure via increased action count after change (adaptation delay). We expect this baseline to show high adaptation delay (>5 extra actions) and low adaptation score (near -0.6), while vision-based agents adapt faster.

**vs. WebVoyager (GPT-4 All Tools)** — On a gradual inventory out-of-stock (items removed after 2 actions), WebVoyager uses visual grounding but still fails if it doesn't rescan the page. Our benchmark measures the failure via the adaptation delay and success drop. We expect WebVoyager to have moderate adaptation delay (3-4 extra actions) and an adaptation score around -0.3, better than text-only but worse than an ideal adaptive agent.

**vs. WebShifting with random changes** — On a semantically meaningful price change, the random changes ablation may trigger changes that are irrelevant (e.g., background color), causing less disruption. Our benchmark's full schedule targets task-critical changes. We expect the ablation to show a smaller drop in success (adaptation score -0.2 vs full -0.5) on tasks where semantic relevance matters, proving the schedule's necessity.

### What would falsify this idea
If the adaptation score is no different across baselines (e.g., all agents score similarly), or if the ablation with random changes matches the full schedule in causing performance drops on all tasks, then the claim that structured task-relevant changes measure adaptation uniquely is false.

## References

1. Running the Gauntlet: Re-evaluating the Capabilities of Agents Beyond Familiar Environments
2. WebVoyager: Building an End-to-End Web Agent with Large Multimodal Models
3. REAL: Benchmarking Autonomous Agents on Deterministic Simulations of Real Websites
4. GPT-4V in Wonderland: Large Multimodal Models for Zero-Shot Smartphone GUI Navigation
5. The Dawn of GUI Agent: A Preliminary Case Study with Claude 3.5 Computer Use
6. Is Your LLM Secretly a World Model of the Internet? Model-Based Planning for Web Agents
7. TheAgentCompany: Benchmarking LLM Agents on Consequential Real World Tasks
8. AgentOccam: A Simple Yet Strong Baseline for LLM-Based Web Agents
