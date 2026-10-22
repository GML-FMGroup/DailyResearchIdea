# Live-Tool Causal Calibration for Synthesized Computer-Use Agent Environments

## Motivation

Synthesized CUA environments such as those produced by WildAWM are validated only by execution-based quality gates (agent rollouts for solvability, cross-judge agreement for verifier reliability), which check that tasks are solvable and judged consistently but never test whether the synthesized environment's action-outcome dynamics match the live tool they purport to model. The same structural gap recurs across SWE-Gym, CodeGym, R2E-Gym, and APIGen: each builds executable or verifiable environments from commits, static problems, or API schemas and assumes the resulting transition structure is faithful to production behavior, so sim-to-real causal fidelity is assumed rather than measured. Because no step ever replays an action sequence against the live tool and compares transitions, environments whose causal structure diverges (e.g., an emulator that accepts an argument the real API rejects, or a GUI stub that ignores a state-mutating action) pass every existing gate and silently poison training.

## Key Insight

A synthesized environment is causally faithful to a live tool exactly when the tool's observable transition function agrees with the synthesized transition function on the action sequences the agent actually visits, so replaying policy-conditioned traces and testing per-transition disagreement isolates sim-to-real divergence that solvability and judge-agreement gates are structurally blind to.

## Method

**What it is.** LiveTool-Cal is a post-synthesis calibration stage for WildAWM that takes a synthesized environment E_syn with its task set T, samples a subset T_cal, and for each task replays the agent's action trace against the live production tool (or a high-fidelity emulator) E_live, producing a per-task causal-fidelity score and a binary flag that removes tasks whose transition structure diverges.

**Load-bearing assumption (stated explicitly).** The method assumes that when an action is executed against E_live from the live tool's *actual current state* (re-synchronized to the live observation before each step), the resulting observation mismatch is attributable to divergence in the environment's transition function rather than to the two environments having already diverged into different states. This assumption fails when replay is action-sequence-grounded without re-grounding: once any earlier step diverges, or the live tool has hidden state, session state, rate limits, or non-idempotent side effects, the replayed action sequence executes in a different state than the synthesized one, and all subsequent observation mismatches are confounded. To make the assumption hold, replay is state-grounded (see step 2b) and divergence is scored only on the first mismatching step per trace (see step 2c).

**How it works.**

```pseudocode
INPUT: E_syn, live tool E_live, task set T, agent policy pi, sample rate r=0.15, divergence threshold tau=0.10, min support k=3
OUTPUT: filtered task set T_kept, per-task fidelity scores

1. T_cal <- stratified_sample(T, rate=r)   # stratify by interface type: GUI / MCP / CLI
2. FOR each task t in T_cal:
   a. Roll out pi in E_syn for n=5 traces; record action sequence a_1..a_m and synthesized observations o_syn_1..o_syn_m
   b. State-grounded replay: initialize E_live to the same initial state as E_syn for t; for i = 1..m:
        - execute a_i against E_live from E_live's *actual current state* (not from a pre-recorded state)
        - record o_live_i
        - re-synchronize E_syn to o_live_i before the next step (so E_syn's state matches E_live's actual state)
        - if o_live_i is non-deterministic (retry up to 3x; drop step if still non-deterministic), mark step i as unscoreable and continue
   c. For each scoreable step i, compute transition_divergence(o_syn_i, o_live_i):
        - success/failure bit mismatch -> 1.0
        - error-class mismatch (e.g. 4xx vs 2xx, exception type) -> 0.8
        - state-mutation mismatch (returned object/DB/UI element differs) -> 0.5
        - textual-only difference with identical state -> 0.0
      Score only the FIRST mismatching step per trace; treat all subsequent steps in that trace as unscoreable (not as independent divergence evidence).
   d. fidelity(t) <- 1 - mean_i transition_divergence_i, averaged over scoreable steps across the n traces (first-mismatch-only per trace)
   e. IF count of scoreable steps < k THEN mark t as under-supported and route to T_review instead of scoring
3. Flag t as divergent IF fidelity(t) < 1 - tau
4. T_kept <- T \ {t : divergent(t)}; T_review <- under-supported tasks
5. Emit per-interface divergence report (which action types, which error classes) for environment repair
```

**Why this design.** We chose to replay the *policy's own* action traces rather than uniformly random action sequences because divergence that matters for training is divergence on the state-action distribution the agent actually occupies, accepting the cost that calibration is policy-dependent and must be rerun when pi changes. We chose a graded divergence taxonomy (success bit / error class / state mutation / text) over a single binary match because a synthesized environment that returns a differently-worded but state-equivalent response is faithful while one that returns a success bit for a call the live tool rejects is not, accepting the cost that the taxonomy weights (1.0/0.8/0.5/0.0) are hand-set and may need per-interface tuning. We chose to route low-support tasks to human review rather than auto-drop them because a task with fewer than k executed transitions cannot be scored without conflating sparsity with divergence, accepting the cost of a small manual review queue. We chose a live production tool with a high-fidelity emulator fallback only when the live tool is rate-limited, and we log which was used, because the calibration claim is about the live tool and an emulator reintroduces the very assumption being tested. We chose state-grounded replay (re-synchronizing E_syn to the live observation before each step and scoring only the first mismatching step per trace) over action-sequence replay because action-sequence replay under drift confounds transition-function divergence with state divergence, accepting the cost that later steps in a trace are unscoreable and that re-synchronization requires E_syn to support state injection. This differs from the WildAWM execution-based quality gates, which use agent rollouts to test solvability and cross-judge agreement to test verifier reliability — neither gate ever executes an action against a live tool, so neither can observe a transition mismatch; a domain expert would not describe LiveTool-Cal as a variant of WildAWM's gates because the gates score *tasks* on solvability/judgeability while LiveTool-Cal scores *environment dynamics* on transition agreement, a quantity the gates never compute.

**Why it measures what we claim.** The computational quantity `transition_divergence(o_syn_i, o_live_i)` measures causal fidelity of the synthesized environment because the assumption that the live tool's transition function is deterministic given the action and prior state makes an observed mismatch equivalent to a divergence in the causal structure rather than sampling noise; this assumption fails when the live tool is non-deterministic (e.g., a search API returning different rankings per call), in which case the metric reflects live-tool stochasticity instead of environment infidelity, which is why non-deterministic steps are retried and dropped. The quantity `fidelity(t)` averaged over the policy's visited trace measures *training-relevant* fidelity because the assumption that the agent's state-action distribution is concentrated on the sampled traces makes trace-average divergence equivalent to expected divergence under training; this assumption fails when the policy is far from the training policy (early in training), in which case the score reflects fidelity on an off-distribution trace. The quantity `support count >= k` measures scoreability because the assumption that at least k deterministic transitions are needed for a stable mean makes low-support tasks unscorable rather than divergent; this assumption fails for tasks with intrinsically short traces, in which case the task is routed to review rather than flagged. The state-grounded replay procedure measures transition-function divergence because the assumption that re-synchronizing E_syn to the live observation before each step keeps both environments in matched states makes a first-step mismatch attributable to the transition function rather than to accumulated state drift; this assumption fails when re-synchronization is impossible (e.g., E_syn cannot be set to arbitrary live states), in which case the first mismatch may reflect state divergence rather than transition infidelity, which is why such tasks are routed to T_review.

## Contribution

(1) LiveTool-Cal, a post-synthesis calibration stage that replays policy-conditioned action traces against live tools and scores per-transition causal fidelity with a graded divergence taxonomy. (2) A design principle: sim-to-real fidelity of synthesized CUA environments is measurable as per-transition agreement on the agent's visited state-action distribution, and solvability/judge-agreement gates are structurally blind to it. (3) A per-interface divergence report that localizes which action types and error classes diverge, enabling targeted environment repair rather than blanket task removal.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | WildAWM synthesized envs across GUI/MCP/CLI interfaces | Directly targets the environments the method calibrates |
| Primary metric | Downstream agent success on live-tool tasks, split by calibration flag | Tests whether flagged divergence predicts real transfer failure |
| Baseline | WildAWM execution-based quality gates (solvability + judge agreement) | Tests sub-claim that task-level gates miss transition infidelity |
| Baseline | Random-action replay calibration | Tests sub-claim that policy-trace replay finds training-relevant divergence |
| Baseline | No calibration (raw WildAWM environments) | Establishes the floor the method must beat |
| Ablation-of-ours | Binary divergence (success-bit only), no graded taxonomy | Tests whether graded taxonomy drives the predicted gain |
| Ablation-of-ours | Action-sequence replay (no state re-synchronization, all steps scored) | Tests whether state-grounded replay is necessary for valid divergence scores |

### Why this setup validates the claim

The central claim is that synthesized-environment transition infidelity is invisible to task-level quality gates and is causally responsible for downstream agent failure on live tools. The design makes this falsifiable by pairing a metric that separates environments by their calibration flag: if LiveTool-Cal is measuring real causal infidelity, then agent success on live tools should be substantially higher on `T_kept` than on the flagged/removed tasks, and the gap should survive after controlling for task difficulty. The WildAWM-gates baseline tests the sub-claim that solvability and judge agreement are orthogonal to transition fidelity — if that baseline already predicts live-tool success as well as ours, the method adds nothing. The random-replay baseline tests the sub-claim that replaying the policy's own trace is what makes divergence training-relevant; if random replay matches us, the policy-distribution argument is wrong. The binary-divergence ablation tests whether the graded taxonomy is doing work or is decorative. The action-sequence-replay ablation tests the load-bearing assumption directly: if scoring all steps under drift yields the same flags and downstream success as state-grounded replay, then re-synchronization is unnecessary; if it yields more false flags (tasks flagged as divergent that succeed live) or misses true divergence, the state-grounded repair is validated. The metric is right because it is the outcome the method claims to protect: live-tool task success, not synthesized-environment success, which the baseline gates already optimize.

### Expected outcome and causal chain

**vs. WildAWM execution-based quality gates** — On a task where the synthesized tool returns a 200 with a plausible-looking object but the live API rejects the same call with a 4xx, the gates pass the task because the rollout succeeds and the judge agrees, yet the agent learns a transition that never occurs live. Our method replays that exact action against the live tool from the live tool's actual state, observes an error-class mismatch on the first divergent step, and flags the task. We expect a clear success gap on the flagged subset (roughly a third of flagged tasks failing live) with near-parity on the kept subset, and a gates-vs-ours gap concentrated on tasks whose synthesized success bit diverges from live.

**vs. Random-action replay calibration** — On a task where the agent's policy reaches a deep state (authenticated session, populated cart) before the divergent call, random replay almost never reaches that state, so it samples shallow transitions that agree and scores the task faithful. Our method replays the policy's own trace with state re-synchronization and hits the divergent deep transition. We expect our divergence report to flag substantially more tasks than random replay, with the extra flags concentrated on deep-trace tasks and correlating with live-tool failures that random replay's kept set still exhibits.

**vs. No calibration (raw WildAWM environments)** — On a task whose synthesized environment silently accepts a malformed mutation, the raw agent trains on a false success and fails at deployment when the live tool raises. Our method removes or repairs that task before training. We expect the raw baseline to show a broad success deficit on live tools, with our calibrated set recovering most of it, and the residual gap shrinking as the divergence threshold is tightened.

**vs. Action-sequence replay ablation** — On a task where an early step diverges (e.g., a login returns a different session token), action-sequence replay executes all subsequent actions in a drifted state, producing many spurious mismatches and flagging the task as divergent even if its later transitions are faithful. Our state-grounded replay re-synchronizes E_syn to the live observation after the first mismatch, scores only that first step, and treats later steps as unscoreable, yielding a lower false-flag rate. We expect the action-sequence ablation to flag more tasks overall, with its extra flags concentrated on tasks that actually succeed live (false positives), and to show a smaller kept-vs-flagged success gap than our method.

### What would falsify this idea

If the downstream success gap between `T_kept` and flagged tasks is uniform across interface types and divergence classes — or if the WildAWM-gates baseline predicts live-tool success as well as our fidelity score — then transition fidelity is not the causal quantity we claim, and the method is merely re-ranking tasks by difficulty. Additionally, if action-sequence replay (without state re-synchronization) produces the same flags and downstream success as state-grounded replay, then the load-bearing assumption is not load-bearing and the repair is unnecessary.

## References

1. Agent World Model in the Wild: Computer-Use Agent Environment Synthesis
2. R2E-Gym: Procedural Environments and Hybrid Verifiers for Scaling Open-Weights SWE Agents
3. Generalizable End-to-End Tool-Use RL with Synthetic CodeGym
4. Training Software Engineering Agents and Verifiers with SWE-Gym
5. APIGen: Automated Pipeline for Generating Verifiable and Diverse Function-Calling Datasets
6. FireAct: Toward Language Agent Fine-tuning
7. Self-Instruct: Aligning Language Models with Self-Generated Instructions
