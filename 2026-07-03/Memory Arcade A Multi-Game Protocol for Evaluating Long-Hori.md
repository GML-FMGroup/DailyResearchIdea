# Memory Arcade: A Multi-Game Protocol for Evaluating Long-Horizon Memory Strategies in LLM Agents

## Motivation

Existing memory evaluation benchmarks like AgenticSTS are confined to single games, allowing agents to exploit game-specific patterns for memorization. DSGBench demonstrated that agents overfit to single-game dynamics in decision-making, and the same structural property applies to memory: single-game memory tests cannot distinguish whether an agent uses generalizable memory strategies or rote memorization. This lack of cross-game validation is a critical gap because it prevents systematic assessment of long-horizon memory strategies and their transfer across environments.

## Key Insight

Because the cognitive structure of memory sub-tasks (e.g., distance-weighted recall, temporally separated integration) is invariant across game surface features, performance differences across games directly measure the agent's ability to generalize memory strategies rather than exploit game-specific patterns.

## Method

**Memory Arcade** is a protocol that defines game-agnostic memory sub-task templates and instantiates them in multiple strategic games via calibrated, bounded-context prompts. The protocol outputs per-agent cross-game accuracy and variance scores that quantify general memory capability and overfitting to game-specific dynamics.

**Pseudocode**:
```pseudocode
Input: Set of strategic games G with logged play sessions; LLM agent; context window W=100.
Output: Cross-game memory scores per sub-task, overfitting score, game diversity metric.

1. Define memory sub-task templates T = {
   "Recall-Old": prompt to recall a specific fact from > L steps ago (L=50),
   "Integrate-Separated": prompt requiring combination of two facts at times t1, t2 with |t1-t2| > D (D=30),
   "Temporal-Ordering": prompt asking the order of two events at times t1 < t2 within window W=100
}

2. Calibration Phase (statistical control):
   a. For each sub-task t and each game g, generate candidate instances from logs.
   b. For each instance i, compute a difficulty score d_i using a small probe LLM (e.g., GPT-2 medium-355M) that predicts the correct answer given the same bounded context; difficulty = 1 - probe accuracy (averaged over 5 runs).
   c. Select instances such that the distribution of d_i is matched across games via stratified sampling (bins of width 0.1).
   d. Optionally, apply a residual correction: after running the target agent, compute per-instance accuracy residuals after regressing out d_i using linear regression, then compute variance on residuals.

3. For each sub-task t, for each game g, sample N=200 instances from the calibrated pool.

4. For each instance, construct a prompt: "You are playing <game_name>. Given the following context (last W=100 steps of history), answer: <query>". Context is truncated to W steps.

5. Run agent on all instances and record accuracy (binary correct/incorrect).

6. Compute per-game average accuracy Acc_{t,g} for each sub-task.

7. Compute cross-game variance V_t = Var_g(Acc_{t,g}) for each sub-task. High V_t indicates game-specific memory strategy.

8. Compute pairwise Jensen-Shannon divergence (JSD) between state-action distributions of the games (from logs). Report average JSD as a diversity metric. A JSD > 0.1 indicates sufficient diversity.

9. Compute overall accuracy per sub-task as measure of general memory capability.

10. (Validation) Run a reference agent (random baseline that guesses uniformly) on the same instances; compute across-game std of accuracy per sub-task; verify std < 0.05, confirming calibration.
```

**Why this design**: We made three key design decisions. First, we use a bounded context window (W=100) rather than full history, following AgenticSTS, to isolate memory capability from context length. The trade-off is that some very long-range dependencies may be truncated, but we set W=100 to cover typical memory demands in strategic games, accepting that tasks requiring longer windows are not tested. Second, we calibrate difficulty across games by using a statistical control: we compute per-instance difficulty via a probe LLM (GPT-2 medium) and regress out difficulty from accuracy before computing variance. This ensures that remaining accuracy differences reflect memory strategy generalization, not residual difficulty. The cost is additional computation but it avoids the incompleteness of surface proxies. Third, we measure cross-game variance as the primary metric rather than simple accuracy. This directly targets the overfitting problem identified in DSGBench. The risk is that if games are too similar, variance may be low even for overfit agents; we mitigate by selecting games with diverse surface features (Slay the Spire, Chess, Poker) and by reporting a diversity metric (JSD). The load-bearing assumption is that after statistical control for difficulty, any remaining accuracy variance across games is solely due to memory strategy generalization. To verify this, we run a random baseline on calibrated instances and check that accuracy variance is low (<0.05).

**Why it measures what we claim**: The bounded context window W=100 operationalizes 'bounded memory': it forces the agent to rely on its memory rather than unbounded context, so accuracy differences reflect memory strategy quality. The assumption is W is large enough to include the relevant information; if not, the metric reflects truncation rather than memory failure. The calibrated difficulty proxies (probe LLM difficulty scores) operationalize 'structural similarity across games': matching them ensures that any accuracy difference across games is due to game-specific adaptation rather than task difficulty. The assumption is that the probe LLM captures all relevant difficulty factors; if not, residual difficulty differences may masquerade as overfitting. Cross-game variance V_t operationalizes 'generalizability of memory strategies': high variance indicates the agent's strategy works in some games but not others, i.e., overfitting. The assumption is that games are sufficiently diverse, which we verify via JSD > 0.1. Each of these assumptions has a named failure mode, ensuring the metric is interpretable.

## Contribution

(1) A protocol for extracting, calibrating, and evaluating memory sub-tasks across multiple strategic games, enabling standardized cross-game memory assessment. (2) A novel metric—cross-game accuracy variance—that quantifies overfitting in memory strategies, directly addressing the lack of generalizability validation in current testbeds. (3) An open-source benchmark suite implementing the protocol on 6 diverse strategic games (e.g., Slay the Spire, Chess, Poker) with automated prompt generation and evaluation scripts.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
|------|--------|-----------|
| Dataset | Slay the Spire, Chess, Poker | Diverse games to generalize memory |
| Primary metric | Cross-game variance V_t per sub-task | Direct measure of overfitting vs generalization |
| Baseline 1 | No-memory agent (AgenticSTS no-store) | Isolates baseline without memory |
| Baseline 2 | Full-context agent (AgenticSTS full-history) | Shows ceiling with unlimited context |
| Baseline 3 | Standard LLM agent (GPT-4, no protocol) | Represents default approach without calibration |
| Ablation-of-ours | Memory Arcade without calibration | Tests importance of difficulty matching |

### Why this setup validates the claim

The combination of diverse games (Slay the Spire, Chess, Poker) and calibrated instances ensures that any accuracy differences reflect memory strategy generalization rather than task difficulty. The primary metric, cross-game variance V_t, directly quantifies overfitting: high variance indicates game-specific adaptation, low variance suggests generalizable memory. Baselines cover the spectrum from no memory (no-store) to unbounded context (full-history) and the default LLM approach (uncalibrated). The ablation removes calibration, isolating its role in reducing spurious difficulty effects. Together, this design forms a falsifiable test: if our method reduces variance without sacrificing overall accuracy, the central claim—that bounded-context calibrated evaluation reveals general memory capability—is supported. If variance remains high or accuracy drops drastically, the claim fails.

### Expected outcome and causal chain

**vs. No-memory agent** — On a recall-old task requiring a fact from 50 steps ago, the no-memory agent has no access to that fact and answers randomly (low accuracy). Our method, using bounded context (W=100), includes the fact in the prompt, so it can rely on the agent's memory to extract it. We expect our method to achieve significantly higher accuracy (e.g., 0.8 vs 0.2) on recall and integrate tasks.

**vs. Full-context agent** — On a temporal-ordering task, the full-context agent sees the entire 500-step history but may be distracted by irrelevant details, leading to inconsistent performance across games (high variance). Our method truncates to W=100, forcing focus on recent context; however, the relevant order may be earlier than 100 steps in some games, causing misses. We expect our method to have lower variance (V_t < 0.1) but slightly lower accuracy on tasks that require information beyond W.

**vs. Standard LLM agent** — On a cross-game integrate-separated task, the standard LLM agent (GPT-4 without protocol) processes full context but is not calibrated for difficulty, so accuracy fluctuates with game-specific distractors. Our method calibrates instances to match difficulty across games, reducing performance gaps due to extraneous factors. We expect our method to show lower cross-game variance (e.g., V_t = 0.05 vs 0.25) while maintaining comparable average accuracy.

### What would falsify this idea

If our method's cross-game variance is not consistently lower than that of the standard LLM agent across all subtasks, or if the calibration ablation does not increase variance, then the claim that bounded-context calibrated evaluation measures general memory capability is wrong.

## References

1. AgenticSTS: A Bounded-Memory Testbed for Long-Horizon LLM Agents
2. DYSTIL: Dynamic Strategy Induction with Large Language Models for Reinforcement Learning
3. DSGBench: A Diverse Strategic Game Benchmark for Evaluating LLM-based Agents in Complex Decision-Making Environments
4. UrzaGPT: LoRA-Tuned Large Language Models for Card Selection in Collectible Card Games
5. The Illusion of Diminishing Returns: Measuring Long Horizon Execution in LLMs
6. ExpeL: LLM Agents Are Experiential Learners
7. Large Language Models can Learn Rules
8. GameBench: Evaluating Strategic Reasoning Abilities of LLM Agents
