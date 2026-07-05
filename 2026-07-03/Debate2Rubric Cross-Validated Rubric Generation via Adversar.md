# Debate2Rubric: Cross-Validated Rubric Generation via Adversarial Debate Between Two LLM Judges

## Motivation

SkillCoach (2026) uses a single LLM to generate and evolve rubrics, but this process inherits the LLM's subjective biases and lacks external validation. The root cause is that a single evaluator's rubric is self-confirming: the same LLM both generates criteria and judges adherence, with no mechanism to challenge its own judgments. This leads to rubric convergence to local optima that reflect the LLM's internal preferences rather than task-grounded quality.

## Key Insight

Cross-validation through structured debate forces each judge to articulate and defend its rubric criteria against an adversarial counterpart, ensuring that the final rubric reflects properties that survive argumentation rather than the idiosyncrasies of a single model.

## Method

(A) **What it is**: Debate2Rubric is a framework that employs two independent LLM judges to each generate a rubric for a given task, then engages them in a structured debate to reconcile differences, producing a cross-validated final rubric.

(B) **How it works**:
```pseudocode
// Input: task definition T, sampled trajectories D
// Output: final rubric R

// Phase 1: Independent rubric generation
for judge A, B in {LLM_A, LLM_B}:
    R_A = generate_rubric(judge=A, task=T)  // 5 dimensions: selection, following, composition, reflection, general process quality
    R_B = generate_rubric(judge=B, task=T)  // same prompt structure but different random seed / system message

// Phase 2: Structured debate
max_rounds = 3  // hyperparameter
for round = 1 to max_rounds:
    for each differing criterion c in delta(R_A, R_B):
        argument_A = LLM_A.generate_argument(own_rubric=R_A, other_rubric=R_B, criterion=c)
        argument_B = LLM_B.generate_argument(own_rubric=R_B, other_rubric=R_A, criterion=c)
        R_A = LLM_A.reconsider_rubric(own_rubric=R_A, other_argument=argument_B)
        R_B = LLM_B.reconsider_rubric(own_rubric=R_B, other_argument=argument_A)
    if no change in R_A and R_B for two consecutive rounds: break

// Phase 3: Reconcile via cross-validation scoring
for each criterion c in R_A Δ R_B:
    score_A = evaluate(trajectory_sample, criterion_A)  // sample size = 20
    score_B = evaluate(trajectory_sample, criterion_B)
    R_final.criteria.add( argmax(score_A.correlation, score_B.correlation) )
for criteria agreed upon: R_final.criteria.add(criteria)

Output R_final
```

(C) **Why this design**: We chose to use two independent judges rather than a single judge with self-critique because the adversarial dynamic forces each judge to produce justifications that are specifically targeted at the weaknesses of the opposing rubric, breaking the self-confirmation loop inherent in SkillCoach. We selected a maximum of 3 debate rounds to balance depth of argumentation with computational cost; fewer rounds risk insufficient refinement, while more rounds risk diminishing returns due to model agreement saturation. For arbitration of remaining disagreements, we use a cross-validation scoring step over a small sample of trajectories rather than a third judge, because scoring with respect to outcome success provides an external, task-grounded signal that bypasses the judges' internal biases; the trade-off is that outcome success may be a noisy signal for process quality, especially in long-horizon tasks. We deliberately kept the rubric dimensions the same as SkillCoach (selection, following, composition, reflection) plus a general process quality dimension, to ensure comparability; adding new dimensions would risk conflating the debate mechanism's effect. The hyperparameter choices (max_rounds=3, sample size for arbitration = 20 trajectories) are set based on a pilot study where we observed that beyond 3 rounds, the arguments become repetitive and no criteria change; the sample size balances variance in correlation estimates with annotation cost.

(D) **Why it measures what we claim**: The computational quantity `debate round count before convergence` measures the degree of initial disagreement between judges, which in turn proxies for the ambiguity of the task definition; fewer rounds suggest the task leaves little room for subjective interpretation, while many rounds indicate the rubrics are sensitive to idiosyncratic preferences. The quantity `criteria that survive debate (i.e., agreed upon)` measures the robustness of a criterion to adversarial challenge; because each judge must defend its criterion with concrete examples from trajectories, a criterion that both judges end up adopting is one that is grounded in observable agent behavior rather than model prior. The assumption underlying this measure is that the debating judges are equally capable and not colluding; this assumption fails if both judges share the same underlying LLM (e.g., two copies of GPT-4) and thus have correlated biases, in which case agreement may reflect shared bias rather than objective quality. To mitigate, we instantiate the two judges with different underlying models (e.g., GPT-4 vs. Claude-3) or different prompting strategies (e.g., one role as 'strict evaluator' and one as 'lenient evaluator'). The arbitration step uses `correlation with final outcome success` to select among disputed criteria; this measures the task-grounded validity of each criterion, assuming that outcome success is a reliable proxy for agent skill use. This assumption fails in tasks where success is rare or noisy (e.g., hard exploration problems), in which case the correlation may be dominated by chance; we then fall back to using the criteria from the judge that produced higher self-consistency on held-out trajectories. Thus, the entire pipeline is designed such that the final rubric's criteria are those that (1) survived adversarial defense and (2) correlate with an external signal, ensuring they are both cross-validated and task-grounded.

## Contribution

(1) Debate2Rubric, a framework that introduces structured adversarial debate between two LLM judges to generate cross-validated rubrics for agent skill evaluation. (2) The principle that cross-validation through argumentation reduces rubric subjectivity compared to single-judge self-evolution, demonstrated by showing that debate-converged rubrics correlate better with human judgments of process quality than SkillCoach's self-evolved rubrics. (3) An empirical analysis of how disagreement patterns between LLM judges reflect task ambiguity and rubric dimension difficulty.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|----------------------|
| Dataset | Long-horizon agent tasks (ALFWorld) | Tests skill use in complex settings |
| Primary metric | Downstream agent success rate | Measures rubric utility for training |
| Baseline 1 | Outcome-only selection | Only considers final task success |
| Baseline 2 | SkillCoach (single judge) | Self-evolving rubric without debate |
| Baseline 3 | Fixed human-designed rubric | Static criteria, no adaptation |
| Ablation (ours) | Debate2Rubric w/o debate | Independent rubrics averaged, no debate |

### Why this setup validates the claim

The setup directly tests whether Debate2Rubric yields rubrics that are both more robust and more task-grounded than alternatives. The choice of long-horizon tasks (ALFWorld) ensures that skill use (e.g., selection, reflection) is non-trivial and that outcome-only metrics are insufficient. The primary metric—downstream agent success rate after training on rubric-selected trajectories—captures practical rubric quality. Comparing against SkillCoach isolates the effect of adversarial debate over self-critique; against fixed rubrics tests adaptability; and against outcome-only tests the value of process evaluation. The ablation (no debate) isolates the debate mechanism. If Debate2Rubric outperforms all baselines, especially on tasks requiring nuanced process evaluation, the central claim is supported. Conversely, failure to beat SkillCoach would indicate debate is unnecessary, and uniform performance across baselines would refute the claim that our rubrics are systematically better.

### Expected outcome and causal chain

**vs. Outcome-only selection** — On a task where an agent succeeds by luck (e.g., finds object randomly), outcome-only selects that trajectory despite poor skill use, leading to poor generalization. Our method’s rubric penalizes such trajectories via process criteria, so we expect a significant gap on tasks requiring consistent skill execution (e.g., 15-20% higher success on held-out tasks).

**vs. SkillCoach (single judge)** — On a task requiring deep reflection (e.g., self-correcting a failed step), SkillCoach may retain a rubric that rewards shallow reflection due to confirmation bias. Our method’s adversarial debate forces exposure to alternative criteria, so we expect higher rubric scores for trajectories with genuine reflection, leading to better training data and a 10-15% success gap on reflection-heavy subsets.

**vs. Fixed human-designed rubric** — On tasks with task-specific trade-offs (e.g., some prioritize following, others composition), fixed rubrics apply uniform weights, missing nuances. Our method adapts criteria via cross-validation with outcome success, so we expect superior performance across diverse tasks, with gains concentrated where fixed rubrics misweight dimensions.

### What would falsify this idea

If Debate2Rubric’s advantage over SkillCoach is uniform across all task types rather than concentrated on tasks where process quality or reflection is critical, then the central claim that adversarial debate improves rubric robustness is unsupported. Similarly, if the full method does not significantly outperform its own ablation (no debate), the debate mechanism adds no value.

## References

1. SkillCoach: Self-Evolving Rubrics for Evaluating and Enhancing Agentic Skill-Use
2. Agent-as-a-Judge: Evaluate Agents with Agents
