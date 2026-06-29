# Ambiguity-Aware Evaluation: A Multi-Interpretation Framework for LLM Agent Benchmarks

## Motivation

Existing evaluators such as WebVoyager's GPT-4V judge and GauntletBench's rule-based metrics assume task instructions are unambiguous, penalizing agents that correctly interpret underspecified goals. This structural flaw causes false negatives and obscures agent adaptability. A framework that explicitly models task ambiguity is needed.

## Key Insight

Explicitly modeling task ambiguity as a set of plausible interpretations transforms the LLM judge's uncertainty from a noise source into a structured evaluation dimension.

## Method

**A) What it is**: Ambiguity-Aware Evaluation (AAE) is a framework that, given a task instruction and agent trajectory, outputs an overall score and an ambiguity report. Inputs: instruction text, sequence of screenshots and actions (from agent). Outputs: score ∈ [0,1] and list of interpretations with per-interpretation scores.

**B) How it works**:
```python
def evaluate(instruction, trajectory):
    # Phase 1: Ambiguity Analysis
    interpretations = llm_judge(prompt=ANALYSIS_PROMPT, instruction=instruction, trajectory=trajectory)
    # interpretations is a list of dicts: {description, likelihood}
    
    # Phase 2: Per-interpretation evaluation
    for interp in interpretations:
        # For deterministic aspects (e.g., field values), use rule-based checks
        rule_score = rule_based_check(trajectory, interp['description'])
        # For underspecified aspects, use LLM judgment
        llm_score = llm_judge(prompt=EVAL_PROMPT, instruction=interp['description'], trajectory=trajectory)
        interp['score'] = 0.5 * rule_score + 0.5 * llm_score  # hyperparameter: weighting
    
    # Phase 3: Aggregate
    overall = sum(interp['likelihood'] * interp['score'] for interp in interpretations)
    return overall, interpretations
```
Hyperparameters: interpretation likelihood threshold (min 0.01), score weighting (default 0.5 rule, 0.5 LLM).

**C) Why this design**: We chose to explicitly generate multiple interpretations rather than have the LLM produce a single score because it allows independent verification of ambiguity and separates the evaluation of deterministic vs. underspecified components. We chose to combine rule-based checks for deterministic aspects with LLM judgments for open-ended aspects because deterministic parts are reliably verifiable programmatically, reducing reliance on the LLM's hallucination-prone scoring. We chose to weight per-interpretation scores by the LLM-estimated likelihood of each interpretation, rather than using uniform weights, because ambiguous instructions have asymmetric plausible interpretations (e.g., "book a flight" most likely means economy class); the trade-off is that the LLM's likelihood estimate may be biased, so we accept a potential misweighting.

**D) Why it measures what we claim**: The generation of interpretations quantifies the degree of underspecification, because we assume that an instruction is ambiguous iff the LLM can enumerate multiple plausible interpretations that are consistent with the environment; this assumption fails when the LLM hallucinates interpretations not implied by the instruction, in which case the metric reflects LLM overcreativity rather than true ambiguity. The per-interpretation score measures how well the agent fulfills each specific interpretation, because we assume that the trajectory's state changes match the requirements of that interpretation; this assumption fails when the trajectory contains multiple valid paths and the LLM judge cannot distinguish, leading to score inflation. The weighted aggregation measures the agent's ability to handle the ambiguity, because we assume that a better agent has higher scores on more likely interpretations; this assumption fails when the likelihood weights are misaligned with human judgment, making the aggregate score an artifact of LLM prior.

## Contribution

(1) A multi-interpretation evaluation framework that explicitly models task ambiguity by decomposing instructions into plausible interpretations and scoring each. (2) A design principle that rule-based metrics and LLM judgments complement each other when ambiguity is identified and separated. (3) A structured prompting methodology for LLM judges to output both ambiguity analysis and per-interpretation evaluations.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|-----------------------|
| Dataset | Ambiguous Web Tasks (AWT) | 50 tasks with multiple plausible interpretations |
| Primary metric | Human-AAE Spearman correlation | Measures alignment with human judgment |
| Baseline 1 | Single LLM Judge | Standard practice, ignores ambiguity |
| Baseline 2 | Rule-based Success | Assumes deterministic success criteria |
| Baseline 3 | Naive Aggregate | Unweighted average of interpretation scores |
| Ablation | AAE w/o rule-based component | Tests importance of deterministic checks |

### Why this setup validates the claim

The combination of an explicitly ambiguous dataset, baselines that each omit a key component of AAE, and a metric that captures human alignment forms a direct falsifiable test. The dataset ensures that ambiguities exist, so differences will surface. Single LLM Judge tests whether explicit interpretation generation helps; Rule-based Success tests whether handling underspecification matters; Naive Aggregate tests whether likelihood weighting is beneficial; and the ablation tests the rule-based component. The primary metric (human-AAE Spearman ρ) directly measures fidelity to human judges, which is the ultimate goal. If AAE improves correlation over baselines specifically on ambiguous subsets, the claim that modeling ambiguity improves evaluation is supported; if gains are uniform or absent, the claim is challenged.

### Expected outcome and causal chain

**vs. Single LLM Judge** — On a task like "book a hotel near the airport" (ambiguous between budget vs. luxury), the Single LLM Judge returns a vague score (e.g., 0.6) because the LLM averages over conflicting interpretations. Our method explicitly enumerates both interpretations, evaluates each separately, and weights by likelihood, giving a score that reflects the agent's actual performance on the likely interpretation(s). We expect AAE's human correlation to be substantially higher (e.g., Δρ ≈ 0.3) on ambiguous tasks, but similar on unambiguous tasks.

**vs. Rule-based Success** — On a task "move the files to the folder" where the folder is underspecified (e.g., which subfolder), a rule-based check may fail entirely if the agent chooses a plausible but non-default folder. AAE's rule-based component only applies to deterministic aspects (e.g., file integrity), while the folder choice is handled by LLM judgment across interpretations. Thus AAE scores partial credit, better matching human assessment. We expect AAE to show higher correlation on tasks with underspecified actions (Δρ ≈ 0.4).

**vs. Naive Aggregate** — On a task where interpretations have very different likelihoods (e.g., "send the email" – most likely to all recipients, unlikely to only manager), the unweighted average inflates scores for unlikely interpretations. AAE's likelihood weighting downweights improbable ones, aligning with human priority. We expect AAE to outperform on tasks with skewed likelihood distributions (Δρ ≈ 0.2).

**vs. AAE w/o rule** — On tasks with deterministic verification (e.g., exact field values), removing rule-based checks leads to LLM hallucinations about state changes, inflating scores. AAE's hybrid design grounds scores. We expect AAE (full) to have higher correlation on tasks with deterministic components (Δρ ≈ 0.1-0.2).

### What would falsify this idea

If our method shows no systematic improvement in human correlation over baselines on ambiguous tasks, or if the gain is uniform across all subsets (including unambiguous ones), then the ambiguity modeling is not the cause but rather an artifact.

## References

1. Running the Gauntlet: Re-evaluating the Capabilities of Agents Beyond Familiar Environments
2. WebVoyager: Building an End-to-End Web Agent with Large Multimodal Models
3. REAL: Benchmarking Autonomous Agents on Deterministic Simulations of Real Websites
4. GPT-4V in Wonderland: Large Multimodal Models for Zero-Shot Smartphone GUI Navigation
5. The Dawn of GUI Agent: A Preliminary Case Study with Claude 3.5 Computer Use
6. Is Your LLM Secretly a World Model of the Internet? Model-Based Planning for Web Agents
7. TheAgentCompany: Benchmarking LLM Agents on Consequential Real World Tasks
8. AgentOccam: A Simple Yet Strong Baseline for LLM-Based Web Agents
