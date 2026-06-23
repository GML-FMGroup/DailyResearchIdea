# FARA: Counterfactual Consistency Enforcement for Role Fidelity in Multi-Agent LLM Evaluation

## Motivation

Existing multi-agent evaluation frameworks (e.g., Multi-Agent-as-Judge, DEBATE) assign personas to LLM agents but assume that a simple instruction is sufficient for sustained role adherence. However, model-specific biases and context drift cause agents to deviate from their assigned perspective, undermining the validity of multi-dimensional evaluation. This structural limitation—role fidelity is unaddressed—recurs across prior work, where each method relies on persona descriptions without verifying sustained adherence.

## Key Insight

Role fidelity can be operationalized as causal sensitivity of an agent's responses to systematic counterfactual changes in its persona description, enabling detection and correction of role drift without external supervision.

## Method

**FARA: Fidelity Assurance via Role Alignment**

(A) **What it is**: FARA is a wrapper module that ensures role fidelity of LLM agents during multi-agent debates. It takes as input a debate transcript (agent responses and assigned personas) and outputs a fidelity-corrected transcript.

(B) **How it works** (pseudocode):

```python
def fara(agent, persona, input_text, debate_context, hyperparams):
    # Step 1: Generate counterfactual personas (e.g., 3 perturbations)
    num_cf = hyperparams['num_cf']  # default 3
    cf_personas = sample_counterfactuals(persona, num_cf, strategy='dimension_swap')
    
    # Step 2: Query agent with each counterfactual persona
    cf_responses = [agent.generate(input_text, persona=cf_p, context=debate_context) for cf_p in cf_personas]
    
    # Step 3: Compute consistency score using calibrated sentence encoder
    # Encoder is fine-tuned on a small (e.g., 200-example) human-validated set of role-relevant changes
    calibration_encoder = hyperparams.get('encoder', 'fine-tuned_all-MiniLM-L6-v2')
    original_response = agent.generate(input_text, persona=persona, context=debate_context)
    consistency = compute_consistency(original_response, cf_responses, encoder=calibration_encoder, metric='cosine_sim')
    # consistency measures invariance: high similarity indicates role drift
    
    threshold = hyperparams['tau']  # default 0.7
    if consistency > threshold:  # drift detected
        corrected_response = contrastive_revision(original_response, cf_responses, persona, strength=hyperparams['alpha'])
        return corrected_response
    else:
        return original_response
```

(C) **Why this design**: We chose intervention-based consistency over simple self-report checks because agents can convincingly assert role adherence while drifting; counterfactual interventions provide an external, behavioral check. We sample multiple counterfactuals (e.g., 3) to avoid overfitting to a single perturbation, accepting the cost of additional API calls. We use semantic similarity as the consistency metric rather than exact match because responses can be semantically equivalent without being identical; this choice risks missing subtle drifts that only affect surface form, but prior work shows embedding-based metrics correlate well with human judgments of role consistency. We apply contrastive revision only when consistency falls below a threshold (tau=0.7) rather than always, because unnecessary corrections may introduce artifacts; this threshold is a tuning parameter that balances sensitivity and specificity.

(D) **Why it measures what we claim**: The computational quantity `consistency` (cosine similarity between original and counterfactual response embeddings) measures role drift (not fidelity) because a role-faithful agent should change its response when the persona is altered in a way that is causally relevant; thus, high consistency indicates that the agent is not attending to the persona change (drift), while low consistency indicates appropriate sensitivity. This assumption fails when the counterfactual perturbations are not causally relevant; in that case, high consistency may still indicate role fidelity if the change is irrelevant. We mitigate this by calibrating the embedder on a small human-validated set of expected response changes for each dimension, ensuring that the similarity metric reflects role-relevant semantic shifts.

## Contribution

(1) FARA, a counterfactual intervention framework for detecting and correcting role drift in multi-agent LLM evaluations. (2) A design principle: role fidelity can be assured via behavioral consistency tests under systematic persona perturbations, reducing reliance on prompt engineering. (3) Empirical analysis of role drift patterns in existing multi-agent debate systems, showing that drift occurs frequently and is detectable by FARA.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|-----------------------|
| Dataset | DebateQA-Bench | Real multi-agent debates with persona labels |
| Primary metric | Human-evaluated Role Fidelity (HRF) | Directly measures role adherence |
| Baseline 1 | Naive Multi-Agent Debate | No role fidelity mechanism |
| Baseline 2 | Single Agent Per Persona | One agent per role, no debate |
| Baseline 3 | DEBATE | State-of-the-art multi-agent evaluator |
| Baseline 4 | Self-Report Check | Agent self-assesses role adherence at each turn |
| Ablation | FARA w/o contrastive revision | Isolates detection from correction |

### Why this setup validates the claim

This setup directly tests the central claim that FARA improves role fidelity by detecting and correcting drift. The dataset, DebateQA-Bench, provides ground-truth persona assignments and human-annotated fidelity scores, enabling precise evaluation. The primary metric, HRF, captures the exact construct FARA aims to improve. Comparing FARA against Naive Multi-Agent and DEBATE isolates the effect of fidelity correction from general multi-agent advantages; the Single Agent baseline tests whether FARA's correction is more effective than simply using one agent per role. The Self-Report Check baseline highlights the novelty of intervention-based detection. The ablation (FARA without correction) reveals whether the detection alone yields improvements, identifying the necessity of the revision step. Together, these comparisons form a falsifiable test: if FARA's gains are not concentrated in cases where drift is predicted, the mechanism is wrong.

### Expected outcome and causal chain

**vs. Naive Multi-Agent Debate** — On a case where a neutral agent drifts to agree with a dominant persona, the naive debate produces a biased consensus because no mechanism checks role fidelity. Our method detects this drift via high consistency between original and counterfactual responses (indicating invariance), then applies contrastive revision to realign the agent's stance. We expect a noticeable gap on debates where persona conflicts are strong, but parity on homogeneous debates.

**vs. Single Agent Per Persona** — On a case requiring nuanced multi-perspective reasoning, a single agent fails to capture the diverse viewpoints because it lacks debate dynamics. Our method leverages multi-agent interaction while ensuring each agent stays faithful, producing richer responses. We expect our method to outperform on tasks requiring synthesis of distinct roles, but be comparable on simple single-role tasks.

**vs. DEBATE** — On a case where DEBATE's adversarial prompts cause agents to exaggerate their roles, DEBATE produces extreme positions because it encourages devil's advocacy without fidelity guard. Our method's counterfactual consistency check prevents such exaggeration by penalizing deviation from the intended persona. We expect our method to show higher fidelity scores on adversarial subsets, while maintaining similar task performance.

**vs. Self-Report Check** — On a case where an agent misreports its own role adherence, self-report fails to detect drift. Our intervention-based method detects drift via behavioral consistency checks, providing a more objective measure. We expect FARA to outperform in scenarios where agents exhibit subtle drift without acknowledging it.

### What would falsify this idea

If FARA shows uniform improvement across all instances rather than concentrated gains on cases where role drift is likely (e.g., high-conflict debates), or if the ablation (detection only) performs as well as the full method, then the central claim that correction is necessary would be wrong.

## References

1. Multi-Agent-as-Judge: Aligning LLM-Agent-Based Automated Evaluation with Multi-Dimensional Human Evaluation
2. MATEval: A Multi-Agent Discussion Framework for Advancing Open-Ended Text Evaluation
3. DEBATE: Devil's Advocate-Based Assessment and Text Evaluation
4. ChatEval: Towards Better LLM-based Evaluators through Multi-Agent Debate
5. Self-Refine: Iterative Refinement with Self-Feedback
6. Chain of Thought Prompting Elicits Reasoning in Large Language Models
7. Improving Factuality and Reasoning in Language Models through Multiagent Debate
8. Towards a Unified Multi-Dimensional Evaluator for Text Generation
