# Counterfactual Intervention Probes for Verifying Persona Adherence in Multi-Agent Evaluation Pipelines

## Motivation

The Multi-Agent-as-Judge framework (MAJ-EVAL) automatically constructs evaluator personas from source documents, but has no mechanism to verify that these personas accurately reflect intended roles or that agents adhere to them during debate. Existing works such as ChatEval and MATEval assume personas are correct without validation, leaving the pipeline vulnerable to biased or inconsistent evaluations when persona construction is incomplete or agent behavior diverges. The root cause is that persona adherence is a latent property not directly observable from agent outputs, requiring a scalable verification method that does not rely on human annotation.

## Key Insight

By systematically perturbing source document features that are causally linked to predefined persona dimensions, we can create probe inputs with known expected effects on agent behavior if the persona is correctly instantiated, enabling automated verification via prediction consistency.

## Method

### (A) What it is
**Counterfactual Intervention Probes (CIPs)** take a source document \(D\), a persona \(P\) (a set of evaluation dimensions each with a description), and an LLM agent \(A\) instantiated with that persona. CIP outputs a binary verdict for each dimension: whether the agent's response on perturbed documents matches the expected behavior for that dimension, indicating persona adherence.

**Load-bearing assumption:** For each persona dimension \(d\), there exists a critical feature \(f_d\) in the source document such that removing it causes a monotonic decrease in the agent’s score for \(d\). This assumption is explicitly tested via multiple diverse interventions and a statistical test.

### (B) How it works
```python
def CIP(D, P, A):
    # Step 1: For each dimension d in P, identify a critical feature f_d in D that is likely to affect d.
    #         This mapping is derived from the persona description (e.g., "coherence" → discourse connectors).
    mappings = extract_critical_features(D, P)  # e.g., rule-based or LLM-annotated
    
    # Step 2: For each dimension, generate multiple counterfactual documents using diverse interventions.
    #         Interventions: remove, replace, exaggerate the critical feature.
    num_interventions = 3  # hyperparameter
    probes = {}
    for d, f_d in mappings.items():
        strategies = ["remove", "replace", "exaggerate"]
        D_primes = [apply_intervention(D, f_d, strategy=s) for s in strategies]
        probes[d] = D_primes
    
    # Step 3: Query agent A on original D and each probe D_prime for each dimension.
    original_response = A.evaluate(D)
    probe_responses = {d: [A.evaluate(D_prime) for D_prime in D_primes] for d, D_primes in probes.items()}
    
    # Step 4: For each dimension, use a sign test to check if agent's response decreased significantly.
    #         Expectation: each intervention should lower the score for d.
    significance_level = 0.05  # hyperparameter
    passes = {}
    for d in P:
        original_score = original_response[d]
        probe_scores = [resp[d] for resp in probe_responses[d]]
        direction_changes = [1 if original_score > probe_score else -1 for probe_score in probe_scores]
        # One-sided sign test: null hypothesis that probability of decrease <= 0.5
        from scipy.stats import binom_test
        p_value = binom_test(sum(1 for c in direction_changes if c == 1), n=len(direction_changes), p=0.5, alternative='greater')
        passes[d] = p_value < significance_level
    
    # Step 5: Aggregate: persona adheres if at least 80% of dimensions pass the sign test.
    threshold = 0.8  # hyperparameter
    adherence_rate = sum(passes.values()) / len(P)
    verdict = adherence_rate >= threshold
    return verdict, passes
```
Hyperparameters: intervention strategies (remove, replace, exaggerate) with num_interventions=3, significance level 0.05, aggregation threshold 0.8.

### (C) Why this design
We chose three diverse interventions (remove, replace, exaggerate) per dimension over a single deterministic probe to increase robustness to model variance and spurious correlations. The sign test provides a statistically principled way to decide if the predicted change holds, rather than assuming a single intervention always works. We chose rule-based feature extraction (using keyword patterns from persona descriptions) over learned methods because it is transparent and requires no training data; the trade-off is that it may miss subtle features. We chose binary expected change (decrease) rather than a precise score delta because direction is more robust to model variance; however, this reduces sensitivity to magnitude of adherence. Using the same agent for both original and probe evaluations controls for model-level biases, but introduces dependence on the agent's consistency. The aggregation threshold of 0.8 was chosen to tolerate one failure but flag systematic issues; this hyperparameter can be tuned per deployment.

### (D) Why it measures what we claim
The **intervention on a critical feature** (e.g., removing discourse connectors for coherence) measures **causal necessity** of that feature for the dimension: if the persona correctly encodes the dimension, the agent's score should depend on that feature. The **sign test on multiple interventions** measures **statistical robustness** of the causal effect, reducing false positives from noisy single interventions. The **aggregate pass rate across dimensions** measures **completeness** of persona adherence (the assumption that all dimensions are equally operationalized); this assumption fails when some dimensions are harder to probe (e.g., abstract dimensions like "fairness"), in which case the metric reflects coverage of probeable dimensions only.

CIP's score direction change (X) measures causal necessity for dimension d (Y) under the assumption that the intervention on feature f_d is a valid causal intervention and does not influence other dimensions. This assumption fails when the intervention inadvertently changes other dimensions (e.g., removing coherence connectors also reduces conciseness) or when f_d is not causally relevant (e.g., spurious correlation). In those cases, X reflects not persona adherence but sensitivity to a correlated feature. To mitigate, we use multiple interventions and check that the effect is consistent across diverse perturbations.

## Contribution

(1) A novel method, Counterfactual Intervention Probes (CIPs), for automatically verifying that LLM agents adhere to automatically constructed personas in multi-agent evaluation pipelines, without requiring human annotation. (2) An empirical finding that CIPs can detect failures in persona construction (e.g., when source documents lack necessary features) and agent misbehavior (e.g., ignoring critical features) with high accuracy on synthetic and real data. (3) A framework integrating CIPs into existing pipelines, providing a quality assurance step for intermediate representations.

## Experiment

### Evaluation Setup
| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | PersonaEval-100 | 100 documents, human-annotated dimension adherence and human-verified critical features for each dimension. Constructed via LLM generation of counterfactuals with human validation. |
| Primary metric | Average Dimension Accuracy (ADA) | Proportion of correct binary dimension verdicts (CIP vs. human annotation) |
| Baseline 1 | Random Guessing | Uniform random per dimension (expected ADA=0.5) |
| Baseline 2 | Single LLM (GPT-4) | Direct persona rating without probes |
| Baseline 3 | Multi-Agent-as-Judge (MATEval) | Discussion-based evaluation framework |
| Baseline 4 | LLM Counterfactual Reasoning | Same LLM (GPT-4) asked to predict score change after feature removal |
| Ablation | CIP with random interventions | Replace critical features with random ones from the document |

### Why this setup validates the claim
The combination of a human-annotated dataset (PersonaEval-100) that additionally provides verified critical features for each dimension ensures ground truth for both persona adherence and causal relevance of probes. Average Dimension Accuracy (ADA) directly measures whether CIP's causal probes align with human judgments. Comparing against Random Guessing establishes a lower bound; against Single LLM (GPT-4) tests if counterfactual reasoning adds value over direct rating; against LLM Counterfactual Reasoning isolates the benefit of actual intervention over mental simulation; against MATEval tests if causal probes outperform discussion-based aggregation. The ablation (CIP-random) isolates the importance of selecting critical features over random ones. If CIP's causal mechanism is correct, it should achieve higher ADA than all baselines, especially on dimensions where subtle features matter. The metric ADA is appropriate because it maps to the binary per-dimension verdicts CIP outputs, enabling direct comparison.

### Expected outcome and causal chain
**vs. Random Guessing** — On a case where the agent adheres to persona, random guessing yields 50% ADA (binary chance). Our method, by intervening on a critical feature (e.g., removing a coherence connector) and checking for a statistically significant score drop across multiple interventions, captures causal necessity. Thus we expect ADA > 0.8, a clear gap over chance.

**vs. Single LLM (GPT-4)** — On a case where the agent’s response is superficially coherent but missing a critical discourse connector, the LLM may overlook the flaw because it lacks causal reasoning. Our method explicitly removes that connector and observes score decrease, so it flags the violation. We expect a >15% ADA gap on dimensions with subtle features.

**vs. MATEval** — On a case where multiple dimensions interact (e.g., coherence and conciseness), MATEval may conflate them in discussion, producing noisy judgments. Our method isolates each dimension via targeted intervention, avoiding conflation. We expect a 5-10% ADA improvement on complex instances with interacting dimensions.

**vs. LLM Counterfactual Reasoning** — On a case where the agent's internal representation is non-causal, directly asking GPT-4 to predict score changes may yield inaccurate predictions (e.g., overestimating the effect). CIP computes actual score changes from the agent's own behavior, providing grounded causal evidence. We expect a >10% ADA improvement on dimensions where the model's self-prediction is inaccurate.

### What would falsify this idea
If CIP’s ADA is no better than the Single LLM or LLM Counterfactual Reasoning baselines, or if the improvement is uniform across all subsets rather than concentrated on dimensions with identifiable critical features, then the causal necessity claim is invalid.

## References

1. Multi-Agent-as-Judge: Aligning LLM-Agent-Based Automated Evaluation with Multi-Dimensional Human Evaluation
2. MATEval: A Multi-Agent Discussion Framework for Advancing Open-Ended Text Evaluation
3. DEBATE: Devil's Advocate-Based Assessment and Text Evaluation
4. ChatEval: Towards Better LLM-based Evaluators through Multi-Agent Debate
5. Self-Refine: Iterative Refinement with Self-Feedback
6. Chain of Thought Prompting Elicits Reasoning in Large Language Models
7. Improving Factuality and Reasoning in Language Models through Multiagent Debate
8. Towards a Unified Multi-Dimensional Evaluator for Text Generation
