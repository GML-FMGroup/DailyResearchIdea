# Plan-then-Trace: Theory-Adherent Derivation Planning for Faithful Moral Reasoning

## Motivation

Existing methods such as the MET framework generate moral reasoning traces by first selecting theory grounds and then reasoning over them, but they do not verify whether the resulting trace faithfully follows the selected theory—the reasoning chain is treated as a black-box output. This meta-gap of missing faithfulness verification means that even if a model claims to follow a theory, the actual inference steps may diverge from the theory's derivation rules. The core structural problem is that no constraint during generation enforces that each reasoning step corresponds to a valid application of the theory's axioms or inference rules.

## Key Insight

If the derivation structure—the ordered set of premises and inference rule applications—is generated before the textual trace, then the trace's faithfulness to the theory is guaranteed by construction because the trace is merely a linguistic instantiation of a plan that already satisfies the theory's logical constraints.

## Method

(A) **What it is**: Theory-Adherent Derivation Planning (TADP) is a two-stage generation method that first produces a symbolic derivation plan specifying the sequence of premises and inference steps mandated by a chosen ethical theory, then generates a natural-language reasoning trace by instantiating the plan with scenario-specific details. Input: a moral dilemma (scenario + question) and a selected theory ground (e.g., deontology, utilitarianism, care ethics). Output: a reasoning trace that is guaranteed to follow the theory's inference structure.

(B) **How it works**:
```python
# Pseudocode for TADP
# Stage 1: Derivation Plan Generation (DPG)
#   Input: scenario S, theory ground T (from a curated set, e.g., from MET's grounds)
#   Step 1.1: Extract relevant ethical principles P from T (e.g., "do not harm", "maximize utility")
#   Step 1.2: Generate a plan plan = [] ordered list of derivation steps
#            Each step: (input_premises, inference_rule, output_conclusion)
#            Use few-shot prompting with examples of theory-adherent plans.
#            The plan length is bounded by max_steps = 5 (hyperparameter).
#   Step 1.3: Verify plan consistency: check that each step's premises exist, rules are valid for T.
#            (Rule validation via a symbolic lookup table of allowed inference rules for each theory.)
# Stage 2: Trace Generation (TG)
#   Input: scenario S, plan plan
#   Step 2.1: For each step in plan, generate a sentence in natural language that expresses
#            applying the inference rule to the premises, conditioned on S.
#            Use greedy decoding with temperature=0 (hyperparameter).
#   Step 2.2: Concatenate all sentences to form the trace.
```

(C) **Why this design**: We chose a two-stage architecture (plan then trace) over joint generation of plan and trace because the plan acts as a structural scaffold that can be verified for theory adherence before trace generation, directly addressing the meta-gap of missing faithfulness verification. We specifically generate the plan as a symbolic sequence of premises and inference rules (rather than a free-text plan) because symbolic plans can be checked against a formal rule inventory, ensuring logical validity; the trade-off is that symbolic plans may be less flexible and require a predefined rule set, limiting coverage to theories with explicit inference rules. We use few-shot prompting for plan generation (rather than a trained extractor) because it leverages the model's existing knowledge of ethical theories while avoiding expensive annotation; the cost is that plan quality depends on the examples selected and may vary across theories. We set the maximum plan length to 5 steps (a design choice based on typical moral reasoning complexity) to balance expressiveness and computational cost; longer plans risk verbosity without improving faithfulness. Finally, we use greedy decoding for trace generation to minimize randomness, accepting that this may reduce lexical diversity, but the plan already ensures structural faithfulness so diversity is secondary.

(D) **Why it measures what we claim**: The derivation plan's step-by-step structure (ordered premises and inference rules) operationalizes **structural faithfulness** because each step explicitly states which premises and rule from the theory are applied; this measures **adherence to theory** under the assumption that the plan's steps correctly instantiate the theory's inference rules; this assumption fails when the generated plan uses a rule that is not actually valid for the selected theory (e.g., applying a utilitarian calculus under deontology), in which case the trace would be unfaithful despite a valid-looking plan. The plan verification step (checking rule validity against a lookup table) operationalizes **theory adherence** because it ensures that only allowed rules appear; this measures **faithfulness by construction** under the assumption that the lookup table correctly captures the theory's inference rules; this assumption fails when the theory allows context-dependent rules not captured in the table, in which case verification may reject valid plans or accept invalid ones. The trace generation step (instantiating plan with scenario) operationalizes **trace correctness** because it follows the plan exactly; this measures **faithfulness as grounding** under the assumption that the scenario-specific content does not contradict the plan's premises; this assumption fails when the scenario introduces exceptional cases that the plan's rules do not cover, in which case the trace may still be faithful to the plan but not to the full moral context.

## Contribution

(1) A novel two-stage generation framework (TADP) that ensures structural faithfulness of moral reasoning traces by generating a theory-adherent derivation plan before the natural-language trace. (2) A symbolic plan verification mechanism that checks inference rules against a theory-specific lookup table, providing an explicit faithfulness guarantee that prior work (e.g., MET) lacks. (3) Design principles for integrating symbolic and neural components in moral reasoning: using few-shot prompting for plan generation and greedy decoding for trace instantiation, balancing flexibility and control.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|-------|--------|-----------|
| Dataset | MCLASH multilingual moral dilemmas | Diverse scenarios, multiple languages |
| Primary metric | Theory adherence score (rule-validity %) | Measures structural faithfulness directly |
| Baseline 1 | Zero-shot Chain-of-Thought | No explicit theory scaffolding |
| Baseline 2 | Translation-based prompting | English-centric reasoning transfer |
| Baseline 3 | MET baseline (no plan verification) | Ablates verification component |
| Ablation of ours | TADP without plan verification | Isolates effect of derivation plan generation |

### Why this setup validates the claim
MCLASH provides a challenging multilingual benchmark with culturally diverse scenarios, testing cross-lingual moral reasoning. Theory adherence score (rule-validity %) directly quantifies how often each inference step conforms to the chosen theory's allowed rules, making it the right metric for structural faithfulness. Zero-shot CoT tests whether implicit reasoning can match structured plans; translation-based prompting tests whether English-centric reasoning transfers poorly; the MET baseline (our method without verification) tests the contribution of the symbolic plan verification step. The ablation removes verification, isolating whether the derivation plan itself improves adherence or whether verification is critical. This combination creates a falsifiable test: if our method's advantage vanishes when verification is removed, the structural scaffold is not sufficient; if our method outperforms baselines only on complex multi-step scenarios, the causal mechanism is plausible.

### Expected outcome and causal chain

**vs. Zero-shot Chain-of-Thought** — On a deontological dilemma where a rule like "do not lie" conflicts with utilitarian outcomes, zero-shot CoT often produces a utilitarian reasoning trace because the model defaults to consequentialist patterns. Our method first generates a plan explicitly listing deontological premises and inference rules (e.g., "categorical imperative"), then instantiates it, so the trace remains rule-bound. We expect a noticeable gap on deontological and duty-based scenarios (e.g., 20-30% higher theory adherence), but parity on simple cases where CoT already adheres.

**vs. Translation-based prompting** — On a culturally specific dilemma from a non-English language (e.g., Hindi), translation-based prompting translates the scenario to English, reasons, then translates back; this loses cultural nuance and may apply Western ethical frameworks incorrectly. Our method uses the original language for trace generation but the plan is language-agnostic (symbolic), so it adapts to local context while maintaining structure. We expect a large gap on scenarios involving localized norms (e.g., family duty in East Asian cultures), with our method showing >15% higher adherence on those subsets.

**vs. MET baseline (no plan verification)** — On a scenario where the generated plan accidentally includes an inference rule not allowed by the chosen theory (e.g., applying "maximize happiness" under deontology), the unverified plan produces an unfaithful trace that still looks structured. Our full TADP catches such invalid rules via the symbolic lookup table, preventing the trace. We expect our full method to outperform the ablation on cases where rule violations are subtle, by about 10-20% in theory adherence.

### What would falsify this idea
If our method's gains are uniform across all baselines and subsets (e.g., a flat 5% improvement everywhere) rather than concentrated on scenarios where the baselines' failure modes are predicted (e.g., deontological rules, non-English cultures, or rule-violation-prone plans), then the central claim that structural scaffolding and verification cause the improvement would be falsified.

## References

1. MET: Theory-Grounded and Culture-Aware Multilingual Moral Reasoning
2. Structured Moral Reasoning in Language Models: A Value-Grounded Evaluation Framework
3. One Model, Many Morals: Uncovering Cross-Linguistic Misalignments in Computational Moral Reasoning
4. MoralBench: Moral Evaluation of LLMs
5. Understanding and Mitigating Language Confusion in LLMs
6. HELP ME THINK: A Simple Prompting Strategy for Non-experts to Create Customized Content with Models
7. Large Language Models are Zero-Shot Reasoners
8. Decomposed Prompting: A Modular Approach for Solving Complex Tasks
