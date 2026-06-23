# Consistency-Driven Falsification: Verifying LLM Implication Derivation via Systematic Hypothesis Perturbations

## Motivation

The Popper framework (Automated Hypothesis Validation with Agentic Sequential Falsifications) relies on LLMs to derive implications from hypotheses for falsification, yet provides no mechanism to verify the correctness of those derivations. This reliance on a single high-capability LLM without verification makes the falsification process vulnerable to reasoning errors. The root cause is that existing methods assume LLMs are reliable reasoners, but no structural property of LLM reasoning is enforced or checked. To bridge this gap, we need a method that detects flawed implication derivations without requiring ground truth.

## Key Insight

Logical consistency under systematic hypothesis perturbations provides a certificate of implication derivation correctness because any sound reasoning must preserve entailment relationships; violations indicate flawed inference.

## Method

(A) **What it is**: Consistency-Driven Falsification (CDF) is a verification layer that takes a hypothesis H and an LLM's derived implications I(H) and outputs a consistency score ρ indicating the reliability of the derivation. It does so by perturbing H into variants H' and checking that entailments among H and H' are mirrored in the implications.

(B) **How it works**:
```python
def cdf_consistency(H, LLM, perturbation_ops, alpha=0.8):
    I_H = LLM.derive_implications(H)
    violations = 0
    total_checks = 0
    for op in perturbation_ops:
        H_prime = op.apply(H)
        I_H_prime = LLM.derive_implications(H_prime)
        # Check entailment in both directions and negation
        for (d1, d2) in [(H, H_prime), (H_prime, H), (neg(H), H_prime), (H_prime, neg(H))]:
            entail = symbolic_entailment_checker.entail(d1, d2)
            if entail != UNKNOWN:
                total_checks += 1
                actual = symbolic_entailment_checker.entail(I_d1, I_d2)  # I_d1 is LLM implications of d1
                if actual != entail:
                    violations += 1
    consistency = 1 - violations / max(1, total_checks)
    return consistency > alpha, consistency
```
Perturbation operators include: add a conjunct, remove a conjunct, replace a predicate with a related one, negate one part, swap conjunction order. Hyperparameters: alpha (default 0.8) for pass threshold; perturbation_ops defined per domain.

(C) **Why this design**: We chose to use systematic hypothesis perturbations rather than random perturbations because systematic perturbations target specific logical relationships, ensuring coverage of entailment types. We use a symbolic entailment checker instead of another LLM because the checker provides ground-truth entailment labels independent of LLM reasoning, avoiding circular verification. We compute a consistency ratio rather than a binary pass/fail because it provides a graded reliability signal that can inform sequential testing. The trade-off is that the symbolic checker requires hypotheses to be expressible in a logical form, limiting scope, but it provides a reliable oracle for the verification.

(D) **Why it measures what we claim**: The consistency ratio ρ measures LLM implication derivation reliability because it operationalizes the invariant that sound logical entailment between hypotheses must be preserved under derived implications; this assumption (that entailment preservation is necessary for correct derivation) fails when the LLM's reasoning is incoherent or when the perturbation does not capture the intended logical relationship (e.g., if the perturbation changes the meaning in a way not captured by logical entailment), in which case ρ reflects the robustness of the LLM to those perturbations rather than correctness per se. The number of violations measures the extent of logical inconsistency; it directly quantifies deviation from sound reasoning because each violation corresponds to a case where the LLM's implications contradict the known entailment, assuming the symbolic checker's entailment judgment is correct; this assumption fails when the symbolic checker itself is inaccurate (e.g., due to underspecified logical forms), in which case violations reflect logical form errors instead of reasoning errors.

## Contribution

(1) A verification method, Consistency-Driven Falsification (CDF), that checks LLM implication derivation by enforcing logical consistency under systematic hypothesis perturbations. (2) A set of perturbation operators designed to cover common logical transformations in scientific hypotheses, enabling robust consistency measurement without ground truth. (3) The finding that consistency score ρ correlates with derivation correctness, providing a practical filter for unreliable implications in autonomous falsification.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Hypotheses with ground-truth implications | Human-verified correct and incorrect derivations. |
| Primary metric | AUC-ROC for correct derivation identification | Measures discriminative power of consistency score. |
| Baseline 1 | Human experts | Upper bound, tests human-level reliability. |
| Baseline 2 | LLM self-consistency | Naive repeated sampling; tests added value of perturbations. |
| Baseline 3 | Raw LLM output | No verification; quantifies improvement. |
| Ablation of ours | CDF with LLM entailment checker | Isolates contribution of symbolic checker. |

### Why this setup validates the claim
This setup directly tests whether the consistency score ρ from CDF reliably distinguishes correct LLM-derived implications from incorrect ones. The curated dataset with ground-truth labels enables computing AUC-ROC, which quantifies the trade-off between true and false positives—picking errors due to entailment violations. The human baseline establishes whether CDF matches or exceeds expert judgment; self-consistency baseline tests if systematic perturbations add signal beyond simple repetition; raw LLM baseline shows the deficit without verification. The ablation (using LLM for entailment checks) reveals the importance of the symbolic oracle. The chosen metric is appropriate because it captures the core claim: a graded reliability score should separate correct and incorrect derivations. The combination of baselines and ablation isolates each sub-claim: human-level performance, perturbation necessity, and symbolic checker indispensability.

### Expected outcome and causal chain

**vs. Human experts** — On a case where a hypothesis requires precise logical entailment (e.g., conjunctive statements), a human may miss subtle inconsistencies due to fatigue or bias; the expert baseline thus produces inconsistent judgments. Our method systematically checks entailment preservation across perturbations, so it maintains consistent scores. We expect CDF to achieve AUC-ROC comparable to or slightly below the human consensus (Δ < 0.05) but with higher reliability across repeated trials.

**vs. LLM self-consistency** — On a case where the LLM is confidently wrong (e.g., a plausible but false implication), self-consistency may show high agreement across samples because the LLM repeats the same flawed reasoning. CDF identifies the flaw because perturbations (e.g., negating a conjunct) change the entailment relation, which the LLM fails to mirror; thus ρ drops. We expect CDF to show a noticeable advantage (AUC-ROC gain >0.1) on the subset where the LLM is confidently incorrect.

**vs. Raw LLM output** — On any case, the raw LLM always outputs it is correct, yielding high recall but low precision (AUC-ROC around 0.5). CDF’s consistency score thresholds allow rejecting false derivations; we expect precision to rise substantially (e.g., from ~0.5 to >0.8) while recall remains high.

**Ablation: CDF with LLM entailment checker** — On a case where entailment relations are logically intricate (e.g., nested quantifiers), the LLM-based checker may err, causing violations to reflect the checker’s mistakes rather than reasoning errors. CDF with symbolic checker avoids this, maintaining correct violation counts. We expect the symbolic version to outperform the LLM version by >0.1 AUC-ROC on the hardest entailment subset.

### What would falsify this idea
If CDF’s AUC-ROC does not significantly exceed the raw LLM baseline, or if the gain is uniform across all perturbation types rather than concentrated where the predicted failure mode (e.g., confident LLM errors) occurs, then the central claim that ρ measures derivation reliability would be contradicted.

## References

1. Automated Hypothesis Validation with Agentic Sequential Falsifications
