# Monotone Evidence-Refinement Distillation for Medical AI Risk Guardrails

## Motivation

Distilled medical guardrails such as LingSante MedGuard (E1) optimize teacher agreement across a fixed set of demonstrations, so the risk head has no structural reason to treat benign additions of clinical evidence (an extra normal lab value, a resolved contraindication, a confirmed negative culture) as non-increasing in harmfulness. This is the same residual gap that recurs across guardrail trees: SafeWatch (E2) and SafetyAnalyst (E3) produce policy- or harm-benefit-scored outputs whose scores are not constrained to move monotonically as the underlying evidence set is refined, because their training objectives reward matching labels rather than respecting the partial order of the evidence lattice. Aggregate distillation therefore cannot close the gap — the missing object is a closure constraint on the evidence set, not more teacher labels.

## Key Insight

Benign evidence refinement forms a partial order on cases, and enforcing a monotone non-increasing constraint on the risk head over that partial order via paired distillation examples converts an unconstrained teacher-agreement objective into one whose feasible set excludes the specific failure mode of raising risk when exculpatory evidence is added.

## Method

**A. What it is.**
We introduce MERD (Monotone Evidence-Refinement Distillation), a training procedure for the risk-decision head of a medical guardrail. Inputs: the LingSante MedGuard student $f_\theta$ with risk head $r_\theta(c, E)$ scoring a case $c$ given evidence set $E$; a guardrail teacher $T$; and a paired corpus $\mathcal{P} = \{(c, E, E^+)\}$ where $E^+ \supset E$ adds only benign clinical evidence. Outputs: updated $\theta$ such that $r_\theta(c, E^+) \le r_\theta(c, E) + \epsilon$ for all pairs, while retaining teacher-agreement accuracy on unpaired cases.

**Load-bearing assumption (stated explicitly).** MERD assumes that the teacher's strict benign rubric can reliably certify an evidence addition as exculpatory (non-harm-increasing) *in isolation*, and that this isolated-evidence certification equals case-conditional non-harm-increase. This assumption fails when the added evidence is benign in isolation but interacts with the case (e.g., a normal potassium in a patient on a potassium-sparing diuretic), in which case the hinge penalizes a legitimately higher risk score. MERD therefore restricts the monotone constraint to a context-conditional subset of pairs where the added evidence is certified non-interacting with the case's active context, and reports coverage of that subset.

**B. How it works.**

```pseudocode
# Phase 1: build benign-refinement pairs (context-conditional)
for each case c in distillation corpus:
    E  = original evidence set (labs, history, meds, imaging)
    # active context = med list + problem list + allergies
    for each candidate benign addition e_benign:
        # rule-based interaction check against active context
        if interacts(e_benign, active_context(c)): skip
        E+ = E ∪ {e_benign}   # k ∈ {1,2,3} additions per pair
        # strict benign rubric: teacher T must label e_benign benign
        # AND confirm non-interaction with active context
        if T.benign_rubric(e_benign, c) != BENIGN: skip
        if T.interaction_check(e_benign, c) != NON_INTERACTING: skip
        add (c, E, E+) to P
# P is the verified subset; coverage = |P| / |all candidate pairs|

# Phase 2: multi-teacher on-policy distillation (inherited from MedGuard)
for student rollout y ~ f_θ(c, E):
    L_distill = CE(r_θ(c,E), T(c,E)) + λ_expl * L_explanation
    # λ_expl = 0.5, as in MedGuard

# Phase 3: monotone-consistency closure term (new, one-sided with slack)
for (c, E, E+) in P:
    s_E  = r_θ(c, E)
    s_Ep = r_θ(c, E+)
    # per-pair learned slack δ(c,E,E+) ≥ 0, initialized 0.0, L2-regularized
    L_mono = mean( relu( s_Ep - s_E + ε - δ ) )   # hinge, ε = 0.02 margin
    # δ allows abstention where exculpatory status is uncertain;
    # δ is clamped to [0, 0.1] and regularized with 1e-3 * ||δ||^2

# Phase 4: joint update
L = L_distill + β * L_mono      # β = 0.3, annealed to 0.1 over training
θ ← θ - η ∇L                     # η = 2e-5, AdamW
```

A calibration probe $g_\phi$ is fit on held-out paired cases to check that the monotone margin holds on unseen benign refinements; if the probe's violation rate exceeds 5%, the pair generator is audited and Phase 1 is rerun with stricter benign labels. The probe also reports coverage: the fraction of held-out candidate pairs that pass the context-conditional filter.

**C. Why this design.**
We chose a hinge penalty on paired benign refinements over a post-hoc calibration layer because post-hoc calibration (as in SafetyAnalyst's interpretable weight fitting, E3) can reorder scores but cannot enforce a lattice constraint that the training signal never saw, accepting the cost that MERD needs a curated benign-refinement corpus. We chose a strict benign-rubric teacher filter for $E^+$ over unlabeled evidence additions because the monotone constraint is only valid if added evidence is genuinely exculpatory, accepting the cost that the pair generator inherits the teacher's labeling errors. We chose a fixed margin $\epsilon=0.02$ over an adaptive margin because a fixed margin is auditable and reproducible, accepting the cost that cases with naturally tiny risk deltas contribute little learning signal. We chose to keep the inherited multi-teacher distillation loss (E1) rather than replace it with a pure monotone objective because teacher agreement still supplies the explanation supervision that the monotone term cannot, accepting the cost that the two objectives can conflict when a teacher labels a benignly-refined case as higher risk — which the hinge explicitly penalizes. We chose a context-conditional pair filter and a per-pair slack $\delta$ over a global partial order because clinical evidence is not closed under context: a benign-in-isolation addition can interact with the case (normal potassium on a potassium-sparing diuretic), and the slack lets the hinge abstain where exculpatory status is not clinically established, accepting the cost that coverage is reduced and must be reported. A domain expert would not describe MERD as a variant of standard distillation or of SafetyAnalyst-style moderation, because neither defines a partial order over evidence sets nor optimizes a closure constraint on that order.

**D. Why it measures what we claim.**
The hinge term $\text{relu}(s_{E^+} - s_E + \epsilon - \delta)$ measures benign-refinement monotonicity because we assume that the teacher's strict benign rubric and interaction check correctly identify evidence additions that cannot increase clinical harm in the case's active context; this assumption fails when the added evidence is benign in isolation but interacts with the case (e.g., a normal potassium in a patient on a potassium-sparing diuretic), in which case the hinge penalizes a legitimately higher risk score. The teacher-agreement term $\text{CE}(r_\theta, T)$ measures distillation fidelity because we assume the teacher's risk decisions are the target distribution; this assumption fails on cases where the teacher itself violates monotonicity, in which case the two losses conflict and the hinge dominates by construction of $\beta$. The calibration probe's violation rate measures generalization of the monotone invariant because we assume held-out pairs are drawn from the same benign-refinement distribution; this assumption fails under distribution shift in evidence types, in which case the probe reflects pair-generator coverage rather than true monotone consistency. The per-pair slack $\delta$ measures the fraction of hinge gradients that are abstained because exculpatory status is uncertain; this assumption fails when $\delta$ is driven to its upper bound on many pairs, in which case the hinge is effectively inactive and the monotone constraint is not being learned.

**Contribution.**
1. MERD, a distillation procedure that augments multi-teacher on-policy distillation with an evidence-refinement monotonicity hinge over a context-conditional benign-refinement pair corpus, yielding a risk head whose scores are constrained to be non-increasing under benign evidence additions where clinically valid.
2. A design principle: benign-refinement pairs filtered by a strict teacher rubric and an interaction check are a sufficient training signal to install a lattice closure constraint that aggregate teacher agreement cannot express, with reported coverage.
3. A paired benign-refinement evaluation protocol for medical guardrails that reports monotone-violation rate alongside standard risk-detection accuracy, stratified by clinician-flagged interaction risk.

**Limitation.** MERD assumes a reliable benign-evidence oracle (the teacher's strict rubric plus interaction check) and a corpus of cases where benign refinements are clinically well-defined; it is designed for single-turn risk scoring over structured evidence sets and does not address multi-turn evidence accumulation or adversarial evidence that is benign in isolation but harmful in combination.

**Inheritance.** paper_ids: ["arxiv:2611.25509"]. what_is_retained: the multi-teacher on-policy distillation pipeline and the risk-decision/explanation supervision structure of LingSante MedGuard (E1), including teacher-generated demonstrations and on-policy feedback.

**Mechanism change.** Relative to MedGuard's pure teacher-agreement distillation (E1), MERD adds a paired benign-refinement corpus and a hinge loss over the evidence lattice, replacing part of the agreement objective with a closure constraint; relative to SafetyAnalyst's interpretable weight fitting (E3), MERD enforces the constraint during distillation rather than post-hoc.

**Mechanism spec.** inputs: case $c$, evidence set $E$, benign-refinement pair $(c, E, E^+)$, teacher $T$, student risk head $r_\theta$. operations: context-conditional benign-pair generation with strict teacher filter and interaction check; multi-teacher on-policy distillation loss; hinge monotone-consistency loss with margin $\epsilon=0.02$, weight $\beta=0.3$, and per-pair slack $\delta \in [0, 0.1]$; joint gradient update. outputs: updated risk head $r_\theta$ satisfying $r_\theta(c, E^+) \le r_\theta(c, E) + \epsilon$ on paired cases in the verified subset.

**Verification.** baseline: LingSante MedGuard student without monotone term (E1). metric: monotone-violation rate on held-out benign-refinement pairs, plus risk-detection F1 on MedGuardBench. ablation: remove hinge ($\beta=0$), remove pair filter, vary $\epsilon$ and $\beta$. falsification: if held-out violation rate is not reduced relative to baseline while F1 is unchanged, the monotone constraint is not being learned from the pair signal.

**Evidence links.**
[{"evidence_id": "E1", "supports_assertion": "LingSante MedGuard integrates a guardrail teacher and six capability experts and consolidates supervision via offline SFT and multi-teacher on-policy distillation."}, {"evidence_id": "E2", "supports_assertion": "SafeWatch provides policy-following guardrail outputs with explanations but does not define a monotone constraint over evidence refinement."}, {"evidence_id": "E3", "supports_assertion": "SafetyAnalyst aggregates harm-benefit effects into a harmfulness score using interpretable weights, without an evidence-lattice monotonicity constraint."}]

## Contribution

(1) MERD, a distillation procedure that augments multi-teacher on-policy distillation with an evidence-refinement monotonicity hinge over a benign-refinement pair corpus, yielding a risk head constrained to be non-increasing under benign evidence additions. (2) A design principle that benign-refinement pairs filtered by a strict teacher rubric are a sufficient training signal to install a lattice closure constraint that aggregate teacher agreement cannot express. (3) A paired benign-refinement evaluation protocol for medical guardrails reporting monotone-violation rate alongside standard risk-detection accuracy.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Held-out benign-refinement pairs from MedGuard corpus | Directly probes the monotone invariant on unseen pairs |
| Primary metric | Monotone-violation rate on held-out pairs | Measures the exact constraint MERD claims to install |
| Baseline | MedGuard student, no monotone term | Tests whether hinge adds anything beyond teacher agreement |
| Baseline | Post-hoc calibration on MedGuard student | Tests whether constraint must be trained, not fitted |
| Baseline | Teacher's own risk head | Tests whether student can exceed the benign-oracle ceiling |
| Ablation-of-ours | MERD with β=0 (hinge removed) | Isolates the hinge term as the causal ingredient |

### Why this setup validates the claim

The central claim is that a hinge over teacher-filtered benign-refinement pairs installs an evidence-lattice closure constraint that teacher agreement alone cannot express. The held-out pair set is the right testbed because each pair is a direct instance of the invariant: a case where risk should not rise. The no-monotone baseline tests the sub-claim that distillation fidelity does not incidentally produce monotonicity; if it did, MERD would be redundant. The post-hoc calibration baseline tests the sub-claim that the constraint must be learned during distillation rather than reordered afterward — a fitted monotone map can flatten scores but cannot recover the evidence-conditional structure the student never saw. The teacher-head baseline tests whether the student can reach the benign-oracle ceiling, separating "MERD works" from "the teacher was already monotone." The β=0 ablation is the sharpest test: it holds data, teacher, and pipeline fixed and removes only the hinge, so any gap between it and full MERD is attributable to the hinge itself. Monotone-violation rate is the correct primary metric because it is precisely the quantity the hinge penalizes; risk-detection F1 is retained as a guard against a degenerate solution that satisfies monotonicity by flattening all scores.

### Expected outcome and causal chain

**vs. MedGuard student, no monotone term** — On a case where a resolved contraindication is added to the evidence set (e.g., a discontinued interacting drug appears in the med list), the baseline produces a risk score that drifts upward or stays flat-but-noisy because its only signal is teacher agreement on the original evidence, and it never receives gradient pressure tying $r_\theta(c,E^+)$ to $r_\theta(c,E)$. Our method instead sees this exact pair during training and the hinge pushes $s_{E^+}$ down whenever it exceeds $s_E + \epsilon$, so we expect a clear violation-rate gap concentrated on refinement types the baseline never supervised, with rough parity on cases where the added evidence is neutral.

**vs. Post-hoc calibration on MedGuard student** — On a case where the benign addition is a normal-range lab value that shifts the student's internal representation rather than its output scale, the baseline produces a monotone-looking but evidence-insensitive score because a fitted calibration map can only reorder or compress the existing score distribution, not condition on the evidence set. Our method instead trains the constraint into the risk head itself, so the score difference between $E$ and $E^+$ is shaped by the hinge during optimization. We expect the post-hoc baseline to show low violation on the calibration split but a rebound on held-out pairs with unseen evidence types, while MERD's violation rate stays low across both.

**vs. Teacher's own risk head** — On a case where the teacher's strict rubric is applied to a benign addition that interacts with the case (e.g., normal potassium in a patient on a potassium-sparing diuretic), the baseline teacher produces a legitimately higher risk score because its rubric evaluates the addition in isolation. Our method instead inherits this behavior through the distillation term but the hinge penalizes the student for reproducing it on paired cases, so we expect MERD to sit slightly below the teacher on violation rate for isolated-benign additions and to match the teacher on the interaction cases the pair filter should have excluded — a signature that the pair generator, not the hinge, is the binding constraint.

### What would falsify this idea

If the β=0 ablation matches full MERD on held-out violation rate, the hinge is not the causal ingredient and the claim that pair-filtered distillation installs the closure constraint is wrong. If MERD's gain is uniform across all refinement types rather than concentrated on the benign-addition subsets the hinge targets, the improvement is a generic regularization effect, not evidence-lattice monotonicity.

## References

1. LingSante MedGuard Technical Report
