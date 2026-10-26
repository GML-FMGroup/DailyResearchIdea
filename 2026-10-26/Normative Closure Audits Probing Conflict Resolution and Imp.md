# Normative Closure Audits: Probing Conflict Resolution and Implicit Norms in Clause-Anchored Ethical Specifications

## Motivation

Constitutional AI and TenetBench both treat a normative specification as a finite list of explicit clauses, and TenetBench's own pipeline generates each probe from a single clause in isolation, so it structurally cannot test whether a model can resolve tensions between clauses or apply professional expectations that no clause states. The root cause is a missing closure property: the evaluation surface is the clause set itself, not the deductive and professional closure of that set, so a model can pass every single-clause probe while failing on exactly the cases where a real professional code is under-determined.

## Key Insight

A specification's normative commitments are only well-defined on its closure — the set of obligations derivable from clause interactions and professional practice — so probing behavior on clause pairs with known tension plus cluster-inferred implicit norms directly tests whether a model's behavior is derivable from the explicit corpus, an invariant that single-clause generation cannot establish.

## Method

**What it is.** Normative Closure Audit (NCA) takes the TenetBench clause-anchored corpus (clauses + benign/conflicting twins) and produces two new probe families over the *closure* of the clause set: (a) **tension pairs** — clause pairs whose joint satisfaction is impossible or professionally ranked, each with a required precedence resolution; and (b) **implicit-norm probes** — behaviors expected of a professional that are *not* entailed by any single clause but are induced from clause clusters. Input: code text + frozen TenetBench tasks. Output: a labeled closure-probe suite plus per-model closure-failure diagnostics.

**Load-bearing assumption (stated explicitly).** The implicit-norm family, the closure-failure metric, and the no-filter ablation all rest on the assumption that the non-entailment filter reliably certifies that an induced norm N is not derivable from any single clause. If the filter has a non-trivial false-negative rate on single-clause entailment (i.e., it labels a genuinely entailed norm as non-entailed), then 'implicit' probes are contaminated with explicit ones, the ablation contrast collapses, and the claim that closure probes measure non-derivability is unsupported. We therefore calibrate and operationally verify the filter as described in Phase 2.

**How it works.**

```pseudocode
INPUT: clauses C = {c_1..c_n} from a professional code; TenetBench twin tasks T
HYPERPARAMS: tension_threshold tau=0.62 (NLI contradiction score);
             cluster_k selected by silhouette over k in [3..12];
             min_support = 0.8 (fraction of cluster clauses a norm must be
             consistent with); judge = fixed LLM with rubric; alpha=0.05;
             filter calibration: target specificity = 0.95 on a labeled
             calibration set of 200 clause->norm pairs (100 entailed, 100 non-entailed);
             clause-ablation reader = fixed strong LLM (same as judge) with
             prompt 'Does this clause entail the norm? Answer yes/no';
             ablation_recover_threshold = 0.5 (LLM yes-probability above which
             the norm is deemed recoverable from that clause)

PHASE 1 -- Tension-pair mining (explicit conflict)
  for each ordered pair (c_i, c_j):
      s_ij = NLI_contradiction(paraphrase(c_i), paraphrase(c_j))   # entailment model
      if s_ij > tau:  add (c_i,c_j) to TENSION_CANDIDATES
  for each candidate, run PRECEDENCE PROBE:
      construct request r that simultaneously triggers c_i and c_j
      (reuse TenetBench twin generator, conditioned on both clauses)
      ask model to act; record whether it (i) satisfies both, (ii) drops one
      silently, (iii) states a precedence rule and justifies it
  keep pair only if a human adjudicator confirms joint impossibility
     (inter-annotator kappa reported); label with adjudicated precedence

PHASE 2 -- Implicit-norm induction (professional closure)
  # 2a. Filter calibration (two-sided, human-anchored)
  build calibration set of 200 clause->norm pairs: 100 entailed (including
     paraphrase-heavy and multi-hop cases) and 100 non-entailed, labeled by
     two human experts (AC1 reported). Tune tau to achieve specificity >= 0.95
     on this set; report the false-negative rate at the chosen tau.

  # 2b. Cluster and induce candidate norms
  embed clauses; cluster with k chosen by silhouette (k in [3..12])
  for each cluster K:
      prompt judge to state a norm N such that N is consistent with
      >= min_support of clauses in K but NOT entailed by any single c in K
      (judge must cite the >=2 clauses it generalizes)

  # 2c. Two-stage non-derivability filter
  for each candidate N:
      # stage 1: NLI non-entailment (tuned tau from 2a)
      if NLI(c_k -> N) >= tau for ANY c_k in K: reject N
      # stage 2: clause-ablation operational check
      for each c_k in K:
          p_yes = LLM_reader(c_k, N)   # probability of 'yes' to 'Does this clause entail the norm?'
          if p_yes > ablation_recover_threshold: reject N
      # consistency check
      if NLI(N -> c_k) < min_support for >= min_support of K: reject N
      keep N

  # 2d. Generate probe requests
  for each surviving N, generate a request r_N whose correct handling
  requires N but where NO clause in C mentions the required behavior
  (checked by lexical+entailment overlap filter); freeze for human audit

PHASE 3 -- Closure audit scoring
  for each model M and probe p in {tension pairs, implicit norms}:
      response = M(p); judge scores on rubric:
         conflict: did M detect tension? reason about precedence? act consistently?
         implicit: did M apply N without being told? warn when under-determined?
  CLOSURE FAILURE(p) = 1 if M's behavior is not derivable from any single
      clause AND M neither flags under-determination nor applies the
      adjudicated closure behavior
  report per-code closure-failure rate; bootstrap 95% CI; alpha=0.05
      significance vs. TenetBench single-clause baseline
  # Stratified reporting by filter confidence
  for each implicit-norm probe, record filter confidence = min over c_k of
      (tau - NLI(c_k -> N)) and ablation margin = min over c_k of
      (ablation_recover_threshold - p_yes). Stratify closure-failure rates
      into tertiles of filter confidence and ablation margin; report rates
      per stratum to isolate filter behavior from model behavior.
```

**Why this design.** We chose NLI-based contradiction mining over human-only pair authoring because the corpus must scale to new codes and jurisdictions, accepting the cost that NLI misses pragmatic tensions that only a professional would recognize (which is why every surviving pair passes human adjudication before entering the suite). We chose cluster-induced norms with an explicit non-entailment filter over prompting an LLM to 'list unwritten norms' directly, because the filter makes the implicit/explicit distinction auditable rather than a matter of the generator's whim, accepting the cost that many genuine professional norms will be rejected when they happen to be entailed by one clause. We chose a fixed-rubric judge over training a bespoke classifier because the failure taxonomy (silent drop vs. stated precedence vs. warning) is small and interpretable, accepting the cost that judge agreement must be validated against adjudicated labels as in TenetBench's Gwet's AC1 protocol. A domain expert would not describe NCA as a variant of Constitutional AI or of TenetBench's generator: both produce probes from single clauses or a principle list, whereas NCA's probes are defined by a non-derivability condition (no single clause entails the required behavior), which is a property of the clause *set* rather than of any clause.

**Why it measures what we claim.** The NLI contradiction score `s_ij` measures *normative tension* because two clauses whose paraphrases are mutually contradictory under a fixed entailment model cannot both be satisfied in a single action; this assumption fails when clauses are contradictory only under a specific factual instantiation, in which case the score reflects lexical/pragmatic opposition rather than operational conflict (hence the paired request in Phase 1 makes the conflict concrete). The non-entailment filter `NLI(c_k -> N) < tau for all c_k` measures *implicitness* because a norm not entailed by any single clause cannot be derived from the explicit corpus by clause lookup; this assumption fails when the entailment model under-detects a genuine entailment, in which case a probe labeled implicit is actually explicit and overstates the model's closure failure — mitigated by the two-stage filter (NLI + clause-ablation) and the calibration set that tunes tau to a target specificity of 0.95. The cluster-support `min_support` measures *professional consensus* because a norm consistent with a large fraction of a code's clauses reflects the code's overall stance; this assumption fails when clusters are topical rather than normative, in which case support reflects topic co-occurrence rather than shared obligation. The closure-failure indicator measures *specification incompleteness* because it flags behavior that is neither derivable from a single clause nor accompanied by a warning; this assumption fails when the judge misreads a legitimate clause-grounded justification as non-derivable, in which case the indicator reflects judge error rather than model behavior — mitigated by the validated Gwet's AC1 judge protocol and by the stratified reporting of closure-failure rates by filter confidence and ablation margin, which separates filter error from model behavior.

## Contribution

(1) The Normative Closure Audit, a pipeline that constructs tension-pair and implicit-norm probes defined by non-derivability from any single clause, extending clause-anchored corpora to the closure of a specification. (2) A design principle for specification-completeness evaluation: implicit norms are operationalized via cluster-induction plus an explicit NLI non-entailment filter, making the explicit/implicit boundary auditable rather than generator-defined. (3) A labeled closure-probe suite over multiple professional codes with adjudicated precedence labels and a per-model closure-failure diagnostic taxonomy (silent clause-drop, stated precedence, under-determination warning).

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | TenetBench clause corpus + frozen twins | Clause-anchored, human-validated, supports closure probes |
| Primary metric | Closure-failure rate on non-derivable probes | Directly measures specification incompleteness |
| Baseline | TenetBench single-clause twin accuracy | Tests whether closure adds signal beyond clause lookup |
| Baseline | Direct LLM "list unwritten norms" prompting | Tests whether non-entailment filter beats free generation |
| Baseline | Human-authored tension pairs only | Tests whether NLI mining scales without losing validity |
| Ablation-of-ours | NCA without non-entailment filter | Isolates the implicitness filter's contribution |
| Ablation-of-ours | NCA with NLI-only filter (no clause-ablation stage) | Isolates the contribution of the operational non-derivability check |
| Calibration set | 200 clause->norm pairs (100 entailed, 100 non-entailed) labeled by two experts | Tunes tau to specificity >= 0.95 and reports false-negative rate |
| Stratification | Tertiles of filter confidence and ablation margin | Separates filter error from model behavior in closure-failure rates |

### Why this setup validates the claim

The central claim is that closure probes — tension pairs and cluster-induced implicit norms defined by non-derivability from any single clause — expose model failures that clause-anchored evaluation cannot see. The single-clause twin baseline tests the sub-claim that closure adds signal: if models pass benign twins but fail on non-derivable probes, the failure is located in specification incompleteness rather than clause comprehension. The free-generation baseline tests the sub-claim that the non-entailment filter is doing real work: if prompting an LLM to list unwritten norms yields the same closure-failure pattern, the filter is unnecessary. The human-authored-pairs baseline tests scalability-without-invalidity: if NLI-mined pairs match human pairs on adjudicated tension, mining is sound. The no-filter ablation is the decisive control — it removes exactly the property (non-derivability) that defines the probes, so any gain attributable to NCA should collapse when the filter is removed. The NLI-only filter ablation isolates the clause-ablation stage, testing whether the operational non-derivability check adds discriminative power beyond the NLI score. The calibration set directly measures the filter's false-negative rate, quantifying how much of the closure-failure signal could be filter error. Stratified reporting by filter confidence and ablation margin ensures that any observed closure-failure gap is not an artifact of filter uncertainty. Closure-failure rate is the right metric because it is defined as behavior neither derivable from a single clause nor accompanied by a warning, which is precisely the quantity the claim asserts is under-measured by existing baselines.

### Expected outcome and causal chain

**vs. TenetBench single-clause twin accuracy** — On a request that triggers two clauses with ranked precedence (e.g., confidentiality vs. disclosure duty), the baseline model satisfies the salient clause and silently drops the other, because its training signal only ever rewards matching one clause at a time and it has no representation of the clause set's joint constraint. Our method instead presents the concrete dual-trigger request and scores whether the model states a precedence rule; a model that silently drops one clause is flagged as a closure failure. We expect a noticeable gap on tension-pair and implicit-norm subsets but near-parity on benign twins, since benign twins are clause-derivable and both setups handle them.

**vs. Direct LLM "list unwritten norms" prompting** — On a professional code with a topical cluster (e.g., several clauses about client communication), the baseline generator emits plausible-sounding norms that are often restatements of a single clause, because it optimizes for fluency and coverage rather than non-derivability. Our method rejects any candidate norm entailed by a single cluster clause via the two-stage non-derivability filter (NLI + clause-ablation), so surviving probes are guaranteed non-derivable. We expect the free-generation baseline to produce a suite that overlaps heavily with clause-lookup probes and thus shows a smaller closure-failure gap, while NCA's filtered suite shows a distinctly larger gap concentrated on probes the baseline would have mislabeled as implicit.

**vs. Human-authored tension pairs only** — On a jurisdiction where clause paraphrases are mutually contradictory under a fixed entailment model but the tension is only pragmatic (e.g., two clauses that conflict under one factual instantiation), the human-authored set misses the pair because authors only encode tensions they personally recognize, and coverage is bounded by authoring effort. Our method surfaces the candidate via NLI contradiction and then filters through human adjudication, so recall scales with the corpus while validity is preserved. We expect NLI-mined pairs to cover a superset of human pairs with comparable adjudicated precision, and closure-failure rates on the mined-only pairs to be similar in magnitude to human-authored pairs — evidence that mining adds coverage without adding noise.

**vs. NCA without non-entailment filter (ablation)** — Removing the filter admits norms that are entailed by a single clause, so the probe suite becomes contaminated with explicit norms. We expect the closure-failure rate on this ablated suite to be significantly lower than on the full NCA suite, and the gap between NCA and the single-clause baseline to shrink, because the ablated probes are largely clause-derivable. If the ablated suite shows the same closure-failure pattern as full NCA, the filter contributes no discriminative signal and the central claim collapses.

**vs. NCA with NLI-only filter (ablation)** — Removing the clause-ablation stage leaves the NLI score as the sole non-derivability check. We expect the NLI-only suite to contain more false-negative (actually entailed) norms than the full two-stage filter, manifesting as a higher closure-failure rate on probes that are in fact clause-derivable. The full filter should show a lower false-positive rate on the calibration set and a larger gap versus the single-clause baseline, demonstrating that the operational check adds value beyond the NLI threshold.

### What would falsify this idea

If the no-filter ablation produces the same closure-failure pattern as full NCA — i.e., the non-entailment filter contributes no discriminative signal — then the probes are not measuring implicitness and the central claim about closure-defined probes collapses. Equally, if NCA's gain over the single-clause baseline is uniform across benign twins and closure probes rather than concentrated on the non-derivable subsets, the failure being measured is generic model weakness, not specification incompleteness. Additionally, if the calibration set reveals a false-negative rate above 10% at the tuned tau and the clause-ablation stage does not reduce it, the implicit-norm probes are contaminated and the closure-failure metric overstates model failure.

## References

1. TenetBench: A Clause-Anchored Russian Corpus for Evaluating Language Models against Professional Codes of Ethics
2. Constitutional AI: Harmlessness from AI Feedback
