# Counterfactual Identifiability Certification for Agent-Evolution Recipes

## Motivation

Across the agent-evolution paper-trees, evaluation is factored out of the mutable surface and frozen as a fixed oracle — RSIHub explicitly keeps evaluation 'outside the mutable surface' so that recipes can only optimize against whatever ranking the evaluator emits. This structural choice means a recipe can appear to improve while its declared objective and the evaluator's ranking silently diverge, and the recipe layer has no mechanism to detect this proxy-optimization; the same oracle-fixation recurs in Darwin Gödel Machine-style self-improvement (where fitness is assumed to track the intended capability) and in ACE-style context curation (where reflection quality is assumed to track downstream utility). The unresolved meta-gap is that no operator certifies whether the fixed evaluator's ranking is identifiable with respect to the recipe's declared objective.

## Key Insight

A recipe can manufacture candidate pairs whose true objective ordering is known by construction (because the recipe itself controls the intervention that differs between them), so agreement between the black-box evaluator's ranking and that known ordering on these counterfactual pairs is a direct, recipe-local certificate of whether the evaluator's signal is the signal the recipe is supposed to optimize.

## Method

## Counterfactual Identifiability Certification (CIC)

**(A) What it is.** CIC is an operator that a recipe invokes to emit a per-recipe *identifiability certificate*: given a black-box evaluator `E` (returns a ranking over candidate agents) and the recipe's declared objective `O` (a natural-language + machine-checkable predicate over agent behavior), CIC returns an agreement statistic `κ ∈ [-1,1]` plus a verdict (`IDENTIFIABLE` / `PROXY-DIVERGENT`) and the specific counterfactual pair families where divergence occurs. Inputs: the recipe's mutable surface, the candidate lineage from RSIHub, and `E`. Output: certificate attached to the recipe, not to any single candidate. **Load-bearing assumption (stated explicitly):** the recipe author's declared monotone intervention operators (`apply_beneficial` / `apply_harmful`) actually induce a true ordering on the latent objective `O`, so that `O(a+) > O(a-)` holds by construction rather than by the author's belief. CIC does not assume this silently: it measures the specification-error rate on a calibration subset and reports it alongside `κ`, so a `PROXY-DIVERGENT` verdict is identifiable between 'E diverges from author belief' and 'author belief diverges from O'.

**(B) How it works.**

```pseudocode
CIC(recipe R, evaluator E, lineage L, budget B=64 pairs/family, calibration fraction c=0.25):
  # Phase 1: enumerate intervention families from the mutable surface
  families = []
  for mut in R.declared_mutable_resources:          # e.g. prompt, tool-set, memory
      for op in R.operators_touching(mut):
          families.append(InterventionFamily(mut, op))

  # Phase 2: for each family, generate counterfactual pairs with KNOWN objective order
  pairs = []
  for fam in families:
      base = sample_from(L, fam)                    # a real lineage candidate
      for k in 1..B:
          # construct a pair (a+, a-) where O(a+) > O(a-) BY CONSTRUCTION
          a_plus  = fam.apply_beneficial(base)      # op with a known-good argument
          a_minus = fam.apply_harmful(base)         # op with a known-bad argument
          assert O.check(a_plus) and not O.check(a_minus)   # construction guard
          pairs.append((a_plus, a_minus, fam))

  # Phase 2b: specification audit on a held-out calibration subset
  # For each family, sample c*B pairs (c=0.25, so 16 pairs/family at B=64)
  # and run an independent, execution-grounded behavioral probe of O
  # (e.g., run the agent on a held-out task and check task completion).
  # Record whether O(a+) > O(a-) holds under the probe.
  audit = {}
  for fam in families:
      cal_pairs = sample(pairs[fam], c*B)
      audit[fam] = mean(1 if probe_O(a_plus) > probe_O(a_minus) else 0
                        for (a_plus, a_minus) in cal_pairs)
  # audit[fam] is the measured specification-error rate for family fam.
  # Families with audit[fam] < 0.5 are flagged as SPECIFICATION-UNRELIABLE
  # and their kappa is reported but excluded from the global minimum.

  # Phase 3: probe the black-box evaluator on each pair
  agree = {}
  for (a_plus, a_minus, fam) in pairs:
      r = E.rank([a_plus, a_minus])                 # E is black box; only ranking read
      agree[fam].append(1 if r.index(a_plus) < r.index(a_minus) else 0)

  # Phase 4: aggregate into certificate
  kappa = {fam: 2*mean(agree[fam]) - 1 for fam in families}   # per-family agreement
  reliable_fams = [fam for fam in families if audit[fam] >= 0.5]
  kappa_global = min(kappa[fam] for fam in reliable_fams)     # worst reliable family governs
  verdict = IDENTIFIABLE if kappa_global >= tau (tau=0.6) else PROXY-DIVERGENT
  return Certificate(kappa, kappa_global, verdict,
                     divergent_families=argmin(kappa over reliable_fams),
                     specification_audit=audit,
                     unreliable_families=[fam for fam in families if audit[fam] < 0.5])
```

The construction guard in Phase 2 is the load-bearing step: `fam.apply_beneficial` / `fam.apply_harmful` are recipe-declared operators whose effect on `O` is monotone by the recipe author's own specification, so `O(a+) > O(a-)` is known *without* consulting `E`. Phase 2b makes the load-bearing assumption explicit and measurable: on a held-out calibration subset of `c*B = 16` pairs per family, an independent execution-grounded probe of `O` checks whether the author's monotonicity claim actually holds, yielding a per-family specification-error rate `audit[fam]`. Families with `audit[fam] < 0.5` are flagged `SPECIFICATION-UNRELIABLE` and excluded from the global minimum, so `kappa_global` is computed only over families whose construction guard passes the external check. Phase 3 reads only the ranking `E` emits — never its internal scores — so CIC works against any oracle. Phase 4 uses the *minimum* over reliable families rather than the mean because a recipe is only as trustworthy as its weakest identifiability channel.

**(C) Why this design.** We chose recipe-declared monotone intervention operators over post-hoc statistical correlation between `E` scores and `O` values because correlation requires `O` to be cheaply evaluable at scale (it is not — `O` is the latent target), accepting the cost that CIC only certifies families the recipe author can name. We chose the *minimum* over families over the mean because a recipe that is identifiable on prompt mutations but proxy-divergent on tool-set mutations is exactly the failure mode RSIHub's fixed-oracle design cannot currently surface, accepting the cost of a stricter pass criterion that flags more recipes as divergent. We chose ranking-only access to `E` over score access because RSIHub's evaluator is a fixed oracle whose scores are not comparable across candidates, accepting the cost of lower statistical power per pair (rank agreement is coarser than score correlation). We added the Phase 2b specification audit because the construction guard substitutes the recipe author's monotonicity belief for the evaluator, and without an external check a `PROXY-DIVERGENT` verdict is unidentifiable between 'E is wrong' and 'the author is wrong'; the audit costs `c*B = 16` execution-grounded probes per family (25% of the pair budget) and yields a measured specification-error rate that is reported alongside `κ`, accepting the cost that families with `audit[fam] < 0.5` are excluded from the global minimum and the certificate under-reports identifiability on those channels. A domain expert would not describe CIC as a variant of standard evaluator-agreement metrics (e.g., Cohen's kappa between two annotators) because those require two *independent* labelers over the *same* items; CIC manufactures the reference label by intervening on the recipe's own mutable surface, so the reference is not an external annotator but a construction the recipe controls.

**(D) Why it measures what we claim.** `kappa[fam]` measures *evaluator-fitness identifiability for family `fam`* because the construction guard makes `O(a+) > O(a-)` true by the recipe author's monotonicity specification, so rank agreement reduces to whether `E` respects a known objective ordering; this assumption fails when the recipe author's `apply_beneficial` is not actually monotone in `O` (e.g., a longer prompt that the author believes helps but that degrades tool-call precision), in which case `kappa` reflects the author's specification error rather than evaluator divergence — which is why Phase 2b reports `audit[fam]` and excludes families with `audit[fam] < 0.5` from `kappa_global`. `kappa_global = min_{fam in reliable} kappa[fam]` measures *recipe-level identifiability* because a recipe is only certified if every declared mutable channel that passes the specification audit is faithfully ranked; this assumption fails when the recipe declares a mutable resource it never actually optimizes (a dead channel), in which case the minimum is dominated by an irrelevant family and the certificate under-reports identifiability. The `PROXY-DIVERGENT` verdict measures *proxy-optimization risk* because divergence on a constructed pair is direct evidence that improving `E`'s ranking need not improve `O`; this assumption fails when `E` is stochastic across calls, in which case a single divergent pair reflects evaluator noise rather than systematic proxy divergence, which is why Phase 3 averages `B=64` pairs per family before thresholding at `tau=0.6`.

## Contribution

(1) The Counterfactual Identifiability Certification (CIC) operator, which converts a recipe's declared mutable surface into a family of counterfactual candidate pairs with construction-guarded objective orderings and returns a per-recipe, per-family identifiability certificate against a black-box evaluator. (2) The design principle that evaluator-fitness identifiability is a *recipe-level* property certifiable by construction rather than a *candidate-level* property estimated by correlation, with the worst-family minimum as the governing statistic. (3) A reusable certificate schema attachable to any RSIHub recipe without modifying the evaluator or the framework core.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | RSIHub recipes × Terminal Bench 2 + τ³ Banking | Real mutable surfaces with declared operators to intervene on |
| Primary metric | κ_global (worst reliable-family rank agreement) | Directly quantifies per-channel evaluator-fitness identifiability |
| Baseline | Fixed-oracle RSIHub (AHE/Hyperagents/A-Evolve/GEPA) | Tests whether CIC surfaces divergence the oracle hides |
| Baseline | Score-correlation agreement (Pearson on E scores) | Tests whether rank-only access suffices vs. score access |
| Baseline | Mean-over-families aggregation | Tests whether worst-family minimum is the load-bearing choice |
| Ablation-of-ours | CIC without construction guard (post-hoc O correlation) | Tests whether manufactured labels, not correlation, drive signal |
| Ablation-of-ours | CIC without Phase 2b specification audit (author belief only) | Tests whether the audit separates E-divergence from author error |
| Diagnostic | Specification-error rate audit[fam] on 16-pair calibration subset | Measures whether author monotonicity claim holds under execution probe |

### Why this setup validates the claim

The central claim is that CIC can certify, per recipe, whether a fixed black-box evaluator's ranking respects the recipe's declared objective — and that the worst-family minimum surfaces proxy divergence that a fixed-oracle RSIHub run cannot. The dataset choice matters because RSIHub recipes are exactly the objects CIC attaches certificates to: they have declared mutable surfaces (prompt, tool-set, memory) and named operators, which is the precondition for Phase 2's construction guard. The fixed-oracle RSIHub baseline tests the sub-claim that divergence exists but is invisible under standard evolution runs; if CIC's certificates flag recipes as PROXY-DIVERGENT while those recipes nonetheless "succeed" under the fixed oracle, the baseline is failing to surface the divergence CIC reports. The score-correlation baseline tests the sub-claim that ranking-only access is sufficient: if score-correlation agreement and CIC's rank agreement rank the same recipes identically, the ranking-only design choice buys nothing and the claim weakens. The mean-over-families baseline tests the sub-claim that the minimum is load-bearing: if mean and min produce the same verdicts across the recipe set, the stricter criterion is decorative. The no-construction-guard ablation tests the deepest sub-claim — that manufactured reference labels, not statistical correlation between E and O, are what make the certificate meaningful; if removing the guard leaves κ_global unchanged, the guard is not doing the work we claim. The no-audit ablation tests the load-bearing assumption directly: if removing Phase 2b leaves κ_global and the verdict set unchanged, the author's monotonicity belief is not a confound in this recipe set; if it changes verdicts, the audit is separating E-divergence from author error as claimed. The specification-error rate audit[fam] is the measured quantity that bounds the gap between the author's monotonicity specification and the true objective: on the 16-pair calibration subset per family, an execution-grounded probe of O checks whether O(a+) > O(a-) holds, and the fraction where it does is reported alongside κ. κ_global is the right metric because it is the certificate's own decision variable: a falsifiable test is whether the certificate's verdicts line up with an independent, held-out behavioral check on the divergent families CIC names.

### Expected outcome and causal chain

**vs. Fixed-oracle RSIHub (AHE/Hyperagents/A-Evolve/GEPA)** — On a recipe whose prompt-mutation channel is faithfully ranked but whose tool-set-mutation channel is not (e.g., adding a tool that improves E's ranking while degrading τ³ Banking task completion), the baseline produces a successful evolution run because its fixed oracle never probes the tool-set channel in isolation. Our method instead constructs beneficial/harmful tool-set pairs and reads E's ranking on them, so we expect a clear split: near-ceiling κ on prompt families, substantially lower κ on tool-set families, and a PROXY-DIVERGENT verdict concentrated on exactly the recipes whose tool-set channels diverge. The Phase 2b audit is expected to show audit[fam] ≥ 0.5 on the prompt families (author monotonicity holds) and audit[fam] < 0.5 on the tool-set families where the author's 'add a tool' belief is mis-specified, so the certificate reports both the divergence and the specification-error rate that explains it.

**vs. Score-correlation agreement (Pearson on E scores)** — On a recipe where E's raw scores are not comparable across candidates (the RSIHub oracle's known property), the baseline produces noisy, occasionally negative correlations that flip sign across candidate subsets because score magnitudes are uncalibrated. Our method instead reads only the ranking of a constructed pair, so we expect CIC's κ to be stable across subsets where score-correlation agreement is unstable, with the gap widest on recipes whose candidate scores span a narrow range.

**vs. Mean-over-families aggregation** — On a recipe that is identifiable on three of four declared families and divergent on the fourth (a dead or mis-specified channel), the baseline produces an IDENTIFIABLE verdict because the mean is pulled up by the three faithful families. Our method instead takes the minimum over reliable families, so we expect mean and min to agree on recipes with uniform family behavior but to diverge sharply on the minority of recipes with one bad channel — and the divergent-family identity CIC reports should match the family where an independent behavioral check shows proxy optimization.

**vs. CIC without Phase 2b specification audit (author belief only)** — On a recipe where the author's `apply_beneficial` is mis-specified (e.g., longer prompt degrades tool-call precision), the no-audit variant computes κ against the author's belief and reports PROXY-DIVERGENT even if E faithfully tracks O, because the reference label is wrong. Our method instead measures audit[fam] on the 16-pair calibration subset, flags the family SPECIFICATION-UNRELIABLE when audit[fam] < 0.5, and excludes it from κ_global, so we expect the no-audit variant to produce false PROXY-DIVERGENT verdicts on exactly the families where audit[fam] < 0.5, while the audited CIC reports the specification-error rate and either certifies the recipe on its reliable families or flags it with the correct attribution.

### What would falsify this idea

If CIC's PROXY-DIVERGENT verdicts on named families do not predict any held-out behavioral divergence (i.e., recipes CIC flags show the same objective improvement under independent evaluation as recipes it certifies), or if removing the construction guard leaves κ_global and the verdict set unchanged, or if the Phase 2b specification audit does not separate families where the author's monotonicity claim fails from families where E diverges (i.e., audit[fam] < 0.5 families show no difference in verdict attribution from audit[fam] ≥ 0.5 families), the claim that manufactured counterfactual labels certify evaluator-fitness identifiability is wrong.

## References

1. RSIHub: A Composable Framework for Agent-Evolution Methods
2. The OpenHands Software Agent SDK: A Composable and Extensible Foundation for Production Agents
3. Darwin Godel Machine: Open-Ended Evolution of Self-Improving Agents
4. Agentic Context Engineering: Evolving Contexts for Self-Improving Language Models
