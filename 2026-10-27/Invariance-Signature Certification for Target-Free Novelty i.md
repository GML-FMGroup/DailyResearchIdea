# Invariance-Signature Certification for Target-Free Novelty in LLM Equation Discovery

## Motivation

Current LLM scientific-discovery benchmarks such as LLM-SRBench and ScienceAgentBench score every candidate artifact against a provided ground-truth target, so the only measurable notion of 'good' is distance-to-target and novelty beyond the target distribution becomes structurally unmeasurable — there is no target to be far from. This is the bottleneck the field's trajectory now creates: as single-artifact, automatically-scored benchmarks scale, the certification signal they produce is definitionally bounded by the reference answer, and a candidate that is genuinely more constrained than anything in the reference distribution is indistinguishable from a candidate that is merely wrong. The root cause is that scoring collapses the artifact to a scalar distance, discarding the artifact's internal, target-free structure.

## Key Insight

A candidate equation carries a computable invariance signature — the subgroup of input transformations under which its prediction is invariant — and this signature is strictly more informative than distance-to-target because two artifacts matching the same target can have different signatures, so novelty can be certified as a proper refinement of the reference class's recoverable invariance group without any ground-truth target.

## Method

**What it is.** INVARIA (INVariance-Refinement Artifact certification) takes as input a candidate artifact `f` (an LLM-produced closed-form equation), the input distribution `D` (a controlled scientific dataset with known transformation generators), and a reference artifact set `R` drawn from the training/reference distribution. It outputs a certification verdict: `NOVEL` iff the invariance subgroup certified for `f` is a strict refinement of the invariance subgroup certified for every artifact in `R`, else `NOT_NOVEL` with the witness generator that breaks the refinement.

**Load-bearing assumption (stated explicitly).** The certification claim rests on the assumption that a candidate's invariance signature being a strict refinement of the reference class's bitwise-AND invariance envelope implies the candidate is a genuinely more constrained, non-reachable hypothesis — i.e., that preserving an extra generator is a semantically meaningful constraint rather than an artifact of the finite generator bank, tolerance threshold, or functional degeneracy. This assumption is not verified by the signature computation alone; it is operationalized by the predictive-adequacy check in Phase 4b, which rejects candidates whose extra invariance degrades held-out fit relative to the reference class. The method therefore certifies 'refinement subject to target-free predictive adequacy', not refinement alone.

**How it works.**

```pseudocode
INPUT: candidate f, dataset D ~ (X, y), reference set R, generator bank G = {g_1..g_K},
       tolerance tau=0.01 (relative), sample budget n=2000, confidence delta=0.05,
       held-out split fraction rho=0.2 (stratified by problem), seed protocol: fixed seed s=0
       for all sampling; resampling for Phase 4 uses seeds s+1..s+K

PHASE 1 — Generator bank construction (target-free, uses X only)
  for each g_k in G:
      verify g_k acts on X (e.g. x -> a*x, x -> x + b, x_i <-> x_j, x -> 1/x)
      keep g_k only if it is a measurable automorphism of the marginal support of X
      # this uses X alone, never y
  # bank size K=4 per domain: scaling (a in {0.5,2}), translation (b in {-1,1}*std(X)),
  # permutation (all pairs i<j), inversion (x -> 1/x, applied only where |x|>0.1*std(X))

PHASE 2 — Invariance signature of an artifact h
  for each g_k in G:
      draw n samples x ~ D_X
      compute residual r_k = E_x[ |h(g_k(x)) - h(x)| / (|h(x)| + eps) ]
      # h is invariant under g_k iff r_k <= tau
      sig_k(h) = 1[ r_k <= tau ]
  return Sig(h) = (sig_1..sig_K)   # a bit-vector = finite subgroup proxy

PHASE 3 — Reference class envelope
  for each h_r in R:
      Sig_r = Sig(h_r)
  Sig_ref = bitwise-AND over r of Sig_r     # generators invariant for ALL references
  # Sig_ref is the recoverable invariance group of the reference class

PHASE 4 — Refinement test on candidate
  Sig_f = Sig(f)
  if (Sig_f AND Sig_ref) != Sig_ref:
      return NOT_NOVEL, witness = argmin_k (Sig_ref_k=1, Sig_f_k=0)
      # candidate breaks a symmetry the reference class preserves -> not a refinement
  extra = Sig_f AND (NOT Sig_ref)          # generators f preserves, references do not
  if popcount(extra) == 0:
      return NOT_NOVEL, witness = None
  # certify each extra generator with a held-out split to control false positives
  for k in extra:
      re-estimate r_k on a fresh sample of size n; keep k only if r_k <= tau at level delta
  if extra is empty after re-estimation:
      return NOT_NOVEL, witness = None

PHASE 4b — Target-free predictive adequacy check (gates the refinement verdict)
  # split D into D_fit (80%) and D_hold (20%) stratified by problem; D_hold is never used in Phases 1-4
  # compute reference-class best fit on D_hold: for each h_r in R, MSE_r = mean((h_r(X_hold)-y_hold)^2)
  # best_ref = min_r MSE_r
  # compute candidate fit on D_hold: MSE_f = mean((f(X_hold)-y_hold)^2)
  # require MSE_f <= best_ref * (1 + gamma), with gamma=0.05 (5% relative slack)
  if MSE_f > best_ref * (1 + gamma):
      return NOT_NOVEL, witness = 'predictive_inadequacy'  # extra symmetry degrades fit
  # non-degeneracy check on f's output variance (guards constant-function collapse)
  if Var_x[f(x)] < 1e-6 * Var_x[y]:
      return NOT_NOVEL, witness = 'degenerate_collapse'
  return NOVEL, certified_generators = surviving extra
```

**Why this design.** We chose a finite generator bank with a bit-vector subgroup proxy over a continuous Lie-algebra estimate because the reference class in LLM-SRBench-style data has known, discrete transformation structure (scaling, translation, permutation, inversion), accepting the cost that signatures are only as expressive as the bank. We chose a bitwise-AND envelope over a per-reference nearest-neighbor match because the certification claim is about the *whole* reference class, not one neighbor — a candidate must refine every reference simultaneously, accepting the cost that a single permissive reference shrinks `Sig_ref` and makes certification easier. We chose a held-out re-estimation of each extra generator over a single-pass signature because a false-positive invariance bit is the dominant failure mode of the test, accepting the cost of a second sampling pass. We added Phase 4b's predictive-adequacy check because the load-bearing assumption — that an extra invariance bit implies a more constrained hypothesis — is not guaranteed by the signature alone: a wrong-but-symmetric artifact can satisfy refinement while degrading fit, so we require the candidate's held-out MSE to be no worse than the reference class's best fit (within 5% slack), accepting the cost that a genuinely novel artifact with slightly worse fit on the finite held-out split is rejected. A domain expert would not describe INVARIA as a variant of equation-matching or of LLM-SRBench scoring: those compare the artifact to a target value, whereas INVARIA never reads `y` in Phases 1–4 and certifies a property (a strict subgroup refinement) that no distance-to-target metric can express; Phase 4b reads `y` only on a held-out split to gate the verdict, not to compute the signature.

**Why it measures what we claim.** The residual `r_k = E_x[|h(g_k(x)) - h(x)|/(|h(x)|+eps)]` measures *invariance of the artifact under generator g_k* because we assume the artifact is a deterministic function evaluated on the input support and the generator is an automorphism of that support, so the expectation is well-defined and finite; this assumption fails when `g_k` pushes samples off the support (e.g. `x -> 1/x` near zero), in which case `r_k` reflects extrapolation error rather than invariance. The bitwise-AND envelope `Sig_ref` measures *the recoverable invariance group of the reference class* because we assume every reference artifact is a faithful function on `D_X` and the class is closed under the bank, so the intersection is the maximal subgroup all references share; this assumption fails when the reference set is small or heterogeneous, in which case `Sig_ref` reflects the least-constrained reference rather than the class's true shared structure. The refinement test `(Sig_f AND Sig_ref) == Sig_ref` with nonempty `extra` measures *strict novelty relative to the reference class* because we assume a finer invariance group implies a strictly more constrained hypothesis class, so a candidate preserving a symmetry no reference preserves is not reachable by the reference distribution's artifacts; this assumption fails when an extra generator is a trivial consequence of the functional form (e.g. any constant function is invariant under everything) or when a wrong-but-symmetric artifact satisfies refinement, in which case the verdict reflects degenerate collapse or spurious symmetry rather than genuine novelty, which is why Phase 4b's held-out predictive-adequacy check (MSE_f <= best_ref * 1.05) and a non-degeneracy check on `f`'s output variance gate the final verdict.

## Contribution

(1) INVARIA, a target-free novelty-certification mechanism that scores LLM-produced scientific artifacts by the strict refinement of their invariance subgroup over a reference class's recoverable invariance group, requiring no ground-truth target at certification time. (2) A design principle — that invariance signature is strictly more informative than distance-to-target because same-target artifacts can differ in signature — operationalized as a bitwise-AND reference envelope plus a held-out refinement test that controls false-positive invariance bits. (3) A certification protocol and witness-generator output that turns a binary novelty verdict into an auditable artifact, applicable to any controlled scientific dataset with a known generator bank.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | LLM-SRBench (239 problems, 4 domains) | Known discrete transformation structure; reference artifacts available |
| Primary metric | Strict-refinement certification rate vs. reference class | Directly operationalizes the novelty claim |
| Baseline | LLM-SRBench target-fit scoring | Tests whether distance-to-target captures subgroup refinement |
| Baseline | Nearest-neighbor invariance match | Tests whether whole-class envelope is necessary |
| Baseline | Single-pass signature (no held-out re-estimation) | Tests whether false-positive gating drives verdicts |
| Ablation-of-ours | INVARIA without non-degeneracy gate | Isolates degenerate-collapse false positives |
| Ablation-of-ours | INVARIA without Phase 4b predictive-adequacy check | Isolates wrong-but-symmetric false positives (the load-bearing assumption) |

### Why this setup validates the claim

The central claim is that novelty is a property of a strict invariance-subgroup refinement relative to the whole reference class, and that no target-distance metric can express it. LLM-SRBench provides the controlled structure — scaling, translation, permutation, inversion — needed for the generator bank to be meaningful, plus a reference artifact set whose shared invariance group is recoverable. The target-fit baseline tests the sub-claim that distance-to-target is blind to subgroup refinement: if INVARIA and target-fit rank candidates identically, the refinement property adds nothing. The nearest-neighbor baseline tests the sub-claim that the whole-class bitwise-AND envelope is required rather than a single permissive reference; a permissive neighbor shrinks the envelope and inflates novelty, so if nearest-neighbor matches INVARIA's verdicts on heterogeneous reference subsets, the envelope is unnecessary. The single-pass ablation tests the sub-claim that held-out re-estimation is what suppresses false-positive invariance bits; if verdicts are unchanged without it, the gating is decorative. The Phase 4b ablation tests the load-bearing assumption directly: if removing the predictive-adequacy check does not increase false-positive certifications on wrong-but-symmetric candidates, the assumption is not doing the work attributed to it. The primary metric — strict-refinement certification rate — is the right detector because it is computed without reading `y` in Phases 1–4, so any gain over target-fit cannot be attributed to target information; Phase 4b reads `y` only on a held-out split to gate the verdict, not to compute the signature. The non-degeneracy ablation isolates the degenerate-collapse failure mode the design explicitly guards against.

### Expected outcome and causal chain

**vs. LLM-SRBench target-fit scoring** — On a case where a candidate equation is target-accurate but preserves an extra symmetry (e.g. a scale-invariant form that fits the same curve), target-fit produces a high score indistinguishable from a genuinely novel artifact because its mechanism compares predictions to `y` and cannot see invariance structure. Our method instead computes the residual under each generator and certifies the extra generator via held-out re-estimation because it never reads `y` in Phases 1–4, so we expect a noticeable gap on the target-accurate-but-symmetric subset and parity on the subset where target-fit and refinement agree.

**vs. Nearest-neighbor invariance match** — On a case where the reference set contains one permissive artifact (e.g. a near-constant fit) whose signature is sparse, nearest-neighbor match adopts that sparse signature as the envelope because its mechanism matches against a single reference, so it certifies candidates as novel that a stricter envelope would reject. Our method instead takes the bitwise-AND over all references because the claim is about the whole class, so we expect nearest-neighbor to over-certify on heterogeneous reference subsets and for INVARIA to show a clear precision advantage concentrated exactly there.

**vs. Single-pass signature (no held-out re-estimation)** — On a case where a candidate's residual under an extra generator sits near the tolerance boundary, the single-pass variant flips the invariance bit on noise because it estimates `r_k` once with no confidence control, yielding spurious novelty verdicts. Our method instead re-estimates each extra generator on a fresh sample at level delta because a false-positive invariance bit is the dominant failure mode, so we expect the single-pass variant to show inflated certification on boundary-residual cases while INVARIA's verdicts remain stable across resampling.

**vs. INVARIA without non-degeneracy gate** — On a case where the candidate is a near-constant function, the ungated variant certifies it as novel because constant functions are invariant under every generator, inflating `extra` trivially. Our method instead applies a non-degeneracy check on output variance because degenerate collapse is not genuine novelty, so we expect the ablation to show spurious certifications concentrated on low-variance candidates while the gated method shows none.

**vs. INVARIA without Phase 4b predictive-adequacy check** — On a case where a candidate is wrong-but-symmetric (e.g. a spurious symmetry imposed by an LLM's inductive bias that degrades fit on held-out data), the ungated variant certifies it as novel because the extra invariance bit is present and the signature computation never reads `y`. Our method instead requires the candidate's held-out MSE to be no worse than the reference class's best fit (within 5% slack) because the load-bearing assumption — that an extra invariance bit implies a more constrained hypothesis — fails for wrong-but-symmetric artifacts, so we expect the ablation to show inflated certification on the wrong-but-symmetric subset while the gated method rejects those candidates and preserves certification on the right-and-more-constrained subset.

### What would falsify this idea

If INVARIA's certification advantage over target-fit and nearest-neighbor is uniform across all subsets rather than concentrated on the predicted failure subsets (target-accurate-but-symmetric cases, heterogeneous reference sets, boundary-residual cases, wrong-but-symmetric cases), then the refinement mechanism is not what drives the result and the central claim is wrong. Equivalently, if the held-out re-estimation, non-degeneracy, and Phase 4b predictive-adequacy ablations leave verdicts unchanged, the design's stated safeguards are not doing the work attributed to them.

## References

1. LLM-SRBench: A New Benchmark for Scientific Equation Discovery with Large Language Models
2. ScienceAgentBench: Toward Rigorous Assessment of Language Agents for Data-Driven Scientific Discovery
