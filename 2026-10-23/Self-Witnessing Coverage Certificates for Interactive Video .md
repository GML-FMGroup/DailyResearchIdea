# Self-Witnessing Coverage Certificates for Interactive Video Generators

## Motivation

Existing interactive video generation benchmarks assume a finite authored script is a sufficient proxy for the space of unscripted interactions: StreamEval predefines 300 scripted test cases with fixed initial assets and requests, so its coverage of informative temporal separations is never verified and failures on unscripted requests are invisible. Symmetrically, training-side approaches such as FCP simply draw supervision from whatever temporal separations happen to exist in training videos, assuming uncurated data is representative. Across both lineages the structural gap is identical — coverage of the interaction space is assumed rather than closed — so a benchmark cannot distinguish a generator that genuinely represents a separation from one that merely interpolates within a narrow authored distribution.

## Key Insight

Coverage can be made a checkable closure property rather than an assumed proxy if the generator is forced to emit, for each interaction class it claims to reach, a self-generated pair of trajectories that diverge only under that class's informative separation — because the reachable set that can produce such witness pairs is exactly the set the generator can control, and closure of the witness set against a probe-derived partition of the interaction space is falsifiable without any human script.

## Method

```pseudocode
# SWC: Self-Witnessing Coverage certificate
# G: interactive video generator (produces trajectory T from initial asset A and request r)
# J: MLLM judge scoring (a) divergence magnitude and (b) separation-specificity
#
# Load-bearing assumption (stated explicitly):
#   The MLLM judge J's separation-specificity score s reliably attributes the
#   divergence between a generator-emitted trajectory pair to the declared
#   separation sep(r) alone, so that an accepted witness certifies the generator
#   genuinely represents that interaction class rather than merely producing a
#   judge-passing artifact. This assumption is load-bearing: if false, the
#   certificate certifies judge-passability, not reachability, and coverage_closure
#   inherits the unverified calibration. We therefore (i) replace the single
#   specificity judgment with a paired, within-witness counterfactual contrast
#   (seed-matched toggle control) that cancels judge bias, and (ii) calibrate J
#   against a human-labeled anchor set and report judge-human agreement as an
#   explicit reliability bound on coverage_closure.

INPUT: generator G, probe interaction space P (unscripted requests, no fixed assets), threshold tau, closure tolerance eps
       seed s0 (fixed per witness), human-labeled anchor set H (N_H=200 witness pairs with binary accept labels),
       judge-human agreement target kappa_min=0.6

PHASE 1 -- Witness emission (generator-intrinsic, no script):
  for each probe request r in P:
    A0 <- G.sample_initial_asset(r)              # generator's own reachable start state
    (T_a, T_b) <- G.emit_witness_pair(A0, r, seed=s0)     # generator proposes two trajectories
    sep(r) <- G.declare_separation(r)            # e.g. "object leaves frame" / "request arrives at t=k"
    # Seed-matched toggle control: same A0, same seed s0, sep(r) toggled off
    (T_a_ctrl, T_b_ctrl) <- G.emit_witness_pair(A0, r, seed=s0, sep_toggle=OFF)
    store witness W_r = (A0, r, sep(r), T_a, T_b, T_a_ctrl, T_b_ctrl, s0)

PHASE 2 -- Witness verification (judge, not human):
  for each W_r:
    d      <- J.divergence(T_a, T_b)                  # are they actually different?
    d_ctrl <- J.divergence(T_a_ctrl, T_b_ctrl)        # divergence under sep(r) toggled off
    s      <- J.separation_specificity(T_a, T_b, sep(r))  # is divergence attributable ONLY to sep(r)?
    # Paired contrast: require sep(r)-varying divergence to significantly exceed control
    accept W_r iff (d - d_ctrl) > tau AND s > tau

PHASE 3 -- Coverage closure (the new part):
  partition P into interaction classes C_1..C_m via probe-request clustering
    (k-means on sentence-embedding of r, fixed k=m, seed=0)
  for each class C_i:
    witnessed(C_i) <- exists accepted W_r with r in C_i and sep(r) informative for C_i
  coverage_closure <- |{i : witnessed(C_i)}| / m
  report residual = classes with no accepted witness (the generator's blind spots)
  # Judge calibration bound: compute Cohen's kappa between J's accept verdict and
  # human labels on H; report coverage_closure with reliability bound
  # [coverage_closure - (1-kappa), coverage_closure + (1-kappa)] clipped to [0,1].
  # If kappa < kappa_min, certificate is marked UNRELIABLE and not issued.
  CERTIFY G iff coverage_closure >= 1 - eps AND kappa >= kappa_min
```

**Why this design.** We chose generator-emitted witness pairs over human-authored test cases (as in StreamEval) because authored cases cannot certify reachability — a scripted request may lie outside the generator's own distribution, making a pass uninformative; the cost is that the generator may emit easy witnesses, which Phase 2's separation-specificity judge is designed to penalize. We chose an MLLM judge over pixel-difference metrics because informative temporal separations (e.g., state change under a delayed request) are semantic, not photometric; the cost is judge noise, mitigated by requiring both divergence and specificity to clear tau, and further mitigated by the seed-matched toggle control (same A0, same seed, sep(r) toggled) which turns the specificity check into a paired within-witness contrast that cancels judge bias. We chose closure over a fixed count of witnesses because a generator can trivially emit many redundant witnesses for one class; partitioning probes into interaction classes and requiring one accepted witness per class makes redundancy non-credit. A domain expert would not describe SWC as a variant of StreamEval: StreamEval samples requests from an authored script and scores execution, whereas SWC requires the generator to *produce the discriminating instance itself* and scores whether the produced witness set closes over the probe space — the object being measured is the generator's reachable separation set, not its accuracy on fixed cases. Nor is it FCP: FCP draws separations from training data, while SWC makes the generator declare and instantiate them.

**Why it measures what we claim.** `coverage_closure` measures the fraction of interaction classes for which the generator can produce a witness pair whose divergence is attributable to the declared separation, because we assume the seed-matched toggle control isolates sep(r) as the only varying factor; this assumption fails when the generator emits a witness whose divergence is driven by an uncontrolled factor (e.g., different random seeds) that the judge misattributes to sep(r), in which case `coverage_closure` reflects judge-passability on confounded pairs rather than genuine reachability. `separation-specificity s` measures causal attribution of divergence to sep(r), because we assume the generator cannot produce a non-sep(r) confound correlated with sep(r) in the witness pair; this assumption fails when the generator's control pair (sep toggled off) still diverges due to an orthogonal factor, in which case `s` reflects surface-level difference rather than separation-specific control. The judge-human agreement `kappa` measures the reliability of J's accept verdict, because we assume the human-labeled anchor set H is representative of the witness distribution; this assumption fails when H is drawn from a different request distribution than P, in which case `kappa` reflects agreement on H but not on the certified witnesses.

## Contribution

(1) A generator-intrinsic coverage-closure certificate (SWC) in which the generator emits temporal-separation witness pairs and an MLLM judge verifies divergence and separation-specificity, turning coverage from an assumed proxy into a falsifiable closure condition over a probe-derived interaction-class partition. (2) The design principle that separation-specificity — not raw divergence — is the load-bearing verification signal, because divergence alone is satisfiable by trivial stochasticity while specificity ties each witness to the interaction class it is meant to cover. (3) A residual report that localizes a generator's blind interaction classes, giving a diagnostic beyond a single aggregate score.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|-----------------------|
| Dataset | Probe interaction space P: 50 hand-curated unscripted requests, no fixed assets; clustered into m=10 interaction classes via k-means (k=10, seed=0) on sentence embeddings | Tests generator's own reachable states, not authored cases |
| Primary metric | Coverage closure (fraction of interaction classes with accepted witness), reported with reliability bound [closure-(1-kappa), closure+(1-kappa)] | Directly measures blind spots over probe space |
| Baseline 1 | StreamEval | Authored-script accuracy baseline, no reachability certification |
| Baseline 2 | FCP (Future Condition Prediction) | Training-data separations, not generator-declared |
| Baseline 3 | Self-Forcing++ | Long-horizon quality, no coverage closure |
| Ablation-of-ours | SWC without Phase 3 (no closure) | Isolates effect of class partitioning and closure |
| Ablation-of-ours | SWC without seed-matched toggle control (single specificity judgment) | Isolates effect of paired contrast on judge bias |
| Calibration set | H: 200 human-labeled witness pairs (binary accept) | Calibrates judge; reports kappa and false-accept rate |
| Oracle study | Synthetic trajectories with injected known object-departure at t=k (N=100) | Measures judge specificity precision/recall against ground truth |

### Why this setup validates the claim

The central claim is that a generator can self-certify coverage of its reachable separation set, and that this certification reveals blind spots that scripted or training-data-driven evaluations miss. To test this, we evaluate on a probe interaction space P of 50 hand-curated unscripted requests with no fixed assets or authored scripts, forcing every method to operate on the generator's own sampled start states. The primary metric, coverage closure, is the fraction of m=10 interaction classes (k-means on request embeddings, fixed k=10, seed=0) for which at least one accepted witness exists. This metric is falsifiable: if a generator has a blind spot, some class will lack an accepted witness, and closure will drop. We report coverage_closure with an explicit reliability bound derived from judge-human agreement kappa on the 200-pair anchor set H, and we run a controlled synthetic study (N=100 injected object-departures at known t=k) to measure judge specificity precision/recall and the false-accept rate that would inflate coverage_closure. StreamEval tests the sub-claim that authored cases cannot certify reachability: it may score high on its 300 scripted cases while missing classes where the generator cannot produce a discriminating instance. FCP tests the sub-claim that separations drawn from training data do not cover the generator's declared separation set. Self-Forcing++ tests the sub-claim that long-horizon quality does not imply coverage; a model can generate long videos but still fail to instantiate specific temporal separations. The ablation (SWC without closure) isolates the contribution of Phase 3: without partitioning and requiring one witness per class, redundant easy witnesses inflate apparent coverage. The second ablation (SWC without seed-matched toggle control) isolates the contribution of the paired contrast in Phase 2: without it, judge bias is uncancelled and false-accept rate should rise. Thus, the combination forms a falsifiable test: if our method's coverage closure is not higher than baselines specifically on classes where baselines fail to produce accepted witnesses, the claim that generator-emitted witnesses certify reachability is wrong.

### Expected outcome and causal chain

**vs. StreamEval** — On a case where the probe requests a delayed object departure (e.g., "object leaves frame at t=5"), StreamEval produces a pass/fail score on its authored script but cannot certify that the generator can reach that state from its own initial asset, because its scripted request may lie outside the generator's distribution. Our method instead requires the generator to emit a witness pair (two trajectories from its own sampled A0) and a judge to verify separation specificity via the seed-matched toggle control, so we expect StreamEval to show high accuracy on scripted cases but low coverage closure on probe classes where the generator's reachable set does not include the scripted separation, yielding a noticeable gap on those classes.

**vs. FCP** — On a case where the generator must declare a separation like "request arrives at t=k" and instantiate it, FCP draws separations from training data and cannot make the generator produce a discriminating instance for its own reachable states. The baseline therefore produces separations that may be unreachable or non-informative for the generator, leading to low coverage closure on classes requiring generator-specific separations. Our method makes the generator declare and emit the witness, so we expect higher coverage closure on classes where FCP's training-data separations are misaligned, but parity on classes where training data happens to cover the generator's reachable set.

**vs. Self-Forcing++** — On a case requiring a specific temporal separation (e.g., "object leaves frame") after a long horizon, Self-Forcing++ focuses on temporal consistency and quality degradation but does not test whether the generator can instantiate that separation from its own start state. It may produce long, consistent videos but fail to cover classes that require precise state changes, resulting in low coverage closure on those classes. Our method explicitly requires an accepted witness per class, so we expect a clear gap on classes involving rare or delayed separations, while both methods may show similar coverage on simple, frequently occurring classes.

**vs. SWC without Phase 3** — On a case where the generator can trivially emit many redundant easy witnesses for one class, the ablation without closure counts all accepted witnesses without partitioning, so apparent coverage is inflated by redundancy. Our method requires one accepted witness per interaction class, so we expect the ablation to show higher raw witness count but lower coverage closure on classes that lack a discriminating instance, confirming that class partitioning and closure are load-bearing.

**vs. SWC without seed-matched toggle control** — On a case where the generator emits a witness whose divergence is driven by an uncontrolled factor (e.g., different random seeds) that the judge misattributes to sep(r), the ablation without the toggle control accepts the confounded witness, inflating coverage_closure. Our method requires the sep(r)-varying pair to significantly exceed the seed-matched control pair, so we expect the ablation to show a higher false-accept rate on the synthetic oracle study and a higher coverage_closure that is not supported by human labels, confirming that the paired contrast cancels judge bias.

### What would falsify this idea

If our method's coverage closure gain over baselines is uniform across all interaction classes rather than concentrated on classes where baselines fail to produce accepted witnesses (i.e., the predicted blind spots), then the claim that generator-emitted witnesses certify reachability is wrong. Similarly, if the ablation without closure achieves the same coverage closure as full SWC, or if the ablation without the seed-matched toggle control achieves the same false-accept rate on the synthetic oracle study, then Phase 3's class partitioning and Phase 2's paired contrast contribute nothing, falsifying the core mechanism.

## References

1. StreamEval: Evaluating Interactive Streaming Video Generation Models
2. Looking Ahead: Modeling Future Evolution in Autoregressive Video Generation
3. Self-Forcing++: Towards Minute-Scale High-Quality Video Generation
4. Diffusion Forcing: Next-token Prediction Meets Full-Sequence Diffusion
5. LTX-Video: Realtime Video Latent Diffusion
6. GAIA-1: A Generative World Model for Autonomous Driving
7. AR-Diffusion: Auto-Regressive Diffusion Model for Text Generation
8. Phenaki: Variable Length Video Generation From Open Domain Textual Description
