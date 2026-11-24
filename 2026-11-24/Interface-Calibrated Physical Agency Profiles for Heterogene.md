# Interface-Calibrated Physical Agency Profiles for Heterogeneous Robot-Learning Agents

## Motivation

WorldLeague compares VLMs, VLAs, and WAMs on a shared 60-task suite, but its Interactive Score is computed from trajectories produced under structurally different control interfaces: VLMs choose discrete targets and motions through a single interface, while VLAs and WAMs act through native continuous controls. Because interface choice determines the reachable trajectory distribution independently of the agent's underlying capability, the resulting Physical Agency Profile confounds interface-induced score shifts with genuine capability differences, so a low score on a capability dimension cannot be attributed to that dimension rather than to the interface. This is a benchmark-level identifiability failure rather than a property of any single evaluated system, and it recurs in any benchmark that admits heterogeneous action interfaces.

## Key Insight

If interface effects enter the trajectory-progress statistic through a dimension-specific additive offset that is shared across all agents using that interface, then a within-interface, dimension-wise centering removes the offset exactly while leaving between-agent capability contrasts intact, making capability-dimension attribution identifiable without ever running a cross-interface comparison.

## Method

**What it is.** We introduce ICAP (Interface-Calibrated Agency Profiles), a post-hoc calibration layer over WorldLeague's Physical Agency Profile. Its inputs are the per-episode trajectory progress scores and binary success indicators from all evaluated agents, tagged by agent, interface type (VLM-discrete, VLA-native, WAM-native), and capability dimension; its outputs are calibrated per-dimension Interactive Scores and success rates, an interface-effect estimate with uncertainty, a per-agent interaction residual term, and a per-agent attribution vector over the six capability dimensions.

**Load-bearing assumption (stated explicitly).** ICAP assumes that interface effects on WorldLeague's per-dimension Interactive Score are additive, dimension-specific, and agent-independent: p(a,k,d) = c(a,d) + b(k,d) + noise. In plain language, this means that switching an agent from one interface to another shifts its progress score by an amount that depends only on the interface and the capability dimension, not on the agent's own capability. This assumption fails when an interface interacts with agent capability (e.g., a strong VLA benefits more from native continuous controls than a weak one), in which case the additive offset absorbs part of the capability contrast and calibrated scores understate between-agent differences. Because interface is perfectly nested with agent family in WorldLeague (all VLMs use the discrete interface, all VLAs/WAMs use native controls), b(k,d) and c(a,d) are not separately identified from the 13 agents in 3 groups; ICAP therefore treats the additive-offset model as a working hypothesis to be tested, not as a guaranteed decomposition.

**How it works.**

```pseudocode
INPUT: episodes E = {(agent a, interface k, dimension d, task t, progress p, success s)}
       dimensions D = 6, interfaces K = {VLM, VLA, WAM}

PHASE 1 — Interface-effect estimation (within-interface centering)
  for each interface k and dimension d:
      mu[k,d]  = mean over episodes with interface k, dimension d of p
      sd[k,d]  = robust scale (MAD) of p over same episodes
  interface_offset[k,d] = mu[k,d] - mean over k' of mu[k',d]   # offset relative to grand mean

PHASE 2 — Per-agent calibrated scores
  for each agent a and dimension d:
      p_cal(a,d) = mean over a's episodes in d of (p - interface_offset[interface(a), d])
      s_cal(a,d) = mean over a's episodes in d of s   # success is interface-invariant by construction
  Physical Agency Profile(a) = {p_cal(a,d), s_cal(a,d)}_{d=1..6}

PHASE 3 — Attribution identifiability test
  for each agent a:
      rank dimensions by p_cal(a,d) descending
      attribution(a) = argmin_d p_cal(a,d)          # weakest calibrated dimension
  identifiability statistic:
      I = between-agent variance of attribution labels /
          within-agent bootstrap variance of attribution labels
  report I with 95% bootstrap CI (1000 resamples over tasks)

PHASE 4 — Interface-effect significance
  permutation test: shuffle interface labels within dimension, recompute
  interface_offset, repeat 1000x; report p-value for H0: offset = 0

PHASE 5 — Interaction residual and sensitivity bound
  # Fit a two-way agent-by-interface model per dimension where possible:
  # p(a,k,d) = c(a,d) + b(k,d) + r(a,k,d) + noise
  # Because no agent spans interfaces in WorldLeague, r(a,k,d) is not directly estimable.
  # Instead, report a bounded sensitivity analysis:
  #   For each dimension d, compute the range of capability contrasts consistent with
  #   arbitrary agent-by-interface interactions bounded by the observed within-interface
  #   variance: |r(a,k,d)| <= 2 * sd[k,d].
  #   Output: interval [p_cal_low(a,d), p_cal_high(a,d)] for each agent and dimension.
  # If a held-out interface-swap subset becomes available (agents evaluated under >1 interface),
  # estimate r(a,k,d) directly as the residual from the additive model and report its variance component.

PHASE 6 — Jointly limiting dimensions (replaces argmin when ties occur)
  For each agent a:
      Let m = min_d p_cal(a,d).
      Define the set of jointly limiting dimensions J(a) = {d : p_cal(a,d) <= m + delta},
      where delta = 0.1 * (max_d p_cal(a,d) - min_d p_cal(a,d)) (10% of the calibrated range).
      If |J(a)| > 1, report J(a) as a set rather than a single argmin label.
```

Hyperparameters: MAD scale (not std) for robustness to heavy-tailed progress; 1000 bootstrap/permutation resamples; attribution defined as argmin over calibrated dimension scores; sensitivity bound uses 2×MAD as the interaction cap; tie threshold delta = 10% of the calibrated range.

**Why this design.** We chose within-interface centering over a learned interface-embedding model because the benchmark supplies only 13 agents across 3 interfaces, which is far too few to fit a flexible embedding without overfitting, accepting the cost that we can only remove additive interface offsets and not interface-by-task interactions. We chose to calibrate Interactive Score only and leave physical success rate uncalibrated because success is a binary goal-achievement event defined identically across interfaces, whereas progress is a continuous trajectory statistic whose scale depends on the action parameterization; the trade-off is that success retains any genuine interface difficulty differences, which we report separately rather than silently correcting. We chose a permutation test on interface labels over a parametric ANOVA because the number of interface groups is three and normality of progress residuals is not established, accepting lower power in exchange for a valid finite-sample null. We chose argmin-based attribution over a regression-based attribution because argmin is invariant to any monotone rescaling of calibrated scores, accepting that it discards magnitude information about how far the weakest dimension falls below the next. A domain expert would not describe ICAP as a variant of latent-disentanglement heads (e.g., the information-bottleneck redundancy reduction of the WorldLeague-adjacent BC work) because ICAP never asserts statistical independence between interface and capability; it exploits a stated additive-offset invariance and verifies it with a permutation null, which is a different structural claim. The addition of Phase 5 (interaction residual and sensitivity bound) and Phase 6 (jointly limiting dimensions) makes the load-bearing assumption explicit and provides a bounded fallback when the additive model is not identified, so the contribution is an identifiability audit rather than a claimed correction.

**Why this measures what we claim.** The computational quantity `interface_offset[k,d]` measures the interface-induced additive shift in Interactive Score because the named assumption is that interface effects are additive and dimension-specific but agent-independent, i.e., p(a,k,d) = c(a,d) + b(k,d) + noise; this assumption fails when an interface interacts with agent capability (a strong VLA benefits more from native controls than a weak one), in which case the offset absorbs part of the capability contrast and calibrated scores understate between-agent differences. The quantity `p_cal(a,d)` measures an agent's interface-free capability on dimension d because the same additivity assumption makes centering an exact projection onto the capability subspace; it fails under the same interaction, in which case p_cal reflects capability plus residual interface interaction. The quantity `attribution(a) = argmin_d p_cal(a,d)` measures which capability dimension a failure belongs to because the assumption that the weakest calibrated dimension is the binding constraint holds when dimensions are separable in the profile; it fails when two dimensions are jointly limiting (e.g., memory and planning co-fail), in which case argmin reports one of the tied dimensions arbitrarily. The identifiability statistic `I` measures whether dimension attribution is stable because the bootstrap-over-tasks assumption treats task sampling as the only source of attribution noise; it fails when interface assignment is confounded with task subset, in which case I overstates identifiability. The sensitivity bound `[p_cal_low(a,d), p_cal_high(a,d)]` measures the range of capability contrasts consistent with arbitrary agent-by-interface interactions bounded by the observed within-interface variance because it explicitly accounts for the non-identifiability of r(a,k,d); it fails when the true interaction exceeds the bound, in which case the interval understates the uncertainty.

## Contribution

(1) ICAP, a post-hoc interface-calibration layer for multi-interface robot-learning benchmarks that separates additive interface effects from agent capability contrasts via within-interface dimension-wise centering, with a permutation-based null for interface-effect significance. (2) An attribution-identifiability statistic that converts a Physical Agency Profile into a stable weakest-dimension label per agent, together with the explicit additivity assumption under which this label is valid. (3) An analysis protocol for WorldLeague-style benchmarks that reports interface effects, calibrated profiles, and attribution stability as three separate outputs rather than a single confounded Interactive Score.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | WorldLeague 60-task suite, 13 agents | Only source with interface-tagged agency profiles |
| Primary metric | Calibrated Interactive Score gap vs. raw | Directly tests additive-offset removal |
| Baseline 1 | Raw WorldLeague Physical Agency Profile | Uncalibrated interface offsets remain |
| Baseline 2 | Pooled-mean centering (no interface labels) | Tests whether interface tagging matters |
| Baseline 3 | Success-only ranking | Tests whether progress calibration adds signal |
| Ablation-of-ours | ICAP without Phase 4 permutation test | Isolates significance check contribution |
| Ablation-of-ours | ICAP without Phase 5 sensitivity bound | Isolates interaction-residual contribution |
| Ablation-of-ours | ICAP with argmin attribution only (no Phase 6 tie set) | Isolates jointly-limiting-dimension contribution |
| Diagnostic | Two-way agent-by-interface interaction variance component per dimension | Directly tests the additivity assumption |

### Why this setup validates the claim

The central claim is that interface-induced additive offsets in WorldLeague's Interactive Score distort cross-interface agent comparisons, and that within-interface centering removes them. Raw Physical Agency Profile is the right baseline because it embodies exactly the distortion we claim exists: if calibration is doing real work, raw profiles should show interface-clustered score shifts that ICAP removes. Pooled-mean centering is the sharper control: it removes a global mean but not per-interface, per-dimension offsets, so any gap between ICAP and pooled centering localizes the effect to interface-specific structure rather than generic re-centering. Success-only ranking checks that progress calibration is not redundant with the binary goal signal. The permutation test ablation separates "offsets exist" from "offsets are statistically distinguishable from noise," which matters because with 13 agents a spurious offset could mimic a real one. The sensitivity-bound ablation tests whether the bounded fallback changes conclusions when the additive model is not identified. The argmin-vs-tie-set ablation tests whether the separability assumption for attribution is load-bearing. The interaction-variance diagnostic directly measures the agent-by-interface interaction term that the additive model assumes away, grounding the central assumption in data rather than assertion. Calibrated Interactive Score gap is the right metric because it is the same quantity WorldLeague reports, so any improvement is directly interpretable as a correction to the published profile rather than a new construct.

### Expected outcome and causal chain

**vs. Raw WorldLeague Physical Agency Profile** — On a case where a VLM agent scores high on WorldShape progress purely because discrete interface progress is scaled generously, the raw profile ranks that VLM above a VLA that is genuinely more capable, because the raw score confounds interface scale with capability. ICAP subtracts the VLM interface offset per dimension, so the VLA's calibrated score overtakes the VLM. We expect a noticeable rank inversion on the two or three dimensions where interface offsets are largest, with near-parity on dimensions where offsets are small.

**vs. Pooled-mean centering (no interface labels)** — On a case where VLA-native and WAM-native interfaces share a similar global mean but differ per dimension, pooled centering leaves the per-dimension residuals intact, so an agent's weakest dimension is misattributed. ICAP's per-interface, per-dimension offsets remove those residuals, so attribution labels shift toward the true binding constraint. We expect ICAP's attribution to disagree with pooled centering on a minority of agents, concentrated on agents whose interface offsets vary across dimensions.

**vs. Success-only ranking** — On a case where two agents have identical success rates but different progress trajectories, success-only ranking declares them tied, because binary goal achievement discards trajectory shape. ICAP's calibrated Interactive Score separates them by removing interface scale, revealing a progress gap. We expect ICAP to break ties that success-only ranking leaves unresolved, with the largest separation on dimensions where success is rare and progress dominates the signal.

**vs. ICAP without Phase 4 permutation test** — Without the permutation test, ICAP reports calibrated scores but cannot distinguish a real interface offset from noise. We expect the permutation test to reject zero offset on the dimensions with the largest raw interface gaps, and to fail to reject on dimensions where offsets are small, confirming that the calibration is not uniformly applied.

**vs. ICAP without Phase 5 sensitivity bound** — Without the sensitivity bound, ICAP reports point estimates of calibrated scores that may overstate precision when the additive model is not identified. We expect the sensitivity intervals to be widest on dimensions where within-interface variance is large, and to contain the raw score on some agents, indicating that the additive correction is not robust for those cases.

**vs. ICAP with argmin attribution only (no Phase 6 tie set)** — Without the tie set, ICAP reports a single weakest dimension per agent even when two dimensions are jointly limiting. We expect the tie set to include more than one dimension for a non-trivial fraction of agents, particularly on dimensions where calibrated scores are close, and to change the attribution label for those agents relative to argmin-only.

**vs. Two-way interaction variance diagnostic** — The diagnostic fits a two-way agent-by-interface model per dimension and reports the interaction variance component. We expect the interaction variance to be non-zero but smaller than the main effects of agent and interface on most dimensions; if it is comparable to or larger than the main effects, the additive model is not supported and the sensitivity bound becomes the primary output.

### What would falsify this idea

If ICAP's calibrated scores produce rank orderings statistically indistinguishable from raw profiles across all six dimensions, or if the permutation test fails to reject zero interface offset on every dimension, then the additive-offset claim is wrong and the calibration layer is doing nothing. A second falsifier: if attribution labels from ICAP match pooled-mean centering on essentially all agents, the interface-specific structure we claim is not present in the data. A third falsifier: if the two-way interaction variance component is large relative to the main effects on most dimensions, the additive model is not supported and the sensitivity bound should be reported instead of point-calibrated scores.

## References

1. WorldLeague: Benchmarking Physical Agency
