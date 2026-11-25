# Chained Counterfactual Fidelity: Certifying Physical Video Editors Under Sequential Edit Composition

## Motivation

Existing physical video editing benchmarks evaluate each edit as an isolated unit: PhyCoVE scores single edits with a counterfactual fidelity metric on 533 cases but never asks whether a model that is faithful on edit A remains faithful when edit A's output becomes the source for edit B, and VEU-Bench similarly treats its 19 understanding tasks as independent classification/reasoning units. The structural root cause is that fidelity is defined as a per-case distance from one source state to one edited state, so the metric carries no information about how physical consequences propagate when the edited state is itself re-consumed as a source; a model can be certified faithful on every isolated case and still compound error or erase the first edit's consequences once edits are chained, a regime that real editing workflows routinely produce.

## Key Insight

Because physical counterfactual edits are state transitions, fidelity under composition is governed by a closure property — a model whose per-edit deviation from the true physical transition is bounded and whose transitions are applied to its own outputs must keep cumulative deviation bounded — so chained fidelity can be certified from single-edit fidelity plus a measured composition-residual, rather than assumed.

## Method

**What it is.** We introduce CCF (Chained Counterfactual Fidelity), a benchmark protocol and scoring rule that turns PhyCoVE's single-edit cases into chains of length L (default L=3) by feeding each edit's produced video into the next edit's source slot, and reports a composition-residual that quantifies how much a model's chained fidelity deviates from what its single-edit fidelity predicts, after controlling for distribution shift via a shift-matched control.

**(B) How it works.**

```pseudocode
INPUT: source video S0, edit script (e1, e2, ..., eL), material solver M_true, editor model E
OUTPUT: CCF score, composition residual R

# Phase 1: build the ground-truth chain (once per chain, human-verified)
S_true[0] = S0
for i in 1..L:
    S_true[i] = M_true.simulate(S_true[i-1], e_i)   # solver matched to material of S_true[i-1]
    human_check(S_true[i], S_true[i-1], e_i)        # reject if physical plausibility fails

# Phase 2: roll out the editor's own chain (closed loop)
S_hat[0] = S0
for i in 1..L:
    S_hat[i] = E.edit(S_hat[i-1], e_i)              # note: source is E's OWN previous output

# Phase 3: score per-step and cumulative fidelity (PhyCoVE counterfactual fidelity CF)
f[i]      = CF(S_hat[i], S_true[i])                 # per-step fidelity, i=1..L
f_open[i] = CF(E.edit(S_true[i-1], e_i), S_true[i]) # open-loop fidelity: same edit, TRUE source

# Phase 3b: shift-matched control (isolates OOD sensitivity from mispropagation)
# Measure how far E's own output drifted from the true chain state at step i-1
d[i-1] = D(S_hat[i-1], S_true[i-1])                 # D = LPIPS + optical-flow endpoint error, normalized to [0,1]
# Construct a ground-truth-perturbed source with the SAME measured drift magnitude
S_ctrl[i-1] = perturb(S_true[i-1], d[i-1])          # perturb = add Gaussian noise in pixel space then re-render with M_true to preserve material statistics
f_ctrl[i]  = CF(E.edit(S_ctrl[i-1], e_i), S_true[i]) # control: edit applied to a source with matched OOD distance

# Phase 4: composition residual and closure test
R = mean_i( f_ctrl[i] - f[i] )                      # degradation attributable to chaining, net of OOD sensitivity
tau = 0.10 * mean_i(f_ctrl[i])                      # tolerance, 10% of shift-matched control fidelity
closure_violated = (R > tau)                        # certified only if R <= tau
CCF = mean_i(f[i]) - lambda * max(0, R - tau),  lambda = 1.0
```

Hyperparameters: chain length L=3; tolerance fraction 0.10; penalty weight lambda=1.0; each chain drawn so consecutive edits share at least one material interaction pattern (rigid/elastic/fluid) so consequences can propagate; drift distance D uses LPIPS (AlexNet backbone) plus mean optical-flow endpoint error from RAFT, each normalized to [0,1] by the 95th percentile over the calibration set of 50 chains; perturbation magnitude d[i-1] is applied as per-pixel Gaussian noise with sigma = d[i-1] * 0.1 in [0,1] pixel space, followed by a single M_true re-render step to restore material statistics.

**(C) Why this design.** We chose a closed-loop rollout in which the editor consumes its own output over an open-loop re-scoring against ground-truth sources because only the closed loop exposes compounding error, accepting the cost that closed-loop chains require the editor to be run L times and can drift into states the human-verified ground truth never visits (we handle this by scoring against the true chain step-by-step rather than requiring state match, and by subtracting a shift-matched control that measures the editor's sensitivity to off-manifold sources of the same magnitude). We chose a solver-generated true chain with human verification over purely real recordings because real single-camera footage cannot supply the post-edit source state needed for the next edit, accepting the cost that simulated chains inherit solver fidelity limits already noted for PhyCoVE. We chose a fixed tolerance tau as a fraction of shift-matched control fidelity over a learned threshold because a learned gate would reintroduce the calibration problem and because the closure rule must be auditable by benchmark users, accepting the cost that tau=0.10 is a design choice we must ablate. This differs from standard benchmark aggregation (e.g., averaging PhyCoVE's per-case counterfactual fidelity) and from sequential-edit scoring in video generation benchmarks, which re-condition on ground-truth prefixes; a domain expert would not describe CCF as a variant of PhyCoVE scoring because the scored quantity is the residual between shift-matched control and closed-loop fidelity, which PhyCoVE's independent-case design cannot produce.

**(D) Why it measures what we claim.** The composition residual R measures propagation correctness because it assumes that the shift-matched control f_ctrl captures the editor's sensitivity to off-manifold sources of the same magnitude as its own drift, so any remaining drop in closed-loop fidelity f is attributable to error compounding across the chain; this assumption fails when the editor's output distribution shifts off the ground-truth manifold in a direction not captured by the pixel-space perturbation (e.g., a systematic material-property bias rather than generic noise), in which case R reflects residual distribution shift rather than physical mispropagation. The closure test (R <= tau) measures certification-under-composition because it assumes the per-edit transition error is approximately stationary across chain positions, so a bounded per-step residual implies bounded cumulative deviation; this assumption fails when early edits systematically corrupt a material quantity (e.g., fluid volume) that later edits amplify, in which case the test reflects amplification sensitivity rather than average fidelity. The per-step fidelity f[i] measures physical faithfulness at position i because it assumes the PhyCoVE counterfactual fidelity metric is comparable across chain positions; this assumption fails when later positions have fewer observable material interactions, in which case f[i] reflects visibility of physical change rather than fidelity.

## Contribution

(1) CCF, a compositionality protocol and closure rule that converts single-edit counterfactual benchmarks into closed-loop chained-edit evaluations with an explicit open-loop-vs-closed-loop composition residual. (2) The empirical design principle that single-edit fidelity is not predictive of chained fidelity, quantified as a measurable residual that separates error compounding from distribution shift. (3) A chained-edit benchmark split derived from PhyCoVE's material/interaction taxonomy, with solver-generated human-verified ground-truth chains and a tolerance-ablated certification rule.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | PhyCoVE chains (L=3, material-linked edits), stratified subset of 100 chains (50 material-linked, 50 materially independent) | Source cases already span rigid/elastic/fluid interactions; 100 chains gives 80% power to detect a 0.05 difference in R at alpha=0.05 (two-sided paired t-test, estimated SD=0.12 from a 10-chain pilot) |
| Primary metric | CCF score with composition residual R (shift-matched) | Directly scores open-vs-closed-loop degradation net of OOD sensitivity |
| Baseline | PhyCoVE per-case CF averaging | Standard aggregation ignores chaining entirely |
| Baseline | Open-loop re-scoring on true prefixes | Tests whether prefix conditioning alone suffices |
| Baseline | Sequential-edit generation baseline | Re-conditions on ground-truth prefixes, not own output |
| Baseline | Generic drift metric (LPIPS + flow endpoint error between S_hat[i] and S_true[i]) | Tests whether a standard autoregressive drift metric fails where CCF succeeds |
| Ablation-of-ours | CCF with tau=0 (no tolerance) | Isolates the tolerance term's contribution to scoring |
| Ablation-of-ours | CCF without shift-matched control (raw f_open - f) | Isolates the contribution of the OOD control to R |

### Why this setup validates the claim

The central claim is that closed-loop composition exposes error compounding that single-edit scoring cannot see, and that the shift-matched residual R isolates this compounding from mere per-edit weakness and from OOD drift. PhyCoVE per-case CF averaging tests the first sub-claim: if chaining did not matter, averaged single-edit fidelity should predict chained fidelity, so a large systematic R against this baseline falsifies the null that composition is neutral. The open-loop re-scoring baseline tests whether degradation is due to the editor consuming its own output versus simply being asked to edit harder states; if open-loop fidelity on true prefixes is high while closed-loop drops, the gap is attributable to self-consumption, which is exactly what R claims to measure. The sequential-edit generation baseline tests whether re-conditioning on ground-truth prefixes (the standard video-generation practice) hides the effect, establishing that our closed-loop choice is load-bearing. The generic drift metric baseline tests whether a standard autoregressive drift measure (LPIPS + flow error) already captures the effect; if CCF separates material-linked from materially independent chains while the generic drift metric does not, the residual is measuring propagation rather than generic drift. The tau=0 ablation isolates whether the tolerance term is doing real work or merely rescaling, confirming the closure test is a genuine gate rather than a cosmetic offset. The no-shift-control ablation isolates whether the OOD control is necessary: if raw f_open - f and shift-matched R produce the same ranking, the control adds nothing; if they diverge on chains with large d[i-1], the control is load-bearing. CCF is the right metric because it is the only one that pairs per-step fidelity with a shift-matched residual term, so a model can score well on average fidelity yet be flagged for compounding.

### Expected outcome and causal chain

**vs. PhyCoVE per-case CF averaging** — On a chain where a fluid-volume edit is followed by a rigid-object insertion, the baseline averages three independent per-case fidelities and reports a healthy mean because each single edit looks locally plausible. Its mechanism gap is that it never feeds an edited output back as a source, so compounding is invisible. Our method runs the closed loop and measures f_ctrl minus f, so we expect a noticeable R on material-linked chains (fluid-rigid, elastic-fluid) but near-parity on chains whose edits are materially independent, producing a concentrated gap rather than a uniform one.

**vs. Open-loop re-scoring on true prefixes** — On a chain where the editor's first output drifts off the ground-truth manifold (e.g., under-deforms an elastic solid), the baseline re-scores each edit against the true previous state and reports near-open-loop fidelity, masking the drift. Its mechanism gap is that it never lets the editor consume its own error. Our method instead scores the editor's own rollout against the true chain and subtracts a shift-matched control, so we expect a widening per-step gap f_ctrl minus f as position i increases, with the largest residual at the final step of material-linked chains.

**vs. Sequential-edit generation baseline** — On a chain where an early edit systematically corrupts a conserved quantity (fluid volume), the baseline re-conditions on ground-truth prefixes at every step and thus never amplifies the early error, so its fidelity stays flat across positions. Its mechanism gap is prefix conditioning, which severs the propagation path. Our method keeps the closed loop, so we expect the composition residual to spike precisely on chains whose consecutive edits share an amplified material quantity, while remaining within tolerance elsewhere.

**vs. Generic drift metric (LPIPS + flow endpoint error)** — On a chain where the editor's output drifts off-manifold but the physical consequences are correctly propagated (e.g., a rigid object moves slightly off-position but the subsequent fluid splash is physically consistent), the generic drift metric reports a large distance because it measures pixel-level deviation, not propagation correctness. Its mechanism gap is that it conflates OOD distance with mispropagation. Our method subtracts the shift-matched control, so we expect the generic drift metric to flag both material-linked and materially independent chains equally, while CCF flags only material-linked chains, producing a significant interaction effect (chain type x metric) that the generic drift metric cannot produce.

**vs. CCF with tau=0** — On chains with small but non-zero R, the tau=0 variant penalizes every deviation, so its ranking may be dominated by noise in R rather than by systematic compounding. Its mechanism gap is that it lacks a tolerance band. Our method with tau=0.10 should produce a ranking that is more stable across random chain draws (measured by Kendall tau between two disjoint 50-chain subsets), confirming the tolerance term reduces variance without changing the top-ranked models.

**vs. CCF without shift-matched control** — On chains where the editor's drift d[i-1] is large (e.g., elastic under-deformation), the raw residual f_open - f conflates OOD sensitivity with mispropagation, so it may flag models that are merely sensitive to off-manifold inputs. Its mechanism gap is the absence of the control. Our method with the shift-matched control should produce a smaller R for models whose closed-loop drop is explained by OOD sensitivity, so we expect the no-control variant to over-flag at least one model that the full CCF certifies, and the difference between the two R values to correlate with mean d[i-1] across chains (Pearson r > 0.5).

### What would falsify this idea

If the shift-matched composition residual R were uniformly small and statistically indistinguishable from zero across all material-linked chains, or if the gap between closed-loop and shift-matched control fidelity were no larger on material-linked chains than on materially independent chains, then the central claim that closed-loop chaining exposes compounding error would be wrong. Likewise, if the tau=0 ablation produced the same ranking as the full CCF, or if the no-shift-control ablation produced the same ranking as the full CCF, the closure test or the OOD control would be doing no diagnostic work.

## References

1. PhyCoVE: Benchmarking Physical Counterfactual Video Editing across Diverse Material Dynamics
