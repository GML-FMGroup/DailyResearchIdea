# Re-Derived Perturbation Certificates for Agent-Authored Perception Code

## Motivation

Across tool-synthesis and policy-freezing loops, a code modification is licensed by its empirical success on a small frozen task sample and then frozen as if that sample were sufficient evidence of deployment safety — HexaShell builds tools on ten layouts of one seed and freezes them, and LightZeroNav freezes a VLM action pipeline validated on a small set of layouts. The structural failure is empirical-selection-as-certification: the same repair loop that edits the code also decides when to stop, so the sample it stopped on is correlated with the edit, and no quantity in either loop names the perturbation family (camera pose, lighting, layout shift, instruction paraphrase) the deployment will face. The result is a certificate that is silently invalidated by the very edits it licenses, because each edit shifts the code's behavior on perturbations that were never in the selection sample.

## Key Insight

If the perturbation family is constructed and held fixed by the agent before any edit is evaluated, then a distributional bound computed against that fixed family is invariant to the edit sequence and can be recomputed per-edit, so the certificate licenses each edit rather than being inherited from the sample that selected it.

## Method

**What it is.** CPG (Certified Perturbation Gate) takes as input an agent-authored perception/tool code edit `e` (from HexaShell-style log-driven repair), a task instance `x`, and a perturbation family `P` that the agent constructs once per task from the calibration metadata before any edit is scored. It outputs a scalar certificate `eps_hat(e)` = a distributional generalization bound on the edit's task error over `P`, plus an accept/reject decision. The certificate is recomputed from scratch for every edit; it is never inherited.

**Load-bearing assumption (stated explicitly).** The certificate is a valid upper bound on deployment error only if the frozen family `P` is a representative sample of the deployment perturbation distribution. Because `P` is authored by the same agent whose code is being certified, this assumption is not guaranteed by construction; it is checked empirically in Phase 1.5 against a small canary set of real deployment perturbations that the agent did not construct. If the canary check fails, the certificate is reported as `family-unvalidated` and the gate does not accept the edit.

**How it works.**

```pseudocode
# Phase 1: family construction (once per task, before any edit)
P = agent.construct_family(x, calibration_metadata,
                           axes=[camera_pose, lighting, layout_shift, instruction_paraphrase],
                           n_strata=8, n_per_stratum=16)
# P is a stratified grid; strata are quantile bins per axis, crossed by a
# Latin-hypercube over the 4 axes. P is FROZEN and hashed: h_P = sha256(P).

# Phase 1.5: family validation against an external canary set (once per task)
C = load_canary_set(x)   # 32 real deployment perturbations, human-curated,
                         # disjoint from P and from the agent's diagnostic probes
for each stratum s in P:
    err_C_s = (1/|C_s|) * sum_{c in C_s} 1[ task_fail(e_ref, c) ]
    # e_ref = the unedited baseline code for task x
    # predicted interval from P: [err_hat_s(e_ref) - margin_s, err_hat_s(e_ref) + margin_s]
    # margin_s = sqrt( (1/(2|S_eval_s|)) * log(2*8/delta) )
    if err_C_s not in predicted_interval_s:
        family_valid = False
family_valid = True if all strata pass
# If family_valid is False, every subsequent certificate is tagged
# 'family-unvalidated' and the gate rejects regardless of eps_hat.

# Phase 2: per-edit certificate (re-derived after every edit)
for each edit e proposed by repair loop:
    # (a) probe set: P itself, but with the agent's own diagnostic probes
    #     filtered out so selection and certification do not share samples
    S_eval = P \ S_diag(e)          # S_diag = instances the agent inspected to write e
    # (b) empirical risk on the frozen family
    err_hat(e) = (1/|S_eval|) * sum_{p in S_eval} 1[ task_fail(e, p) ]
    # (c) distributional bound: Hoeffding over the stratified grid, with
    #     a per-stratum variance correction from the two worst strata
    eps_hat(e) = err_hat(e) + sqrt( (1/(2|S_eval|)) * log(2*8/delta) )
                 + max_{s in strata} ( err_hat_s(e) - err_hat(e) )
    # delta = 0.05; the third term is the worst-stratum gap and is the
    # certificate's sensitivity to the axis the edit most damaged.
    # (d) gate: accept iff eps_hat(e) <= eps_target (0.10) AND h_P unchanged
    #     AND family_valid == True
    if eps_hat(e) <= eps_target and sha256(P) == h_P and family_valid:
        accept(e); log(e, eps_hat(e), h_P, family_valid)
    else:
        reject(e); return eps_hat(e), family_valid to agent as a repair signal
# The repair loop may edit code, but may NOT edit P; P is re-hashed each round.
```

A diagram of the data flow:

```
 calibration_metadata ──► [Phase 1: family constructor] ──► P (frozen, hashed)
                                                              │
 canary set C ──────────► [Phase 1.5: family validator] ──────► family_valid
                                                              │
 repair loop ──► edit e ──► [Phase 2a: diag-probe filter] ──► S_eval
                                                              │
                          [Phase 2b: stratified risk] ──► err_hat, err_hat_s
                                                              │
                          [Phase 2c: Hoeffding + worst-stratum gap] ──► eps_hat(e)
                                                              │
                          [Phase 2d: gate eps_hat <= 0.10, hash match, family_valid] ──► accept/reject
```

**Why this design.** We chose a frozen, agent-constructed stratified family over an agent-adaptive one because an adaptive family would let the same loop that edits the code also select the evidence, which is exactly the empirical-selection-as-certification failure we are targeting; the cost is that a family that is badly constructed in Phase 1 cannot be repaired mid-loop, so we accept lower coverage on axes the agent failed to anticipate. We chose a Hoeffding-plus-worst-stratum bound over a PAC-Bayes or Rademacher bound because the latter requires a prior over code edits that the agent's repair loop does not supply, and inventing one would smuggle in an unstated assumption; the cost is a looser bound (larger `eps_hat`) that rejects some edits a tighter bound would license. We chose to filter the agent's diagnostic probes `S_diag(e)` out of `S_eval` over reusing them, because the repair loop conditions on exactly those instances and reusing them would make `err_hat` optimistic in a way that is correlated with the edit; the cost is a smaller effective sample and thus a wider bound. We chose to re-derive `eps_hat` per edit rather than inherit a running certificate, because the whole point is that the certificate must survive the edits it licenses; the cost is `|P|` evaluations per edit, which is tractable only because `P` is small (8 strata x 16 = 128 instances). We chose to add Phase 1.5 (canary validation) because the load-bearing assumption that `P` is representative is not guaranteed when the agent authors `P`; the canary set is 32 real deployment perturbations the agent did not construct, and the check is a per-stratum interval test against the unedited baseline, so a misspecified family is flagged as `family-unvalidated` rather than silently licensing edits. This is not a variant of AlphaEvolve or AutoEval: AlphaEvolve scores candidates on a fixed evaluator and never names a perturbation family, and AutoEval automates the *execution* of a real-world evaluation but inherits whatever sample the user submits — neither constructs a family that the code edit is forbidden to touch, and neither re-derives a bound per edit.

**Why it measures what we claim.** `err_hat(e)` on the frozen family `P` measures the edit's deployment error over the perturbation axes the deployment will face because `P` is a stratified sample of exactly those axes drawn before the edit was written, under the assumption that the four named axes (camera pose, lighting, layout shift, paraphrase) are the axes along which deployment varies; this assumption fails when deployment exhibits a fifth axis (e.g., occlusion by a new object class), in which case `err_hat` reflects error on a family that is merely *named* rather than *complete*. The worst-stratum gap `max_s (err_hat_s - err_hat)` measures the edit's *axis-specific fragility* because a stratum is a quantile bin of one perturbation axis, under the assumption that within-stratum variation is small relative to between-stratum variation; this assumption fails when an axis is non-monotone within a bin (e.g., lighting that is easy at both extremes but hard at mid-range), in which case the gap reflects binning noise rather than fragility. The Hoeffding term `sqrt((1/(2|S_eval|)) log(2*8/delta))` measures the *sampling uncertainty* of the certificate because the grid is drawn i.i.d. within strata and the bound is a union bound over 8 strata, under the assumption that the per-instance failure indicators are bounded in [0,1] and the strata are independent; this assumption fails when the Latin-hypercube induces correlation across strata, in which case the term under-covers and `eps_hat` is optimistic. The hash check `sha256(P) == h_P` measures *certificate validity preservation across edits* because it certifies that the family the bound was computed on is the family the current edit is being scored against, under the assumption that the agent cannot construct a second family that collides in hash; this assumption fails only under a hash collision, in which case the gate reflects a stale certificate. The Phase 1.5 canary check measures *family representativeness* because it compares the unedited baseline's error on 32 external canary perturbations against the interval predicted by `P`, under the assumption that the canary set is drawn from the deployment distribution and is disjoint from `P`; this assumption fails when the canary set is itself unrepresentative or overlaps the agent's diagnostic probes, in which case `family_valid` reflects the canary's coverage rather than the family's.

## Contribution

(1) A per-edit certification procedure, CPG, that freezes an agent-constructed stratified perturbation family before any edit is scored and re-derives a distributional bound on every edit, with a hash invariant that makes the certificate valid across the repair sequence rather than inherited from the selection sample. (2) A design principle — diagnostic-probe filtering — that separates the instances a repair loop conditions on from the instances its certificate is computed on, which is the concrete mechanism by which empirical-selection-as-certification is broken. (3) An open perturbation-family constructor and certificate log format that records, per edit, the family hash, the stratified risk, and the worst-stratum gap, enabling audit of which axis each edit traded against.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | HexaShell robot-coding tasks, held-out layouts | Matches deployment axis; tests generalization to unseen layouts |
| Primary metric | Accepted-edit deployment error on held-out layouts | Directly measures whether gate licenses robust edits |
| Secondary metric | Acceptance rate at eps_target = 0.10 | Pairs with primary metric so a gate that rejects everything is not trivially best |
| Baseline | HexaShell repair loop, no gate | Tests whether gating itself reduces deployment error |
| Baseline | Fixed-evaluator scoring (AlphaEvolve-style) | Tests whether frozen family beats fixed evaluator |
| Baseline | Inherited running certificate | Tests whether per-edit re-derivation matters |
| Baseline | Adaptive-family certificate (family re-selected per edit from the same calibration metadata) | Tests whether freezing the family, not just bounding, drives the effect |
| Ablation-of-ours | CPG without diag-probe filter | Tests whether probe filtering prevents optimistic bias |
| Ablation-of-ours | CPG without Phase 1.5 canary validation | Tests whether family validation is load-bearing |
| Stress test | Injected fifth axis (occlusion) and non-monotone lighting bin | Measures coverage gap of eps_hat under known misspecification |

### Why this setup validates the claim

The central claim is that a frozen, agent-constructed perturbation family plus a per-edit distributional bound yields accepted edits whose held-out deployment error stays under target, whereas loops that score edits on a fixed evaluator or inherit a certificate do not. The HexaShell layout split is the right testbed because its train/test layout separation is exactly the camera-pose and layout-shift axes we name, so held-out error is a genuine out-of-family measurement rather than a re-scoring of the same samples. The ungated repair loop tests the null: if gating adds nothing, its accepted-edit error should match. The fixed-evaluator baseline tests whether the frozen stratified family, rather than any evaluator, drives the effect. The inherited-certificate baseline isolates per-edit re-derivation. The adaptive-family baseline tests whether freezing the family is necessary or whether any per-edit bound suffices. The no-filter ablation is the sharpest test: if filtering diagnostic probes is what keeps `err_hat` honest, removing it should produce accepted edits with inflated held-out error concentrated on the strata the agent inspected. The no-canary ablation tests whether Phase 1.5 is load-bearing: if removing it does not change accepted-edit error, the family-validation story is not doing the work. The stress test converts the named failure modes (fifth axis, non-monotone bin) into measured coverage gaps by injecting a known occlusion axis and a non-monotone lighting bin into the held-out set and reporting the difference between eps_hat and true held-out error. The primary metric is accepted-edit held-out error, not acceptance rate, because a gate that rejects everything trivially bounds error; pairing it with acceptance rate in prose keeps the claim falsifiable.

### Expected outcome and causal chain

**vs. HexaShell repair loop, no gate** — On a case where the agent writes an edit that fixes the ten training layouts by overfitting to their lighting range, the ungated loop accepts it because its only signal is training-layout success, and deployment fails on the held-out mid-range lighting stratum. Our method instead scores the edit on the frozen stratified family, where the worst-stratum gap surfaces that lighting bin, so `eps_hat` exceeds 0.10 and the edit is rejected and returned as a repair signal. We expect a noticeable gap in accepted-edit held-out error, concentrated on the lighting and layout-shift strata, with parity on the easy camera-pose strata.

**vs. Fixed-evaluator scoring (AlphaEvolve-style)** — On a case where the agent's edit improves the fixed evaluator's aggregate score by exploiting one layout, the fixed evaluator accepts it because it never names which perturbation axes it covers, so a single-axis gain masks cross-axis regression. Our method instead computes per-stratum risk on a family constructed before the edit and takes the worst-stratum gap, so the same edit is rejected when its damage concentrates on a named axis. We expect the fixed-evaluator baseline to show higher held-out error specifically on strata it never sampled, with the gap widening as the number of perturbation axes the edit touches grows.

**vs. Inherited running certificate** — On a case where the first edit is genuinely robust but the third edit in the loop reintroduces a fragility on paraphrase, the inherited-certificate baseline accepts the third edit because its bound was computed once and never re-derived, so the certificate describes a stale code state. Our method instead re-derives `eps_hat` from scratch per edit and re-checks the family hash, so the paraphrase fragility appears in the worst-stratum gap and the edit is rejected. We expect the inherited baseline's accepted-edit error to drift upward across loop rounds while ours stays flat, with the divergence appearing only after the second edit.

**vs. Adaptive-family certificate** — On a case where the agent's edit damages a stratum that the adaptive family would have down-weighted after seeing the edit, the adaptive baseline accepts because its family was re-selected to exclude the damaged stratum. Our method instead holds `P` fixed and hashed, so the damaged stratum remains in `S_eval` and the worst-stratum gap surfaces it. We expect the adaptive baseline to show higher held-out error on the strata it down-weighted, with the gap appearing on edits that touch more than one axis.

**vs. CPG without diag-probe filter** — On a case where the agent inspects a specific lighting instance to write an edit, the no-filter ablation includes that instance in `S_eval`, so `err_hat` is optimistic on exactly the stratum the agent inspected. Our method filters `S_diag(e)` out, so the same edit's `err_hat` is higher and the worst-stratum gap is not masked. We expect the no-filter ablation to accept edits with inflated held-out error concentrated on the inspected strata, with the gap largest on edits whose diagnostic probes are concentrated in one stratum.

**vs. CPG without Phase 1.5 canary validation** — On a case where the agent constructs a family that omits the occlusion axis, the no-canary ablation accepts edits that fail on occlusion because `family_valid` is never checked. Our method's canary set includes occlusion perturbations, so the per-stratum interval test fails and the certificate is tagged `family-unvalidated`, causing rejection. We expect the no-canary ablation to show higher held-out error on the occlusion stress test, with the gap appearing only when the canary set contains the missing axis.

### What would falsify this idea

If the no-filter ablation produces the same accepted-edit held-out error as full CPG, then diagnostic-probe filtering is not doing the work and the empirical-selection story is wrong. If our gain over the ungated loop is uniform across all strata rather than concentrated on the lighting and layout-shift strata where the predicted failure mode lives, the mechanism is not the one we claim. If the no-canary ablation matches full CPG on the occlusion stress test, then Phase 1.5 is not load-bearing and the family-validation story is wrong.

## References

1. HexaShell: A Thin Harness for Exploring the Limits of VLM Agents as Robot Coding Agents
2. AlphaEvolve: A coding agent for scientific and algorithmic discovery
3. MoGe-2: Accurate Monocular Geometry with Metric Scale and Sharp Details
4. AutoEval: Autonomous Evaluation of Generalist Robot Manipulation Policies in the Real World
5. MoGe: Unlocking Accurate Monocular Geometry Estimation for Open-Domain Images with Optimal Training Supervision
6. Multi-objective Evolution of Heuristic Using Large Language Model
7. Amplifying human performance in combinatorial competitive programming
8. Autonomous Improvement of Instruction Following Skills via Foundation Models
