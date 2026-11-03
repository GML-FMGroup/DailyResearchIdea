# Counterfactual Replay Contracts: Verifying Video Caption Faithfulness by Executable Consistency

## Motivation

Across both paper-trees, faithfulness of an intermediate video representation (code captions in CodeCap-Bench, VLM-narrative-derived transition boundaries in SBS) is inferred from downstream QA accuracy or reconstruction fidelity rather than checked against the source video, so a syntactically fluent but semantically wrong representation can score well. CodeCap-Bench itself concedes that whether MLLMs can generate *faithful* code captions is the key unresolved bottleneck, yet its multi-level framework still grades faithfulness through downstream QA and reconstruction — proxies that a representation can game by matching surface statistics without predicting the video's actual state transitions. The structural gap is that no pipeline treats faithfulness as an identifiability condition verifiable by intervention on the video's own observable trajectory.

## Key Insight

Video supplies a dense, partially observable state trajectory, so a faithful intermediate representation is identifiable exactly when its predicted transition signature — replayed under a spanning set of controlled counterfactual edits — matches the video's observed transitions, making faithfulness a checkable consistency property rather than a downstream correlation.

## Method

**What it is.** CRaFT (Counterfactual Replay and Fault-localization Test) takes a video `V` and an intermediate representation `R` (executable code caption, scene graph, or policy plan) and outputs a partial-identifiability certificate over the expressible edit subspace: a binary verdict in {FAITHFUL, UNFAITHFUL, UNDEFINED} plus a segment-localized residual map. It requires no retraining and no downstream QA; it only requires that `R` be *executable* to produce a predicted observable-state trajectory and that edits to `V` be expressible as edit operators on `R`'s inputs. The load-bearing assumption, stated explicitly: every edit operator `e` used in the contract must have a well-defined, locally-editable syntactic analogue in `R`, such that `R_e` is well-formed and a null-effect edit yields `delta_pred = 0`; edits failing this precondition are excluded from the spanning set and the verdict is reported as a partial-identifiability certificate over the expressible subspace rather than a global FAITHFUL/UNFAITHFUL label.

**How it works.**

```pseudocode
INPUT: video V (frames f_1..f_T), representation R, edit-operator library E
OUTPUT: verdict in {FAITHFUL, UNFAITHFUL, UNDEFINED}, residual map r_1..r_T,
        expressibility mask M (one bit per edit e in E)

# Phase 1 — Extract observable state trajectory from V
S_obs = [phi(f_t) for t in 1..T]        # phi = fixed off-the-shelf detector for
                                         # the category's observable predicates
                                         # (object presence, position, count)

# Phase 2 — Execute R on the *unperturbed* video inputs
S_pred = execute(R, inputs=observed_initial_state)   # replay forward
residual_0 = d(S_pred, S_obs)            # d = per-predicate Hamming distance

# Phase 3 — Build a spanning set of counterfactual edits, with expressibility mask
EDITS = E_spanning(V, E)                 # k edits chosen so their
                                         # effect-vectors span the predicate
                                         # space; k = rank of the effect matrix
for each edit e in EDITS:
    R_e = R with e's syntactic analogue substituted
    M[e] = 1 iff R_e is well-formed AND a null-effect edit yields delta_pred = 0
    # edits with M[e]=0 are excluded from the spanning set and reported
    # as expressivity gaps, not infidelity
EDITS_expressible = {e in EDITS : M[e]=1}

# Phase 4 — Bidirectional executable-consistency contract
for each edit e in EDITS_expressible:
    V_e      = apply_edit(V, e)          # e.g. delete object, shift onset time
    S_obs_e  = [phi(f_t) for f_t in V_e]
    S_pred_e = execute(R_e, inputs=observed_initial_state)
    delta_obs  = S_obs_e  - S_obs        # observed effect of e
    delta_pred = S_pred_e - S_pred       # predicted effect of e
    residual_e = d(delta_pred, delta_obs)   # syntax-invariant: both are
                                            # *differences* of trajectories

# Phase 5 — Faithfulness verdict + localization
r_t = aggregate over edits in EDITS_expressible of |residual_e restricted to segment t|
if |EDITS_expressible| < k_min:          # k_min = 0.5 * k, calibrated on held-out set
    verdict = UNDEFINED                  # insufficient expressible edits for identifiability
else:
    verdict = FAITHFUL iff  max_{e in EDITS_expressible} residual_e <= tau
                            AND  residual_0 <= tau_0
                                         # tau, tau_0 calibrated on a
                                         # held-out faithful-reference set

# Expressibility report: fraction of edits in E with M[e]=1, per representation family
```

The key quantity is `residual_e`: it compares *differences* of trajectories, so it is invariant to whether `R` writes the same effect as a `for`-loop, a scene-graph edge, or a plan step. The residual map `r_t` localizes the failure to a segment because each edit's predicted-vs-observed effect is attributed to the frame interval where the edit's syntactic analogue lives. The expressibility mask `M` makes the abstraction mapping explicit: only edits with a well-formed syntactic analogue enter the contract, so `residual_e` is an identifiability residual on a declared abstraction rather than a conflation of infidelity with non-expressibility.

**Why this design.** We chose to compare *effect differences* (`delta_pred` vs `delta_obs`) rather than raw trajectories because raw-trajectory distance rewards surface-syntactic match, which is exactly the failure mode CodeCap-Bench's downstream-QA proxy exhibits; the cost is that an edit operator must have a well-defined syntactic analogue in `R`, which excludes representations whose syntax cannot be edited locally. We chose a *spanning* edit set (rank of the effect matrix) over a random edit sample because random edits leave directions of the predicate space untested and cannot certify identifiability; the cost is that computing the spanning set requires a one-time linear-algebra pass over the effect matrix, which is cheap but assumes the predicate space is finite and enumerable. We chose a held-out calibration set for the thresholds `tau, tau_0` over a fixed heuristic because faithfulness is a binary decision and the thresholds must be tied to the noise floor of `phi`; the cost is that the verdict is only as calibrated as that reference set. We chose an explicit per-edit expressibility mask `M` over a binary 'edit has a syntactic analogue' precondition because the causal-abstraction literature (Beckers & Halpern 2019; Geiger et al. 2021) shows that counterfactual-consistency tests are only valid under a specified abstraction mapping; the cost is that the verdict becomes a partial-identifiability certificate over the expressible subspace rather than a global label, and representations with low expressibility yield UNDEFINED rather than a verdict. This is *not* the composition of a VLM captioner with a downstream QA scorer (the CodeCap-Bench baseline): a domain expert would not describe CRaFT as a variant of CodeCap-Bench because CRaFT never invokes a QA model — it invokes the representation's own executor and asks whether the representation's *predicted transition signature* is invariant under the same edits as the video's.

**Why it measures what we claim.** `residual_e` measures *faithfulness* of `R` because the assumption that a representation is faithful iff its predicted effect under every edit in a spanning set matches the video's observed effect makes residual_e an identifiability residual rather than a correlation; this assumption fails when an edit operator has no faithful syntactic analogue in `R` (e.g., a scene-graph representation cannot express a temporal reordering edit), in which case residual_e reflects the representational *expressivity gap* rather than infidelity — the expressibility mask `M` excludes such edits and reports them separately. The spanning-set rank condition measures *coverage of the counterfactual space* because the assumption that effect-vectors of edits span the predicate space makes the spanning set a certificate that no untested direction remains; this assumption fails when the predicate space is not finite (open-vocabulary scenes), in which case the spanning set reflects only the enumerated predicates and the verdict is reported as a partial-identifiability certificate over the enumerated subspace. The per-segment residual map `r_t` measures *failure localizability* because the assumption that each edit's syntactic analogue lives in a known frame interval makes residual attribution exact; this assumption fails for edits whose effects propagate globally (e.g., a global color shift), in which case `r_t` reflects effect-propagation rather than the true fault location. The threshold `tau` measures *the noise floor of the observable detector `phi`* because the assumption that the held-out faithful-reference set is itself faithful makes `tau` a calibrated false-positive bound; this assumption fails when the reference set contains subtly unfaithful representations, in which case `tau` is inflated and the verdict becomes permissive.

## Contribution

(1) A faithfulness-verification framework, CRaFT, that replaces downstream-proxy scoring with a bidirectional executable-consistency contract: a representation is faithful iff its predicted transition signature matches the video's under a spanning set of counterfactual edits, with a syntax-invariant residual and a per-segment localization map. (2) A construction of spanning counterfactual edit sets whose effect-vectors certify coverage of the observable predicate space, together with a held-out calibration procedure for the faithfulness thresholds tied to the detector noise floor. (3) A design principle — that faithfulness of any intermediate video representation is an identifiability condition verifiable by intervention on the video's own trajectory — demonstrated across code captions, scene graphs, and policy plans.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | CodeCap-Bench (200-video human-labeled calibration split; 800-video test split) | Provides executable code captions and downstream-QA baseline. |
| Primary metric | Faithfulness verdict accuracy vs. human labels (3-way: FAITHFUL/UNFAITHFUL/UNDEFINED) | Binary verdict directly tests central claim. |
| Secondary metric | Expressibility fraction per edit operator (fraction of representations admitting a faithful syntactic analogue) | Bounds the expressivity gap empirically. |
| Baseline | CodeCap-Bench downstream-QA scorer | Tests whether QA proxy captures faithfulness. |
| Baseline | Natural-language caption + QA | Tests whether syntax-agnostic effects are necessary. |
| Baseline | Raw-trajectory distance (no counterfactual edits) | Tests necessity of counterfactual edit spanning set. |
| Ablation-of-ours | CRaFT with random edit sample (not spanning) | Tests whether spanning set is required for identifiability. |
| Ablation-of-ours | CRaFT without expressibility mask (all edits enter contract) | Tests whether mask prevents expressivity-gap false negatives. |

### Why this setup validates the claim

The central claim is that CRaFT's counterfactual effect-difference residual yields a calibrated binary faithfulness verdict that localizes faults, and that this verdict is not obtainable from downstream QA or raw-trajectory distance. CodeCap-Bench is the right dataset because it supplies executable code captions and human faithfulness annotations, so the verdict can be compared against ground truth rather than a proxy. The 200-video human-labeled calibration split is used to set `tau` and `tau_0` and to compute the expressibility fraction per edit operator; the 800-video test split is held out for verdict accuracy. The CodeCap-Bench downstream-QA baseline tests the sub-claim that QA scores are not faithfulness measures: if CRaFT agrees with human labels while QA does not, the proxy failure is confirmed. The natural-language caption baseline tests whether the effect-difference mechanism is necessary even when syntax is not code. The raw-trajectory baseline tests whether counterfactual edits add anything beyond replay residual. The random-edit ablation isolates the spanning-set rank condition: if random edits match spanning edits, the identifiability argument is wrong. The no-mask ablation isolates the expressibility mask: if removing the mask does not change verdicts on low-expressibility representations, the mask is unnecessary. Verdict accuracy is the right metric because it is the method's actual output and is falsifiable against human labels; localization is secondary and checked only on detected unfaithful cases. The expressibility fraction is reported per edit operator to bound the assumption's failure mode empirically.

### Expected outcome and causal chain

**vs. CodeCap-Bench downstream-QA scorer** — On a case where a code caption is syntactically fluent but mispredicts an object's disappearance after a deletion edit, the QA scorer produces a high score because it asks questions whose answers are surface-consistent with the caption's own narrative. Our method instead executes the caption under the deletion edit and compares the predicted effect difference to the observed effect difference, so we expect a noticeable verdict-accuracy gap concentrated on unfaithful-but-fluent captions, with parity on clearly faithful captions.

**vs. Natural-language caption + QA** — On a case where a natural-language caption describes the right objects but omits a temporal reordering, the QA baseline produces a faithful verdict because its questions do not probe the reordering. Our method applies the reordering edit to the code caption's syntactic analogue and detects the predicted-vs-observed effect mismatch, so we expect CRaFT to flag unfaithful temporal captions that the NL baseline passes, producing a gap on the temporal-edit subset but parity on object-presence captions.

**vs. Raw-trajectory distance (no counterfactual edits)** — On a case where a code caption uses a different loop structure but produces the same observable trajectory, raw-trajectory distance produces a large residual and an unfaithful verdict because it compares raw trajectories and rewards surface-syntactic match. Our method compares effect differences under edits, which are invariant to loop structure, so we expect CRaFT to yield a faithful verdict on this case while raw-trajectory distance fails, producing a gap on syntax-variant faithful captions and parity on unfaithful captions.

**vs. CRaFT with random edit sample** — On a case where the unfaithfulness lies in a predicate direction not hit by random edits, the random-edit ablation produces a faithful verdict because its residuals are small on the sampled directions. Our method uses a spanning set that covers the predicate space, so it detects the residual in the untested direction, and we expect a verdict-accuracy gap on unfaithful cases whose fault direction is rare, with parity on cases whose fault direction is common.

**vs. CRaFT without expressibility mask** — On a case where a scene-graph representation cannot express a temporal reordering edit, the no-mask ablation produces an unfaithful verdict because the edit's syntactic analogue is ill-formed and `delta_pred` is undefined or zero, which the contract scores as a large residual. Our method's expressibility mask excludes this edit and reports UNDEFINED or a verdict over the expressible subspace, so we expect the no-mask ablation to produce false UNFAITHFUL verdicts on low-expressibility representations, with parity on high-expressibility representations.

### What would falsify this idea

If CRaFT's verdict accuracy is uniform across subsets rather than concentrated on the cases where the predicted failure modes occur (fluent-but-unfaithful, temporal-edit, syntax-variant, rare-direction), then the counterfactual effect-difference mechanism is not doing the work we claim and the central argument is wrong. Similarly, if the spanning-set ablation matches random edits on rare-direction faults, the identifiability claim fails; and if the expressibility fraction is near zero for a representation family, the partial-identifiability certificate is vacuous for that family.

## References

1. CodeCap-Bench: Can Code Serve as a Good Video Caption?
2. Seeing Before Synthesizing: VLM-Guided Transition Event Discovery for Weakly-Supervised Dense Video Captioning
3. Implicit Location-Caption Alignment via Complementary Masking for Weakly-Supervised Dense Video Captioning
4. SmolVLM: Redefining small and efficient multimodal models
5. NVILA: Efficient Frontier Visual Language Models
