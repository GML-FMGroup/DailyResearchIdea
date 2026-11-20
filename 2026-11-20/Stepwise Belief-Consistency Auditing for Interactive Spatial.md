# Stepwise Belief-Consistency Auditing for Interactive Spatial CAPTCHAs

## Motivation

TuringSpace records only terminal pass/fail and its failure analysis shows models hallucinate evidence to defend wrong 3D beliefs across click/drag iterations; ORIGAMISPACE similarly assumes correct terminal answers imply correct spatial reasoning, leaving intermediate belief refinement unaudited. The structural gap is that no existing checker enforces an invariant on the belief trajectory itself, so hallucination persistence and self-correction are invisible to evaluation.

## Key Insight

A monotone-consistency invariant between each emitted 3D layout belief and the oracle's post-action state makes belief revision a per-step, oracle-constrained object that hallucinated evidence cannot satisfy without contradicting observed feedback.

## Method

**What it is.** BELIEF-AUDIT is an oracle-conditioned wrapper around TuringSpace's interactive CAPTCHA loop. At each step t the model receives the screenshot, its previous belief B_{t-1}, and the checker's last verdict, and must emit (a) a *target cell* c_t on the 16×16 occupancy grid and (b) an explicit coarse 3D belief B_t = (scene graph of object nodes with 3D centers/extents, a 2D occupancy grid at 16×16 cells, and a categorical free/occupied/unknown label per cell). **The action a_t is a deterministic function of the declared belief: a_t = planner(B_t, c_t), where planner is a fixed, non-learned routine that converts the committed target cell c_t and the belief's occupancy labels into the CAPTCHA's click/drag primitive (click at c_t if the task is single-click; drag from the nearest occupied cell to c_t if the task is drag).** The CAPTCHA program's oracle applies a_t, producing post-action state S_t; a belief verifier then returns a structured verdict V_t that gates the next step. **Load-bearing assumption (stated explicitly):** the emitted B_t is the sufficient statistic that determines a_t, because a_t is computed from B_t by the fixed planner and cannot be chosen independently; therefore oracle-falsifying labels in B_t diagnose the belief that drove behavior, not a decorative post-hoc artifact. **The 'unknown' label is not an escape hatch: every queried cell must carry a definite free/occupied label, and an abstention (unknown) on a queried cell is counted as a contradiction in K_t.**

**How it works.**
```pseudocode
INPUT: task program P (checker C, oracle O), model M, step budget T=10
OUTPUT: trajectory tau = [(a_t, B_t, V_t)], terminal score, audit log

B_0 = M.emit_belief(screenshot_0, empty_history)   # all 256 cells labeled free/occupied
for t in 1..T:
    s_t = screenshot(P, state)
    (c_t, B_t) = M.act_and_believe(s_t, B_{t-1}, V_{t-1})   # joint emission; c_t is a grid cell
    a_t = planner(B_t, c_t)                                  # deterministic: action derived from belief
    S_t = O.apply(state, a_t)                                # oracle post-action state
    V_t = verify(B_t, B_{t-1}, S_t, a_t, C)                  # structured verdict
    if V_t.status == REJECT:
        M.receive(V_t); continue                            # no state advance, belief must be revised
    state = S_t; log(a_t, B_t, V_t)
terminal = C.check(state); audit = summarize(tau)
```

`verify` returns three sub-scores, each in [0,1] plus a binary REJECT flag:
1. **Oracle agreement** A_t = fraction of queried cells/objects whose occupancy or 3D position label matches S_t (queried set = cells within the action's reachable neighborhood, radius 2 cells; object nodes matched by checker-assigned IDs). **Abstentions on queried cells count as mismatches in A_t.**
2. **Monotone consistency** M_t = 1 − |A_t − A_{t-1}| / max(A_{t-1}, ε): belief quality may not drop after oracle feedback; a drop triggers REJECT. **Exception: if the action a_t is a drag that reveals previously occluded cells (detected by the oracle as newly rendered cells in S_t that were not rendered in S_{t-1}), the queried set for M_t is restricted to cells rendered in both S_{t-1} and S_t, so legitimate revision on newly revealed cells does not penalize M_t.**
3. **Contradiction count** K_t = number of (object, cell) pairs where B_t asserts a label that S_t directly falsifies, **plus the number of queried cells where B_t abstains (unknown)**; K_t > 0 forces REJECT regardless of A_t.
The model's training signal (when fine-tuning, mirroring TuringSpace's gym) is the per-step reward r_t = A_t · M_t − λ·K_t with λ=2, plus terminal reward from C.

**Why this design.** We chose a *joint* emission of action and belief over two separate heads (as in a router/controller decomposition) because the belief must be the sufficient statistic the action was conditioned on — splitting them lets a model emit a plausible action with a decorative belief, which is exactly the hallucination mode TuringSpace reports; the cost is that a single emission failure corrupts both, so REJECT must roll back the action. **We close the decorative-belief channel by making the action a deterministic function of the declared belief (a_t = planner(B_t, c_t)), so the belief cannot be post-hoc rationalization: the model must commit to a target cell c_t inside B_t, and the planner reads only B_t and c_t.** We chose oracle-grounded *cell/object queries* over free-form belief text because free-form beliefs are unverifiable without a learned judge (the anti-pattern of a learned independence head over coupled variables); the cost is a coarser belief than a full scene graph, which we accept because the audit's purpose is trajectory consistency, not metric reconstruction. We chose a hard REJECT flag over a soft penalty alone because a soft penalty permits the model to trade contradiction for agreement, reproducing hallucination persistence; the cost is that legitimate belief revision after surprising feedback can be misflagged, which we mitigate by restricting the queried set to the action's reachable neighborhood and by the newly-revealed-cell exception in M_t. **We remove the 'unknown' escape hatch by counting abstentions as contradictions, so a model cannot satisfy the invariant with a low-commitment belief.** This is not a variant of process reward models (Lightman 2023) or VinePPO per-step rollouts: those assign a scalar to a reasoning prefix via a learned or Monte-Carlo value, whereas our verdict is computed by the task's oracle against a declared belief object, so the signal is not a learned proxy for correctness but a falsification test on the belief itself.

**Why it measures what we claim.** The oracle agreement A_t measures *belief correctness at step t* because the oracle's post-action state is the ground-truth 3D layout by construction of the CAPTCHA program; this assumption fails when the oracle's state is under-specified (e.g., occluded cells the program never renders), in which case A_t reflects agreement with a partial ground truth and under-counts errors. The monotone consistency M_t measures *self-correction rather than hallucination persistence* because it assumes that a model receiving oracle feedback should not lose belief accuracy — i.e., feedback is informative; this assumption fails when the action itself legitimately changes the scene (a drag that reveals new occluded regions), in which case M_t penalizes correct revision, which is why we restrict queries to the reachable neighborhood and exclude newly revealed cells from the M_t queried set. The contradiction count K_t measures *hallucinated evidence* because it counts beliefs the oracle directly falsifies rather than beliefs that are merely wrong; this assumption fails when the model's belief format and the oracle's state use incompatible granularity, in which case K_t reflects format mismatch rather than hallucination, which we bound by fixing the 16×16 grid and checker-assigned object IDs. **The action-determinism property (a_t = planner(B_t, c_t)) measures *the belief that drove behavior* because the planner reads only B_t and c_t; this assumption fails when the planner's fixed routine is ambiguous for a given (B_t, c_t) pair (e.g., multiple occupied cells equidistant from c_t), in which case a_t reflects the planner's tie-breaking rather than the model's belief, which we bound by having the planner break ties by lowest cell index and logging the tie-break in the audit.**

## Contribution

["BELIEF-AUDIT: an oracle-conditioned stepwise belief-consistency verifier that converts TuringSpace's terminal checker into a per-step constraint on an explicit 3D layout belief, with a hard REJECT rule on oracle contradictions.", 'A trajectory-level audit metric family (oracle agreement, monotone consistency, contradiction count) that separates hallucination persistence from legitimate belief revision in interactive spatial tasks.', 'A fine-tuning gym extension in which per-step verifier verdicts supply dense rewards, enabling measurement of whether belief-consistency constraints improve terminal CAPTCHA success.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | TuringSpace 125 interactive CAPTCHAs | Action-grounded spatial tasks with oracle and checker |
| Primary metric | Task solve rate under step budget T=10 | Directly reflects action-and-belief loop success |
| Secondary metric | Verifier precision/recall on human-labeled trajectories | Bounds false-REJECT and false-ACCEPT rates |
| Baseline | GPT-6 Astra (no belief audit) | Strongest reported model, isolates wrapper effect |
| Baseline | Kimi K3 (no belief audit) | Open-weight weak point, tests generalizability |
| Baseline | Coarse-Correspondence prompting | Training-free spatial boost, tests belief structure |
| Baseline | Monte-Carlo/VinePPO-style per-step value | Tests whether scalar value suffices vs. contradiction-typed verdict |
| Baseline | Learned process-reward-model (PRM) per-step scalar | Tests whether learned proxy suffices vs. oracle falsification |
| Ablation-of-ours | Soft-penalty only, no REJECT | Tests whether hard gating is load-bearing |
| Ablation-of-ours | Allow 'unknown' labels (escape hatch restored) | Tests whether abstention counting is load-bearing |
| Ablation-of-ours | Remove action-determinism (separate action head) | Tests whether belief-driven action is load-bearing |
| Calibration set | 512 human-annotated trajectories (held-out) | Estimates verifier precision/recall vs. radius and grid |

### Why this setup validates the claim

The central claim is that a joint action-plus-belief emission, gated by an oracle-falsification verdict, converts TuringSpace's reported hallucination-persistence failure into self-correction. TuringSpace is the right dataset because its CAPTCHA programs supply both an oracle post-action state and a terminal checker, which is exactly what the belief verifier needs; no other benchmark in the evidence provides per-step ground truth against a declared belief object. The two unmodified baselines test the sub-claim that the gain comes from the audit loop rather than from a stronger base model: if both GPT-6 Astra and Kimi K3 improve by roughly the same margin, the mechanism is model-agnostic rather than a capability artifact. Coarse-Correspondence prompting tests whether structured belief alone (without gating) suffices, isolating the verifier's contribution. The Monte-Carlo/VinePPO-style per-step value baseline and the learned PRM baseline test whether a scalar per-step signal can substitute for the contradiction-typed verdict; if they match our method, the oracle-falsification structure is not load-bearing. The soft-penalty ablation is the load-bearing control: if removing the hard REJECT flag restores the baseline's hallucination pattern, then contradiction gating — not merely the reward shaping — is what produces the effect. The 'unknown'-allowed ablation tests whether abstention counting closes the Goodhart channel. The action-determinism ablation tests whether the belief-driven action is load-bearing. The 512-trajectory calibration set provides the false-REJECT rate on legitimate post-drag revisions and false-ACCEPT rate on hallucinated beliefs, reported as precision/recall of the verifier as a function of neighborhood radius (1, 2, 3 cells) and grid resolution (8×8, 16×16, 32×32), so the audit verdict is shown to measure belief quality rather than format or occlusion artifacts. Solve rate is the right primary metric because TuringSpace's own analysis attributes failures to misread screenshots that reasoning fails to correct, so a mechanism that repairs beliefs should show up as terminal solve-rate gains concentrated on the interactive, multi-step subset.

### Expected outcome and causal chain

**vs. GPT-6 Astra (no belief audit)** — On a drag task where the first screenshot occludes the target cell, the baseline commits to a wrong 3D layout from the visible view, drags in the wrong direction, and its reasoning then defends the wrong belief rather than revising it, which TuringSpace identifies as its dominant 25.9% failure mode. Our method emits a coarse belief whose queried reachable-neighborhood cells contradict the oracle's post-action state, triggering REJECT and forcing a revised belief before the state advances. We expect a noticeable gap on the interactive multi-step subset but near-parity on single-click recognition tasks, where no belief revision is needed.

**vs. Kimi K3 (no belief audit)** — On a multi-step task requiring spatial tracking across drags, the baseline's 68.0% failure rate concentrates in exactly the cases where it must maintain a belief across steps; without an explicit belief object it has no persistent state to correct, so each screenshot is interpreted afresh and errors compound. Our method carries B_{t-1} forward and gates each emission against the oracle, so compounding is interrupted at the first contradiction. We expect a larger absolute gain for Kimi K3 than for GPT-6 Astra, since the weaker prior benefits more from externalized belief state.

**vs. Coarse-Correspondence prompting** — On tasks where the model's belief is plausible but the action is inconsistent with it, prompting-only correspondence injection supplies better spatial cues but no mechanism to detect that the emitted action does not follow from the belief, so it can produce a correct-looking belief with a decorative wrong action. Our joint emission plus oracle agreement check ties the action to the belief and rejects mismatches. We expect prompting to improve early-step accuracy but plateau across steps, while our method's advantage grows with step index.

**vs. Monte-Carlo/VinePPO-style per-step value** — The value baseline assigns a scalar to each reasoning prefix via Monte-Carlo rollouts but cannot express which (object, cell) pair is contradicted, so its correction signal is undirected. Our verdict returns a contradiction-typed correction (which cells/objects are falsified), which we expect to produce faster belief revision on the multi-step subset; if the scalar baseline matches our solve rate, the typed verdict is not load-bearing.

**vs. Learned process-reward-model (PRM)** — The PRM assigns a learned scalar per step and is trained on the same trajectories; it can reward plausible-looking beliefs without oracle falsification. We expect the PRM to be vulnerable to the decorative-belief mode (it cannot distinguish a conservative belief from a correct one) and to show a smaller gain on the contradiction-triggering subset; if the PRM matches our method, the oracle-falsification signal is not load-bearing.

### What would falsify this idea

If the improvement over unmodified baselines is uniform across single-step recognition and multi-step interactive subsets rather than concentrated on the contradiction-triggering multi-step subset, the audit loop is not doing the claimed work. Likewise, if the soft-penalty ablation, the 'unknown'-allowed ablation, or the action-determinism ablation matches the full method, the corresponding component (hard REJECT, abstention counting, or belief-driven action) is not load-bearing and the central mechanism claim fails. If the calibration study shows false-REJECT rate above 10% on human-labeled legitimate revisions or false-ACCEPT rate above 10% on hallucinated beliefs at the chosen radius (2 cells) and grid (16×16), the verifier is measuring format or occlusion artifacts rather than belief quality.

## References

1. TuringSpace: Grounding Spatial Intelligence in Action with Interactive CAPTCHAs
