# Revision-Algebraic Monotone Consistency for MLLM World Models

## Motivation

WM^MLLM (arXiv:2611.24193) stores an external panorama with a revision history and lets an MLLM translate requests into state operations and 'check every candidate before it becomes the next state', but the check is an opaque MLLM judgment with no invariant that ties a candidate revision to previously verified properties of the state. Structurally, the verifier and the editor draw on the same MLLM, so an inconsistency introduced by an earlier edit can be silently inherited by every later candidate: the system guarantees pixel-level locality and exact revisit-by-revision-identity, but not that accumulated edits preserve properties the system already certified. This is the meta-level gap: persistent world state is treated as a mutable buffer whose verification is a per-request judgment rather than a monotone refinement of a certified belief.

## Key Insight

If every committed revision carries a machine-checkable certificate of the properties it preserves, and each new candidate must produce a certificate that entails the previous one on the untouched region, then silent inconsistency becomes detectable as a certificate that fails to entail — a structural property that per-request MLLM verification cannot provide because it has no memory of what was certified.

## Method

**What it is.** RAMC (Revision-Algebraic Monotone Consistency) is a wrapper around WM^MLLM's revision store that gives each committed panorama revision a *certificate*: a finite set of region-indexed property atoms (e.g. object-identity at region r, lighting-class at r, geometric-consistency at r) plus a revision pointer. Inputs are a user request and the current certified state; outputs are either a new certified revision or a rejection with the violated atom.

**Atom vocabulary (pre-registered, fixed).** The atom set is a typed predicate vocabulary of size 5: `object_identity(r) ∈ {class label from a fixed 80-class list}`, `object_count(r) ∈ ℕ`, `lighting_class(r) ∈ {day, night, dusk, indoor}`, `weather_class(r) ∈ {clear, rain, snow, fog}`, and `geometric_consistency(r) ∈ {pass, fail}` (pass iff the epipolar reprojection error of the region's rendered view against the parent panorama is below 2.0 px). Region masks are produced by a fixed procedure: the MLLM's operation set returns a bounding box per operation; the edit mask R_edit is the union of these boxes dilated by 8 px, snapped to a 16×16 grid. This vocabulary and mask procedure are fixed before any experiment and are not tuned.

**How it works.**

```pseudocode
# State: revision chain C_0 -> C_1 -> ... -> C_t, each C_i = (panorama_i, atoms_i, parent_i)
# atoms_i: dict region_id -> set of (property, value) with region masks
# Hyperparameters: K=5 probes per atom, τ_noise=0.2 (per-atom drop threshold), repair_budget=3

function PROPOSE(request, C_t):
    ops   = MLLM.translate(request, C_t)        # WM^MLLM operation set
    cand  = apply_ops(ops, C_t.panorama)        # local edit, region mask R_edit
    return cand, ops, R_edit

function ESTIMATE_ATOMS(panorama, regions, K):
    # Calibrated repeated-probe estimator: for each region r and atom type a,
    # query the MLLM K times with paraphrased probes (K=5 fixed paraphrases)
    # and return the majority value plus an empirical drop frequency.
    est = {}
    for r in regions:
        for a in ATOM_TYPES:
            votes = [MLLM.query_atom(panorama, r, a, paraphrase=j) for j in 1..K]
            est[(r,a)] = (majority(votes), drop_freq = 1 - count(majority)/K)
    return est

function CERTIFY(cand, ops, R_edit, C_t):
    # 1. Seed atoms: untouched pixels inherit C_t atoms verbatim
    inherited = {r: a for r,a in C_t.atoms if mask(r) disjoint from R_edit}
    # 2. Re-derive atoms inside R_edit with calibrated repeated probes
    edited_est = ESTIMATE_ATOMS(cand, regions(R_edit), K=5)
    # 2b. Re-derive the SAME regions on the parent panorama to subtract parent-side noise
    parent_est = ESTIMATE_ATOMS(C_t.panorama, regions(R_edit), K=5)
    edited = {}
    for (r,a) in edited_est:
        child_val, child_drop = edited_est[(r,a)]
        parent_val, parent_drop = parent_est[(r,a)]
        # Atom is treated as ABSENT only if the child's majority value differs
        # from the parent's majority value AND the drop exceeds the measured
        # per-atom noise threshold τ_noise=0.2 on unedited control regions.
        if child_val != parent_val and child_drop > τ_noise and parent_drop <= τ_noise:
            edited[(r,a)] = ABSENT
        else:
            edited[(r,a)] = child_val
    cand_atoms = merge(inherited, edited)
    # 3. Monotone entailment check against parent (using re-derived parent atoms)
    for (r, a) in C_t.atoms where r touches R_edit:
        for atom in a:
            if edited.get((r,atom)) == ABSENT and not ops.declares_removal(atom):
                return REJECT(violated=atom, region=r)   # silent change
    # 4. Revisit reproduction check (cheap, deterministic)
    for rev in sample(C_t.history, k=8):
        if render(cand, rev.view) != render(C_t.panorama, rev.view) outside R_edit:
            return REJECT(violated=revisit, region=rev.view)
    return COMMIT(cand, cand_atoms, parent=C_t)

function EDIT_LOOP(request, C_t):
    for attempt in 1..3:                       # budget = 3 repair rounds
        cand, ops, R = PROPOSE(request, C_t)
        result = CERTIFY(cand, ops, R, C_t)
        if result is COMMIT: return result
        request = MLLM.repair_prompt(request, result.violated)   # targeted retry
    return REJECT_FINAL(result)
```

**Why this design.** We chose a *certificate-carrying revision chain* over WM^MLLM's per-request MLLM verification because a stateless verifier cannot distinguish a legitimate property change (declared by the operation) from a silent one introduced by a prior edit, accepting the cost that every revision stores an atom set and every edit pays a per-region recheck. We chose *region-scoped atom inheritance* over full re-verification of the whole panorama because full re-verification is O(panorama) per edit and would make interactive editing infeasible, accepting the cost that atoms outside the edited mask are trusted rather than re-derived. We chose a *bounded 3-round repair loop* over unbounded retry because unbounded retry lets the MLLM drift the request until the certificate passes, accepting the cost that some legitimate edits are rejected outright. A domain expert would not describe this as 'WM^MLLM with a checker': WM^MLLM's checker is a fresh MLLM judgment on each candidate, whereas RAMC's load-bearing object is the entailment relation between successive certificates, which is what makes monotone consistency a property of the chain rather than of any single request.

**Why it measures what we claim.** `atoms_i` on the untouched region measures *preservation of previously certified properties* because the inheritance assumption — that pixels outside the edit mask are bit-identical to the parent and therefore retain their properties — makes the atom set a sufficient statistic for those pixels; this assumption fails when a generator's output leaks outside its declared mask, in which case the atoms reflect the parent rather than the actual child. The `declares_removal` flag measures *explicit versus silent property change* because the assumption that WM^MLLM's operation vocabulary names every intentional removal makes an undeclared disappearance a genuine inconsistency; this assumption fails when the MLLM omits a removal from its operation list while intending it, in which case a legitimate edit is flagged as a violation. The revisit-reproduction check measures *monotone consistency under belief refinement* because the assumption that render(view) is a deterministic function of panorama plus viewpoint makes any mismatch a real state divergence rather than a sampling artifact; this assumption fails under stochastic generators, in which case the check reflects generator noise instead of state inconsistency. The calibrated repeated-probe estimator measures *genuine atom absence* because the assumption that the MLLM's per-region judgments are stable enough that a drop exceeding τ_noise=0.2 on K=5 paraphrased probes reflects a real state change rather than query noise; this assumption fails when the MLLM's visual-question answering is dominated by textual patterns (as in SymDPO, arXiv:2411.11909) or when object-level visual information is only bound to question tokens in middle layers (as in cross-modal information flow work, f88c232e...), in which case the estimator's drop frequency reflects probe phrasing rather than the region's actual state.

## Contribution

(1) A revision-algebraic certificate mechanism (RAMC) that turns WM^MLLM's opaque per-request verification into a chain-level monotone-consistency invariant with region-scoped atom inheritance and declared-removal semantics. (2) A design principle for persistent MLLM world models: consistency should be certified as an entailment relation between successive revisions rather than re-judged per request, with a bounded repair loop that localizes violations to specific regions and atoms. (3) A rejection-with-witness interface that exposes which property atom a candidate revision would silently violate, enabling targeted repair rather than whole-request retry.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | MBench-A multi-edit sequences | WM^MLLM's own benchmark; supports revisit and edit chains |
| Dataset | 200 human-authored multi-edit sequences (5–10 edits each) on MBench-A scenes | Tests generality beyond synthetic chains |
| Primary metric | Silent-change rate over edit chains, corrected against human-annotated ground truth | Directly measures undeclared property loss across revisions |
| Secondary metric | False-rejection rate (fraction of legitimate edits rejected) | Separates verifier noise from genuine inconsistency |
| Secondary metric | Mask-leakage rate (fraction of pixels changed outside R_edit) and atom-sufficiency error (fraction of inherited atoms contradicted by full re-derivation) | Quantifies certificate soundness envelope |
| Secondary metric | Request-fidelity score (human 1–5 rating of committed panorama vs. original request) | Measures drift under repair |
| Baseline | WM^MLLM stateless MLLM checker | Tests claim that chain-level certificates beat per-request verification |
| Baseline | WM^MLLM with full-panorama recheck | Tests whether region-scoped inheritance suffices |
| Baseline | WM^MLLM with unbounded repair retries | Tests whether bounded repair prevents request drift |
| Baseline | RAMC with single-shot recheck (K=1, no parent re-derivation) | Isolates contribution of calibrated repeated-probe estimator |
| Ablation-of-ours | RAMC without revisit-reproduction check | Isolates contribution of deterministic revisit consistency |
| Ablation-of-ours | RAMC without parent-side re-derivation (uses stored parent atoms) | Isolates contribution of noise subtraction |

### Why this setup validates the claim

The central claim is that monotone consistency is a property of the revision chain, not of any single request, and that this distinction is what suppresses silent property changes across multi-edit sequences. MBench-A is the right arena because it is WM^MLLM's native environment-recovery benchmark, so any gain cannot be attributed to a friendlier dataset; and because its edit-and-revisit structure forces the chain to be exercised rather than a single isolated edit. The human-authored sequences test whether the effect survives outside synthetic chains. The silent-change rate is the right primary metric because it is the direct behavioral signature of the failure the method targets: a property atom present in a parent certificate and absent in a child without a declared removal. Because the calibrated repeated-probe estimator can itself err, we report silent-change rate corrected against a human-annotated ground-truth subset (200 edits, two annotators, Cohen's κ reported) and also report raw false-rejection rate, so that a gain cannot be an artifact of verifier noise. The stateless-checker baseline tests the sub-claim that per-request MLLM judgment cannot distinguish legitimate from silent change; the full-recheck baseline tests whether region-scoped inheritance is a sufficient substitute for whole-panorama re-derivation; the unbounded-retry baseline tests whether the 3-round budget is doing real work or merely throttling; the single-shot-recheck baseline tests whether calibration is load-bearing. The ablation isolates the revisit check, which is the only component that detects divergence not visible in the atom set alone. Together these form a falsifiable test: if RAMC's advantage is real, it must be concentrated on chains where silent changes are the predicted failure, it must vanish when the certificate chain is reduced to a single-request check, and it must not be explained by a lower false-rejection rate alone.

### Expected outcome and causal chain

**vs. WM^MLLM stateless MLLM checker** — On a case where an edit at revision 3 removes a lamp that was certified at revision 1 but the operation list only mentions repainting the wall, the stateless checker sees a plausible panorama and a plausible request and commits, because it has no record that the lamp's identity atom was ever certified. Our method instead walks the parent's atoms touching the edit mask, finds the lamp identity atom absent (child majority value differs from parent majority value and drop exceeds τ_noise=0.2) and undeclared, and rejects with that atom named, triggering a repair round. We expect a noticeable gap in silent-change rate that grows with chain length, with near-parity on single-edit chains.

**vs. WM^MLLM with full-panorama recheck** — On a case where a request edits one region of a large panorama, the full-recheck baseline re-derives atoms for every region, and because the MLLM's per-region judgments are noisy, it spuriously drops atoms in untouched regions and either rejects legitimate edits or commits with altered atoms. Our method inherits untouched atoms verbatim and re-derives only inside the edit mask with K=5 paraphrased probes and parent-side noise subtraction, so the noise budget is confined to the edited region and parent-side noise is subtracted rather than inherited. We expect our method to match or beat full recheck on silent-change rate while showing a large reduction in false rejections on edits whose masks are small relative to the panorama.

**vs. WM^MLLM with unbounded repair retries** — On a case where the initial proposal genuinely violates a certified atom, the unbounded baseline keeps re-prompting until the certificate passes, and the MLLM gradually rewrites the request into a different, weaker edit that no longer resembles what the user asked for. Our method caps repair at three rounds and rejects outright, so the committed revision either satisfies the original request or is not committed. We expect a visible divergence in request-fidelity on the subset of edits that require more than three repairs, where the baseline's committed panoramas drift from the original request while ours are rejected.

**vs. RAMC with single-shot recheck (K=1, no parent re-derivation)** — On a case where the MLLM's single-shot judgment on a cropped region is dominated by the textual property query rather than the pixels (as SymDPO and cross-modal information flow work suggest), the single-shot baseline will drop parent atoms on noise and reject legitimate edits. Our calibrated estimator requires a drop exceeding τ_noise=0.2 across K=5 paraphrased probes and subtracts parent-side noise, so it should show a lower false-rejection rate at matched silent-change rate, confirming that calibration is load-bearing rather than cosmetic.

### What would falsify this idea

If RAMC's improvement in silent-change rate were uniform across chain lengths and edit-mask sizes rather than concentrated on long chains and small masks — where the certificate chain and region-scoped inheritance are the predicted causal drivers — or if the gain disappeared after correcting for verifier error against human-annotated ground truth, the central claim that chain-level entailment is the load-bearing mechanism would be wrong, and the gain would instead be attributable to the extra MLLM calls alone.

## References

1. WM^MLLM: Turning MLLMs into World Models
