# Conflict-Signature Indexing: Feasibility-Gated Skill Retrieval for GUI Agents

## Motivation

Existing GUI agents retrieve skills by instruction similarity alone, ignoring instruction-GUI feasibility conflicts that determine whether a skill can be safely executed. CONFLICTGUARD detects these conflicts at inference time but discards the resulting labels, so the skill index cannot learn which skills are feasible under which conflicts. This structural gap—an independent feasibility gate whose outcomes never feed back into the retrieval key space—recurs across skill-library systems like VISUALSKILL and Agent Workflow Memory, which likewise treat retrieval and feasibility as separate, unlinked modules.

## Key Insight

The feasibility gate's conflict signature is a sufficient statistic for the conditional feasibility of a retrieved skill, so augmenting the retrieval key with this signature yields a conflict-conditioned index that is provably more discriminative than instruction-only retrieval without requiring any new training loop.

## Method

```pseudocode
# CONFLICT-SIGNATURE INDEXING (CSI)
# Inputs: skill index I (entries keyed by instruction embedding), feasibility gate G (from CONFLICTGUARD),
#         instruction x, GUI state s
# Output: conflict-conditioned skill retrieval
#
# Load-bearing assumption (stated explicitly): the conflict signature emitted by G is a sufficient
# statistic for the conditional feasibility of a retrieved skill — i.e., two instructions sharing the
# same signature require the same skill variant. This is a discrete-taxonomy sufficiency claim; it is
# verified empirically (see step 6) rather than asserted, and the guarded key is built from a calibrated
# continuous score (step 2) so that signature coarseness is bounded rather than hidden.

1. Base retrieval: r = top-k(I, embed(x)), k=5

2. Gate evaluation: (outcome, conflict_sig, p_conflict) = G(x, s, r)
   # outcome ∈ {proceed, abstain, terminate}
   # conflict_sig ∈ fixed taxonomy of C=8 conflict types from CONFLICTGUI
   # p_conflict ∈ [0,1] is the gate's calibrated conflict probability (temperature-scaled logit;
   #   temperature T fit on a 512-example held-out calibration split from CONFLICTGUI)

3. If outcome == proceed:
       return r

4. If outcome ∈ {abstain, terminate}:
       # Split the blocked skill entry into a guarded variant keyed by a soft, score-weighted signature
       # guarded_key = concat(embed(x), conflict_sig_onehot(C), p_conflict)  # dim = d_embed + C + 1
       guarded_key = concat(embed(x), onehot(conflict_sig, C), p_conflict)
       if guarded_key not in I:
           I[guarded_key] = copy(r) with metadata {conflict_sig, p_conflict, guard_policy: outcome}
       # Retrieve using the conflict-conditioned key; score-weighted so that low p_conflict
       #   (ambiguous conflict) does not hard-route to a guarded variant
       r' = top-k(I, guarded_key) weighted by p_conflict  # soft weighting, not hard concat
       return r'

5. Update: after execution, if outcome == proceed and task succeeds, reinforce the guarded entry's weight;
   if outcome == terminate and task would have failed, keep the guard as a hard filter.

6. Sufficiency verification (run once before claiming the signature is a sufficient statistic):
   On a held-out split of CONFLICTGUI, fit a logistic probe predicting skill success from
   embed(x) alone vs. embed(x) + conflict_sig + p_conflict. Report ΔAUC. If ΔAUC ≤ 0.02,
   the signature does not predict skill success beyond instruction embedding and the
   key-augmentation claim is not supported; report this as a negative result.

7. Index pruning: cap guarded variants at V=3 per base skill entry; when the cap is exceeded,
   evict the guarded variant with lowest reinforcement weight. Report index size (base + guarded)
   vs. conflict-task success at V ∈ {1, 3, 5}.
```

**Why this design**
We chose to split the index entry rather than down-weight it because down-weighting conflates two distinct failure modes—skill infeasibility under a specific conflict versus general low relevance—whereas splitting preserves the original skill for conflict-free contexts while creating a guarded variant whose key includes the conflict signature, accepting the cost of a larger index that must be pruned periodically (capped at V=3 variants per base skill, step 7). We chose to make the conflict signature a retrieval key rather than a post-retrieval filter because filtering discards the gate's signal for ranking, whereas key-augmentation lets the same instruction retrieve different skills depending on detectable conflicts, accepting the cost that the gate must produce a discrete signature (we use a fixed taxonomy of C=8 conflict types from CONFLICTGUI) and that the signature may be coarse; we therefore pair the discrete label with the gate's calibrated p_conflict so the guarded key is soft rather than a hard concat. We chose to keep the base index unchanged for proceed outcomes rather than updating all entries, because proceed outcomes are the common case and updating them would add noise; this accepts the cost that guarded variants are only created when conflicts are actually detected, which may under-cover rare conflicts. Unlike standard retrieval-augmented skill selection (e.g., VISUALSKILL's load_topic or Agent Workflow Memory's workflow retrieval), which treats the index as static with respect to feasibility, CSI edits the index's key space using the gate's labels—a prior method would not describe CSI as a variant of retrieval because the retrieval key itself is a function of a counterfactual feasibility assessment, not just instruction similarity.

**Why it measures what we claim**
The conflict signature produced by the gate measures the conditional feasibility of a skill under the current instruction-GUI pair because we assume the gate's conflict taxonomy is exhaustive for the conflict types it detects; this assumption fails when the GUI exhibits a conflict outside the taxonomy (e.g., a novel UI element that contradicts the instruction in an unanticipated way), in which case the signature reflects only the detected subset and the guarded variant may still be retrieved in an infeasible context. The outcome label (proceed/abstain/terminate) measures the gate's decision confidence because we assume the gate's threshold is calibrated for the conflict detection task (as validated in CONFLICTGUARD's benchmark, and re-checked here via temperature scaling on a 512-example calibration split); this assumption fails when the gate is overconfident on ambiguous conflicts, in which case the label reflects the gate's bias rather than true feasibility—we bound this by including p_conflict in the key and reporting reliability diagrams on the conflict subset. The retrieval key concatenation measures conflict-conditioned relevance because we assume that instructions with the same conflict signature require the same skill variant; this assumption fails when two instructions share a signature but differ in other task-relevant ways, in which case the guarded variant may be too coarse and retrieve a suboptimal skill—we test this directly with the sufficiency probe in step 6 (ΔAUC of embed(x)+signature over embed(x) alone) and with a random-label key-augmentation control in the experiment.

## Contribution

["A conflict-signature indexing mechanism that uses the feasibility gate's outcomes as a supervision signal to split skill index entries into guarded variants keyed by conflict signatures, enabling conflict-conditioned retrieval without retraining.", 'An empirical demonstration that conflict-conditioned retrieval improves conflict task success rate over instruction-only retrieval on the CONFLICTGUI benchmark, with analysis showing that the gain comes from retrieving different skills under different detectable conflicts.', 'A lightweight index-editing procedure that operates at inference time and integrates with existing skill-library systems (e.g., VISUALSKILL, Agent Workflow Memory) without modifying their base retrieval or gate components.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | CONFLICTGUI benchmark (primary); held-out AndroidWorld conflict split (transfer check) | CONFLICTGUI provides paired feasible and conflict tasks per instruction; AndroidWorld conflict split tests whether the guarded-index mechanism generalizes beyond one taxonomy |
| Primary metric | Conflict-task success rate at parity on feasible tasks (parity defined as within 2 percentage points of the CONFLICTGUARD-only baseline on the feasible subset) | Isolates conditional feasibility gain, not raw skill relevance |
| Baseline | CONFLICTGUARD-only agent | Tests whether index editing adds value beyond the gate |
| Baseline | VISUALSKILL static retrieval | Tests whether feasibility-conditioned keys beat static keys |
| Baseline | Agent S2 generalist | Tests against strong compositional retrieval without conflict handling |
| Baseline | Instruction-embedding + conflict-label concat before a standard retriever (no guarded-variant split) | Isolates whether the guarded-variant key is reproducible by simply appending the conflict label to the instruction embedding |
| Ablation-of-ours | Post-retrieval filter instead of key augmentation | Isolates the key-augmentation mechanism from mere gating |
| Ablation-of-ours | Random conflict label in the key (control) | Shows the gain is not from key dimensionality alone |
| Ablation-of-ours | Guarded-variant cap V ∈ {1, 3, 5} | Measures index size vs. conflict-task success, bounding the pruning cost |
| Calibration analysis | Reliability diagram on CONFLICTGUI conflict subset; report ECE and how guarded-variant retrieval degrades as a function of signature coarseness (number of distinct signatures collapsed) | Empirically bounds the exhaustiveness and calibration assumptions rather than asserting them |

### Why this setup validates the claim

The central claim is that making the retrieval key a function of a counterfactual feasibility assessment—rather than treating the index as static—lets the same instruction retrieve different skills depending on detectable conflicts, improving conflict-task success without degrading feasible-task success. CONFLICTGUI is the right dataset because it pairs feasible and conflict variants of the same instruction-GUI family, so any gain can be attributed to conflict conditioning rather than instruction coverage; the held-out AndroidWorld conflict split checks that the mechanism is not overfit to one taxonomy. The CONFLICTGUARD-only baseline tests the sub-claim that gating alone (abstain/terminate) is insufficient: it can stop bad actions but cannot route to a viable guarded skill variant, so if CSI's advantage is real it must appear as higher conflict-task completion, not just lower harmful-action rate. The VISUALSKILL static-retrieval baseline tests the sub-claim that feasibility must enter the key space: if static top-k already succeeds on conflict tasks, CSI's key editing is unnecessary. The instruction-embedding + conflict-label concat baseline is the decisive novelty control—it holds the gate signal fixed but uses a standard retriever without the guarded-variant split, so if CSI's gain disappears under this baseline, the mechanism is the guarded-variant key rather than the label alone. The post-retrieval-filter ablation holds the gate signal fixed but discards it for ranking, so if CSI's gain disappears under filtering, the mechanism is key-augmentation rather than gating. The random-label control shows the gain is not from key dimensionality alone. Conflict-task success rate at feasible-task parity is the correct metric because the predicted effect is a conditional improvement localized to conflict subsets; a uniform gain across all subsets would indicate a generic skill-retrieval improvement rather than the claimed conflict-conditioned effect. The calibration analysis (reliability diagram, ECE) and the sufficiency probe (ΔAUC of embed(x)+signature over embed(x) alone) directly test the load-bearing assumption that the signature is a sufficient statistic; if ΔAUC ≤ 0.02, the key-augmentation claim is not supported and we report a negative result.

### Expected outcome and causal chain

**vs. CONFLICTGUARD-only agent** — On a case where the instruction is feasible only under a specific UI conflict (e.g., a modal blocking the target button), the baseline produces an abstain or terminate with no fallback skill because its gate signal never re-enters retrieval. Our method instead concatenates the conflict signature into the key and retrieves the guarded variant that encodes the modal-dismissal policy, because the gate's label edits the key space rather than only halting execution. We expect a clear gap on the conflict subset—higher completion on blocked-but-recoverable tasks—with parity on feasible tasks where the gate proceeds and the base index is untouched.

**vs. VISUALSKILL static retrieval** — On a case where two instructions share high embedding similarity but differ in conflict status, the baseline produces the same top-k skill for both because its index is static with respect to feasibility, so the conflict variant retrieves a skill that ignores the blocking element. Our method instead returns different guarded entries for the two instructions because the conflict signature differentiates their keys. We expect the gap to concentrate on conflict tasks with near-duplicate feasible twins, and near-zero gap on conflict tasks whose instructions are already lexically distinct—the signature of key-space conditioning rather than general skill quality.

**vs. Agent S2 generalist** — On a case where a compositional subtask becomes infeasible mid-trajectory due to a detected conflict, the baseline produces continued execution of the infeasible subtask because its retrieval has no feasibility-conditioned variant to fall back to. Our method instead routes to the guarded variant whose guard policy matches the detected conflict type, because the outcome label and signature jointly select a conflict-specific skill. We expect a noticeable gap on multi-step conflict trajectories but rough parity on short feasible trajectories, consistent with the predicted localization.

**vs. Instruction-embedding + conflict-label concat baseline** — On a case where the conflict label is appended to the instruction embedding before a standard retriever, the baseline produces the same top-k skill as CSI on conflict tasks because the label is present in both keys; the difference is that CSI splits the index into a guarded variant with its own guard policy and reinforcement weight, whereas the concat baseline has no separate entry to reinforce or filter. We expect CSI to outperform this baseline on conflict tasks where the guarded variant's guard policy (abstain vs. terminate) matters, and to match it where the label alone suffices—isolating the guarded-variant split as the genuinely new component.

**vs. Post-retrieval filter ablation** — On a case where the gate detects a conflict but the filter only removes the blocked skill without rerouting, the baseline produces a fallback to the next-highest instruction-similarity skill, which may be infeasible for the same conflict. Our method instead retrieves a guarded variant whose key includes the conflict signature, so the fallback is conflict-conditioned. We expect the gap to appear specifically on conflict tasks where the top-k contains multiple skills infeasible under the same conflict, and to vanish where the second-ranked skill happens to be feasible.

**vs. Random conflict label control** — On a case where the key is augmented with a random conflict label, the baseline produces a guarded variant whose key is uncorrelated with the true conflict, so retrieval is no better than random routing among variants. Our method instead uses the gate's calibrated signature, so the guarded variant matches the detected conflict. We expect CSI to outperform the random-label control by a margin comparable to its gain over the post-retrieval filter, showing the gain is from the gate's signal rather than key dimensionality.

### What would falsify this idea

If CSI's gain over the post-retrieval-filter ablation is uniform across conflict and feasible subsets—or if the CONFLICTGUARD-only baseline matches CSI on blocked-but-recoverable tasks—then key augmentation is not doing the claimed work and the central claim is wrong. Likewise, if the gain appears on feasible tasks as much as conflict tasks, the effect is generic retrieval improvement rather than conflict-conditioned indexing; and if the sufficiency probe yields ΔAUC ≤ 0.02 or the random-label control matches CSI, the signature is not a sufficient statistic and the key-augmentation novelty collapses to gate + post-retrieval filter.

## References

1. Do GUI Agents Know When Not to Act? Enabling Conflict-Aware Termination for Multimodal GUI Agents
2. VISUALSKILL: Multimodal Skills for Computer-Use Agents
3. Agent S2: A Compositional Generalist-Specialist Framework for Computer Use Agents
4. Inducing Programmatic Skills for Agentic Tasks
5. Agent S: An Open Agentic Framework that Uses Computers Like a Human
6. AgentStore: Scalable Integration of Heterogeneous Agents As Specialized Generalist Computer Assistant
7. Proposer-Agent-Evaluator(PAE): Autonomous Skill Discovery For Foundation Model Internet Agents
8. Agent Workflow Memory
