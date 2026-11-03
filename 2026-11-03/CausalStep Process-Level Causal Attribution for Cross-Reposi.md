# CausalStep: Process-Level Causal Attribution for Cross-Repository Coding Agents via Dependency-Edge Ablation

## Motivation

CrossSWE and its sibling benchmarks (SWE-bench Pro, SWE-Exp) score only final repository states, so a Pass@1 of 58.48% cannot distinguish an agent that genuinely reasoned over cross-repository dependency structure from one that memorized recurring patch patterns that happen to satisfy tests; the 44.2% cross-repository-inconsistency failure rate is reported as a terminal outcome with no attribution to the intermediate reasoning steps that produced it. This reasoning-identifiability gap is structural, not incidental: because success is defined as a conjunction over end-state tests, any internal cross-repo step that is redundant with a memorized template is unobservable, and the field's shared assumption that end-state success suffices silently conflates coordination with recall across every benchmark in this tree.

## Key Insight

A cross-repository reasoning step is causally necessary for a coordinated edit exactly when intervening on the dependency edge it traverses — not merely observing the agent's emitted trace — flips the joint-edit outcome, so necessity is identifiable as a contrast between matched counterfactual rollouts that differ only in one edge's availability rather than as a correlation with final pass rates.

## Method

**What it is.** CausalStep is an evaluation instrument that wraps a CrossSWE-style agent with a dependency-graph state machine and produces, per instance, a *necessity score* for each intermediate cross-repository resolution step. Inputs: a multi-repo task with its repository dependency graph G = (R, E) (E = import/call/interface edges mined from build manifests and cross-repo symbol references), and the agent's rollout. Outputs: a per-edge causal-necessity vector and a per-instance attribution report distinguishing coordinated reasoning from pattern recall.

**Load-bearing assumption (stated explicitly).** The necessity contrast `J(C0) − max(J(C1), J(C2))` identifies the causal role of edge e only if the intervention actually removes all reconstructable evidence of e from the agent's visible context, so that the agent cannot route around the mask or perturb by recovering the symbol from sibling repos, test files, build manifests, or pretraining memory. When this fails, the contrast measures the agent's resourcefulness (redundancy of e) rather than e's causal role. We therefore certify each intervention before scoring (Phase 2b) and report non-identifiable edges as such rather than assigning them necessity zero.

**How it works.**

```pseudocode
PHASE 1  Trace capture (instrumentation, no model change)
  For each agent tool-call t during rollout on instance x:
    record (t, repo_r, symbol_s, edge_set E_t touched, emitted_edit_delta)
  Build step graph S = {nodes: resolution steps, edges: data/control dependence between steps}
  Map each S-node to the subset of E it traverses  ->  edge-attribution map A

PHASE 2  Matched counterfactual ablation (the causal core)
  for each edge e in E that A marks as load-bearing (|A^-1(e)| >= 1):
    construct THREE matched conditions, replayed from the SAME prefix checkpoint:
      C0 = unmodified rollout (reference)
      C1 = mask_e : hide cross-repo context reachable only via e (import stubs, interface signatures)
      C2 = perturb_e : flip e's direction / retarget to a structurally-similar decoy repo
    run agent to completion under each; score joint-edit pass J(.) in {0,1}
    necessity(e) = J(C0) - max(J(C1), J(C2))          # matched-contrast, bounded in [-1,1]
    specificity(e) = J(C0) - J(C1)  minus  J(C0) - J(C2)  (sign tells mask- vs direction-sensitivity)

PHASE 2b  Intervention certification (verified-intervention protocol)
  For each edge e and each intervention condition C1 (mask) and C2 (perturb):
    run a held-out probe agent (same base model, temperature 0) on the post-intervention context
    with the single instruction: "recover the definition/signature of symbol s reachable via e".
    If the probe returns the correct symbol definition (string match on signature) -> intervention FAILS.
    Additionally, scan the agent's own retrieval trace under C1/C2 for any tool-call that returns
    the masked symbol from an alternative path (sibling repo grep, test file, manifest).
    If either check succeeds -> mark e as NON-IDENTIFIABLE for that condition.
  Scoring rule:
    if C1 certified effective and C2 certified effective: report necessity(e) as above.
    if only C1 certified effective: report necessity(e) = J(C0) - J(C1), flag specificity as collapsed.
    if neither certified effective: report necessity(e) = NA (non-identifiable), exclude from CRI.

PHASE 3  Aggregation and attribution
  Necessity vector N = [necessity(e)] over load-bearing edges with certified interventions
  Coordinated-reasoning index  CRI(x) = mean_e necessity(e) * coverage(e)
     where coverage(e) = fraction of S-nodes traversing e that precede the final edit
  Report per-instance: (CRI, {e: necessity, specificity, certified}, residual = J(C0) - sum-contribution)
  Residual > tau (=0.2) flags success unexplained by any ablated edge -> pattern-recall suspect
```

Ablation budget is capped at K=8 edges per instance (top-K by coverage), replay is from a stored prefix checkpoint so C0/C1/C2 share identical pre-ablation history, and each condition is run with 3 seeds with majority vote on J to suppress sampling noise. Decoy repos in C2 are selected by graph-isomorphism of their local dependency neighborhood, so C2 perturbs *which* repo is depended on without perturbing task difficulty. The certification probe in Phase 2b uses the same base model as the agent under evaluation, temperature 0, and a fixed prompt template (provided in supplementary); it adds 2 probe rollouts per edge per condition (C1, C2), i.e., at most 2*K*2 = 32 extra short rollouts per instance, each capped at 3 tool calls.

**Why this design.** We chose matched counterfactual replay from a shared prefix checkpoint over simply ablating edges in the prompt because observational trace analysis cannot separate a step the agent *used* from a step it *mentioned*, accepting the cost of K× extra rollouts per instance. We chose *two* intervention types (mask and perturb) over a single mask because masking conflates "this edge was unused" with "the agent routed around the mask," whereas the perturb condition forces traversal of a wrong edge and exposes direction-sensitive reasoning; the trade-off is that perturb requires constructing graph-isomorphic decoys, which is not always possible for singleton-dependency repos. We chose a bounded necessity contrast J(C0) − max(J(C1),J(C2)) over a Shapley-style marginal contribution because Shapley values over non-commutative edit sequences are not well-defined when later steps depend on earlier edits, accepting a coarser but replay-valid per-edge score. The certification step (Phase 2b) is added because the load-bearing assumption — that the intervention removes all reconstructable evidence of e — is not guaranteed by construction; without certification, the contrast would conflate e's necessity with the agent's ability to route around the mask, which is a property of the agent, not of the reasoning step. This is not a variant of process reward models (Lightman 2023) or per-step Monte-Carlo value estimation (VinePPO): those *learn or estimate* a value for each step from outcome statistics, whereas CausalStep *intervenes on the environment's dependency graph* and reads off a contrast — the signal is generated by an experimental manipulation the agent cannot observe, not by a value head fit to the same rollouts. Nor is it descriptor-plus-retrieval: nothing is matched against a library; the quantity is a counterfactual outcome difference.

**Why this measures what we claim.** `necessity(e) = J(C0) − max(J(C1), J(C2))` measures *causal necessity of the cross-repository reasoning step traversing e* because the matched-prefix replay makes C0 and C1/C2 differ only in edge e's availability, so under the assumption that the agent's policy is a deterministic function of its visible context (plus bounded sampling noise removed by majority-vote over 3 seeds) **and that the intervention is certified to remove all reconstructable evidence of e (Phase 2b)**, any outcome change is attributable to e; this assumption fails when the agent has a fallback path that reconstructs the masked context from other repos, in which case necessity(e) reflects *redundancy of e* rather than its causal role — and we report such edges as non-identifiable rather than assigning them necessity zero. `specificity(e)` measures *whether the step required the correct dependency direction/identity versus merely any cross-repo context* because the mask and perturb conditions differ only in whether the substituted repo is graph-isomorphic to the true one; this fails when no isomorphic decoy exists, in which case specificity collapses to the mask contrast alone and cannot distinguish direction-sensitivity from context-hunger. `CRI(x)` measures *the degree to which instance success is attributable to coordinated cross-repo reasoning* because it weights each edge's necessity by the fraction of pre-edit steps traversing it, under the assumption that load-bearing edges identified by coverage are a superset of causally relevant ones; this fails when a necessary step touches no mined edge (e.g., implicit runtime coupling), in which case CRI under-counts and the residual term absorbs the unexplained success, which is exactly the pattern-recall flag we report rather than hide. The residual term additionally assumes that success unexplained by any ablated load-bearing edge implies template memorization rather than an unmined implicit coupling; this assumption fails when implicit runtime coupling inflates the residual, in which case the residual reflects missing edge coverage rather than recall — we test this by comparing residual-flagged successes against a patch-template nearest-neighbor similarity score (see experiment).

## Contribution

(1) CausalStep, a process-level instrumentation and matched-counterfactual ablation protocol that converts a cross-repository agent rollout into a per-dependency-edge causal-necessity and specificity vector, yielding a Coordinated-Reasoning Index and a residual-based pattern-recall flag. (2) The design principle that cross-repo reasoning identifiability requires *two* intervention types (context masking and graph-isomorphic edge perturbation) replayed from a shared prefix checkpoint, since masking alone cannot separate unused dependencies from rerouted ones. (3) An open attribution report format and analysis over CrossSWE instances that partitions end-state success into edge-attributable coordination versus unexplained residual, providing the first step-level diagnostic layer for multi-repository coding benchmarks.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | CrossSWE multi-repo instances (full test split, N=500) | Native cross-repo dependency graphs, joint-edit ground truth |
| Primary metric | CRI–Pass@1 correlation (per-instance) and residual-flag rate | Detects whether attribution tracks genuine coordinated reasoning |
| Baseline | GPT-6.1 Sol Pass@1 | Strongest reported agent, upper-bound recall suspect |
| Baseline | Trace-only step salience | Observational alternative our causal core must beat |
| Baseline | Single-mask ablation only | Tests whether perturb condition adds direction signal |
| Ablation-of-ours | CausalStep without perturb (mask-only) | Isolates contribution of the C2 direction probe |
| Ablation-of-ours | CausalStep without certification (Phase 2b removed) | Tests whether verified-intervention protocol changes attribution |
| Validation metric | Patch-template nearest-neighbor similarity | Independent operationalization of pattern recall |
| Calibration | J-variance across seeds on C0 | Bounds noise floor for necessity contrast |

### Why this setup validates the claim

The central claim is that CausalStep's matched counterfactual necessity scores identify *which* cross-repo steps are causally load-bearing and separate coordinated reasoning from pattern recall. CrossSWE is the right dataset because its instances genuinely require traversing mined import/interface edges, so edge-attribution has real ground truth to align against. The primary metric — the per-instance correlation between CRI and joint-edit success, plus the residual-flag rate — is the right detector because the theory predicts a specific shape: successful instances should show high CRI (load-bearing edges are necessary), while a subset of successes should show high residual (success unexplained by any ablated edge), which is exactly the pattern-recall signature. The trace-only salience baseline tests the sub-claim that observational analysis conflates *used* with *mentioned*: if it matches CausalStep's attribution, our causal core is unnecessary. The single-mask baseline and the mask-only ablation test whether the perturb condition earns its cost by exposing direction-sensitivity that masking cannot. The certification ablation tests whether the verified-intervention protocol changes the attribution: if removing certification leaves CRI–Pass@1 correlation unchanged, the load-bearing assumption is empirically harmless on this dataset; if it changes, certification is necessary. The patch-template nearest-neighbor similarity provides an independent operationalization of pattern recall: we train a 1-NN classifier on the training split using normalized edit-distance between the agent's emitted patch and the nearest training patch for the same repository pair; residual-flagged successes should have significantly higher template similarity than high-CRI successes, validating that the residual measures recall rather than noise. The calibration experiment (J-variance across seeds on C0) bounds the noise floor: we report necessity values only where the contrast exceeds the empirically measured floor (e.g., 2 standard deviations of J(C0) across seeds). Together these form a falsifiable test: the method must produce a *concentrated* signal (high CRI on coordinated successes, high residual on recall successes) that the observational baseline cannot reproduce, and the residual must align with an independent recall signature.

### Expected outcome and causal chain

**vs. GPT-6.1 Sol Pass@1** — On an instance where the agent must edit repo A's interface and propagate the signature change into dependent repo B, the baseline may reach a passing joint edit by recalling a common refactor pattern and editing B speculatively without ever traversing the mined A→B edge. Our method masks and perturbs that edge, replaying from the same prefix; if the edit survives masking, necessity(e) collapses toward zero and the residual term stays high, flagging the success as pattern-recall. We expect the baseline's reported 58% Pass@1 to split into a coordinated-reasoning subset with high CRI and a recall-suspect subset with residual above tau, so CausalStep reveals a meaningful gap between raw success and causally-grounded success that Pass@1 alone hides. We further expect residual-flagged successes to have significantly higher patch-template nearest-neighbor similarity than high-CRI successes (e.g., mean similarity 0.8 vs 0.4, p<0.01 by Mann-Whitney U), confirming that the residual measures recall rather than noise.

**vs. Trace-only step salience** — On a case where the agent's trace explicitly mentions repo B's interface in its reasoning but the eventual edit to B is driven by a default value that happens to satisfy the test, trace salience ranks the B-edge as highly used. Our method's mask condition leaves the outcome unchanged, driving necessity(e) to zero, contradicting the trace. We expect the two attributions to disagree on a substantial fraction of load-bearing edges — specifically, trace salience should over-rank edges the agent *mentioned* but did not depend on, producing a noticeable divergence concentrated on instances with high residual.

**vs. Single-mask ablation only** — On an instance where the correct dependency direction is A→B but a graph-isomorphic decoy repo C sits in the same neighborhood, masking the A→B edge lets the agent route through C and still pass, yielding necessity near zero under mask-only. Our perturb condition retargets to C and forces traversal of the *wrong* repo, so a direction-sensitive agent fails and specificity becomes positive. We expect the mask-only ablation to systematically under-report necessity on direction-sensitive instances, while the full CausalStep recovers it — a gap concentrated exactly on the singleton-decoy subset where C2 is constructible.

**vs. CausalStep without certification** — Without Phase 2b, edges where the agent can reconstruct the masked symbol from sibling repos or test files will be assigned necessity zero (because the agent routes around the mask), under-reporting CRI on coordinated successes. With certification, such edges are marked non-identifiable and excluded from CRI, so CRI should increase on instances where reconstruction is possible. We expect the certification ablation to show a significant drop in CRI–Pass@1 correlation (e.g., from 0.6 to 0.4) on the subset of instances where the probe agent successfully recovers the masked symbol, confirming that the verified-intervention protocol is necessary for valid attribution.

### What would falsify this idea

If CRI gains over the trace-only baseline are uniform across subsets rather than concentrated on instances where the predicted failure mode (mentioned-but-unused edges, direction-sensitive decoys) actually occurs, the causal-core claim is wrong. Likewise, if residual-flagged successes do not align with any independently observable recall signature (e.g., they pass equally when the entire cross-repo context is masked, or their patch-template similarity is not higher than high-CRI successes), then the residual is measuring noise rather than pattern recall.

## References

1. CrossSWE: Benchmarking Cross-Repository Joint Development for Coding Agents
