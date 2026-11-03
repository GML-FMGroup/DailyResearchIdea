# Online Discovery of Coordination-Fault Labels from Execution-Trace Residuals

## Motivation

Existing multi-agent orchestration systems condition their planners on a closed, pre-enumerated failure taxonomy: DynaFlow trains its main agent using supervision restricted to three MAST-derived defect types (unsupported premises, biased verification, information-flow defects), while MovieAgent and StoryAgent evaluate against fixed plan-schema and role-consistency violations. This closure assumption is structural, not incidental — the label set is derived from a small annotated trace corpus (MAST's 150 expert-analyzed traces) and is never revised as new execution traces are observed, so any coordination fault whose signature was absent from the annotation corpus is permanently invisible to the orchestrator's supervision and reward.

## Key Insight

A coordination fault is identifiable from a single execution trace exactly when its message/plan-graph signature has a nonzero component orthogonal to the span of signatures already admitted as labels, so the supervision target can be grown monotonically by a residual-energy test whose threshold is set by the trace distribution itself rather than by a human annotator.

## Method

```pseudocode
# COORDINATE: Coverage-Oriented Residual Discovery for Identifiability of New Task-failure labels
# Inputs:  orchestrator policy pi (e.g., DynaFlow main agent), sub-agent pool A, task stream D
# Outputs: monotonically growing label set L = {l_1..l_K}, each with a signature basis vector phi_k,
#          and a retrained fault classifier f_theta over L

# Load-bearing assumption (stated explicitly):
#   Coordination faults leave signatures lying in distinct linear subspaces of the
#   size-normalized, edge-type-stratified message-graph feature vector, so that the
#   residual norm after projecting onto the span of admitted label signatures measures
#   the presence of a genuinely novel fault rather than the magnitude of ordinary
#   within-task variation. This assumption is verified empirically in Phase 0 and
#   monitored at runtime via the principal-angle drift diagnostic (Phase 2b).

Phase 0 — Subspace-assumption pre-check (run once on MAST-Data expert-labeled traces)
for each expert-labeled fault type c in MAST-Data (7 frameworks):
    S_c = { phi(tau) : tau labeled c }              # per-fault signature set
    U_c = top-3 principal subspace of S_c           # via SVD on centered S_c
for each pair (c, c'):
    theta_{c,c'} = principal angles between U_c and U_{c'}   # vector of 3 angles
report median and 10th-percentile of min-angle over pairs; if median min-angle < 15 deg,
    flag the linear-subspace premise as violated for that framework and report it.

Phase 1 — Signature extraction (per trace)
for each task x in D:
    tau = rollout(pi, A, x)                      # full inter-agent message/plan graph
    G   = build_graph(tau)                       # nodes = agent calls; edges = (output, consumption)
    for each edge e = (u -> v) in G:
        s_e = [ precond_satisfied(v, out(u)),     # binary: does out(u) meet v's declared precondition?
                info_type_match(out(u), in(v)),   # categorical: schema/type agreement (one-hot, 4 levels)
                temporal_order_ok(u, v),          # binary: did v run before u finished?
                redundant_consumption(u, v) ]     # binary: was out(u) also consumed by a sibling?
        # edge-type stratum id: (info_type_match level, temporal_order_ok, redundant_consumption) -> 16 strata
    # size-normalized, edge-type-stratified aggregation (replaces sum-pooling):
    for each stratum g in {1..16}:
        phi_g(tau) = mean_{e in stratum g} s_e        # mean, not sum -> invariant to |E|
    phi(tau) = concat_{g=1..16} phi_g(tau)            # fixed-length vector in R^d, d = 16 * 4 = 64
    # (strata with zero edges contribute the zero vector; graph size enters only via
    #  which strata are non-empty, not via the magnitude of phi)

Phase 2 — Residual test against current label span (per edge-type stratum)
L = {}; Phi_L = empty basis matrix (d x 0)
for each new trace tau:
    for each stratum g with at least one edge in tau:
        r_g = phi_g(tau) - Proj_{span(Phi_L[g])}(phi_g(tau))   # residual within stratum g
        if ||r_g||_2 > gamma_tau(g, size_bucket(tau)):          # matched-stratum threshold, see Phase 3
            admit new label l_{K+1} with signature r_g / ||r_g||_2 in stratum g
            Phi_L[g] = [Phi_L[g] ; r_g / ||r_g||_2]            # Gram-Schmidt against existing basis in g
            K = K + 1
    # a trace may admit at most one new label per stratum per step (co-occurrence handled
    # by separate strata; superposition within a single edge is reported as a limitation)

Phase 2b — Runtime assumption diagnostic (principal-angle drift)
for each admitted label l_k in stratum g:
    theta_drift(k) = min principal angle between U_k (top-3 subspace of traces assigned to l_k
                     in the current window W) and the stored basis vector phi_k
if median theta_drift over labels > 30 deg:
    flag 'subspace assumption drifting' in the run log; report per-label drift in results.

Phase 3 — Trace-derived identifiability threshold (size- and type-matched null)
gamma_tau(g, b) = quantile_{0.95}( ||phi_g(tau') - Proj_{span(Phi_L[g])}(phi_g(tau'))||_2
                                   for tau' in W_g,b )
# W_g,b = sliding window of the last 200 traces that (i) have at least one edge in stratum g
#         and (ii) fall in graph-size bucket b (b = quartile of |E| over the last 1000 traces);
#         the 95th percentile is the null residual scale under the current label span within
#         the matched stratum and size bucket, so a new label is admitted only when the trace's
#         residual exceeds the noise floor of the matched trace distribution itself.

Phase 4 — Retrain fault classifier on grown label set
whenever K increases:
    relabel the window W with f_theta (argmax over L);
    fine-tune f_theta on (phi(tau), argmax_{l in L} cos(phi(tau), phi_l)) for tau in W
    with cross-entropy; freeze Phi_L.
return L, Phi_L, f_theta
```

**Why this design.** We chose a *residual-energy* admission rule over a *classifier-confidence* admission rule because classifier confidence on a closed label set is structurally incapable of signalling the existence of an unseen label — the softmax is normalized over L, so a novel fault is forced onto the nearest existing class with high confidence; the residual rule instead measures distance from the span of admitted signatures, which is well-defined even when L is empty. We chose a *trace-derived quantile threshold* gamma_tau over a fixed epsilon because coordination-fault signatures have task-dependent scale (a retrieval trace has more edges than a math trace), so a global epsilon either over-admits on dense graphs or under-admits on sparse ones; the sliding-window quantile adapts the noise floor to the current task distribution, accepting the cost that the label set can grow during distribution shift. We chose *orthogonalized admission* (Gram–Schmidt against Phi_L) over simple clustering because clustering re-partitions existing labels and violates monotonicity, whereas orthogonalization guarantees that an admitted label contributes a genuinely new direction, accepting the cost that near-duplicate faults are merged rather than separately named. We chose to *freeze Phi_L during classifier retraining* over joint optimization because joint optimization can rotate the basis and retroactively invalidate earlier admissions, breaking the monotonic-growth property that is the whole point. A domain expert would not describe this as a variant of DynaFlow's filtered SFT: DynaFlow filters a fixed corpus for three known defect types, whereas COORDINATE's supervision target is a function of the trace geometry observed so far.

**Why it measures what we claim.** The residual norm ||phi_g(tau) - Proj_{span(Phi_L[g])}(phi_g(tau))||_2 measures *label novelty* because we assume that coordination faults leave signatures lying in distinct linear subspaces of the size-normalized, edge-type-stratified feature space — an assumption that holds when the fault is localized to a specific edge type (e.g., precondition violation produces a signature in the precond_satisfied coordinate that no other admitted label spans); this assumption fails when two faults co-occur on the same edge and their signatures superpose, in which case the residual reflects the *combined* fault rather than a single novel one, and it fails when the per-stratum mean is dominated by within-task variation, in which case the residual reflects task density rather than fault identity (Phase 0 pre-check and Phase 2b drift diagnostic flag both failure modes). The threshold gamma_tau(g, b) measures *identifiability* because we assume the 95th-percentile residual over the size- and type-matched sliding window is a consistent estimator of the null residual scale under the current label span; this assumption holds when the matched window is stationary; it fails under abrupt task-distribution shift, in which case gamma_tau reflects the shift magnitude rather than the noise floor. The orthogonalized basis Phi_L measures *coverage* because we assume that the span of admitted signatures is the set of fault directions the classifier can already discriminate; this fails when a fault's signature is a nonlinear function of admitted directions, in which case coverage is underestimated and the label set grows more slowly than the true fault diversity.

## Contribution

['COORDINATE, an online label-admission procedure that treats the multi-agent failure taxonomy as a partially observed object and grows it monotonically via a residual-energy test against the span of admitted fault signatures, replacing the fixed MAST-derived label set used by DynaFlow.', 'A trace-derived identifiability threshold (sliding-window 95th-percentile residual) that adapts the admission criterion to the task distribution without human annotation, converting the annotation bottleneck into a coverage-estimation problem.', 'An orthogonalized signature basis that guarantees each admitted label contributes a genuinely new direction in message-graph feature space, enabling classifier retraining without retroactively invalidating earlier admissions.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | MAST-Data traces, 3 of 7 frameworks (AutoGen, MetaGPT, ChatDev) | Real coordination-fault traces; scoped for feasibility |
| Primary metric | Novel-label recall on held-out fault types | Directly tests discovery of unseen faults |
| Secondary metric | Task success rate on held-out tasks after feeding discovered labels into planner conditioning | Ties discovery to orchestration performance |
| Baseline | DynaFlow filtered SFT | Fixed three-defect supervision, no discovery |
| Baseline | Closed-set softmax classifier | Confidence cannot signal unseen labels |
| Baseline | K-means clustering over signatures | Re-partitions labels, breaks monotonic growth |
| Baseline | Incremental PCA / streaming OOD detector on phi(tau) | Prior open-set subspace-tracking method |
| Ablation-of-ours | COORDINATE with fixed epsilon threshold | Isolates trace-derived quantile contribution |
| Ablation-of-ours | COORDINATE with sum-pooling (no size normalization) | Isolates size-normalization contribution |
| Diagnostic | Phase 0 principal-angle pre-check on expert labels | Tests linear-subspace premise before admission |
| Diagnostic | Phase 2b principal-angle drift | Flags runtime assumption violation |

Compute budget: 3 frameworks × ~50 expert traces each = 150 traces for Phase 0; main stream = 2,000 rollouts per framework (6,000 total); classifier retraining = 2-layer MLP (hidden=256, GeLU), ≤20 retrains per run, ~0.5 GPU-hour per retrain on a single A100 → ≤10 GPU-hours total; rollouts at ~30 s each → ~50 GPU-hours. Total ≤60 GPU-hours, feasible in 3–6 months.

### Why this setup validates the claim

The central claim is that residual-energy admission over an orthogonalized signature basis discovers genuinely new coordination-fault labels that closed-set supervision cannot name, while remaining monotone. MAST-Data is the right substrate because it contains expert-annotated failure traces spanning frameworks, so held-out fault types give a ground-truth notion of "unseen label" that we can withhold during admission and score against afterward; scoping to three frameworks (AutoGen, MetaGPT, ChatDev) keeps the compute budget at ≤60 GPU-hours while retaining cross-framework diversity. The primary metric — recall on withheld fault types — is the only metric that directly detects the predicted effect: a method that forces novel faults onto known classes will score near zero on it regardless of its closed-set accuracy, so it cleanly separates our mechanism from confidence-based ones. The secondary metric — task success rate after feeding discovered labels into planner conditioning — ties the discovery claim to orchestration performance rather than to label recall alone. DynaFlow filtered SFT tests the sub-claim that fixed-corpus supervision cannot grow its label set; the closed-set softmax baseline tests the sub-claim that softmax confidence is structurally blind to novelty; K-means tests the sub-claim that clustering violates monotonicity; incremental PCA / streaming OOD detection tests the sub-claim that our residual rule is not merely a re-implementation of prior subspace tracking. The fixed-epsilon ablation is the load-bearing control: it shares our residual machinery but removes the trace-derived quantile, so any gain concentrated on dense-graph subsets is attributable to adaptive thresholding rather than to residual admission itself. The sum-pooling ablation isolates the size-normalization contribution: if our gains vanish when we revert to sum-pooling, the size-normalization repair is doing the work. The Phase 0 principal-angle pre-check reports whether the linear-subspace premise holds on expert labels before any admission rule is built on it; if median min-angle < 15°, we report the premise as violated for that framework and interpret admission results accordingly.

### Expected outcome and causal chain

**vs. DynaFlow filtered SFT** — On a trace where a novel fault (e.g., a temporal-order violation between planner and executor) appears, DynaFlow's fixed three-defect supervision assigns it to the nearest known defect because its label set is closed by construction. Our method instead computes the per-stratum residual against the admitted signature span, sees energy outside every known direction, and admits a new label. We expect a clear gap on the withheld-fault subset and near-parity on the three known-defect subsets.

**vs. Closed-set softmax classifier** — On a case where a genuinely new fault fires, the softmax classifier returns high confidence on an existing class because the softmax is normalized over L and must place mass somewhere. Our method ignores classifier confidence during admission and thresholds the per-stratum residual norm instead, so the novel direction is admitted rather than absorbed. We expect our novel-label recall to be substantially higher while closed-set accuracy on known faults stays comparable.

**vs. K-means clustering over signatures** — On a stream where a new fault arrives after several labels are established, K-means re-partitions all points and relabels earlier traces, so previously named faults change identity and monotonicity breaks. Our orthogonalized admission appends a new basis direction without touching earlier ones, so earlier labels are stable. We expect label-identity churn to be near zero for us and visibly nonzero for K-means, with our advantage growing as more labels accumulate.

**vs. Incremental PCA / streaming OOD detector** — On a stream where a new fault arrives, incremental PCA updates its principal subspace using all data and can rotate earlier components, while a streaming OOD detector thresholds a reconstruction error without an orthogonalized label basis. Our method freezes Phi_L during retraining and admits only directions orthogonal to the existing span. We expect comparable novel-label recall on the first few faults but higher label-identity stability (lower churn) and higher recall on later faults as the basis accumulates.

**vs. COORDINATE with fixed epsilon** — On a mixed stream of sparse retrieval traces and dense planning traces, a fixed epsilon either over-admits on dense graphs (noise exceeds epsilon) or under-admits on sparse ones (true novelty falls below epsilon). Our trace-derived quantile rescales the threshold to each matched stratum and size bucket's null residual scale. We expect our gain over the ablation to concentrate on the dense-graph subset, with parity on sparse traces where the two thresholds coincide.

**vs. COORDINATE with sum-pooling** — On a stream mixing dense retrieval traces (many edges) and sparse math traces (few edges), sum-pooling makes the residual norm scale with |E|, so the threshold admits labels for task-density shifts rather than fault types. Our mean-pooling per stratum removes the |E| dependence. We expect our gain over this ablation to concentrate on the dense-graph subset and on the size-bucket-matched evaluation, with parity on traces of uniform size.

**Task success rate (secondary metric)** — After discovery, we feed the grown label set L into the orchestrator's planner conditioning (as in DynaFlow's SFT loop) and measure task success rate on held-out tasks. We expect a positive gap over DynaFlow filtered SFT on tasks whose failures involve withheld fault types, and near-parity on tasks whose failures involve the three known defect types.

### What would falsify this idea

If our advantage over the closed-set softmax baseline is uniform across withheld-fault and known-fault subsets rather than concentrated on withheld faults, then residual admission is not doing what we claim and the gain is a generic classifier effect. Likewise, if the fixed-epsilon ablation matches us on dense-graph traces, the trace-derived quantile contributes nothing and the identifiability argument is wrong; if the sum-pooling ablation matches us, the size-normalization repair contributes nothing; and if the Phase 0 principal-angle pre-check reports median min-angle < 15° on the expert labels, the linear-subspace premise is violated and the residual-energy admission rule is not measuring label novelty as claimed.

## References

1. DynaFlow: Learning Free-Form Multi-Agent Orchestration
2. Automated Movie Generation via Multi-Agent CoT Planning
3. StoryAgent: Customized Storytelling Video Generation via Multi-Agent Collaboration
4. Why Do Multi-Agent LLM Systems Fail?
5. Challenges in Human-Agent Communication
6. AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation
7. The Rise and Potential of Large Language Model Based Agents: A Survey
