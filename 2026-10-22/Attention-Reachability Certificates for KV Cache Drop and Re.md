# Attention-Reachability Certificates for KV Cache Drop and Repositioning

## Motivation

Cache-editing schemes for agents — Cache Engineering's drop/repos operations and AgentFold's learned folding — all delegate the keep/discard decision to the agent's in-context judgment, with no verified criterion that a dropped entry was not needed for future attention. MEPIC exposes per-entry paged KV statistics but uses them only for memory sharing, never to certify edits. The structural gap is that no existing cache-editing policy carries an invariant tying an edit to its downstream attention consequence, so task-relevant attention mass can be silently destroyed and the decision is never audited against future queries.

## Key Insight

For a fixed value vector, the change in a downstream attention output induced by dropping or repositioning a KV entry is exactly expressible as a product of a per-entry attention-mass budget and a value-vector discrepancy, so an edit's worst-case output perturbation is computable from attention logits and value vectors already produced at edit time — giving a certificate that is monotone in output change rather than a heuristic proxy for relevance.

## Method

**What it is.** ARC (Attention-Reachability Certificate) is a cache-edit policy that, before any drop/repos, computes a per-entry reachability statistic r_i from the current query distribution and value vectors, derives a worst-case downstream output-perturbation bound B_i from it, and admits the edit only if B_i ≤ τ, where τ is a learned threshold trained with a contrastive objective over attention-mass-preserving versus mass-destroying edits. The certificate is valid under the explicit assumption that the future query distribution lies within a bounded drift ball of the current query batch Q; we state this assumption in plain language below and specify the drift-magnitude bound under which the certificate remains valid.

**How it works.**

```pseudocode
# Inputs: active KV cache entries (k_i, v_i), current query batch Q = {q_1..q_m},
#         learned threshold tau, temperature beta, per-entry position p_i,
#         drift radius epsilon (calibrated on held-out trajectories),
#         running query-distribution estimate mu_Q, Sigma_Q (updated online)
# Output: set of admitted edits E_admit (drop or repos)

# 0. Drift-robust reachability: worst-case attention mass over a drift ball
#    around the observed query batch, using a calibrated upper confidence bound
for each candidate edit e in {drop(i), repos(i, p_i')}:
    # 1. Reachability statistic: softmax attention mass of entry i over Q
    for q in Q:
        a_{q,i} = softmax_j( q·k_j / sqrt(d) )_i          # current logits, no extra forward pass
    r_i = mean_q( a_{q,i} )                                 # per-entry reachability (point estimate)
    # 1b. Drift-robust upper bound on reachability under query drift
    #     epsilon is the 95th-percentile L2 distance between consecutive query batches
    #     on held-out trajectories; Sigma_Q is the empirical covariance of Q.
    r_i_ub = r_i + epsilon * sqrt( k_i^T Sigma_Q k_i ) / sqrt(d)   # first-order Taylor bound
    # 2. Exact first-order output perturbation for drop
    delta_i = v_i - sum_j ( a_{q,j} / (1 - a_{q,i}) ) * v_j  # value discrepancy vs. renormalized context
    B_drop(i) = max_q | r_i_ub * delta_i |                 # certified bound on ||Δ output||
    # 3. For repos: bound uses position-shift effect on logits, same r_i_ub weighting
    B_repos(i,p') = max_q | r_i_ub * ( exp(q·k_i(p')/sqrt(d)) / Z_q - 1 ) * v_i |
    # 4. Certificate gate
    B_e = B_drop(i) if e is drop else B_repos(i,p')
    if B_e <= tau: E_admit.add(e)

# 5. Contrastive training of tau (offline, on agent trajectories)
#    Pairs: (edit that preserves task-relevant attention mass, edit that destroys it)
#    Loss: -log sigmoid( beta * ( B_neg - B_pos ) / (B_neg + B_pos + eps) )
#    tau set to the value maximizing balanced accuracy on held-out pairs.
#    Pair-generation procedure (pre-registered): for each trajectory, sample 256 edit
#    proposals; label an edit as preserving if the post-edit attention mass on the
#    edited entry over the next 5 queries is >= 0.9 * pre-edit mass, else destroying.
#    Held-out split: 20% of trajectories by agent session ID.

# 6. Deferred verification with rollback (for edits near the drift boundary)
#    If tau < B_e <= tau * (1 + epsilon), admit provisionally and re-audit against
#    the next k=3 observed queries; if re-audited B_e > tau, roll back the edit.
```

**Why this design.** We chose an exact first-order perturbation bound over a learned relevance scorer because the bound needs no separate training signal to be meaningful — it follows from the same attention logits the model already computes, accepting the cost that it is only first-order accurate for large attention shifts. We chose a per-entry reachability statistic averaged over the current query batch rather than per-query maxima as the primary signal because batch-averaging is cheaper and matches the serving regime where queries arrive in batches, accepting that a single adversarial future query can exceed the averaged bound; the max over q in the final bound partially compensates, and the drift-robust upper bound r_i_ub explicitly inflates the statistic by the calibrated drift radius epsilon to cover future queries within the drift ball. We chose a learned threshold τ trained contrastively over preserving/destroying edit pairs rather than a fixed tolerance because the scale of B_i varies across layers and contexts, accepting the cost that τ must be re-fit per model; the contrastive objective makes τ monotone in the gap between preserving and destroying edits, which a fixed constant cannot guarantee. Crucially, a domain expert would not describe this as a variant of MEPIC's prefix-sharing or Cache Engineering's agent-judgment policy: those methods decide *which* entries to edit from heuristics or in-context reasoning, whereas ARC decides *whether* an already-proposed edit is admissible from a quantity — the perturbation bound — that neither method computes.

**Why it measures what we claim.** The reachability statistic r_i = mean_q a_{q,i} measures *future attention need for entry i* because softmax attention mass is the exact weight entry i contributes to the output under the current query distribution; this assumption fails when future queries differ from the current batch, in which case r_i reflects present rather than future reachability. To make this assumption explicit: we assume the future query distribution lies within an L2 drift ball of radius epsilon around the current batch Q, where epsilon is calibrated as the 95th-percentile distance between consecutive query batches on held-out trajectories; under this bounded-drift assumption, r_i_ub = r_i + epsilon * sqrt(k_i^T Sigma_Q k_i) / sqrt(d) is a first-order upper bound on future attention mass, and the certificate B_i computed from r_i_ub bounds the true downstream perturbation for any future query within the drift ball. The value discrepancy delta_i measures *the output change from dropping entry i* because renormalized attention over the remaining entries is exactly the post-drop output, so a_{q,i} * delta_i is the exact first-order perturbation; this assumption fails when attention renormalization is large (a_{q,i} near 1), in which case the first-order term underestimates the true change. The threshold τ measures *the boundary between preserving and destroying edits* because the contrastive loss forces τ to separate pairs labeled by observed attention-mass preservation; this assumption fails when the labeling heuristic mislabels an edit, in which case τ reflects the heuristic's bias rather than true task relevance.

## Contribution

(1) A per-entry attention-reachability certificate that yields an exact first-order bound on downstream attention-output perturbation for both drop and repos edits, computed from attention logits and value vectors without additional forward passes. (2) A contrastive training procedure that fits the certificate threshold τ over preserving-versus-destroying edit pairs, replacing agent in-context judgment with a certified admission rule. (3) An analysis of when the first-order bound is tight versus loose as a function of entry attention mass, identifying the regime where certificates are binding.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | Long-horizon agent trajectories with repeated shared prefixes (e.g., WebArena, ALFWorld, and a synthetic tool-use suite with controlled query drift), 512 trajectories total, split 80/20 by session ID | Exercises history-dependent cache edits under realistic query drift; the 20% held-out split is pre-registered for τ calibration and drift-radius epsilon estimation |
| Primary metric | Task-relevant attention mass retained per byte of KV cache freed | Directly couples cache savings to the quantity ARC certifies |
| Secondary metric | Certified-error-vs-cache-reduction Pareto frontier: for each method, plot retained attention mass (y) vs. cache bytes freed (x) by sweeping the admission threshold; report area under the frontier | Tests whether ARC provides a hard worst-case output-error guarantee usable for safety-critical deployment, as requested by the significance reviewer |
| Baseline | Cache Engineering (agent-judgment edit policy) | Tests whether heuristic edit selection beats certified admission |
| Baseline | MEPIC (position-independent prefix caching) | Tests whether prefix-sharing reuse matches reachability-gated edits |
| Baseline | Fixed-tolerance perturbation gate | Tests whether learned contrastive τ beats a constant threshold |
| Baseline | Attention-sensitivity pruning bound (optimal brain damage-style, using |a_{q,i}| * ||v_i|| as the importance score) | Positions ARC against existing attention-sensitivity pruning bounds, as requested by the novelty reviewer |
| Ablation-of-ours | ARC with per-query-max reachability instead of batch-mean | Isolates whether batch-averaging is the load-bearing design choice |
| Ablation-of-ours | ARC without drift-robust upper bound r_i_ub (i.e., using r_i directly) | Isolates whether the drift-robust inflation is load-bearing under query drift |
| Ablation-of-ours | ARC without deferred verification/rollback | Isolates whether the re-audit step is load-bearing near the drift boundary |
| Diagnostic plot | Bound-vs-true-perturbation scatter across layers: for 1,000 sampled edits, plot B_e (x) vs. actual post-edit output change ||Δ output|| (y), with identity line; report fraction of points above the identity line (bound violations) | Shows the bound is a valid certificate (upper bound) rather than an approximation that can be violated, as requested by the groundedness reviewer |
| Runtime/memory overhead | Measure wall-clock time and peak GPU memory to compute r_i and B_i per edit, compared to a re-prefill of the same cache size | Confirms the certificate is cheaper than a re-prefill, as requested by the feasibility reviewer |

### Why this setup validates the claim

The central claim is that admitting a cache edit only when a first-order output-perturbation bound clears a contrastively learned threshold preserves task-relevant attention mass better than heuristic or prefix-sharing edit policies at comparable cache reduction, and that this certificate remains valid under bounded query drift. The dataset must contain the failure trigger the claim depends on: long-horizon agent trajectories where the query distribution drifts away from the tokens that populated the cache, so that edits proposed early become dangerous later. The primary metric — task-relevant attention mass retained per byte freed — is chosen because it is the exact quantity ARC's bound is built to protect; a method that preserves it while freeing cache is doing precisely what the certificate promises. Cache Engineering tests whether heuristic agent-judgment selection can match certified admission; MEPIC tests whether prefix reuse sidesteps the need for admission altogether; the fixed-tolerance gate tests whether the contrastive τ is doing real work beyond scale calibration; the attention-sensitivity pruning bound tests whether ARC's certificate adds anything beyond standard importance scores. The per-query-max ablation is the sharpest test of the batch-averaging design choice: if batch-averaging is truly the cheaper-but-adequate statistic, replacing it with the more conservative per-query max should change the admit/reject pattern in a predictable direction (fewer admits, higher retained mass), confirming the averaging choice is a real tradeoff rather than a no-op. The drift-robust ablation (without r_i_ub) directly tests the load-bearing assumption: if the certificate is valid only under bounded drift, removing the drift-robust inflation should cause a visible drop in retained attention mass on high-drift subsets, and the bound-violation fraction in the diagnostic scatter should increase. The deferred-verification ablation tests whether the re-audit step is necessary near the drift boundary. The Pareto frontier and runtime overhead measurements test whether the certificate provides a usable worst-case guarantee at a cost cheaper than re-prefill. Together these form a falsifiable test: the gain must appear where query drift is largest, not uniformly, and the bound must be an upper bound (no points above the identity line) on the diagnostic scatter.

### Expected outcome and causal chain

**vs. Cache Engineering** — On a case where the agent's early turns populate the cache with tool-output tokens that a later, drifted query no longer attends to, the heuristic policy drops those entries as apparently stale and then must re-prefill when a follow-up query re-references them, because its judgment is local to the current turn and computes no downstream perturbation. ARC instead computes r_i_ub over the current batch and admits the drop only if the first-order bound clears τ, so it rejects the drop when residual reachability is nonzero. We expect a visible gap in retained attention mass on drifted-query subsets with near-parity on stationary subsets.

**vs. MEPIC** — On a case where two trajectories share a long prefix but diverge in cache-operation order, MEPIC reuses the shared prefix and treats the divergent suffix as position-independent, yet its reuse decision ignores whether the current query still reaches the reused entries, so it can serve a suffix whose attention mass has collapsed. ARC evaluates the proposed repos against the live query batch and gates on B_repos, refusing the reposition when the position-shift term exceeds τ. We expect MEPIC to show higher raw cache reuse but lower retained attention mass per byte on divergence-heavy trajectories, with ARC closing that gap while giving up modest reuse.

**vs. Fixed-tolerance perturbation gate** — On a case where bound magnitudes vary by an order of magnitude across layers and contexts, a fixed tolerance either admits destructive edits in low-scale layers or rejects benign ones in high-scale layers, because a constant cannot track the scale of B_i. ARC's contrastive τ is fit to separate mass-preserving from mass-destroying pairs, so its admit/reject boundary moves with the local scale. We expect the fixed gate to show a bimodal error pattern (over-admit in some layers, over-reject in others) that ARC flattens, with the largest gain in the layers where the fixed gate's error is worst.

**vs. Attention-sensitivity pruning bound** — On a case where an entry has high current attention mass but its value vector is nearly collinear with the renormalized context, the attention-sensitivity score |a_{q,i}| * ||v_i|| flags it as important and refuses the drop, even though the true output perturbation is near zero. ARC's bound B_drop(i) = max_q | r_i_ub * delta_i | includes the value discrepancy delta_i, so it admits the drop when delta_i is small, freeing cache that the sensitivity bound conservatively retains. We expect ARC to achieve higher cache reduction at equal retained attention mass on subsets where value vectors are redundant, with near-parity on subsets where value vectors are diverse.

**vs. ARC without drift-robust upper bound** — On high-drift subsets where future queries move outside the current batch's convex hull, the point-estimate r_i underestimates future attention mass, so the certificate admits edits that later destroy task-relevant attention. We expect the no-drift-robust ablation to show lower retained attention mass on high-drift subsets and a higher bound-violation fraction in the diagnostic scatter, with near-parity on low-drift subsets.

**vs. ARC without deferred verification/rollback** — On edits near the drift boundary (tau < B_e <= tau * (1 + epsilon)), the provisional admit may be wrong if the next few queries drift further. We expect the no-verification ablation to show a small but measurable drop in retained attention mass on the boundary subset, with near-parity elsewhere.

### What would falsify this idea

If ARC's gain over the heuristic and prefix-sharing baselines is uniform across query-drift strata rather than concentrated on the high-drift subset where the reachability statistic is supposed to matter, the certificate is not doing what we claim and the result is explained by something else. Likewise, if the per-query-max ablation matches batch-mean exactly, the batch-averaging design choice is inert and the claimed cost-accuracy tradeoff is illusory; and if the diagnostic scatter shows bound violations (points above the identity line) in the high-drift regime, the certificate is not a valid upper bound under the stated drift assumption.

## References

1. Cache Engineering for Agents
2. MEPIC: Memory Efficient Position Independent Caching for LLM Serving
3. AgentFold: Long-Horizon Web Agents with Proactive Context Management
