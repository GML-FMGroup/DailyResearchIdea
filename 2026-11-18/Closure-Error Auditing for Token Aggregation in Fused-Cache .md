# Closure-Error Auditing for Token Aggregation in Fused-Cache and Linear-Attention Transformers

## Motivation

Across the efficient-transformer line, aggregation is treated as free: Multi-Branch Transformer's Joint-KV builds a shared KV cache from branch representations at each depth and fuses branches without ever asking whether the fused statistic is realizable by any per-token attention operator, and TSSA replaces pairwise-similarity attention with token-statistic attention whose linear complexity likewise assumes that summarizing tokens into statistics preserves the attention computation. Both inherit the same unexamined premise — that a token-set aggregation has a closure/equivalence rule mapping it back to an equivalent per-token operator — and because no single branch states this rule, neither can detect when fusion silently changes the function being computed. The structural gap is not a missing module but a missing invariant: aggregation is assumed, never checked.

## Key Insight

A token-set aggregation is faithful exactly when its induced per-token attention output is recoverable up to a computable residual, so an explicit closure operator with a measurable closure error turns an implicit aggregation assumption into a checkable invariant that localizes fusion-induced distortion.

## Method

**A) What it is.** We introduce **CLOC** (Closure-Error Operator for Token Aggregation), an audit-and-correction layer that sits on top of any token-aggregating transformer component (Joint-KV fused caches or a TSSA-style linear attention module). Its inputs are the per-token key/value/query tensors before aggregation and the aggregated statistics after aggregation; its outputs are a scalar **closure error** ε per aggregation site and a per-token correction that restores the closure identity within a tolerance τ.

**Load-bearing assumption (stated explicitly).** CLOC measures deviation from each component's *declared output contract*, not from a universal softmax-attention reference. For a component that is explicitly posed as a softmax approximation (e.g., a linear-fusion KV cache whose stated contract is to reproduce softmax(QK^T)V up to a cache-sharing term), the reference O_ref is the full-resolution softmax output. For a component that is *not* posed as a softmax approximation (e.g., TSSA, whose declared contract is the statistic-attention output its block is defined to compute), the reference O_ref is that component's own declared output computed at full token resolution without aggregation. In both cases ε is nonzero only when the aggregation changes the function the component declares it computes; if the component intentionally changes semantics, ε measures that intentional divergence and is reported as such, not as repairable error. This assumption is the load-bearing one: if a component's declared contract is not the semantics CLOC uses as reference, ε is not a faithfulness measure for that component.

**B) How it works.**

```pseudocode
# Inputs: per-token Q, K, V in R^{n x d}; aggregation map A (Joint-KV fusion or TSSA statistic)
# Hyperparameters: tolerance tau (default 1e-2 relative), probe count m (default 8), correction gain gamma (default 1.0)
# Reference selector: ref_mode in {softmax, declared}
#   softmax  -> O_ref = softmax(Q K^T / sqrt(d)) V            (for softmax-approximation aggregations)
#   declared -> O_ref = F_declared(Q, K, V) at full resolution (for TSSA-style statistic attention)

1. Compute reference output at full token resolution according to ref_mode:
     O_ref = softmax(Q K^T / sqrt(d)) V            if ref_mode == softmax   # n x d
     O_ref = F_declared(Q, K, V)                   if ref_mode == declared  # n x d, component's own contract

2. Apply the aggregation map A to get aggregated statistics:
     (K_agg, V_agg) = A(K, V)                       # e.g., Joint-KV shared cache or TSSA token statistics

3. Define the closure operator C_A: given aggregated stats, reconstruct the
   per-token operator that A *claims* to be equivalent to under ref_mode:
     O_agg = softmax(Q K_agg^T / sqrt(d)) V_agg    if ref_mode == softmax
     O_agg = F_declared(Q, K_agg, V_agg)           if ref_mode == declared

4. Closure error at this site:
     eps = ||O_ref - O_agg||_F / (||O_ref||_F + 1e-8)

5. If eps > tau, estimate the closure residual via m random probe vectors
   {p_j} drawn from the current batch's token distribution:
     r = (1/m) * sum_j ( O_ref(p_j) - O_agg(p_j) )   # d-dimensional residual direction

6. Apply a gain-limited correction to the aggregated values:
     V_agg' = V_agg + gamma * r                     # closes the gap on the probe subspace
     O_closed = softmax(Q K_agg^T / sqrt(d)) V_agg'  if ref_mode == softmax
     O_closed = F_declared(Q, K_agg, V_agg')         if ref_mode == declared

7. Return eps, O_closed, and the per-site residual norm ||r||.
```

A **site** is any location where a token set is collapsed: each Joint-KV depth in MBT, and each TSSA statistic block. CLOC does not replace either component; it wraps their aggregation with the closure check above. The audit is run in two modes: *diagnostic* (report ε per site, no correction, used in analysis) and *enforced* (apply step 6, used in the main experiments).

**C) Why this design.** We chose an explicit operator-level closure identity (step 3) over a learned auxiliary head that predicts whether aggregation is safe, because the closure identity is checkable without labels and a learned head would inherit the very coupling it is meant to detect — the trade-off is that the identity is exact only for attention maps whose softmax normalizer is preserved under aggregation, so we accept a residual that must be reported rather than driven to zero. We chose probe-based residual estimation (step 5) over full per-token recomputation, because recomputing O_ref for every site defeats the efficiency purpose of aggregation; the cost is that the residual is estimated on a random subspace and can miss directions orthogonal to all probes, which is why we report ||r|| alongside ε rather than trusting the correction blindly. We chose a gain-limited additive correction (step 6, γ=1.0, no learned parameters) over fine-tuning the aggregated cache, because a learned correction would confound the audit with training and make the closure error uninterpretable; the trade-off is that a fixed additive correction cannot repair nonlinear aggregation maps, so on such maps CLOC reports high ε and refuses to correct rather than silently masking the failure. A domain expert would not describe this as a variant of Joint-KV or TSSA: those papers propose aggregation mechanisms, whereas CLOC proposes an invariant those mechanisms must satisfy and a measurement of when they do not — the mechanism is the audit, not the aggregation.

**D) Why it measures what we claim.** The closure error ε = ||O_ref − O_agg||_F / ||O_ref||_F measures **aggregation faithfulness** because it is equivalent to the relative operator norm of the difference between the component's declared output at full resolution and the output induced by aggregated statistics, under the assumption that the declared output is the semantics the aggregation claims to preserve (ref_mode selects that semantics); this assumption fails when the aggregation is *intended* to change semantics (e.g., a deliberate long-context eviction), in which case ε reflects intentional divergence rather than error. The probe residual r measures the **correctable component of the closure gap** because it is the sample mean of the output difference over m probes drawn from the batch token distribution, under the assumption that the batch distribution covers the directions in which aggregation distorts the operator; this assumption fails when a site's distortion is concentrated on rare tokens absent from the probe sample, in which case r underestimates the true residual and the reported ε (which is exact) remains the trustworthy signal. The per-site ε profile measures **where in the architecture closure breaks** because each site is an independent aggregation event and ε is computed against that site's own pre-aggregation reference, under the assumption that sites do not interact through downstream nonlinearities; this assumption fails in deep stacks where an early closure error is amplified by later layers, in which case the per-site ε is a lower bound on the end-to-end distortion rather than a decomposition of it.

## Contribution

['A closure operator and measurable closure-error metric (CLOC) that formalizes token-set aggregation faithfulness for fused-cache and linear-attention transformers, with a probe-based residual estimator and a gain-limited correction.', 'A design principle — aggregation mechanisms should be audited against a per-token closure identity rather than assumed faithful — instantiated as a drop-in wrapper for both Joint-KV branch fusion and TSSA-style token-statistic attention.', 'A per-site closure-error profile analysis tool that localizes where fusion or statistic aggregation distorts the attention operator in a transformer stack.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Long-context language modeling suite (PG-19 / long-document QA) | Stresses token aggregation where closure breaks |
| Primary metric | Closure error ε (ref_mode per component) plus end-to-end task loss gap | Directly measures faithfulness and downstream cost |
| Baseline | Sequential Transformer (matched compute) | Tests whether aggregation is even needed |
| Baseline | MBT + Joint-KV (no CLOC, ref_mode=softmax) | Tests whether CLOC repairs aggregation distortion |
| Baseline | TSSA linear attention (no CLOC, ref_mode=declared) | Tests CLOC on a second aggregation family |
| Ablation-of-ours | CLOC diagnostic-only (ε reported, no correction) | Isolates correction from mere measurement |
| Ablation-of-ours | Probe-count sweep m ∈ {1, 2, 4, 8, 16, 32} | Tests whether probe residual spans distortion subspace |
| Ablation-of-ours | Probe-distribution sweep: uniform vs. frequency-matched vs. adversarial rare-token | Tests assumption A directly |
| Ablation-of-ours | ref_mode swap: softmax reference applied to TSSA, declared reference applied to Joint-KV | Tests whether reference choice changes ε interpretation |

### Why this setup validates the claim

The central claim is that token aggregation silently violates a closure identity and that auditing plus a gain-limited correction recovers faithfulness without giving up efficiency. The sequential Transformer baseline establishes the reference semantics: if CLOC-wrapped aggregation does not approach its quality on long-context tasks, the correction is not doing its job. The MBT+Joint-KV and TSSA baselines are the actual targets — each tests a different aggregation family (fused KV caches versus statistic-based linear attention), so a gain that appears only on one family would reveal the closure identity is not universal. The diagnostic-only ablation is the key falsification lever: it separates "we measured a gap" from "we closed it." If correction adds nothing beyond diagnosis, the mechanism claim collapses to a monitoring tool. The probe-count and probe-distribution sweeps test the load-bearing assumption that the m batch-distribution probes span the distortion subspace: if ε and ||r|| are stable across m and probe distributions, the residual estimate is robust; if they vary, the assumption fails and the reported ε (which is exact) remains the trustworthy signal. The ref_mode swap tests whether the reference choice changes the interpretation of ε: if ε is high under softmax reference but low under declared reference for TSSA, that confirms TSSA's divergence is intentional and the declared reference is the correct one. Closure error ε is the right primary metric because it is computed against each site's own pre-aggregation reference under the component's declared contract, so it directly indexes the quantity the method claims to reduce, and pairing it with end-to-end loss checks that reducing ε actually matters rather than being a cosmetic norm.

### Expected outcome and causal chain

**vs. Sequential Transformer (matched compute)** — On a long-context instance where the aggregation collapses many tokens into shared statistics, the sequential baseline attends over all tokens and pays quadratic cost but preserves exact per-token semantics. Our CLOC-wrapped model starts from the same aggregated statistics but applies the probe residual, so the output operator is pulled back toward the full-resolution reference. We expect CLOC-wrapped aggregation to land close to sequential quality on long-context subsets while retaining the aggregation speedup, with the residual gap concentrated at sites where ε remains above τ after correction.

**vs. MBT + Joint-KV (no CLOC)** — On a depth where Joint-KV fuses branch representations into a shared cache, the uncorrected model produces an output whose softmax normalizer no longer matches the per-token reference, so tokens that were rare in the fusion get systematically mis-weighted. CLOC detects this as elevated ε at that depth and adds the probe-estimated residual to the fused values, restoring the operator on the probe subspace. We expect a visible ε drop at fused depths and a matching loss improvement on inputs with skewed token frequency, with near-parity on balanced inputs where the fusion was already faithful.

**vs. TSSA linear attention (no CLOC)** — On a sequence where token statistics are dominated by a few high-magnitude tokens, TSSA's statistic-based attention departs from pairwise semantics and the closure gap grows with token-count skew. CLOC measures ε against TSSA's declared statistic-attention output at full resolution (ref_mode=declared) and corrects the aggregated values along the residual direction. We expect the correction to help most on high-skew sequences and to report high ε with little correction on sites where the aggregation is intentionally lossy, giving a clean separation between repairable and intentional divergence. Under ref_mode=softmax, we expect TSSA to show uniformly high ε regardless of skew, confirming that the softmax reference measures intentional divergence rather than repairable error.

### What would falsify this idea

If the CLOC gain over the diagnostic-only ablation is uniform across sites and subsets rather than concentrated where predicted high-ε sites occur, then the correction is not closing the closure gap and the central mechanism claim is wrong. Equally fatal: if ε drops after correction while end-to-end loss does not improve, the closure identity is not the quantity that governs aggregation quality; and if ε and ||r|| vary wildly across probe counts and distributions without converging, the probe residual does not span the distortion subspace and the correction is unreliable.

## References

1. Multi-Branch Transformer
