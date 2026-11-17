# Full-Budget Parameter Accounting for Looped Transformers at Matched Executed Depth

## Motivation

Looped-transformer efficiency claims are currently made inside a fixed accounting convention: the Loopback Transformer reports lower validation loss than standard Transformers using 40–75% of their parameters, but those parameter counts explicitly exclude token embeddings and the output head, and the comparison stops at a 3B reference scale. This matters structurally because embedding and output-head matrices scale with vocabulary and hidden width rather than with depth, so excluding them systematically favors architectures that trade depth for width — precisely the trade a looped, shared-depth design makes. The field therefore lacks an efficiency comparison whose denominator is a function of executed depth and total parameter budget, which is the recurring gap across looped and depth-reuse work rather than a local omission.

## Key Insight

When the efficiency denominator is recomputed as a function of executed depth and total parameters including embeddings and the output head, geometric-decay update weighting in a shared stack remains the dominant lever only if its advantage survives the depth-cost term that embedding/output-head inclusion makes explicit — a term that is fixed in width but amortized differently at every executed depth.

## Method

**What it is.** We introduce FBPA (Full-Budget Parameter Accounting), a measurement-and-training protocol that takes a model family (looped or standard), a reference parameter budget, and a set of executed depths, and returns an effective-parameter curve P_eff(D) together with a matched-budget comparison at each depth. Its output is the quantity that Loopback-style claims must be evaluated against, not a new architecture.

**How it works.**

```pseudocode
# Phase 0: analytical grounding on published configs (no training)
input: published Loopback configs at 191M, 1.4B, 3B (vocab V, d_model, n_layers, tied flag)
for each published config:
    P_emb   = V * d_model
    P_head  = 0 if tied else V * d_model
    P_block = per_block_params(n_layers=1, d_model, n_heads)
    P_eff(D) = P_emb + P_head + P_block * D
    tax_frac(D) = (P_emb + P_head) / P_eff(D)
    budget_ratio(D) = P_eff(D) / P_ref
report: tax_frac(D) and budget_ratio(D) for each published config
# (this yields the predicted compression of the reported 40-75% fraction before any training)

# Phase 1: full-budget accounting
input: model config (n_layers, d_model, n_heads, vocab V, tied_embeddings flag)
       executed depth D, reference budget P_ref
P_emb   = V * d_model                      # token embedding matrix
P_head  = 0 if tied else V * d_model       # output head
P_block = per_block_params(n_layers=1, d_model, n_heads)
P_eff(D) = P_emb + P_head + P_block * D    # depth-linear body term
budget_ratio(D) = P_eff(D) / P_ref

# Phase 2: matched-budget construction with joint (d_model, rho) search
for each target D in {matched depths}:
    solve d_model(D) s.t. P_eff(D) = P_ref     # width absorbs the depth cost
    instantiate standard baseline at (D, d_model(D))
    instantiate Loopback at (D, d_model(D)) with geometric decay rho
    # joint budget-constrained search for BOTH arms:
    for standard baseline: tune learning-rate schedule over {1e-4, 3e-4, 1e-3} and width schedule (constant vs. linear warmup over first 10% steps) at fixed D and d_model(D)
    for Loopback: sweep rho in {0.5, 0.7, 0.85, 0.95} at fixed D and d_model(D)
    record per-arm optimum loss(D) and accuracy(D) over the freed variables

# Phase 3: matched-depth training
for each D:
    train both models on ClimbMix, identical tokens, identical executed depth D
    record validation loss and 13-task mean accuracy at the per-arm optimum from Phase 2

# Phase 4: decay-sensitivity sweep (full surface)
for rho in {0.5, 0.7, 0.85, 0.95}:
    retrain Loopback at matched budget; record loss(D, rho)
report: loss(D) and accuracy(D) as functions of budget_ratio(D), not of raw parameter count;
        also report the full loss(D, d_model, rho) surface and claim only orderings that hold across it
```

Hyperparameters held fixed across arms: token count, executed depth D, optimizer (AdamW, beta1=0.9, beta2=0.95, weight decay=0.1), and P_ref. The free variables per arm are width (solved in closed form from the accounting equation) and, for Loopback, rho (swept); for the standard baseline, learning-rate schedule and width schedule are tuned at each matched budget so that both arms are allowed to re-tune under the budget constraint.

**Why this design.** We chose a closed-form width solve over a grid search of widths because the accounting equation P_eff(D) = P_emb + P_head + P_block·D is linear in d_model and therefore invertible, accepting the cost that the solve assumes a fixed vocabulary and untied-or-tied embedding choice rather than jointly optimizing them. We chose to sweep geometric decay rho rather than hold it at the reported default because the Loopback contribution attributes its gain to decay-controlled update weighting, and an accounting protocol that fixes rho cannot tell whether the gain is a property of the architecture or of one hyperparameter setting; the cost is a 4× training multiplier at each depth. We chose to report loss as a function of budget_ratio(D) rather than of raw parameter count because raw counts are exactly the convention that excludes embeddings and the head, and a domain expert would not describe FBPA as a variant of the Loopback architecture paper: FBPA changes the denominator and the scale range of the comparison, not the residual-path mechanism, and it would be equally applicable to any depth-reuse design. We chose ClimbMix and the thirteen downstream tasks over a new benchmark because reuse of the original evaluation makes the accounting change the only difference between arms. We chose to tune the standard baseline's learning-rate and width schedules at each matched budget (rather than holding them fixed) because the standard transformer scaling literature establishes that optimal hyperparameters shift with width/depth allocation, so a comparison in which only one arm re-tunes would confound the accounting change with a hyperparameter-transfer effect; the cost is an additional 3× learning-rate sweep for the standard arm at each depth.

**Why it measures what we claim.** The computational quantity budget_ratio(D) measures full-budget parameter efficiency at matched executed depth because the assumption that embedding and output-head matrices are depth-invariant while the body is depth-linear makes P_eff(D) an exact count rather than an estimate; this assumption fails when the vocabulary is resized or embeddings are tied differently across arms, in which case budget_ratio reflects a vocabulary choice rather than a depth-efficiency trade. The quantity loss(D, rho) measures the architectural contribution of geometric-decay update weighting because the assumption that width is the only variable freed by the budget solve makes any loss difference attributable to the update-weighting mechanism; this assumption fails when width itself interacts with decay strength (a wider model may prefer a different rho), in which case the measured difference reflects a width-decay interaction rather than decay alone. The quantity accuracy(D) over thirteen tasks measures downstream equivalence because the assumption that the task suite is sensitive to the depth-width trade makes matched-budget accuracy a valid equivalence test; this assumption fails when tasks are saturated at all tested depths, in which case accuracy reflects task insensitivity rather than equivalence. The quantity loss(D) at the per-arm optimum measures the residual architectural advantage of Loopback over the matched-budget standard Transformer because the assumption that width and depth are interchangeable capacity at fixed P_eff (i.e., that the width-solved standard baseline is capacity-equivalent to Loopback at equal P_eff) licenses attributing any residual loss gap to the update-weighting mechanism; this assumption fails when width and depth are not capacity-equivalent, in which case the loss gap reflects a depth-width capacity mismatch rather than the update-weighting mechanism, and we therefore pair the parameter-matched comparison with a FLOPs-per-token-matched control (see experiment).

## Contribution

(1) FBPA, a full-budget parameter-accounting protocol that recomputes effective parameter count as a closed-form function of executed depth and total parameters including token embeddings and the output head, with a width-solve that constructs matched-budget standard and looped baselines at each depth. (2) A matched-budget, matched-executed-depth comparison of geometric-decay update weighting against standard residual Transformers at reference scales beyond 3B, reporting loss and downstream accuracy as functions of budget ratio rather than raw parameter count. (3) A decay-sensitivity sweep that separates the architectural contribution of update weighting from the contribution of a single decay hyperparameter under full-budget accounting.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | ClimbMix | Reuses original evaluation; accounting is only change |
| Primary metric | Validation loss vs. budget_ratio(D) | Directly tests full-budget efficiency claim |
| Secondary metric | FLOPs-per-token-matched loss | Controls for depth-width capacity equivalence |
| Baseline | Raw-count Loopback (reported convention) | Tests whether excluding embeddings inflates gain |
| Baseline | Matched-budget standard Transformer (tuned LR/width schedule) | Tests whether depth-reuse survives width absorption with both arms re-tuned |
| Baseline | Fixed-rho Loopback | Tests whether gain is decay-specific |
| Ablation-of-ours | FBPA without decay sweep | Isolates accounting from hyperparameter contribution |
| Ablation-of-ours | FLOPs-per-token-matched control | Tests whether residual gap is depth-width capacity mismatch |

Primary matched-budget and rho-sweep experiments are run at reference scales 50M–200M parameters (where the 4-rho × multiple-depth grid is tractable: 4 rho × 3 depths × 2 arms = 24 training runs, each ≤ 8 GPU-hours on A100-40GB, total ≤ 192 GPU-hours). The accounting-only prediction (Phase 0, no retraining) is validated at 1.4B and 3B by recomputing P_eff from published configs, reserving one confirmatory 1.4B training run for the single most informative depth (the depth where Phase 0 predicts the largest compression of the reported advantage).

### Why this setup validates the claim

The central claim is that Loopback-style depth-reuse gains are partly an artifact of a parameter-counting convention that excludes embeddings and the output head, and that under a full-budget denominator the advantage shrinks or reorganizes. The raw-count Loopback baseline reproduces the original convention, so any gap between it and FBPA isolates the accounting change alone; the matched-budget standard Transformer tests whether depth-reuse still wins once width absorbs the depth cost and both arms are allowed to re-tune their hyperparameters at the matched budget; the fixed-rho baseline tests whether the residual gain is architectural or a single hyperparameter setting. Validation loss plotted against budget_ratio(D) is the right metric because it makes the denominator explicit and continuous, so a genuine efficiency effect must appear as a leftward shift of the loss curve at equal full-budget cost, not merely at equal raw body parameters. The FLOPs-per-token-matched control addresses the width-depth capacity-equivalence assumption: if the residual loss gap persists when FLOPs-per-token are matched, the gap is attributable to the update-weighting mechanism; if it vanishes, the gap reflects a depth-width capacity mismatch. The ablation removes the decay sweep, so if the sweep is what produces the difference, the ablation will match the fixed-rho baseline rather than the full protocol.

### Expected outcome and causal chain

**vs. Raw-count Loopback (reported convention)** — On a case where the vocabulary is large relative to the body, the baseline reports Loopback matching the standard Transformer at 40-75% of parameters because its denominator omits the embedding and head matrices, which are depth-invariant and therefore a fixed tax. FBPA instead charges those matrices at every depth, so budget_ratio(D) rises toward and past parity at small D. Phase 0 computes tax_frac(D) for the published 191M, 1.4B, and 3B configs and yields the predicted compression before any training. We expect the reported parameter-fraction advantage to compress substantially, with the largest compression at shallow depths where the fixed embedding cost dominates.

**vs. Matched-budget standard Transformer (tuned LR/width schedule)** — On a case where the budget solve forces the standard baseline to a wider d_model at the same executed depth, the baseline gains capacity that the raw-count comparison never granted it, and the per-arm LR/width-schedule tuning ensures the baseline is not handicapped by hyperparameters transferred from a different width regime. FBPA instantiates exactly this wider, re-tuned baseline, so the depth-reuse advantage must now be earned against a fairer opponent. We expect Loopback to retain a modest loss edge at moderate depths but to approach parity at shallow depths where width absorption is cheapest.

**vs. Fixed-rho Loopback** — On a case where the reported default rho is near-optimal only for one width, the baseline's gain partly reflects a lucky hyperparameter match, and a wider matched-budget model may prefer a different decay. FBPA sweeps rho and reports loss(D, rho), so any gain that survives across the sweep is attributable to the update-weighting mechanism itself. We expect a clear rho-by-width interaction: the best rho shifts as width grows, and the fixed-rho baseline underperforms the swept optimum most at the extremes of the depth range.

**vs. FLOPs-per-token-matched control** — On a case where the width-solved standard baseline and Loopback have different FLOPs-per-token at equal P_eff, the parameter-matched comparison may attribute a depth-width capacity mismatch to the update-weighting mechanism. The FLOPs-per-token-matched control instantiates both arms at equal FLOPs-per-token (by adjusting d_model or n_heads within the budget), so any residual loss gap is attributable to the update-weighting mechanism rather than to capacity mismatch. We expect the residual gap to shrink but not vanish under FLOPs matching, confirming that part of the parameter-matched advantage is a capacity effect and part is architectural.

### What would falsify this idea

If Loopback's advantage over the matched-budget standard Transformer is uniform across all depths and rho settings rather than concentrated at shallow depths and sensitive to the width-decay interaction, then the accounting change is not the operative variable and the central claim is wrong. Equivalently, if the FBPA curve coincides with the raw-count curve after rescaling, or if the residual gap vanishes entirely under FLOPs-per-token matching, the convention critique fails or the gap is a capacity mismatch rather than an architectural effect.

## References

1. Loopback Transformer: Towards a Unified View of Residual Looped Transformers
