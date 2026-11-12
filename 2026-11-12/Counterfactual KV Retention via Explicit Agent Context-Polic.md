# Counterfactual KV Retention via Explicit Agent Context-Policy Models

## Motivation

Agent-serving systems such as Evolving Infra estimate KV-cache utility from runtime access recency and frequency signals, which are correlational summaries of past attention rather than a model of why the agent will re-read a block. Because an agent's context-management policy (which tool outputs it re-injects, which scratchpad it summarizes, which files it re-opens) is the actual mechanism generating future KV reuse, recency/frequency heuristics systematically mis-rank blocks whose value is mediated by a planned future step rather than by recent access; the structural failure is that no existing system represents the intervention 'evict this block' against the agent's policy, so utility is never counterfactual. This limitation is shared across the DeepSeek-V2 and DeepSeek-V3 branches of Evolving Infra and is not addressed by any access-statistics heuristic.

## Key Insight

An agent's context-management policy is an explicit, inspectable program over its own history (tool calls, file reads, summarization triggers), so the value of a KV block is identifiable as the difference in the policy's future re-reference probability under an intervention that removes the block, which access statistics cannot express because they condition on the block already being present.

## Method

```pseudocode
INPUT: KV blocks B = {b_1..b_n} for an active agent trajectory, agent context-policy program P (the agent's tool-use/context-selection rules, e.g. a ReAct-style loop with explicit read_file/summarize/re-inject actions), budget k, horizon H
OUTPUT: retained set R ⊂ B, |R| = k

0. ASSUMPTIONS (stated explicitly; see block D for failure modes)
   A1 (sufficiency of symbolic state): s_t = (g, F, S, Q, t) is a sufficient statistic for the read set of every future action a_t drawn from the declared action grammar.
   A2 (single-origin provenance): each block b_i has exactly one originating tool call / observation / reasoning span, so provenance is a partition of the context.
   A3 (horizon truncation): re-fetch cost beyond H is dominated by re-prefill and does not change the eviction ranking.
   A4 (calibrated reference model): the reference predicates references(a, b_i) are a calibrated approximation of the agent's stochastic policy; the unresolved-reference mass ε (fraction of actions whose read set falls outside the provenance-declared set) is measured on the target traces and is small enough that U_i is identified (see step 3b).

1. TRACE EXTRACTION
   Parse the running agent transcript into a symbolic state s_t = (task goal g, open files F, scratchpad summary S, pending tool calls Q, turn index t).
   For each block b_i record its provenance label: which tool call / observation / reasoning span produced it. Blocks with multiple origins (concatenations) are split into per-origin sub-blocks b_i^(j) at extraction time; the split is recorded so the over-crediting error can be measured (see experiment ablation).

2. POLICY MODEL
   Instantiate P as a deterministic transition function over symbolic state:
     s_{t+1} = P(s_t, a_t),  a_t drawn from the agent's declared action grammar.
   P exposes, for each action a, a predicate references(a, b_i) that is TRUE iff executing a would emit a context read whose provenance resolves to b_i (e.g., read_file(F) references all blocks whose provenance is a prior read of F; re-inject(S) references blocks summarized into S).
   For each action type τ (read_file, grep, semantic_search, summarize, re-inject, open_recent), fit a per-action-type read-set predictor from the agent's own logged rollouts: a lookup table mapping (τ, symbolic-state features) → empirical distribution over provenance labels read, with a Laplace-smoothed count over the calibration set of 512 logged actions per type. The predictor returns a reference probability p(a, b_i) ∈ [0,1] and an unresolved mass ε_τ = 1 - Σ_i p(a, b_i), the empirical rate at which actions of type τ read blocks outside their provenance-declared set.

3. COUNTERFACTUAL UTILITY (do-calculus, backdoor-free by construction)
   For each b_i, compute over horizon H the expected number of future references under the factual policy and under the intervention do(evict b_i):
     U_i = E[ sum_{t'=t+1}^{t+H} p(a_{t'}, b_i) | s_t, P ]
         - E[ sum_{t'=t+1}^{t+H} p(a_{t'}, b_i) | s_t, P, do(evict b_i) ]
   The intervention is implemented by masking b_i from the provenance resolver and forcing any action that would read it to instead re-fetch (counts as a miss), so the second term is the re-fetch cost the eviction induces.
   (3b) UNCERTAINTY-AWARE FALLBACK: for each b_i compute the unresolved-reference mass ε_i = Σ_{t'=t+1}^{t+H} ε_{τ(a_{t'})} · 1{b_i is in the candidate read set of a_{t'}}. If ε_i > ε_max (set to 0.10, i.e. >10% of the horizon's reference mass is unresolved for this block), b_i is treated as high-variance and retained by a fallback score (recency/attention) rather than ranked by U_i; U_i is logged with a flag `unresolved=true` for audit. The threshold ε_max is validated on the target traces before claiming the counterfactual is identified: if the fraction of blocks with ε_i > ε_max exceeds 20% on a trace subset, that subset is reported as outside the identified regime.

4. EVICTION
   Rank blocks by U_i descending; ties broken by smaller byte footprint (cheaper to keep). Blocks flagged `unresolved=true` in step 3b are ranked by the fallback score and placed after all resolved blocks with U_i > 0. Retain top-k. Emit R to the KV manager; log U_i, ε_i, and the inducing action for audit.
```

**Why this design.** We chose an explicit symbolic policy program P over a learned utility regressor because the agent's context-management rules are already exposed as tool-call grammar in the serving stack, and a program makes the intervention do(evict b_i) mechanically well-defined, accepting the cost that P must be kept in sync with the agent's actual loop. We chose provenance-resolved reference predicates over attention-weight or recency scores because provenance is the only signal that ties a block to a future action rather than to past attention, accepting the cost of instrumenting tool outputs with provenance labels. We chose a finite horizon H matched to the agent's planning depth over an infinite-horizon estimate because re-fetch cost beyond the next few steps is dominated by re-prefill and is not the eviction decision's job, accepting the cost that very long-range re-references are under-weighted. We chose to calibrate the reference predicates from 512 logged actions per action type and to attach an explicit unresolved-reference mass ε_i with a fallback, rather than trusting the hand-written predicates, because the agent's next action is sampled from a stochastic policy and the read set of grep/semantic-search/summarize is data-dependent; this accepts the cost of a calibration pass over logged rollouts in exchange for making the sufficiency assumption A1 testable and bounding the spurious-counterfactual error. A domain expert would not describe this as a variant of VinePPO or per-step Monte-Carlo value estimation: those estimate a scalar reward-to-go from sampled rollouts, whereas here the quantity is a difference in reference counts induced by a structural intervention on the policy's provenance resolver, which sampling cannot produce without the resolver.

**Why it measures what we claim.** The quantity U_i measures the causal necessity of block b_i for future agent steps because the assumption A1 that P's action grammar is a sufficient statistic for which blocks a future action reads makes the reference-count difference equal the eviction-induced re-fetch cost; this assumption fails when the agent issues an action whose read set depends on content not captured in the symbolic state (e.g., a free-form grep whose match set is data-dependent), in which case U_i reflects only the provenance-declared reads and under-counts content-discovered re-references — this is exactly the unresolved mass ε_i that step 3b measures and falls back on. The provenance label measures block-to-action attribution because the assumption A2 that each block has a single originating tool call makes provenance a partition of the context; this assumption fails when a block is a concatenation of multiple observations, in which case the label attributes the whole block to the first origin and over-credits it — this is why step 1 splits multi-origin blocks into per-origin sub-blocks and the experiment measures the over-crediting error directly. The horizon H measures planning-depth relevance because the assumption A3 that re-fetch cost beyond H is dominated by re-prefill makes truncation lossless for the eviction decision; this assumption fails when an agent re-opens a file many turns later, in which case U_i under-ranks a block that will be needed. The calibrated reference probability p(a, b_i) measures the agent's actual read set because assumption A4 that the per-action-type predictor fitted on 512 logged actions is calibrated makes the empirical reference rate equal the policy's true reference probability; this assumption fails when an action type is under-represented in the calibration set or when the symbolic-state features miss a read-set determinant, in which case p(a, b_i) reflects the calibration distribution rather than the current trajectory's policy, and ε_i flags the affected blocks for fallback.

## Contribution

(1) A counterfactual KV-utility estimator that replaces access-statistics eviction with an intervention on an explicit agent context-policy program, yielding per-block utilities that are differences in policy-induced reference counts rather than recency/frequency scores. (2) A provenance-resolved reference predicate that makes do(evict) mechanically executable in a serving stack, together with a finite-horizon truncation argument that bounds when the estimate is faithful to re-fetch cost. (3) An audit log of U_i and its inducing action that exposes, per eviction, which future agent step justified retention, enabling diagnosis of mis-ranked blocks.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | Agentic coding/math/computer-use traces (GLM-5.2, Kimi K2.6), scoped first to a single ReAct loop with a fixed tool grammar and a single model (Kimi K2.6) | Matches serving workload with real tool-call grammar; single-framework scope makes the symbolic state s_t well-defined and the calibration set of 512 logged actions per action type constructible |
| Trace subsets (pre-registered) | (i) long-horizon re-inject (≥8 intervening turns before re-reference), (ii) summary-mediated re-reference (block re-read only through a scratchpad summary), (iii) direct-read (re-reference within 2 turns) | Pre-registered before running so the concentration test in 'What would falsify' is not post-hoc; direct-read is the near-parity control |
| Primary metric | Re-fetch rate per retained block under fixed KV budget | Directly measures eviction-induced re-prefill cost |
| Secondary metric | Fraction of blocks with unresolved mass ε_i > ε_max=0.10, reported per trace subset | Validates assumption A4: if >20% of blocks on a subset are unresolved, that subset is reported as outside the identified regime |
| Baseline | Dynamo default KV eviction | Production reference for agentic serving |
| Baseline | Recency/LRU eviction | Tests whether provenance beats temporal locality |
| Baseline | Attention-score eviction | Tests whether future-reference beats past-attention |
| Baseline | Program-dependence prefetcher (software-cache analogue: prefetch blocks whose provenance is in the read set of the next declared action) | Bounds the novelty of the counterfactual ranking against a dependence-analysis prefetcher; on at least one trace we report whether it makes the same decision as our method |
| Ablation-of-ours | U_i with provenance masked to recency-only (keeps horizon and intervention machinery) | Isolates the provenance-resolver contribution |
| Ablation-of-ours | U_i without the step-3b unresolved-mass fallback (ε_max = ∞) | Isolates the calibration/fallback contribution and measures the spurious-counterfactual error |
| Ablation-of-ours | Per-origin block splitting disabled (multi-origin blocks kept whole) | Quantifies the over-crediting error from assumption A2 directly |
| Replay harness | Reuses cached prefill states so each eviction policy is evaluated on identical trajectories without re-running the LLM | Makes the comparison causal: only the eviction decision varies across policies |

### Why this setup validates the claim

The central claim is that eviction utility should be the causal re-fetch cost induced by removing a block from the agent's future action grammar, and that provenance-resolved reference counts estimate this better than recency or attention. The dataset supplies real agent loops where re-references are non-local in time, which is exactly the regime where the claim is testable; scoping to a single ReAct loop with a fixed tool grammar makes the symbolic state s_t and the reference predicates well-defined, and the replay harness with cached prefill states ensures every policy sees identical trajectories so only the eviction decision varies. Dynamo's default eviction tests the end-to-end serving claim; LRU tests whether temporal locality alone suffices, isolating the value of provenance; attention-score eviction tests whether past-attention is a sufficient proxy for future reference, isolating the value of the counterfactual; the program-dependence prefetcher bounds the novelty against a software-cache analogue. The provenance-masked ablation, which keeps the horizon and intervention machinery but collapses provenance to recency, isolates whether any gain comes from provenance resolution specifically rather than from horizon truncation or the eviction ranking itself. The no-fallback ablation and the per-origin-splitting ablation directly measure the two load-bearing assumptions A4 and A2: if the fallback or the splitting changes the ranking materially, the assumptions are doing work and must be reported. The secondary metric (fraction of blocks with ε_i > ε_max) validates A4 on the target traces before any counterfactual claim is made. Re-fetch rate per retained block is the right primary metric because U_i is defined as exactly the quantity that predicts re-fetch cost; if the method works, the measured re-fetch rate should track the predicted U_i ranking, and if it does not, the causal chain is broken regardless of any latency side effects.

### Expected outcome and causal chain

**vs. Dynamo default KV eviction** — On a case where an agent reads a file early, summarizes it into a scratchpad, and re-injects the summary many turns later, Dynamo's default policy evicts the early file blocks because they are cold by recency and attention, forcing a full re-prefill on re-inject. Our method retains them because P's re-inject predicate resolves provenance to those blocks and U_i counts the induced re-fetch, so we expect a clear re-fetch-rate gap concentrated on long-horizon re-inject workloads with parity on short single-pass tasks.

**vs. Recency/LRU eviction** — On a case where a computer-use agent opens a tool result, switches to a different window for several turns, then returns to the first, LRU has already evicted the first result because intervening turns dominate the recency window, producing a re-fetch on return. Our method keeps it because references(a, b_i) under the return action resolves to that block regardless of intervening turns, so we expect a gap that widens with the number of intervening turns and shrinks to zero when re-references are immediate (the direct-read subset is the near-parity control).

**vs. Attention-score eviction** — On a case where a block was heavily attended during a past reasoning span but the agent's policy program never references it again, attention-score eviction retains it and evicts a quiet block that the next tool call will read, causing a miss on the read. Our method ranks the quiet block higher because U_i is a forward-looking reference-count difference, not a backward attention mass, so we expect our gain to concentrate on blocks with high past attention but zero future references, and near-parity on blocks where past attention and future reference coincide.

**vs. Program-dependence prefetcher** — On a trace where the next declared action's read set is fully resolved by provenance (e.g., a direct read_file(F) whose provenance is a single prior read of F), the dependence prefetcher and our method make the same decision; we expect parity there and a gap only on traces where the counterfactual (do(evict)) changes the ranking relative to the factual read set, bounding the novelty to the intervention rather than to dependence analysis alone.

**vs. Provenance-masked ablation** — On a case where a block is re-referenced only through a scratchpad summary rather than a direct file read, the recency-only ablation cannot resolve the summary back to the block and under-ranks it, producing a re-fetch on re-inject. Our method resolves the summary's provenance to the block and retains it, so we expect the ablation to sit between LRU and our full method, with the gap largest on summarize-then-re-inject traces and near-zero on direct-read traces.

**vs. No-fallback ablation (ε_max = ∞)** — On a trace containing a data-dependent grep whose match set is not resolvable from the symbolic state, the no-fallback ablation ranks the grep-referenced blocks by a spurious U_i and can evict a block the grep will actually read, producing a re-fetch; our method flags those blocks as unresolved (ε_i > 0.10) and retains them by fallback. We expect the no-fallback ablation to underperform on the subset where the unresolved-mass metric exceeds 20%, and to match our method on subsets where ε_i is small, which is exactly the validation of assumption A4.

**vs. Per-origin splitting ablation** — On a trace where a block concatenates two observations and only the second is re-referenced, the unsplit ablation attributes the whole block to the first origin and over-credits it, retaining a block whose second half is never read and evicting a block that is; our method splits the block into per-origin sub-blocks and ranks each by its own U_i. We expect the splitting ablation to show a measurable over-crediting error (higher re-fetch rate) on traces with concatenated blocks, quantifying the cost of assumption A2.

### What would falsify this idea

If our gain over LRU and attention-score eviction is uniform across all trace subsets rather than concentrated on long-horizon re-inject and summary-mediated re-reference cases, then provenance-resolved counterfactual utility is not the causal driver and the claim is wrong. Likewise, if the provenance-masked ablation matches our full method on those subsets, the provenance resolver contributes nothing and the mechanism is not what we claim; and if the fraction of blocks with ε_i > ε_max exceeds 20% on the target subsets, the counterfactual is not identified there and the claim is restricted to the resolved regime.

## References

1. Evolving Infra: Agent-Adaptive LLM Serving at Scale
