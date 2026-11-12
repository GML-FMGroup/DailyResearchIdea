# Type-Certified Weight Generation for EEG Classifiers via Dependent Architectural Invariants

## Motivation

Weight-space programming (Mixed Signals, the only prior work that has a language model emit EEG/ECG classifier weights with no gradient step) produces tensors that are empirically plausible but structurally unverified: nothing in that pipeline guarantees a generated weight tensor is even dimensionally composable with its architecture, let alone that it preserves residual-stream compatibility or channel/temporal filter constraints, so failures surface only after forward-pass evaluation and cannot be attributed to a specific violated invariant. The root cause is that legality in that pipeline is a post-hoc property of the assembled network rather than a generation-time property of the parameter manifold, and no ancestor supplies a structural verification layer — Tracr enforces legality by construction but only for a fixed computation graph, and causal tracing enforces minimality but not legality. This leaves the shared residual gap of joint legality-and-minimality certification unresolved at the meta level across biosignal architecture families.

## Key Insight

A dependent type system whose indices are the architecture's dimensional and compatibility invariants turns weight generation into type inhabitation, so every emitted tensor is a member of the certified parameter manifold by construction and type errors become a search signal that prunes the generator's candidate space rather than a post-hoc test.

## Method

**What it is.** We present TYPED-WEIGHTS, a compiler that takes a language-model prompt (architecture spec, task, subject subset) and emits a weight tensor only if it inhabits a dependent-type-indexed parameter manifold; every rejection returns a structured type error that is fed back into the generator for backtracking. The load-bearing assumption is explicit: an LLM prompted with a serialized dependent-type context and structured type-error witnesses will emit weight tensors that satisfy the encoded value-level indices at a rate high enough that type inhabitation is a tractable generation criterion — i.e., the LLM's token-level generation of numeric tensors is sensitive to symbolic shape and value constraints rather than being shape-blind noise. To make this assumption testable rather than assumed, every index in Gamma is a predicate over numeric content, not only declared shapes, and the primary metric is reported against a shape-only null baseline (random correctly-shaped tensors) so that any gain attributable to the LLM beyond shape conformance is visible.

**How it works.**

```pseudocode
Input: arch spec A (layer list, channels, temporal kernel sizes, heads, residual widths)
       task T, subject subset S, LLM L, max_retries R=8
Output: certified weight tensor W* or structured failure

1. INVARIANT EXTRACTION
   Gamma <- extract_indices(A)
   # Gamma is a dependent context of index terms, each a predicate over numeric content:
   #   residual_width[l] = C_l  for every residual block l
   #   head_dim[l,h] = d_model_l / H_l  (integer divisibility constraint)
   #   kernel_span[l] <= T_l (temporal receptive field bound)
   #   channel_count[l] = C_{l-1} for conv stack
   #   each weight matrix W_l : R^{C_l x C_{l-1}} typed as Matrix(C_l, C_{l-1})
   # Value-level indices (checked on the numbers, not the annotations):
   #   spectral_norm(W_l) <= s_max[l]  for every residual projection (Lipschitz bound)
   #   support(W_l^{kernel}) within T_l: the temporal kernel has nonzero entries only
   #     in the first T_l positions of its support (kernel-span support constraint)
   #   block_structure(W_l^{head}) matches H_l blocks of size d_model_l/H_l
   #     (head-dimension factorization reflected in the tensor's block structure)
   #   norm_coupling(W_l^{norm}) consistent with subject subset S: per-channel scale
   #     vector has L2 norm in [0.5, 2.0] and is shared across residual blocks
   #     (data-dependent normalization coupling tied to S)

2. GENERATION WITH TYPE FEEDBACK
   for r in 1..R:
     candidate <- L(prompt(A, T, S, Gamma, prior_errors))
     # candidate is a serialized tensor + declared shape annotations
     errs <- typecheck(candidate, Gamma)
     if errs == []:
        W* <- candidate; break
     prior_errors <- prior_errors + [errs]   # type-error feedback

3. TYPE CHECKER (dependent, value-level)
   for each weight tensor w in candidate:
     require shape(w) == declared_type(w)     # dimensional consistency
     require residual_projection(w) preserves C_l  # residual-stream compatibility
     require kernel_span(w) <= T_l             # temporal filter constraint
     require divisibility(d_model_l, H_l)      # attention-head consistency
     require spectral_norm(w) <= s_max[l]      # Lipschitz bound on residual projections
     require support(w^{kernel}) within T_l     # nonzero entries only in first T_l positions
     require block_structure(w^{head}) matches H_l blocks of size d_model_l/H_l
     require norm_coupling(w^{norm}) in [0.5, 2.0] and shared across blocks
   return list of violated indices with witness values

4. CERTIFICATION RECORD
   emit (W*, Gamma, derivation) where derivation is the type-inhabitation trace
```

**Why this design.** We chose a dependent type system over a fixed-graph legality checker (the Tracr approach) because Tracr's legality is bound to one computation graph and cannot certify weights for architectures the graph does not encode, accepting the cost that we must hand-write index terms for each new biosignal architecture family. We chose backtracking with structured type-error feedback over rejection sampling because rejection sampling discards the location of the violation and would require exponentially many LLM calls for deep residual stacks, accepting the cost that the generator must be re-prompted with a serialized error witness (extra tokens per retry, capped at R=8). We chose to type the parameter manifold rather than the forward-pass output because output-level checks conflate numerical instability with structural illegality, accepting the cost that some numerically valid but structurally unusual tensors are rejected. A domain expert would not describe TYPED-WEIGHTS as a variant of Mixed Signals weight-space programming: Mixed Signals emits weights and only then evaluates them, whereas TYPED-WEIGHTS never emits an uncertified tensor and uses type inhabitation as the generation criterion, not as a post-hoc filter.

**Why it measures what we claim.** The fraction of candidates that pass `typecheck` on the first attempt measures *generation-time legality* because the dependent indices in Gamma are exactly the architectural invariants (residual width, head divisibility, kernel span, channel count, spectral norm, kernel support, block structure, norm coupling) whose violation makes a tensor non-composable with the architecture; this assumption fails when an architecture has an invariant not expressible as a dependent index (e.g., a data-dependent normalization coupling not captured by the norm-coupling index), in which case the pass rate reflects only the encoded subset of legality. The number of backtracking steps R_used measures *type-error informativeness* because each retry is conditioned on the previous violation witness, so a monotone decrease in violations per retry indicates the feedback is causally used by the generator; this assumption fails when the LLM ignores the witness and resamples independently, in which case R_used reflects resampling luck rather than feedback utility. The certification record's derivation depth measures *minimality of the legality certificate* because each index in the derivation corresponds to exactly one invariant that had to be discharged; this assumption fails when indices are redundant (two indices encode the same constraint), in which case derivation depth over-counts distinct invariants. To separate legality from functionality, we additionally report downstream motor-imagery accuracy of certified weights versus random legal weights of identical shape, so that certification is shown to buy more than dimensional composability.

## Contribution

['A dependent-type-indexed parameter-manifold compiler (TYPED-WEIGHTS) that converts EEG/ECG weight generation into type inhabitation, with a structured type-error feedback channel that lets a language-model generator backtrack on violated architectural invariants.', 'The finding that type-error feedback monotonically reduces violation counts across retries, establishing that legality can be enforced at generation time rather than by post-hoc filtering for biosignal classifiers.', 'A certification record format (weights + index context + derivation trace) that makes the legality of a generated biosignal network auditable per-invariant.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | EEG motor-imagery subject subsets (physionet-style), 2 subjects held out for zero-shot transfer test | Subject subsets exercise residual-stack depth and channel-count invariants; held-out subjects test whether certified weights transfer |
| Primary metric | First-attempt typecheck pass rate, reported against a shape-only null baseline (random correctly-shaped tensors) | Directly measures generation-time legality of emitted tensors beyond shape conformance |
| Secondary metric | Downstream motor-imagery accuracy of certified weights vs. random legal weights of identical shape | Separates legality from functionality |
| Baseline | Gradient-descent training | Tests whether type inhabitation beats optimization from scratch |
| Baseline | Mixed Signals weight-space LM | Tests whether pre-emission certification beats post-hoc evaluation |
| Baseline | Tracr fixed-graph legality checker | Tests whether dependent indices generalize beyond one encoded graph |
| Baseline | Shape-only null (random correctly-shaped tensors) | Tests whether the LLM adds anything beyond shape conformance |
| Ablation-of-ours | TYPED-WEIGHTS minus type-error feedback | Isolates whether structured witnesses, not resampling, drive retries |
| Ablation-of-ours | TYPED-WEIGHTS with shape-only indices (no value-level indices) | Isolates whether value-level indices are necessary for legality |

### Why this setup validates the claim

The central claim is that type inhabitation, enforced before emission, yields legal weights for biosignal architectures that fixed-graph or post-hoc methods cannot certify. Gradient-descent training tests the sub-claim that a generator can reach legal, functional tensors without iterative optimization; if TYPED-WEIGHTS matches it on legality while emitting in one shot, the generation criterion is doing real work. Mixed Signals tests the sub-claim that pre-emission certification matters: it emits first and evaluates later, so its legality rate should collapse exactly where deep residual stacks make post-hoc rejection expensive. Tracr tests generality: its legality is bound to one computation graph, so on architectures outside that graph it either cannot certify or must be re-encoded, revealing whether dependent indices travel further. The shape-only null baseline tests the load-bearing assumption directly: if random correctly-shaped tensors achieve the same first-attempt pass rate as the LLM, then type inhabitation reduces to shape conformance and the LLM adds nothing beyond shape annotation. The ablation without type-error feedback isolates the causal role of structured witnesses versus blind resampling, and the shape-only-index ablation isolates whether value-level indices are necessary. First-attempt pass rate is the right primary metric because the dependent context Gamma is exactly the set of architectural invariants whose violation makes a tensor non-composable; a metric on downstream accuracy alone would conflate numerical quality with structural legality and could not distinguish a legal-but-weak tensor from an illegal one, so we report both. This combination is falsifiable: if legality gains are uniform across architectures rather than concentrated where invariants are hard to satisfy by chance, the mechanism claim is unsupported.

### Expected outcome and causal chain

**vs. Gradient-descent training** — On a deep residual stack with a tight temporal kernel-span bound, gradient descent reaches legal shapes only because the architecture fixes them, but it needs many optimization steps and can drift into numerically unstable regions that still pass shape checks. Our method instead emits a tensor only after discharging each index in Gamma, so residual-width, kernel-span, spectral-norm, and norm-coupling constraints are satisfied by construction rather than by convergence. We expect TYPED-WEIGHTS to show a large first-attempt legality advantage on deep stacks, with parity on shallow single-block architectures where the constraint surface is trivial, and we expect downstream accuracy of certified weights to be above random legal weights of identical shape, confirming that certification buys functionality, not just composability.

**vs. Mixed Signals weight-space LM** — On an architecture with several residual blocks and non-trivial head divisibility, the baseline emits a full tensor then evaluates it, so a single violated residual projection invalidates the whole emission and the violation location is lost. Our method instead typechecks before emission and returns a structured witness naming the violated index, feeding that witness back into the next generation attempt. We expect the baseline's legality rate to drop sharply with stack depth while ours stays high, and we expect violations-per-retry under our method to decrease monotonically across the capped retries, whereas the baseline shows no such trend. We also expect the shape-only null to match the baseline on shape conformance but fail on value-level indices, showing that pre-emission value-level certification is the differentiator.

**vs. Tracr fixed-graph legality checker** — On an architecture family outside the single computation graph Tracr encodes, such as a variant with a different attention-head factorization, the baseline cannot certify because its legality predicate is tied to that graph and would need re-encoding. Our method instead hand-writes index terms for the new family and checks the dependent context directly, so certification extends to the variant. We expect a visible gap on out-of-graph architecture variants but near-parity on the in-graph architecture, which is the signature that dependent indices, not graph encoding, carry the legality guarantee.

**vs. Shape-only null (random correctly-shaped tensors)** — Random correctly-shaped tensors satisfy every shape index by construction but violate value-level indices (spectral norm, kernel support, block structure, norm coupling) with high probability. We expect the null to achieve near-100% first-attempt pass rate on shape-only indices but near-zero on value-level indices, while TYPED-WEIGHTS achieves high pass rates on both. This gap is the direct evidence that the LLM is sensitive to symbolic value constraints beyond shape conformance, validating the load-bearing assumption.

### What would falsify this idea

If the first-attempt legality gain over Mixed Signals is uniform across shallow and deep architectures rather than concentrated on deep residual stacks and out-of-graph variants, the claim that type inhabitation is the causal mechanism would be wrong. Likewise, if the ablation without type-error feedback matches ours on retries-used, the structured-witness feedback is not doing the work we attribute to it. If the shape-only null matches TYPED-WEIGHTS on value-level indices, the LLM is shape-blind and the load-bearing assumption is false.

## References

1. Mixed Signals: Weight-Space Programming for EEG and ECG
