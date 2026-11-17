# Joint Tree Acceptance for Composed Speculative Drafts

## Motivation

Speculative decoding systems increasingly compose heterogeneous draft sources into a single candidate tree — LoopFill fuses suffix-lookup drafts with shallow-loop predictions in one tree — yet verification remains per-token and per-depth, so the composed tree is never treated as a single proposal distribution. Consequently the acceptance test cannot certify distributional equivalence of the tree as a whole: per-token rules accept paths whose joint probability under the target differs from the product of per-token acceptances, and the same closure gap appears across GRIFFIN (token-alignment), PEARL (adaptive draft length), HASS (representation harmonization), and polybasic (multi-model drafts), none of which establishes joint acceptance closure under heterogeneous draft composition.

## Key Insight

Any tree of candidate paths can be flattened into a single proposal distribution over complete sequences whose partition function is computable in one pass, so a single residual-based acceptance test on whole paths — not on individual tokens — yields exact target-distribution sampling while accepting or rejecting entire composed paths at once.

## Method

**What it is.** JTAD (Joint Tree Acceptance Decoding) takes as input a target model p, a composed draft tree T whose nodes carry draft probabilities from heterogeneous sources (suffix-lookup counts and shallow-loop softmaxes), and a flattening order over root-to-leaf paths; it outputs a sampled sequence distributed exactly as p, together with the set of accepted paths. Unlike per-token verification, JTAD samples one complete path from a normalized joint proposal q_T over the tree's leaves and applies a single accept/reject test to that whole path.

**Load-bearing assumption (stated explicitly).** JTAD's exactness holds under three conditions on the composed tree T: (C1) the root-to-leaf paths partition the proposal support — every candidate sequence is reachable as exactly one path; (C2) each node's source probability is normalized (suffix-lookup counts are converted to a categorical over the node's children by dividing by the total count at that node; shallow-loop softmaxes are already normalized); (C3) no token sequence is emitted at two different depths. When C1–C3 hold, q_T is a valid proposal over the leaf set and the residual p_res(pi) ∝ max(0, p(pi) − q_T(pi)) is exact over that set. When they fail, JTAD falls back to the standard per-token residual test at the first position where the sampled path leaves the tree's support, so exactness is restored independent of tree coverage and source normalization; the joint proposal is retained only for path selection.

**How it works.**

```pseudocode
Input: target p, composed tree T (nodes with draft logprobs from suffix source and shallow-loop source), temperature tau, max path length L, leaf budget K_max=32
Output: accepted sequence y

0. Enforce C2 (per-node normalization): for each suffix-source node, replace raw count c with c / sum_{c' in children} c'; for each shallow-loop node, use the softmax logprob as-is.
1. Enumerate root-to-leaf paths P = {pi_1,...,pi_K}; each pi_k = (t_1,...,t_{L_k}).
   Enforce C3 (de-duplication): if two paths yield the same token sequence, merge them into one leaf with q_raw summed; if K > K_max, keep the K_max highest-q_raw paths and record the dropped mass m_drop = 1 - sum_kept q_raw.
2. For each path pi_k, compute joint draft logprob:
     log q_raw(pi_k) = sum_{i=1..L_k} log q_source(i)(t_i | prefix)
   where q_source(i) is the source that produced node i (normalized suffix count / shallow-loop softmax).
3. Normalize over paths: log q_T(pi_k) = log q_raw(pi_k) - logsumexp_j log q_raw(pi_j).
4. Sample one path pi* ~ q_T (categorical over K paths).
5. Compute target joint logprob of pi*: log p(pi*) = sum_i log p(t_i | prefix).
   Compute residual mass: log r = log p(pi*) - log q_T(pi*).
6. If r >= 0: accept pi* entirely; return pi*.
   Else: accept with probability exp(r); on reject, sample a replacement path from the residual distribution
     p_res(pi) ∝ max(0, p(pi) - q_T(pi))  (normalized over paths; computed by one target forward over all K paths in a batch).
7. Out-of-tree fallback (restores exactness when C1 fails): if the sampled path pi* reaches a leaf before length L and the target assigns non-zero mass to continuations outside T, or if m_drop > 0, run the standard per-token residual test at the first position where pi* leaves the tree's support: draw u ~ Uniform(0,1); accept the token if u < min(1, p(t)/q_T(t)); on reject, sample the replacement token from p_res(t) ∝ max(0, p(t) - q_T(t)) over the full vocabulary at that position. Continue autoregressively from the accepted prefix.
8. Return accepted path (truncated at first divergent token if the caller requires token-level output).
```

Hyperparameters: temperature tau used inside both p and q_source; max path length L fixed to the tree depth; K is the leaf count of T after de-duplication, capped at K_max = 32 (typically 8–32). The out-of-tree fallback in step 7 is the only place the full-vocabulary residual is computed; it fires only when C1 fails or m_drop > 0.

**Why this design.** We chose a single joint proposal over the whole tree rather than per-token acceptance (as in LoopFill, GRIFFIN, PEARL) because per-token rules cannot express the correlation between draft sources once a suffix lookup extends a shallow-loop prediction; the cost is that we must enumerate and normalize over all leaves, which is linear in K and dominated by one batched target forward. We chose a categorical sample of one path followed by a residual replacement over the full path set rather than accepting each token independently because only the joint residual preserves exactness under heterogeneous sources; the cost is that a single rejected path discards its entire prefix, so we compensate by batching all K paths through the target once. We chose exact path enumeration over approximate beam search because the partition function must be exact for the residual test to be distributionally equivalent to p; the cost is limited to trees whose leaf count fits one batch, which we set K <= 32. The out-of-tree fallback in step 7 is what makes the exactness claim independent of tree coverage: when the composed tree does not cover the target's mass, the per-token residual at the first out-of-tree position restores exactness, and the joint proposal is used only for path selection. A domain expert would not describe this as 'LoopFill with a different verifier': LoopFill inherits a per-depth full verification and its tree is only a scheduling device, whereas JTAD changes the object being sampled — the tree is the proposal distribution and the acceptance test operates on paths, with the partition function computed from the composed sources themselves.

**Why this measures what we claim.** The computational quantity log q_T(pi*) - log p(pi*) measures distributional equivalence between the composed tree proposal and the target because the assumption is that the tree's leaves partition the support of the proposal (every candidate sequence is reachable as exactly one root-to-leaf path); this assumption fails when two sources emit the same token sequence at different depths, in which case q_T double-counts that sequence and the residual test rejects more often than necessary, reflecting path multiplicity rather than target mismatch. The quantity p_res(pi) ∝ max(0, p(pi) - q_T(pi)) measures the residual mass that per-token verification leaves implicit because the assumption is that the rejection step can be carried out over the same flattened path set used for proposal; this fails when the target assigns mass to sequences outside the tree, in which case the residual sampler must fall back to ordinary autoregressive sampling for the remaining suffix, and the metric reflects only the in-tree correction. The joint log-probability sum over a path measures source heterogeneity because the assumption is that suffix-lookup counts and shallow-loop softmaxes are both normalized per node; this fails when a suffix source returns an unnormalized count, in which case the path probability is biased toward the source with larger counts, and the acceptance rate reflects source calibration rather than draft quality. The out-of-tree fallback in step 7 measures the fraction of target mass outside the enumerated leaf set because the assumption is that this fraction is small enough that the joint path test dominates the acceptance decision; this fails when m_drop is large, in which case the acceptance rate reflects the per-token fallback rather than joint path acceptance.

## Contribution

['A tree-level speculative sampling rule, JTAD, that flattens a composed draft tree from heterogeneous sources into a single normalized joint proposal and accepts or rejects whole paths via one residual test, replacing per-token and per-depth verification.', 'A construction that makes the partition function of the composed tree computable in one batched target forward, enabling exact target-distribution sampling from trees whose nodes originate from suffix lookup and shallow-loop prediction simultaneously.', 'An acceptance-path diagnostic that separates rejection due to source miscalibration from rejection due to target mismatch, giving a per-tree signal for when heterogeneous draft composition is worth its enumeration cost.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Seven text-generation benchmarks on looped LLMs (Ouro-2.6B, Huginn-3.5B) | Matches LoopFill's setting where composed draft trees arise naturally |
| Primary metric | Average accepted-path length and wall-clock speedup vs. greedy AR | Directly reflects joint-path acceptance yield |
| Secondary metric | Out-of-tree mass fraction m_drop and fallback-fire rate | Quantifies how often the joint path test dominates |
| Baseline | LoopAhead | Strongest pipelined self-speculative baseline with per-depth verification |
| Baseline | GRIFFIN | Token-alignment draft model with per-token acceptance |
| Baseline | PEARL | Adaptive draft length with parallel draft/verify phases |
| Ablation-of-ours | JTAD with per-token acceptance instead of joint path acceptance | Isolates the joint-path proposal as the causal ingredient |
| Ablation-of-ours | JTAD without de-duplication (C3 relaxed) | Isolates the multiplicity effect from the acceptance-object effect |
| Ablation-of-ours | JTAD without per-node normalization (C2 relaxed) | Isolates source-calibration effect from acceptance-object effect |

### Why this setup validates the claim

The central claim is that sampling and accepting a whole root-to-leaf path from a composed, heterogeneous tree proposal yields exact target-distributed output with higher acceptance than per-token verification, because per-token rules cannot express cross-source correlation once a suffix lookup extends a shallow-loop prediction. LoopAhead tests the sub-claim that pipelined shallow-loop drafting alone leaves acceptance on the table: if JTAD beats it, the gain must come from joint path acceptance rather than from the drafting scheduler, since both use the same looped-transformer shallow predictions. GRIFFIN tests the sub-claim against a strong trained draft model with token-level acceptance: parity or a gain here shows the joint tree proposal is competitive even without a separately trained aligner. PEARL tests the sub-claim against adaptive-length parallel drafting, which shares the mutual-waiting motivation but keeps per-token verification. The per-token-acceptance ablation is the decisive control: swapping joint path acceptance for per-token acceptance inside JTAD keeps the tree, the sources, and the batched target forward identical, so any drop isolates the acceptance object as the cause. The de-duplication ablation isolates the multiplicity effect: if removing de-duplication (C3 relaxed) collapses the gain, the gain is due to path multiplicity rather than joint acceptance. The normalization ablation isolates source calibration: if removing per-node normalization (C2 relaxed) collapses the gain, the gain is due to source calibration rather than joint acceptance. The secondary metric m_drop and fallback-fire rate quantify how often the joint path test dominates the acceptance decision; if m_drop is large, the significance claim is reframed around a bounded-K regime with an explicit cost model. Accepted-path length and wall-clock speedup are the right metric because the claim is about how much of each proposal survives verification and how much target compute that saves; a metric like perplexity or downstream accuracy would be blind to the acceptance mechanism and could not detect the predicted effect. Exactness is checked separately by confirming output matches greedy autoregressive decoding on the same inputs, which is a correctness gate rather than a performance claim; the out-of-tree fallback in step 7 is validated against a brute-force exact-sampling oracle on a small vocabulary (V=64) and a small tree (K=8) to confirm the fallback restores exactness when C1 fails.

### Expected outcome and causal chain

**vs. LoopAhead** — On a case where a shallow-loop prediction is immediately extended by a suffix lookup, LoopAhead processes tokens at different loop depths in the same batch but each token must reach drafting depth before the next enters, and its per-depth verification accepts or rejects tokens independently, so an extension that is jointly plausible but individually marginal gets truncated at the first divergent token. Our method instead samples one complete path from the normalized joint proposal over leaves, so the suffix extension and the shallow prediction are scored as a single correlated object, and the batched target forward scores all K paths at once. We expect a noticeable accepted-path-length gap concentrated on inputs with long repeated suffixes, with near-parity on inputs where suffix lookup rarely fires. The de-duplication ablation is expected to show a smaller but non-zero drop, confirming that part of the gain is due to path multiplicity rather than joint acceptance alone.

**vs. GRIFFIN** — On a case where the draft model's training and decoding contexts are misaligned, GRIFFIN applies loss masking and token-alignable inputs but still verifies token by token, so a misaligned token late in a path rejects the prefix that was already correct, wasting the accepted work. Our method instead conditions the accept/reject test on the whole path's joint likelihood ratio, so a single low-probability token does not discard the entire accepted prefix unless the path as a whole is improbable under the target. We expect comparable or better speedup on LLaMA-family benchmarks, with the gap widening on prompts where GRIFFIN's alignment corrections are weakest, and parity where alignment is already good. The normalization ablation is expected to show a drop when suffix counts are unnormalized, confirming that source calibration matters for the joint path test.

**vs. PEARL** — On a case where the draft length is poorly matched to input difficulty, PEARL mitigates mutual waiting via pre-verify and post-verify but retains fixed per-token verification semantics, so over-long drafts produce many rejected tokens that were never jointly viable. Our method instead normalizes over the leaf set, which adapts the effective proposal mass to the tree's actual branching, and the residual replacement reallocates rejected mass across paths rather than restarting autoregressively. We expect a speedup advantage over vanilla speculative decoding of the order shown by PEARL's own reported margins, with the largest gains on heterogeneous-branching inputs and parity on uniform low-branching inputs. The out-of-tree fallback is expected to fire rarely on these inputs; if it fires often, the gain is attributed to the fallback rather than joint path acceptance.

### What would falsify this idea

If the gain over LoopAhead and PEARL is uniform across all subsets rather than concentrated on inputs where suffix lookup extends shallow-loop predictions, then the joint-path acceptance object is not the cause and the central claim is wrong. Likewise, if the per-token-acceptance ablation matches joint path acceptance, or if the de-duplication and normalization ablations show that the gain is entirely due to path multiplicity or source calibration rather than joint acceptance, the proposed mechanism contributes nothing beyond the composed tree itself.

## References

1. LoopFill: Pipelined Multi-Token Speculative Decoding for Looped Transformers
