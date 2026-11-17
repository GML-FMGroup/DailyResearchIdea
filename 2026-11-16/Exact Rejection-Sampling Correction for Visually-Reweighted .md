# Exact Rejection-Sampling Correction for Visually-Reweighted Diffusion Drafters

## Motivation

MMFlash adaptively upweights visually dependent tokens during drafting and asserts losslessness because the target model verifies all proposed tokens, but it never re-derives the rejection-sampling kernel under the reweighted proposal; the same structural gap recurs across speculative-decoding work that reshapes the drafter without re-deriving the verifier (GRIFFIN's token-alignable training, HASS's harmonized representations, PEARL's parallel drafting). The root cause is that adaptive reweighting changes the proposal distribution q while the standard acceptance rule min(1, p/q) is only valid for the unmodified q, so an unmodified verifier cannot restore the target distribution p when q has been tilted by token-dependent weights.

## Key Insight

Treating MMFlash's token-level upweighting as a multiplicative tilt of the proposal distribution yields a closed-form corrected acceptance ratio whose derivation makes the tilted proposal exactly cancel, so the verified output is distributed as the target p regardless of how aggressively visual-dependence weights are applied.

## Method

**What it is.** We introduce TILT-AC (Tilted Acceptance Correction), a verifier-side correction that takes as input a target distribution p, a base drafter distribution q0, and a token-dependent tilt w (MMFlash's visual-dependence upweighting), and outputs an acceptance probability per proposed token such that the accepted sequence is distributed exactly as p. TILT-AC is applied only when the drafting step is instrumented to log the exact per-token sampling distribution q_used actually used to draw x_t; if the logged q_used equals q0 (i.e., no inference-time tilt is present), TILT-AC reduces to the standard min(1, p/q0) rule and adds no correction.

**How it works.**

```pseudocode
# Drafting (unchanged from MMFlash, with instrumentation)
for each draft position t:
    v_t = KL( p(.|x, visual) || p(.|x, no-visual) )      # visual dependence
    w_t = 1 + alpha * normalize(v_t)                     # tilt weight, alpha in [0, 2]
    q_t = softmax( log q0_t + log w_t )                  # tilted proposal (if MMFlash applies tilt at inference)
    x_t ~ q_t
    log q_used_t = q_t                                   # instrument: record exact sampling distribution

# Verification (TILT-AC, replaces min(1, p/q0) with min(1, p/q_used_corrected))
for each proposed token x_t:
    q0_t = drafter_dist(x_t)
    p_t   = target_dist(x_t)
    w_t   = 1 + alpha * normalize(v_t)
    q_tilted = q0_t * w_t / Z_t                          # Z_t = sum_v q0_t(v) w_t(v)
    q_used = log q_used_t                                # from instrumentation; equals q_tilted if tilt applied at inference
    r_t = min(1, p_t / q_used)                           # corrected acceptance ratio
    u ~ Uniform(0,1)
    if u < r_t: accept x_t
    else: sample x_t ~ normalize( max(0, p_t - q_used) ); break
```

**Why this design.** We chose to correct the acceptance ratio on the verifier side rather than retrain the drafter to absorb the tilt, because MMFlash's tilt is computed at inference time from the visual input and therefore cannot be baked into training; the cost is that the verifier must recompute Z_t per token, which adds a normalization pass over the vocabulary but no extra target-model forward pass. We chose a multiplicative tilt q_tilted ∝ q0 · w over an additive shift of logits because multiplicative tilting has a closed-form partition function Z_t and therefore admits an exact rejection-sampling kernel, whereas additive shifts would require numerical integration over the vocabulary that is neither exact nor cheap; the cost is that w must be strictly positive, which we enforce by w = 1 + alpha·normalize(v) with alpha ≥ 0. We chose to keep MMFlash's sparse-attention drafter unchanged rather than co-designing attention sparsity with the correction, because the correction is agnostic to how q0 was produced and co-design would couple two otherwise independent knobs; the cost is that we forgo potential joint gains from sparsity-aware tilting. A domain expert would not describe TILT-AC as a variant of MMFlash: MMFlash changes the drafter and leaves the verifier as standard speculative decoding, whereas TILT-AC changes the verifier's acceptance kernel to match the tilted drafter, and the two compose rather than compete.

**Why it measures what we claim.** The computational quantity `q_tilted = q0 · w / Z_t` measures the actual proposal distribution used to sample x_t because the sampling step draws from exactly this normalized tilted distribution, under the assumption that w is computed from the same visual input that conditions q0; this assumption fails when the visual-dependence estimate v_t is computed with dropout or a different image resolution than the drafter's forward pass, in which case q_tilted reflects a stale tilt rather than the sampling distribution actually used. The quantity `r_t = min(1, p_t / q_used)` measures the rejection-sampling acceptance probability that restores the target p because the standard rejection-sampling argument requires the proposal in the denominator to be the distribution from which x_t was drawn, under the assumption that the drafter's per-token distribution is normalized and the target p is the same distribution used at verification; this assumption fails when the target model's p is computed with a different KV-cache prefix than the one used to compute q0, in which case r_t reflects an acceptance ratio between mismatched contexts rather than the true p/q_used ratio. The quantity `Z_t` measures the normalization that makes the tilt a valid probability distribution because it is the exact sum of q0·w over the vocabulary, under the assumption that the vocabulary is closed and finite at drafting time; this assumption fails under vocabulary truncation (top-k sampling), in which case Z_t reflects a truncated partition function and the acceptance ratio is exact only over the truncated support, so we report speedups with and without truncation to expose this boundary. The instrumentation `log q_used_t` measures the actual sampling distribution because it is recorded at the exact point where x_t is drawn, under the assumption that the logging code is not bypassed by a fused kernel or a different sampling path; this assumption fails when the drafter uses a compiled sampling routine that does not expose the normalized distribution, in which case q_used_t reflects a recomputed approximation rather than the true sampling distribution, and we fall back to the standard verifier and report the mismatch rate.

## Contribution

['A corrected rejection-sampling acceptance kernel (TILT-AC) for speculative decoding with token-tilted drafters, derived by treating adaptive upweighting as a multiplicative proposal tilt and computing the exact acceptance ratio against the normalized tilted proposal.', 'A design principle: any inference-time drafter reshaping that can be expressed as a positive multiplicative tilt admits a closed-form verifier correction, making adaptive drafting lossless by construction rather than by empirical assertion.', 'An open reference implementation and evaluation protocol that reports acceptance-rate versus tilt-strength curves on multimodal benchmarks, exposing where the correction is exact and where vocabulary truncation breaks exactness.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Multimodal video QA benchmarks with high visual-dependence tokens | Probes the exact regime where tilt matters |
| Primary metric | Acceptance length (tokens accepted per verification) | Directly reflects acceptance-kernel correctness and speedup |
| Baseline | MMFlash (tilted drafter, standard verifier) | Tests whether verifier correction adds over tilt alone |
| Baseline | EAGLE-3 | Tests against a strong non-tilted speculative baseline |
| Baseline | Vanilla SD with MMFlash drafter, no tilt | Isolates tilt contribution from correction contribution |
| Ablation-of-ours | TILT-AC with alpha=0 (correction disabled) | Checks that the correction, not the code path, drives gains |
| Diagnostic | Empirical proposal distribution vs q0·w/Z (TV distance) | Confirms the load-bearing assumption that tilt enters sampling |

### Why this setup validates the claim

The central claim is that MMFlash's inference-time tilt changes the actual proposal distribution, so the standard verifier's min(1, p/q0) ratio is mismatched and TILT-AC's corrected min(1, p/q_used) restores exactness while preserving the tilt's acceptance benefit. To falsify this, we need a comparison that separates the tilt from the correction. MMFlash (tilted drafter, standard verifier) tests whether the correction adds anything once the tilt is present: if TILT-AC matches or beats MMFlash's acceptance length, the correction is doing real work; if it ties, the verifier mismatch was benign. EAGLE-3 tests whether the whole tilted-plus-corrected pipeline beats a strong non-tilted baseline on the same multimodal benchmarks. The vanilla-SD-with-MMFlash-drafter baseline isolates the tilt's contribution from the correction's, and the alpha=0 ablation checks that the correction path itself, not incidental code changes, produces the gain. The diagnostic directly measures the empirical proposal distribution against q0·w/Z via total-variation distance on a held-out set of 512 prefixes; if TV distance is near zero, the tilt is not entering sampling and TILT-AC reduces to the standard rule. Acceptance length is the right metric because it is the direct observable of a correct rejection-sampling kernel: a mismatched verifier either over-accepts (distribution drift, undetected by acceptance length alone) or under-accepts (lost speedup). We therefore pair acceptance length with a distribution-match check on the accepted outputs to catch over-acceptance, which is exactly the failure mode a wrong denominator produces. Together these form a falsifiable test: the correction must show a gain concentrated where visual dependence is high, not uniformly.

### Expected outcome and causal chain

**vs. MMFlash** — On a case where a token has high visual dependence (top quartile, where MMFlash reports acceptance dropping toward 22%), MMFlash's drafter proposes from q_tilted but its verifier accepts with min(1, p/q0), so the denominator reflects the untilted proposal and the acceptance decision is systematically miscalibrated against the actual sampling distribution. Our method instead recomputes the denominator as q_used (logged from the sampling step) with the exact partition Z_t, so the acceptance ratio matches the distribution the token was actually drawn from. We expect a noticeable acceptance-length gain concentrated on the high-visual-dependence subset, with parity on the low-dependence quartile where the tilt is near-uniform. The diagnostic TV distance between empirical proposal and q0·w/Z should be below 0.01 on the high-dependence subset if the tilt is truly an inference-time sampling modification.

**vs. EAGLE-3** — On a case where the drafter is a diffusion-style block proposer and the visual input strongly conditions the next token, EAGLE-3's drafter is not tilt-aware, so its proposals underweight visually dependent tokens and acceptance drops on exactly those positions, forcing more target-model forwards. Our method keeps the tilt in the proposal and corrects the verifier to match, so the acceptance benefit of tilting survives verification instead of being clipped by a mismatched ratio. We expect a speedup gap that widens with the fraction of high-visual-dependence tokens in the sequence, and near-parity on text-only or low-dependence inputs.

**vs. Vanilla SD with MMFlash drafter, no tilt** — On a case where visual dependence is high but the drafter is run untilted, the proposal q0 is well matched to the standard verifier, so acceptance is exact but leaves the tilt's acceptance gains on the table; the pipeline is correct but slower than it could be. Our method adds the tilt and corrects the verifier, so it should dominate: exactness preserved and acceptance length higher. We expect the gain over this baseline to track the tilt magnitude alpha, growing from parity at alpha=0 to a clear acceptance-length advantage at moderate alpha, which also confirms the alpha=0 ablation recovers baseline behavior.

### What would falsify this idea

If TILT-AC's acceptance-length gain over MMFlash is uniform across visual-dependence quartiles rather than concentrated in the high-dependence quartile, the correction is not addressing the verifier-mismatch mechanism we claim. If the diagnostic TV distance between the empirical proposal distribution and q0·w/Z is near zero (indicating the tilt does not enter sampling), then the load-bearing assumption fails and TILT-AC reduces to the standard verifier, so the central claim is unsupported.

## References

1. MMFlash: Faster Speculative Decoding for Vision-Language Models
