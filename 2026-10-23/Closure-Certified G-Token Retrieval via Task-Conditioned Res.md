# Closure-Certified G-Token Retrieval via Task-Conditioned Residual Certificates

## Motivation

Across unified multimodal trees (GAFU, UniHetero, BAGEL), the retrieval interface for compact generation-side tokens is defined relative to an understanding pathway that is assumed correct: GAFU retrieves G tokens 'in a task-conditioned manner by the understanding pathway' with no constraint tying the selection rule to the pathway's own error modes, and UniHetero shows the generation-understanding coupling is real but treats the understanding module as the arbiter of what helps. This is a structural meta-gap: the arbiter and the interface are two separate objects, so when the understanding pathway errs the selection rule has no closure guarantee, and the compact token interface is assumed (not shown) to be compositionally sufficient for external reasoning. The root cause is that no shared quantity forces the retrieval rule and the pathway's residual error to be the same object.

## Key Insight

If the admissible G-token set is defined as the level set of a low-dimensional residual certificate that is a sufficient statistic of the understanding pathway's error, then the retrieval map is invariant to every pathway perturbation that preserves the certificate, so the interface carries its own closure witness rather than deferring to a second arbiter.

## Method

**What it is.** CLOSURE-G (Closure-Certified G-token retrieval) learns, per task, a G-token codebook and a residual certificate `c(x,t)` — a low-dimensional sufficient statistic of the understanding pathway's residual error — and defines retrieval as the unique codebook element whose certificate matches `c`. External reasoning blueprints may only query tokens through `c`, so the certificate and the interface are one object.

**How it works.**

```pseudocode
Inputs: unified model M = (generation pathway G, understanding pathway U)
        task t, image x, codebook size K=64, certificate dim d=8, margin m=0.1
        Phi: 2-layer MLP (residual_dim -> 256 -> 256, GeLU) trained jointly with W_t
        W_t: linear map (256 -> d=8), task-conditioned

# Phase 1: residual certificate extraction (frozen U)
r  = U.residual(x, t)              # per-token logit residual vs. U's own answer, dim = vocab_size
c  = W_t · Phi(r)                  # d-dim task-conditioned projection; W_t and Phi learned

# Phase 2: certificate-conditioned codebook
for k in 1..K:
    g_k = G.compress(x)[anchor_k]  # generation-side feature at anchor_k
    a_k = A_theta(c, t)[k]         # certificate-conditioned address (2-layer MLP, hidden=128)
score_k = -|| a_k - W_t·Phi(r) ||^2
retrieve S(c) = { g_k : score_k >= max_j score_j - m }   # margin level set

# Phase 3: closure training (task-conditioned)
L_ret  = -log softmax(score)[y]                       # retrieval correctness
L_cert = || c - c_target(x,t) ||^2                    # certificate predicts U's error mode
L_blue = -log p_blueprint(answer | S(c))              # blueprint must answer from S(c) alone
L = L_ret + lambda1*L_cert + lambda2*L_blue,  lambda1=0.5, lambda2=1.0

# Phase 4: closure check at inference
# Perturbation family: Gaussian noise eps ~ N(0, sigma^2 I) in U's penultimate layer,
# projected onto null(W_t·Phi) (i.e., directions that leave c unchanged).
# sigma = 0.01 * std(penultimate activations), 100 samples per input.
if not invariant(S(c), U-perturbations preserving c): flag 'no closure certificate'
return S(c)  # the ONLY channel blueprints may read
```

**Why this design.** We chose to make the certificate a *projection of U's residual* rather than a learned verifier head because a verifier bolted onto a selector reintroduces the second-arbiter problem that GAFU's task-conditioned retrieval already suffers from; the cost is that `W_t` must be identifiable, which we enforce by requiring `c` to reconstruct the residual's task-relevant subspace (Phase 3 `L_cert`). We chose a *margin level set* `S(c)` over top-1 selection because a single selected token cannot be checked for compositional sufficiency — the level set gives the blueprint a closed admissible output set, accepting the cost that retrieval precision drops when the margin `m` is large. We chose to gate blueprint access through `c` (Phase 2 address `A_theta(c,t)`) rather than expose raw tokens because this makes the certificate the sole query channel, accepting that blueprints lose direct token access and must be retrained against `c`. A domain expert would not describe this as a variant of Toolformer/Gorilla retrieval: those compute a descriptor and nearest-neighbor match with no residual certificate and no closure check, whereas here the retrieved set is defined as an invariant level set of U's own error statistic.

**Why it measures what we claim.** The certificate `c = W_t·Phi(r)` measures *the understanding pathway's error mode* because `Phi` is assumed to be a sufficient statistic of the residual's task-relevant subspace — i.e., two residuals with equal `c` induce the same retrieval behavior; this assumption fails when U's errors have task-irrelevant high-variance directions that `Phi` discards, in which case `c` reflects the dominant error direction rather than the full error. The margin level set `S(c)` measures *compositional sufficiency of the interface* because the blueprint loss `L_blue` is assumed to be minimized only when `S(c)` contains all tokens the blueprint needs; this fails when the blueprint can shortcut via priors, in which case `S(c)` reflects blueprint prior fit rather than interface sufficiency. The closure check measures *invariance of retrieval to U-perturbations* because perturbations preserving `c` are assumed to be exactly the perturbations that leave `S(c)` fixed; this fails when `Phi` is non-injective on the perturbation set, in which case the check reflects certificate stability rather than true closure.

**(D) Equivalence sub-blocks.**

- **D1: `c` measures U's error mode.** Assumption: `Phi` is injective on the task-relevant residual subspace (i.e., no two distinct task-relevant residuals map to the same `c`). Failure mode: high-variance task-irrelevant directions dominate `Phi`, so `c` reflects the dominant error direction rather than the full error. Operationalized by `L_cert` and the reconstruction error of the task-relevant subspace from `c` (reported in experiments).
- **D2: `S(c)` measures compositional sufficiency.** Assumption: `L_blue` is minimized only when `S(c)` contains all tokens the blueprint needs. Failure mode: blueprint shortcuts via priors, so `S(c)` reflects blueprint prior fit rather than interface sufficiency. Operationalized by `L_blue` and the accuracy on blueprint answers gated through `S(c)`.
- **D3: closure check measures invariance.** Assumption: perturbations preserving `c` are exactly those that leave `S(c)` fixed (i.e., `Phi` is injective on the perturbation set). Failure mode: `Phi` non-injective on the perturbation set, so the check reflects certificate stability rather than true closure. Operationalized by the finite perturbation family in Phase 4 and the false-flag rate on a validation split.

## Contribution

(1) A closure-certified G-token retrieval operator that defines admissible outputs as a margin level set of a task-conditioned residual certificate, making the error statistic and the token interface the same object rather than a selector plus verifier. (2) A task-conditioned certificate codebook training procedure whose blueprint loss forces the certificate to be the sole query channel, yielding a design principle: interface sufficiency is testable only when the interface carries its own closure witness. (3) A closure-invariance diagnostic that flags retrieval configurations where the certificate fails to be a sufficient statistic of the understanding pathway's error mode.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Fine-grained VQA with generation-side anchors (e.g., GAFU-style benchmark) | Tests residual-certificate retrieval where U's errors are fine-grained |
| Primary metric | Accuracy on blueprint answers gated through S(c) | Directly measures interface sufficiency, not raw retrieval recall |
| Baseline | GAFU task-conditioned retrieval | Tests whether certificate gating beats descriptor nearest-neighbor |
| Baseline | Standard VLM understanding-only | Tests whether generation-side tokens help at all |
| Baseline | Toolformer/Gorilla-style descriptor retrieval | Tests whether residual certificate adds over plain descriptor match |
| Ablation-of-ours | CLOSURE-G with margin m→0 (top-1 selection) | Tests whether the level set, not the certificate, drives gains |
| Ablation-of-ours | CLOSURE-G with injected task-irrelevant residual variance (D1 break) | Tests whether gain collapses when Phi sufficiency fails |
| Ablation-of-ours | CLOSURE-G with blueprint prior shortcut (D2 break) | Tests whether S(c) sufficiency is real or prior-driven |
| Ablation-of-ours | CLOSURE-G with non-injective Phi on perturbation set (D3 break) | Tests whether closure check reflects true invariance |

### Why this setup validates the claim

The central claim is that a residual certificate `c` of the understanding pathway's error mode, used as the sole query channel into a margin level set of generation tokens, yields compositional sufficiency that descriptor-based retrieval cannot. The GAFU baseline tests the sub-claim that task-conditioned retrieval without a residual certificate reintroduces a second arbiter; if our method beats GAFU specifically on cases where U's residual is high-variance in task-irrelevant directions, the certificate is doing work. The understanding-only baseline tests whether generation-side tokens contribute anything at all, isolating the retrieval mechanism from the generation pathway's raw value. The Toolformer/Gorilla baseline tests whether the residual certificate adds over a descriptor match — if we tie on it, the certificate is decorative. The top-1 ablation is the critical control: if the margin level set is what enables blueprint composition, collapsing it to top-1 should reproduce the baseline failure pattern even with the certificate intact. The D1–D3 ablations directly break each equivalence assumption: D1 injects task-irrelevant residual variance to test whether `c` remains a sufficient statistic; D2 adds a blueprint prior shortcut to test whether `S(c)` sufficiency is real; D3 uses a non-injective `Phi` on the perturbation set to test whether the closure check reflects true invariance. Accuracy on blueprint answers is the right metric because it is the only observable that requires S(c) to actually contain the tokens the blueprint needs; retrieval recall or certificate reconstruction loss would not detect a level set that is closed but insufficient.

### Expected outcome and causal chain

**vs. GAFU task-conditioned retrieval** — On a case where the understanding pathway's residual concentrates on a fine-grained attribute (e.g., distinguishing a subtle material from a similar one) while high-variance task-irrelevant residual directions dominate, GAFU's descriptor nearest-neighbor retrieves tokens matched to the dominant direction because it has no residual statistic to condition on. Our method projects the residual through `W_t` so `c` encodes the task-relevant subspace, and the margin level set admits the material-distinguishing token. We expect a noticeable accuracy gap concentrated on fine-grained attribute subsets, with parity on coarse object-identity subsets where the dominant direction is task-relevant.

**vs. Standard VLM understanding-only** — On a case requiring compositional reasoning over an image region the understanding pathway under-resolves (e.g., counting or spatial relation), the understanding-only model answers from U's own logits and fails because it has no generation-side feature to supplement the under-resolved region. Our method retrieves generation tokens anchored at that region via `c`, giving the blueprint a closed set containing the missing compositional pieces. We expect a gap that grows with the compositional depth of the question, and near-parity on single-hop attribute questions where U alone suffices.

**vs. Toolformer/Gorilla-style descriptor retrieval** — On a case where two images share a descriptor but differ in U's error mode (same scene, different residual pattern), descriptor retrieval returns the same token set for both because its descriptor discards the residual. Our method returns different level sets because `c` differs, and the closure check flags the case if invariance fails. We expect a gap on paired-instance subsets matched on descriptor but differing in residual, and parity on subsets where descriptor and residual are correlated.

**vs. CLOSURE-G with margin m→0 (top-1 selection)** — On a case requiring compositional reasoning over multiple tokens, top-1 selection returns a single token, so the blueprint cannot compose the needed pieces even if the certificate is correct. Our method's margin level set returns a closed set containing all tokens within `m` of the max score, enabling composition. We expect a gap on compositional subsets, with parity on single-token subsets where top-1 suffices.

**vs. CLOSURE-G with injected task-irrelevant residual variance (D1 break)** — On a case where we artificially inflate the variance of task-irrelevant residual directions (e.g., by adding Gaussian noise to U's penultimate layer along directions orthogonal to the task-relevant subspace), `Phi` discards these directions, so `c` reflects the dominant error direction rather than the full error. Our method's gain over GAFU should collapse because the certificate no longer measures U's error mode. We expect the accuracy gap to shrink monotonically with the injected variance, confirming D1.

**vs. CLOSURE-G with blueprint prior shortcut (D2 break)** — On a case where the blueprint can answer correctly from priors alone (e.g., common object co-occurrences), `L_blue` is minimized even if `S(c)` is insufficient. Our method's accuracy should remain high but the gap over a baseline that ignores `S(c)` should vanish, indicating that `S(c)` sufficiency is not the driver. We expect this ablation to show that gains on prior-solvable subsets are not evidence for compositional sufficiency, confirming D2.

**vs. CLOSURE-G with non-injective Phi on perturbation set (D3 break)** — On a case where `Phi` is non-injective on the perturbation set (e.g., by using a low-rank `Phi` that collapses multiple perturbation directions), the closure check may pass even when `S(c)` changes under perturbations that preserve `c`. Our method's closure check should flag fewer cases than the true invariance violation rate, indicating that the check reflects certificate stability rather than true closure. We expect the false-flag rate to increase with the degree of non-injectivity, confirming D3.

### What would falsify this idea

If the top-1 ablation (m→0) matches full CLOSURE-G on the fine-grained subsets, then the margin level set is not what enables compositional sufficiency and the central mechanism claim is wrong. If our gain over GAFU is uniform across fine-grained and coarse subsets rather than concentrated where the residual's task-irrelevant variance is high, the certificate is not measuring U's error mode as claimed.

## References

1. What Generation Knows: Unlocking Fine-Grained Understanding in Unified Multimodal Models
2. UniHetero: Could Generation Enhance Understanding for Vision-Language-Model at Large Data Scale?
3. Emerging Properties in Unified Multimodal Pretraining
4. Expanding Performance Boundaries of Open-Source Multimodal Models with Model, Data, and Test-Time Scaling
