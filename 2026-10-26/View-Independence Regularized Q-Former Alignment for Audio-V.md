# View-Independence Regularized Q-Former Alignment for Audio-Visual Hallucination Detection

## Motivation

Omni-Woodpecker's conflict detector assumes that frozen Q-former-aligned audio and visual views expose cross-modal conflicts, but because the encoders and projection are never trained to keep the two views statistically separable, correlated evidence collapses the view-divergence signal that the detector depends on; the inference-time verification action cannot repair this because it never touches encoder training. This is a recurring structural failure across the frozen-encoder alignment pipeline family (Video-LLaMA, Qwen2.5-Omni block-wise encoders, MIO's alignment pre-training stage): projection into a shared LLM embedding space is optimized only for reconstruction/next-token loss, which is invariant to whether the two modality views remain distinguishable, so downstream conflict-based mitigation inherits a degenerate divergence signal.

## Key Insight

Independence between audio and visual Q-former outputs is not a property the shared-embedding projection can create on its own, but it can be enforced as a training-time constraint on the pre-projection views so that view divergence remains a non-degenerate, conflict-sensitive statistic after projection.

## Method

**What it is.** VIR-Q (View-Independence Regularized Q-Former) is a training objective added to the alignment stage of a frozen-encoder Omni pipeline. Inputs: paired audio and video clips with their Q-former outputs $z_a, z_v \in \mathbb{R}^{d}$ (one pooled vector per clip per modality, $d=768$). Output: a Q-former whose audio and visual views remain separable after projection into the LLM embedding space, so that Omni-Woodpecker's view-divergence statistic $D = \|z_a - z_v\|$ is informative rather than collapsed.

**Load-bearing assumption (stated explicitly).** The method assumes that enforcing statistical independence between the pre-projection Q-former views via HSIC will cause the post-projection divergence $D = \|z_a_p - z_v_p\|$ to become a monotone, conflict-sensitive statistic that Omni-Woodpecker's orthogonal decomposition can read, even though the shared projection is trained only on captioning CE and is free to re-mix the views. This assumption is not guaranteed by HSIC on inputs (Gretton et al. 2005); it is measured and enforced by the post-projection probe and Lipschitz constraint described below.

**How it works.**

```pseudocode
# Per training batch of N paired (audio, video) clips
z_a, z_v = Qformer(audio), Qformer(video)       # frozen encoders, trainable Q-former, d=768
z_a_p, z_v_p = project(z_a), project(z_v)        # shared LLM embedding space, d_llm=4096

# (1) Reconstruction / captioning loss (standard alignment)
L_align = CE(LLM(z_a_p, z_v_p), caption)

# (2) Cross-view independence on PRE-projection views via HSIC
K_a = RBF(z_a, sigma=median_pairwise(z_a))       # N x N Gram, sigma set per batch
K_v = RBF(z_v, sigma=median_pairwise(z_v))
HSIC = trace(K_a @ H @ K_v @ H) / (N-1)^2        # H = centering matrix
L_indep = HSIC

# (3) View-separation margin: enforce D large on conflict pairs, small on agree pairs
#     conflict labels come from a frozen off-the-shelf AV conflict classifier (no gradient)
#     classifier: Qwen2.5-Omni-7B zero-shot prompted as conflict/agree, threshold 0.5
L_margin = mean( relu(m - D) [agree] ) + mean( relu(D - M) [conflict] )
#     m = 0.2, M = 1.0, D = ||z_a_p - z_v_p||

# (4) Post-projection invariance check (probe) and Lipschitz constraint
#     probe: linear regression from D to ground-truth conflict label, trained on held-out 512 clips
#     probe R^2 must exceed 0.3; if not, increase lambda_lip
L_lip = max(0, ||z_a_p - z_v_p|| - L_lip_bound * ||z_a - z_v||)  # L_lip_bound = 1.5

L = L_align + lambda_indep * L_indep + lambda_margin * L_margin + lambda_lip * L_lip
# lambda_indep=0.05, lambda_margin=0.1, lambda_lip=0.01
```

**Why this design.** We chose HSIC on the *pre-projection* Q-former views rather than on the post-projection LLM embeddings because the shared projection is precisely the operator that mixes the two views, so penalizing independence after projection would fight the alignment loss and degrade captioning; putting the constraint upstream lets the projection remain free to fuse while the views stay separable, accepting the cost that the constraint is one step removed from the space Omni-Woodpecker reads. We chose a *margin* term on the projected divergence in addition to HSIC because HSIC alone enforces only pairwise statistical independence and does not guarantee that the divergence magnitude tracks conflict, so we add a supervised separation term using frozen conflict pseudo-labels, accepting the cost of pseudo-label noise. We chose HSIC over an adversarial mutual-information estimator because HSIC has a closed-form unbiased estimator on finite batches and does not require a discriminator that can collapse, accepting the cost that HSIC is only a dependence proxy. We chose to keep encoders frozen (matching Video-LLaMA and Qwen2.5-Omni) rather than fine-tuning them, because the failure is in the projection/alignment stage, not the encoder, accepting the cost that encoder-level correlations are not removed. A domain expert would not describe this as Omni-Woodpecker-plus-a-loss because the mechanism is a training-time constraint on the alignment operator, not an inference-time hidden-state intervention; it is also not a generic "independence head" because the constraint is placed on the exact quantity (view divergence) that the downstream detector consumes, and the margin term ties that quantity to conflict labels. The post-projection probe and Lipschitz constraint make the transfer from pre- to post-projection geometry measured rather than assumed.

**Why it measures what we claim.** The HSIC statistic measures *cross-view statistical independence* because the assumption is that paired audio/video clips that share no content-generating factor should have Q-former representations whose kernel Gram matrices are uncorrelated; this assumption fails when the dataset contains clips where audio and video are genuinely redundant (e.g., a talking head), in which case HSIC reflects *dataset redundancy* rather than a Q-former defect. The margin term measures *conflict-sensitivity of view divergence* because the assumption is that the frozen conflict classifier's pseudo-labels are a valid proxy for ground-truth cross-modal conflict; this assumption fails when the classifier is systematically wrong on a modality (e.g., silent video), in which case the margin reflects *classifier bias* rather than true conflict. The downstream view-divergence statistic $D$ measures *recoverable conflict signal* because the assumption is that Omni-Woodpecker's orthogonal decomposition reads divergence monotonically; this assumption fails when the LLM's hidden state has already fused the views beyond the projection, in which case $D$ reflects *residual projection noise* rather than usable conflict evidence. The post-projection probe measures *transfer of pre-projection independence to post-projection divergence* because the assumption is that the Lipschitz constraint keeps the projection from re-mixing the views; this assumption fails when the projection's Lipschitz constant exceeds the bound, in which case the probe $R^2$ drops and the margin term operates on a distorted geometry.

## Contribution

(1) A training objective, VIR-Q, that places an HSIC independence penalty on pre-projection Q-former views plus a conflict-margin penalty on post-projection divergence, making the frozen-encoder alignment pipeline produce modality views that stay separable after projection. (2) The design principle that independence constraints belong upstream of the shared projection while conflict-sensitivity constraints belong downstream, because the projection is the operator that destroys separability. (3) A reproducible training recipe (hyperparameters, pseudo-label source, HSIC estimator) for integrating VIR-Q into any frozen-encoder Omni pipeline such as Video-LLaMA or Qwen2.5-Omni.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | AVHBench conflict subset (1,200 clips) and agree subset (1,200 clips) | Paired AV with labeled cross-modal conflict; agree subset controls for generic capability boost |
| Primary metric | Hallucination rate on conflict clips | Directly reflects conflict-driven hallucination |
| Secondary metric | Probe $R^2$ of linear regression from $D$ to ground-truth conflict label on held-out 512 clips | Measures whether post-projection divergence is conflict-sensitive |
| Baseline | Qwen2.5-Omni-7B unmitigated | Establishes raw backbone hallucination floor |
| Baseline | Omni-Woodpecker (inference-time) | Tests whether training-time constraint beats intervention |
| Baseline | Output-level mitigation (post-hoc filtering of captions by a frozen conflict classifier) | Tests whether divergence signal is needed at all |
| Ablation-of-ours | VIR-Q without margin term ($\lambda_{\text{margin}}=0$) | Isolates margin term's contribution to conflict sensitivity |
| Ablation-of-ours | VIR-Q with post-projection HSIC instead of pre-projection HSIC | Tests whether pre-projection placement is load-bearing |
| Ablation-of-ours | VIR-Q with mutual-information contrastive estimator instead of HSIC | Tests whether HSIC choice is interchangeable |
| Ablation-of-ours | VIR-Q without Lipschitz constraint ($\lambda_{\text{lip}}=0$) | Tests whether the invariance check is necessary |

### Why this setup validates the claim

The central claim is that constraining view separability during alignment makes the view-divergence statistic informative, and that this informativeness is what reduces hallucination on conflict clips. AVHBench's conflict subset is the right dataset because it is exactly the regime where audio and video disagree, so a method that only helps on redundant clips would show no gain there. The unmitigated backbone baseline tests the sub-claim that the raw pipeline already collapses divergence; the inference-time Omni-Woodpecker baseline tests the sub-claim that fixing the read-out without fixing the representation is insufficient; the output-level baseline tests whether any post-hoc correction can substitute for an informative divergence. The no-margin ablation is the load-bearing control: it keeps HSIC but removes the supervised separation, so if our gain survives without the margin, the margin was decorative, and if the gain vanishes, the margin is doing the causal work. The post-projection HSIC and MI-contrastive ablations test whether the pre-projection placement and HSIC choice are the novel, load-bearing choices rather than interchangeable regularizers. The Lipschitz ablation tests whether the invariance check is necessary for the transfer claim. Hallucination rate on conflict clips is the right metric because it is the quantity the divergence statistic is supposed to protect, and a uniform gain across conflict and agree subsets would signal a generic capability boost rather than the predicted mechanism. The probe $R^2$ directly measures the load-bearing assumption that pre-projection independence transfers to post-projection divergence.

### Expected outcome and causal chain

**vs. Qwen2.5-Omni-7B unmitigated** — On a case where the audio says "a dog barking" while the video shows a cat, the baseline produces a caption naming the dog because its Q-former views have collapsed into a shared content prior that the projection cannot separate, so the audio token dominates. Our method instead keeps the pre-projection views statistically independent via HSIC and pushes the projected divergence above the conflict margin, so the LLM receives two distinguishable evidence streams and the divergence statistic flags the disagreement. We expect a clear hallucination-rate drop concentrated on conflict clips (e.g., from 32% to 18%), with near-parity on agree clips (e.g., 8% to 7%). The probe $R^2$ should rise from ~0.05 (unmitigated) to >0.3.

**vs. Omni-Woodpecker (inference-time)** — On the same dog/cat case, the inference-time baseline reads a divergence statistic that is already collapsed, so its orthogonal decomposition attributes the conflict to noise and rebalances toward the wrong evidence, producing a confident but still wrong caption. Our method instead makes that same statistic informative at training time, so the baseline's own read-out mechanism would now have signal to act on. We expect our method to beat this baseline specifically on clips where the baseline's correction flips toward the wrong modality, with the gap largest where pre-projection collapse was strongest (measured by low probe $R^2$ on the unmitigated backbone).

**vs. Output-level mitigation** — On a case where the conflict classifier is correct but the caption is already generated, output-level filtering can only suppress or re-rank, not recover the correct evidence. Our method instead provides an informative divergence signal at generation time, so the LLM can attend to the correct modality. We expect our method to outperform output-level filtering on conflict clips where the correct answer requires integrating the non-dominant modality.

**vs. VIR-Q without margin term** — On a case where HSIC is satisfied but the projected divergence magnitude is small for a genuine conflict, the ablation produces views that are independent yet not conflict-separated, so the downstream statistic stays below threshold and the hallucination persists. Our method instead uses the margin term to tie divergence magnitude to pseudo-labeled conflict, so the same clip is pushed above the separation threshold. We expect the ablation to recover part of the gain on redundant clips but to lag our full method on conflict clips (e.g., 24% vs. 18% hallucination rate), isolating the margin as the conflict-specific driver.

**vs. VIR-Q with post-projection HSIC** — On a case where the projection re-mixes the views, post-projection HSIC directly penalizes dependence in the space Omni-Woodpecker reads, but it fights the captioning loss and may degrade caption quality. We expect post-projection HSIC to achieve similar conflict-clip hallucination reduction but with higher captioning loss (e.g., +0.5 CE) and lower agree-clip accuracy, demonstrating that pre-projection placement is the better trade-off.

**vs. VIR-Q with mutual-information contrastive estimator** — On a case where the discriminator collapses, the MI estimator provides a noisy independence signal. We expect HSIC to be more stable (lower variance across seeds) and to yield slightly better conflict-clip performance, confirming HSIC as the load-bearing choice.

**vs. VIR-Q without Lipschitz constraint** — On a case where the projection has a large Lipschitz constant, pre-projection independence does not transfer to post-projection divergence, so the probe $R^2$ drops and the margin term operates on distorted geometry. We expect the no-Lipschitz ablation to show lower probe $R^2$ (e.g., 0.15 vs. 0.35) and higher conflict-clip hallucination rate, confirming the necessity of the invariance check.

### What would falsify this idea

If our hallucination-rate gain is uniform across conflict and agree subsets rather than concentrated on conflict clips, the mechanism is not conflict-sensitive view separation but a generic capability boost, and the central claim is wrong. A second falsifier: if the no-margin ablation matches the full method on conflict clips, the margin term is not doing the causal work and the HSIC-only story collapses. A third falsifier: if the post-projection probe $R^2$ remains below 0.1 even with the Lipschitz constraint, the load-bearing assumption that pre-projection independence transfers to post-projection divergence is false.

## References

1. Omni-Woodpecker: Hallucination Correction via Evidence-Prior Rebalancing for Omni LLMs
2. Qwen2.5-Omni Technical Report
3. CosyVoice 2: Scalable Streaming Speech Synthesis with Large Language Models
4. MIO: A Foundation Model on Multimodal Tokens
5. F5-TTS: A Fairytaler that Fakes Fluent and Faithful Speech with Flow Matching
