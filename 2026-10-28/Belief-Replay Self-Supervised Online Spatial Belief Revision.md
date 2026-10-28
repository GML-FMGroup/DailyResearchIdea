# Belief-Replay: Self-Supervised Online Spatial Belief Revision for Streaming Vision-Language Models

## Motivation

Current spatial VLMs, including ChronoReplay, VideoLLM-online, and SpatialVLM, all treat spatial reasoning as a function of a fixed, offline-annotated observation set: ChronoReplay reorders offline annotations into chronological episodes but never revises its spatial beliefs when new unannotated frames arrive, and VideoLLM-online converts offline temporal annotations into streaming dialogue but still depends on those annotations for supervision. Structurally, this is because the spatial state is implicit in the language model's activations and is never given an explicit, updatable representation with a training signal that does not require labels; the model can only 'replay' what it was annotated on. This shared residual gap — online belief revision under unannotated streaming input — is the bottleneck the field's trajectory now exposes.

## Key Insight

A spatial belief is revision-stable if and only if it is a sufficient statistic for predicting the next frame's spatial observation, so a self-supervised next-belief prediction loss on unannotated streams forces the belief to encode exactly the spatial information that future observations will confirm or contradict, making label-free revision well-posed rather than a free-floating heuristic.

## Method

**What it is.** We introduce **SPARC** (Self-supervised Predictive And Revision-Consistent belief), a lightweight recurrent spatial belief module bolted onto a frozen spatial VLM (e.g., the ChronoReplay checkpoint). SPARC maintains a low-dimensional belief vector `b_t` (K=4 slots, each d_b=256) that is updated from each incoming unannotated frame's VLM visual tokens, and it is trained with a prediction-consistency objective that requires `b_t` to predict the VLM's spatial-token embedding of frame `t+1`, plus a revision-consistency term that requires the belief to reconstruct a held-out masked spatial region of the current frame.

**How it works.**

```pseudocode
# Frozen spatial VLM V (ChronoReplay checkpoint). Trainable: SPARC only.
# Hyperparameters: d_b=256 (belief dim), K=4 (belief slots), lambda_rev=0.5, lambda_var=0.1, tau=0.1, window=8, mask_ratio=0.25
# Momentum target encoder: V_mom = EMA(V) with decay 0.999 (updated per step)

initialize b_0 = zeros(K, d_b)
for each unannotated frame x_t in stream:
    h_t = V.visual_tokens(x_t)              # frozen, dim d_v
    z_t = V.spatial_pool(h_t)               # frozen spatial-token summary, dim d_v
    z_t_mom = V_mom.spatial_pool(V_mom.visual_tokens(x_t))  # momentum target, dim d_v

    # --- belief update (recurrent, gated) ---
    e_t   = W_e z_t                         # project to d_b
    g_t   = sigmoid(W_g [b_{t-1}; e_t])     # per-slot write gate
    c_t   = tanh(W_c [b_{t-1}; e_t])        # candidate belief
    b_t   = (1 - g_t) * b_{t-1} + g_t * c_t # convex revision: old belief retained where gate low

    # --- self-supervised prediction-consistency loss (momentum target) ---
    z_hat_{t+1} = W_p b_t                   # predict NEXT spatial summary
    L_pred = || z_hat_{t+1} - sg(z_{t+1}_mom) ||^2   # sg = stop-gradient on momentum target

    # --- revision-consistency loss (masked spatial reconstruction) ---
    # Mask a random 25% of spatial tokens in z_t, reconstruct them from b_t.
    m_t = random_binary_mask(d_v, ratio=0.25)  # 1 = masked
    z_t_masked = z_t * (1 - m_t)
    z_hat_t = W_r b_t                       # reconstruct masked tokens
    L_rev = || m_t * (z_hat_t - sg(z_t)) ||^2 / sum(m_t)  # only masked positions

    # --- variance/covariance regularizer (VICReg-style) ---
    # Flatten b_t to (K*d_b) vector; compute variance and covariance over batch.
    L_var = (1/(K*d_b)) * sum_j max(0, 1 - std(b_t[:,j])) + (1/(K*d_b)) * sum_{i!=j} cov(b_t[:,i], b_t[:,j])^2

    L = L_pred + lambda_rev * L_rev + lambda_var * L_var
    update SPARC params by Adam on L

at inference (no labels):
    b_t = update(b_{t-1}, z_t)   # belief revised online; downstream head reads b_t
```

At inference, the downstream spatial answer head (inherited from the frozen VLM) is conditioned on `b_t` via cross-attention, so spatial beliefs are revised as frames arrive without any annotation.

**Why this design.** We chose a *convex gated revision* `b_t = (1-g_t)b_{t-1} + g_t c_t` over a full overwrite because a full overwrite discards spatial information that earlier frames established and that later frames may still depend on, accepting the cost that the belief can lag behind abrupt spatial changes when `g_t` is under-confident. We chose a *next-belief prediction* target over reconstructing the current frame because predicting the future forces the belief to retain only information that is predictive of upcoming spatial structure, accepting the cost that transient, non-predictive spatial details are deliberately dropped. We chose to *freeze the VLM and train only SPARC* over end-to-end fine-tuning because end-to-end training would require the offline annotations that ChronoReplay already consumes and would re-introduce the very dependency we remove, accepting the cost that SPARC cannot correct errors baked into the frozen VLM's spatial tokens. We chose a *stop-gradient target* over a jointly-trained predictor because a jointly-trained predictor can collapse to a trivial constant, accepting the cost that the target is noisier; we additionally use a momentum-updated target encoder (EMA decay 0.999) to temporally smooth the target and a VICReg-style variance/covariance regularizer (lambda_var=0.1) to prevent dimensional collapse of `b_t`. We chose a *masked spatial reconstruction* for `L_rev` over driving a residual to zero because a zero-residual objective is trivially minimized by `W_r -> 0`, whereas reconstructing a held-out 25% of spatial tokens forces the belief to encode frame-level spatial content that is not already trivially present in the unmasked input. Unlike VideoLLM-online, which still requires offline temporal annotations converted into streaming dialogue, SPARC's only training signal is the frozen VLM's own future spatial tokens and masked current tokens — a domain expert would not describe SPARC as a variant of VideoLLM-online because VideoLLM-online's supervision is the annotation stream, whereas SPARC's supervision is a self-prediction consistency condition on the belief itself.

**Why it measures what we claim.** The prediction-consistency loss `L_pred` measures *belief revision quality* because it is equivalent to requiring `b_t` to be a sufficient statistic for the next spatial observation under the assumption that the momentum target encoder's spatial summary `z_{t+1}_mom` is an unbiased, temporally consistent encoding of the next frame's spatial content; this assumption fails when the frozen VLM's spatial tokens are systematically biased (e.g., on out-of-distribution scenes), in which case `L_pred` reflects the VLM's token bias rather than genuine spatial predictiveness. The revision-consistency term `L_rev` measures *belief-frame coherence* because it is equivalent to requiring the belief to reconstruct a held-out masked spatial region of the current frame under the assumption that the masked tokens are predictable from the unmasked context and the belief; this assumption fails when the frame contains spatial content the frozen VLM did not encode or when the masked region is intrinsically unpredictable, in which case `L_rev` reflects encoder coverage or intrinsic uncertainty rather than belief coherence. The gate `g_t` measures *revision necessity* because it is equivalent to the fraction of the belief that must change to predict the next frame under the assumption that the belief slots are linearly independent; this assumption fails when slots are correlated, in which case `g_t` overstates revision necessity. The variance/covariance term `L_var` measures *belief non-degeneracy* because it is equivalent to requiring each belief dimension to have unit variance and zero pairwise covariance under the assumption that the belief dimensions are meant to encode distinct spatial factors; this assumption fails when the true spatial factors are inherently correlated, in which case `L_var` penalizes useful redundancy. Together these quantities operationalize the motivation-level concept 'online belief revision under unannotated input' as a self-consistency condition on the belief trajectory.

## Contribution

(1) SPARC, a self-supervised recurrent spatial belief module that revises a frozen spatial VLM's spatial beliefs online from unannotated streaming frames via a next-belief prediction-consistency objective, removing the offline-annotation dependency that ChronoReplay and VideoLLM-online share. (2) A convex gated revision rule whose write gate is trained purely by prediction consistency, providing a label-free mechanism for deciding how much of the prior spatial belief to retain versus overwrite as new frames arrive. (3) A design principle: freezing the VLM and training only a low-dimensional belief module is sufficient to obtain online spatial belief revision, isolating the revision capability from the offline-annotation supervision that prior spatial VLMs entangle it with.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | OVO-S-Bench streaming spatial episodes (pre-registered split: 1,200 cross-frame questions, 1,800 single-frame questions) | Tests online spatial belief under unannotated frame streams |
| Primary metric | Streaming spatial QA accuracy (overall and per-subset) | Directly reflects belief revision quality at answer time |
| Baseline | ChronoReplay frozen checkpoint (no SPARC) | Isolates whether SPARC adds anything to frozen VLM |
| Baseline | Full-overwrite belief module (b_t = c_t) | Tests convex-gated revision vs. overwrite claim |
| Baseline | Current-frame reconstruction objective (L_pred replaced by ||W_p b_t - z_t||^2) | Tests next-prediction target vs. reconstruction claim |
| Baseline | Standard GRU world-model (single 1024-dim state, same L_pred + L_var) | Tests convex-gated revision vs. standard recurrent state |
| Baseline | SPARC with momentum target only (no L_var) | Isolates variance regularizer contribution |
| Ablation-of-ours | SPARC without revision-consistency term (lambda_rev=0) | Isolates contribution of L_rev to belief coherence |
| Ablation-of-ours | SPARC without variance regularizer (lambda_var=0) | Isolates contribution of L_var to non-degeneracy |
| Ablation-of-ours | SPARC with zero-residual L_rev (original placeholder) | Tests masked reconstruction vs. trivial residual |
| Transfer test | Same SPARC module trained on SpatialVLM frozen backbone | Tests backbone-agnostic mechanism |

### Why this setup validates the claim

The central claim is that a self-supervised, prediction-consistent recurrent belief bolted onto a frozen spatial VLM improves online spatial answering under unannotated streaming input, and that this gain comes specifically from convex gated revision, next-frame prediction, and non-degenerate belief regularization rather than from added capacity. OVO-S-Bench is the right dataset because it presents spatial questions mid-stream, so the answer depends on what the belief has retained and revised up to that point — exactly the quantity SPARC is designed to shape. The pre-registered split (1,200 cross-frame, 1,800 single-frame questions) ensures the falsification test is well-powered: with these counts, a 5% accuracy difference on the cross-frame subset is detectable at 80% power (two-proportion z-test, alpha=0.05). The frozen ChronoReplay baseline tests whether any gain exists at all beyond the pretrained spatial tokens; the full-overwrite baseline tests the convex-revision sub-claim; the reconstruction-objective baseline tests the next-prediction sub-claim; the standard GRU world-model baseline tests whether the slot-wise convex gate adds anything over a generic recurrent state; the momentum-only and no-L_var ablations test the anti-collapse mechanisms; and the no-L_rev and zero-residual ablations test whether the masked reconstruction term matters or is decorative. The SpatialVLM transfer test checks that the mechanism is backbone-agnostic rather than a single-checkpoint artifact. Streaming QA accuracy is the correct primary metric because it is measured at the moment the belief is read, so it detects lag, overwrite-induced forgetting, and incoherent revisions directly rather than through a proxy.

### Expected outcome and causal chain

**vs. ChronoReplay frozen checkpoint (no SPARC)** — On an episode where an early frame establishes a spatial layout (e.g., object on the left) and a later frame shows a partial view, the frozen checkpoint answers from the current frame's tokens alone because it has no persistent belief state, producing answers that contradict the earlier established layout. Our method instead carries `b_t` forward with a low gate on the layout slots, so the earlier spatial fact survives into the answer, and we expect a noticeable accuracy gap concentrated on the 1,200 cross-frame questions (target: +8–12 points) with near-parity on the 1,800 single-frame questions (target: ±1 point).

**vs. Full-overwrite belief module** — On a case where a frame is heavily occluded or low-information, the full-overwrite baseline replaces its entire belief with the current summary because `b_t = c_t`, discarding the layout that earlier frames established, and then answers subsequent questions from a corrupted state. Our method keeps `(1-g_t)b_{t-1}` with a low gate on uninformative frames, so the prior layout is retained and only the informative slots are revised, and we expect the gap to concentrate on episodes containing occlusion or abrupt viewpoint shifts (target: +6–10 points on the occlusion subset), with parity on clean, uniformly informative streams.

**vs. Current-frame reconstruction objective** — On a case where transient spatial detail (a passing distractor) appears in one frame and is irrelevant to later questions, the reconstruction baseline is forced to encode that detail into the belief because its loss rewards reconstructing the current frame, crowding out predictive layout information and hurting later answers. Our method instead drops non-predictive detail because `L_pred` only rewards information that predicts `z_{t+1}`, so the belief stays layout-focused, and we expect a modest but consistent gain on questions about persistent structure (target: +3–5 points) and no regression on questions about transient content.

**vs. Standard GRU world-model** — On episodes with abrupt spatial changes, the GRU baseline must overwrite its entire 1024-dim state because it lacks slot-wise gating, whereas SPARC's per-slot gate can revise only the affected slots. We expect SPARC to outperform the GRU baseline on the cross-frame subset (target: +4–6 points) and to show a slot-wise gate analysis where occlusion frames produce high gate values on a subset of slots (e.g., 1–2 of 4) rather than uniform gating.

**vs. SPARC with momentum target only (no L_var)** — Without the variance regularizer, we expect dimensional collapse of `b_t` (measured by effective rank < 50% of K*d_b) and a corresponding drop in cross-frame accuracy (target: −5–8 points vs. full SPARC), confirming that L_var is necessary for non-degenerate belief.

**vs. SPARC without revision-consistency term (lambda_rev=0)** — Without L_rev, the belief is trained only on next-frame prediction, so we expect a drop in belief-frame coherence (measured by masked reconstruction error on held-out frames) and a smaller but detectable drop in cross-frame accuracy (target: −2–4 points), confirming that the masked reconstruction term contributes beyond L_pred.

**vs. SPARC with zero-residual L_rev** — The zero-residual placeholder is trivially minimized by W_r→0, so we expect it to match the no-L_rev ablation (target: within 1 point), confirming that the masked reconstruction is the operative form of L_rev.

**vs. SpatialVLM transfer** — We expect the same SPARC module (same hyperparameters) trained on SpatialVLM to show a similar cross-frame gain over its frozen baseline (target: +6–10 points), confirming backbone-agnosticism.

### What would falsify this idea

If SPARC's accuracy gain over the frozen ChronoReplay checkpoint is uniform across the pre-registered single-frame and cross-frame question subsets (e.g., both +5 points), rather than concentrated on the 1,200 cross-frame questions, then the belief is not doing the causal work we claim and the central mechanism is unsupported. Likewise, if the full-overwrite and reconstruction baselines match SPARC on occlusion-heavy and transient-detail episodes respectively, or if the standard GRU world-model matches SPARC on cross-frame questions, the specific design choices are not the source of any gain.

## References

1. ChronoReplay: One Answer Interface Is Not Enough for Streaming Spatial Intelligence
2. SpatialVLM: Endowing Vision-Language Models with Spatial Reasoning Capabilities
3. VideoLLM-online: Online Video Large Language Model for Streaming Video
4. SpatialBot: Precise Spatial Understanding with Vision Language Models
5. RoboVQA: Multimodal Long-Horizon Reasoning for Robotics
6. RT-1: Robotics Transformer for Real-World Control at Scale
7. Interactive Language: Talking to Robots in Real Time
8. VideoCoCa: Video-Text Modeling with Zero-Shot Transfer from Contrastive Captioners
