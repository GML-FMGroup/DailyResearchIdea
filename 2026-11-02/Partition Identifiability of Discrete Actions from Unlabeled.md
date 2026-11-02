# Partition Identifiability of Discrete Actions from Unlabeled Gameplay Video

## Motivation

Across the interactive video generation literature, the discrete action variable is never recovered from the data itself: GameNGen-style methods require recorded RL actions for supervised next-frame prediction, and Yan's hierarchical autoregressive captioning translates actions into text before conditioning the VDM. Both therefore assume the action-to-frame-change mapping must be supplied externally, which means compositional generalization to unseen action combinations can only come from enumerating them in captions or from held-out RL logs. The structural question — when is a finite action set recoverable from raw frame transitions alone — is never posed, so no existing objective makes the action variable identifiable from unlabeled gameplay video.

## Key Insight

If a finite action set acts locally injectively on frames modulo nuisance transformations, the local frame-transition manifold admits a partition whose number of equivalence classes equals |A| and whose boundaries are invariant to camera jitter, texture, and lighting, so the action label is the unique class assignment recovered by clustering transitions rather than a signal that must be supervised or verbalized.

## Method

**What it is.** We introduce PACT (Partition-Action Consistency Training), which takes unlabeled gameplay video and outputs (i) a discrete action codebook of cardinality |A| discovered from data and (ii) a frame-transition encoder whose latent space is partitioned into exactly |A| nuisance-invariant cells, used to condition an autoregressive video generator without any recorded actions or captions.

**Load-bearing assumption (stated explicitly).** PACT assumes that the nuisance-invariant contrastive latent space preserves the action partition as a set of well-separated clusters whose cardinality equals |A|, i.e., that discarding nuisance dimensions via InfoNCE does not also discard or entangle the action-relevant dimensions that define the partition boundaries. Concretely, we assume (A1) the augmentation set N acts as a group on the transition manifold whose orbits are contained within true action cells, and (A2) the between-cell margins exceed bootstrap sampling noise. If A1 fails (e.g., a camera pan is itself action-induced), the contrastive step can suppress action-relevant dimensions, collapsing distinct actions into one cluster and making the stability plateau report |A_eff| < |A| for reasons unrelated to non-injectivity. We therefore add two diagnostics (Step 3b and Step 4b) that measure orbit containment and per-cell margin, so that a plateau at K < |A| can be attributed to non-injectivity rather than to contrastive dimensional collapse.

**How it works.**

```text
Input: video V = (f_1 ... f_T), candidate cardinality K (swept over [2..12]), nuisance aug set N
Step 1. Transition extraction
    For each t: x_t = Enc(f_t, f_{t+1})            # patch-token pair encoder: ViT-B/16 backbone,
                                                    # tokens from both frames concatenated, 2-layer MLP head to d=128,
                                                    # L2-normalized; 8-frame stride
Step 2. Nuisance-invariant contrastive pretraining
    For each x_t, draw n_t = Aug(x_t) with camera jitter (random crop 0.8-1.0, rotation ±5°),
        texture (random grayscale p=0.2, color jitter 0.4), lighting (brightness/contrast 0.4)
    Loss L_inv = -log exp(sim(x_t,n_t)/tau) / sum_j exp(sim(x_t,n_j)/tau), tau=0.1
    Train Enc with AdamW lr=1e-4, batch=256, 100 epochs; freeze Enc after this step.
Step 3. Finite-partition clustering (the core operation)
    Initialize K centroids c_1..c_K by k-means++ on L2-normalized x_t
    Alternate for 200 iters:
        assign: a_t = argmin_k || x_t - c_k ||^2
        update: c_k = mean of assigned x_t
    Record per-iteration assignment agreement A_k = fraction of transitions
    whose label is unchanged between consecutive iterations
Step 3b. Orbit-containment diagnostic (tests assumption A1)
    For each transition x_t and its augmented view n_t, compute the latent displacement
        d_aug = n_t - x_t
    and the between-cluster displacement
        d_between = c_{a_t} - c_{a'_t}, where a'_t = argmin_{k != a_t} ||x_t - c_k||^2
    Report the mean cosine similarity cos(d_aug, d_between) over all t.
    If cos > 0.3, augmentation moves transitions toward other clusters (A1 violated);
    flag the dataset and report the fraction of transitions with cos > 0.3.
Step 4. Cardinality selection via stability plateau
    Sweep K in [2..12]; for each K compute partition stability
        S(K) = mean over t of agreement of a_t across 5 bootstrap resamples
    Select K* = smallest K where S(K) plateaus (delta S < 0.01)
    Freeze codebook C* = {c_1..c_K*}
Step 4b. Per-cell margin diagnostic (tests assumption A2)
    For each cluster k, compute margin m_k = min_{t: a_t=k} ( ||x_t - c_k||^2 - min_{j != k} ||x_t - c_j||^2 )
    Report the distribution of m_k across cells. If any m_k < 0.1 * median(m), flag that cell as
    low-margin; a plateau at K < |A| with low-margin cells indicates non-injectivity (A2 failure).
Step 5. Action-algebra probe (compositionality check)
    For pairs (a,b) observed consecutively, compute transition composition
    T_ab = c_a + c_b projected onto manifold; test commutativity of T_ab vs T_ba
    by cosine distance < 0.15 -> store commutation graph G
    Validate G on a synthetic dataset with known action algebra (e.g., 4 actions with
    known commutation relations) before trusting it as a diagnostic.
Step 6. Conditioning interface
    Emit (x_t, a_t) pairs; downstream autoregressive generator conditions on a_t
    via cross-attention to frozen codebook C*.
```

**Why this design.** We chose a *stability-plateau* cardinality criterion over a Bayesian nonparametric prior (e.g., Dirichlet process) because the identifiability claim is about partition *boundaries* being stable under nuisance, not about posterior concentration, accepting that we must sweep K explicitly rather than infer it in one pass. We chose *contrastive nuisance invariance* over reconstruction-based denoising because reconstruction lets the encoder cheat by encoding texture, whereas contrastive pairs force the latent to discard exactly the nuisance dimensions, accepting the cost that the augmentation set must cover the true nuisance factors or the partition will absorb them. We chose *hard k-means assignment* over soft EM because the identifiability criterion requires a discrete partition whose cardinality is counted exactly, accepting the cost of non-differentiability (we therefore train the encoder with L_inv only and freeze it before clustering). A domain expert would not describe PACT as a variant of Toolformer-style retrieval or of GameNGen-style supervised next-frame prediction: retrieval has no notion of partition cardinality, and GameNGen never discovers the action label — it consumes it.

**Why it measures what we claim.** The per-iteration assignment agreement A_k measures *partition stability* because the assumption that nuisance augmentation is a group action on the transition manifold implies stable transitions map to the same cell across resamples; this assumption fails when augmentation introduces label-changing effects (e.g., a lighting change that mimics a state transition), in which case A_k reflects augmentation artifacts rather than action identity. The bootstrap stability S(K) measures *identifiability of |A|* because the assumption that the true partition has |A| cells with margins larger than sampling noise implies stability plateaus exactly at K=|A|; this assumption fails when two actions are locally non-injective (produce identical frame deltas), in which case S(K) plateaus at the effective cardinality |A_eff| < |A| and reports a merged action class. The commutation graph G measures *action algebra* because the assumption that sequential composition of transition codes is associative on the manifold makes T_ab vs T_ba a valid test of commutativity; this assumption fails when the generator's latent is not closed under composition, in which case G reflects encoder geometry rather than game mechanics. The orbit-containment diagnostic (Step 3b) measures *whether augmentation orbits are contained within true action cells* because the assumption that N is a group action implies d_aug is orthogonal to d_between; this assumption fails when augmentation mimics action-induced effects (e.g., camera pan), in which case cos(d_aug, d_between) > 0.3 and the contrastive step suppresses action-relevant dimensions. The per-cell margin diagnostic (Step 4b) measures *whether between-cell margins exceed sampling noise* because the assumption that true cells are well-separated implies all m_k are positive and comparable; this assumption fails when two actions are non-injective, in which case some m_k are near zero and the plateau reports |A_eff| < |A|.

## Contribution

['PACT, a training objective and pipeline that recovers a discrete action codebook of cardinality |A| from unlabeled gameplay video via nuisance-invariant contrastive encoding plus a stability-plateau cardinality criterion, requiring no recorded actions, captions, or game-specific priors.', 'A design principle: partition stability under nuisance augmentation is the operational signal that separates true action identity from visual nuisance, and the plateau of bootstrap stability as a function of K is the empirical signature of the identifiability condition.', 'An action-algebra probe that extracts a commutation graph over discovered action classes, enabling compositional generalization tests on unseen action pairs without caption enumeration.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Unlabeled gameplay video with known action logs (held-out) | Ground-truth actions only for scoring, never training |
| Primary metric | Adjusted mutual information between predicted and true action partitions | Penalizes both over- and under-segmentation of cardinality |
| Baseline | GameNGen-style supervised next-frame predictor | Tests whether labels are needed vs. discovered |
| Baseline | k-means on raw frame deltas (no invariance) | Tests whether nuisance contrastive step matters |
| Baseline | Dirichlet-process clustering of transitions | Tests whether stability-plateau beats one-pass cardinality |
| Ablation-of-ours | PACT without nuisance contrastive pretraining | Isolates the invariance contribution to partition quality |
| Diagnostic | Orbit-containment (cos(d_aug, d_between)) and per-cell margin m_k | Tests load-bearing assumption A1/A2 |

### Why this setup validates the claim

The central claim is that PACT recovers a discrete action partition matching true game mechanics from unlabeled video, with cardinality selected by stability rather than supervision. The supervised GameNGen-style baseline tests the necessity of action labels: if PACT approaches its partition quality without labels, the discovery claim holds. The raw k-means baseline tests the nuisance-invariance sub-claim: without contrastive pretraining, camera jitter and lighting should corrupt assignment boundaries, so any gain PACT shows over it is attributable to invariance. The DP-clustering baseline tests the cardinality sub-claim: it infers K in one pass via a prior, so if PACT's stability plateau recovers K* more accurately, the explicit sweep is justified. The ablation removes invariance while keeping everything else, isolating that single mechanism. Adjusted mutual information is the right metric because it is cardinality-aware: a method that merges two actions or splits one is penalized, directly detecting the failure modes the design predicts (merged classes under non-injectivity, absorbed nuisance cells). The orbit-containment and per-cell margin diagnostics directly test the load-bearing assumption: if cos(d_aug, d_between) > 0.3 on a dataset, the contrastive step may be suppressing action-relevant dimensions, and any AMI drop can be attributed to that violation rather than to non-injectivity; if margins m_k are near zero for some cells, a plateau at K < |A| is diagnostic of non-injectivity rather than contrastive collapse.

### Expected outcome and causal chain

**vs. GameNGen-style supervised next-frame predictor** — On a case where the generator must condition on a rare action (e.g., a brief crouch) that appears sparsely in video, the supervised baseline produces blurred or averaged transitions because it consumes recorded labels and never learns a stable discrete cell for rare actions, overfitting frequent ones. Our method instead assigns rare transitions to their own stable cell because nuisance contrastive pretraining pulls them into a distinct cluster with margins exceeding sampling noise, so we expect near-parity on frequent actions but a noticeable AMI gap concentrated on the rare-action subset. The per-cell margin diagnostic should show that the rare-action cell has m_k > 0.1 * median(m), confirming it is not a low-margin artifact.

**vs. k-means on raw frame deltas (no invariance)** — On a case where a camera pan coincides with a genuine state change, the raw baseline produces a spurious split because texture and viewpoint dimensions dominate the L2 distance, so the partition absorbs nuisance as extra cells. Our method instead maps both the pan and the true transition to the same cell because contrastive pairs force the latent to discard viewpoint dimensions, so we expect the raw baseline to overshoot K* (plateau at K>|A|) while PACT plateaus at the true cardinality, yielding a clear AMI gap on camera-jitter-heavy subsets. The orbit-containment diagnostic should show cos(d_aug, d_between) < 0.3 on these subsets for PACT, confirming that augmentation orbits are contained within action cells.

**vs. Dirichlet-process clustering of transitions** — On a case where two actions are locally non-injective (identical frame deltas), the DP baseline produces a merged class in one pass without diagnostic signal because its prior cannot distinguish effective from true cardinality. Our method instead reports a stability plateau at |A_eff|<|A| and flags the merge via the commutation graph, so we expect comparable AMI on injective subsets but a diagnostic divergence on non-injective ones, where PACT's stability curve reveals the merge rather than silently absorbing it. The per-cell margin diagnostic should show near-zero m_k for the merged cell, distinguishing non-injectivity from contrastive collapse (which would show low margins across many cells).

### What would falsify this idea

If PACT's AMI gain over the raw k-means baseline is uniform across nuisance-heavy and nuisance-light subsets rather than concentrated where camera jitter and lighting are present, the invariance mechanism is not doing the work claimed. Likewise, if the stability plateau S(K) fails to align with the true cardinality on injective-action datasets, or if the orbit-containment diagnostic shows cos(d_aug, d_between) > 0.3 on datasets where PACT still claims success, the cardinality-selection and load-bearing assumptions are wrong.

## References

1. Yan: Foundational Interactive Video Generation
2. HunyuanVideo: A Systematic Framework For Large Video Generative Models
3. From Slow Bidirectional to Fast Autoregressive Video Diffusion Models
4. StreamEval: Evaluating Interactive Streaming Video Generation Models
