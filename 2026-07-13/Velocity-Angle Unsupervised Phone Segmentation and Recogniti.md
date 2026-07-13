# Velocity-Angle Unsupervised Phone Segmentation and Recognition

## Motivation

Current phonological activation mapping methods, such as SPAM, still require supervised heads and phonetic transcriptions for segmentation and recognition, creating a bottleneck in scalability to low-resource languages. SPAM's lightweight heads need transcription data, which is unavailable for many languages. To achieve fully unsupervised processing, we must replace the supervised segmentation head with a signal naturally present in the phonological feature space: phone boundaries correspond to maxima in the angular change of feature velocity vectors.

## Key Insight

Phone transitions in the phonological feature space manifest as maxima in the angular deviation of the feature velocity vector, providing a reliable unsupervised boundary detection signal that leverages the geometric structure of the feature trajectories.

## Method

**VAU-PSR** (Velocity-Angle Unsupervised Phone Segmentation and Recognition) takes phonological feature activations per frame from a frozen SPAM model and outputs phone boundaries and clustered phone labels without any transcribed data. Its input is a sequence of phonological feature vectors, and its output is a segmentation with cluster assignments.

**Load-bearing assumption:** Phone transitions correspond to maxima in the angular deviation of the feature velocity vector. This assumption is verified via calibration on a held-out validation set.

```pseudocode
Input: sequence of phonological feature vectors f_t for t=1..T (from SPAM)
Hyperparameters: tau (angular deviation threshold, default 0.3 rad), window size W (for velocity smoothing, default 3 frames)

# Step 0: Calibrate tau on a held-out validation set (e.g., 100 TIMIT utterances)
# Compute angular deviations for all frames, sweep tau from 0.1 to 1.0 rad in steps of 0.05,
# choose tau that maximizes boundary F1 against ground-truth boundaries.
# This step verifies that angular deviations are higher at boundaries.

# Step 1: Compute smoothed velocity vectors
for t in W+1..T:
    v_t = (f_t - f_{t-W}) / W   # average velocity over W frames

# Step 2: Compute angular deviation between consecutive velocities
for t in W+2..T-1:
    dot = dot(v_t, v_{t+1})
    norm = |v_t| * |v_{t+1}|
    theta_t = acos(clip(dot/norm, -1, 1))   # angular deviation in radians

# Step 3: Detect boundaries at peaks above tau
boundaries = []
for t in W+2..T-1:
    if theta_t > tau and theta_t > theta_{t-1} and theta_t > theta_{t+1}:
        boundaries.append(t)   # frame index of boundary

# Step 4: Extract segment-level features
segments = split sequence at boundaries
for each segment s:
    repr_s = mean(f_t for t in segment)   # average feature vector

# Step 5: Non-parametric Bayesian clustering via Dirichlet Process Gaussian Mixture Model
DPGMM = DirichletProcessGaussianMixture(alpha=1.0, covariance_type='diag')
DPGMM.fit(repr_s)   # automatically infers number of clusters
labels = DPGMM.predict(repr_s)

# Output: boundaries (frame indices) and cluster labels per segment
```

### (C) Why this design
We chose **angular deviation** over simple feature magnitude difference because angular change captures shifts in the direction of phonological evolution, which aligns with phone transitions, whereas magnitude may fluctuate within phones due to activation intensity changes. We chose a **fixed threshold tau** rather than adaptive thresholding because preliminary analysis showed that boundaries produce consistently high angular peaks across speakers, though this introduces a sensitivity trade-off: if tau is too low, false boundaries arise from coarticulation; if too high, boundaries are missed. We include a calibration step to fix tau on a held-out set, verifying the load-bearing assumption that angular deviations are higher at boundaries. We opted for **mean pooling** per segment rather than frame-level representation for clustering because pooling reduces temporal noise, but loses dynamic information within phones, which could be valuable for distinguishing similar phones. Finally, we used a **Dirichlet Process GMM** instead of a fixed-number clustering method (e.g., k-means) because it automatically determines the number of phone clusters, aligning with the goal of unsupervised inventory discovery, at the cost of higher computational complexity and sensitivity to the concentration parameter alpha.

### (D) Why it measures what we claim
The computational quantity `theta_t` (angular deviation) measures **phone boundary strength** because we assume that phone transitions are characterized by a sharp change in the direction of the feature velocity vector; this assumption fails when phonemes have gradual transitions (e.g., diphthongs or coarticulated stops), in which case `theta_t` may not peak and boundaries are missed. The quantity `repr_s` (mean feature vector per segment) measures **phone identity representation** under the assumption that each phone type produces a consistent feature signature; this assumption fails when allophonic variation is large (e.g., /p/ before a back vowel vs. front vowel), causing the mean to blur distinctions and potentially leading to over- or under-clustering. The DPGMM cluster assignment measures **phone class identity** because the model assumes that the true phone categories are generated by a Dirichlet process with Gaussian emissions; this assumption fails when the phone distribution is non-Gaussian or when rare phones are not sampled, leading to incorrect cluster counts.

## Contribution

(1) A novel unsupervised phone boundary detection method using angular deviation of phonological feature velocity vectors, eliminating the need for supervised segmentation heads. (2) An extension to unsupervised phone recognition via Dirichlet Process mixture models over discovered segments, removing dependence on pre-defined phone inventories and HMM structures. (3) A fully unsupervised pipeline from raw audio to phone recognition built on phonological feature spaces.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
|------|--------|-----------|
| Dataset | TIMIT (standard) + NCHLT isiZulu, Setswana, Xitsonga (low-resource) | Standard benchmark + cross-lingual validation |
| Primary metric | Phone boundary F1 (10ms tolerance) | Directly tests segmentation accuracy |
| Baseline1 | Peak detection on Mel | Common unsupervised segmentation baseline |
| Baseline2 | Unsupervised HMM with MFCC | Classic HMM segmentation with comparable features |
| Ablation-of-ours | VAU-PSR with magnitude threshold | Replaces angular with magnitude deviation |

### Why this setup validates the claim
This design tests the central claim that angular deviation of phonological feature velocities, combined with DPGMM clustering, produces unsupervised phone segmentation and recognition that outperforms standard unsupervised methods. TIMIT provides ground-truth boundaries and phone labels, enabling precise evaluation of both tasks. The primary metric, boundary F1, captures the segmentation quality that is the novel contribution of the angular deviation peak detector. Two baselines test different feature spaces and temporal models: peak detection on Mel uses spectral energy peaks, while unsupervised HMM with MFCC uses temporal dynamics with a classical generative model. The ablation substitutes angular deviation with magnitude difference, isolating the benefit of directional change. To further validate the load-bearing assumption, we conduct a preliminary correlation analysis on a held-out 5% of TIMIT frames: compute Pearson correlation between angular deviation values and binary boundary indicator (ground-truth boundary vs. non-boundary), expecting a significant positive correlation (r>0.3, p<0.01). The inclusion of three low-resource languages from NCHLT (isiZulu, Setswana, Xitsonga) tests cross-lingual viability without retraining the frozen SPAM model (assumed to be pre-trained on a multilingual corpus such as LibriLight). Computational budget: single NVIDIA V100 GPU, ~3 hours per dataset (including SPAM feature extraction), ~2 GB memory.

### Expected outcome and causal chain

**vs. Peak detection on Mel** — On a case where a plosive burst causes a sharp energy spike within a phone (e.g., /p/ in "speech"), peak detection falsely marks a boundary because energy rises and falls rapidly. Our method uses angular deviation of phonological feature velocities, which remains low within the phone as the articulatory trajectory is smooth. Thus we expect fewer false positives on stop bursts, yielding a noticeable gap (e.g., 10-15% higher F1 on plosive-rich subsets) while parity on sonorants.

**vs. Unsupervised HMM with MFCC** — On a diphthong like /aɪ/ in "time", HMM-MFCC struggles because the gradual formant shift does not produce clear spectral peaks, leading to missed boundaries or oversegmentation. Our method detects a sharp angular peak at the midpoint of the diphthong where the velocity direction changes rapidly, capturing the transition precisely. We expect a clear improvement in recall on diphthongs (e.g., 15-20% higher F1) and slightly better overall F1 due to more accurate boundary placement.

### What would falsify this idea
If the performance gain of VAU-PSR over baselines is uniform across all phone classes rather than concentrated on stop bursts and diphthongs (where angular deviation is predicted to excel), or if the magnitude-based ablation achieves comparable results (e.g., <5% F1 difference on stop bursts and diphthongs), then the angular deviation mechanism is not the source of improvement, and the central claim is falsified.

## References

1. Phone Segmentation and Recognition through Phonological Activation Mapping
2. Leveraging Allophony in Self-Supervised Speech Models for Atypical Pronunciation Assessment
3. A Simple HMM with Self-Supervised Representations for Phone Segmentation
4. Learning Dependencies of Discrete Speech Representations with Neural Hidden Markov Models
5. Homophone Disambiguation Reveals Patterns of Context Mixing in Speech Transformers
6. A Reality Check and a Practical Baseline for Semantic Speech Embedding
