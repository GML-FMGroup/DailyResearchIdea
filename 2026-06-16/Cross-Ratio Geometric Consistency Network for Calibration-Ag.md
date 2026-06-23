# Cross-Ratio Geometric Consistency Network for Calibration-Agnostic Stereo and Manipulation Verification

## Motivation

Existing hybrid consistency methods like HGC-Net rely on epipolar geometry that requires known camera calibration, and dexterous manipulation representations such as DexRepNet assume perfect depth sensing. Both fail under real-world sensor imperfections (unknown intrinsics, depth noise, textureless regions) because they depend on metric geometry that is not invariant to these variations. The root cause is the underutilization of projective invariants like the cross-ratio, which can be computed directly from collinear points without calibration.

## Key Insight

Cross-ratio is a projective invariant that remains unchanged under any projective transformation, so it can serve as a calibration-agnostic and noise-robust measure of geometric consistency as long as collinear points can be identified in the scene.

## Method

### (A) What it is
**CRGC-Net** (Cross-Ratio Geometric Consistency Network) takes as input either a stereo pair (for stereo consistency assessment) or a point cloud with line features (for manipulation verification) and outputs a scalar consistency score. The score is computed from cross-ratio differences of corresponding collinear point quadruples across views or time, aggregated by a learned attention mechanism.

### (B) How it works
```pseudocode
1. Input: Stereo pair (I_L, I_R) or point cloud P
2. Feature extraction: Detect line segments (LSD, threshold τ=5 pixels min length, min 10 pixels)
3. For each line, sample N=10 collinear points uniformly → set of quadruples Q (all combinations of 4 points on same line, but to reduce combinatorial explosion we sample at most 50 quadruples per line via random selection)
4. Compute cross-ratio for each quadruple: CR = (a-b)*(c-d) / ((a-c)*(b-d)) where (a,b,c,d) are projective coordinates along the line (normalized by line length)
   - For stereo: compute CR_L and CR_R, difference dCR = |CR_L - CR_R|
   - For manipulation: compute CR_object from current point cloud, CR_template from reference, difference dCR = |CR_object - CR_template|
5. Aggregation: Feed all dCR values into a learnable attention layer (single-head, key/query/value dimensions 64) that outputs weights via softmax, then scalar score S = softmax(MLP(dCR)) · dCR
   - MLP: 2-layer, hidden dim 64, ReLU activation
   - Hyperparameters: learning rate 1e-4, batch size 16, Adam optimizer, trained for 50 epochs on synthetic data
6. Output: Consistency score S (lower means more consistent)
```

### (C) Why this design
We chose cross-ratio over epipolar geometry (as in HGC-Net) because it is invariant to unknown camera intrinsics, eliminating the need for calibration; the cost is that cross-ratio requires collinear points (abundant in line-rich scenes but scarce in purely textured ones), limiting applicability to environments with linear features. We chose LSD for line detection over learning-based methods (e.g., DeepLSD) to maintain calibration-agnostic operation (no learned camera biases), accepting lower detection accuracy in noise. We chose attention-based aggregation over simple averaging to weight reliable quadruples more heavily; this adds computational overhead but improves robustness to outliers (e.g., mismatched lines). Together, these decisions prioritize invariance and robustness over generality.

### (D) Why it measures what we claim
Every dCR value measures geometric inconsistency because cross-ratio is invariant under any projective transformation; the assumption is that corresponding quadruples across views or time are related by a projective transformation (true for planar scenes or a single moving camera with constant intrinsics, but not for general stereo with non-zero baseline on non-planar scenes). Under non-projective deformations (e.g., non-rigid deformations or wide-baseline stereo on 3D scenes), dCR measures shape change rather than calibration error. The attention layer weights each dCR by its reliability (learned from synthetic data), so the final score reflects the most consistent subset of quadruples under the projective assumption, isolating calibration error from other noise sources. **Load-bearing assumption (explicit):** Cross-ratio is invariant only if corresponding quadruples lie on a plane or are related by a projective transformation; for arbitrary 3D scenes with significant baseline, this assumption fails. Therefore, the method is best suited for planar scenes or temporal consistency from a single moving camera; for general stereo, self-calibration would be necessary.

## Contribution

(1) A novel geometric consistency metric based on cross-ratio that operates without camera calibration or depth priors. (2) A deep learning architecture (CRGC-Net) that integrates this metric into a trainable consistency score for both stereo and manipulation tasks. (3) An empirical demonstration that cross-ratio features are more robust to synthetic calibration errors and depth noise compared to epipolar-based methods (e.g., HGC-Net).

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
|------|--------|-----------|
| Dataset (synthetic) | Middlebury 2014 stereo with synthetic distortions (random focal length [0.8,1.2]*original, principal point shift up to 10% image dimensions, additive Gaussian noise σ=0.05 rad for depth noise) | Standard dataset for stereo consistency with controlled distortions |
| Dataset (real-world) | YCB-Video dataset (subset of 20 scenes with line-rich objects: block, cracker box, etc.) | Demonstrates practical impact on robotic scenes with real sensor noise |
| Primary metric | Binary classification accuracy of misalignment detection (threshold: S > 0.1 → misaligned) | Directly measures consistency assessment |
| Baseline 1 | HGC-Net (epipolar geometry) | Epipolar baseline requiring calibration |
| Baseline 2 | CR-mean (cross-ratio with mean aggregation) | Tests need for attention mechanism |
| Baseline 3 | CRGC-Net with anharmonic ratio (anharmonic = log of cross-ratio, equivalent but numerically stable) | Isolates unique benefit of standard cross-ratio formulation |
| Ablation | CRGC-Net with uniform attention weights (all weights = 1/|Q|) | Removes learned attention to test its contribution |

### Why this setup validates the claim
This setup isolates the central claims: cross-ratio invariance eliminates calibration dependency, and attention improves robustness. The synthetic dataset with controlled distortions provides ground truth misalignment, enabling controlled tests. The YCB dataset tests generalization to real robotic scenes with depth noise and textureless regions. Baseline HGC-Net tests if epipolar geometry fails when calibration is unknown. Baseline CR-mean tests if simple aggregation suffices. Baseline 3 (anharmonic ratio) tests whether the specific formulation of cross-ratio matters, or if any projective invariant would work. The ablation tests if learned attention is critical. Accuracy on misalignment detection directly reflects how well the consistency score captures true geometric error, making this a falsifiable test of the method's ability to assess consistency without calibration.

### Expected outcome and causal chain

**vs. HGC-Net** — On a case where stereo intrinsics are unknown and distorting, HGC-Net computes epipolar error that depends on assumed intrinsics, leading to inconsistent scores that correlate poorly with actual misalignment because it assumes known calibration. Our method instead uses cross-ratio, which is invariant to intrinsics, so we expect high accuracy on misalignment detection (e.g., >85%) even with random intrinsics, whereas HGC-Net performs near chance (<60%) on unseen camera parameters.

**vs. CR-mean** — On a case with outlier line detections (e.g., due to noise or mismatches), CR-mean averages all cross-ratio differences, including large outliers, producing unreliable scores that fail to distinguish small misalignments. Our attention mechanism weights quadruples by learned reliability, suppressing outliers, so we expect consistent improvement (e.g., 5-10% accuracy gain) specifically on noisy subsets where outliers occur, while performing similarly on clean data.

**vs. Anharmonic ratio (Baseline 3)** — On synthetic data with moderate noise, anharmonic ratio yields identical performance to cross-ratio because it is a monotonic transformation. However, on real data with large numerical errors, anharmonic ratio may be slightly more stable; we expect CRGC-Net to match or slightly outperform Baseline 3, indicating that the specific formula is not critical (the invariance property is what matters).

### What would falsify this idea
If the accuracy gap between CRGC-Net and CR-mean is uniform across all distortion levels (i.e., no advantage on noisy data), the claim that attention improves robustness would be falsified. If HGC-Net matches our performance even under unknown intrinsics, the core invariance claim fails. If the anharmonic ratio baseline significantly outperforms standard cross-ratio on real data, then the choice of cross-ratio formula may be suboptimal.

## References

1. Hybrid Deep–Geometric Approach for Efficient Consistency Assessment of Stereo Images
2. DexRepNet++: Learning Dexterous Robotic Manipulation With Geometric and Spatial Hand-Object Representations
3. DexRepNet: Learning Dexterous Robotic Grasping Network with Geometric and Spatial Hand-Object Representations
