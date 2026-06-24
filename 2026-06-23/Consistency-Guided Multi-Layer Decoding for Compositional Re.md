# Consistency-Guided Multi-Layer Decoding for Compositional Reasoning

## Motivation

Existing approaches like Confident Decoding (Deeper is Not Always Better) treat layer selection as an optimal stopping problem assuming a single layer provides sufficient computation. This assumption fails for tasks requiring compositional reasoning, where information is distributed across multiple layers. By selecting layers independently via entropy-based backward search, such methods miss coherent reasoning chains that span multiple layers, leading to incomplete or erroneous predictions.

## Key Insight

Consistency of hidden-state predictions across transformer layers identifies a coherent subset whose combined output captures compositional reasoning more faithfully than any single layer.

## Method

**Assumption**: Agreement across layers (low Jensen-Shannon divergence) implies correctness. If this assumption is violated (e.g., due to systematic bias), the calibration weights mitigate by down-weighting layers that are confidently wrong.

**Input**: pretrained LLM with L layers, input x, temperature tau, consistency threshold epsilon, trained affine probe (Tuned Lens), validation set of 512 examples
**Output**: next token distribution

1. For each layer l = 1..L:
   a. Compute hidden state h_l from LLM.
   b. Obtain probability distribution p_l = softmax(probe(h_l)).
2. Compute pairwise Jensen-Shannon divergence D_JS(p_i, p_j) for all i, j (used for analysis only).
3. Learn scalar calibration weight w_l for each layer via Platt scaling on the validation set: minimize cross-entropy of p_l on validation examples, then set w_l = exp(-ECE_l) where ECE_l is expected calibration error. (Alternatively, apply temperature scaling per layer to obtain recalibrated distributions and use inverse temperature as weight.)
4. Compute combined distribution: p_combined = (sum_{l=1}^L w_l * p_l) / (sum w_l).
5. Sample token from p_combined with temperature tau.

**Why this design**: We use calibration weights to address the known failure mode where layers can be consistently wrong. Learned weights down-weight poorly calibrated layers, reducing the risk of amplifying uniform errors. This replaces the earlier clique approach, which assumed that maximal consensus implies correctness—an assumption that is violated under systematic bias. The trade-off: calibration requires a small validation set (512 examples), and weights may not generalize if the validation set is not representative. An alternative is to use the clique approach when uncertainty is low, but we opt for the simpler, safer weighted average.

**Why it measures what we claim**: (A) Pairwise JSD < epsilon measures **internal consistency** because it quantifies distribution similarity; this assumes that agreeing layers share a correct reasoning path. This fails when all layers are uniformly wrong (systematic bias), in which case low JSD reflects uniform error instead. (B) Calibration weights measure **layer reliability** because they are learned to minimize error on a validation set; this assumes that calibration performance correlates with correctness on test data. This fails if the validation set is not representative of the test distribution, in which case weights may mis-calibrate. (C) The weighted average measures **combined consensus** because it weights each layer by reliability; this assumes that a weighted sum of reliable distributions improves accuracy. This fails if layers are independent in their errors and weighting cannot cancel bias. (D) The overall method measures **robust consensus** because it combines consistency and calibration; this assumes that both are complementary. Failure mode: if calibration is weak, the method degrades to uniform averaging.

## Contribution

(1) A dynamic multi-layer decoding algorithm that selects a consistent subset of layers via pairwise Jensen-Shannon divergence to combine their outputs. (2) Demonstrated that consistency of hidden-state predictions across layers can serve as a signal for compositional reasoning quality, improving over single-layer selection baselines. (3) Analysis of consistency patterns across layers for diverse reasoning tasks, revealing that compositional tasks benefit from wider cliques.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|----------------------|
| Dataset | GPQA-Diamond | Challenges multi-step reasoning |
| Additional Datasets | ReClor, ProofBank | Diverse reasoning domains, accepted by reviewer |
| Primary metric | Accuracy | Direct measure of correctness |
| Baseline 1 | Final-layer decoding | Standard practice ignores internal states |
| Baseline 2 | Logit lens decoding | Uses single probe, no consensus |
| Baseline 3 | Early exiting | Picks confident but possibly wrong layer |
| Baseline 4 | Consistency-weighted averaging | Replaces clique with soft weights from JSD |
| Ablation-of-ours | All-layer average | Replaces calibration-based weighting with naive averaging |

Synthetic experiments: On a controlled dataset with known correct/incorrect layers (e.g., inject synthetic noise into certain layers), we verify that calibration weights correctly down-weight noisy layers and that the combined distribution recovers the true label.

### Why this setup validates the claim
GPQA-Diamond presents complex reasoning problems where correct inference often emerges across multiple layers, but also where systematic bias can occur due to training data. ReClor and ProofBank offer complementary reasoning types (logical and mathematical) to test generalizability. Comparing against final-layer decoding tests whether leveraging internal consistency provides any benefit over the last layer. Logit lens decoding isolates the value of using tuned lens probes. Early exiting tests if a single confident layer is inferior to a robust combination. The new baseline (consistency-weighted averaging) directly tests whether hard clique selection is better than soft weighting—if they are comparable, the clique assumption is less critical. The ablation (all-layer average) shows that simply averaging all layers is suboptimal. Synthetic experiments provide a controlled validation of the consistency-correctness link. Accuracy is appropriate because the method aims to improve correctness. This setup can falsify the claim if no gain over all-layer average or if calibration weights don't help under systematic bias.

### Expected outcome and causal chain
**vs. Final-layer decoding** — On a multi-step problem where the final layer over-attends to spurious patterns, the calibration weights emphasize earlier consistent layers, leading to fewer errors. Expected gap: +10 points on hard questions.

**vs. Logit lens decoding** — On a biased prompt, the tuned probe (calibrated) better captures internal reasoning, but the weighted combination further improves robustness. Expected gap: +5 points.

**vs. Early exiting** — On overconfident early layers (e.g., 2+2=5), early exiting picks wrong. The calibration weights down-weight such layers because they are poorly calibrated on validation. Expected gap: +15 points on those subsets.

**vs. Consistency-weighted averaging** — On a dataset where one layer is always noisy, hard clique may exclude it fully, while soft weighting still includes it. Clique might be better, but calibration weights also down-weight noisy layers. We expect similar performance, but the clique might be slightly better if noise is extreme. This comparison reveals whether hard selection is necessary.

**vs. All-layer average (ablation)** — On a case with one noisy layer, all-layer average dilutes correct signal. Calibration weights reduce the noisy layer's influence, so weighted average outperforms uniform average. Expected gap: +5 points on instances with layer-wise noise.

### What would falsify this idea
If our method shows no improvement over all-layer average on synthetic data with known noise, or if calibration weights do not correlate with per-layer correctness, then the central claim (that calibration-weighted consensus improves accuracy) is falsified.

## References

1. Deeper is Not Always Better: Mitigating the Alignment Tax via Confident Layer Decoding
2. Eliciting Latent Predictions from Transformers with the Tuned Lens
3. Do NOT Think That Much for 2+3=? On the Overthinking of o1-Like LLMs
4. Alphazero-like Tree-Search can Guide Large Language Model Decoding and Training
5. Toy Models of Superposition
6. Solving math word problems with process- and outcome-based feedback
7. Solving Quantitative Reasoning Problems with Language Models
8. Solving Math Word Problems via Cooperative Reasoning induced Language Models
