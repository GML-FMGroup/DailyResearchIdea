# Temporal Path Keyframe Extraction via Coherence-Aware Dynamic Programming

## Motivation

Existing keyframe extraction methods, such as TASKER, score frames independently based on task relevance and scene dynamics, discarding temporal dependencies between frames. This structural limitation prevents the selected keyframe set from coherently capturing action transitions and semantic progression, which is critical for video understanding tasks like VideoQA and agentic planning. Without modeling inter-frame coherence, the selected frames may omit intermediate states or introduce redundant jumps, degrading downstream task performance.

## Key Insight

Temporal coherence can be enforced by formulating keyframe selection as a path optimization problem in a graph where edge weights reflect optical flow consistency and semantic similarity, ensuring that selected frames form a temporally smooth sequence that preserves the rate of semantic change.

## Method

### (A) What it is
Temporal Path Keyframe Extraction (TPKE) is a method that selects a set of keyframes from a video by solving a maximum-weight path problem in a temporal consistency graph. Input: a video with T frames, a salience score s(t) per frame (e.g., from a task-relevance model like VideoQA attention weights), and a coherence cost c(t,t') between any two frames t<t'. Output: a sequence of keyframe indices (t_1,...,t_K) that maximizes total salience minus total coherence cost, under a constraint on the number of keyframes K.

### (B) How it works
```python
# Pseudocode for TPKE
Input: video frames F[1..T], salience scores S[1..T], coherence model C(t,t'), max keyframes K
# Hyperparameters: lambda=0.5 (coherence weight), alpha=0.7 (optical flow vs. semantic weight)
# Optical flow computed using RAFT (Teed and Deng, 2020) at resolution 224x224, average magnitude normalized by max flow in video
# Semantic embeddings from CLIP ViT-B/32 (Radford et al., 2021) 512-dim
Output: keyframe indices KF[1..K]

# Step 1: Build temporal consistency graph G with nodes 1..T
# Edge weight from i to j (i < j): w(i,j) = S[j] - lambda * C(i, j)
# where C(i,j) = alpha * optical_flow_cost(i,j) + (1-alpha) * semantic_cost(i,j)
# optical_flow_cost(i,j): mean optical flow magnitude between frame i and j, computed as average over all pixels at each intermediate frame, then normalized by the maximum such mean over all pairs in the video (range [0,1])
# semantic_cost(i,j): 1 - cosine similarity between CLIP embeddings of frames i and j (range [0,2] but typically [0,1])

# Step 2: Dynamic programming for K-length path
# DP[k][t] = max total score for path of k keyframes ending at frame t
# Complexity O(T^2 K) with T up to 5000 (pre-processed to reduce to 500 frames via uniform sampling if T>500)
# Initialize DP[1][t] = S[t] for all t
for k in 2..K:
    for t in k..T:
        DP[k][t] = max_{i < t} ( DP[k-1][i] + S[t] - lambda * C(i,t) )
        best_prev[t] = argmax_i ( DP[k-1][i] - lambda * C(i,t) )
# Step 3: Backtrack to find optimal path
# Choose t* = argmax_t DP[K][t]
# Reconstruct path using best_prev
KF = [t*] 
for k from K down to 2:
    t* = best_prev[t*]
    KF.prepend(t*)
return KF
```

### (C) Why this design
Three key design decisions: (1) **Weighted edge function**: We combine optical flow and semantic similarity into a single coherence cost, with a linear interpolation parameter α. We chose this over a learned fusion network because it is interpretable and requires no training data, accepting the cost that the optimal α may vary across video domains. (2) **Dynamic programming over greedy selection**: DP guarantees the globally optimal path under the given edge weights, whereas greedy per-step selection would sacrifice future coherence for immediate salience. The trade-off is O(T^2 K) complexity, which is acceptable for offline extraction but may be slow for real-time applications. (3) **Salience as additive node reward**: We add the salience of the destination frame to the path score rather than incorporating salience into the edge weight, which decouples the independent frame quality from temporal coherence. This design assumes salience scores are sufficiently reliable; if salience is noisy, the path may still select low-salience frames to maintain coherence. Compared to TASKER's independent scoring, our method explicitly couples frames, and unlike simple uniform sampling, it adapts to content dynamics.

### (D) Why it measures what we claim
The computational quantity `-lambda * C(i,j)` in the edge weight measures **temporal coherence** because it penalizes transitions with high optical flow magnitude or low semantic similarity, under the explicit assumption that a coherent action transition should exhibit small motion and consistent semantics between consecutive keyframes. This assumption fails when an action involves rapid but meaningful changes (e.g., a jump cut or fast motion), in which case the cost may incorrectly penalize a necessary transition, and the metric reflects motion magnitude rather than semantic discontinuity. The `S[t]` term measures **frame salience** because it is derived from task relevance (e.g., attention weights from a VLM), assuming that important frames are those that the model would attend to for answering questions. This assumption fails when the task relevance model is misaligned with the actual question, in which case S[t] reflects model bias rather than true importance. The combined DP objective `max Σ (S[t] - lambda * C(i,j))` operationalizes the motivation-level concept of **temporal coherence in keyframe sets** because it enforces that the selected frames form a path maximizing salience while minimizing pairwise discontinuity, assuming that the ideal keyframe sequence is a smooth path through the video manifold. This assumption fails when the optimal keyframe set includes intentional discontinuity (e.g., summary of distinct events), in which case the objective over-penalizes jumps and the selected frames may miss important non-consecutive events. To calibrate the hyperparameters α and λ, we use a small validation set of 100 videos with human-annotated keyframes that optimize downstream VideoQA accuracy (accuracy on a held-out question set). We grid-search α in {0.3,0.5,0.7,0.9} and λ in {0.1,0.3,0.5,0.7} to maximize validation accuracy. This calibration step ensures the coherence metric is grounded in task performance, partially mitigating the failure modes.

## Contribution

(1) A novel keyframe extraction algorithm (TPKE) that formulates selection as a maximum-weight path problem in a temporal consistency graph, enforcing inter-frame coherence via optical flow and semantic similarity. (2) A design principle that temporally coherent keyframe sets can be obtained by balancing frame salience with pairwise coherence, which is operationalized through a dynamic programming solution with tunable hyperparameters. (3) An empirical finding that TPKE improves downstream VideoQA accuracy by up to 5% over TASKER on the VideoMME benchmark (to be verified).

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | VideoMME (full set, 900 videos) | Tests generalization across video domains. |
| Primary metric | Accuracy (exact match) | Direct measure of answer correctness. |
| Additional metric | Human coherence score (Likert 1-5) | Human evaluation of semantic progression in selected keyframes (50 random videos, 3 annotators). |
| Baseline 1 | Uniform sampling (K frames spaced evenly) | Baseline for temporal coverage. |
| Baseline 2 | Greedy selection (pick top-K salience frames sequentially, no coherence) | Isolates coherence contribution. |
| Baseline 3 | TASKER (Sun et al., 2022) with default parameters | Strong learned keyframe extractor. |
| Baseline 4 | Submodular selection (facility location function) | Stronger optimization baseline. |
| Ablation-of-ours | TPKE (λ=0) | Removes coherence penalty from objective. |

### Why this setup validates the claim
Accuracy on VideoMME directly tests if TPKE’s keyframes retain task-relevant information for video QA. Uniform sampling isolates the benefit of adaptive selection; greedy selection tests whether coherence adds value beyond salience; TASKER compares against a learned baseline; submodular selection provides a state-of-the-art combinatorial baseline. The ablation with λ=0 quantifies the coherence term’s impact. The human coherence score validates that our coherence metric aligns with human perception of action progression. If TPKE outperforms all four baselines and the ablation, the central claim that temporal coherence improves keyframe selection is substantiated. This setup is falsifiable because each baseline targets a specific sub-claim: if TPKE fails against uniform sampling on temporally diverse videos, the adaptive selection fails; if it ties with the ablation, coherence is redundant.

### Expected outcome and causal chain
**vs. Uniform sampling** — On a video with fast action followed by slow scenes (e.g., a cooking recipe with quick chopping then slow simmering), uniform sampling evenly spreads keyframes, missing critical quick cuts. Our method uses salience peaks and coherence costs to concentrate frames in dense motion, capturing the chopping steps. Expect a noticeable accuracy gap on high-variation videos (e.g., >10% absolute improvement) but near parity on static videos.

**vs. Greedy selection** — On a video with a sudden scene change (e.g., cut from exterior to interior), greedy picks both salient frames but jumps cause a large coherence cost. Our method may drop one frame to maintain a smooth path, selecting a transitional frame that better preserves context. Expect our method to outperform on videos with abrupt transitions (e.g., >5% gain), while greedy may have higher recall on isolated salient moments.

**vs. TASKER** — On out-of-domain video (e.g., a novel GUI task not in TASKER’s training set), TASKER’s learned features may miss relevant frames. Our handcrafted coherence (optical flow + CLIP) and task-relevant salience generalize better. Expect our method to match or exceed TASKER on out-of-domain subsets (e.g., similar accuracy), while TASKER leads on in-domain (e.g., >3% higher).

**vs. Submodular selection** — On videos with multiple distinct events (e.g., news compilation), submodular selection may evenly cover events, while our coherence path may skip transitions. Expect submodular to perform similarly or slightly better on such videos, but on action sequences with smooth progression, our method should have higher accuracy (e.g., >3% improvement).

### What would falsify this idea
If TPKE does not outperform uniform sampling on videos with temporal variation, or if the ablation (λ=0) matches TPKE’s accuracy, then the coherence term provides no benefit, challenging the central claim that temporal coherence in keyframe selection improves video QA performance.

## References

1. Bridging VideoQA and Video-Guided Agentic Tasks via Generalized Keyframe Extraction
2. Tool-Augmented Spatiotemporal Reasoning for Streamlining Video Question Answering Task
3. Scalable Video-to-Dataset Generation for Cross-Platform Mobile Agents
4. VideoChat-A1: Thinking with Long Videos by Chain-of-Shot Reasoning
5. Watch and Learn: Learning to Use Computers from Online Videos
6. Computer-Use Agents as Judges for Generative User Interface
7. AgentTrek: Agent Trajectory Synthesis via Guiding Replay with Web Tutorials
8. AMEX: Android Multi-annotation Expo Dataset for Mobile GUI Agents
