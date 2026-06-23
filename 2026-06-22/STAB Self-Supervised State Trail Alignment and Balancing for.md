# STAB: Self-Supervised State Trail Alignment and Balancing for Noisy Multimodal Embodied Agents

## Motivation

Existing embodied agent memory frameworks such as WorldLines build state trails from clean, structured data (e.g., simulated dialogues and actions), but real-world agents face noisy, asynchronous, or missing modalities. Without a mechanism to calibrate per-modality noise online, these methods fail to maintain coherent state representations. The root cause is that they assume a fixed, reliable input stream; temporal and cross-modal inconsistencies are treated as data rather than as signals for noise estimation.

## Key Insight

Temporal and cross-modal consistency provides a self-supervised signal to estimate per-modality noise levels because the correct information is redundantly distributed across modalities and time, so deviations from consensus identify noise without requiring ground truth.

## Method

**STAB (State Trail Alignment and Balancing)**

### (A) What it is
STAB is an online memory framework that takes a continuous stream of noisy, unstructured multimodal observations (video frames, audio clips, text transcripts) and outputs a sequence of state trails (entity- and action-level states). It calibrates per-modality noise levels in real time using cross-modal similarity statistics and temporal consistency checks, then fuses modalities weighted by estimated reliability.

### (B) How it works
```pseudocode
Input: Multimodal stream O = {(v_t, a_t, txt_t)} for t = 1..T
Output: State trail S = {s_1, s_2, ..., s_K}

Parameters: window size W = 10, buffer size B = 100, similarity threshold τ = 0.7, fusion decay α = 0.1

Initialize: noise_estimates = {0.0 for each modality}, buffer = empty list, state_trail = empty

for each time step t:
    # 1. Encode each modality into feature vectors
    z_v = CLIP_visual(v_t)
    z_a = Whisper_audio(a_t)
    z_t = BERT_text(txt_t)

    # 2. Compute pairwise cosine similarities
    s_va = cos(z_v, z_a)
    s_vt = cos(z_v, z_t)
    s_at = cos(z_a, z_t)
    avg_sim = (s_va + s_vt + s_at) / 3

    # 3. Update noise calibration buffer
    buffer.append( (t, avg_sim, z_v, z_a, z_t) )
    if len(buffer) > B: pop oldest

    # 4. Estimate noise level per modality using GMM on similarities
    #    For each modality m, collect all similarity values involving m from buffer
    #    Fit a 2-component GMM on the set; the component with lower mean is "noisy"
    λ_m = probability that the sample belongs to the low-similarity component
    #    Update running estimate with EMA: noise_estimates[m] = α * λ_m + (1-α) * noise_estimates[m]

    # 5. Fuse modalities with inverse-noise weighting
    weights = [1 - noise_estimates[m] for m in ('v','a','t')]
    f_t = sum( weights[i] * z_i ) / sum(weights)

    # 6. Update state trail
    if state_trail is empty:
        add new state s_1 = f_t
    else:
        last_s = state_trail[-1]
        if cos(f_t, last_s) < τ:
            add new state s_{k+1} = f_t
        else:
            # merge: running average with momentum
            state_trail[-1] = (1-α) * last_s + α * f_t

    # 7. Periodic temporal consistency calibration (every W steps)
    if t % W == 0:
        for each pair of states (s_i, s_j) with |i-j| ≤ 2:
            if cos(s_i, s_j) < τ:
                # Indicates possible misalignment; increase noise estimates for modalities that had low similarity in that window
                # Aggregate per-modality similarities during those steps; if below median, increase their λ by 0.1
                # This step refines noise calibration based on temporally nearby states
    
```
### (C) Why this design
We chose a similarity-based fusion with inverse-noise weighting over learned attention because it is transparent, computationally cheap, and directly links cross-modal consistency to reliability. The trade-off is that it assumes linear combination of features is sufficient, which may miss complex cross-modal interactions, but in online settings this simplicity enables real-time operation. We use a 2-component GMM on historical similarities rather than a fixed threshold to adaptively separate noise from signal per modality; the cost is that the buffer size introduces a latency in noise estimation and requires tuning. For state update, we use a cosine threshold τ to decide when to create a new state rather than a learned detection head, because the threshold is interpretable and avoids training data requirements; the consequence is that τ may need to be tuned per environment, and rapid state changes can cause oversegmentation. The temporal consistency check (step 7) is designed as a periodic recalibration rather than a continuous process to avoid computational overhead; this means noise estimates may lag behind sudden environment changes, but the benefit is stable operation across long streams. We chose EMA for noise estimates and state averaging to provide smooth adaptation; the trade-off is that the method may respond slowly to abrupt noise increases, but this is acceptable for streaming settings where most noise is stationary or slowly varying.

### (D) Why it measures what we claim
The cross-modal similarity values (s_va, s_vt, s_at) **measure** the consistency between modalities, which is a proxy for per-modality noise because under the assumption that correct information is redundantly present across at least two modalities, low similarity indicates a modality is corrupted by noise. This assumption fails when modalities are inherently asynchronous (e.g., video lags behind audio), in which case the metric reflects temporal misalignment rather than intrinsic noise, and the method may underestimate true noise in the lagging modality. The noise estimates λ_m, derived from GMM, **measure** the probability that a modality's representation is unreliable; this operationalizes the concept of 'noise calibration' because the GMM separates the distribution of similarities into a 'coherent' and 'incoherent' component. The assumption that GMM captures a bimodal structure holds when noise is sporadic; it fails when noise is pervasive (all similarities are low), in which case λ_m becomes unreliable (both components collapse). The fused representation f_t and the state update rule **measure** the robustness of the state trail by weighting according to estimated reliability; the assumption is that state transitions in the clean latent space are smooth, so a cosine change above τ flags a genuine new state. This assumption fails when state changes are gradual, causing many spurious new states that degrade trail coherence, but the merging step mitigates this. The temporal consistency calibration (step 7) **measures** the accuracy of the state trail by checking that nearby states in time remain similar; if they are not, it signals that noise estimates need tightening. This relies on the assumption that the true underlying state changes slowly relative to the observation frequency; it fails in highly dynamic environments where rapid true state changes are indistinguishable from noise-induced instability.

## Contribution

(1) A self-supervised, online framework (STAB) that calibrates per-modality noise levels using cross-modal and temporal consistency, enabling robust state trail construction from noisy multimodal streams without clean pre-structured data. (2) A design principle that noise can be estimated as the inverse of average cross-modal similarity, validated by a GMM-based separation of coherent and incoherent observations. (3) A temporal consistency check as a fine-tuning mechanism for noise estimates, demonstrating that state trail stability can be used as a self-supervisory signal.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
|------|--------|-----------|
| Dataset | WorldLines Suite | Multimodal long-horizon embodied traces |
| Primary metric | State Recognition Accuracy | Measures state trail correctness |
| Baseline 1 | Episodic Memory | Stores observations; no online fusion |
| Baseline 2 | LSTM | Learns temporal dynamics; ignores multimodal reliability |
| Baseline 3 | Full-Context LLM | Processes entire history; expensive and noisy |
| Ablation of ours | STAB w/o temporal consistency | Tests value of periodic recalibration |

### Why this setup validates the claim
WorldLines provides multimodal streams with realistic noise (e.g., modality asynchrony, dropouts), directly testing STAB's core claim: online noise calibration via cross-modal similarity yields robust state trails. The baselines represent distinct failure modes: Episodic Memory cannot adapt to noise, LSTM overfits to irregular temporal patterns, and Full-Context LLM is overwhelmed by redundant yet conflicting signals. State Recognition Accuracy is the direct measure of state trail fidelity, so a clear advantage on this metric—especially on noisy subsets—would confirm the method's effectiveness. The ablation isolates the temporal consistency calibration step, probing its contribution to noise adaptivity.

### Expected outcome and causal chain

**vs. Episodic Memory** — On a case where audio is corrupted but video is clear, Episodic Memory retrieves states based on raw feature similarity, often pulling in incorrect audio-dominated recollections because it treats all modalities equally. Our method instead estimates audio as noisy (low cross-modal similarity) and downweights it, relying more on video, so the fused state remains correct. We expect a noticeable gap (e.g., 15–25% higher accuracy) on multimodal sequences where modality noise varies, with parity on clean sequences.

**vs. LSTM** — On a sequence with periodic visual dropout (e.g., occlusion), LSTM learns a temporal pattern that includes the dropout intervals, but during testing it may hallucinate visual states during dropouts because its hidden state is contaminated by previous noisy inputs. STAB detects the drop in visual-text coherence for those frames, increases visual noise estimate, and fuses primarily audio and text, maintaining a stable state trail. We expect a measurable advantage (e.g., 10–20% higher accuracy) on segments with intermittent sensor failures, with similar performance on smoothly varying data.

**vs. Full-Context LLM** — On a long interaction where video and audio are slightly misaligned (e.g., audio leads by 0.5s), Full-Context LLM attends to all tokens and may produce contradictory state descriptions because it tries to reconcile temporally offset information. STAB's state update threshold and merging prevent spurious states from minor asynchrony, and its noise calibration does not treat asynchrony as noise (similarities remain high across modalities). We expect STAB to maintain coherent states while Full-Context LLM may produce fragmented or inconsistent outputs, leading to a gap of at least 10% on long traces with mild misalignment.

### What would falsify this idea
If STAB's performance is no better on noisy multimodal subsets than on clean ones, or if the ablation (STAB w/o temporal consistency) performs equally well, then the causal link between cross-modal noise calibration and state trail accuracy is unsupported.

## References

1. WorldLines: Benchmarking and Modeling Long-Horizon Stateful Embodied Agents
2. Memory OS of AI Agent
3. Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory
4. Embodied AI Agents: Modeling the World
5. Generative Dense Retrieval: Memory Can Be a Burden
6. LLM-based Medical Assistant Personalization with Short- and Long-Term Memory Coordination
7. Explore Theory of Mind: Program-guided adversarial data generation for theory of mind reasoning
8. ELLMA-T: an Embodied LLM-agent for Supporting English Language Learning in Social VR
