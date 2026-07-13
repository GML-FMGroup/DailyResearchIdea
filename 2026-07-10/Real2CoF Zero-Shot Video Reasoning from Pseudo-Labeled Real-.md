# Real2CoF: Zero-Shot Video Reasoning from Pseudo-Labeled Real-World Captions

## Motivation

OpenCoF relies on synthetic mazes (OpenCoF-17K) which fail to generalize to real-world video reasoning due to distribution shift and limited visual complexity. The root cause is that procedural generation cannot capture the full variety of causal and temporal structures in natural videos, making zero-shot transfer impossible. By incorporating real-world videos with pseudo-reasoning labels from a captioning model, we relax the synthetic-only assumption and enable direct exposure to natural reasoning patterns.

## Key Insight

Because real-world videos share the same underlying causal and temporal reasoning principles as synthetic ones, pseudo-labels from a video captioning model can bridge the domain gap by providing dense supervisory signals that encode the reasoning chain, allowing the model to learn invariant reasoning patterns across both domains.

## Method

Real2CoF fine-tunes a video generation model (Wan2.2-I2V-A14B) on real-world videos with pseudo-reasoning labels derived from a pre-trained video captioning model and an LLM, inserting visual and textual reasoning tokens to enable zero-shot reasoning transfer.

**How it works**:
```pseudocode
Input: Real-world videos V = {v_i}, captioning model C (BLIP-2), reasoning chain extractor L (GPT-4), video generation model G (Wan2.2-I2V-A14B)

Procedure:
0. (Optional) For feasibility, initial experiments use Wan2.2-I2V-1B (1B params) instead of 14B, with same architecture but fewer parameters.
1. For each video v_i in V:
   a. Sample key frames f_1,...,f_T (uniformly spaced, T=16)
   b. Generate caption c_i = C(f_1,...,f_T)  // BLIP-2 full video description
   c. Extract reasoning chain r_i = [e_1,...,e_K] = L("Given caption: {c_i}. Produce a sequence of events (cause-effect and temporal order) in the video. Each event: a short phrase.")  // e.g., ["person picks up cup", "cup moves left", "person pours water"]
   d. For each event e_j, create textual reasoning token t_j via embedding lookup (dim=2048)
   e. For each frame f_t, create visual reasoning tokens v_t (as in OpenCoF: spatial feature maps from patch embeddings; 256 tokens, each 1024-dim)
   f. Verification: Filter out videos where (i) BLIP-2 confidence (softmax probability of generated caption) < 0.7, or (ii) LLM chain fails consistency check (events not in chronological order by timestamps extracted from caption, or semantic similarity between consecutive events < 0.5 via Sentence-BERT). Additionally, from 10% of remaining videos, randomly select 512 as a calibration set for human annotation of reasoning chains. These human annotations are used only to estimate label noise (e.g., ROUGE-L between pseudo and human chains) and are not used for training.
2. Build training pairs: (input = f_1...f_{t-1} + all t_j + all v_t, target = f_t) for t=2..T
3. Fine-tune G with combined loss: L = L_mse(G(input), f_t) + λ * L_aux, where L_aux = -cosine(t_output, e_embedding) for each textual token, λ=0.1
   Hyperparameters: learning rate=1e-5, batch size=8 (adjust to fit GPU memory; for 1B model batch size=32), epochs=10, optimizer=AdamW, weight decay=0.01, gradient clipping norm=1.0.
4. Return fine-tuned model G'
```

**Why this design**:
We chose to use a video captioning model (BLIP-2) over manual annotation to scale to thousands of videos, accepting label noise that may degrade reasoning accuracy but enabling coverage of diverse real-world scenarios. We extract reasoning chains via an LLM (GPT-4) rather than using the caption directly as a single token, because structured event sequences align better with the token-level reasoning tokens in Wan2.2, trading additional computational cost for finer-grained supervision. We retain both visual and textual reasoning tokens from OpenCoF instead of only textual tokens, because visual tokens capture low-level spatial cues (e.g., object positions) that complement semantic priors, increasing model capacity at the risk of learning redundant representations. We employ an auxiliary cosine similarity loss on textual token outputs rather than relying solely on the frame-generation MSE, because this directly incentivizes the tokens to encode the reasoning chain, but introduces a trade-off where the token embeddings may drift from the language space if not properly regularized. To mitigate pseudo-label noise, we add a verification step that filters low-confidence captions and inconsistent chains, and we hold out a calibration set of 512 videos with human-annotated reasoning chains to quantify label noise.

**Why it measures what we claim**:
The pseudo-reasoning chain extracted from the caption serves as a high-level semantic prior for causal and temporal structure. The auxiliary loss L_aux measures the alignment between the model's textual token outputs and the event embeddings; this computational quantity operationalizes the concept of 'reasoning chain fidelity' under the assumption that the LLM-generated chain accurately captures the true causal structure of the video. This assumption fails when the captioning model produces a false or incomplete description (e.g., missing a critical interaction), in which case L_aux reflects alignment with a flawed prior rather than true reasoning. The visual reasoning tokens from key frames measure low-level spatial cues relevant to reasoning, assuming relevant objects and motions are visually salient; this fails when reasoning depends on invisible factors (e.g., forces), in which case tokens encode spurious correlations. The combined training on real-world videos with these pseudo-labels measures the model's ability to transfer to real-world reasoning tasks by exposing it to natural visual variation and causal structures not present in synthetic data, under the assumption that the pseudo-labels and the generation loss together induce a reasoning-consistent latent space. This chain closes the gap from method to motivation: the pseudo-labels and joint training provide the missing real-world experience, enabling zero-shot transfer.

## Contribution

(1) A method to automatically generate pseudo-reasoning labels for real-world videos by combining a video captioning model with an LLM for chain extraction, enabling scalable annotation of causal/temporal structure. (2) A fine-tuning procedure that integrates these labels as visual and textual reasoning tokens into a video generation model, reducing domain gap for zero-shot video reasoning. (3) Demonstration that synthetic-only training is not necessary; pseudo-labeled real-world data can serve as a substitute, opening the door to reasoning over diverse natural videos.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
|------|--------|-----------|
| Dataset | Ego4D + Something-Something v2 | Diverse real-world causal structures |
| Primary metric | Accuracy on VCR+CATER | Measures causal and temporal reasoning |
| Additional metric | ROUGE-L agreement with human-annotated reasoning chains on calibration set (512 videos) | Quantifies pseudo-label noise |
| Additional metric | Human evaluation of generated reasoning chains (100 videos, 3 annotators, Cohen's κ > 0.7) | Assesses practical utility of reasoning chains |
| Baseline | Wan2.2-I2V-A14B (vanilla) | No reasoning tokens or fine-tuning |
| Baseline | Real2CoF without auxiliary loss | Isolates effect of reasoning token loss |
| Baseline | Real2CoF without visual tokens | Isolates role of visual reasoning tokens |
| Baseline | Real2CoF without LLM chains | Tests need for structured event sequences |
| Baseline | Real2CoF with human-annotated chains (on calibration set only, 512 examples) | Upper bound on reasoning quality |
| Ablation | Real2CoF without verification step | Tests impact of noisy pseudo-labels |

### Why this setup validates the claim
This combination of dataset, baselines, and metrics forms a falsifiable test of the central claim that inserting visual and textual reasoning tokens with pseudo-reasoning supervision enables zero-shot reasoning transfer. The chosen dataset includes real-world videos with diverse causal structures (Ego4D: egocentric actions; Something-Something: object interactions), matching our method's target domain. The baseline Wan2.2-I2V-A14B tests whether any fine-tuning helps; the ablation without auxiliary loss tests whether the reasoning token loss is necessary beyond standard video generation fine-tuning; the ablation without visual tokens tests whether spatial cues add value; and the ablation without LLM chains tests whether structured event sequences outperform a single caption token. The primary metric (accuracy on VCR+CATER) directly measures reasoning capabilities where our method should excel if the reasoning tokens capture causal/temporal structure. The additional metrics (ROUGE-L on calibration set, human evaluation) explicitly validate the load-bearing assumption about pseudo-label quality. The baseline with human-annotated chains provides an upper bound. If our method outperforms all baselines on these benchmarks and the pseudo-labels show high agreement with human annotations, the claim is supported; if not, the specific failure pattern will isolate which component is flawed.

### Expected outcome and causal chain

**vs. Wan2.2-I2V-A14B (vanilla)** — On a case like "person pours water into cup," the vanilla model generates a frame sequence that maintains visual consistency but fails to respect causal order (e.g., water appears in cup before pouring starts). This happens because the model lacks explicit reasoning tokens to enforce temporal causality. Our method encodes the chain "pick cup, tilt bottle, water flows" into textual tokens and visual tokens marking object positions, so the generation loss and auxiliary loss jointly align the latent space with the event order. We expect a noticeable accuracy gap on temporal ordering tasks (e.g., CATER) but parity on simple appearance-based tasks.

**vs. Real2CoF without auxiliary loss** — On a case where the pseudo-reasoning chain is accurate (e.g., "hand opens drawer, drawer slides out"), the model trained only with MSE can reproduce the visual flow but may skip or reorder events because it has no direct signal to align token embeddings with the event sequence. Our method with the auxiliary loss explicitly pushes textual token outputs toward event embeddings, so the model learns to attend to the reasoning chain. We expect a clear advantage on tasks requiring precise step ordering (e.g., Something-Something temporal order) but similar performance on static recognition tasks.

**vs. Real2CoF without visual tokens** — On a case like "ball rolls behind box and reappears," the textual-only model may reason about occlusion but lacks spatial grounding, leading to imprecise object positions in generated frames. Our method with visual tokens encodes spatial feature maps from key frames, providing low-level cues for object locations. We expect better performance on spatial reasoning tasks (e.g., CATER object tracking) but comparable results on purely semantic reasoning.

**vs. Real2CoF without verification step** — On a case where the captioning model produces a low-confidence or inconsistent chain (e.g., missing a critical action), the model without verification may learn from faulty pseudo-labels, degrading reasoning quality. Our method filters those out, so we expect higher accuracy on tasks sensitive to causal structure, especially for videos where the captioning model is uncertain.

**vs. Real2CoF with human-annotated chains (upper bound)** — On the calibration set, we expect the model with human chains to achieve near-perfect reasoning accuracy, while our model with pseudo-labels will show some gap. The size of this gap, measured by accuracy and ROUGE-L, quantifies the cost of pseudo-label noise.

### What would falsify this idea
If our method performs no better than the vanilla Wan2.2-I2V-A14B on all subsets, or if the gain is uniform across tasks instead of concentrated on reasoning-heavy subsets where the baseline fails, then the central claim that reasoning tokens drive the improvement would be falsified. Additionally, if the ROUGE-L between pseudo and human chains is below 0.3 (low), the load-bearing assumption is violated, indicating the pseudo-labels are too noisy for effective training.

## References

1. OpenCoF: Learning to Reason Through Video Generation
