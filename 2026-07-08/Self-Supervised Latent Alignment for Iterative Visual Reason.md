# Self-Supervised Latent Alignment for Iterative Visual Reasoning over Rendered Graphics

## Motivation

MentalThink relies solely on final-answer reward, which is sparse and cannot provide feedback on the quality of each reasoning step, especially when the final answer is unavailable. This structural limitation prevents fine-grained credit assignment across multiple turns and limits applicability to tasks without explicit ground-truth answers.

## Key Insight

The deterministic rendering engine provides a stationary environment where the prediction error between the model's latent representation of the next rendered image and the actual rendering gives a dense, self-supervised measure of reasoning progress without external annotations.

## Method

(A) **What it is**: SLAIR (Self-supervised Latent Alignment for Iterative Reasoning) is a multi-turn reinforcement learning method that trains a visual-language model to reason over SVG programs by augmenting the final-answer reward with a dense intermediate reward derived from the model's ability to predict the latent encoding of the next rendered image. The input is a visual question and an initial blank scene; the output is a final answer after iterative refinement.

(B) **How it works**:
1. At each turn t, the model receives the question and the rendered image I_{t-1} from the previous turn (or a blank at t=0). It generates an SVG program S_t.
2. After generating S_t, we extract the model's internal hidden state h_t (final hidden state of the language decoder after the last token of S_t).
3. A lightweight prediction head f_θ (two-layer MLP with 512 hidden units) maps h_t to a predicted latent vector z_pred_t = f_θ(h_t).
4. The SVG program S_t is rendered by a deterministic renderer (e.g., Cairo) to produce image I_t.
5. A frozen pretrained vision encoder (CLIP ViT-L/14) encodes I_t into a target latent vector z_t = E(I_t). E is frozen.
6. Self-supervised intermediate reward for turn t: r_t = -∥z_pred_t - z_t∥², clipped to [-1, 0].
7. After final turn T, model outputs answer. Reward: R_final = +1 if correct, else -0.1.
8. Total trajectory reward: R_total = R_final + Σ_{t=1}^{T-1} λ * r_t, λ=0.1.
9. Model trained with PPO using R_total, optimizing both f_θ and base model (via LoRA).

(C) **Why this design**: We chose a frozen pretrained image encoder over a learned one to provide a stable target representation, preventing the reward from decaying as the model overfits to its own predictions; this stability is critical for the self-supervised signal to be consistent across training iterations. Using L2 distance rather than cosine similarity ensures the reward magnitude correlates with absolute prediction quality, which is important for credit assignment in PPO where reward scale matters. The scaling factor λ=0.1 is set small to keep the intermediate reward as a regularizer rather than dominating the final answer reward, because the final correctness is the ultimate objective and we want to avoid distracting the model with high intermediate rewards that may not align with final success. We chose a two-layer MLP over a linear probe or deeper network because it provides sufficient expressiveness to map the hidden state to the visual feature space while being lightweight enough to train without overfitting on limited turns. The clipping of intermediate rewards to [-1,0] normalizes the magnitude across different tasks and prevents extreme outliers from destabilizing PPO.

(D) **Why it measures what we claim**: The negative L2 distance between predicted latent z_pred_t and actual latent z_t measures the model's ability to foresee the visual consequence of its generated SVG program. This operationalizes "reasoning progress" under the assumption that accurate prediction of the next rendered image implies the model has correctly inferred how its current program transforms the visual scene; this assumption fails when the prediction error is dominated by irrelevant visual details (e.g., texture or lighting) that are not semantically important for reasoning, in which case a low reward may penalize the model for failing to predict superficial features even though its reasoning is correct. To mitigate this, we use a latent space from a pretrained vision encoder that prioritizes semantic content over low-level textures, as CLIP's feature space is trained to align with language descriptions. However, if the reasoning task requires fine-grained spatial relationships that CLIP does not capture well, the reward may not fully reflect reasoning quality.

## Contribution

(1) A novel self-supervised intermediate reward for multi-turn visual reasoning over rendered graphics, computed from latent prediction error. (2) The finding that stable target representations from a frozen pretrained encoder enable dense reward signals without ground-truth step annotations. (3) A training framework that combines this reward with final-answer PPO to improve credit assignment and handle tasks without explicit answers.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|-----------------------|
| Dataset | VSI-Bench | Tests spatial reasoning in diverse scenes. |
| Primary metric | Answer accuracy | Directly measures reasoning success. |
| Baseline 1 | Vanilla MLLM | No intermediate representation, directly answers. |
| Baseline 2 | MLLM + CoT (textual) | Uses language chain-of-thought without vision grounding. |
| Baseline 3 | MLLM + SVG without RL | SVG generated in one shot, no iterative refinement. |
| Ablation | SLAIR w/o intermediate reward | Removes self-supervised signal to isolate its effect. |

### Why this setup validates the claim
This combination directly tests the central claim: that self-supervised prediction of next rendered image provides a useful dense reward for iterative visual reasoning. VSI-Bench requires understanding spatial relationships, which is exactly where predicting visual consequences should matter. The baselines isolate key sub-claims: Vanilla MLLM tests whether any intermediate representation is needed; CoT tests whether language-only reasoning suffices; one-shot SVG tests the value of iterative refinement. The ablation removes the intermediate reward to measure its incremental contribution above final correctness. Using accuracy as the metric captures the ultimate goal of correct reasoning, making the test falsifiable: if our method outperforms all baselines and the ablation, the claim is supported; if not, the mechanism fails.

### Expected outcome and causal chain

**vs. Vanilla MLLM** — On a case where the question involves relative positions of three objects, the baseline produces an incorrect answer because it lacks explicit spatial representation and may overfit to language priors. Our method iteratively generates SVG programs, and the intermediate reward encourages predicting the visual outcome of each step, leading to accurate spatial reasoning. We expect our method to significantly outperform on questions requiring multi-step spatial inference, but similar performance on simple perceptual tasks.

**vs. MLLM + CoT (textual)** — On a case where reasoning requires precise geometric transformations (e.g., rotation), CoT may produce plausible but geometrically invalid steps because it lacks grounding in actual visual changes. Our method, by predicting the rendered image after each SVG command, is forced to ensure geometric consistency. We expect a noticeable gap on geometric constraints, with our method achieving higher accuracy, while both perform similarly on semantic reasoning.

**vs. MLLM + SVG without RL** — On a case where the initial SVG guess contains a positional error, the one-shot baseline cannot correct it, leading to wrong final answer. Our method uses iterative refinement: the low intermediate reward from failing to predict the correct next image signals the error, and PPO updates the policy to adjust the program. We expect our method to show improved accuracy on multi-step correction tasks, with the gap widening as number of required iterations increases.

### What would falsify this idea
If the ablation (SLAIR w/o intermediate reward) achieves comparable performance to the full SLAIR, then the intermediate reward is not driving improvement. Additionally, if our method fails to outperform the one-shot SVG baseline on tasks requiring iterative correction, the core mechanism of self-supervised latent alignment does not provide useful guidance.

## References

1. MentalThink: Shaping Thoughts in Mental SVG World
2. Euclid's Gift: Enhancing Spatial Perception and Reasoning in Vision-Language Models via Geometric Surrogate Tasks
3. Learning to See Before Seeing: Demystifying LLM Visual Priors from Language Pre-training
4. SpatialLadder: Progressive Training for Spatial Reasoning in Vision-Language Models
5. Machine Mental Imagery: Empower Multimodal Reasoning with Latent Visual Tokens
6. Open Vision Reasoner: Transferring Linguistic Cognitive Behavior for Visual Reasoning
7. MedVisionLlama: Leveraging Pre-Trained Large Language Model Layers to Enhance Medical Image Segmentation
8. Aioli: A Unified Optimization Framework for Language Model Data Mixing
