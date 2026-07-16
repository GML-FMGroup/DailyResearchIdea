# Sparse Latent UI Tracker: Modeling Dynamic State with Identifiable Sparse Transitions

## Motivation

Existing GUI agents like KnowAct-GUIClaw treat each screenshot as a fully observable state, ignoring the fact that UI state evolves dynamically through actions; after an element is clicked, its visual appearance may change or disappear, yet the agent's next decision is based on a single snapshot. This structural assumption—that one screenshot captures all necessary state—fails when temporal dependencies matter, such as tracking a dropdown menu that persists across multiple steps or remembering a previously entered form field. The root cause is that current architectures lack an explicit state model that accounts for action-induced changes over time.

## Key Insight

The sparsity of UI changes—only a few elements are modified per action—implies that latent state updates can be modeled as a sparse additive function, enabling state tracking without full visual re-observation.

## Method

(A) **What it is**: Sparse Latent UI Tracker (SLUT) learns a latent state representation that evolves via a deterministic transition function with L1-regularized sparse updates. Its input is a sequence of screenshots and actions; its output is a compact latent state that can be decoded for action prediction. (B) **How it works** (pseudocode):

```python
# Encoder: f_enc (ViT + MLP) projects screenshot to latent vector z (dim=256)
# Transition: f_trans (MLP + hard thresholding) outputs sparse delta_z conditioned on action embedding
# Identity predictor: h (small Transformer) takes a 2-frame screenshot sequence and predicts z
# Decoder: f_dec (MLP + deconv) reconstructs screenshot for auxiliary loss

# Training loop
for each batch of (s_t, a_t, s_{t+1}):
    z_t = f_enc(s_t)
    # action embedding: linear projection of one-hot action a_t
    delta_z = f_trans(z_t, a_t)
    # enforce sparsity: apply element-wise hard threshold (threshold=0.1) to delta_z
    delta_z = delta_z * (abs(delta_z) > 0.1).float()
    z_{t+1} = z_t + delta_z
    
    # Reconstruction loss (optional, for visual grounding)
    loss_rec = MSE(f_dec(z_{t+1}), s_{t+1})
    
    # Identifiability loss: force z_{t+1} to be predictable from previous frames
    # h: takes [s_t, s_{t+1}] (concatenated) and outputs predicted z_{t+1}
    z_pred = h(cat(s_t, s_{t+1}))
    loss_id = MSE(z_pred, z_{t+1})
    
    # Total loss: loss_total = loss_rec + lambda * loss_id (lambda=0.5)
    # plus L1 penalty on delta_z before thresholding (l1_weight=0.01)
    loss_total += l1_weight * delta_z.abs().mean()
```

(C) **Why this design**: We chose hard thresholding over learned gating (e.g., Gumbel-Softmax) because it is deterministic and prevents the model from learning to use dense updates when sparsity is genuinely present, accepting the risk that a fixed threshold may be suboptimal for some UI patterns. We chose a predictive identifiability loss (MSE between transition-based and prediction-based states) rather than a contrastive objective because MSE directly minimizes representation distance, ensuring the latent space is metric-smooth; the trade-off is that with very large state spaces, MSE may be sensitive to scale differences. We used a two-frame input for the identity predictor instead of longer sequences because UI dynamics are Markovian in practice—a single previous frame suffices to resolve most ambiguities—and longer sequences would increase computational cost. The reconstruction loss serves as a visual regularizer but is not strictly necessary; we include it to maintain pixel-level correspondence, accepting that excellent reconstruction does not guarantee downstream task performance. (D) **Why it measures what we claim**: The sparse update delta_z measures which latent dimensions correspond to UI elements changed by the action, because the L1 penalty forces the model to concentrate change information into as few dimensions as possible; this assumption fails when a single action triggers a global change (e.g., page reload), in which case delta_z becomes dense and the measure reflects a breakdown of the sparsity prior. The identifiability loss (MSE between z_{t+1} from transition and from prediction) measures how well the latent state can be recovered from visual history alone, because the predictor h must learn to invert the transition dynamics using only screenshots; this assumption fails when the rendering engine introduces stochastic delays (e.g., animation) that make visual history non-deterministic, in which case the loss reflects aleatoric uncertainty rather than identifiability. Together, these computational quantities operationalize the concepts of sparse causal state and visual identifiability, providing a principled proxy for robust tracking under partial observability.

## Contribution

(1) A sparse latent state tracker (SLUT) with a deterministic transition function and L1-regularized updates, enabling efficient state modeling for GUI agent tasks. (2) An identifiability constraint that forces the latent representation to be recoverable from a short screenshot sequence, bridging temporal dynamics and visual input. (3) A conceptual framework for evaluating partial observability in GUI agents, with implications for designing agents that do not assume full observability.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | OSWorld (desktop GUI tasks) | Diverse tasks with partial observability |
| Primary metric | Task success rate | Directly measures agent effectiveness |
| Baseline 1 | KnowAct-GUIClaw | State-of-the-art memory-based agent |
| Baseline 2 | LearnAct (vanilla prompt) | No sparse latent tracking |
| Baseline 3 | Dense latent SLUT (no L1) | Tests sparsity prior necessity |
| Ablation-of-ours | SLUT w/o ID loss | Tests identifiability necessity |

### Why this setup validates the claim

This combination forms a falsifiable test of our central claim that sparse latent transitions with identifiability improve GUI agent robustness. The primary metric (task success on OSWorld) directly measures the end goal. Comparing against a dense latent variant isolates the benefit of sparsity, while comparing against SLUT without identifiability loss isolates the benefit of state predictability. State-of-the-art baselines like KnowAct and LearnAct provide real-world benchmarks; if SLUT outperforms them on tasks requiring fine-grained tracking but not on simple tasks, the sparsity and identifiability claims are supported. The ablation clarifies the necessity of each component.

### Expected outcome and causal chain

**vs. KnowAct-GUIClaw** — On a task where a UI element changes subtly (e.g., a checkbox becomes checked without visible text change), KnowAct relies on explicit memory of past states, which may be absent or stale, causing it to miss the change and repeat actions. Our SLUT directly encodes visual changes into sparse latent deltas, tracking the state transition without external memory. Thus we expect SLUT to show a noticeable gap (e.g., 15-20% higher success) on tasks with frequent but local UI changes, but near parity on tasks where memory suffices.

**vs. LearnAct (vanilla prompt)** — On a long-horizon task like multi-step form filling with partial observability (e.g., fields appear after previous input), LearnAct, lacking state tracking, may misinterpret the current screen and repeat or skip steps due to confusion about what has been done. SLUT maintains a latent state that evolves sparsely with actions, enabling it to track progress and context. Expected outcome: SLUT achieves >25% higher success on long-horizon tasks, while similar on short tasks.

**vs. Dense latent SLUT (no L1)** — On a task where a single action changes only one UI element (e.g., clicking a button to toggle visibility of a small widget), the dense latent update spreads change across many dimensions, making the state less interpretable and potentially causing interference with future predictions. Our sparse update concentrates change into few dimensions. We expect SLUT to outperform by ~10% on tasks with isolated changes, but perform similarly on tasks with global changes (e.g., page reload) where dense updates are appropriate.

### What would falsify this idea

If SLUT's performance gain over dense latent is uniform across all task subsets (both isolated and global changes), or if removing the identifiability loss does not degrade performance on tasks requiring visual history recovery, then the central claims about sparsity and identifiability are not supported.

## References

1. KnowAct-GUIClaw: Know Deeply, Act Perfectly, Personal GUI Assistant with Self-Evolving Memory and Skill
2. LearnAct: Few-Shot Mobile GUI Agent with a Unified Demonstration Benchmark
3. MAI-UI Technical Report: Real-World Centric Foundation GUI Agents
