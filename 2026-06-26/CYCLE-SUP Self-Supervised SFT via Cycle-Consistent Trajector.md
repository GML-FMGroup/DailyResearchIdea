# CYCLE-SUP: Self-Supervised SFT via Cycle-Consistent Trajectory Imputation for Tool-Use RL

## Motivation

Interleaving supervised fine-tuning (SFT) with reinforcement learning (RL) prevents policy collapse in multi-step tool-use tasks, but its effectiveness relies on high-quality human demonstrations for the SFT phases. This dependence on human annotation creates a scalability bottleneck, as generating diverse, in-distribution demonstrations for every tool-use scenario is impractical. 'Why Multi-Step Tool-Use Reinforcement Learning Collapses and How Supervisory Signals Help' identifies the stability benefits of interleaved SFT but relies on human-curated supervision, whose absence leads to performance degradation under out-of-distribution evaluations. The core problem is that without human demonstrations, the SFT phase lacks a supervisory signal that remains on the policy's own trajectory distribution, causing the RL updates to drift away from effective exploration.

## Key Insight

Cycle-consistency between forward and backward predictions of intermediate tool-use steps ensures that self-imputed demonstrations lie on the policy's manifold, because a step that is invariant under both inference directions must be consistent with the policy's implicit transition dynamics.

## Method

### (A) What it is
CYCLE-SUP is a self-supervised framework that generates SFT demonstrations by imputing masked intermediate steps in tool-use trajectories using a cycle-consistency constraint, replacing the need for human annotations in interleaved SFT+RL training.

### (B) How it works
```python
# Pseudocode for one training iteration
# Input: policy π (LLM) on tool-use tasks
# Hyperparameters: mask_rate = 0.2, λ_cyc = 1.0, λ_l2 = 0.01
for each rollouts from π:
    τ = [(s1, a1), ..., (sT, aT)]  # state-action pairs
    # Mask one random action (tool call) per trajectory
    t_mask = random.randint(1, T-1)
    τ_masked = remove a_t_mask from τ
    
    # Forward imputation: predict a_t_mask from prefix
    prefix = τ_masked[:t_mask]
    fwd_pred = ImputationNet(prefix)  # lightweight MLP
    
    # Backward imputation: predict a_t_mask from suffix
    suffix = τ_masked[t_mask:]
    bwd_pred = ImputationNet(reversed(suffix))  # predict previous action
    
    # Cycle-consistency loss
    L_cyc = MSE(fwd_pred, bwd_pred)
    L_l2 = λ_l2 * (||fwd_pred||^2 + ||bwd_pred||^2)
    update ImputationNet via SGD on L = λ_cyc * L_cyc + L_l2
    
    # Generate demonstration from partial trajectory using fwd_pred
    τ_demo = insert fwd_pred at position t_mask into τ_masked
    
# Interleave: SFT on τ_demo (cross-entropy) and RL on original τ (PPO)
update π with combined objective: L_total = L_SFT + L_RL
```

### (C) Why this design
We chose cycle-consistency over generative adversarial training (e.g., GAIL) because it requires no discriminator and directly aligns forward/backward predictions, avoiding mode collapse and additional adversarial optimization. The trade-off is that cycle-consistency assumes invertibility of tool-use steps; for irreversible actions (e.g., deleting a file), the backward prediction may be ill-defined, but in practice tool-use actions often have deterministic effects and context dependencies that make the forward-backward mapping nearly bijective within the policy's experience. We mask only one random step per trajectory rather than multiple steps to keep the imputation tractable—multiple masks would require a more complex autoregressive model—accepting that partial demonstrations may yield less diverse SFT data but with higher quality per step. We train a separate lightweight imputation network (a 3-layer MLP with 256 hidden units) instead of fine-tuning the policy itself for imputation, to avoid interfering with the RL updates that rely on the policy's own exploration; the cost is additional parameters (~1M) but preserves the policy's RL stability. Unlike standard cycle-consistency methods (e.g., CycleGAN) that operate across domains, our setting is single-domain, and we exploit the temporal structure of tool-use trajectories—this structural difference avoids collapsing to a simple reconstruction baseline.

### (D) Why it measures what we claim
The cycle-consistency loss `MSE(fwd_pred, bwd_pred)` measures the degree to which the imputed step is invariant under both forward and backward inference directions; this invariance proxies for the concept of "in-distribution" because if the imputed action is consistent with the policy's learned forward transition (given prefix) and backward transition (given suffix), it must lie on the policy's manifold of plausible actions. This equivalence holds under the assumption that the policy's transition dynamics are approximately deterministic and that the context window captures all relevant information; when this assumption fails—for example, when the policy exhibits multimodal behavior (multiple equally valid next actions given the same prefix)—the MSE may average over modes, yielding an imputed action that is not in the policy's actual distribution (e.g., a mean of two distinct tool calls). In that failure case, the MSE reflects a degenerate centroid, not a valid demonstration. To mitigate, we ensure that the tool-use actions are often discrete and context-dependent, reducing mode multiplicity. Additionally, we use the forward prediction (not the cycle-consensus) as the demonstration, which is biased toward the most likely mode, providing partial robustness. The `L_l2` regularizer further prevents the imputation from producing extreme values that are unlikely under the policy.

## Contribution

(1) A self-supervised framework (CYCLE-SUP) that generates SFT demonstrations for tool-use RL by imputing masked steps via cycle-consistency, eliminating human annotation for interleaved training. (2) A demonstration that cycle-consistency within a single trajectory distribution produces in-distribution supervision, validated by comparing performance against human-demonstrated SFT in interleaved RL. (3) An analysis of the conditions under which cycle-consistency fails (multimodal actions) and a mitigation via forward-prediction selection.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|-----------------------|
| Dataset | ToolBench (10 tasks) | Diverse tool-use scenarios with ground-truth. |
| Primary metric | Task success rate | Directly measures task completion. |
| Baseline 1 | RL-only | Tests necessity of SFT demonstrations. |
| Baseline 2 | Human SFT+RL | Upper bound with expert annotations. |
| Baseline 3 | Random imputation SFT+RL | Ablates cycle-consistency structure. |
| Ablation-of-ours | CYCLE-SUP w/o cycle loss | Isolates cycle-consistency contribution. |

### Why this setup validates the claim
This combination forms a falsifiable test of whether cycle-consistency imputation generates useful SFT demonstrations. ToolBench provides realistic tool-use tasks with known optimal trajectories. RL-only isolates the effect of SFT; human SFT+RL gives an upper bound; random imputation tests if arbitrary imputation suffices. Task success rate directly reflects the quality of learned policies. The ablation (w/o cycle loss) pinpoints whether cycle consistency itself drives any improvement. If our method outperforms RL-only and random imputation, and approaches human SFT+RL, the claim is supported. If not, the mechanism is invalid.

### Expected outcome and causal chain

**vs. RL-only** — On a case where the policy must call multiple tools in sequence, RL-only produces action collapse (e.g., repeating a single tool call) because sparse reward and insufficient exploration cause mode collapse. Our method instead generates plausible intermediate actions via cycle-consistency imputation, providing dense supervisory signals, so we expect a noticeable gap (e.g., >20% success rate) on multi-step tasks but parity on single-step tasks.

**vs. Human SFT+RL** — On a case involving nuanced tool parameters (e.g., formatting a search query), human SFT+RL achieves high accuracy because expert demonstrations cover exact syntax. Our method imputes actions from policy rollouts, which may be slightly off-distribution, so we expect a small gap (e.g., 5-10% lower success rate) on such precision-critical steps, but parity on tasks with deterministic tool effects.

**vs. Random imputation SFT+RL** — On a case where the tool-use trajectory has context-dependent steps (e.g., using a retrieved result to form a subsequent query), random imputation often produces invalid actions (e.g., mismatched arguments) because it ignores temporal dependencies. Our method uses cycle consistency to enforce temporal coherence, so we expect a large gap (e.g., >15% success rate) on context-sensitive tasks, but minimal difference on simple linear chains.

### What would falsify this idea
If CYCLE-SUP does not significantly outperform random imputation on multi-step or context-sensitive tasks, or if its performance is uniformly worse than human SFT+RL across all subsets, then cycle-consistency does not capture the intended trajectory structure and the central claim is wrong.

## References

1. Why Multi-Step Tool-Use Reinforcement Learning Collapses and How Supervisory Signals Fix It
2. Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning
3. SRFT: A Single-Stage Method with Supervised and Reinforcement Fine-Tuning for Reasoning
4. Boosting MLLM Reasoning with Text-Debiased Hint-GRPO
5. SFT Memorizes, RL Generalizes: A Comparative Study of Foundation Model Post-training
6. Grounding by Trying: LLMs with Reinforcement Learning-Enhanced Retrieval
7. Training Language Models to Self-Correct via Reinforcement Learning
8. Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution
