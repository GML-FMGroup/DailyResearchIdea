# Dense Intrinsic Rewards via Conditional KL Divergence for Joint Perception-Reasoning Optimization

## Motivation

Existing methods like P2R decouple perception and reasoning but optimize perception only through sparse final-answer rewards, providing weak gradient signal for region selection. The root cause is that the reward for the perceiver is delayed until after reasoning, leading to high variance and misalignment between localization quality and final accuracy. This structural problem — lack of intermediate supervision for perception in joint optimization — is unresolved across multiple approaches.

## Key Insight

The KL divergence between the reasoning model's output distributions with and without a candidate region directly quantifies the causal contribution of that region to the reasoning outcome, providing a dense, principled supervision for perception without external annotations.

## Method

(A) **What it is**: Causal Perception Reward (CPR) is an end-to-end training framework that computes a dense intrinsic reward for the perception module as the KL divergence between the reasoning model's output distribution when a candidate region is included versus when it is masked. The perception module is trained via REINFORCE with a combined reward: the KL divergence plus a final-answer correctness reward.

(B) **How it works**:
```python
# Training loop
for batch in data:
    # 1. Perception: sample region from localization head
    region_dist = perceiver(image, question)  # categorical over grid cells
    region = gumbel_softmax(region_dist, tau=1.0, hard=True)  # discrete sample
    log_prob = region_dist.log_prob(region)
    
    # 2. Reasoning with region attended
    p_yes = reasoner(image, question, region)  # softmax over answers
    
    # 3. Reasoning with region masked (replace region features with learned [MASK] token)
    p_no = reasoner(image, question, mask_region(region))
    
    # 4. Intrinsic reward: KL(p_yes || p_no)
    kl = (p_yes * (p_yes.log() - p_no.log())).sum()
    
    # 5. Extrinsic reward: correctness (1 if correct, 0 else)
    correct = answer == batch['answer']
    final_reward = 1.0 if correct else 0.0
    
    # 6. Combined reward for perception
    total_reward = kl + 0.1 * final_reward  # lambda=0.1
    
    # 7. Policy gradient for perception
    loss_perceiver = -total_reward * log_prob
    
    # 8. Standard cross-entropy for reasoning
    loss_reasoner = -p_yes.log()[batch['answer_index']]
    
    # 9. Backpropagate both losses
    (loss_perceiver + loss_reasoner).backward()
    optimizer.step()
```
Hyperparameters: tau=1.0 for Gumbel-Softmax temperature, lambda=0.1 for reward mixing.

(C) **Why this design**: We chose KL divergence over alternative intrinsic rewards like attention entropy or feature similarity because KL directly measures the change in reasoning distribution when the region is ablated, which is a causal quantity, while attention may only reflect correlation; the cost is that computing KL requires a second forward pass through the reasoner, doubling inference cost per update. We chose REINFORCE with a Gumbel-Softmax estimator over continuous relaxation (e.g., soft-attention) because the region selection is inherently discrete in our cropping-based pipeline, and REINFORCE provides unbiased gradients; the trade-off is higher variance, which we mitigate by using the dense KL reward. We chose to mask the region by replacing its features with a learned [MASK] token rather than cropping it out entirely because cropping would alter the spatial structure and introduce a confound (missing context), while feature masking isolates the region's contribution; the downside is that the model may learn to rely on the mask token, which we mitigate by randomizing mask embeddings during training.

(D) **Why it measures what we claim**: The KL divergence between p(y|x,region) and p(y|x,region\_masked) measures the causal necessity of the perceived region for reasoning because the key assumption is that masking the region is a valid intervention that removes its causal influence while keeping everything else identical; this assumption fails when the mask leaks information (e.g., if the mask token carries statistical regularities from training), in which case the KL reflects reliance on the mask rather than region content. The log-prob of the sampled region in REINFORCE measures the exploration probability of selecting that region, and using total\_reward (KL + final) as the reward signal directly operationalizes the goal of finding regions that are both causally important and lead to correct answers; the failure mode is that KL may be high simply because the model is uncertain about the masked output, not because the region is specifically needed—this is mitigated by combining KL with final reward to prefer regions that actually help correctness.

## Contribution

(1) A novel dense intrinsic reward for perception in visual reasoning, computed as the KL divergence between reasoning distributions conditioned on the presence vs. absence of a candidate region. (2) A principled justification for using this KL as a measure of causal necessity, linking it to the reasoning model's own behavior without external annotations. (3) An end-to-end training framework that jointly optimizes perception and reasoning using a combination of this intrinsic reward and final answer reward, providing intermediate supervision for perception.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|-----------------------|
| Dataset | HRBench8K | High-resolution visual reasoning benchmark. |
| Primary metric | Visual reasoning accuracy | Direct measure of method success. |
| Baseline 1 | Qwen3-VL-Instruct | Standard MLLM without CPR training. |
| Baseline 2 | CPR with attention entropy | Tests KL vs. entropy as causal measure. |
| Baseline 3 | CPR with soft-attention | Tests discrete sampling vs. continuous. |
| Ablation | CPR without final reward | Tests necessity of correctness signal. |

### Why this setup validates the claim

This experimental design isolates each component of the proposed Causal Perception Reward (CPR) framework. HRBench8K is chosen because its high-resolution images require fine-grained perception to locate small but critical regions. The primary metric, visual reasoning accuracy, directly reflects whether the perception module successfully supports reasoning. Baseline 1 (Qwen3-VL-Instruct) tests the overall benefit of CPR over a standard MLLM. Baseline 2 (attention entropy) tests whether KL divergence is a better measure of causal importance than attention. Baseline 3 (soft-attention) tests whether discrete region selection via REINFORCE is advantageous over continuous blending. The ablation (without final reward) tests the necessity of the correctness signal for guiding perception. If any baseline matches CPR's performance, or if the ablation performs similarly, the corresponding sub-claim fails, providing a falsifiable test of the central idea.

### Expected outcome and causal chain

**vs. Qwen3-VL-Instruct** — On a case where a tiny logo is the key to answering a question, the baseline's global attention spreads over the whole image, diluting the logo's influence, leading to incorrect answer because it lacks explicit region-level perception. Our method's perception module, trained with KL reward, learns to select that specific region, and the reasoner uses it to give correct answer. We expect CPR to significantly outperform on questions requiring pinpoint localization, with a large gap (e.g., >10% on HRBench8K's fine-grained subsets).

**vs. CPR with attention entropy** — On a case with a visually salient but irrelevant region (a red herring), the entropy baseline may reward high attention entropy (uncertainty) or low entropy depending on design, but it does not measure causal impact; it might still select the salient region. Our KL reward directly captures how much the reasoning distribution changes when that region is masked, avoiding spurious attention. We expect our method to avoid selecting irrelevant regions, leading to higher accuracy on questions with distractors, with a noticeable gap (e.g., 5-8%).

**vs. CPR with soft-attention** — On a case with overlapping candidate regions, the soft-attention baseline blends features from all regions, diluting each region's causal effect, making it harder to attribute importance. Our discrete selection forces a hard choice, enabling clearer causal attribution via KL. We expect our method to achieve higher accuracy on ambiguous scenes where a single region is critical, with a moderate but consistent gap (e.g., 3-5%).

### What would falsify this idea

If CPR underperforms baseline 1 on fine-grained subsets, or if the gain is uniform across all questions rather than concentrated on those requiring precise localization, the claim that KL-based causal reward is effective would be falsified.

## References

1. Perceive-to-Reason: Decoupling Perception and Reasoning for Fine-Grained Visual Reasoning
2. HiDe: Rethinking The Zoom-IN method in High Resolution MLLMs via Hierarchical Decoupling
3. Divide, Conquer and Combine: A Training-Free Framework for High-Resolution Image Perception in Multimodal Large Language Models
4. ZoomEye: Enhancing Multimodal LLMs with Human-Like Zooming Capabilities through Tree-Based Image Exploration
5. Alphazero-like Tree-Search can Guide Large Language Model Decoding and Training
