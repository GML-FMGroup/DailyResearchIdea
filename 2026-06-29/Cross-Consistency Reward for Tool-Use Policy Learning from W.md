# Cross-Consistency Reward for Tool-Use Policy Learning from Weak Multimodal Signals

## Motivation

Existing methods like ProMSA and SenseNova-MARS train tool-use policies via rejection-sampling SFT followed by RL, requiring high-quality trajectories annotated with tool-beneficial labels. This reliance on curated data limits scalability and prevents learning from weak or sparse success signals—a structural limitation across both systems. A self-supervised reward that leverages cross-modal coherence, computed without human annotation, could replace curated labels and enable policy learning from the environment directly.

## Key Insight

Cross-modal consistency between retrieved images and text from different tool invocations provides a dense, self-supervised reward that correlates with task success because correct tool use produces coherent multimodal evidence, even without ground-truth annotations.

## Method

### (A) What It Is
CRAFT (Cross-consistency Reward for Agent Tool-use) is a reinforcement learning algorithm that trains a multimodal agent to select tool-use actions (image search, text search, stop) by maximizing a self-supervised reward defined as the cosine similarity between embeddings of retrieved image and text content, computed by a frozen multimodal encoder (e.g., CLIP ViT-B/32 with 512-dim output). Input: image-question query, set of tools; output: a trained policy that produces tool-use trajectories.

### (B) How It Works
```python
# Pseudocode for CRAFT training loop
initialize policy π (e.g., GRPO-based but sequence-level)
initialize frozen CLIP encoder E
# Load-bearing assumption: Cross-modal CLIP similarity between retrieved image and text pairs is a valid reward signal that correlates with task success (correct answer).
for each rollout:
  state = (image, question)
  trajectory = []
  for t in 1..max_steps:
    action = π.sample(state)
    if action == image_search:
      retrieved_image = image_retrieve(state)  # e.g., using CLIP ViT-L/14 for image retrieval from database of 10M images
      state += retrieved_image
    elif action == text_search:
      retrieved_text = text_retrieve(state)   # e.g., using BM25 for text retrieval from Wikipedia corpus
      state += retrieved_text
    elif action == stop:
      answer = VLM.generate(state)           # e.g., LLaVA-1.5-13B
      break
  
  # Compute cross-modal consistency reward
  reward = 0
  pairs = []
  for each image_retrieved_i in trajectory:
    for each text_retrieved_j in trajectory:
      pairs.append( (image_retrieved_i, text_retrieved_j) )
  if pairs:
    similarities = [cos_sim(E(img), E(txt)) for img,txt in pairs]  # cos_sim = dot(x,y)/(||x||*||y||)
    reward = mean(similarities)
  else:
    reward = 0  # no cross-modal pair; perhaps terminal reward?
  
  # Update policy via sequence-level RL (e.g., GSPO)
  loss = GSPO_loss(log_prob_seq, reward, clip_epsilon=0.2)  # GSPO: Group Sequence Policy Optimization, batch_size=64
  optimizer.step(loss)  # AdamW, lr=1e-5
```
Hyperparameters: max_steps=5, GSPO clipping epsilon=0.2, batch_size=64, learning_rate=1e-5, policy network is a 3-layer MLP with hidden size 256 and GeLU activation.

### (C) Why This Design
We chose a self-supervised reward based on cross-modal CLIP similarity over a handcrafted heuristic reward (e.g., keyword overlap) because CLIP provides a learned, continuous measure of semantic alignment that generalizes across domains, accepting the cost that CLIP may be biased toward common co-occurrences rather than factual accuracy. We used a mean over all pairs instead of a max or last-step reward because averaging provides a denser signal over the entire trajectory, but it may dilute strong single pair evidence. We adopted GSPO over token-level PPO because sequence-level clipping stabilizes learning when reward is sparse and noisy, though it increases computational cost per update. We opted for a frozen CLIP encoder rather than fine-tuning it to avoid reward hacking, but this limits adaptability to domain-specific consistency patterns.

### (D) Why It Measures What We Claim
The mean CLIP similarity over all image–text pairs in the trajectory measures cross-modal coherence because it quantifies how well the retrieved visual and textual content align in a shared semantic space; this assumes that correct tool use retrieves complementary information that is semantically consistent (e.g., an image of a monument and a text snippet about its location). The assumption fails when the agent retrieves two pieces of information that are both consistent with each other but irrelevant to the question (e.g., an image and text about the same unrelated concept) — in that case the reward signal reflects coherence but not utility, potentially reinforcing off-target behavior. The GSPO loss function operationalizes policy optimization under this reward by normalizing trajectory probabilities; its clipping mechanism measures the extent to which we trust the reward signal, because a high clip threshold allows large updates even from noisy rewards — this assumption holds when rewards are well-calibrated (which we assume via frozen CLIP) but fails if reward noise is high, in which case the clipping prevents excessive variance at the cost of slower learning.

**Load-bearing assumption explicitly stated:** Cross-modal CLIP similarity between retrieved image and text pairs is a valid reward signal that correlates with task success (correct answer). To verify this, we compute Pearson correlation between the mean CLIP similarity and binary answer correctness on a held-out calibration set of 512 trajectories. If correlation < 0.3, we flag a warning; in experiments we report this correlation to assess the reward's validity.

## Contribution

(1) A self-supervised reward function for tool-use policies derived from cross-modal consistency between retrieved images and text, computed via a frozen multimodal encoder without human annotation. (2) A demonstration that RL-based policy learning with this reward can replace curated trajectory data, enabling scalable training from weak environmental signals. (3) A design principle: the structural property of cross-modal coherence serves as a dense, naturally available signal for tool-use learning in multimodal RAG agents.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
|------|--------|-----------|
| Dataset | InfoSeek | Knowledge-based VQA needs search. |
| Primary metric | Top-1 accuracy | Measures answer correctness directly. |
| Additional metric | Correlation between mean CLIP similarity and answer correctness on held-out calibration set (512 trajectories) | Validates the load-bearing assumption. |
| Baseline 1 | Strong RAG | No tool selection; tests retrieval quality. |
| Baseline 2 | ProMSA | Agent but without RL reward shaping. |
| Baseline 3 | SenseNova-MARS | RL agent with different reward design. |
| Ablation 1 | CRAFT w/o cross-modal reward | Reward is random (uniform in [-1,1]); isolates reward effect. |
| Ablation 2 | CRAFT with question-conditioned reward | Cross-modal similarity weighted by relevance to query: reward = mean(cos_sim(E(img), E(txt)) * cos_sim(E(query), E(img)) * cos_sim(E(query), E(txt))). Tests the load-bearing assumption directly. |

### Why this setup validates the claim

InfoSeek requires retrieving both visual and textual knowledge, making it ideal to test cross-modal coherence. Comparing against Strong RAG isolates the benefit of iterative tool selection, while ProMSA and SenseNova-MARS test whether our self-supervised reward improves over sequential or sparse-reward baselines. The ablation with random reward directly tests the causal role of CLIP similarity. The additional ablation with question-conditioned reward tests whether ignoring query relevance is detrimental. The correlation metric on a held-out set provides empirical evidence for (or against) the load-bearing assumption. This combination forms a falsifiable test: if cross-modal reward is effective, our method should outperform all baselines especially on ambiguous multi-step queries where coherence matters.

### Expected outcome and causal chain

**vs. Strong RAG** — On a case where the query requires disambiguation (e.g., "What style is this building?" from a blurry image), Strong RAG retrieves a single text snippet that may be irrelevant because it cannot iteratively refine search. Our method instead performs sequential image and text retrieval, computing cross-modal CLIP similarity between all pairs, and selects actions that maximize alignment, ensuring the final answer is grounded in consistent multimodal evidence. We expect a noticeable gap on multi-step examples but parity on single-step ones.

**vs. ProMSA** — On a case requiring multiple tool invocations (e.g., first retrieve an image of a landmark, then retrieve its history), ProMSA's progressive search lacks a reward signal for cross-modal coherence, so it may retrieve an irrelevant image first and then a text that does not align, leading to wrong answer. Our method's dense reward penalizes such mismatches, guiding the agent to choose complementary retrievals. We expect our method to show higher accuracy on trajectories with >2 steps.

**vs. SenseNova-MARS** — On a case where the final answer is correct but retrieved pairs are inconsistent (e.g., correct answer by luck), SenseNova-MARS's sparse final-answer reward fails to penalize the inconsistency, potentially reinforcing poor retrieval habits. Our dense cross-modal reward directly encourages alignment at each step, leading to more robust tool use. We expect our method to achieve faster convergence and higher final accuracy, especially on hard examples with multiple relevant retrievals.

**Ablation 2 (question-conditioned reward)** — If the load-bearing assumption holds, this ablation should perform similarly to the original CRAFT, because the extra weighting by query relevance may not add much if retrieved pairs are already relevant. If it significantly outperforms, then the original reward suffers from the irrelevant-coherence failure mode, supporting the need for redesign.

### What would falsify this idea

If our method performs similarly to the no-reward ablation, or if gains are uniform across all subsets rather than concentrated on multi-step or ambiguous queries, then the cross-modal reward is not the causal driver of improvement. Additionally, if the correlation between mean CLIP similarity and answer correctness on the calibration set is low (e.g., <0.2), that would directly falsify the load-bearing assumption.

## References

1. ProMSA:Progressive Multimodal Search Agents for Knowledge-Based Visual Question Answering
2. SenseNova-MARS: Empowering Multimodal Agentic Reasoning and Search via Reinforcement Learning
3. Group Sequence Policy Optimization
4. VisRAG: Vision-based Retrieval-augmented Generation on Multi-modality Documents
