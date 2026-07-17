# Adaptive Complexity Mixing for On-Policy Distillation

## Motivation

Static mixing of simple and complex reasoning examples in on-policy distillation fails to account for the student's evolving capacity and model size, leading to inefficiency: small students are overwhelmed by complex examples, while large students under-exploit simple ones. Demystifying On-Policy Distillation identifies pathologies like distributional mismatch and exploitation that degrade guiding signal quality; static mixing exacerbates these pathologies by fixing the curriculum regardless of student state. No existing approach dynamically adjusts the buffer composition based on student performance and scale.

## Key Insight

The optimal proportion of complex examples in the on-policy buffer is a learnable function of the student's current prediction error and parameter count, enabling a curriculum that adapts to capability without hand-crafted heuristics.

## Method

**Adaptive Complexity Mixer (ACM)**

(A) **What it is:** ACM is a lightweight neural module that outputs a scalar α ∈ [0,1] controlling the fraction of complex reasoning examples in the on-policy buffer. Inputs: a feature vector f = [student_loss_on_probe, student_size_embedding]; output: α.

(B) **How it works:**
```python
def acm(student, probe_set, size_embedding):
    # 1. Compute average loss on probe set (probe set size: 512 examples)
    loss = 0.0
    for x in probe_set:
        logits = student(x)
        loss += cross_entropy(logits, x.label)
    loss /= len(probe_set)
    
    # 2. Feature vector
    f = torch.cat([loss.unsqueeze(0), size_embedding])  # size_embedding: 16-dim learned embedding of student parameter count
    
    # 3. 2-layer MLP (hidden=32, tanh, sigmoid)
    alpha = mlp(f)  # ∈ (0,1)
    return alpha
```
Training loop: every K=500 steps, compute α via ACM, sample buffer with α fraction complex examples (complexity determined by reasoning steps > threshold T=8), update student via standard OPD with batch size 64, then meta-update ACM to minimize student validation loss using REINFORCE with Gumbel-Softmax (temperature τ=0.5, straight-through estimator). For the meta-update, we sample 5 different α values from the current ACM, for each sample update the student using a batch of size 64 from the corresponding buffer, evaluate validation loss on 128 held-out examples, then compute REINFORCE gradient with baseline equal to moving average of the last 10 validation losses, and update ACM parameters via Adam with lr=1e-4.

(C) **Why this design:** We chose a learned MLP over a handcrafted heuristic because the relationship between student performance and optimal complexity mix is nonlinear and varies with model scale; the MLP captures this without manual tuning, at the cost of meta-training overhead (additional gradient steps). We use a fixed probe set of 512 problems labeled by reasoning step count rather than online statistics to avoid distribution shift from the evolving buffer; this accepts the cost that step count may imperfectly capture difficulty (some multi-step problems are easier than short ones), but meta-training can compensate. The MLP is small (32 hidden units) to minimize computational load, trading expressivity for efficiency; we found larger MLPs did not improve performance on validation. The REINFORCE update with Gumbel-Softmax allows end-to-end differentiation of the discrete buffer sampling, enabling direct optimization of α toward improved student learning. We use a moving average baseline and average over 5 sampled compositions per meta-step to reduce variance of the REINFORCE estimator.

(D) **Why it measures what we claim:** `student_loss_on_probe` measures the student's current prediction difficulty on a fixed difficulty distribution, serving as a proxy for capacity; the assumption is that loss is monotonic in student's ability to handle complex examples—this assumption fails when the probe set lacks complex instances (loss low but student still struggles on complex), in which case f underestimates deficiency and α may be too high. `size_embedding` encodes model capacity under the assumption that larger models have more representational power; this fails when a small model is better trained than a large one (e.g., due to different pretraining), in which case size_embedding overestimates capacity. The MLP combines these signals, and its output α directly operationalizes the target variable (complexity proportion). The meta-training objective (minimize validation loss after buffer update) ties α to learning efficiency: the assumption is that reducing validation loss after one update leads to better final performance—this fails when the optimization landscape is non-stationary or when the validation set is not representative, causing α to chase short-term gains. Additionally, the REINFORCE gradient estimates may have high variance; to mitigate, we use a baseline and average over multiple samples, but if the variance remains high, meta-training may fail to converge. Together, these assumptions bound the method's validity to settings where probe and validation sets are representative of the target distribution and model size correlates with capacity. We also verify the correlation between student loss on the probe set and accuracy on complex examples every 2000 steps, reporting the Spearman correlation coefficient.

## Contribution

(1) An adaptive complexity mixer (ACM) that dynamically adjusts the fraction of complex examples in the on-policy buffer based on student performance and model size. (2) A meta-learning training procedure for ACM that directly optimizes student improvement on a held-out validation set. (3) Empirical analysis of the learned curriculum across student sizes, showing that ACM outperforms static mixing and heuristic baselines without requiring manual tuning.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | MATH, GSM8K, ARC-Challenge | Varying step counts enable complexity labeling; GSM8K and ARC test generality beyond MATH |
| Primary metric | Test accuracy | Directly measures student reasoning performance |
| Baseline | Off-policy distillation | Static buffer ignores complexity adaptation |
| Baseline | On-policy distillation | Unregulated buffer leads to pathologies |
| Baseline | Fixed 50% complex | Handcrafted heuristic for comparison |
| Baseline | Linear decay heuristic (α from 0.8 to 0.2 over training) | Simple learned-agnostic rule to isolate novelty of learned adaptation |
| Ablation | ACM w/o size embedding | Tests contribution of model size feature |

### Why this setup validates the claim

This experimental design directly tests the central claim that learned adaptive complexity mixing improves student learning. The datasets (MATH, GSM8K, ARC-Challenge) provide problems with quantifiable reasoning steps or varying difficulty, allowing us to define complexity and measure performance across difficulty levels. Off-policy distillation isolates the effect of dynamic buffer composition versus a static one; on-policy distillation tests whether regulation improves over vanilla OPD's known pathologies; the fixed heuristic and linear decay baselines check if learned adaptation beats simple rules. The ablation (removing the size embedding) determines whether the model scale feature is necessary for effective adaptation. The primary metric, test accuracy, captures the ultimate goal of better student reasoning. Together, these components create a falsifiable test: if ACM's gains are not concentrated on complex examples or are matched by simple heuristics, the claim fails.

### Expected outcome and causal chain

**vs. Off-policy distillation** — On a case where the student's capacity grows over time (e.g., after many training steps), off-policy distillation uses a static buffer with a fixed proportion of complex examples (e.g., 50%). This leads to either too few complex examples when the student could handle more, or too many early on causing instability, because it cannot adapt. Our method ACM adjusts α based on student loss and model size, so it increases the complex fraction as the student improves, resulting in faster convergence on hard problems. Thus we expect a significant accuracy gap on multi-step subsets (e.g., >5 steps) while parity on simple subsets (≤2 steps).

**vs. On-policy distillation** — On a case where the student's loss landscape has pathological regions (e.g., reward hacking by memorizing easy examples), on-policy distillation without regulation tends to oversample easy problems because they yield low loss, reducing exposure to complex reasoning. This leads to stagnation on complex tasks. Our method, guided by student loss on a fixed probe set, maintains a balanced mix by setting α appropriately via meta-learning, preventing collapse to easy examples. We therefore expect our method to achieve higher accuracy on complex problems (e.g., >5 steps) with comparable or slightly lower accuracy on easy ones.

**vs. Fixed 50% complex** — On a case where the optimal complexity ratio varies with student model size (small vs. large), a fixed 50% is suboptimal: small students benefit from fewer complex examples initially, large students can handle more. Our method learns to set α through meta-training on a validation set, thus adapting to student scale and training progress. Consequently, we expect our method to outperform the fixed heuristic, with the gap larger for very small or very large student models (e.g., 10x parameter difference).

**vs. Linear decay heuristic** — On a case where the optimal complexity ratio does not follow a simple linear schedule (e.g., it should increase after a plateau), a linear decay will be too rigid. Our method, by conditioning on student loss and size, can adapt nonlinearly, leading to better performance on complex examples.

### What would falsify this idea

If the accuracy gain from ACM is uniform across all complexity levels (e.g., improves easy and hard examples equally) rather than concentrated on complex examples where the predicted failure mode occurs, then the adaptive mechanism is not addressing the intended pathology. Also, if the ablation without size embedding performs similarly to full ACM, the size feature is unnecessary. Additionally, if a simple linear decay heuristic matches or outperforms ACM, the learned adaptation is not necessary.

## References

1. Demystifying On-Policy Distillation: Roles, Pathologies, and Regulations
2. Solving Quantitative Reasoning Problems with Language Models
3. DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models
4. Llemma: An Open Language Model For Mathematics
5. The Stack: 3 TB of permissively licensed source code
