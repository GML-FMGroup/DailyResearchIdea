# Surpassing the Teacher: Self-Play Distillation with Verified Trajectory Reservoirs for Language Models

## Motivation

Existing on-policy distillation methods, such as those analyzed in 'Demystifying On-Policy Distillation', are fundamentally bounded by the teacher's knowledge because the student only imitates teacher-generated trajectories. Even with exploration, the student cannot correct teacher errors or produce novel reasoning strategies that go beyond the teacher's capabilities, as the teacher's outputs define the target distribution. This structural limitation caps the student's performance at the teacher's level, preventing any improvement beyond it.

## Key Insight

The student can surpass the teacher by learning from its own self-generated trajectories that are verified as correct, because these trajectories provide a training signal that is decoupled from the teacher's distribution and can include reasoning paths the teacher never produced.

## Method

(A) **What it is**: Self-Improving Distillation with Verified Trajectories (SID-VT) is an iterative distillation algorithm that trains a student language model by combining on-policy distillation from a teacher with a dynamically growing reservoir of the student's own correct self-generated trajectories. Input: teacher T, student S, problem set D with a correctness verification function V (e.g., exact answer match). Output: improved student S*. **Load-bearing assumption**: Correctly answered self-generated trajectories from the student, as judged by exact match verification augmented with self-consistency (majority voting over multiple samples), contain transferable reasoning strategies that improve the student's future performance.

(B) **How it works** (pseudocode):
```python
def SID_VT(T, S, D, V, reservoir_capacity=10000, num_iterations=5, temperature=1.0, batch_size=32, num_selfplay_samples=8):
    R = []  # knowledge reservoir
    for iteration in range(num_iterations):
        # Phase 1: Distillation from teacher and reservoir
        for batch in D.batches(batch_size):
            t_traj = T.generate(batch.prompts, temperature=temperature)
            # Sample up to 2 reservoir trajectories per prompt if available
            if len(R) > 100:
                res_trajs = random.sample(R, min(2 * len(batch), len(R)))
            else:
                res_trajs = []
            positive_trajs = [t_traj] + res_trajs
            # Train student with cross-entropy on positive trajectories
            S.train(positive_trajs, loss='cross_entropy', temperature=temperature)
        # Phase 2: Self-play generation and reservoir update
        all_student_trajs = S.generate(D.prompts, temperature=temperature, num_samples=num_selfplay_samples)
        for i, prompt in enumerate(D.prompts):
            traj_list = all_student_trajs[i]
            # Compute majority answer among self-play samples
            answers = [traj.answer for traj in traj_list]
            majority_answer = max(set(answers), key=answers.count)  # simple majority
            for traj in traj_list:
                if V(traj, prompt.answer) and traj.answer == majority_answer:
                    if traj not in R:  # simple deduplication
                        R.append(traj)
                        if len(R) > reservoir_capacity:
                            R.pop(0)  # FIFO eviction
    return S
```
Hyperparameters: reservoir_capacity=10000, num_iterations=5, temperature=1.0, batch_size=32, num_selfplay_samples=8.

(C) **Why this design**: We chose on-policy distillation (phase 1) over off-policy because on-policy ensures the student is trained on trajectories from the current teacher, maintaining alignment with the teacher's distribution early on; the trade-off is computational cost of regenerating teacher outputs each iteration, but this prevents distribution mismatch. Using a reservoir with FIFO eviction rather than priority-based eviction (e.g., by confidence) simplifies the method and avoids introducing a separate scoring model; the cost is that older, potentially less relevant correct trajectories may persist too long, but we mitigate this by the student's improving generation quality over time. Sampling reservoir trajectories randomly rather than by relevance avoids bias toward high-frequency patterns; however, it may include low-diversity examples, but the reservoir size keeps diversity manageable. Finally, we use correctness combined with self-consistency (majority voting over multiple self-play samples) as the filter for reservoir addition rather than a confidence threshold because this dual filter ensures both answer correctness and consistent reasoning across samples; the trade-off is that some correct but novel trajectories may be discarded if they deviate from majority, but this reduces the risk of memorization artifacts.

(D) **Why it measures what we claim**: The computational quantity `correctness of self-generated trajectory` measures the student's ability to produce a correct reasoning path beyond the teacher because correctness is defined by the problem's answer, independent of the teacher's output distribution; this assumption fails when the verification function V is noisy (e.g., exact match for problems with multiple valid answer formats), in which case a trajectory labeled correct may actually be flawed. The additional self-consistency filter assesses reasoning quality by requiring agreement across multiple samples, capturing transferable strategies; this assumption fails if the student's self-play samples are highly correlated (e.g., due to low temperature), in which case majority voting offers little filtering. The quantity `reservoir inclusion` operationalizes the concept of storing high-quality student-generated examples; the underlying assumption is that correctly answered and self-consistent trajectories reflect effective reasoning strategies that generalize; this assumption fails when the student relies on spurious patterns (e.g., GSM-Symbolic-like shortcuts), causing the reservoir to contain examples that do not transfer. The quantity `on-policy distillation loss on mixed trajectories` measures how well the student aligns with both teacher and reservoir examples; the underlying assumption is that the reservoir trajectories are at least as informative as teacher outputs, which holds when the student's correct generations cover reasoning gaps not present in the teacher's distribution; this fails if the reservoir contains mostly redundant or low-diversity trajectories, in which case the loss measures only overfitting to a narrow set of patterns.

## Contribution

(1) A self-play distillation framework (SID-VT) that integrates on-policy distillation with a dynamically updated reservoir of student-generated correct trajectories, enabling the student to surpass the teacher. (2) The finding that correctness-filtered self-generated data, when mixed with teacher outputs in on-policy distillation, allows the student to discover and reinforce novel reasoning paths beyond the teacher's demonstrations. (3) Analysis of the reservoir's effect on distillation dynamics, showing that mixing reservoir samples accelerates convergence and improves final accuracy on mathematical reasoning benchmarks.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
|------|--------|-----------|
| Dataset | MATH | Challenging reasoning benchmark; diverse. |
| Primary metric | Exact Match accuracy | Direct measure of answer correctness. |
| Baseline 1 | Standard Distillation (off-policy) | Off-policy baseline; no iterative updates. |
| Baseline 2 | On-Policy Distillation (no reservoir) | Iterative teacher only; no student self-gen. |
| Ablation | SID-VT with confidence-based reservoir | Tests necessity of correctness filtering. |

### Why this setup validates the claim
MATH is a sufficiently difficult and varied reasoning benchmark that stresses both teacher and student models. Exact match accuracy provides an unambiguous measure of output correctness, directly testing whether the student's self-generated trajectories improve reasoning. Comparing against standard off-policy distillation isolates the effect of iterative on-policy updates; comparing against on-policy distillation without a reservoir isolates the effect of including the student’s own correct trajectories. The ablation using confidence-based filtering (instead of raw correctness) tests whether the verification function is essential or whether any positive signal suffices. Together, these contrasts form a falsifiable test: if SID-VT outperforms both baselines and the ablation, the central claim—that a reservoir of verified student trajectories boosts distillation—is supported; if not, the assumed mechanism is undermined.

### Expected outcome and causal chain

**vs. Standard Distillation (off-policy)** — On a problem where the teacher makes a systematic reasoning error (e.g., misapplies the quadratic formula), a standard distillation student merely reproduces that error because it only observes the teacher's fixed output distribution. Our method instead encounters the student's own correct trajectories on similar problems (e.g., applying the formula correctly after repeated practice) in the reservoir; these correct paths are used alongside teacher outputs during training, giving the student a chance to override the teacher’s mistake. Consequently, we expect a noticeable gap on error-prone subcategories (e.g., algebra word problems) where teacher errors are consistent, with SID-VT achieving perhaps 5–10% higher accuracy on those subsets while performing similarly on others.

**vs. On-Policy Distillation (no reservoir)** — On a problem where the student initially generates a correct but under-explored solution path (e.g., a non-standard factoring trick), on-policy distillation only sees the teacher’s standard method and may never reinforce the student’s alternative correct reasoning, leading to stagnation. Our method adds that correct student trajectory to the reservoir, so in later distillations the student is trained on both the teacher’s and its own successful methods, promoting diversity and deeper understanding. We expect SID-VT to show a progressively widening gap over iterations (e.g., after iteration 3, the student’s accuracy grows faster on problems requiring multiple solution strategies), with a final improvement of 2–4% overall.

**vs. SID-VT with confidence-based reservoir** — On a problem where the student is poorly calibrated (e.g., low confidence despite a correct answer), the confidence-based filter discards the trajectory, starving the reservoir of valuable correct examples. Our method retains it because correctness is verified independently, leading to a richer reservoir. We expect the confidence-based ablation to underperform on problems where student confidence is miscalibrated (e.g., novel problem types), with accuracy drops of 3–6% on those subsets, confirming that correctness-based filtering is crucial.

### What would falsify this idea
If the performance of SID-VT is no better than on-policy distillation without a reservoir, or if the confidence-based ablation matches SID-VT, then the central claim that a correctness-filtered reservoir is beneficial would be refuted. Specifically, the diagnostic pattern is: if the gain is uniform across all subsets rather than concentrated in error-prone or confidence-miscalibrated areas, the proposed mechanism is unsupported.

## References

1. Demystifying On-Policy Distillation: Roles, Pathologies, and Regulations
2. Solving Quantitative Reasoning Problems with Language Models
3. DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models
4. Llemma: An Open Language Model For Mathematics
5. The Stack: 3 TB of permissively licensed source code
