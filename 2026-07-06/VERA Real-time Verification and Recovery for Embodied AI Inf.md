# VERA: Real-time Verification and Recovery for Embodied AI Inference Runtimes

## Motivation

Embodied.cpp provides a portable runtime for VLA models but assumes the model outputs are always correct, leading to failures under noisy demonstrations or ambiguous states. This limitation persists because the runtime has no mechanism to detect and correct reasoning errors during long-horizon tasks, which is critical for reliable robot deployment. Prior work like Embodied.cpp demonstrates the need for runtime adaptations to handle such errors.

## Key Insight

Action errors in long-horizon tasks can be detected by the mismatch between the predicted action's expected state transition and the actual observation, as measured by the model's internal feature embeddings, without requiring ground truth labels.

## Method

VERA is a lightweight runtime verification module that takes the current observation, action prediction, and model's intermediate features, and outputs a decision (execute, re-prompt, or safe fallback) to correct reasoning errors in real-time.

```pseudocode
class VERA:
    def __init__(self, model, th_logp=-1.0, th_shift=0.5, max_fails=3):
        self.model = model
        self.th_logp = -1.0      # log-probability threshold, tuned on validation set of 100 episodes
        self.th_shift = 0.5      # feature shift threshold, tuned on same validation set
        self.max_fails = 3
        self.fail_count = 0
        self.history = []        # list of (obs_feat, action), max length 10

    def verify(self, obs, action):
        # Step 1: Compute log-probability of action given obs
        log_prob = self.model.action_log_prob(obs, action)
        # Step 2: Extract intermediate feature (e.g., penultimate layer, dimension 512)
        feat = self.model.extract_feature(obs)
        # Step 3: Check confidence
        if log_prob < self.th_logp:
            # Low confidence -> check feature consistency
            if len(self.history) > 0:
                prev_feat = self.history[-1][0]
                shift = np.linalg.norm(feat - prev_feat)
                if shift > self.th_shift:
                    # Likely error: re-prompt
                    return 're-prompt'
            # else: execute despite low confidence
        # Update history (FIFO, max length 10)
        self.history.append((feat, action))
        if len(self.history) > 10:
            self.history.pop(0)
        return 'execute'

    def recover(self, obs):
        # After max_fails consecutive re-prompts, fallback to stop
        self.fail_count += 1
        if self.fail_count >= self.max_fails:
            self.fail_count = 0
            return 'safe_stop'
        return 're-prompt'
```

(C) **Why this design**: We chose log-probability as a cheap first-pass filter over more expensive uncertainty quantification (e.g., MC dropout) to minimize latency, accepting that log-prob may be miscalibrated for correctness. We selected feature shift over pixel-level differencing because features capture semantic task progress, though they may be sensitive to non-task variations like lighting. We used a fixed threshold handset rather than a learned classifier to avoid requiring labeled error data, accepting that thresholds may require task-specific tuning. Third, we limited re-prompt attempts to prevent infinite loops, trading off between recovery chance and operational safety.

(D) **Why it measures what we claim**: The computational quantity `log_prob` measures model confidence under the assumption that the model is well-calibrated for its training distribution; this assumption fails when novel states cause overconfident incorrect predictions, in which case `log_prob` reflects fluency rather than error probability. The `feature shift` measures discordance between predicted and observed state transitions because the model's features encode task-relevant progress; this assumption fails when features respond to irrelevant sensory changes (e.g., camera noise), in which case `shift` reflects spurious variation. **Load-bearing assumption**: The conjunction of low log-probability and high feature shift reliably detects reasoning errors without being triggered by miscalibration or irrelevant sensory changes. By requiring both low `log_prob` and high `shift` to trigger re-prompt, we exploit the statistical conjunction: true reasoning errors tend to depress calibration and perturb features, while single-signal false positives (e.g., one low-prob but consistent correct action) are filtered out.

## Contribution

(1) VERA, a runtime verification module that integrates into existing embodied inference runtimes like Embodied.cpp to detect and recover from action errors in long-horizon tasks. (2) The design principle of combining model confidence (log-probability) with temporal feature consistency to achieve error detection without ground truth labels. (3) A recovery policy that triggers re-prompting or safe fallback based on consecutive failures.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Habitat object rearrangement & LIBERO object manipulation (10 tasks) | Standard long-horizon embodied benchmarks; diverse tasks |
| Primary metric | Task success rate | Directly measures task completion |
| Baseline 1 | No verification | Baseline without any error correction |
| Baseline 2 | Log-prob only | Isolates effect of log-prob threshold |
| Baseline 3 | MC Dropout (10 samples) | Gold-standard uncertainty baseline |
| Baseline 4 | Learned verification classifier (2-layer MLP, hidden=256) | Assess heuristic vs. learned approach |
| Ablation-of-ours | No feature shift | Test contribution of feature shift |
| Diagnostic analysis | Ground-truth error labels on validation set | Verify that feature shift correlates with failure |

We also release an open-source implementation with a Docker environment to ensure reproducibility.

### Why this setup validates the claim
This setup validates the central claim by pitting VERA against four baselines that test distinct sub-claims. Habitat object rearrangement and LIBERO provide diverse long-horizon tasks where reasoning errors occur due to partial observability and novel states. The primary metric, task success rate, directly captures whether VERA's runtime verification improves task completion by correcting errors. Comparing to No verification measures absolute gain. Log-prob only isolates the confidence filter's effect, showing the added value of feature shift. MC Dropout represents a more expensive uncertainty method; parity on performance but lower latency would support VERA's efficiency claim. The learned verification classifier baseline tests whether a data-driven approach outperforms the heuristic; we expect VERA to be competitive with lower labeling cost. The ablation without feature shift tests whether the shift component is essential. The diagnostic analysis (plot of feature shift vs. error using ground-truth labels) directly validates the causal mechanism. Together, these comparisons form a falsifiable test: if VERA's advantage vanishes on subsets without feature shift triggers, the mechanism is validated.

### Expected outcome and causal chain

**vs. No verification** — On a case where the model overconfidently predicts a "grasp" action when the object is out of reach, the baseline executes blindly, causing a failed grasp and derailing the task. Our method detects low log-prob (since grasp is unlikely in empty space) and high feature shift (because visual features don't change as expected after a grasp) and triggers re-prompt, leading to a corrective action like moving closer. We expect a noticeable gap in success rate on subtasks involving manipulation under uncertainty, with parity on straightforward navigation segments.

**vs. Log-prob only** — On a case where the model is confidently wrong, e.g., a novel object appearance causes high log-prob for a wrong action, the log-prob-only baseline executes erroneously. Our method additionally detects high feature shift because the model's internal state differs from the expected transition, and triggers re-prompt. Thus, we expect VERA to outperform specifically on instances where the model is confident but semantically wrong, while both methods perform similarly on low-confidence errors.

**vs. MC Dropout** — On a case where MC dropout correctly flags uncertainty that log-prob misses (e.g., due to distribution shift), MC dropout triggers re-prompt while VERA might not. However, VERA's feature shift can catch such cases via temporal inconsistency. On easy instances, both succeed; on hard ones, MC dropout may have slightly higher recall but at higher latency. We expect comparable success rates but VERA with significantly lower average inference time, especially on time-critical steps.

**vs. Learned verification classifier** — On a case where the learned classifier has been trained on similar errors, it may match VERA's accuracy. But VERA's unsupervised heuristic avoids the need for labeled error data. We expect VERA to perform comparably on the primary metric while requiring less offline supervision.

### What would falsify this idea
If VERA's improvement over log-prob-only is uniformly distributed across all failure modes rather than concentrated on cases with high feature shift, then the feature shift mechanism is not working as hypothesized, and the central claim of exploiting joint low confidence and high shift is unsupported.

## References

1. Embodied.cpp: A Portable Inference Runtime of Embodied AI Models on Heterogeneous Robots
