# Deadline-Aware Conversational Infill: Bounded-Latency Knowledge Distillation for Responsive Voice Agents

## Motivation

Existing inference-time knowledge distillation methods like ConvFill assume real-time feasibility without explicit latency bounds, causing stalls when reasoner inference varies. This structural gap prevents deployment in latency-critical applications because the talker cannot guarantee timely output. A bounded-latency coordination protocol is needed to enforce token-level deadlines for reasoner knowledge injection, adapting talker generation complexity to available time budgets.

## Key Insight

By modeling talker generation as a real-time system with predictable latency distributions, we can precompute optimal decision thresholds for waiting versus generating locally, ensuring fluency without sacrificing knowledge integration.

## Method

(A) **What it is**: Deadline-Aware Conversational Infill (DACI) is a coordination protocol between a talker (fast, local model) and a reasoner (powerful, slower model) that enforces token-level deadlines. The talker uses a hierarchical decoder with multiple detail levels, and a deadline scheduler decides at each step whether to wait for reasoner knowledge or proceed with the current best output.

(B) **How it works**:
```pseudocode
Input: user query Q, deadline per token D_t
Output: response tokens

Initialization:
  talker_state = initial hidden state from Q
  reasoner_input = Q
  time_remaining = D_t
  generation_level = 0  // 0: fast, 1: detailed

While not EOS:
  start_time = current time
  // Phase 1: Reasoner inference (asynchronous)
  if reasoner not busy:
    spawn reasoner_inference(reasoner_input)  // returns knowledge chunk update
  
  // Phase 2: Talker decides to wait or generate
  expected_reasoner_time = estimate_reasoner_latency()
  if time_remaining > expected_reasoner_time + margin:
    wait_for_reasoner_update()
    update talker_state with reasoner knowledge
    generation_level = 1
  else:
    // Generate locally without waiting
    token = local_generate(talker_state, level=generation_level)
  
  // Phase 3: Output token and update budget
  output(token)
  elapsed = current_time - start_time
  time_remaining -= elapsed
  time_remaining = min(time_remaining + D_t, D_t)  // sliding window
  generation_level = max(0, 1 - (expected_reasoner_time / D_t))  // adapt
end

Hyperparameters: D_t (token deadline, e.g., 50ms), margin (e.g., 10ms), estimate_reasoner_latency() uses a moving average of historical inference times.
```

(C) **Why this design**: We chose a token-level deadline over an utterance-level deadline because token-level granularity allows the talker to adapt after each output, preventing accumulation of delay. We chose a fixed threshold rule based on expected reasoner latency rather than an adaptive reinforcement learning policy because the decision space is low-dimensional and the threshold can be analytically derived from latency distributions, avoiding training instability. We chose a hierarchical decoder with only two complexity levels (fast vs. detailed) instead of a continuous adaptation because binary switching is simpler to implement and sufficient to cover the main trade-off: waiting incurs risk of deadline miss, while local generation sacrifices knowledge quality. The trade-off is that a fixed threshold may be suboptimal under distribution shift, and two levels may limit fine-grained control, but the simplicity enables real-time guarantees.

(D) **Why it measures what we claim**: The wait-decision threshold (time_remaining > expected_reasoner_time + margin) measures latency budget enforceability because it directly controls whether the talker can wait for reasoner knowledge without exceeding the token deadline; this assumption relies on the estimator being accurate, and fails when reasoner latency spikes unpredictably, in which case the talker may incorrectly wait or skip updates. The generation level adjustment (1 - expected_reasoner_time / D_t) measures dynamic complexity adaptation because it reduces detail level when latency budget is tight, ensuring faster generation; this assumes a linear relationship between detail level and generation time, which fails when the model's computation is not monotonic in detail, in which case the adaptation may not achieve the intended speedup. The token-level sliding window (time_remaining = min(time_remaining + D_t, D_t)) measures bounded latency per token because it resets the budget each token, preventing carryover delays; this assumes independence of token generation times, which fails under bursty processing, but the max operation caps budget to D_t, ensuring no single token consumes more than D_t.

## Contribution

(1) A deadline-aware coordination protocol for inference-time knowledge distillation that enforces token-level latency bounds. (2) A method for dynamic talker complexity adjustment based on time budgets, using a hierarchical decoder with binary detail levels. (3) A token-level deadline scheduling mechanism with analytical threshold determination.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Synthetic conversational infill (Thinking While Speaking) | Covers diverse domains with ground truth reasoner output. |
| Primary metric | Deadline-constrained accuracy | Measures both speed and quality under token deadlines. |
| Baseline 1 | Talker-only (no reasoner) | Tests lower bound without knowledge transfer. |
| Baseline 2 | Reasoner-only (frontier model) | Tests upper bound ignoring deadlines. |
| Baseline 3 | Always wait (ignore deadline) | Tests cost of waiting without adaptation. |
| Ablation of ours | No deadline adaptation (fixed generation level) | Isolates benefit of adaptive complexity. |

### Why this setup validates the claim
This experimental design directly tests the central claim that DACI enables responsive and intelligent conversation by coordinating a fast talker and slow reasoner under token-level deadlines. The dataset provides controlled inputs with known reasoner outputs, enabling precise measurement of accuracy degradation due to deadline pressure. The talker-only baseline establishes the quality floor when no knowledge transfer occurs, while the reasoner-only baseline shows the quality ceiling but without responsiveness. The always-wait baseline demonstrates the failure mode of ignoring deadlines entirely, highlighting the need for adaptation. Our ablation, which removes deadline-aware generation level adjustment, isolates the contribution of dynamic complexity adaptation. The primary metric, deadline-constrained accuracy, is chosen because it jointly captures both aspects: it penalizes tokens that miss the deadline (each treated as a failure) and rewards accuracy relative to the reasoner's output. This metric is sensitive to the predicted effect: DACI should maintain high accuracy while meeting deadlines, whereas baselines will either sacrifice accuracy (talker-only) or miss deadlines (reasoner-only, always-wait). The ablation should show lower accuracy under tight deadlines compared to full DACI, confirming the adaptation mechanism's role.

### Expected outcome and causal chain

**vs. Talker-only (no reasoner)** — On a case where the reasoner would provide crucial knowledge (e.g., resolving a pronoun reference), the talker-only model produces a generic or incorrect token because it lacks access to the reasoner's context understanding. Our method instead waits for the reasoner update when the deadline budget allows, so it incorporates the correct knowledge and outputs the right token. We expect a noticeable gap (e.g., 15-30% higher deadline-constrained accuracy on reasoning-heavy subsets) but parity on simple factual queries where local generation suffices.

**vs. Reasoner-only (frontier model)** — On every token, the reasoner-only model prioritizes quality over speed; but under a tight token deadline (e.g., 50ms), it inevitably exceeds the deadline on longer tokens or complex reasoning steps, causing many missed deadlines. Our method's talker generates locally when the reasoner is too slow, meeting the deadline for those tokens while still following the reasoner's general plan. We thus expect reasoner-only to have high raw accuracy but low deadline-constrained accuracy (e.g., <50%) compared to DACI (e.g., >80%) under strict deadlines, with the gap narrowing when deadlines are relaxed.

**vs. Always wait (ignore deadlines)** — On a sequence of tokens with varying reasoner latency (e.g., sudden spike due to model load), the always-wait baseline waits for every token, accumulating delay and eventually violating the sliding-window deadline on subsequent tokens, leading to a cascade of misses. Our method's deadline scheduler skips waiting when time remaining is insufficient, generating a locally plausible token instead and resetting the budget. Thus, we expect always-wait to show high accuracy on early tokens but steep degradation after latency spikes, whereas DACI maintains stable accuracy. The observable signal is a higher variance in token-by-token performance for always-wait.

### What would falsify this idea
If our method shows no improvement in deadline-constrained accuracy over the talker-only baseline under all deadline conditions, then the knowledge transfer adaptation is ineffective. Alternatively, if the advantage of DACI over always-wait is uniform across all latency distributions (rather than concentrated in high-variance settings), the deadline scheduler is not correctly identifying when to wait.

## References

1. Thinking While Speaking: Inference-Time Knowledge Transfer for Responsive and Intelligent Conversational Voice Agents
2. Agents Thinking Fast and Slow: A Talker-Reasoner Architecture
3. Reasoning with Language Model is Planning with World Model
4. DayDreamer: World Models for Physical Robot Learning
5. Inner Monologue: Embodied Reasoning through Planning with Language Models
6. ProgPrompt: Generating Situated Robot Task Plans using Large Language Models
7. Language Models Are Greedy Reasoners: A Systematic Formal Analysis of Chain-of-Thought
