# Spatial-Temporal Act2Answer: Evaluating VLA Knowledge Retention Under Dynamic Spatial Configurations

## Motivation

Existing VLA evaluations, such as Act2Answer from 'Does VLA Even Know the Basics?', assess knowledge retention using static tabletop tasks with fixed object positions. This fails to capture whether VLAs can retain knowledge when spatial configurations vary over time, a critical capability for real-world deployment where environments change dynamically. Without spatial-temporal grounding, models may rely on location shortcuts rather than semantic understanding, leading to brittle performance.

## Key Insight

The consistency of a VLA's action across spatially varying temporal sequences isolates whether its knowledge is grounded in semantic object properties rather than spatial position.

## Method

**Spatial-Temporal Act2Answer (STA2A)**

(A) **What it is:** STA2A is an evaluation protocol that extends Act2Answer by requiring a VLA to answer a question via an action sequence over multiple timesteps, each with a different spatial layout of objects. The input is a question Q and a temporal sequence of T tabletop configurations {C_1,...,C_T}, each placing objects at different positions. The output is the action-grounded success rate and a temporal consistency metric.

(B) **How it works:**
```python
# Pseudocode for STA2A evaluation
Input: VLA model M, question Q, answer options {a_i}, 
       spatial-temporal conditions: static, spatial-only, temporal-only, spatiotemporal
       For each condition, generate T configurations {C_t} where C_t defines positions of objects.

for condition in conditions:
    for episode in N_questions:
        answer = correct_answer_label
        action_history = []
        for t in 1..T:
            # Set up environment with configuration C_t
            observation = setup_tabletop(C_t)
            # VLA performs action (pointing to a location) based on observation and history
            action_t = M(point_action, observation, action_history)
            action_history.append(action_t)
        # Final action action_T should point to location corresponding to answer
        success = (decode_action_position(action_T) == answer)
        record success
        # Layerwise probing: extract hidden states at each timestep
        store_probes(M.hidden_states)
    Compute success rate per condition.
```
**Assumption verification:** We pre-filter questions to exclude those where the answer could depend on spatial relations (e.g., 'What is left of X?') using a rule-based filter based on question templates (e.g., avoid 'left of', 'right of', 'behind', 'in front of'). Additionally, for each question, we validate answer invariance via a human annotation batch: 50 random questions are shown to 3 annotators who verify that the answer is independent of object positions; any question with disagreement is discarded. This ensures that changes in action across timesteps reflect temporal inconsistency, not a change in correct answer.

Hyperparameters: T=5 timesteps, spatial perturbation radius=10cm (uniform random within area), condition types as above.

We introduce two additional metrics computed per episode:
- **Position-based action consistency:** Compute the Pearson correlation between the sequence of chosen action positions and the sequence of the correct answer's object position (which should be invariant). High correlation (r>0.8) indicates the model is using position shortcuts rather than semantic grounding.
- **Semantic grounding consistency:** Compute the correlation between action trajectories and object identity changes across timesteps. If the model changes action when a distractor object moves but the target object stays, it indicates sensitivity to irrelevant changes, i.e., low grounding consistency.

(C) **Why this design:** We chose a sequential action requirement over a single action to force the VLA to maintain temporal consistency, as opposed to static Act2Answer which only tests a snapshot. The spatial perturbations are designed to be semantically irrelevant (e.g., moving objects to arbitrary but distinct locations) to isolate whether the model relies on position. A core assumption of this design is that the correct answer remains invariant under these perturbations, which we verify via the pre-filtering and human annotation. We opted for pointing actions (mapping to answer options) rather than full object manipulation to reduce control confounds, accepting that this simpler action space may not generalize to all manipulation tasks. The layerwise probing enables identifying where spatial-temporal information is lost, a feature absent in Act2Answer. We kept the number of timesteps small (T=5) to avoid excessive episode length, balancing duration with the ability to measure temporal drift.

(D) **Why it measures what we claim:** The **success rate** under the spatiotemporal condition measures the VLA's spatial-temporal grounding of knowledge, because we assume the correct answer is invariant under the applied spatial perturbations; this assumption fails when the question involves spatial relations (e.g., "What is left of X?"), in which case the metric reflects spatial reasoning ability rather than knowledge retention — we mitigate this via our pre-filtering. The **temporal consistency** across timesteps (e.g., whether action_t changes) measures the model's stability of knowledge representation over time, assuming that a grounded model should maintain consistent answer; this assumption fails when the model adapts to new evidence (not present here), in which case variability indicates uncertainty. The **layerwise probe correlation** with answer correctness measures where in the network spatial-temporal information is encoded, under the assumption that linear probes can recover answer-relevant features; this assumption fails when features are nonlinearly distributed, in which case probes may underestimate knowledge presence. The **position-based action consistency** measures the extent to which the model relies on location shortcuts (high correlation = shortcut usage). The **semantic grounding consistency** measures whether action changes are aligned with object identity changes (low correlation = poor grounding).

## Contribution

(1) A novel evaluation protocol, Spatial-Temporal Act2Answer, that extends existing knowledge retention tests for VLAs to incorporate spatial and temporal variations, enabling systematic assessment of spatial-temporal grounding. (2) Empirical finding that VLAs show significant performance drops under spatial-temporal conditions compared to static ones, revealing a structural gap in their ability to maintain knowledge across changing layouts. (3) A publicly available benchmark dataset comprising 500+ questions from commonsense and world knowledge categories, each with multiple spatial-temporal episode configurations.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
|------|--------|-----------|
| Dataset | Custom spatiotemporal tabletop QA | Controls spatial-temporal conditions; 500 questions, each with 5 timesteps, objects randomly placed within 30x30cm area. |
| Primary metric | Spatiotemporal success rate + Position-based consistency + Semantic grounding consistency | Three metrics capture grounding, shortcut usage, and sensitivity to irrelevant changes. |
| Baseline 1 | Act2Answer (static) | Tests without temporal dimension; single timestep, same config. |
| Baseline 2 | VLM VQA (LLaVA) | Tests pure language grounding; output is text, not action. |
| Baseline 3 | Random pointing (chance) | Lower bound for performance; uniform among 4 options. |
| Ablation of ours | STA2A with T=1 (single timestep) | Isolates temporal consistency effect. |
| Control condition | Semantic-movement control: objects move such that correct answer changes (e.g., swap positions of two objects) | Validates that temporal consistency metric captures grounding vs. adaptation; we compare performance when answer changes vs. invariant. |
| Models tested | RT-2 (55B), Octo (100M), OpenVLA (7B), and a smaller VLA (300M) | Tests scale effects and generalizability. |
| Open-source | All code and dataset generation scripts released at [anonymous repo]. | Lowers adoption barrier. |

### Why this setup validates the claim

The dataset systematically varies spatial and temporal conditions across four conditions (static, spatial-only, temporal-only, spatiotemporal), enabling isolation of the VLA's ability to maintain knowledge under irrelevant perturbations. The control condition (semantic-movement) tests whether the model can adapt when the answer actually changes, distinguishing between grounded adaptation and positional shortcutting. Comparing against Act2Answer (static) isolates the temporal dimension, while VLM VQA reveals the gap between language-only and action-grounded reasoning. Random pointing provides a floor. The spatiotemporal success rate directly measures whether the model can consistently ground knowledge across changing configurations, forming a falsifiable test: if our method performs no better on spatiotemporal trials than static trials, the claim that temporal consistency matters is falsified. The additional metrics (position-based consistency, semantic grounding consistency) provide finer-grained diagnostics. Testing on diverse architectures (RT-2, Octo, OpenVLA) demonstrates the prevalence of failures.

### Expected outcome and causal chain

**vs. Act2Answer (static)** — On a case where the same question (e.g., "What is the color of the apple?") is asked across 5 timesteps with the apple placed at different locations, the static method sees only the first configuration and thus cannot detect temporal drift; it maintains a fixed answer regardless of later observations, leading to inconsistent pointing when the answer location changes. Our method tracks action decisions over time, so it can adapt to each new configuration (spatial perturbation) while maintaining semantic answer, leading to higher success on spatiotemporal trials. We expect a noticeable gap (e.g., 20%+ higher) on spatiotemporal condition but parity on static single-timestep trials.

**vs. VLM VQA (LLaVA)** — On a case requiring pointing to the object containing milk (e.g., "Which cup has milk?"), the VLM answers with text ("the blue cup") but fails when asked to output a pointing action because it lacks action head and spatial reasoning for the specific table layout. Our method uses a VLA trained to emit pointing actions conditioned on visual input, so it can map the semantic answer to the correct spatial location. We expect our method to achieve high success on all conditions, while VLM VQA will achieve zero under the pointing-based metric (since it cannot produce actions), highlighting the necessity of action grounding.

**vs. Random pointing (chance)** — On any trial, random pointing chooses a location uniformly among answer options, achieving 1/N accuracy (e.g., 25% for 4 options). Our method should significantly exceed this on all conditions, particularly on the static condition where no temporal confound exists. If our method is near chance on spatiotemporal condition, it would indicate that the VLA cannot handle spatial-temporal variations at all, falsifying the claim of robust grounding.

**vs. Semantic-movement control** — When object movements change the correct answer (e.g., two objects swap places), a grounded model should adapt its action to point to the new correct location. Our temporal consistency metric will show increased variability, but the success rate will remain high if the model adapts correctly. If the model fails to adapt (low success rate) or continues pointing to the old location (high position-based consistency), it indicates reliance on positional shortcuts rather than semantic grounding.

### What would falsify this idea

If our method's success rate on spatiotemporal trials is similar to that on static trials (within 5%), then the claim that temporal consistency is a distinct challenge would be unsupported. Alternatively, if the gain over Act2Answer is uniform across all subsets rather than concentrated on spatiotemporal trials, the specific spatial-temporal failure mode is not isolated. Also, if the position-based consistency metric is high (r>0.8) on spatiotemporal trials, it would indicate that models rely on position shortcuts, contradicting the claim of semantic grounding.

## References

1. Does VLA Even Know the Basics? Measuring Commonsense and World Knowledge Retention in Vision-Language-Action Models
2. Cosmos-Reason1: From Physical Common Sense To Embodied Reasoning
3. The Llama 3 Herd of Models
4. HybridFlow: A Flexible and Efficient RLHF Framework
5. Expanding Performance Boundaries of Open-Source Multimodal Models with Model, Data, and Test-Time Scaling
6. Robotic Control via Embodied Chain-of-Thought Reasoning
7. NVLM: Open Frontier-Class Multimodal LLMs
8. BridgeData V2: A Dataset for Robot Learning at Scale
