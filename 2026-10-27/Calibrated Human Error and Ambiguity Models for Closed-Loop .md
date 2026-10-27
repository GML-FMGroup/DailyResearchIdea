# Calibrated Human Error and Ambiguity Models for Closed-Loop Evaluation of Embodied Multimodal Assistants

## Motivation

Existing closed-loop embodied assistant benchmarks such as InteractiveWorld evaluate MLLMs against a fixed scripted human policy, so the measured robustness never reflects the stochastic misinterpretations, clarification requests, and execution errors that real users exhibit. DialFRED partially addresses human variability by adding active question-asking, but it relies on an oracle that returns clean answers, structurally removing the noise and ambiguity that dominate real human–assistant interaction. The field therefore lacks a closed-loop environment in which the human's stochastic error-and-ambiguity behavior is calibrated from real user data, making robustness to human variability unmeasurable.

## Key Insight

The distribution of human misinterpretations, clarification requests, and execution errors is a low-dimensional, task-conditioned stochastic process whose parameters can be estimated from a small real user study and then replayed exactly inside the simulator, so closed-loop task success under this distribution is an unbiased estimator of assistant robustness to real human variability.

## Method

**What it is.** We introduce **CHAOS** (Calibrated Human Ambiguity and errOr Simulation), a drop-in replacement for the scripted human policy in InteractiveWorld. Its input is the assistant's natural-language instruction plus the current egocentric scene and dialogue history; its output is a sampled human response drawn from an empirically calibrated stochastic model — a correct action, a misinterpreted action, an execution slip, or a clarification request — together with the resulting scene update. The model has two layers: (i) an *instruction-level ambiguity model* that predicts, from instruction text, scene features, **and assistant-utterance features**, the probability that a real user would ask for clarification before acting; and (ii) an *execution-level error model* that predicts, conditioned on the instruction being acted upon **and on the sampled clarification content (if any)**, the distribution over correct action, misinterpretation, and slip.

**How it works.**

```pseudocode
Input: instruction u, scene s, dialogue history h, assistant utterance a, calibrated parameter vector theta
Output: human response r (action or clarification), updated scene s'

# --- Offline calibration (run once on real user study) ---
1. Collect N=40 participants x 120 InteractiveBench tasks in a Wizard-of-Oz setup.
   For each turn record: (u, s_features, h, a, response_type, executed_action, latency).
2. Encode each turn as feature vector phi = [CLIP(s), SBERT(u), SBERT(a), turn_index, dialogue_length].
3. Fit a two-stage logistic model:
     P(clarify | phi) = sigmoid(w_c . phi)          # ambiguity stage
     P(action | phi, not clarify) = softmax(W_a . phi) # execution stage
   with L2 regularization lambda=0.01, fit by maximum likelihood.
   Additionally fit a stage-2 model conditioned on the sampled clarification content c:
     P(action | phi, c, not clarify) = softmax(W_a' . [phi; SBERT(c)])
   where c is the clarification template text emitted in step 8.
4. Estimate per-action slip probabilities from the executed_action vs. intended_action
   confusion matrix, smoothed with a Dirichlet(alpha=1) prior.
5. Store theta = (w_c, W_a, W_a', slip_matrix) as the calibrated human model.

# --- Online simulation (replaces scripted policy) ---
6. At each turn, compute phi from (u, s, h, a).
7. Sample response_type ~ Categorical([P(clarify), 1-P(clarify)]).
8. If clarify: emit a clarification request drawn from a template distribution
   conditioned on the ambiguous entity; do NOT update scene.
9. Else: sample intended_action ~ Categorical(softmax(W_a . phi));
   sample executed_action ~ slip_matrix[intended_action]; update scene s'.
10. Log (response_type, intended_action, executed_action, clarification_content) for recovery analysis.
```

**Why this design.** We chose a two-stage logistic model over a single end-to-end neural human simulator because the two-stage factorization makes the ambiguity and execution parameters separately identifiable from the same user-study data, accepting the cost that cross-stage correlations (e.g., ambiguity that also biases which misaction occurs) are not captured. We chose maximum-likelihood logistic regression over a deep generative model because with N=40 participants the deep model would overfit and its parameters would not be interpretable as error rates, accepting the cost that nonlinear feature interactions are missed. We chose to sample clarification requests from a template distribution conditioned on the ambiguous entity rather than generating them with an LLM because template sampling guarantees the clarification is grounded in a real scene entity, accepting the cost of reduced linguistic diversity. A domain expert would not describe CHAOS as a variant of DialFRED's questioner–performer framework: DialFRED trains an agent to ask questions and answers them with an oracle, whereas CHAOS is an evaluation-side human model whose parameters are estimated from real user data and whose purpose is to make the *assistant's* robustness measurable, not to train a question-asker.

**Why it measures what we claim.** The clarification probability `P(clarify | phi)` measures the motivation-level concept of *instruction ambiguity as experienced by real users* because the assumption that the user-study participants' clarification behavior is representative of the target population makes the fitted logistic coefficient equivalent to the population ambiguity rate; this assumption fails when the study population is demographically skewed (e.g., all participants are sighted adults), in which case the quantity reflects the ambiguity rate of that subpopulation rather than of users at large. The execution slip matrix `slip_matrix[intended_action]` measures *realistic execution error* because the assumption that the Wizard-of-Oz executed action is the ground-truth intended action makes the confusion matrix an unbiased estimate of the slip distribution; this assumption fails when the wizard corrects the participant's action mid-execution, in which case the matrix undercounts slips. The closed-loop task success under sampled responses measures *assistant robustness to human variability* because the assumption that sampling from the calibrated distribution is equivalent to encountering a random real user makes the success rate an unbiased estimator of expected real-world success; this assumption fails when the assistant's behavior is correlated with the sampled error type (e.g., the assistant learns to exploit a specific clarification template), in which case the success rate overestimates robustness. **Additionally, the template-sampled clarification measures real user clarification behavior because the assumption that assistants cannot distinguish template from free-form clarification (no template-exploitation) holds; this assumption fails when an assistant overfits to template phrasing and inflates success, in which case the measured robustness reflects template exploitation rather than genuine clarification handling.**

## Contribution

['CHAOS, a calibrated two-stage stochastic human model (ambiguity stage + execution stage) that replaces the scripted policy in InteractiveWorld and is estimated from a 40-participant Wizard-of-Oz user study.', 'A closed-loop evaluation protocol that measures assistant robustness as expected task success under the calibrated human distribution, including recovery from clarification and correction, and decomposes failures into ambiguity-induced, misinterpretation-induced, and slip-induced categories.', 'An empirical finding that MLLM assistants which perform well against the scripted human policy exhibit a characteristic failure mode under calibrated human variability: they issue instructions that are unambiguous to the model but ambiguous to the calibrated human, and they fail to recover when a clarification request is issued.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | InteractiveBench (120 tasks) | Matches the environment CHAOS was calibrated for |
| Primary metric | Closed-loop task success rate | Directly measures assistant robustness to human variability |
| Baseline | Scripted human policy (InteractiveWorld default) | Tests whether realism of human model changes rankings |
| Baseline | Oracle human (always correct, never clarifies) | Upper-bound ceiling for the success metric |
| Baseline | DialFRED-style oracle answerer | Tests whether question-asking alone explains gains |
| Baseline | LLM-prompted human simulator (GPT-4 with user-study transcripts) | Tests whether the two-stage factorization is better calibrated than a neural alternative |
| Ablation-of-ours | CHAOS with ambiguity stage frozen (no clarification sampling) | Isolates contribution of the ambiguity layer |
| Ablation-of-ours | CHAOS with LLM-generated clarifications (replacing templates) | Tests template-exploitation assumption |

### Why this setup validates the claim

The central claim is that a calibrated stochastic human model changes which assistants look robust, and that the change is attributable to two separable mechanisms: ambiguity-driven clarification and execution-level error. The scripted-policy baseline tests the first sub-claim directly: if the scripted policy already produces the same assistant ranking as CHAOS, then realism of the human model does not matter and the calibration effort is unjustified. The oracle baseline anchors the metric by showing what success looks like when human variability is removed, so any gap between CHAOS and oracle is attributable to human variability rather than task difficulty. The DialFRED-style oracle answerer tests whether the assistant's apparent robustness is just an artifact of getting clean answers to its questions, which would mean CHAOS's clarification sampling is doing no work. The LLM-prompted human simulator baseline tests whether the two-stage logistic factorization is not merely simpler but actually better-calibrated than the obvious neural alternative, by comparing per-turn log-likelihood and calibration error on held-out user turns. The frozen-ambiguity ablation is the key diagnostic: it removes exactly one mechanism while holding the execution-error layer fixed, so any drop in the CHAOS-vs-ablation gap can be attributed to ambiguity sampling rather than to the presence of stochasticity per se. The LLM-clarification ablation tests the template-exploitation assumption: if rankings shift when templates are replaced with LLM-generated clarifications, then the template distribution was doing work that real user clarification would not. Closed-loop success rate is the right primary metric because the claim is about robustness under realistic human behavior, not about per-turn accuracy; a model that improves per-turn action prediction but not end-to-end success would not support the claim.

### Expected outcome and causal chain

**vs. Scripted human policy** — On a case where the instruction is underspecified (e.g., "put the cup on the shelf" with two shelves in view), the scripted baseline produces a deterministic correct action because its policy ignores ambiguity and always executes the canonical interpretation. Our method instead samples a clarification request at the calibrated rate because the ambiguity stage was fit on real users who asked in exactly this configuration, so we expect a visible gap on the ambiguous-task subset (assistants that fail to handle clarification will drop several points there) but near-parity on unambiguous tasks.

**vs. Oracle human** — On a case where the assistant gives a slightly wrong instruction, the oracle baseline produces the correct action anyway because it never errs or misinterprets, so the assistant's instruction quality is never penalized. Our method instead samples a misinterpretation or slip at the calibrated rate because the execution stage reflects real user error, so we expect the CHAOS success rate to sit noticeably below the oracle ceiling, with the gap concentrated on tasks where instructions are long or contain spatial references.

**vs. DialFRED-style oracle answerer** — On a case where the assistant asks a clarification question, the oracle-answerer baseline returns a clean, correct answer because its answerer is a ground-truth oracle, so the assistant's question-asking is rewarded without testing whether the question was well-posed. Our method instead returns a template-grounded clarification drawn from the calibrated entity distribution because the ambiguity stage conditions on scene features, so we expect assistants that ask vague questions to show lower recovery rates under CHAOS than under the oracle answerer, while assistants that ask entity-grounded questions show parity.

**vs. LLM-prompted human simulator** — On held-out user turns, the LLM-prompted simulator is expected to produce plausible but poorly calibrated responses because it is not fit to the user-study distribution, so we expect CHAOS to achieve lower Brier score and better reliability-diagram calibration for P(clarify|phi) and higher top-1 accuracy for the slip matrix, while the LLM simulator may show higher linguistic diversity but worse calibration.

**vs. CHAOS with ambiguity frozen** — On a case where the instruction is ambiguous, the frozen ablation produces a deterministic action because the clarification branch is disabled, so the assistant never gets a chance to recover from ambiguity. Our method instead samples clarification at the calibrated rate, so we expect the frozen ablation to overestimate assistant robustness on the ambiguous subset specifically, with the two conditions converging on unambiguous tasks — a pattern that would confirm the ambiguity layer, not stochasticity in general, drives the effect.

**vs. CHAOS with LLM-generated clarifications** — On a case where the assistant asks a clarification question, the LLM-generated clarification may be more linguistically diverse but less grounded in the specific ambiguous entity, so we expect rankings to shift if assistants exploit template phrasing; if rankings are stable, the template-exploitation assumption is supported.

### What would falsify this idea

If the CHAOS-vs-scripted gap is uniform across ambiguous and unambiguous task subsets rather than concentrated on the ambiguous subset, then the calibration is not measuring instruction ambiguity as claimed and the two-stage factorization is not doing the work attributed to it. Similarly, if the frozen-ambiguity ablation matches full CHAOS on the ambiguous subset, the ambiguity stage is inert and the central mechanism claim is wrong. Additionally, if the LLM-prompted simulator matches or exceeds CHAOS on held-out calibration metrics, the two-stage factorization is not better-calibrated than the neural alternative, and if rankings shift substantially under LLM-generated clarifications, the template-exploitation assumption fails.

## References

1. InteractiveWorld: A Verifiable Human-Centric Environment for Building Better Interactive MLLM Assistants
2. DialFRED: Dialogue-Enabled Agents for Embodied Instruction Following
