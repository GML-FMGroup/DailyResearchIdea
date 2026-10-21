# Noise-Propagating Compositional QA over Visual Primitives

## Motivation

Existing low-light VQA benchmarks, such as DarkQA, generate only single-hop questions over visual primitives and thus cannot test whether vision-language models compose noisy perceptual evidence across reasoning steps. This structural gap persists because prior generators lack a mechanism to build multi-hop reasoning graphs and propagate sensor-noise uncertainty through them, so compositional generalization under noise remains unmeasured.

## Key Insight

A question's answer is uncertain exactly to the degree that the sensor-noise posterior over its constituent visual primitives makes the reasoning graph's output variable, which is captured without retraining by pushing noise-perturbed primitive distributions through the graph's deterministic operations.

## Method

**What it is** — NPC-QA (Noise-Propagating Compositional QA) is a generator-plus-validator pipeline. Given a clean indoor image and a simulated low-light RAW noise model, it outputs multi-hop yes/no questions whose answers are validated with a calibrated uncertainty score derived from noise-perturbed primitive posteriors.

**How it works** —
```pseudocode
# Input: clean image I, noise model N(σ) from DarkQA RAW pipeline
# Output: multi-hop QA pairs with noise-aware validity scores

1. PRIMITIVE EXTRACTION
   Detect objects O = {o_i} with attributes (color, size, material)
   For each o_i and attribute a, estimate posterior p(a|o_i, I) via
   repeated forward passes under K=20 RAW-noise perturbations I_k ~ N(σ)
   → primitive distribution P_i^a
   # Calibration: fit temperature scaling on a held-out set of 512 clean images
   # (split from DarkQA train) to calibrate the empirical distribution P_i^a;
   # temperature T is chosen to minimize negative log-likelihood of true attributes.

2. REASONING GRAPH CONSTRUCTION
   Sample a graph template G from a fixed library of 12 templates
   (3 per relation type: REL, SIZE, COUNT, EXIST) with depth d∈{2,3}
   Instantiate nodes with objects/attributes; edges = spatial/logical relations
   (e.g., left-of, larger-than, same-color)
   → deterministic function f_G mapping primitive values to a boolean answer

3. NOISE PROPAGATION
   For each graph G, compute answer distribution q_G(y) by Monte-Carlo:
     sample primitive values from {P_i^a}, evaluate f_G, repeat M=1000 times
   → q_G(True), q_G(False)
   # Joint sampling: draw a shared illumination field and spatially correlated
   # RAW noise (as in DarkQA) across all objects in the scene, then sample
   # primitive values conditionally on the joint noise realization.

4. NOISE-AWARE VALIDATION
   Accept question iff entropy H(q_G) ≤ τ_ent (τ_ent=0.3) AND
   margin |q_G(True)-0.5| ≥ τ_marg (τ_marg=0.2)
   Attach q_G as ground-truth uncertainty label

5. EVALUATION PROTOCOL
   For each accepted (question, q_G), query VLM on clean and low-light images
   Compare VLM confidence to q_G via Brier score and ECE
```
Each graph template is a fixed Boolean circuit over primitive predicates; f_G is deterministic, so all uncertainty in q_G comes from the primitive posteriors P_i^a. The Monte-Carlo sample size M=1000 is chosen so that the standard error of q_G is below 0.02, and K=20 perturbations balance posterior fidelity against generation cost. The primitive posteriors are calibrated via temperature scaling on a held-out set of 512 images, and the joint sampling in step 3 accounts for spatially correlated noise by drawing shared illumination and correlated RAW noise fields across objects.

**Why this design** — We chose to propagate noise through a deterministic graph rather than train a learned uncertainty head because the graph's Boolean structure gives an exact, assumption-light mapping from primitive noise to answer uncertainty, accepting the cost that graph templates must be hand-specified and may not cover all compositional forms. We chose Monte-Carlo sampling over analytic propagation because the predicates include non-linear spatial relations (e.g., left-of under noisy bounding boxes) where closed-form convolution is intractable, accepting higher compute per question (M=1000 evaluations) in exchange for a consistent estimator. We chose to accept only low-entropy questions (τ_ent=0.3) rather than all generated questions because high-entropy items have answers dominated by noise and would make VLM evaluation uninformative, accepting a reduced yield of valid QA pairs. This design is not a variant of Toolformer-style retrieval or of a learned router: there is no library lookup and no controller deciding between modules; the uncertainty is computed by exact propagation through a known circuit, a property no retrieval or routing baseline possesses.

**Why it measures what we claim** — The Monte-Carlo answer entropy H(q_G) measures the compositional uncertainty induced by sensor noise because the assumption that the primitive posteriors P_i^a are calibrated and independent across objects makes the push-forward distribution q_G equal to the true answer distribution under the noise model; this assumption fails when noise correlations across nearby objects (e.g., shared illumination) are strong, in which case H(q_G) underestimates the true uncertainty. The acceptance margin |q_G(True)-0.5| measures question determinacy because the assumption that f_G is a deterministic Boolean circuit makes the margin equal to the distance from a coin-flip answer; this fails when the graph contains a near-tie predicate (e.g., sizes within noise), in which case the margin reflects predicate ambiguity rather than noise. The Brier score between VLM confidence and q_G measures the VLM's noise-awareness because the assumption that q_G is the Bayes-optimal answer distribution makes the Brier score a proper scoring rule for calibration; this fails when the VLM's errors are systematic (e.g., bias toward 'yes'), in which case the score reflects bias rather than miscalibration.

## Contribution

['A compositional question generator that constructs multi-hop reasoning graphs over visual primitives and derives exact answer-uncertainty labels by Monte-Carlo propagation of RAW-domain sensor-noise posteriors through deterministic Boolean circuits.', 'A noise-aware answer validator that accepts only questions whose propagated answer entropy and margin exceed thresholds, yielding a benchmark of multi-hop low-light QA pairs with calibrated uncertainty ground truth.', 'An evaluation protocol that scores VLM confidence against the propagated uncertainty via Brier score and ECE, enabling direct measurement of compositional noise-awareness in vision-language models.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | DarkQA low-light indoor QA pairs | Controlled RAW noise with ground-truth primitives |
| Primary metric | Brier score vs. q_G on accepted questions | Proper scoring rule for noise-aware calibration |
| Secondary metric | Reliability/resolution decomposition of Brier score | Separates bias from miscalibration |
| Baseline | VLM without LLIE preprocessing | Tests raw noise sensitivity |
| Baseline | VLM with LLIE preprocessing | Tests whether denoising alone fixes uncertainty |
| Baseline | Chain-of-thought prompting | Tests linguistic reasoning without noise propagation |
| Baseline | Learned Bayesian VLM uncertainty head | Tests whether learned uncertainty matches exact propagation |
| Baseline | Probabilistic soft-logic generator | Tests whether soft-logic propagation matches exact circuit |
| Ablation-of-ours | Uniform q_G (drop noise propagation) | Isolates value of propagated uncertainty labels |
| Ablation-of-ours | Independent per-object sampling (drop joint sampling) | Tests effect of correlated noise modeling |
| Ablation-of-ours | Uncalibrated primitive posteriors (drop temperature scaling) | Tests effect of calibration |

### Why this setup validates the claim

The central claim is that q_G, computed by exact Monte-Carlo propagation through a known Boolean circuit, is a calibrated ground-truth answer distribution under sensor noise, and that VLMs fail to match it in a noise-structured way. DarkQA is the right dataset because it supplies both clean and RAW-noise low-light versions with primitive-level annotations, so the same question can be posed under controlled noise levels and q_G can be computed from actual posteriors rather than assumed ones. The Brier score is the right metric because it is a proper scoring rule: it rewards a VLM only if its confidence equals the true answer probability, which is exactly what q_G claims to be. The three baselines partition the sub-claims: no-LLIE tests raw noise sensitivity, LLIE tests whether preprocessing alone restores calibration, and chain-of-thought tests whether linguistic reasoning substitutes for probabilistic noise propagation. The uniform-q_G ablation is the key falsifier: if replacing our propagated labels with a constant 0.5 does not worsen the diagnostic, then the propagation machinery adds nothing and the claim collapses. The additional baselines (learned Bayesian VLM uncertainty head and probabilistic soft-logic generator) test whether exact circuit propagation outperforms learned or approximate uncertainty propagation on the same accepted question set. The joint-sampling and calibration ablations directly test the load-bearing assumption that primitive posteriors are calibrated and independent; if removing joint sampling or calibration does not degrade the Brier score, the assumption is not critical, but if it does, the necessity of these components is confirmed. The reliability/resolution decomposition further separates whether VLM errors stem from systematic bias (reliability) or from poor discrimination (resolution).

### Expected outcome and causal chain

**vs. VLM without LLIE preprocessing** — On a case where two objects are close in size and the RAW noise inflates bounding-box jitter, the no-LLIE baseline answers "yes" to a size-comparison question with high confidence because its primitive detector treats a noisy edge as a stable boundary, ignoring the perturbation distribution entirely. Our method instead assigns q_G near 0.5 because the Monte-Carlo push-forward shows the size predicate flips under plausible noise draws, so the accepted question is either filtered out or labeled low-confidence. We expect the baseline to show a large Brier gap concentrated on size- and count-subset questions under high σ, with near-parity on existence questions where noise does not flip the predicate.

**vs. VLM with LLIE preprocessing** — On a case where denoising smooths a shared-illumination region, the LLIE baseline confidently answers a same-color multi-hop question because preprocessing removes visible noise but also erases the correlation structure that makes the answer genuinely uncertain; its confidence reflects post-denoising crispness, not the underlying answer distribution. Our method instead propagates the residual primitive posteriors through the color-equality circuit and reports q_G reflecting the true flip probability. We expect LLIE to improve raw accuracy over no-LLIE but to leave a persistent Brier gap on compositional (depth-3) questions, since denoising cannot recover information the noise destroyed, while matching us on depth-2 questions.

**vs. Chain-of-thought prompting** — On a case where a depth-3 graph chains left-of, larger-than, and same-color, the CoT baseline verbalizes each hop confidently and compounds errors because each linguistic step commits to a hard primitive value with no uncertainty carry-over; the final answer is overconfident exactly where the circuit has multiple near-tie predicates. Our method instead evaluates the full circuit under sampled primitives, so near-ties in early hops attenuate the final answer distribution rather than being silently resolved. We expect CoT to show worse Brier scores than plain QA specifically on depth-3 graphs with at least two near-tie predicates, and parity elsewhere, which is the signature of missing uncertainty propagation rather than missing reasoning.

**vs. Learned Bayesian VLM uncertainty head** — On a case where the VLM's uncertainty head is trained on clean images but tested on noisy ones, the Bayesian head may produce overconfident predictions because it has not seen the noise distribution during training, whereas our method computes q_G directly from the noise model without retraining. We expect the Bayesian head to underperform our method on high-σ images, with the gap concentrated on depth-3 questions where uncertainty compounds.

**vs. Probabilistic soft-logic generator** — On a case where the soft-logic generator approximates the Boolean circuit with continuous relaxations, it may smooth over near-tie predicates and produce a q_G that is less sharp than the exact Monte-Carlo estimate. We expect our method to yield lower Brier scores on questions with multiple near-tie predicates, as the exact propagation captures the discrete flips that soft-logic misses.

**vs. Ablation: Uniform q_G** — Replacing q_G with a constant 0.5 removes all noise-propagated uncertainty. We expect the Brier score to worsen uniformly across all question types, confirming that the propagated labels carry information beyond a trivial baseline.

**vs. Ablation: Independent per-object sampling** — Dropping joint sampling ignores correlated noise, leading to underestimated uncertainty (lower entropy) for nearby objects. We expect the Brier score to be worse on questions involving spatially proximate objects, especially under high noise correlation, validating the need for joint sampling.

**vs. Ablation: Uncalibrated primitive posteriors** — Without temperature scaling, the primitive posteriors may be miscalibrated, propagating bias into q_G. We expect the Brier score to degrade, particularly on attributes with skewed empirical distributions, confirming the importance of calibration.

### What would falsify this idea

If the Brier advantage of our q_G labels over the uniform-q_G ablation is uniform across depth-2 and depth-3 questions and across noise levels rather than concentrated on the compositional, near-tie subsets where the causal chain predicts it, then q_G is not capturing noise-induced compositional uncertainty and the central claim is wrong. Likewise, if LLIE preprocessing closes the gap to our method entirely, then the propagation machinery is unnecessary and simple denoising suffices.

## References

1. DarkQA: Benchmarking Vision-Language Models on Visual-Primitive Question Answering in Low-Light Indoor Scenes
2. Improved Baselines with Visual Instruction Tuning
3. Thinking in Space: How Multimodal Large Language Models See, Remember, and Recall Spaces
