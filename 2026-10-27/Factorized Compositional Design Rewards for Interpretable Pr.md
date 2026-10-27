# Factorized Compositional Design Rewards for Interpretable Prompt Enhancement

## Motivation

Existing prompt enhancement methods such as Qwen-Image-PE rely on a monolithic scalar Groupwise Design Reward that collapses multiple design sub-attributes (layout, color, typography, composition) into a single preference score, inheriting HPSv3's paradigm without any verified decomposition. This structural collapse makes it impossible to attribute a low reward to a specific design weakness, so targeted prompt enhancement cannot be performed without risking degradation in other attributes. The problem is meta-level: across BeautifulPrompt and HPSv3 branches, no method establishes compositionality across design sub-attributes, leaving prompt enhancement opaque and uncontrollable.

## Key Insight

A reward model whose aggregate score is constrained to equal a learned composition of independently predicted sub-attribute scores yields a verified decomposition that is identifiable from pairwise preference data, because the consistency loss forces the multi-head outputs to be sufficient statistics for the aggregate reward.

## Method

**What it is** — We introduce FACTOR-PE (Factorized Compositional Design Reward for Prompt Enhancement), a prompt enhancement framework that replaces the scalar Groupwise Design Reward with a multi-head reward model whose aggregate score is verified to equal a learned composition of sub-attribute scores. Inputs: a user request and a candidate enhanced prompt; outputs: a targeted prompt edit that improves a specific design sub-attribute (layout, color, typography, composition) without degrading the others.

**Load-bearing assumption (stated explicitly)** — The four attribute heads are identifiable and disentangled from pairwise preference data alone, so that each head's score reflects its own design sub-attribute rather than a shared confound, making the max-min selection and 'improve {weakest_attr}' edit causally valid. This assumption is enforced by the attribute-specific supervision term L_attr (defined below) and verified by the permutation-identification test (defined below); without them, L_pref and L_cons are invariant to relabeling the four heads and the decomposition is ungrounded.

**How it works** — We train a multi-head reward model R_φ with four attribute heads (layout, color, typography, composition) and a composition head g_ψ that maps the four scores to an aggregate. The training loss has three terms: (1) a standard pairwise preference loss on the aggregate score, (2) a consistency loss that penalizes the squared difference between the aggregate head's output and g_ψ applied to the sub-attribute scores, with weight λ=0.5, and (3) an attribute-specific supervision loss L_attr that uses attribute-level pairwise labels (for each pair, which image is better on layout, on color, on typography, on composition) to train each head directly. After training, for a given request we generate K=8 candidate enhanced prompts, score each with the four heads, and select the candidate whose weakest sub-attribute score is maximized (max-min selection). The selected prompt's sub-attribute scores are then used to generate a targeted edit instruction (e.g., 'improve color harmony') that is fed back to the prompt enhancer for one refinement step. We verify identifiability with a permutation-identification test: on a held-out set of attribute-labeled pairs, for each head h we compute the accuracy of predicting the attribute label for attribute a; the head is identified if the diagonal accuracy (head h predicting attribute h) exceeds the off-diagonal accuracies by a margin of at least 0.1.

```pseudocode
# Training FACTOR-PE
for each batch of (request, image_pair, preference_label, attr_labels):
    # image_pair = (image from enhanced prompt A, image from enhanced prompt B)
    # preference_label ∈ {0,1} for aggregate preference
    # attr_labels = {layout: 0/1, color: 0/1, typography: 0/1, composition: 0/1}
    s_A = [R_φ^layout(A), R_φ^color(A), R_φ^typography(A), R_φ^composition(A)]
    s_B = [R_φ^layout(B), R_φ^color(B), R_φ^typography(B), R_φ^composition(B)]
    agg_A = g_ψ(s_A)   # composition head
    agg_B = g_ψ(s_B)
    L_pref = -log σ(agg_A - agg_B) * preference_label  # pairwise preference
    L_cons = (agg_A - R_φ^agg(A))^2 + (agg_B - R_φ^agg(B))^2  # consistency
    L_attr = sum_{a in ATTRS} -log σ( R_φ^a(A) - R_φ^a(B) ) * attr_labels[a]  # attribute-specific supervision
    L = L_pref + λ * L_cons + μ * L_attr  # λ=0.5, μ=1.0
    update φ, ψ

# Inference: targeted enhancement
for a user request:
    candidates = [enhance(request) for _ in range(K=8)]
    scores = [[R_φ^attr(c) for attr in ATTRS] for c in candidates]
    best = argmax_c min(scores[c])  # max-min selection
    weakest_attr = argmin(scores[best])
    edit_instruction = f"improve {weakest_attr}"
    refined_prompt = enhance(request + " " + edit_instruction)
    return refined_prompt
```

**Why this design** — We chose a multi-head reward with a learned composition head over a single scalar reward because the scalar reward collapses all attributes and provides no attribution, accepting the cost that the composition head must be trained jointly and may underfit if attributes are strongly correlated. We chose a consistency loss over post-hoc decomposition because post-hoc methods (e.g., linear probing of a scalar reward) do not guarantee that the decomposed scores are sufficient statistics for the aggregate, accepting the cost that the consistency loss adds a hyperparameter λ. We chose max-min selection over weighted-sum selection because max-min directly targets the weakest attribute, avoiding the risk that a high score in one attribute masks a low score in another, accepting the cost that max-min may be overly conservative when attributes are imbalanced. We chose a single refinement step over iterative refinement because iterative refinement can drift the prompt away from the original request, accepting the cost that one step may not fully correct severe weaknesses. Unlike HPSv3, which uses a monolithic scalar reward, FACTOR-PE enforces a verified decomposition through the consistency loss and attribute-specific supervision, making the sub-attribute scores identifiable and usable for targeted enhancement.

**Why it measures what we claim** — The consistency loss L_cons measures the compositionality of the reward model because it forces the aggregate head's output to equal the learned composition of sub-attribute scores; this assumption fails when the composition function g_ψ is misspecified (e.g., attributes interact non-additively), in which case L_cons reflects the mismatch between the aggregate head and the composition head rather than true compositionality. The attribute-specific supervision L_attr measures the identifiability of each head because it directly trains each head on its named attribute's pairwise labels; this assumption fails when the attribute labels are noisy or when the heads collapse to constants, in which case L_attr reflects label noise or degenerate solutions rather than true attribute semantics. The max-min selection measures targeted enhancement because it selects the candidate with the highest weakest attribute score; this assumption fails when the weakest attribute is not the bottleneck for overall design quality (e.g., when layout is already satisfactory but color is poor), in which case max-min may over-optimize an already-adequate attribute. The sub-attribute scores measure design attributes because each head is trained on preference data annotated for that attribute; this assumption fails when the preference annotations are noisy or biased toward one attribute, in which case the scores reflect annotation bias rather than true attribute quality.

## Contribution

['A factorized compositional design reward model (FACTOR-PE) that decomposes the Groupwise Design Reward into four sub-attribute heads and a learned composition head, with a consistency loss that verifies the aggregate reward equals the composition of sub-attribute rewards.', 'A max-min selection and targeted refinement procedure that uses the verified decomposition to generate attribute-specific prompt edits, enabling interpretable and controllable prompt enhancement.', 'An empirical demonstration that FACTOR-PE improves the weakest design sub-attribute without degrading others, and that the consistency loss yields sub-attribute scores that are predictive of human judgments of each attribute.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Qwen-Image-Bench prompts with attribute-annotated preferences (≥10k pairs, each with 4 attribute labels) | Attribute-level labels needed to test decomposition claim. |
| Primary metric | Weakest-attribute design score on refined prompts | Directly measures the max-min targeting claim. |
| Baseline | Qwen-Image-PE (scalar PE) | Tests whether factorized reward beats monolithic enhancement. |
| Baseline | HPSv3-style scalar reward reranking | Tests whether attribution, not just scoring, drives gains. |
| Baseline | Weighted-sum multi-head selection | Tests whether max-min beats aggregate optimization. |
| Baseline | Post-hoc linear probe on scalar reward features | Tests whether consistency-constrained heads recover sub-attributes beyond a probe. |
| Ablation-of-ours | FACTOR-PE without consistency loss | Tests whether verified decomposition is load-bearing. |
| Ablation-of-ours | FACTOR-PE without attribute-specific supervision (L_attr) | Tests whether attribute supervision is load-bearing for identifiability. |
| Diagnostic | Permutation-identification test (diagonal accuracy vs. off-diagonal) | Verifies each head predicts its named attribute best. |

### Why this setup validates the claim

The central claim is that a verified factorized reward yields identifiable sub-attribute scores that enable targeted, non-degrading prompt edits. Qwen-Image-Bench supplies prompts where design attributes can be scored independently, so the weakest-attribute metric can actually move. Qwen-Image-PE tests the sub-claim that scalar enhancement leaves the weakest attribute behind; the HPSv3-style scalar reranker isolates whether attribution matters beyond mere multi-candidate selection; the weighted-sum variant isolates whether max-min selection, not just multi-head scoring, produces the gain. The post-hoc linear probe baseline tests whether the consistency-constrained heads recover sub-attribute scores that a probe on a scalar reward cannot, distinguishing our mechanism from mere multi-head capacity. The no-consistency ablation is the decisive control: if removing the consistency loss collapses the weakest-attribute gain while leaving aggregate scores intact, the decomposition itself is doing the work rather than extra capacity. The no-attribute-supervision ablation tests whether L_attr is necessary for identifiability, as the load-bearing assumption requires. The permutation-identification test directly verifies that each head's score predicts its named attribute better than the others, providing empirical evidence for the identifiability claim. The weakest-attribute metric is chosen because it is the only quantity the method explicitly optimizes and the only one where scalar baselines have a structural blind spot.

### Expected outcome and causal chain

**vs. Qwen-Image-PE (scalar PE)** — On a request where the enhanced prompt already fixes layout but leaves color harmony muddy, the scalar PE baseline produces a fluent prompt with no signal about which attribute is weak, because its training objective collapses all design aspects into one preference score. Our method instead scores all eight candidates per attribute, selects the one with the highest floor, and emits an explicit "improve color harmony" edit, because the four heads expose per-attribute deficits. We expect a clear weakest-attribute gain on prompts whose bottleneck is color or typography, with near-parity on prompts already balanced across attributes.

**vs. HPSv3-style scalar reward reranking** — On a case where two candidate prompts have similar aggregate reward but one has a collapsed layout and strong color, the scalar reranker picks arbitrarily or favors the flashy-color candidate, because a monolithic score cannot represent the trade-off. Our method instead rejects that candidate at max-min selection and refines the balanced one, because the composition head plus four heads make the weak layout score visible. We expect the gap to concentrate on prompts with high attribute variance across candidates and to shrink to parity on low-variance prompts, which is the signature of attribution rather than raw scoring power.

**vs. Weighted-sum multi-head selection** — On a case where typography is severely weak but color and composition are excellent, weighted-sum selection keeps the high-total candidate and never triggers a typography edit, because the sum lets strong heads mask the weak one. Our method instead selects on the minimum and issues a typography-targeted refinement, because max-min explicitly optimizes the floor. We expect a noticeable weakest-attribute advantage on imbalanced-attribute prompts, with the weighted-sum variant possibly matching or slightly exceeding us on mean aggregate score — a divergence that confirms the mechanism is doing targeted repair rather than generic quality inflation.

**vs. Post-hoc linear probe on scalar reward features** — On a controlled synthetic attribute perturbation test (e.g., systematically varying color harmony while holding layout, typography, and composition fixed), the linear probe on a scalar reward's penultimate features cannot recover the perturbed attribute because the scalar reward was trained only on aggregate preferences and its features are not constrained to be sufficient statistics for sub-attributes. Our method, with consistency and attribute-specific supervision, recovers the perturbed attribute score with higher correlation (expected Pearson r > 0.7 vs. < 0.4 for the probe), because L_attr directly trains each head on its named attribute and L_cons ties the aggregate to the composition. We expect the probe to perform near chance on the permutation-identification test, while our heads show diagonal accuracy exceeding off-diagonal by at least 0.1.

### What would falsify this idea

If the no-consistency ablation matches full FACTOR-PE on weakest-attribute score, the verified-decomposition claim is false and the gain is just multi-candidate selection. If our improvement is uniform across balanced and imbalanced prompts rather than concentrated where the predicted weakest-attribute bottleneck occurs, the targeted-enhancement causal chain is not what is producing the result. If the permutation-identification test shows diagonal accuracy not exceeding off-diagonal by at least 0.1, the attribute heads are not identifiable and the load-bearing assumption fails.

## References

1. Qwen-Image-PE: Learning Visual Design for Text-to-Image Prompt Enhancement
