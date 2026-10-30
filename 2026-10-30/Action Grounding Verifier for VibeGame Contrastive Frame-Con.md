# Action Grounding Verifier for VibeGame: Contrastive Frame-Consequence Identification

## Motivation

VibeGame's AI-native engine exposes a fixed discrete action API with schema-validated state, but agents have no mechanism to verify that executing an action produces the visual consequence its name implies. This is a concrete instance of the broader identifiability gap left by Genie's latent action model: discrete action semantics are never tied to frame-level effects, so adversarial agents can only check schema validity, not semantic grounding. Without such verification, any game built through the engine may pass all structural checks while its actions are visually inert or semantically mismatched, and the self-evolution loop distills flawed action contracts.

## Key Insight

The mapping from each discrete action to its frame-level visual consequence is identifiable from engine rollouts because the engine's frame-synchronous control lets us hold all non-action variables fixed, so a contrastive objective over matched and mismatched (action, before/after frame) pairs can isolate the action's causal visual signature without any learned independence assumption.

## Method

**What it is.** We introduce the Action Grounding Verifier (AGV), a contrastive model trained on engine rollouts that takes a discrete action `a` and a pair of consecutive frames `(f_t, f_{t+1})` and outputs a grounding score `g(a, f_t, f_{t+1})` estimating whether the visual transition is the consequence of executing `a` at `f_t`. AGV is queried by VibeGame's adversarial review agents to check whether an action's intended semantics match its observed visual outcome.

**How it works.**

```python
# Phase 1: Rollout collection with controlled intervention
# For each positive (s, a), we generate counterfactual negatives by stepping
# the engine with every a' != a from the SAME state s, holding (f_t, s) fixed.
for each game project P produced by VibeGame:
    for each action a in P.action_set:
        for each reachable state s (sampled via engine frame-synchronous stepping):
            f_t = render(s)
            f_t_plus_1 = engine.step(s, a)          # execute ONLY action a
            store (a, f_t, f_t_plus_1, s) as a positive pair
            # counterfactual negatives: same state s, same f_t, different action a'
            for each a' != a in P.action_set:
                f_t_plus_1_neg = engine.step(s, a')  # execute a' from SAME state s
                store (a, f_t, f_t_plus_1_neg, s) as a negative pair

# Phase 2: Contrastive training
# Encoder architecture:
#   encoder_A: discrete action embedding lookup, dim 64
#   encoder_F: shared CNN over f_t and f_t+1 (two 3x64x64 RGB frames),
#              concat features -> 2-layer MLP (hidden=256, GeLU), output dim 256
#   g(a, f_t, f_t+1) = cosine_similarity(encoder_A(a), encoder_F(f_t, f_t+1))
# Loss: InfoNCE with temperature tau=0.07 over a batch of (a, f_t, f_t+1) positives
#       and counterfactual negatives (a, f_t, f_t+1_neg) from the SAME state s.
#       Batch size = 256 positives, each with up to |A|-1 negatives.
# Semantic-completeness regularizer:
#   For each action a, maximize min over states of g(a, f_t, f_t+1) - max over a' of g(a', f_t, f_t+1)
#   (margin 0.2) to ensure every action has a detectable signature.
#   Implemented as a hinge loss: max(0, 0.2 - (g(a, f_t, f_t+1) - max_{a'} g(a', f_t, f_t+1)))
#   averaged over sampled states, added to InfoNCE with weight 0.5.

# Phase 3: Integration into adversarial review loop
for each action a in proposed game:
    sample K=20 reachable states via engine
    compute g(a, f_t, f_t+1) for each
    if mean_g < threshold (0.5) or variance_g > 0.3:
        flag action as semantically ungrounded or context-dependent
    # Agent queries AGV with intended semantics: e.g., "does 'jump' produce upward motion?"
    # AGV returns grounding score; agent uses it to revise action implementation.
```

**Why this design.** We chose contrastive InfoNCE over a generative frame predictor because the engine's frame-synchronous stepping lets us form exact counterfactual negatives by swapping actions from the same state, which a generative model cannot exploit without modeling all visual dynamics; the cost is that AGV only certifies discriminability, not full causal sufficiency. We chose a shared frame-pair encoder over separate per-action classifiers because sharing forces the representation to capture action-discriminative transitions rather than action identity, accepting that rare actions may have noisier embeddings. We added the semantic-completeness regularizer over plain contrastive loss because plain InfoNCE can satisfy the objective by making one action dominate all transitions; the margin term forces every action to have a detectable signature, accepting a harder optimization. Unlike a retrieval baseline that would match action names to frame descriptors via nearest neighbors, AGV's score is a learned contrastive margin that requires intervening on the action while holding the state fixed — a counterfactual that retrieval cannot compute from a static library. Crucially, the counterfactual negatives are generated by stepping the engine with every a' != a from the SAME state s, so that state s and f_t are held constant across the contrast set; this makes the only feature that can separate positives from negatives the action-conditional frame delta, removing the incentive for the encoder to encode state identity.

**Why it measures what we claim.** The grounding score `g(a, f_t, f_t+1)` measures action-semantic identifiability because the contrastive objective with counterfactual same-state negatives approximates the log-odds that the observed transition was caused by `a` rather than any other action, under the assumption that the engine's step function is deterministic and that all non-action state is held fixed during rollout collection; this assumption fails when the engine has stochastic transitions (e.g., physics jitter), in which case the score reflects average transition discriminability rather than exact causal attribution. The semantic-completeness regularizer measures whether every action has a detectable visual signature because the margin term enforces a positive gap between the true action's score and the best impostor's score across sampled states; this assumption fails when an action's effect is only visible in rare states not reached by the sampling procedure, in which case the regularizer reflects coverage of the sampled state distribution rather than global completeness. The variance threshold in Phase 3 measures context-dependence of an action's visual consequence because high variance across states indicates the action's effect is not a stable function of the transition; this assumption fails when the state sampler is biased toward visually diverse states, in which case variance reflects sampler bias rather than true context-dependence.

## Contribution

["A contrastive Action Grounding Verifier (AGV) that learns to map VibeGame's discrete engine actions to frame-level visual consequences using action-swapped negatives from frame-synchronous rollouts, with a semantic-completeness regularizer that enforces every action has a detectable signature.", "A concrete integration protocol that closes the grounding gap in VibeGame's adversarial review loop: review agents query AGV to flag actions whose observed visual outcomes do not match their intended semantics, enabling revision before project acceptance.", 'An empirical design principle: grounding scores computed from controlled engine interventions (holding state fixed, swapping actions) are more discriminative than retrieval-based action-frame matching because they require counterfactual action intervention rather than static descriptor similarity.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | VibeGame PA2G games with engine rollouts | Engine stepping supplies exact action-swapped negatives |
| Primary metric | Action-grounding precision@flag on held-out games | Directly measures whether flagged actions are truly ungrounded |
| Baseline | Retrieval action-name-to-frame matcher | Tests whether learned contrast beats static name matching |
| Baseline | Generative frame predictor residual | Tests whether contrastive negatives beat generative dynamics |
| Baseline | Per-action binary classifier | Tests whether shared encoder beats action-identity shortcuts |
| Baseline | Causal pixel-delta signature (mean absolute frame delta conditioned on action, with action-swapped deltas as contrast) | Tests whether learned encoder beats trivial causal estimator |
| Ablation-of-ours | AGV without semantic-completeness regularizer | Isolates the margin term's contribution to coverage |
| Ablation-of-ours | AGV with biased state sampler (control) | Quantifies sampler-bias failure mode on variance metric |

### Why this setup validates the claim

The central claim is that AGV's grounding score identifies actions whose intended semantics do not match their observed visual consequence, and that this signal is usable by adversarial review agents. The retrieval baseline tests the sub-claim that contrastive margin over counterfactual action swaps beats static name-to-descriptor matching; if retrieval matched AGV, the counterfactual machinery would be unnecessary. The generative predictor baseline tests whether explicit dynamics modeling is needed; if it matched AGV, the contrastive framing would be a needless simplification. The per-action classifier tests whether the shared frame-pair encoder actually forces action-discriminative transition representations rather than memorizing action identity. The causal pixel-delta baseline tests whether the learned encoder adds value over a trivial hand-crafted causal estimator; if AGV does not beat it, the novelty collapses to 'learned vs. hand-crafted contrast'. The no-regularizer ablation isolates whether the semantic-completeness margin is what gives uniform per-action detectability. The biased-sampler control tests the assumption that variance reflects action-effect instability rather than sampler bias. Precision@flag is the right primary metric because the downstream use is a review agent that flags actions for revision: what matters is whether flagged actions are genuinely ungrounded (precision), not whether AGV's raw scores correlate with anything. Held-out games prevent the encoder from memorizing project-specific visual styles, so a win must come from transferable transition semantics.

### Expected outcome and causal chain

**vs. Retrieval action-name-to-frame matcher** — On a case where an action is named "jump" but its implementation produces a horizontal dash, retrieval matches the name to upward-motion descriptors in its static library and scores the transition as grounded, because it never intervenes on the action while holding state fixed. Our method instead computes a contrastive margin between the observed transition and the action-swapped negatives from the same state, so the horizontal dash scores low against the "jump" embedding and high against the dash embedding. We expect a noticeable precision@flag gap on mislabeled-action subsets and rough parity on correctly-implemented actions.

**vs. Generative frame predictor residual** — On a case where the engine has mild physics jitter, the generative predictor's residual is dominated by unpredictable pixel-level noise and it cannot cleanly separate a wrong action from a right one, because it models absolute visual dynamics rather than action-conditional discriminability. Our method instead compares the observed transition to action-swapped alternatives from the same state, so shared jitter cancels and only the action-conditional signal remains. We expect AGV to hold precision under jittered engines where the predictor degrades, with the gap concentrated on stochastic-transition games.

**vs. Per-action binary classifier** — On a case where a rarely-used action shares most of its visual signature with a common one, the per-action classifier learns action identity from correlated state features and assigns high confidence to the wrong action, because separate classifiers can exploit state priors instead of transition content. Our method instead shares one frame-pair encoder across actions, so the representation must encode what changed between frames rather than which state it was in. We expect higher precision on rare-action and visually-ambiguous subsets, with parity on common, visually-distinct actions.

**vs. Causal pixel-delta signature** — On a case where an action's visual effect is subtle (e.g., a small color shift) but consistent, the pixel-delta baseline computes mean absolute frame delta conditioned on action and may be dominated by large but irrelevant background motion, because it uses raw pixel differences without learning to isolate the action-conditional component. Our method instead learns a contrastive embedding that can suppress background motion and focus on the action's causal signature. We expect AGV to outperform the pixel-delta baseline on subtle-effect and visually-cluttered subsets, with parity on actions with large, distinct visual effects.

### What would falsify this idea

If AGV's precision@flag gain over the retrieval, classifier, and pixel-delta baselines were uniform across all action subsets rather than concentrated on mislabeled, jittered, and rare-action cases where the predicted failure modes occur, the contrastive-margin mechanism would not be doing the claimed work. Likewise, if the no-regularizer ablation matched full AGV on per-action detectability, the semantic-completeness regularizer would be unnecessary and the coverage claim would be unsupported; and if the biased-sampler control showed variance_g > 0.3 for actions with stable effects, the variance metric would reflect sampler bias rather than context-dependence.

## References

1. VibeGame: Prompt-to-Game Development with AI-Native Engine and Self-Evolving Adversarial Agent Team
