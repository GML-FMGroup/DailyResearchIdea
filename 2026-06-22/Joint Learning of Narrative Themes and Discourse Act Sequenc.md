# Joint Learning of Narrative Themes and Discourse Act Sequences via Hierarchical Bayesian Nonparametrics

## Motivation

Existing narrative analysis frameworks rely on fixed human-annotated dimensions (e.g., the 11-dimension framework in NarraBERT by Kayser et al.), which fail to capture domain-specific narrative patterns and require costly re-annotation for each new domain. This limitation arises because these dimensions are designed a priori and assume universal narrative structures, ignoring that narrative themes and their sequential organization vary across genres and communities. A data-driven approach that automatically discovers thematic and sequential structure is needed.

## Key Insight

The hierarchical coupling between latent themes and discourse act transitions, captured through a hierarchical Dirichlet process, enables joint discovery of narrative themes and coherent act sequences without human-defined dimensions.

## Method

(A) **What it is**: We present NarBayes, a hierarchical Bayesian nonparametric model that jointly discovers narrative themes and discourse act sequences. Input: a corpus of narrative texts (tokenized into sentences). Output: for each narrative, a latent theme assignment and a sequence of latent discourse acts; globally, a set of theme prototypes (distributions over discourse acts) and act transition matrices.

(B) **How it works**: The model extends the Hierarchical Dirichlet Process Hidden Markov Model (HDP-HMM) with a document-level theme variable that modulates act transitions.

```python
# Hyperparameters:
#   alpha: DP concentration for themes
#   beta: DP concentration for act transitions
#   gamma: stick-breaking prior for global act base distribution
#   K: number of discourse acts (fixed, e.g., 6)

# Generative process:
# 1. Global: Draw infinite theme base distribution G0 ~ DP(alpha, H)
# 2. For each theme k (indexed by stick-breaking):
#    - Draw transition matrix theta_k ~ Dirichlet(beta) over K acts
# 3. For each narrative d:
#    - Draw theme z_d ~ GEM(alpha)  # stick-breaking over themes
#    - For each sentence t in 1..T_d:
#        - Draw act a_{d,t} ~ Categorical(theta_{z_d}[a_{d,t-1}])
#        - Draw sentence words from emission distribution given a_{d,t} (e.g., topic model)

# Inference: Collapsed Gibbs sampling using Chinese Restaurant Franchise representation.
# Update theme assignments using conditional likelihood of act sequences given theme.
# Update act assignments using forward-backward sampling given theme and transition.
```

(C) **Why this design**: We choose a hierarchical Bayesian nonparametric approach over fixed-dimension frameworks for three reasons. First, using a stick-breaking prior for themes allows the model to automatically determine the number of themes from data, avoiding the need for human-defined dimensions. This addresses the key limitation of NarraBERT's fixed 11 dimensions. Second, the theme-conditional transition matrices capture the idea that narrative structure varies by theme (e.g., a romance narrative may follow different act sequences than a tragedy). This is a structural coupling that previous work treats as independent. Third, we use collapsed Gibbs sampling (instead of variational inference) because it provides asymptotically exact samples and handles the hierarchical structures without needing to specify a variational family, accepting that it may be slower for very large corpora. The design choice of modeling acts as discrete labels (rather than continuous representations) ensures interpretability and aligns with narrative theory's focus on functional units. The trade-off is that we assume act boundaries align with sentence boundaries, which may not hold for all narratives, but we accept this simplification for tractability.

(D) **Why it measures what we claim**: The computational quantity `theme z_d` measures the narrative macro-structure (theme) because it aggregates act transition patterns across the document; the assumption is that narratives with similar act sequence dynamics share a common theme. This assumption fails when two very different themes accidentally yield similar transition matrices (e.g., if both use similar act patterns), in which case the model may conflate them, but the stick-breaking prior encourages few distinct themes to reduce such conflation. The act sequence `a_{d,t}` measures discourse-level narrative progression because each act label represents a functional narrative unit (e.g., setup, conflict) based on its role in the sequence; the assumption is that these functions are common across narratives. This assumption fails when the same act label corresponds to different narrative functions in different contexts, but the theme-specific transitions help disambiguate by conditioning on the theme. The joint distribution `p(a_{d,t}|z_d, a_{d,t-1})` captures narrative coherence because it enforces that act sequences follow theme-consistent transition probabilities; the assumption is that coherent narratives exhibit systematic act transitions. This assumption fails when narratives are intentionally incoherent (e.g., postmodern stories), in which case the model would assign lower likelihood to such texts.

## Contribution

(1) A hierarchical Bayesian nonparametric model (NarBayes) that jointly learns narrative themes and discourse act sequences without any human-defined dimensions. (2) A demonstration that the discovered themes correspond to interpretable narrative structures (e.g., romantic comedy, tragedy) and that the act transition patterns differ across themes, validated on a diverse corpus of stories. (3) An open-source implementation of the inference algorithm.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | ROCStories (Story Cloze Test) | Coherence discrimination task with correct/incorrect endings |
| Primary metric | Accuracy on story ending prediction | Directly tests narrative coherence capture |
| Baseline 1 | HDP-HMM without theme (only acts) | Isolates benefit of joint theme modeling |
| Baseline 2 | LDA topic model on sentences | Tests theme discovery without act structure |
| Baseline 3 | Fixed-dimension NarraBERT | Compares to predefined theme taxonomy |
| Ablation-of-ours | Ours with shared transition matrix | Tests necessity of theme-conditional transitions |

### Why this setup validates the claim
This experimental design provides a falsifiable test of our central claim—that jointly discovering narrative themes and discourse act sequences with theme-conditional transitions improves narrative coherence modeling. The Story Cloze Test measures the ability to distinguish plausible from implausible story endings, which requires understanding both high-level theme (macrostructure) and low-level act progression (microstructure). The HDP-HMM baseline (acts only) tests whether themes add value beyond act dynamics alone; if our method outperforms it, theme information is beneficial. The LDA baseline tests whether themes alone (without act structure) are sufficient; if our method outperforms it, act sequencing matters. NarraBERT tests whether a fixed, predefined set of themes is as effective as automatically discovered themes. The ablation (shared transitions) tests whether conditioning act transitions on themes is crucial. The chosen metric—accuracy—captures overall coherence discrimination; a significant improvement over baselines on the full dataset, especially concentrated on stories with strong thematic cues, would confirm our hypothesis. Conversely, if our method performs no better than the HDP-HMM baseline, the theme variable is unnecessary; if it matches LDA, act structure is irrelevant; if it matches NarraBERT, automatic discovery offers no advantage; and if the ablation matches the full model, theme-conditional transitions are superfluous.

### Expected outcome and causal chain

**vs. HDP-HMM without theme** — On a story that transitions from a romantic scene to a tragic event (e.g., a couple's argument after a wedding), the HDP-HMM predicts act sequences based on global transition probabilities, missing that romance themes have different act transitions (e.g., more dialogues, fewer action scenes). Our method, with theme-conditional transitions, captures that romance stories prefer dialogue-heavy acts while tragedy may shift to action. Thus, our model assigns higher coherence probability to the correct ending, yielding a noticeable accuracy gap on such multi-theme stories (e.g., +5–10% on those with high theme entropy), while performing similarly on single-theme stories.

**vs. LDA topic model** — On a story where the same topic (e.g., "beach") appears in both a happy vacation narrative and a sad farewell narrative, LDA assigns similar topic mixtures and cannot distinguish coherent versus incoherent endings because it ignores act order. Our method incorporates act sequences (e.g., vacation theme may follow a "setup→fun→resolution" pattern, while farewell follows "setup→conflict→climax"), so it correctly penalizes endings that deviate from the expected act progression. We expect our method to achieve higher accuracy on stories where topic alone is ambiguous but act ordering disambiguates, e.g., a gap of +3–8% on stories with high topical overlap but distinct narrative structures.

**vs. Fixed-dimension NarraBERT** — On a narrative with a theme not covered by NarraBERT's 11 predefined dimensions (e.g., a "heist" story that blends planning and betrayal), NarraBERT forces it into the closest dimension (e.g., "Suspense"), missing theme-specific act transitions. Our method discovers novel themes from data, capturing the heist's distinct act pattern (e.g., setup→conflict→twist→resolution). Therefore, our model assigns higher likelihood to the correct ending, leading to a larger accuracy advantage on out-of-distribution themes (e.g., +7–12% on stories from rare genres), while performing similarly on those matching predefined dimensions.

### What would falsify this idea
If our model's accuracy gains are uniform across all story subsets (e.g., similar improvement on both high- and low-theme-variation stories) rather than concentrated on subsets where the hypothesized failure modes of baselines occur (e.g., stories with ambiguous topics, multiple themes, or rare genres), then the claim that theme-conditional act transitions drive the improvement is false; instead, some global advantage (e.g., better optimization) would be indicated.

## References

1. Characterizing Narrative Content in Web-scale LLM Pretraining Data
2. Stories That Heal: Characterizing and Supporting Narrative for Suicide Bereavement
3. Where Do People Tell Stories Online? Story Detection Across Online Communities
4. Toward a Data-Driven Theory of Narrativity
