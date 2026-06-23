# Causal Decomposition of Intersectional Biases in Multimodal Models via Full-Factorial Attribute Manipulation

## Motivation

Existing bias evaluation methods (e.g., StylisticBias, TIBET) rely on single-attribute perturbations, implicitly assuming that bias decomposes into independent additive contributions of each attribute. This assumption ignores interaction effects between attributes (e.g., age and gender jointly affecting stereotype), leading to incomplete and potentially misleading bias estimates, especially for intersectional demographic groups. The root cause is a structural reliance on additive decomposability without validation, leaving the field without a method to assess whether such decomposability holds or to quantify interaction-driven biases.

## Key Insight

Full factorial design ensures that attribute effects are orthogonal, enabling unbiased estimation of main effects and interactions without assuming additive decomposability.

## Method

(A) **What it is**: FACIAL (Full-factorial Attribute Causal Intersectional Analysis for biases) is a causal inference framework that generates images with controlled combinatorial manipulations of demographic attributes and uses ANOVA-style decomposition to quantify main and interaction effects of these attributes on social biases measured by multimodal LLMs. Input: a set of K binary attributes (e.g., gender, age, skin tone), a generative model with independent attribute control, and a target MLLM for bias judgments. Output: effect sizes (partial eta-squared) for each attribute and each interaction, plus a test of whether the additive decomposition assumption is valid.

(B) **How it works** (pseudocode):
```pseudocode
# Phase 1: Generate full factorial design matrix
attributes = ['gender', 'age', 'skin_tone']  # K=3 binary, example
levels = [2, 2, 2]  # number of levels per attribute (binary)
design_matrix = full_factorial(levels)  # 2^K = 8 rows, each a unique combination

# Phase 2: Generate controlled images
base_latent = sample_fixed_identity()  # one common identity latent for all
images = []
for combo in design_matrix:
    image = edit_attributes(base_latent, combo)  # use StyleGAN3 with selective editing
    images.append(image)

# Phase 3: Collect bias judgments from target MLLM
judgments = []
for image in images:
    score = MLLM_judge(image, target_judgment="trustworthy")  # scale 0-10
    judgments.append(score)

# Phase 4: ANOVA decomposition
# Fit linear model: judgment ~ C(gender) + C(age) + C(skin_tone) + gender:age + gender:skin_tone + age:skin_tone + gender:age:skin_tone
import statsmodels.formula.api as sm
model = sm.ols('judgment ~ C(gender) * C(age) * C(skin_tone)', data=df).fit()
# Compute partial eta-squared for each term using type III sums of squares
anova_results = sm.stats.anova_lm(model, typ=3)
# Output effect sizes and p-values
```

(C) **Why this design**: We chose a full factorial experimental design over commonly used single-attribute perturbation (as in StylisticBias) because it is the only way to unbiasedly estimate interaction effects without assuming additivity. This design introduces combinatorial explosion (e.g., 2^K conditions for binary attributes), but we counterbalance it by (1) limiting attributes to those identified as most impactful via prior studies (e.g., StylisticBias found ~15 attributes explain most variance), (2) using a fixed base latent to minimize noise from identity variation, and (3) leveraging the efficiency of modern generative models (e.g., StyleGAN3) for rapid image generation. We chose ANOVA over causal forests for interpretability, as our primary goal is to test the structural assumption of decomposability, not to predict individual-level bias. The trade-off is that ANOVA assumes linear and homoscedastic errors, which may be violated if bias responses are bounded or heteroscedastic; however, we treat the F-test as a heuristic and validate with non-parametric permutation tests. We deliberately fixed the base latent across all conditions to control confounding from identity-specific features, accepting that the generated images may be less diverse than real identities, but this is necessary for causal identification of attribute effects.

(D) **Why it measures what we claim**: The main effect size (partial eta^2 of attribute i) measures the average causal effect of that attribute on bias because the full factorial design ensures attribute levels are independent and manipulated without confounding; this assumption holds if the generative model can vary attributes independently (as demonstrated by StylisticBias). When the generative model fails to perfectly isolate attributes (e.g., editing age also introduces wrinkles that correlate with skin tone changes), the estimated effects may be biased towards attribute correlations; in that case, the measured main effect confounds with other attributes. The interaction term (beta_ij) measures the deviation from additivity, i.e., the degree to which the effect of attribute i depends on attribute j; it captures intersectional bias not accounted by single-attribute perturbations. This interaction term is zero if and only if bias is additive across attributes, which is the decomposability assumption we test. The R^2 difference between main-effects-only and full model directly quantifies the validity of decomposability: if interaction R^2 is large, the decomposability assumption is violated. Thus, FACIAL operationalizes the meta-gap by providing a statistical test for the additivity assumption inherent in prior work.

## Contribution

(1) A causal inference framework (FACIAL) for decomposing multimodal model biases into main and interaction effects of demographic attributes via full factorial manipulation. (2) An empirical test of the decomposability assumption, revealing that intersectional biases often involve significant interactions, thus invalidating the additive assumption in prior single-attribute methods. (3) A synthetic dataset of images with full factorial combinations of demographic attributes for bias evaluation across multiple MLLMs.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
|------|--------|----------|
| Dataset | FACIAL-generated full factorial image set | Full factorial design for causal identification |
| Primary metric | Interaction partial eta-squared | Quantifies deviation from additivity |
| Baseline 1 | Single-attribute perturbation (StylisticBias) | Assumes additivity; ignores interactions |
| Baseline 2 | TIBET (predefined axis bias detection) | No causal decomposition; predefined axes |
| Baseline 3 | Human judgment (ground truth) | Gold standard for bias perception |
| Ablation-of-ours | Without fixed base latent (random identities) | Confounds identity with attribute effects |

### Why this setup validates the claim

The full factorial dataset ensures that attribute effects are unconfounded, enabling causal estimation of main and interaction effects. The primary metric (interaction partial eta-squared) directly quantifies the deviation from additivity that our method aims to detect. Baselines test the necessity of interaction analysis: StylisticBias shows the insufficiency of single-attribute approaches, TIBET highlights the limitation of pre-defined axes without combinatorial coverage, and human judgment provides ecological validity. The ablation (random identities) isolates the benefit of controlling identity confounds, proving that our fixed-base strategy is essential for clean causal identification. Combined, these components form a falsifiable test: if interaction effects are negligible even when known intersectional biases exist, the additive assumption holds and our method is unnecessary.

### Expected outcome and causal chain

**vs. StylisticBias** — On a case where bias depends jointly on gender and skin tone (e.g., dark-skinned women are judged less trustworthy than expected from additive effects), StylisticBias measures only average effects of each attribute independently, missing the interaction. Our method decomposes the total variance into main and interaction components, so we expect a noticeable gap in interaction R² (e.g., >0.1) for such attribute pairs, while StylisticBias reports near-zero interaction because it never estimates them.

**vs. TIBET** — On a case where the bias arises from an unusual combination not covered by predefined axes (e.g., age plus facial hair), TIBET may fail to detect it because it only tests preset attribute pairs. Our method exhaustively manipulates all combinations, so we expect to detect significant interaction effects for any combination that humans perceive as biased, whereas TIBET would show null results for those combinations.

**vs. Human judgment** — On a case where humans consistently report intersectional bias (e.g., older Asian women being rated less competent), our method should yield a significant interaction term for age:gender:ethnicity. If our method fails to produce a corresponding effect size while humans agree, that would indicate a failure of the generative model or metric. We expect high correlation between human interaction ratings and our estimated interaction partial eta-squared across attribute sets.

### What would falsify this idea
If the interaction R² is uniformly near zero across all attribute sets despite prior evidence of intersectional bias from human judgments or social science, then the additive decomposition assumption holds and our central claim (that interactions are crucial) is false. Alternatively, if our gain over StylisticBias is uniform across all attribute pairs rather than concentrated on pairs with known interactions, the improvement is not due to interaction detection but to some other factor (e.g., better image quality).

## References

1. StylisticBias: A Few Human Visual Cues Drive Most Social Biases in MLLMs
2. MLLM-as-a-Judge: Assessing Multimodal LLM-as-a-Judge with Vision-Language Benchmark
3. TIBET: Identifying and Evaluating Biases in Text-to-Image Generative Models
4. Easily Accessible Text-to-Image Generation Amplifies Demographic Stereotypes at Large Scale
5. DALL-EVAL: Probing the Reasoning Skills and Social Biases of Text-to-Image Generation Models
