# Contrastive Likelihood Calibration: Zero-Shot Reward for Text-to-Image Alignment via Minimal Prompt Baseline

## Motivation

Existing likelihood-based rewards like SpectraReward (Read It Back) compute the average log-likelihood of the full prompt given the generated image, but this likelihood conflates generic visual alignment (e.g., 'a photo') with instruction-specific alignment, leading to misalignment with human preference under diverse instructions. We propose to calibrate by subtracting the likelihood of a minimal semantic core of the prompt, which isolates the instruction-specific contribution.

## Key Insight

Subtracting the likelihood of a minimal semantic core cancels the MLLM's generic prior because the core captures the most generic visual association, leaving only the instruction-specific alignment.

## Method

### Contrastive Likelihood Calibration (CLC)

(A) **What it is**: CLC is a zero-shot reward function that outputs a scalar alignment score for a generated image I given a prompt P. It computes the difference between the average log-likelihood of P under I and that of a minimal semantic core C(P) under I, where the prompts are preprocessed to place core tokens as a prefix.

(B) **How it works**:
```python
import spacy
nlp = spacy.load('en_core_web_sm')

def extract_core(prompt):
    # Rule-based core extraction using spaCy
    doc = nlp(prompt)
    root = [token for token in doc if token.head == token][0]
    subject = [child for child in root.children if child.dep_ == 'nsubj'][0]
    objects = [child for child in root.children if child.dep_ in ('dobj', 'pobj')]
    core_tokens = [subject.text, root.text] + [obj.text for obj in objects]
    return ' '.join(core_tokens)

def reorder_prompt(prompt, core):
    # Reorder prompt tokens so that core tokens appear first (preserving internal order)
    core_tokens = core.split()
    non_core_tokens = [t for t in prompt.split() if t not in core_tokens]
    # preserve relative order within core and within non-core
    return ' '.join(core_tokens + non_core_tokens)

def clc_reward(image, prompt):
    core = extract_core(prompt)
    reordered_prompt = reorder_prompt(prompt, core)
    # Use a pretrained MLLM (e.g., BLIP-2) to compute average log-likelihood per token
    L_full = mllm.likelihood(image, reordered_prompt, mode='teacher_forcing')  # average log-likelihood over all tokens
    L_core = mllm.likelihood(image, core, mode='teacher_forcing')               # average log-likelihood over core tokens
    reward = L_full - L_core   # equivalent to conditional log-likelihood of non-core given core
    return reward
```

(C) **Why this design**: We chose a rule-based minimal core extraction over learned extraction to maintain zero-shot generality and avoid training data bias, accepting the trade-off that rule-based extraction may miss some semantic nuances or over-remove critical modifiers. We chose to subtract the core likelihood rather than divide because subtraction preserves the additive decomposition of log-probabilities under the model's autoregressive factorization, and after reordering, L_full - L_core equals the average log-likelihood of the non-core tokens conditioned on the core. We chose to use average log-likelihood (not sum) to normalize for prompt length, ensuring that longer prompts do not dominate. We chose to use a single minimal core rather than multiple contrastive baselines to keep computation low, accepting that one baseline may not capture all prior biases (e.g., stylistic biases). **Load-bearing assumption**: The minimal core captures all generic visual priors (e.g., typical subject-object relations), so that L_full - L_core isolates instruction-specific alignment. This assumption is explicitly verified in our experiments by testing alternative core extraction strategies (random core, contradictory core) and by analyzing failure cases where core tokens themselves carry instruction-specific information (e.g., rare subjects).

(D) **Why it measures what we claim**: The computational quantity L_full measures the model's conditional probability of the reordered full instruction (core as prefix) given the image. L_core measures the model's conditional probability of the minimal core given the image. Due to the reordering, L_full - L_core equals the average log-likelihood of the non-core tokens conditioned on the core tokens, which isolates the instruction-specific alignment because it cancels the generic prior captured by the core. This relies on the assumption that the minimal core captures the most generic visual features that the model overweighs. This assumption fails when the minimal core is not sufficient to capture all generic biases (e.g., the model might have a prior for 'dog' over 'cat' even in the core, but since core is common to both, it cancels out; however, if the core itself is biased, the contrast may not fully remove bias). In that case, the metric reflects residual bias from the core. The core extraction rule also assumes that the main subject and action are the most essential; this fails for prompts where the crucial information is in a modifier (e.g., 'a blue car' — the color is essential but not in core); then the difference may underestimate alignment, since the color is treated as instruction-specific but it should be. Additionally, **missing equivalence**: CLC measures instruction-specific alignment as L_full - L_core under the assumption that L_core captures all generic visual priors. When core tokens carry instruction-specific information (e.g., rare subjects), CLC may underestimate the reward because the core itself contains instruction-specific signal. To address this, we propose a calibration step: for prompts where the core contains rare or unusual tokens (e.g., 'a zebra painting a fence'), we flag them as potential underestimation and recommend using an alternative core extraction (e.g., LLM-based) that better separates generic and specific content.

## Contribution

(1) A zero-shot reward calibration method that uses a minimal prompt contrastive baseline to improve alignment by subtracting generic prior. (2) The finding that simple rule-based core extraction suffices to significantly improve reward quality over raw likelihood. (3) An open-source implementation of the core extractor and reward computation.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|-----------------------|
| Dataset | PartiPrompts extended with Pick-a-Pic, HPD v2 | Accelerate evaluation with existing human preference data |
| Primary metric | Human Preference Accuracy (HPA) | Directly measures alignment with humans |
| Baseline 1 | CLIPScore (CLIP) | Unimodal alignment baseline |
| Baseline 2 | ImageReward | Supervised reward model trained on human preferences |
| Baseline 3 | L_full without contrast (ablation) | Isolates effect of contrastive subtraction |
| Ablation-of-ours | CLC with alternative core extraction: (a) LLM-based (GPT-4o), (b) random core, (c) contradictory core (e.g., opposite subject) | Tests robustness of contrastive mechanism to core definition |
| Downstream task | Best-of-16 image selection using CLC reward | Demonstrates practical utility beyond HPA |

### Why this setup validates the claim
This combination of datasets, baselines, and metrics directly tests the central claim that CLC's contrastive mechanism better captures instruction-specific alignment than existing approaches. PartiPrompts offers diverse prompts covering various compositional and stylistic instructions, enabling evaluation on the very scenarios where CLC is predicted to excel (e.g., prompts where the core+modifiers structure matters). The addition of Pick-a-Pic and HPD v2 provides larger-scale human preference data for more reliable and faster comparisons. Human Preference Accuracy (HPA) is the gold standard because it directly measures agreement with human judgments, which is the ultimate goal of a reward model. The baselines test distinct sub-claims: CLIPScore reveals whether simple unimodal similarity suffices; ImageReward tests if supervised training on human preferences provides an advantage (and whether CLC can rival it zero-shot). The ablation (CLC without contrast) isolates the effect of the core subtraction, validating that the contrastive design is essential. The alternative core ablations (LLM-based, random, contradictory) explicitly test the load-bearing assumption that the minimal core captures generic priors; if CLC performs well across these variations, the contrastive mechanism is robust. Finally, the downstream task shows that CLC reward can translate to real-world image generation selection, demonstrating practical significance. The metric is sensitive to the predicted effect because HPA reflects nuanced alignment that reward models must capture; a model that overweights generic features or misses instruction-specific details will receive lower HPA. Thus, if CLC outperforms baselines on relevant subsets (e.g., compositional or uncommon prompts), it confirms our mechanism.

### Expected outcome and causal chain

**vs. CLIPScore** — On a case with compositional reversal (e.g., "a dog chasing a cat" vs "a cat chasing a dog"), CLIPScore assigns similar scores to both images because it independently matches objects without ordering. Our method uses MLLM likelihood conditioned on the full prompt, which captures word order and relational structure, so it correctly prefers the matching image. We expect a noticeable gap on compositional prompts (e.g., +10% HPA) but parity on simple prompts with few modifiers.

**vs. ImageReward** — On a case with an uncommon instruction (e.g., "a blue car with pink polka dots"), ImageReward may assign low reward because such combinations are rare in its training data. Our method is zero-shot and relies on the MLLM's flexible conditional likelihood, so it can handle novel modifiers. We expect a larger improvement on out-of-distribution prompts (e.g., +15% HPA) than on common ones (e.g., +5%).

**vs. L_full without contrast** — On a case where the core is needed (e.g., "a dog sits on a bench" — the core "dog sits" may yield high likelihood for any dog-sitting image, but the full prompt adds location detail), L_full alone rewards any dog-sitting image, even without the bench. CLC subtracts core likelihood, so it rewards only images that specifically include the bench. We expect a clear advantage on prompts with critical non-core modifiers (e.g., +20% HPA on such prompts).

**Alternative core ablations** — For LLM-based core extraction, we expect similar HPA as rule-based, validating that contrastive mechanism is not tied to extraction method. For random core (where core is a random subset of prompt tokens), we expect CLC performance to degrade to near L_full, confirming that meaningful core extraction matters. For contradictory core (e.g., opposite subject), CLC may assign negative rewards to correct images, illustrating the assumption's boundary; we expect this to occur only rarely.

### What would falsify this idea
If CLC's HPA is not significantly higher than L_full alone on prompts where the core omits critical modifiers, or if CLC fails to outperform CLIPScore on compositional prompts, then the contrastive subtraction is not effectively isolating instruction-specific alignment. A uniform gain across all subsets would also falsify the claim that the advantage stems from the predicted failure mode. Moreover, if CLC with random core still outperforms L_full, it would suggest that the contrastive effect is not due to meaningful core extraction but to some other factor (e.g., length normalization).

## References

1. Read It Back: Pretrained MLLMs Are Zero-Shot Reward Models for Text-to-Image Generation
2. SRUM: Fine-Grained Self-Rewarding for Unified Multimodal Models
3. Uniworld-V2: Reinforce Image Editing with Diffusion Negative-aware Finetuning and MLLM Implicit Feedback
4. RewardDance: Reward Scaling in Visual Generation
5. Emu3.5: Native Multimodal Models are World Learners
6. VisionReward: Fine-Grained Multi-Dimensional Human Preference Learning for Image and Video Generation
7. Mixture-of-Transformers: A Sparse and Scalable Architecture for Multi-Modal Foundation Models
8. LMFusion: Adapting Pretrained Language Models for Multimodal Generation
