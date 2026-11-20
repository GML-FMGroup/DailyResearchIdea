# Shared-Token Bidirectional Grounding for Unified Vision-Language Understanding and Generation

## Motivation

Unified multimodal models such as HiUni achieve extensive parameter sharing by reusing a pretrained language-aligned ViT and Transformer backbone for both understanding and generation, but the shared representation inherits only the coarse, understanding-side patch-level alignment from the pretrained VLM (E1/E8 show caption-loss projectors yield weak patch-word alignment), while output-side modules—the Pixel Diffusion Head and HiGate—cannot retroactively impose patch-to-word correspondence on the backbone. The structural gap recurs across the field: understanding-side alignment work (E1, E8) and generation-side unification work (E6) each optimize their own surface without enforcing a shared invariant on the tokens both branches consume, so generation-side spatial fidelity and understanding-side grounding remain decoupled even when the backbone is nominally shared.

## Key Insight

A contrastive patch-word objective applied to the shared backbone tokens before the output-side branches establishes a single bidirectional grounding invariant—each visual token must be predictive of its semantic words and each word of its patches—so that spatial control for generation and referential grounding for understanding are constrained to be readouts of the same token geometry rather than separately acquired capabilities.

## Method

**A. What it is.** We introduce **BiGround**, a training objective and lightweight adapter that sits on HiUni's shared ViT/Transformer token stream, before the Pixel Diffusion Head and HiGate. Inputs: paired (image, caption, region-box, region-word) tuples. Outputs: a shared token matrix that is simultaneously optimized for (i) the original understanding loss, (ii) the original pixel-diffusion generation loss, and (iii) a bidirectional patch-word contrastive loss computed on the shared tokens. **Load-bearing assumption (stated explicitly):** BiGround assumes that a symmetric patch-word contrastive loss applied to the shared backbone tokens upstream of HiUni's Pixel Diffusion Head and HiGate will propagate into both branches' outputs—i.e., that the generation branch's spatial fidelity is a readout of the grounded token geometry rather than being dominated by HiGate's own bypass spatial pathway. This assumption is load-bearing because HiUni (E6) introduces HiGate precisely to supply fine spatial details that the shared tokens lack; a loss on those shared tokens cannot constrain a pathway designed to route around them. To make this assumption testable rather than assumed, we add a method-level control (HiGate-severed ablation) and a direct measurement (gradient alignment) inside the existing training loop, as specified in B and D.

**B. How it works.**

```pseudocode
# Shared backbone from HiUni: ViT + dense Transformer, tokens T = [t_1..t_N]
# Output-side branches (frozen interface): PixelDiffusionHead, HiGate

for each batch of (image I, caption C, boxes B, region-words W):
    T = SharedBackbone(I)                      # N patch tokens, dim d
    # --- Bidirectional grounding on shared tokens ---
    Z_v = PatchProject(T)                      # linear, d -> 256, L2-normalized
    Z_w = WordEncoder(C, W)                    # frozen text encoder, 256-d, L2-normalized
    S = Z_v @ Z_w.T / tau                      # tau = 0.07, patch x word logits
    # positive pairs from box-word correspondence; negatives = all other pairs
    L_pw = symmetric_CE(S, M_pos)              # patch->word
    L_wp = symmetric_CE(S.T, M_pos.T)          # word->patch
    L_ground = 0.5*(L_pw + L_wp)
    # --- Preserve original branches ---
    L_und = UnderstandingLoss(T, C)            # token-level LM loss
    L_gen = DiffusionLoss(PixelDiffusionHead(T), I) + HiGateRecon(T, I)
    L = L_und + L_gen + lambda_g * L_ground     # lambda_g = 0.5, warmup 2000 steps
    update SharedBackbone, PatchProject only

# --- HiGate-severed control (method-level ablation, run as a separate training job) ---
# Identical to above except HiGateRecon(T, I) is replaced by HiGateRecon(detach(T), I)
# so that gradients from the reconstruction path do not reach the shared tokens.
# This isolates whether generation-side gains come from the shared-token invariant
# or from HiGate's independent spatial channel.

# --- Gradient-alignment diagnostic (logged every 500 steps, no gradient update) ---
# g_ground = d L_ground / d T ; g_gen = d L_gen / d T ; g_und = d L_und / d T
# cos_ground_gen = cosine(g_ground, g_gen) ; cos_ground_und = cosine(g_ground, g_und)
# Report mean over batch; values near 0 indicate the grounded subspace is not used by that branch.
```

Diagram of token flow:

```
image -> ViT -> dense Transformer -> shared tokens T
                                        |---> Understanding head (LM loss)
                                        |---> PatchProject -> contrastive vs WordEncoder (BiGround loss)
                                        |---> PixelDiffusionHead (gen loss)
                                        |---> HiGate (recon loss)
```

Hyperparameters: projection dim 256, temperature tau=0.07, grounding weight lambda_g=0.5 with 2000-step linear warmup, batch region-word pairs per image up to 8, symmetric cross-entropy over in-batch negatives. Gradient-alignment diagnostic logged every 500 steps. HiGate-severed control uses the same hyperparameters with detach(T) on the HiGate input.

**C. Why this design.** We chose to apply the grounding loss on the shared tokens *before* the output-side branches rather than on the Pixel Diffusion Head's internal features, because HiUni's design confines task-specific computation to the output side and any loss placed inside that head would be invisible to the understanding pathway—accepting the cost that the shared tokens must now serve two objectives and may need a longer warmup. We chose a symmetric patch→word and word→patch contrastive objective over a one-directional captioning loss because E1/E8 show caption-loss projectors produce only weak, coarse patch-level alignment; the reverse direction (word→patch) is what forces each semantic word to localize, which is the property generation-side spatial control needs, accepting the cost of requiring region-word supervision rather than captions alone. We chose a frozen text encoder as the word anchor rather than a learned word projector, because a learned projector can collapse the word side and trivially satisfy the contrastive objective without grounding—accepting the cost that the frozen encoder's vocabulary coverage bounds which words can be grounded. We chose to keep the original understanding and generation losses active with a warmup-weighted grounding term rather than a staged two-phase schedule, because staged training risks catastrophic drift in the shared backbone between phases—accepting the cost that early grounding gradients are noisy while the projection head is still random. A domain expert would not describe this as a variant of CLIP-style image-text contrastive pretraining, because CLIP aligns a single global image embedding to a caption, whereas BiGround operates on per-patch shared tokens inside a generative backbone and enforces the reverse word→patch direction that CLIP never computes. To make the load-bearing assumption testable, we add a HiGate-severed control (detach(T) on the HiGate input) and a gradient-alignment diagnostic (cosine between dL_ground/dT and dL_gen/dT, dL_und/dT) inside the same training loop; these do not change the core training procedure but measure whether the grounded token subspace is actually used by each branch.

**D. Why it measures what we claim.** The symmetric contrastive loss L_ground measures **bidirectional patch-word grounding** because the assumption that box-word correspondences define the positive set makes the patch→word direction equivalent to referential localization and the word→patch direction equivalent to spatial control; this assumption fails when a word's referent is not box-annotated (e.g., background adjectives, relational words), in which case L_ground reflects spurious alignment to the nearest annotated region rather than true grounding. The shared-token placement of the loss measures **cross-branch coupling** because the assumption that both output heads read the same token matrix makes any grounding improvement visible to both branches; this assumption fails when HiGate's reconstruction path bypasses the grounded token subspace, in which case generation-side fidelity improvements reflect HiGate's own spatial pathway rather than the shared invariant. We operationalize this assumption with two concrete measurements: (i) the cosine similarity between dL_ground/dT and dL_gen/dT (and dL_und/dT) on shared tokens, logged every 500 steps, where values near zero indicate the grounded subspace is not used by that branch; and (ii) the HiGate-severed control (detach(T) on the HiGate input), where any generation-side gain that survives severing is attributable to the shared-token invariant rather than to HiGate's bypass. The frozen WordEncoder anchor measures **semantic stability** because the assumption that the frozen encoder's embedding geometry is fixed makes contrastive drift attributable to the visual side; this assumption fails for out-of-vocabulary medical or technical terms, in which case the loss measures alignment to a nearest in-vocabulary neighbor rather than the intended concept.

## Contribution

['BiGround: a bidirectional patch-word contrastive objective plus lightweight projection adapter applied to the shared backbone tokens of a unified understanding-generation model, with a warmup-weighted multi-task schedule that preserves both original branches.', 'A design principle: grounding on shared tokens before output-side task-specific modules is the locus that couples referential understanding and spatial generation, as opposed to grounding inside either branch.', 'A paired image-caption-region-word training recipe compatible with HiUni-style shared backbones, including the frozen-encoder anchoring choice that prevents word-side collapse.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | RefCOCO/RefCOCO+/RefCOCOg referring-expression grounding + Visual Genome region captions with box-word labels | Provides region-word supervision needed for bidirectional loss |
| Primary metric | Word→patch localization accuracy (pointing game, top-1 patch within box) | Directly probes reverse direction CLIP never computes |
| Secondary metric | Gradient alignment: cosine(dL_ground/dT, dL_gen/dT) and cosine(dL_ground/dT, dL_und/dT) on shared tokens, logged every 500 steps | Directly measures whether grounded subspace is used by each branch |
| Baseline | Pretrained VLM (HiUni's init) | Tests whether grounding improves over unmodified shared backbone |
| Baseline | Unidirectional patch→word contrastive (E1-style) | Tests whether reverse direction adds localization beyond caption-loss alignment |
| Baseline | HiUni with generation-only fine-tuning | Tests whether gains come from coupling, not extra training |
| Ablation-of-ours | BiGround with frozen WordEncoder replaced by learned projector | Tests whether frozen anchor prevents word-side collapse |
| Ablation-of-ours | BiGround with HiGate-severed control (detach(T) on HiGate input) | Isolates whether generation-side gains come from shared-token invariant or HiGate bypass |
| Ablation-of-ours | BiGround with grounding loss applied to Pixel Diffusion Head internal features instead of shared tokens | Tests whether shared-token placement is the novel ingredient |

### Why this setup validates the claim

The central claim is that a symmetric patch-word contrastive loss on HiUni's shared tokens produces bidirectional grounding that is visible to both the understanding and generation branches, and that the reverse word→patch direction is what enables spatial control. The pretrained VLM baseline tests sub-claim (a): that grounding on shared tokens improves over an ungrounded backbone. The unidirectional patch→word baseline tests sub-claim (b): that the reverse direction is not redundant with caption-loss-style alignment — if our symmetric loss only matched it, the E1/E8 critique would apply and the reverse direction would be unnecessary. The generation-only baseline tests sub-claim (c): that coupling through shared tokens, not merely more gradient steps, drives the gain. The learned-projector ablation tests the frozen-anchor design choice by checking whether a learned word side collapses the contrastive objective. The HiGate-severed control directly tests the load-bearing assumption: if generation-side gains survive severing HiGate's gradient path to the shared tokens, they are attributable to the shared-token invariant; if they vanish, HiGate's bypass dominates. The gradient-alignment diagnostic converts the 'visible to both branches' claim from an assumption into a measured quantity: cosine similarity near zero between dL_ground/dT and dL_gen/dT would indicate the grounded subspace is not used by the generation branch. The Pixel-Diffusion-Head-internal ablation tests whether the shared-token placement—not the contrastive loss itself—is what produces cross-branch coupling. Localization accuracy is the right primary metric because it is the only measure that directly reads the word→patch direction; captioning quality and generation fidelity are secondary signals that could improve for reasons unrelated to grounding.

### Expected outcome and causal chain

**vs. Pretrained VLM (HiUni init)** — On a case where a caption names a small object in a cluttered scene, the baseline produces a fluent caption but its shared tokens carry only coarse, globally-averaged visual features because caption loss never forces per-patch discrimination (E1/E8). Our method instead pushes each region-word pair's patch tokens toward their word embedding and away from all other words, because the symmetric contrastive loss operates directly on per-patch shared tokens. We expect a clear localization-accuracy gap on cluttered small-object subsets, with near-parity on single-dominant-object images where coarse alignment already suffices.

**vs. Unidirectional patch→word contrastive (E1-style)** — On a case where the word "handle" must bind to the correct sub-region of a mug, the unidirectional baseline produces patch→word scores that are high for many patches sharing the word's broad semantics, because it only asks which word each patch matches, never which patch each word selects. Our method instead computes the transpose direction, forcing each word embedding to concentrate mass on its box-annotated patches, because the word→patch cross-entropy penalizes diffuse word-side attention. We expect a noticeable gap on part-level and relational referring expressions, with parity on whole-object referring where both directions coincide.

**vs. HiUni with generation-only fine-tuning** — On a case where a generated image must place an object at a specified location, the baseline produces plausible pixels but drifts from the requested region because its generation head receives tokens never shaped by region-word correspondence. Our method instead shapes the shared token matrix before both heads, because the grounding loss sits upstream of the Pixel Diffusion Head and HiGate. We expect generation-side spatial fidelity to improve specifically on position-specified prompts, while unconditional generation quality stays flat — the diagnostic signature of coupling rather than capacity. The HiGate-severed control and gradient-alignment diagnostic will confirm whether this gain is attributable to the shared-token invariant (cosine(dL_ground/dT, dL_gen/dT) > 0 and gain persists under severing) or to HiGate's bypass (cosine near zero and gain vanishes under severing).

**vs. BiGround with HiGate-severed control** — On the same position-specified generation prompts, the severed control receives identical grounding gradients on shared tokens but HiGate's reconstruction path is detached from them. If our load-bearing assumption holds, the severed control should show a smaller but nonzero generation-side gain, and the gradient-alignment diagnostic should show cosine(dL_ground/dT, dL_gen/dT) > 0. If the severed control matches the full model, HiGate's bypass dominates and the shared-token invariant is not the source of generation fidelity.

**vs. BiGround with grounding loss on Pixel Diffusion Head internal features** — On part-level referring expressions, this ablation tests whether shared-token placement is the novel ingredient. If the internal-feature variant matches BiGround on localization accuracy but fails to improve understanding-side grounding, the cross-branch coupling claim is supported; if it matches on both, the shared-token placement is not necessary.

### What would falsify this idea

If the symmetric loss's gain over the unidirectional baseline is uniform across part-level, relational, and whole-object referring subsets rather than concentrated where word→patch localization is required, the reverse-direction claim is wrong. Likewise, if the learned-projector ablation matches the frozen anchor without collapse, the frozen-encoder design rationale is unsupported. If the HiGate-severed control matches the full model on generation-side spatial fidelity and cosine(dL_ground/dT, dL_gen/dT) is near zero, the load-bearing assumption that generation fidelity is a readout of the grounded token geometry is false.

## References

1. HiUni: Harmonizing Understanding and Generation in Shared Visual Space
