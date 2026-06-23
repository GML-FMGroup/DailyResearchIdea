# DOPA: Depth-Order-Preserving Attention for Occlusion-Aware Counting in Vision-Language Models

## Motivation

Existing VLMs evaluated on CAPTURe rely solely on RGB inputs and systematically undercount occluded objects because they lack 3D spatial priors that encode depth ordering. Without metric depth, the model cannot distinguish whether a partially visible object is behind or in front of an occluder, leading to failures in occlusion reasoning. While Metric3D v2 provides consistent metric depth, naively adding depth as a fourth channel disrupts the pretrained visual encoder's distribution, as demonstrated by the distribution shift problem.

## Key Insight

Metric depth provides an invariant ordering of objects along the line of sight that is orthogonal to appearance, and this ordering can be propagated into the language decoder via a cross-modal attention mechanism that preserves relative depth rankings without requiring per-object depth estimation.

## Method

**DOPA: Depth-Order-Preserving Attention**

(A) **What it is:** DOPA is a cross-modal attention module inserted between the vision encoder and language decoder of a VLM. It takes RGB visual features (V, shape B×N×C) and a metric depth map (D, shape B×H×W) from Metric3D v2 as inputs, and outputs enriched features that encode depth ordering into the language model's hidden states.

**Load-bearing assumption:** The metric depth map from Metric3D v2 provides an accurate relative depth ordering for all objects in occluded scenes, even for thin, reflective, or distant objects. This assumption is critical because the depth ordering matrix D_order directly depends on accurate depth differences. To mitigate this, we replace the hard sign-based D_order with a soft, confidence-weighted ordering derived from a lightweight confidence predictor.

(B) **How it works:**
```python
# Inputs: V (RGB features, BxNxC), D (metric depth, BxHxW) where N=H*W
import torch
import torch.nn.functional as F

# --- Smallest repair: confidence-weighted ordering ---
# Step 0: Predict per-pixel confidence from depth map using a small CNN
# Input: D (Bx1xHxW) -> 3-layer CNN with ReLU, hidden channels [16, 8, 1], kernel 3, padding 1
conf_net = nn.Sequential(
    nn.Conv2d(1, 16, 3, padding=1),
    nn.ReLU(),
    nn.Conv2d(16, 8, 3, padding=1),
    nn.ReLU(),
    nn.Conv2d(8, 1, 3, padding=1),
    nn.Sigmoid()  # output in [0,1]
)  # (Bx1xHxW)
C = conf_net(D.unsqueeze(1))  # Bx1xHxW, learned confidence
C_flat = C.view(B, N)  # BxN

# Step 1: Depth encoding and normalization
D_flat = D.view(B, N)  # BxN
min_d = D_flat.min(dim=1, keepdim=True).values
max_d = D_flat.max(dim=1, keepdim=True).values
D_norm = (D_flat - min_d) / (max_d - min_d + 1e-8)  # scale to [0,1]

# Step 2: Depth ordering matrix (soft, confidence-weighted)
D_diff = D_norm.unsqueeze(2) - D_norm.unsqueeze(1)  # BxNxN
# Soft sign using tanh with temperature tau
soft_order = torch.tanh(D_diff / 0.1)  # tau=0.1, range (-1,1)
# Weight by minimum confidence of each pair
C_min = torch.min(C_flat.unsqueeze(2), C_flat.unsqueeze(1))  # BxNxN
D_order = soft_order * C_min  # BxNxN; unreliable pairs are downweighted

# Step 3: Cross-modal attention with depth ordering bias
Q = V @ W_Q    # W_Q shape C x d, d=64
K = V @ W_K    # W_K shape C x d
scores = Q @ K.transpose(1,2) / math.sqrt(d)  # BxNxN
# Inject learnable depth ordering bias
alpha = nn.Parameter(torch.tensor(0.1))  # initialized to 0.1
scores = scores + alpha * D_order
attn_weights = F.softmax(scores, dim=-1)  # BxNxN

# Step 4: Enriched feature by concatenating attended RGB and depth-modulated features
V_depth = D_norm.unsqueeze(-1) * V  # element-wise modulation: BxNxC
E_rgb = attn_weights @ V            # BxNxC
E_dep = attn_weights @ V_depth      # BxNxC
E = torch.cat([E_rgb, E_dep], dim=-1)  # BxNx2C

# Step 5: Project to language decoder dimension
W_proj = nn.Linear(2*C, C)  # learnable
E_out = W_proj(E)           # BxNxC
```
Hyperparameters: attention dimension d=64, alpha initialized to 0.1 and learnable, confidence network: 3 Conv2d layers with ReLU hidden 16,8 and sigmoid output.

(C) **Why this design:** We chose a cross-modal attention with depth ordering bias (D_order) over naive depth-channel concatenation because concatenation does not enforce spatial reasoning; the model could learn to ignore the depth channel. By injecting a pairwise ordering bias directly into attention scores, we force the model to attend to pixels based on their relative depth, which is critical for occlusion reasoning (e.g., counting objects behind an occluder requires ignoring the occluder while considering the pattern of partially visible objects). We used a learnable alpha over a fixed large bias to allow the model to calibrate the importance of depth ordering relative to appearance similarity; the trade-off is that very poor depth estimates could misguide attention if alpha becomes large. We concatenated both attended RGB and attended depth-modulated features (rather than only depth-modulated ones) to preserve appearance details needed for object identification, at the cost of doubling the feature dimension and increasing computational overhead. We avoided a full two-stream architecture (separate RGB and depth streams fused late) because it would double parameters and risk overfitting on limited occlusion data; the single-stream design with cross-modal bias keeps the pretrained vision encoder intact and is more parameter-efficient. The confidence weighting addresses the load-bearing assumption by downweighting unreliable depth regions, making the ordering bias robust to noisy depth estimates.

(D) **Why it measures what we claim:** The depth ordering bias D_order operationalizes the invariant of depth ordering: the computational quantity D_diff (and its sign) measures relative depth ordering, which is a proxy for occlusion relationships because closer objects occlude farther ones; this assumption fails when objects are at the same depth and occlude each other (e.g., overlapping), where D_order treats them equally and the model may miss occlusion relationships. The enriched feature E is formed by concatenating attended RGB features E_rgb and attended depth-modulated features E_dep: E_dep measures the integration of spatial ordering with appearance by weighting each pixel's appearance by its normalized depth (D_norm), assuming that objects at different depths are independent in appearance; this assumption fails when two objects at the same depth occlude each other (e.g., overlapping), causing the modulation to treat them similarly and potentially miss occlusion relationships. The attention weights themselves measure the model's learned affinity between pixel pairs under the joint influence of appearance similarity and depth ordering; this captures the motivation-level concept of 'spatial-relation-enriched feature representation' because it encodes both which pixels are similar in appearance and how they are arranged in depth.

**Change summary (warnings):** The load-bearing assumption about Metric3D v2 accuracy is fatal — we added a confidence weighting mechanism as smallest repair, but this is a new module; if confidence prediction is unreliable, the method may still fail. We mark this as WARN: load-bearing assumption is fatal — method needs redesign.

## Contribution

(1) A depth-order-preserving cross-modal attention mechanism (DOPA) that injects metric depth ordering into vision-language models without modifying the pretrained encoder or decoder, enabling occlusion-aware counting. (2) The design principle that explicit pairwise depth ordering bias, rather than depth as an additional input channel, is necessary to preserve the invariant spatial structure required for counting partially occluded objects. (3) An analysis of the failure modes of depth-based attention when depth estimates are inconsistent or when objects overlap at the same depth.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | CAPTUREreal | Real occluded scenes with ground-truth counts |
| Primary metric | Counting accuracy | Direct measure of occlusion reasoning |
| Baseline 1 | LLaVA-1.5 | Standard VLM without depth cues |
| Baseline 2 | BLIP-2 | Another VLM baseline |
| Baseline 3 | DepthConcat | VLM with naive depth concatenation |
| Ablation-of-ours | DOPA w/o depth ordering bias (also w/o confidence weighting) | Isolates bias contribution |
| Qualitative analysis | Attention map visualization | Grounding: correlate attention weights with depth ordering |

### Why this setup validates the claim
CAPTUREreal is designed to require reasoning about occluded objects, thus directly testing the central claim that depth ordering improves occlusion understanding. Comparing DOPA against standard VLMs (LLaVA-1.5, BLIP-2) isolates the benefit of any depth information, while comparing against DepthConcat isolates the specific advantage of the ordering bias over naive integration. The ablation (DOPA without bias) pinpoints whether the bias mechanism drives improvement. Counting accuracy is the appropriate metric because it demands precise spatial reasoning—a model must correctly identify and count objects behind occluders, which directly benefits from depth ordering. Additionally, we will visualize attention maps from DOPA and correlate them with depth ordering (e.g., show that attention weights are higher for pairs with consistent depth order) to further ground the claim. This combination of controlled baselines, targeted metric, and qualitative analysis creates a falsifiable test: if DOPA significantly outperforms DepthConcat on occluded object counts but not on non-occluded ones, the bias is effective.

### Expected outcome and causal chain

**vs. LLaVA-1.5** — On a scene where a cup is half-occluded by a bottle, LLaVA-1.5 counts only the bottle because it lacks depth awareness and attends mostly to visible areas. Our method uses depth ordering bias to prioritize pixels behind the occluder, attending to the cup's visible parts via ordering cues, so we expect a ~15-20% accuracy gap on occluded subsets but parity on unoccluded ones.

**vs. BLIP-2** — Similar failure pattern: BLIP-2's attention is appearance-driven, so on a shelf with a box in front of a toy, it misses the toy because its features are masked. DOPA's bias forces attention to depth-ordered regions, recovering the toy. Expect a noticeable gap (~10-15%) on scenes with moderate occlusion.

**vs. DepthConcat** — On a cluttered table with multiple overlapping objects, DepthConcat may learn to ignore depth or treat it as an extra channel, still miscounting due to shallow integration. DOPA's ordering bias directly enforces relational reasoning, so we expect a gap of ~5-10% on complex occluded scenes, with equal performance on simple scenes.

**Qualitative grounding:** Visualization of attention weights for occluded pixel pairs will show higher weights when depth ordering matches occlusion (closer attends to farther) compared to non-occluded pairs, confirming that the bias works as intended.

### What would falsify this idea
If DOPA shows no improvement over DepthConcat on occluded object counts, or the improvement is uniform across all difficulty levels, then the depth ordering bias is not providing the intended spatial reasoning advantage. Additionally, if attention map visualization reveals no correlation with depth ordering (e.g., occluded regions do not receive higher attention), the mechanism is not functioning as claimed.

## References

1. InternSpatial: A Comprehensive Dataset for Spatial Reasoning in Vision-Language Models
2. CAPTURe: Evaluating Spatial Reasoning in Vision Language Models via Occluded Object Counting
3. Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution
4. Are Deep Learning Models Robust to Partial Object Occlusion in Visual Recognition Tasks?
5. Metric3D v2: A Versatile Monocular Geometric Foundation Model for Zero-Shot Metric Depth and Surface Normal Estimation
6. Are We on the Right Way for Evaluating Large Vision-Language Models?
7. PonderV2: Pave the Way for 3D Foundation Model with A Universal Pre-training Paradigm
8. DG-Recon: Depth-Guided Neural 3D Scene Reconstruction
