# CompoNeRF: Category-Agnostic Generative 3D Scene Synthesis via Composable Proto-Shape Implicit Fields

## Motivation

Existing 3D GANs require category-specific pre-training and cannot generalize to novel object classes or multi-object scenes with arbitrary compositions. For example, ZIGNeRF and NeRFInvertor both rely on a pre-trained 3D GAN per object category (e.g., faces, cats), limiting their applicability to open-set scenarios. The root cause is that current generators entangle geometry and appearance in a monolithic latent space, preventing the reuse of shape primitives across categories.

## Key Insight

The joint distribution of geometry and appearance over an open set of categories can be factorized into a finite set of learnable proto-shapes that are spatialized by a compositional arrangement network, because objects across categories share common structural primitives (e.g., spheres, cylinders, or parts).

## Method

### CompoNeRF: Category-Agnostic Generative 3D Scene Synthesis via Composable Proto-Shape Implicit Fields

**(A) What it is:** We propose **CompoNeRF**, a generative adversarial framework that learns a set of learnable proto-shapes (small NeRFs) and a spatial arrangement network that composes them to form arbitrary 3D scenes. The generator takes a latent code and outputs a set of transformations (translation, rotation, scale) for each proto-shape, which are then composed via volume rendering. The discriminator distinguishes real vs. synthesized rendered views of full scenes (single or multi-object). Training is end-to-end on unlabeled multi-view images of scenes containing objects from various categories.

**(B) How it works** (pseudocode):
```pseudocode
# Proto-Shape Set: S = {S_1, ..., S_K}, each S_k is a small NeRF with architecture:
#   - 4-layer MLP with skip connection at layer 2, hidden dim=256, ReLU activations.
#   - Input: position (3D) + view direction (2D spherical coordinates) with positional encoding (6 frequencies for pos, 4 for dir).
#   - Output: density sigma_k (linear), color c_k (3D, sigmoid).
# Hyperparameter: K=32 proto-shapes.

# Spatial Arrangement Network: A = MLP(L) -> {T_1,...,T_K} where T_k = (translation t_k in R^3, rotation R_k in R^6 (continuous 6D representation), scale s_k in R^3).
# Architecture: 2 hidden layers, 512 dims each, LeakyReLU (0.2). Output dimension = K*(3+6+3).

# Scene Renderer: For a given camera pose, sample 128 stratified points along rays. For each point, transform it into each proto-shape's canonical space using T_k, query all S_k to get (sigma_k, c_k). Combine densities: sigma = sum_k sigma_k. Combine colors: c = (1/sigma) * sum_k sigma_k * c_k (weighted sum, assuming sigma > 0). Then volume rendering via quadrature to produce pixel color.

Training loop:
1. Sample latent code L ~ N(0,I) (dim=512).
2. Compute arrangements {T_k} = A(L).
3. For each proto-shape S_k, sample 128 points along rays, compute density sigma_k and color c_k.
4. Combine and render image I_rendered via volume rendering (128 coarse samples, 64 fine samples via importance sampling).
5. Discriminator D(I_rendered) vs. real image I_real (from dataset of unposed multi-view images of various scenes). D is a convolutional network with spectral normalization.
6. Update generator (proto-shapes + arrangement network) and discriminator via non-saturating GAN loss with R1 gradient penalty (gamma=10).
7. Optional: add regularization to encourage sparse usage of proto-shapes:
   - L1 penalty on per-point sigma_k, weight=0.001.
   - L1 penalty on sum_k sigma_k at each point (to reduce overlap), weight=0.01.
8. Training details: batch size 16, learning rate 0.0002, Adam optimizer (betas=(0,0.9)), 500k iterations on 4 A100 GPUs (~5 days).
```

**(C) Why this design:** We chose to learn proto-shapes as small NeRFs rather than using a single large NeRF because decomposition into reusable primitives enables generalization across categories without retraining. This design accepts the cost that scenes with truly novel shape structures (not decomposable into learned primitives) may be poorly modeled. We chose a spatial arrangement network that outputs explicit 6-DoF transformations rather than implicit composition (e.g., via attention) because explicit transformations allow direct manipulation and are easier to regularize for spatial consistency. The trade-off is that the arrangement network must handle occlusions and complex spatial relations, requiring a powerful enough MLP. We chose to combine densities by simple sum rather than more complex operations (e.g., max) because sum is differentiable and preserves the probability interpretation of volume density; however, it can lead to double-counting at overlapping locations, which we mitigate by introducing a soft-occupancy regularizer. We chose not to use a slot-based attention mechanism (as in some object-centric scene representations) because the fixed set of proto-shapes with learnable arrangement provides a simpler inductive bias for the shapes themselves, at the cost of not being able to handle an arbitrary number of objects beyond K. We accept that K must be chosen large enough to cover typical scenes. **This design assumes that a fixed set of K=32 proto-shapes can represent the geometry and appearance of objects from any category, because objects share common structural primitives. We verify this assumption by measuring primitive reuse across categories (see experiment).**

**(D) Why it measures what we claim:** The combination of proto-shapes and arrangement network operationalizes the idea of a category-agnostic generative prior. Specifically, the set of learnable proto-shapes, when trained jointly on scenes from multiple categories, must capture reusable geometric and appearance primitives that are common across categories. The spatial arrangement network, which outputs transformations for each proto-shape, measures the compositional structure of a scene because it specifies how to spatially combine these primitives to form the observed scene. The sum of densities from all proto-shapes measures the total occupancy of the scene under the assumption that proto-shapes are independent in their occupancy contributions; this assumption fails when two proto-shapes are placed at the same location (overlapping), in which case the sum density is doubled and may occlude others. To prevent this, we enforce a scene-level prior that encourages sparse activation of proto-shapes per point via an L1 penalty on the per-point weight of each proto-shape. The discrimination loss measures the realism of the rendered scene, which indirectly tests whether the compositional decomposition yields plausible geometry and appearance. The GAN training ensures that the generator's distribution matches the real data distribution, thus measuring the ability to generate scenes with arbitrary objects by composition. Additionally, we explicitly measure primitive reuse: for each proto-shape, compute its average density on a validation set of single-object scenes per category; a proto-shape is considered reused if its density exceeds a threshold (0.1) on at least two categories.

## Contribution

(1) We introduce CompoNeRF, the first category-agnostic generative 3D scene synthesis framework that learns to compose scenes from a shared set of learnable proto-shapes without requiring category-specific pre-training. (2) We demonstrate that a spatial arrangement network can effectively parameterize the locations, orientations, and scales of proto-shapes, enabling the generation of both single- and multi-object scenes with diverse object categories from a single generator. (3) We provide a method for training such a compositional generator end-to-end using only unposed multi-view images, using volume rendering and adversarial loss, along with regularization to encourage sparse and meaningful decomposition.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|---|---|---|
| Dataset | ShapeNet SRN (cars, chairs, airplanes, tables, lamps, sofas) + Blender multi-object composites | Diverse categories and compositions; real multi-category testbed. |
| Primary metric | FID on rendered views (128x128) | Standard measure of generative realism. |
| Secondary metric | Chamfer distance (CD) between predicted object centroids and ground truth | Direct measure of compositional accuracy in multi-object scenes. |
| Baseline 1 | Monolithic NeRF-GAN (GRAF) | Tests necessity of compositional decomposition. |
| Baseline 2 | Object-specific Template GAN (GIRAFFE) | Tests advantage of shared primitives. |
| Baseline 3 | Slot-Attention NeRF (OC-NeRF) | Tests explicit spatial arrangements. |
| Baseline 4 | Mixture of NeRFs (Component NeRF) | Tests effect of learnable shared primitives vs. per-scene components. |
| Ablation-of-ours | Fixed primitive shapes (spheres with learnable appearance) | Tests benefit of learnable proto-shapes. |
| Ablation-of-ours | Proto-shapes with K=8 and weight sharing | Tests sensitivity to K and weight tying. |
| Ablation-of-ours | No sparsity regularizer | Shows necessity of sparsity loss. |

### Why this setup validates the claim

This experimental design directly tests the central claim that learnable proto-shapes with an explicit arrangement network enable category-agnostic scene synthesis. The multi-category dataset ensures that the model must generalize across object types, while the baselines isolate different mechanisms: monolithic NeRF lacks composition, object-specific templates prevent part sharing, slot-attention omits explicit transforms, and mixture of NeRFs lacks shared primitives. The fixed-primitive ablation disentangles the effect of learning primitives from the compositional framework. The K=8 weight-sharing ablation investigates expressiveness vs. efficiency. The sparsity ablation tests the regularizer's role in preventing overlap. FID measures realism of generated renderings, and Chamfer distance directly evaluates spatial arrangement accuracy. Additionally, we compute primitive reuse: for each proto-shape, we tally the number of categories where its average density exceeds a threshold (0.1) on validation single-object renders. High reuse (e.g., average >2 categories per proto-shape) would confirm that the fixed set captures cross-category primitives. If reuse is low, the assumption fails. This concretely measures the factorization claim.

### Expected outcome and causal chain

**vs. Monolithic NeRF-GAN** — On a case where a scene contains a novel combination of two object categories (e.g., a chair and a car together), the monolithic NeRF-GAN produces unrealistic or missing shapes because it encodes all geometry in a single network without decomposition, forcing it to memorize seen category-specific structures. Our method instead composes the scene from reusable proto-shapes (e.g., a leg primitive for the chair and a wheel primitive for the car) because its generator learns shape parts that transfer across categories, so we expect a noticeable FID gap on scenes with multiple categories, especially unseen combinations, while parity on single-category scenes.

**vs. Object-specific Template GAN (GIRAFFE)** — On a case where a stool shares leg shape with a chair (i.e., cross-category part reuse), GIRAFFE learns separate templates per category, so the stool's legs must be reconstructed from scratch, leading to higher sample complexity and potentially unrealistic shapes. Our method, by sharing proto-shapes across categories, can reuse the leg primitive, producing a plausible stool with fewer data. Thus, we expect our method to achieve lower FID on categories with overlapping parts, but similar FID on categories with unique shapes.

**vs. Slot-Attention NeRF (OC-NeRF)** — On a case where objects have precise spatial relations (e.g., a ball stacked on a cube), OC-NeRF uses attention to assign slots, which can misplace or merge objects due to unconstrained attention, resulting in inaccurate geometry or occlusions. Our method uses explicit transformations (translation, rotation, scale) output by the arrangement network, ensuring exact placement. Therefore, we expect our method to produce sharper renderings with correct occlusions, reflected in a lower FID on scenes requiring precise stacking or interlocking, while being similar on simple side-by-side arrangements.

**vs. Mixture of NeRFs (Component NeRF)** — On scenes where parts are shared across categories (e.g., chair legs and table legs), Mixture of NeRFs uses per-scene latent codes for each component, so it does not explicitly reuse primitives across categories. Our method's proto-shapes are reused across scenes, leading to more consistent geometry and lower FID on categories with shared parts, while Mixture of NeRFs may perform similarly on unique shapes. The Chamfer distance on multi-object scenes should favor our method due to explicit transforms.

### What would falsify this idea

If the FID improvement over GIRAFFE is uniform across all scene types rather than concentrated on cross-category part reuse, then the advantage is not due to shared primitives. Alternatively, if our method fails on scenes requiring extensive part sharing (e.g., novel combinations of parts), or if primitive reuse metric shows low cross-category activation (<1 category per proto-shape on average), the central claim of reusable proto-shapes is invalid.

## References

1. Zero-Shot 3D Scene Representation With Invertible Generative Neural Radiance Fields
2. NeRFInvertor: High Fidelity NeRF-GAN Inversion for Single-Shot Real Image Animation
