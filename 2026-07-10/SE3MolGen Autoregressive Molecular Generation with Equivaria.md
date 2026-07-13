# SE3MolGen: Autoregressive Molecular Generation with Equivariant Latent 3D Conditioning

## Motivation

Existing autoregressive molecular generation models (e.g., DrugGen 2, DrugGPT) work on tokenized representations like SMILES, which inherently discard 3D geometric information. This leads to generated molecules that may be geometrically implausible or fail to capture steric and electronic constraints essential for bioactivity. The structural root cause is that token-level autoregressive decoding operates on a 1D sequence, with no mechanism to enforce 3D physical symmetries or spatial consistency during generation.

## Key Insight

By embedding an SE(3)-equivariant attention mechanism that operates on a dynamically updated latent 3D point cloud within the autoregressive decoder, the model preserves physical symmetries and ensures that each generation step is conditioned on a geometrically consistent partial structure, making the output inherently respect 3D constraints.

## Method

### (A) What it is
SE3MolGen is an autoregressive token decoder that, at each step, attends to a latent 3D point cloud representing the partial molecule. It outputs the next token (atom or fragment) and updates the 3D point cloud equivariantly via an equivariant refinement head. Input: a sequence of tokens (starting with a start token) and a set of learnable 3D anchor points (initially empty). Output: a completed molecule and its full 3D structure.

### (B) How it works
```python
# Pseudocode for one generation step
def generate_step(decoder, equivariant_attention, point_updater, tokens_so_far, latent_3d_points):
    # tokens_so_far: list of token embeddings (e.g., atom types)
    # latent_3d_points: set of 3D points (x,y,z) with features (point features and token indices)
    
    # Encode tokens with rotary position embeddings (RoPE) for sequence order
    token_embeds = decoder.token_embedding(tokens_so_far) + rotary_pos_emb
    
    # Equivariant attention: each token attends to all 3D points and vice versa
    # Input: token queries Q_t, point keys K_p, point values V_p; also point queries Q_p, token keys K_t, token values V_t
    # The attention uses relative 3D positions encoded via spherical harmonics (up to L_max=4) for SE(3) invariance
    # Implementation uses e3nn library for spherical harmonics and a 2-layer equivariant message-passing module from e3nn.examples
    attn_tokens = equivariant_attention(Q_t, K_p, V_p)  # token context from points
    attn_points = equivariant_attention(Q_p, K_t, V_t)  # point context from tokens
    
    # Decoder self-attention over token sequence (with causal mask)
    token_hidden = decoder.self_attention(token_embeds + attn_tokens)
    token_hidden = decoder.ffn(token_hidden)
    
    # Predict next token logits (over vocabulary)
    logits = decoder.lm_head(token_hidden[-1])
    next_token = sample(logits, temperature=1.0)
    
    # Update latent 3D points: given the new token, refine point positions and features
    # Use an equivariant message-passing or a small PointNet++ module (with SE(3)-invariant features)
    new_points = point_updater(latent_3d_points, token_hidden, next_token_embed)
    
    return next_token, new_points
```

### (C) Why this design
We chose an autoregressive decoder over a diffusion model because autoregressive generation is faster and directly applicable to lead optimization, accepting the cost that it may struggle with global 3D consistency if the latent representation is not updated properly. We use a latent 3D point cloud rather than a voxel grid or full coordinate graph because points are sparse and equivariant attention can scale to long molecules (≤100 heavy atoms), whereas voxel grids explode in memory. We integrated spherical harmonic encoding for relative positions up to L=4 (instead of a simpler distance-based kernel) because it provides exact SE(3) invariance for the attention weights, enabling the model to treat rotated molecule fragments identically; the cost is higher computational overhead per attention head. A load-bearing assumption is that L=4 provides sufficient angular resolution to distinguish chiral configurations; we verify this by testing attention weight invariance under random rotations (quantified by mean relative error < 1e-5). We update the point cloud dynamically after each token (instead of every N steps) to maintain fine-grained 3D context, but this increases training steps; we mitigate by using a lightweight equivariant module (2-layer message-passing) for the update. Finally, we condition the token decoder on the 3D features via cross-attention to the point cloud (rather than concatenating a global vector) so that the model can spatially localize which part of the molecule to extend, at the cost of increased attention computation.

### (D) Why it measures what we claim
The computational quantities that operationalize our motivation are: (i) the equivariant attention outputs (which measure spatial consistency between tokens and 3D points under SE(3) transformations) and (ii) the point update mechanism (which measures the geometric plausibility of the new token's placement). Equivariant attention measures SE(3) invariance because the spherical harmonic encoding of relative positions ensures that attention weights are unchanged if the entire point cloud is rotated/translated; this assumption fails when the relative positions are very large (beyond the encoding's receptive field), in which case the attention becomes rotationally degenerate. The point update module measures geometric plausibility (X) by predicting atom positions conditioned on known partial geometry (Y). The linking assumption (A) is that the learned bond-length/angle distribution is representative of physical conformers; this fails (F) for novel ring systems, in which case the updated points reflect an interpolation towards the average geometry rather than a physically realistic one. To detect this failure, we compute a per-molecule plausibility score (e.g., inverse of predicted strain energy from a fast force field) and flag molecules where plausibility drops below a threshold (e.g., 0.5 on a normalized scale).

## Contribution

(1) A novel autoregressive molecular generation architecture that integrates a latent SE(3)-equivariant 3D point cloud as a conditioning context, enabling token-by-token generation with explicit 3D awareness. (2) A design principle demonstrating how to incorporate equivariant attention into a causal transformer decoder without breaking autoregressive property, using spherical harmonic relative position encoding. (3) An analysis of the trade-off between 3D detail and sequence length through the latent point cloud update frequency.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | ZINC250k | Standard drug-like molecule benchmark |
| Primary metric | Geometric validity (%) | Tests 3D consistency and physical plausibility |
| Baseline1 | DrugGPT | Token-only autoregressive, no 3D conditioning |
| Baseline2 | EDM | Equivariant diffusion, strong 3D generation |
| Baseline3 | G-SchNet | Autoregressive equivariant, similar paradigm |
| Ablation | SE3MolGen w/o point update | Isolates effect of dynamic point cloud updates |

### Why this setup validates the claim
This combination of dataset, baselines, metric, and ablation forms a falsifiable test of the central claim that SE3MolGen's autoregressive decoding with dynamic 3D point cloud conditioning yields high-quality 3D molecular structures. Using ZINC250k ensures drug-like diversity. Geometric validity directly tests the physical plausibility of generated coordinates, addressing the key claim of improved 3D consistency. DrugGPT (no 3D) tests whether 3D conditioning adds value; EDM (diffusion) tests the speed-accuracy trade-off; G-SchNet (equivariant autoregressive) tests the impact of our specific equivariant attention and dynamic updates. The ablation isolates the point update mechanism. If SE3MolGen outperforms DrugGPT but matches EDM, it validates the efficiency claim; if it beats G-SchNet, it shows our module design is superior. Failure of the ablation to drop performance would falsify the importance of dynamic updates.

### Expected outcome and causal chain

**vs. DrugGPT** — On a case where a molecule requires precise 3D geometry (e.g., a strained ring with specific dihedral angles), DrugGPT produces a sequence of atom types but no spatial coordinates, relying on external tools to guess 3D conformers, often resulting in unrealistic bond angles or steric clashes. Our method instead generates coordinates stepwise via equivariant attention to the point cloud, ensuring that each new atom's placement respects the partial geometry. We expect a noticeable gap on molecules with high conformational complexity (e.g., macrocycles) where DrugGPT's validity drops below 50% while ours remains above 80%, but parity on simple linear fragments.

**vs. EDM** — On a case where fast generation is desired (e.g., generating 10,000 molecules for virtual screening), EDM requires iterative denoising steps (e.g., 1000 steps) leading to high latency, while our autoregressive model generates a molecule in O(n) steps with n~50. However, EDM may produce more globally consistent 3D structures due to its diffusion process. We expect that on metrics of geometric validity, EDM achieves ~95% vs our ~90%, but on generation speed (molecules per second), we are 10-100x faster. The trade-off is acceptable for lead optimization where speed matters.

**vs. G-SchNet** — On a case where a molecule has multiple symmetrically equivalent conformations (e.g., rotation of a methyl group), G-SchNet's fixed torsion angles may force a specific orientation, leading to higher steric energy. Our method uses spherical harmonic encoding up to L=2 for equivariant attention, which treats rotated fragments identically, allowing the model to generate the most stable orientation without bias. We expect that on molecules with flexible rotatable bonds, our method yields 5-10% higher geometric validity because it avoids penalty for symmetric placements.

### What would falsify this idea
If our method's geometric validity is not significantly higher than DrugGPT on molecules requiring 3D precision (e.g., macrocycles), or if the ablation with static points matches the full model, the central claim that dynamic point updates are crucial would be falsified. Specifically, uniform improvement across all subsets rather than concentration on geometrically challenging cases would indicate the model is not leveraging 3D information as intended.

## References

1. DrugGen 2: A disease-aware language model for enhancing drug discovery
2. DrugGen enhances drug discovery with large language models and reinforcement learning
3. Token-Mol 1.0: tokenized drug design with large language models
4. TamGen: drug design with target-aware molecule generation through a chemical language model
5. DrugGPT: A GPT-based Strategy for Designing Potential Ligands Targeting Specific Proteins
6. Generation of 3D molecules in pockets via a language model
