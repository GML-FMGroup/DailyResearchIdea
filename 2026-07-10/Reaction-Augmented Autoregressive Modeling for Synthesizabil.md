# Reaction-Augmented Autoregressive Modeling for Synthesizability-Guaranteed Molecular Generation

## Motivation

Existing autoregressive molecular generators like DrugGen 2 (cited from Idea Cards) produce molecules with high predicted affinity but no guarantee of synthetic accessibility. The root cause is that these models generate token sequences representing atoms/bonds, which do not correspond to synthetic pathways, leading to many generated molecules being impractical to synthesize and limiting real-world utility.

## Key Insight

By generating molecules as a sequence of reaction steps from a differentiable library of reaction templates, we enforce that every generated intermediate and final molecule is synthesizable from commercially available building blocks, enabling end-to-end gradient-based optimization of synthesizability constraints.

## Method

### (A) What it is
We propose **SynthGen**, an autoregressive transformer that generates molecules as a sequence of chemical reaction steps. Input: purchasable building blocks (molecular fragments). Output: a valid synthetic route ending with the target molecule. The model uses a differentiable reaction template library to allow gradient flow through reaction selection. **Load-bearing assumption**: The differentiable softmax over templates provides faithful gradients for learning reaction sequences. We verify this in the experiment (see expected outcome vs. non-differentiable baseline).

### (B) How it works
```
Algorithm: SynthGen Autoregressive Reaction Generation
Input: Building block library B, reaction template library T (differentiable), target property reward R
Output: Reaction sequence (r1, r2, ..., rn) and final molecule M

1. Initialize sequence S = [START] token.
2. For step t = 1 to max_steps (max_steps=20):
   a. Encode current molecular state: from S, decode molecular graph G_t using reaction templates.
   b. Compute context vector: h_t = TransformerDecoder(S, G_t, target_embedding)  (num_layers=6, hidden_dim=512).
   c. Predict next reaction: P(r_t | S, G_t) = softmax(MLP(h_t) · template_embedding_matrix)  (MLP: 2 layers, hidden=256, GeLU).
   d. Sample r_t from P, or use argmax at inference.
   e. Apply reaction: G_{t+1} = apply_template(G_t, r_t)  (differentiable via graph neural network update).
   f. Append r_t to S.
   g. If no more reactions possible or max steps, stop.
3. Compute reward R(final_molecule) for RL fine-tuning (R = docking_score + synthetic_accessibility_score).
4. Train with language modeling loss on S plus policy gradient for R (learning_rate=1e-4).

Verification: We compare the differentiable softmax variant with a discrete policy gradient variant (REINFORCE with self-critical baseline) to assess gradient faithfulness.
```

### (C) Why this design
We chose autoregressive reaction sequences over atom-level generation because they inherently enforce synthetic feasibility by construction, whereas atom-level models require post-hoc filtering. We used a differentiable template library instead of discrete lookup to enable gradient-based optimization of synthesizability; this trades off a small computational overhead for end-to-end training. We conditioned on target embeddings from a pre-trained protein encoder (e.g., from TamGen) to guide generation toward specific targets, accepting that target information may be noisy. We employed RL fine-tuning with a composite reward (validity, docking score, synthetic accessibility) to balance multiple objectives; this introduces training instability but is necessary to optimize properties beyond synthesizability.

### (D) Why it measures what we claim
The differentiable reaction template probability P(r_t) measures the likelihood of a feasible reaction step because it is parameterized by a softmax over templates that represent known chemical reactions verified in the literature; **assumption A**: all realizable reactions belong to library T. **Failure mode F**: novel or condition-dependent reactions are missed, in which case P(r_t) reflects only the closest known reaction, reducing synthesizability guarantee. We test this by evaluating out-of-template reactions via a retrosynthesis predictor (e.g., using an external retrosynthesis model to check if the predicted path can be executed with alternative templates). The autoregressive generation of reaction sequences measures synthetic accessibility because each step consumes a building block and transforms it via a validated template; this assumption fails when building blocks are not purchasable or reactions require non-standard conditions, in which case the sequence still describes a synthetic path but may not be practically executable.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Target-specific binding dataset (e.g., SARS-CoV-2 Mpro) | Tests synthesis-aware generation for real targets |
| Primary metric | Top-100 average docking score (AutoDock Vina) among synthesizable molecules | Combines docking quality and synthesizability |
| Baseline 1 | DrugGPT | Generic SMILES generation without synthesis awareness |
| Baseline 2 | Token-Mol | Token-level generation with RL, no reaction modeling |
| Baseline 3 | Non-differentiable template lookup (argmax) | Isolates benefit of differentiable gradients |
| Ablation-of-ours | SynthGen without RL fine-tuning | Isolates effect of property optimization via RL |
| Real-world validation | Medicinal chemists evaluate synthesizability of top‑10 molecules | Human expert assessment of practicality |

### Why this setup validates the claim

This experimental design directly tests the central claim that autoregressive reaction generation enforces synthetic feasibility while enabling property optimization. By using a target-specific dataset (e.g., SARS-CoV-2 Mpro), we evaluate the model's ability to generate synthesizable molecules with high binding affinity—a practical drug discovery scenario. DrugGPT serves as a baseline that ignores synthesizability, testing whether our method improves the ratio of synthesizable high-docking molecules. Token-Mol uses RL but without reaction templates, testing whether differentiable templates provide an additional advantage. Baseline 3 (non-differentiable template) isolates the effect of gradient-based learning, while the ablation (no RL) isolates the contribution of RL fine-tuning to docking score improvement. The primary metric—average docking score among the top 100 synthesizable molecules—captures both synthesizability and potency, ensuring that gains are not merely from filtering but from generating better synthesizable candidates. Additionally, real-world validation with medicinal chemists provides a human assessment of synthesizability, grounding our automated metric in expert judgment. If our method outperforms baselines on these metrics, it validates the claim that reaction-constrained generation yields more practical drug candidates.

### Expected outcome and causal chain

**vs. DrugGPT** — On a case where a high-docking molecular topology is sterically complex (e.g., a macrocycle), DrugGPT often generates a molecule that requires non-standard reactions for synthesis, leading to a low synthesizability score. The model lacks reaction constraints, so it naively optimizes docking without regard to feasibility. Our method, constrained by the differentiable reaction template library, only generates molecules that are reachable via known reaction steps; it naturally avoids infeasible topologies. Therefore, we expect a substantial gap in the fraction of synthesizable molecules among top-docking outputs (e.g., ~75% for SynthGen vs ~30% for DrugGPT), while docking scores for synthesizable molecules are comparable.

**vs. Token-Mol** — Consider a molecule where the optimal binding pose requires a specific chiral center. Token-Mol, while using RL to optimize docking, may generate molecules with unstable synthetic intermediates because its token-level generation does not enforce stepwise reaction feasibility. Our method's differentiable templates allow the gradient to flow through each reaction step, so the model can learn to avoid synthetic dead-ends while still optimizing docking. We thus expect SynthGen to produce a higher proportion of synthesizable molecules among top-scoring candidates (e.g., 90% vs 60%) and slightly better average docking scores (e.g., -10.2 vs -9.8 kcal/mol) because it can explore a more focused synthesizable space.

**vs. Non-differentiable template lookup** — The non-differentiable baseline uses argmax over template probabilities, breaking gradient flow. Without proper gradients, the model cannot learn which reaction steps lead to high reward. We expect SynthGen with differentiable softmax to achieve higher docking scores (e.g., -10.2 vs -8.0 kcal/mol) and a higher fraction of synthesizable top molecules because the gradient signal guides the model toward feasible, high-affinity routes. This comparison directly tests the load-bearing assumption that differentiable templates provide better optimization.

**vs. Ablation (no RL)** — Without RL fine-tuning, SynthGen generates diverse synthesizable molecules but with docking scores no better than random selection from the template library. With RL, the model explicitly optimizes for docking score through policy gradient, shifting the distribution toward higher binding affinity. We expect a clear improvement in average docking score (e.g., -10.2 vs -8.5 kcal/mol), while maintaining high synthesizability.

### What would falsify this idea

If SynthGen's synthesizable hit rate is not significantly higher than DrugGPT's, or if the docking score improvement over Token-Mol is uniform across all subsets (indicating the advantage is not due to reaction constraints but rather other model differences), then the central claim—that reaction-level generation improves synthesizability with property optimization—would be falsified. Additionally, if the differentiable softmax variant does not outperform the non-differentiable baseline, the load-bearing assumption about gradient faithfulness would be invalidated.

## References

1. DrugGen 2: A disease-aware language model for enhancing drug discovery
2. DrugGen enhances drug discovery with large language models and reinforcement learning
3. Token-Mol 1.0: tokenized drug design with large language models
4. TamGen: drug design with target-aware molecule generation through a chemical language model
5. DrugGPT: A GPT-based Strategy for Designing Potential Ligands Targeting Specific Proteins
6. Generation of 3D molecules in pockets via a language model
