# CycleState: Self-Supervised Latent State Inference from Fragmented Embodied Interactions via Cycle-Consistency

## Motivation

Existing memory benchmarks such as WorldLines rely on structured trace generation that assumes clean, contiguous input sequences. However, real-world embodied agents encounter noisy, fragmented interactions where observations are incomplete and state transitions are overwritten. This structural mismatch prevents existing methods from extracting coherent state representations from naturalistic data because they depend on well-defined boundaries and complete state information, which are unavailable in practice.

## Key Insight

Cycle-consistency across overlapping temporal fragments provides a self-supervised training signal that forces latent state representations to be invariant to fragment boundaries and observation noise, because the composition of encoding and decoding across fragments must recover the same latent state.

## Method

**CycleState: Self-Supervised Latent State Inference from Fragmented Interactions**

(A) **What it is:** CycleState is a self-supervised framework that takes as input a set of observation-action fragments (each fragment is a short sequence of observations and actions with potentially incomplete state information) and learns an encoder that maps each fragment to a sequence of latent states, and a decoder that reconstructs future observations/actions from latent states. The key training signal is cycle-consistency across overlapping fragments.

(B) **How it works (pseudocode):**
```pseudocode
Input: Set of fragments F = {f_i}, each fragment f_i = [(o_{i,1}, a_{i,1}), ..., (o_{i,T_i}, a_{i,T_i})]
Hyperparameters: latent_dim=256, cycle_consistency_weight lambda_c=1.0, alignment_window W=10, soft-DTW temperature gamma=0.1
For each training batch:
  Sample two fragments f_a, f_b that share temporal overlap (detected via timestamps or learned alignment)
  # Step 1: Encode each fragment to latent state sequences
  z_a = Encoder(f_a)  # (T_a, latent_dim)
  z_b = Encoder(f_b)  # (T_b, latent_dim)
  # Step 2: Align latent sequences using soft-DTW (differentiable) with window W
  # Concrete soft-DTW implementation (PyTorch-like):
  # Compute pairwise distance matrix
  dist_mat = torch.cdist(z_a, z_b, p=2)  # (T_a, T_b)
  # Initialize cumulative cost matrix R (size T_a+1 x T_b+1) with large values, R[0,0]=0
  R = torch.full((T_a+1, T_b+1), 1e10)
  R[0,0] = 0
  for i in range(1, T_a+1):
      for j in range(1, T_b+1):
          if abs(i-j) <= W:  # window constraint
              # soft minimum with gamma
              soft_min = -gamma * torch.logsumexp(torch.tensor([-R[i-1,j]/gamma, -R[i-1,j-1]/gamma, -R[i,j-1]/gamma]), dim=0)
              R[i,j] = dist_mat[i-1,j-1] + soft_min
  dist = R[-1,-1]  # soft-DTW distance (not used in loss directly)
  # Alignment matrix: approximate via difference of cumulative costs (simplified here)
  alignment = torch.sigmoid((R[1:,1:] - R[:-1,:-1]) / gamma)  # (T_a, T_b)
  # Step 3: Cycle consistency: reconstruct fragment a using latents from fragment b
  # For each time t in z_a, reconstruct using weighted average of decoder outputs from aligned z_b
  # Weights: w_{i,j} = alignment[i,j] / sum_j alignment[i,j] (normalized)
  w = alignment / alignment.sum(dim=1, keepdim=True)  # (T_a, T_b)
  hat_o_a = sum_j w[i,j] * Decoder(z_b[j])  # reconstructed observations for a
  hat_a_a = sum_j w[i,j] * Decoder_action(z_b[j])  # reconstructed actions
  loss_cycle_a = MSE(hat_o_a, o_a) + MSE(hat_a_a, a_a)
  # Symmetric: reconstruct b from a
  w_sym = alignment.T / alignment.T.sum(dim=1, keepdim=True)
  hat_o_b = sum_i w_sym[j,i] * Decoder(z_a[i])
  hat_a_b = sum_i w_sym[j,i] * Decoder_action(z_a[i])
  loss_cycle_b = MSE(hat_o_b, o_b) + MSE(hat_a_b, a_b)
  loss_cycle = loss_cycle_a + loss_cycle_b
  # Step 4: Reconstruction loss within fragment (one-step prediction)
  loss_recon_a = MSE( Decoder(z_a[:-1,:]), (o_a[1:,:], a_a[1:,:]) )
  loss_recon_b = MSE( Decoder(z_b[:-1,:]), (o_b[1:,:], a_b[1:,:]) )
  loss_recon = loss_recon_a + loss_recon_b
  # Step 5: Total loss
  loss = loss_recon + lambda_c * loss_cycle
  Update Encoder and Decoder
```

(C) **Why this design:** We chose cycle-consistency over direct alignment (e.g., cross-entropy on aligned pairs) because cycle-consistency forces latent states to be coherent under composition, preventing the encoder from exploiting fragment-specific cues. We used soft-DTW alignment instead of hard DTW to maintain differentiability and allow gradient flow, accepting the cost of approximate alignment that may blur sharp transitions. We incorporated reconstruction loss within fragments to anchor the latent space to actual observations, preventing degenerate solutions where cycle-consistency alone could be satisfied with trivial constant latents. The alignment window W limits the temporal spread to avoid matching temporally distant states, assuming that overlap is local; this trades off robustness to long-range dependencies for computational efficiency. The cycle-consistency weight lambda_c balances the two objectives; we set lambda_c=1.0 based on a small grid search on a validation set. Compared to prior gradient-based alignment methods (e.g., soft-DTW for time series), our novelty lies in using it across heterogeneous fragments with a reconstruction objective, not just alignment distance.

(D) **Why it measures what we claim:** The cycle-consistency loss measures temporal alignment consistency (X) as a proxy for state trajectory coherence (Y). The linking assumption (A) is that overlapping fragments share the same underlying state sequence. This fails (F) when fragments are non-overlapping or observe different state branches, leading to forced false alignments. In that case, the loss reflects reconstruction fidelity rather than state coherence. The reconstruction loss within fragments measures the encoder's ability to compress observations into predictive latent states, assuming that the Markov property holds for the state dynamics; if the dynamics are non-Markovian, the reconstruction loss becomes an overestimate of state sufficiency. Together, the two losses operationalize the concept of coherent latent state inference: cycle-consistency enforces cross-fragment consistency, and reconstruction ensures the latent states are predictive of future observations.

**Load-bearing assumption verification:** The method assumes that soft-DTW alignment on latent representations of fragments whose temporal overlap is detected via timestamps (available in WorldLines) produces accurate temporal correspondences. To verify this, we hold out a calibration set of 512 fragments with ground-truth timestamps and compute the alignment F1 score (comparing aligned indices) after each epoch. If F1 drops below 0.85, we decrease the soft-DTW temperature gamma by 0.02 (minimum 0.05) and increase window W by 2 (maximum 15), ensuring that alignment remains reliable throughout training.

## Contribution

(1) CycleState, a self-supervised framework for learning latent state representations from fragmented embodied interactions without requiring ground-truth state trajectories or clean contiguous sequences. (2) A novel application of cycle-consistency loss across temporal fragments for embodied agent memory, demonstrating that overlapping observation sequences can serve as a self-supervised signal for state inference. (3) Empirical validation on a modified version of the WorldLines benchmark with artificially fragmented traces, showing improved performance in downstream memory QA and planning tasks compared to baselines that ignore fragmentation.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | WorldLines Memory QA | Standard benchmark for stateful agents |
| Primary metric | Memory QA accuracy | Measures state inference correctness |
| Baseline 1 | Episodic Memory (MemNN) | Strong memory-based baseline |
| Baseline 2 | VAE without cycle consistency | Tests generative latent dynamics |
| Ablation-of-ours | Ours w/o cycle-consistency | Isolates cycle-consistency effect |

### Why this setup validates the claim

WorldLines Memory QA provides overlapping observation-action fragments from embodied agent trajectories, requiring accurate state inference from incomplete views. By comparing against MemNN (explicit memory retrieval) and VAE (latent dynamics without cross-fragment alignment), we isolate the benefit of cycle-consistency. The metric directly measures state correctness, which is the core claim. Our ablation (ours without cycle-consistency) tests whether the cycle loss drives performance, not just the encoder-decoder architecture. If cycle-consistency is essential, our full method should outperform the ablation and baselines specifically on queries requiring integration across overlapping fragments.

### Expected outcome and causal chain

**vs. Episodic Memory (MemNN)** — On a case where two fragments overlap but contain different observation perspectives (e.g., one from camera A, another from camera B), MemNN stores them as separate episodes without alignment; when queried about the agent's location at an overlapping time, it retrieves conflicting information or fails to retrieve, producing incorrect state. Our method aligns latents via soft-DTW and cycle-consistency, merging the perspectives into a single coherent state sequence, thus correctly inferring location. We expect a >15% accuracy gap on multi-fragment queries, but parity on single-fragment queries.

**vs. VAE (without cycle consistency)** — On a case where two fragments temporally overlap but have noisy observations (e.g., occlusion), VAE learns independent latent states per fragment; without cross-fragment alignment, its latents for the same underlying time diverge, causing inconsistent state inference. Our cycle-consistency loss forces latents to coincide, improving state coherence. We expect VAE to achieve <50% accuracy on overlapping queries, while ours maintains >80%.

### What would falsify this idea

If our full method does not outperform the ablation on overlapping queries, or if the gain is uniform across all query types rather than concentrated on multi-fragment ones, then cycle-consistency is not the driving factor and the central claim is invalid.

## References

1. WorldLines: Benchmarking and Modeling Long-Horizon Stateful Embodied Agents
2. Memory OS of AI Agent
3. Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory
4. Embodied AI Agents: Modeling the World
5. Generative Dense Retrieval: Memory Can Be a Burden
6. LLM-based Medical Assistant Personalization with Short- and Long-Term Memory Coordination
7. Explore Theory of Mind: Program-guided adversarial data generation for theory of mind reasoning
8. ELLMA-T: an Embodied LLM-agent for Supporting English Language Learning in Social VR
