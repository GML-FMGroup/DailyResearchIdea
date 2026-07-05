# CompoDiff: Compositional Skill Discovery via Additive Discrete Interface Vectors for Continuous Trajectory Generation

## Motivation

Code-as-policy methods (e.g., ASPIRE, Code as Policies) represent robot skills as executable code, restricting expressiveness to discrete symbolic actions and making it difficult to compose continuous motions. Even when continuous skills are learned, they lack a principled composition mechanism, forcing reliance on hand-crafted sequencing or further training. We need a representation that preserves continuous expressiveness while enabling compositional generalization via a simple algebraic operation.

## Key Insight

The additive composition of discrete interface vectors, learned via vector quantization, induces a linear structure in skill embedding space that mirrors the compositional structure of tasks, enabling zero-shot composition without retraining.

## Method

### (A) What it is
CompoDiff is a skill discovery framework that learns a continuous latent diffusion model for trajectory generation conditioned on composable discrete interface vectors. Input: unlabeled demonstration trajectories of various skills. Output: a codebook of K=64 discrete skill primitives with associated embedding vectors e_i in R^d (d=128) and a conditional diffusion policy that generates trajectories for both primitive and composite skills by adding interface vectors (e_i + e_j). The method assumes that additive composition of interface vectors directly corresponds to the trajectory of the composite skill.

### (B) How it works (pseudocode)
```python
# Hyperparameters
K, d = 64, 128  # codebook size and dimension
T = 1000  # diffusion steps
beta_schedule = linear(0.0001, 0.02)
batch_size = 64
lr = 1e-4
lambda_comp = 1.0  # composition loss weight
tau_gumbel = 0.5   # Gumbel-Softmax temperature

# Architecture
# VAE encoder: 3-layer MLP (input trajectory dim -> 256 -> 128 -> 64 (z))
# VAE decoder: symmetric 3-layer MLP
# Diffusion denoiser: 1D UNet with 4 down/up blocks, channels [64,128,256,512]
#               conditioned on c = [z, v] via cross-attention

# Initialize
codebook = [e_i for i in range(K)]  # each e_i in R^d
encoder = VAE_encoder()
decoder = VAE_decoder()
denoiser = DiffusionUNet()

# Training
for each trajectory tau in dataset:
    z = encoder(tau)  # latent z in R^64
    # Discrete interface: quantize to nearest codebook entry
    logits = -||encoder(tau) - e_i||^2 / tau_gumbel
    indices = Gumbel_Softmax(logits, tau=tau_gumbel, hard=True)  # straight-through
    v = sum(indices_i * e_i)  # weighted sum with straight-through gradient
    c = concat(z, v)  # condition for diffusion
    # Diffusion loss
    t ~ Uniform(1,T), epsilon ~ N(0,1)
    tau_t = sqrt(alpha_bar_t) * tau + sqrt(1-alpha_bar_t) * epsilon
    L_diff = MSE(epsilon, denoiser(tau_t, t, c))
    # VAE reconstruction loss
    L_recon = MSE(tau, decoder(z))
    # Composition loss: sample two skills i,j and composite trajectory tau_ij
    # Synthetic composite: concatenate trajectory of skill i then skill j with smooth transition (10-step linear interpolation)
    v_comp = e_i + e_j
    z_comp = encoder(tau_ij)
    c_comp = concat(z_comp, v_comp)
    L_comp = MSE(epsilon, denoiser(tau_ij_t, t, c_comp))
    # Total loss
    total_loss = L_diff + L_recon + lambda_comp * L_comp
    optimize(total_loss)

# Inference for composite skill (i, j):
v = e_i + e_j
sample z ~ N(0,I)
reverse diffusion conditioned on [z, v] to generate tau
```

### (C) Why this design (≥80 words)
We chose a discrete representation (codebook) over a continuous embedding because discrete codes naturally enforce a finite set of composable primitives, avoiding overlapping or redundant skill representations. The additive composition of interface vectors is simpler than learned composition operators (e.g., MLP over concatenated codes) because it preserves linear structure, enabling generalization to unseen combinations. We combine a VAE for continuous latent variation with diffusion for trajectory generation because VAEs capture low-dimensional skill structure while diffusion models generate high-fidelity trajectories. The straight-through estimator for codebook selection maintains differentiability, accepting the cost of biased gradients. The composition loss directly supervises composite trajectories, which is crucial for learning additive semantics, but requires composite demonstration data (which we generate synthetically by concatenating primitive trajectories with a smooth transition of 10 interpolation steps).

### (D) Why it measures what we claim (≥60 words)
The composition loss L_comp operationalizes the concept of 'compositionality' by enforcing that the additive combination of interface vectors (e_i + e_j) approximates the latent and trajectory of the composite skill. This relies on the assumption that the composite trajectory tau_ij can be represented as a simple additive function in the embedding space; this assumption fails when skills interact in non-linear ways (e.g., mutual interference), in which case L_comp reflects additive error rather than true compositionality. The discrete interface vector v measures 'skill identity' via nearest-neighbor assignment, assuming that trajectories belonging to the same skill are closer in latent space; if skills have high intra-class variation, v may capture irrelevant aspects. The continuous latent z measures 'intra-skill variation' via VAE reconstruction, assuming that sufficient information is retained; if reconstruction is imperfect, z omits details needed for composition. The additive composition of interface vectors (e_i + e_j) measures compositional generalization under the explicit assumption that the embedding space is linear w.r.t. skill combinations. Failure mode: when skills interfere non-additively (e.g., one skill changes the state for the next), L_comp measures additive error rather than true compositionality.

## Contribution

(1) A novel skill representation that pairs a continuous latent trajectory generator (VAE+diffusion) with a discrete interface vector learned via vector quantization, where composition is defined as vector addition. (2) A composition loss that enforces added interface vectors to yield correct composite trajectories, enabling zero-shot skill composition. (3) Empirical demonstration that CompoDiff discovers composable continuous skills from separate demonstrations without manual annotation, outperforming code-as-policy baselines in compositional generalization.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|-------|--------|----------------------|
| Dataset | LIBERO-Spatial | Contains composite tasks with known primitives |
| Primary metric | Composition success rate | Measures ability to combine skills correctly |
| Baseline 1 | DDO (unsupervised skill discovery) | Discovers skills without composition mechanism |
| Baseline 2 | HIL (hierarchical imitation learning) | Learns policies for each primitive, not composition |
| Baseline 3 | CompoDiff-MLP (continuous composition operator) | 2-layer MLP (hidden=256, ReLU) over concatenated e_i,e_j, trained same way |
| Ablation | CompoDiff w/o composition loss | Tests necessity of additive interface supervision |

### Why this setup validates the claim
This setup tests the central claim that a discrete, additively composable codebook enables generalization to unseen composite skills. LIBERO-Spatial provides a controlled environment where primitive skills (e.g., pick, place, push) and their sequential compositions are known. DDO lacks any composition mechanism, so it serves as a baseline for the necessity of explicit compositional structure; HIL treats each composite as a separate primitive, failing to recombine skills. CompoDiff-MLP uses a learned composition operator to quantify the benefit of additive structure. The ablation without composition loss isolates the contribution of additive supervision. The primary metric, composition success rate, directly captures the ability to generate correct trajectories for novel composites, which is exactly what the method claims to enable. By comparing performance on seen vs. unseen composites, we can falsify the claim if gains are not concentrated on unseen composites.

### Expected outcome and causal chain
**vs. DDO** — On a case where the robot must perform a novel composite task (e.g., pick cup then push to the left), DDO's unsupervised skills are not tied to semantic actions; it would likely generate a trajectory that randomly mixes primitive behaviors, failing because it lacks a way to specify the intended combination. Our method instead uses additive interface vectors (e_i + e_j) to condition the diffusion process, generating a coherent trajectory that sequentially executes the correct primitives. Thus, we expect a noticeable gap (e.g., >30% higher success rate) on composite tasks, with parity on single-skill tasks.

**vs. HIL** — On a case where the composite task reuses known primitives in an unseen order (e.g., push then pick instead of pick then push), HIL fails because it was trained to treat each specific sequence as a distinct skill, and cannot reorder them. Our method's additive codebook allows any combination, so it can generate the reversed order correctly. We expect HIL to achieve high success only on composites seen during training, while ours generalizes to unseen orders, leading to a significant difference (e.g., >40% gap on unseen sequences).

**vs. CompoDiff-MLP** — On a case where the composite involves simple sequential execution (e.g., pick then place), both methods succeed. On composites with strong temporal dependencies or interference (e.g., simultaneous push and pick), the additive composition may fail while MLP can learn non-linear interactions. We expect CompoDiff-MLP to show higher success on non-additive composites (e.g., 15-20% higher) but lower on novel sequences due to overfitting, while our method generalizes better to unseen orderings.

**vs. Ablation (w/o composition loss)** — On a case where the composition requires precise coordination between skills (e.g., pick then place with a specific offset), the ablation lacks supervision for the additive interface; its codebook vectors may not combine linearly, causing failures in generating consistent composites. Our full method enforces additive semantics via L_comp, so we expect the ablation to underperform on composite tasks, especially those requiring accurate skill boundaries (e.g., 10-20% lower success).

### What would falsify this idea
If the composition success rate of our method is not significantly higher than both baselines on unseen composite tasks, or if the ablation without composition loss matches our method's performance, then the additive composition mechanism is not critical and the central claim is invalidated. Additionally, if CompoDiff-MLP outperforms our method on a majority of composites, it would indicate that additive assumptions are too restrictive.

## References

1. ASPIRE: Agentic /Skills Discovery for Robotics
2. EvoEngineer: Mastering Automated CUDA Kernel Code Evolution with Large Language Models
3. Eurekaverse: Environment Curriculum Generation via Large Language Models
4. Eureka: Human-Level Reward Design via Coding Large Language Models
5. RoboGen: Towards Unleashing Infinite Data for Automated Robot Learning via Generative Simulation
6. DeXtreme: Transfer of Agile In-hand Manipulation from Simulation to Reality
7. Code as Policies: Language Model Programs for Embodied Control
8. Evolution through Large Models
