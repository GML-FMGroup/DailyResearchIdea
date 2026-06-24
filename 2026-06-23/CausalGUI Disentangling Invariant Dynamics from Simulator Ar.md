# CausalGUI: Disentangling Invariant Dynamics from Simulator Artifacts for Synthetic GUI Trajectory Generation

## Motivation

Existing methods for training phone-use agents, such as PhoneBuddy, rely on static mock environments that fail to capture dynamic app behaviors like animations and async events, necessitating costly real-device data collection. The root cause is that current approaches treat all aspects of GUI dynamics as monolithic, entangling invariant transition mechanisms (e.g., logical state changes triggered by actions) with simulator-specific artifacts (e.g., timing, visual effects). This prevents the synthesis of realistic dynamic scenarios that challenge the agent, leaving a fundamental gap in sim-to-real transfer.

## Key Insight

GUI dynamics can be modeled as a causal generative process where the invariant transition mechanism is conditionally independent of simulator artifacts given the latent state, enabling controlled synthesis of diverse dynamic behaviors by intervening on artifacts alone.

## Method

(A) **What it is**: CausalGUI is a conditional variational autoencoder that learns a disentangled representation of GUI transitions from mixed real and mock trajectories. Input: a sequence of (state, action, next_state) triples. Output: synthetic trajectories with varied dynamic behaviors (e.g., different animation speeds, random async event timings).

(B) **How it works**:
```python
def causal_de_train(trajectories):
    for (s_t, a_t, s_{t+1}) in mixed_batch:
        # Encode transition into two latent variables
        mu_inv, logvar_inv = encoder_inv(s_t, a_t, s_{t+1})
        mu_art, logvar_art = encoder_art(s_t, a_t, s_{t+1})
        z_inv = reparameterize(mu_inv, logvar_inv)
        z_art = reparameterize(mu_art, logvar_art)
        # Decode next state from action and latents
        s_pred = decoder(z_inv, z_art, a_t)
        # Reconstruction loss
        recon_loss = MSE(s_pred, s_{t+1})
        # KL divergence for VAE
        kl_loss = KL(q(z_inv)||p(z_inv)) + KL(q(z_art)||p(z_art))
        # Disentanglement: minimize mutual information between z_inv and z_art
        mi_loss = mutual_information_estimator(z_inv, z_art)
        # Adversarial invariance: predict z_inv from (s_t, a_t, s_{t+1}) across environments
        adv_loss = adversarial_loss(env_predictor(s_t, a_t, s_{t+1}), z_inv)
        loss = recon_loss + beta_vae * kl_loss + lambda_mi * mi_loss + lambda_adv * adv_loss
        optimize(loss)
```
Hyperparameters: beta_vae=1.0 (standard), lambda_mi=0.1, lambda_adv=0.5.

(C) **Why this design**: We chose a VAE over a GAN because the likelihood-based objective naturally handles stochastic transitions and provides a principled latent space for intervention, at the cost of potentially blurry generated states. We split the latent space into invariant and artifact-specific components rather than using a single latent, because we need to manipulate artifacts independently; this enables controlled synthesis but requires regularization to prevent information leakage between latents. We used adversarial training to enforce that z_inv is not predictive of the environment (real vs. mock), which is a stronger condition than standard disentanglement; this introduces training instability but is necessary to ensure that z_inv truly captures invariant mechanisms. The mutual information penalty further reduces correlation between latents. Compared to prior work like PhoneBuddy, which simply mixes data, CausalGUI explicitly models the generative process, allowing synthesis of unseen dynamics.

(D) **Why it measures what we claim**: The reconstruction loss ensures that z_inv and z_art together capture all information needed to predict the next state, but the disentanglement losses isolate the invariant mechanism. Specifically, the adversarial loss (`adv_loss`) measures the degree to which z_inv is independent of environment type because a domain expert would expect that invariant transitions are environment-agnostic; this assumption fails when real and mock environments have fundamentally different transition probabilities (e.g., a real app may have a timeout that does not exist in mock), in which case z_inv would be forced to encode environment-specific information, leading to poor synthesis. The mutual information penalty (`mi_loss`) measures the statistical dependence between z_inv and z_art; we assume that true invariant mechanisms are orthogonal to artifact details, but this assumption fails when artifacts correlate with transitions (e.g., animation duration affects when a next state is detected), in which case the model may under-represent the invariant component. These regularizers together operationalize the causal separation required for controllable generation.

## Contribution

(1) A causal generative model for GUI dynamics that explicitly disentangles invariant transition mechanisms from simulator-specific artifacts via a structured VAE with adversarial and mutual information regularization. (2) A method to synthesize diverse dynamic app behaviors without additional real-world data by intervening on the artifact latent variable. (3) Empirical demonstration that policies trained on synthetic data from CausalGUI achieve higher task success on dynamic real-device tasks compared to policies trained on static mock data or mixed data alone.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|-----------------------|
| Dataset | Mixed real+mock phone trajectories | Captures both environments |
| Primary metric | RL task success on real phone | Tests invariant generalization |
| Baseline 1 | Mix-only training | Lacks latent separation |
| Baseline 2 | Single-latent VAE | No artifact-invariant split |
| Baseline 3 | CausalGUI w/o adv loss | Adversarial invariance missing |
| Ablation | CausalGUI w/o MI penalty | Mutual information not penalized |

### Why this setup validates the claim

The combination of a mixed real-and-mock dataset, downstream RL success metric, and three targeted baselines (mixing only, single-latent VAE, and no adversarial loss) along with an MI-ablation forms a falsifiable test of the central claim: CausalGUI learns a disentangled representation where invariant mechanisms can be controlled independently of artifacts. If the claim holds, CausalGUI should significantly outperform all baselines on tasks where artifact variation (e.g., animation speed, async events) distracts learning, while matching baselines on artifact-free tasks. The mix-only baseline tests whether naïve data augmentation suffices; the single-latent VAE tests whether a non-disentangled latent space can still generalize; the no-adversarial baseline tests the specific role of adversarial invariance. The MI-ablation isolates the contribution of mutual information regularization. The chosen metric directly measures the downstream utility of synthesized trajectories, which is the ultimate goal of the method.

### Expected outcome and causal chain

**vs. Mix-only** — On a task where real and mock differ in animation speed (e.g., a button that triggers a slow animation in real but instant in mock), mix-only synthesizes trajectories with incorrect timing, causing the RL agent to mispredict when the next state appears. Our method's invariant latent captures the deterministic button-triggered transition independent of delay; the artifact latent models delay variation. Expected: a noticeable gap (e.g., 20% success on variable-timing tasks vs. 50% for ours) but parity on static-timing tasks.

**vs. Single-latent VAE** — On a task with frequent async ads (artifact), a single latent must compress both the invariant ad-dismissal logic and the stochastic ad occurrence, leading to blurry reconstructions and poor RL policy. Our split latent isolates the invariant action-effect from the artifact’s randomness. Expected: our method achieves >60% success on high-artifact tasks while single-latent VAE stays below 30%.

**vs. CausalGUI w/o adv loss** — On a task where real environment has a timeout not present in mock (e.g., real app logs out after 10s idle), the model lacking adversarial loss may encode timeout info into z_inv, mimicking mock behavior and failing in real. Our adversarial loss forces z_inv to be environment-agnostic. Expected: full model succeeds (>70%) while the ablation fails (<20%) on timeout-sensitive tasks.

### What would falsify this idea

If the performance gain of full CausalGUI over the ablation without adversarial loss is uniform across all task subsets rather than concentrated where real/mock transition probabilities differ, then the adversarial loss is not enforcing true invariance. Similarly, if single-latent VAE matches CausalGUI overall, the latent split is unnecessary.

## References

1. Training Open Models for Agentic Phone Use
2. Mobile-Bench: An Evaluation Benchmark for LLM-based Mobile Agents
3. UI-Venus Technical Report: Building High-performance UI Agents with RFT
4. Expanding Performance Boundaries of Open-Source Multimodal Models with Model, Data, and Test-Time Scaling
5. Aguvis: Unified Pure Vision Agents for Autonomous GUI Interaction
6. PPTC Benchmark: Evaluating Large Language Models for PowerPoint Task Completion
7. DroidBot-GPT: GPT-powered UI Automation for Android
8. Android in the Wild: A Large-Scale Dataset for Android Device Control
