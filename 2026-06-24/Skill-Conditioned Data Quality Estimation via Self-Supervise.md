# Skill-Conditioned Data Quality Estimation via Self-Supervised Compositional World Models

## Motivation

World value models (WVM) require task-specific trajectory data to learn value functions, limiting generalization to unseen tasks. This structural dependence stems from the assumption that data quality is inherently task-defined, yet robotic manipulation trajectories share reusable skill primitives (e.g., grasping, pushing) that can be assessed independently of task goals. Prior self-supervised methods like V-JEPA learn rich video representations but do not exploit compositionality for data filtering, while Re-Mix optimizes domain weights but requires predefined task domains. We address this by factorizing trajectories into discrete skill components via a self-supervised world model, enabling data quality estimation based on skill transition coherence without task labels.

## Key Insight

By enforcing a discrete latent bottleneck that captures temporally extended skill primitives, the model's prediction error on skill-conditioned future frames directly measures data quality as the coherence of skill transitions, independent of task labels.

## Method

### (A) What it is
**Self-Supervised Skill-Factorized World Model (SF-WM)** learns a compositional generative model of unlabeled robot trajectories by factorizing them into discrete latent skill codes and a dynamics predictor. Input: video-action trajectories; output: a per-trajectory quality score (inverse reconstruction error).

### (B) How it works
```python
# Training SF-WM
Input: Unlabeled trajectory dataset D (each τ: (o_1,a_1,...,o_T,a_T))
Hyperparameters: K=16 skill codes, β=0.1 commitment, λ_rec=1.0
Initialize: Encoder E (CNN+Transformer) → skill logits; Predictor P (MLP); Decoder D (CNN decoder)

for each batch of trajectories:
    # (1) Encode each observation into a discrete skill code via Gumbel-Softmax
    for t in 1..T:
        logits = E(o_t)
        z_t = gumbel_softmax(logits, tau=0.5)  # one-hot vector
    
    # (2) Predict next observation from skill and action
    for t in 1..T-1:
        pred_latent = P(z_t, a_t)
        o_hat_{t+1} = D(pred_latent)
    
    # (3) Losses
    L_rec = MSE([o_hat_2..o_hat_T], [o_2..o_T])  # reconstruction
    L_commit = ||z_t - sg(E(o_t))||^2  # stop-gradient on encoder
    L_total = L_rec + β * L_commit
    update E, P, D via Adam

# Data quality estimation (post-training)
def quality_score(τ):
    # Encode trajectory into skill sequence
    z_seq = [argmax(E(o_t)) for t in 1..T]
    # Compute per-step reconstruction error
    errors = [MSE(o_{t+1}, D(P(z_t, a_t))) for t in 1..T-1]
    return -mean(errors)  # negative error → higher is better
```

### (C) Why this design
We chose three key design decisions: (1) **Discrete skill codes** (Gumbel-Softmax) over continuous latent variables because discrete codes naturally enforce a composition of reusable primitives, enabling interpretable skill transitions; the trade-off is that discretization may lose fine-grained motion details, which we accept because quality estimation benefits from categorical structure. (2) **Reconstruction loss** (MSE) rather than contrastive or predictive loss in latent space; reconstruction forces the model to preserve visual details essential for detecting novel skill combinations, but it increases computational cost—we accept this as the payoff for robustness against spurious latent correlations. (3) **Commitment loss** to prevent skill assignment from collapsing; without it, the encoder would ignore the codebook and produce uniform representations, but adding it introduces a hyperparameter β that must be tuned—we set β=0.1 via grid search on a held-out set. These choices are distinct from prior work: V-JEPA uses joint-embedding prediction without discretization, and DINO-WM uses pre-trained features without a compositional bottleneck. Our design explicitly forces a structured decomposition that prior methods lack.

### (D) Why it measures what we claim
The computational quantity **reconstruction error** measures **data quality** (coherent skill transitions) because the model's prediction relies entirely on the learned skill dynamics; low error implies that the observed trajectory can be explained by the known skill set and their transition patterns, i.e., it is 'well-composed'. This assumption fails when a trajectory contains static frames or repetitive motions that are trivially predictable independent of skill structure—in that case, low reconstruction error reflects simplicity rather than skill coherence. However, such trajectories are generally low-quality for learning diverse skills, so the metric still aligns with practical utility. The **discrete skill codes** measure **skill purity** via the commitment loss value: high commitment to a single skill over short windows indicates consistent execution; this assumption fails when the encoder assigns codes arbitrarily due to distribution shift, but the stop-gradient design prevents drift. Taken together, the model's components operationalize the concept of compositional data quality in a task-agnostic manner.

## Contribution

(1) SF-WM, a self-supervised method that factorizes robotic manipulation trajectories into discrete skill codes and a dynamics predictor without task labels, enabling compositional trajectory understanding. (2) An empirical finding that reconstruction error under the skill-factorized world model serves as an effective task-agnostic data quality metric for data filtering in imitation learning, validated across multiple skill combinations. (3) A new behavioral benchmark for evaluating compositional skill discovery in manipulation, using synthetic mixtures of primitive skills to measure factorization accuracy.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
|------|--------|-----------|
| Dataset | Diverse unlabeled robot trajectories | Contains varied quality |
| Primary metric | Spearman correlation with human quality | Measures ranking accuracy |
| Baseline | World Value Model (WVM) | Learns value from trajectories |
| Baseline | V-JEPA 2 | Self-supervised video model |
| Baseline | DataMIL | Data selection via influence |
| Ablation | SF-WM with continuous latents | Tests discretization benefit |

### Why this setup validates the claim

The dataset includes both coherent and incoherent trajectories, enabling assessment of quality estimation against human judgment. WVM requires reward labels and may fail in unlabeled settings; our method is fully self-supervised. V-JEPA 2 predicts latent features without compositional bottleneck; our method forces skill decomposition via discrete codes. DataMIL relies on training dynamics of a downstream policy, which is task-specific; our method is task-agnostic. The ablation (continuous latents) isolates the effect of discretization. The metric (correlation with human quality) directly measures alignment with the intended notion of data quality, providing a falsifiable test of whether skill-factorization improves quality estimation.

### Expected outcome and causal chain

**vs. World Value Model (WVM)** — On a trajectory with repetitive but irrelevant motions, WVM assigns high value because such motions are common in training data, but our method detects high reconstruction error due to lack of skill structure (repeated motions are trivially predictable but not composed of diverse skills). WVM's value function is trained on expert data and cannot distinguish repetitive from purposeful behavior. Our method's reconstruction error penalizes transitions that are not well-explained by the skill codebook, so we expect a larger gap on repetitive/static trajectories (e.g., 0.2 vs. 0.6 correlation) but parity on clean expert data (≈0.8).

**vs. V-JEPA 2** — On a trajectory with a novel skill combination (e.g., push then grasp), V-JEPA 2's latent prediction may still be accurate because it captures motion features without explicit skill separation. However, it cannot identify whether the trajectory is composed of known primitives; our method's reconstruction error will be high if the skill sequence is rare or unseen. Thus, we expect our method to better identify unusual combinations, showing a 0.3 higher correlation on the subset of trajectories with novel skill transitions.

**vs. DataMIL** — DataMIL selects data based on training loss of a downstream policy; it is task-specific. On a trajectory that is high-quality for one task but low for another, DataMIL may misjudge. Our method's reconstruction error is task-agnostic and reflects intrinsic compositionality. Therefore, we expect our method to generalize across tasks (e.g., consistent 0.75 correlation), while DataMIL's correlation varies by task (between 0.4 and 0.9).

### What would falsify this idea

If our method shows high correlation with human quality only on trajectories with obvious motion errors but fails on subtle skill mismatches (e.g., same motions but wrong order), then the discretization is not capturing meaningful composition and the central claim is false.

## References

1. World Value Models for Robotic Manipulation
2. Re-Mix: Optimizing Data Mixtures for Large Scale Imitation Learning
3. V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning
4. DataMIL: Selecting Data for Robot Imitation Learning with Datamodels
5. Stop Regressing: Training Value Functions via Classification for Scalable Deep RL
6. RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control
7. Scaling 4D Representations
8. DINO-WM: World Models on Pre-trained Visual Features enable Zero-shot Planning
