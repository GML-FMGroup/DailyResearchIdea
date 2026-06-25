# CT-WVM: Continuous-Time World Value Models for Delay-Robust Robotic Manipulation

## Motivation

Current world value models (e.g., WVM) assume fixed timesteps and ignore varying inference delays, causing value estimates to become misaligned with actual task progress when delays change. This structural failure arises because discrete-time dynamics cannot represent the elapsed time between state updates, leading to temporal inconsistency. We build on WVM to address this limitation by introducing a continuous-time formulation.

## Key Insight

By parameterizing the world model with a Neural ODE, the latent state evolves continuously, allowing the value function to be integrated over the actual elapsed time, making it invariant to inference delays of arbitrary duration.

## Method

## Method: CT-WVM (with hybrid dynamics)

(A) **What it is**: ...

## Contribution

(1) CT-WVM, a continuous-time world value model that uses Neural ODEs to handle variable inference delays, enabling value estimation under arbitrary delay durations. (2) A demonstration that combining continuous-time dynamics with categorical value distribution training improves temporal consistency and robustness in robotic manipulation value models.

## Experiment

### Evaluation Setup
| Role | Choice | Rationale |
|...|...|...|

### Why this setup validates the claim
...

### Expected outcome and causal chain
**vs. Discrete WVM** ...
**vs. MSE WVM** ...
**vs. RNN WVM** ...
**vs. CT-WVM w/ learnable discount** ...

### What would falsify this idea
...

## References

1. World Value Models for Robotic Manipulation
2. Re-Mix: Optimizing Data Mixtures for Large Scale Imitation Learning
3. V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning
4. DataMIL: Selecting Data for Robot Imitation Learning with Datamodels
5. Stop Regressing: Training Value Functions via Classification for Scalable Deep RL
6. RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control
7. Scaling 4D Representations
8. DINO-WM: World Models on Pre-trained Visual Features enable Zero-shot Planning
