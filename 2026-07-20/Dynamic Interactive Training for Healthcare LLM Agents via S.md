# Dynamic Interactive Training for Healthcare LLM Agents via Simulated Patient Encounters

## Motivation

Existing training methods for healthcare LLMs, such as Cura 1T, rely on static benchmark trajectories that fail to capture the interactive dynamics of real patient-physician interactions. This structural limitation arises because static evaluation assumes a fixed set of user intents and responses, whereas real clinical workflows involve dynamic information gathering and adaptive decision-making. MedAgentBench provides a realistic EHR environment for evaluation, but its use as a training environment has not been explored, leaving the gap in learning interactive coordination.

## Key Insight

Training in a simulated interactive environment with an LLM-based patient model exposes the agent to a continuous distribution of user behaviors, enabling it to learn a policy that is robust to unseen patient responses, a property static trajectories cannot provide.

## Method

We propose **Interactive Agentic Training (IAT)**, a reinforcement learning framework that trains a healthcare LLM agent through repeated interactions with a simulated patient environment (SIMPAT). The input is a pre-trained LLM (e.g., Cura 1T) and the output is a policy fine-tuned on interactive tasks.

**How it works:**
```pseudocode
# Hyperparameters: γ=0.99, λ=0.95, KL_coeff=0.1, lr=1e-5, T_max=15 (max steps per episode)
Initialize policy π_θ (Cura 1T)
Initialize patient simulator P (fine-tuned LLaMA-2-7B on MIMIC-III dialogue corpus, with dynamic profile from Gaussian mixture model over demographics and symptoms)
for iteration i = 1..N do
  # Run K episodes (K=64)
  for episode e = 1..K do
    Sample patient profile and task from distribution D (tasks from MedAgentBench, profiles from real EHR statistics)
    Reset SIMPAT environment with patient profile
    state s_0 = initial observation (EHR data, task description)
    for t = 1..T_max do
      action a_t ~ π_θ(·|s_t)  # tool call or natural language
      s_{t+1}, reward r_t = step(a_t)  # r_t = task_completion (binary) + patient_satisfaction (continuous 0-1 from a separately trained classifier on satisfaction annotations)
      store (s_t, a_t, r_t, s_{t+1}, old_logprob)
    end
    Compute advantages A_t using GAE
  end
  # Update policy using PPO
  Update θ by maximizing clipped surrogate objective with KL penalty
  
  # Calibration loop (load-bearing assumption check)
  if i % 1000 == 0:
    Compute Jensen-Shannon divergence between action distributions of real patient logs (200 held-out interactions) and simulator responses
    if JS_div > 0.15:
      Fine-tune simulator P on those 200 logs for 1 epoch (lr=1e-6)
    end
  end
end
```

**Why this design:** We chose PPO over DPO because PPO provides explicit control over the KL divergence from the base model, which is critical for maintaining general capabilities in healthcare. The LLM-based patient simulator (rather than rule-based) captures realistic variation in patient responses, accepting the cost of increased computational overhead and potential mode collapse of the simulator. This design relies on the load-bearing assumption that the simulator's response distribution accurately mirrors real clinical interactions. To mitigate distribution shift, we incorporate a periodic calibration loop that fine-tunes the simulator on real patient logs when divergence exceeds a threshold. Hyperparameter K=64 balances exploration efficiency and training stability; a smaller K leads to high variance, while larger K delays updates. The task reward is binary (task completion) plus a continuous patient satisfaction score from the simulator, incentivizing both goal achievement and communication quality; this dual signal avoids the sparsity of binary rewards alone.

**Why it measures what we claim:** The interactive reward collected from SIMPAT operationalizes the concept of 'interactive coordination' because the reward directly depends on the model's ability to adapt its actions based on the patient's dynamic responses; this equivalence relies on the assumption that the simulator accurately reflects real patient behavior patterns. This assumption fails when the simulator exhibits distribution shift (e.g., overly compliant or adversarial responses), in which case the reward reflects only performance on simulated distribution, not true clinical coordination. The calibration loop aims to keep the simulator realistic, but if the real interaction logs are biased or insufficient, the assumption remains partially unverified. The PPO update with KL penalty measures 'capability retention' because the KL divergence constraint ensures the policy does not deviate far from the base model's safe region; this assumption holds as long as the base model captures sufficient medical knowledge, but fails when the task requires new knowledge not present in the base model, in which case the KL penalty hinders learning.

## Contribution

(1) A training framework that integrates an interactive simulated patient environment into the RL fine-tuning of healthcare LLM agents. (2) A demonstration that interactive training yields models that are more robust to unseen patient queries compared to static trajectory fine-tuning. (3) The introduction of a patient satisfaction reward signal that goes beyond binary task completion.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|------------------------|
| Dataset | MedAgentBench virtual EHR | Realistic interactive healthcare tasks |
| Primary metric | Task completion rate | Direct measure of goal achievement |
| Secondary metric | Patient satisfaction (simulator-rated, 0-1) | Measures communication quality |
| Secondary metric | Diagnostic accuracy (post-hoc clinician review on 100 episodes) | Measures clinical correctness |
| Baseline | GPT-4 (zero-shot) | Frontier general-purpose LLM |
| Baseline | Cura 1T (untrained) | Base model without IAT |
| Baseline | PPO without KL penalty | Tests importance of KL constraint |
| Baseline | RLHF (Cura 1T + human preferences on MedAgentBench trajectories) | Tests advantage over preference-based learning |
| Baseline | Behavioral cloning (Cura 1T fine-tuned on expert trajectories from MedAgentBench) | Tests imitation learning baseline |
| Ablation | IAT with rule-based simulator | Isolates impact of LLM simulator |

### Why this setup validates the claim

This experimental design forms a falsifiable test of IAT's central claim: that interactive coordination and capability retention can be simultaneously improved via PPO with a realistic LLM-based simulator and KL penalty. The dataset MedAgentBench provides realistic interactive EHR tasks requiring tool use and patient communication. Task completion rate directly measures the agent's ability to achieve goals through interaction, capturing both coordination and knowledge application. Patient satisfaction and diagnostic accuracy provide finer-grained evaluation of communication and clinical correctness. GPT-4 tests whether general models already excel; Cura 1T establishes the base performance; PPO without KL isolates the effect of the KL constraint on retention. RLHF and behavioral cloning baselines test whether simpler methods can achieve similar gains, and the rule-based simulator ablation tests the necessity of the LLM simulator's realism. Together, they distinguish the contributions of each component and identify failure modes.

### Expected outcome and causal chain

**vs. GPT-4** — On a case where a patient provides vague symptoms, GPT-4 may ask a single question and then guess incorrectly because it lacks iterative learning from the environment. Our method, trained on thousands of simulated interactions, learns to ask follow-ups and adjust based on patient responses, leading to higher task completion and patient satisfaction. We expect a noticeable gap on complex interactive tasks (e.g., >15% improvement in task completion, >0.1 in satisfaction) but parity on simple single-step tasks.

**vs. Cura 1T (untrained)** — On a case requiring rare disease knowledge, Cura 1T might hallucinate treatments because it has not been fine-tuned for task-specific interaction. Our method uses PPO with KL penalty, which retains the base model's knowledge while learning to apply it correctly via reward. We expect IAT to outperform Cura 1T on knowledge-intensive tasks by ~20% in task completion while maintaining similar recall on factual queries (measured by diagnostic accuracy).

**vs. PPO without KL penalty** — On a case where the environment reward encourages a shortcut (e.g., prescribing unnecessary tests), PPO without KL may exploit this, causing the model to degrade in general capabilities. Our method's KL penalty prevents large policy deviations, preserving safe behavior. We expect IAT with KL to outperform without KL on out-of-distribution tasks (e.g., novel symptoms) by ~10% in task completion and ~0.05 in satisfaction, while both perform similarly on in-distribution tasks.

**vs. RLHF baseline** — On a case where human preferences are sparse or inconsistent, RLHF may converge slowly or to suboptimal policies. IAT's dense reward from the simulator provides more frequent signals, leading to faster convergence and higher final performance. We expect IAT to exceed RLHF by ~5-10% in task completion after same number of environment steps.

**vs. Behavioral cloning** — On a case where expert demonstrations are limited or suboptimal, BC may overfit to narrow behaviors. IAT's reinforcement learning explores beyond demonstrations, discovering more robust strategies. We expect IAT to outperform BC by ~15% on complex multi-step tasks.

### What would falsify this idea

If IAT shows uniform improvement across all task types rather than concentrated gains in interactive coordination, the claimed mechanism is unsupported. Alternatively, if the KL penalty variant underperforms the no-KL variant on capability retention metrics (e.g., factual recall drops by >5%), the constraint may be too restrictive. If the rule-based simulator matches the LLM simulator's performance, the LLM simulator's realism is unnecessary. If RLHF or BC match or exceed IAT on all metrics, then the interactive training paradigm is not advantageous over simpler alternatives.

## References

1. Cura 1T: Specialized Model for Agentic Healthcare
2. MedAgentBench: A Realistic Virtual EHR Environment to Benchmark Medical LLM Agents
