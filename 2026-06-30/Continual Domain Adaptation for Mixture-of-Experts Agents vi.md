# Continual Domain Adaptation for Mixture-of-Experts Agents via Expert Augmentation and Replay

## Motivation

The static training pipeline of Agents-A1 assumes the six training domains are comprehensive; performance degrades on unseen domains because no mechanism exists to incorporate novel domain knowledge after training. This structural limitation — the lack of a continual learning mechanism that can detect when inputs fall outside known domains and adapt without forgetting — prevents deployment in open-ended environments where new domains emerge.

## Key Insight

By exploiting the MoE architecture's modularity, we can spawn new experts for novel domains and selectively fine-tune them while using importance-weighted replay of prior expert trajectories to preserve previously learned behaviors, because expert isolation prevents interference and replay counteracts forgetting.

## Method

(A) **What it is**: DAFTER (Domain-Adaptive Fine-Tuning via Expert Augmentation and Replay) is a continual learning pipeline for MoE-based agents that detects new domains via a statistical shift in expert routing distributions, spawns a new expert, and adapts it using synthetic trajectories from a verifier-guided generator while replaying a subset of prior domain trajectories.

(B) **How it works**:
```python
# DAFTER: Continual Domain Adaptation for MoE Agents
Input: pretrained MoE model M (expert set E, N experts), 
       stream S of incoming task batches,
       threshold θ (e.g., 0.1) for domain shift detection,
       replay memory R (initially empty, capacity C=100 per expert),
       synthetic trajectory generator G (using RLVR verifier),
       adaptation steps K (50).

Compute baseline_distribution = mean expert routing over first training batch.

for batch b in S:
    current_distribution = mean routing weights over b
    kl_div = KL(baseline_distribution || current_distribution)
    if kl_div > θ:
        # Identify low-activation experts
        low_act_experts = {e | mean_activation(e, b) < threshold_act (e.g., 0.05)}
        if low_act_experts:
            # Spawn new expert initialized from lowest-activated expert
            new_expert = initialize_expert(architecture, init_from=argmin_activation)
            Add new_expert to E, initialize routing weight = 0.01
            # Generate synthetic trajectories for new domain
            trajectories = G.generate_verifiable_trajectories(domain_context=b, verifier)
            # Fine-tune new expert with replay
            for step in 1..K:
                replay_batch = sample_replay(R, expert_importance_weights)
                loss = L_new(trajectories) + λ (0.5) * L_replay(replay_batch)
                # Gradient flows only into new expert and its routing weight
                update_parameters(new_expert)
            # Update replay memory: add some trajectories, evict oldest
            append_to_replay(R, trajectories, expert_id)
```

(C) **Why this design**: We chose to detect domain shift via KL divergence of routing distributions rather than input embeddings (anti-pattern 2) because the routing distribution directly captures the model's internal representation of domain familiarity; this avoids the need for a separate domain classifier whose training would itself suffer from distribution shift. We spawn a new expert for each gap rather than fine-tune all experts (common in lifelong learning) because MoE's expert isolation ensures that updating one expert minimally impacts others, making it suitable for incremental learning without catastrophic forgetting; the cost is that the total number of experts grows linearly with detected domains, which we accept given the computational budget of agent deployment. We use a verifier-guided trajectory generator (inspired by TÜLU 3's RLVR) to produce high-quality synthetic data without human annotation, leveraging verifiable rewards for correctness; this trade-off limits applicability to domains with objectively verifiable outcomes. Finally, we incorporate replay of prior trajectories weighted by expert activation importance to prevent forgetting; this introduces memory cost but ensures stability of previously learned behaviors.

(D) **Why it measures what we claim**: The KL divergence of expert routing distributions measures domain shift because we assume that the model's routing decisions are a sufficient statistic for input domain identity; this assumption fails when a new domain induces a different routing distribution that still yields correct outputs (false positive), leading to unnecessary expert spawning and increased model size without performance gain. The low-activation threshold measures coverage gaps because we assume that low activation indicates the domain is not well-covered; this assumption fails when the domain is covered but the model routes to few experts due to sparsity bias (e.g., top-2 routing), causing redundant expert creation. The fine-tuning loss L_new on synthetic trajectories measures adaptation to the new domain because we assume the verifier-guided generator produces correct and representative trajectories; this assumption fails when the verifier is incomplete or the generator exploits blind spots (e.g., reward hacking), resulting in the new expert learning spurious patterns. The replay loss L_replay measures retention of prior domains because we assume that replayed trajectories are drawn from the same distribution as training; this assumption fails when replay memory is biased or outdated (e.g., due to capacity constraints), causing forgetting despite replay. By explicitly naming these assumptions, we provide a causal bridge between computational quantities and the motivation-level concepts of domain adaptation and forgetting.

## Contribution

(1) DAFTER, a continual learning pipeline for Mixture-of-Experts agents that detects domain gaps via routing distribution shifts and stably adapts by spawning new experts with selective fine-tuning and replay. (2) A principle for leveraging expert modularity in MoEs to enable lifelong learning without catastrophic forgetting, demonstrated by the design of gap detection and expert augmentation. (3) An analysis of the assumptions underlying the method's operationalization of domain adaptation and forgetting prevention, providing a causal bridge between computational quantities and motivation-level concepts.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | Long-horizon agent benchmarks (6 domains) | Covers diverse domain shifts for detection |
| Primary metric | Average success rate across all seen domains | Measures adaptation and forgetting |
| Baseline 1 | Full fine-tune all experts | Tests catastrophic forgetting baseline |
| Baseline 2 | Domain classifier + new expert | Tests embedding shift detection baseline |
| Baseline 3 | DAFTER without replay | Ablates replay retention component |
| Ablation-of-ours | DAFTER with random trajectories | Ablates verifier-guided generator |

### Why this setup validates the claim

This setup provides a falsifiable test of DAFTER's central claim: that routing-distribution-based domain detection and expert spawning with replayed synthetic trajectories enable continual adaptation without forgetting. The long-horizon benchmarks span six distinct domains, creating natural distribution shifts that trigger our detection mechanism. The primary metric—average success across all seen domains—directly captures the trade-off between new domain performance and retention of old domains. Baseline 1 (full fine-tune) isolates the forgetting problem; Baseline 2 (domain classifier) tests whether our routing-based detection is superior; and Baseline 3 (no replay) tests the necessity of replay. Our ablation of random trajectories isolates the value of verifier-guided generation. Together, these comparisons expose whether our method's gains stem from each designed component or from confounding factors.

### Expected outcome and causal chain

**vs. Full fine-tune all experts** — On a case where a new domain arrives after training on several old domains, full fine-tune updates all experts, causing the model to overwrite weights specialized for prior domains (e.g., a coding agent losing math skills). Our method instead spawns a new expert and only updates that expert and its routing weight, leaving other experts untouched, so we expect a large gap on old-domain tasks (e.g., ≥20% higher success) while matching or slightly exceeding full fine-tune on the new domain.

**vs. Domain classifier + new expert** — On a case where the new domain's input embeddings are similar to an old domain but the routing distribution differs (e.g., two distinct natural language instructions requiring different tools), the domain classifier misclassifies it as old domain and fails to spawn an expert, leading to poor adaptation. Our method detects the shift via routing distribution, spawns an expert, and adapts correctly, so we expect a noticeable gap on domains with high embedding similarity but different routing patterns (e.g., >15% higher success on those subsets).

**vs. DAFTER without replay** — On a case where a previously learned domain is revisited after learning several new domains, DAFTER without replay gradually forgets because the new expert's training distracts routing away from old experts. Our method with replay samples old trajectories to maintain routing stability, so we expect a smaller drop on old domains (e.g., <5% drop vs. >15% drop for no replay) while achieving similar new domain performance.

### What would falsify this idea
We would consider the central claim falsified if the gain of DAFTER over baselines is uniform across all subsets rather than concentrated on domains where the predicted failure mode (e.g., forgetting, embedding similarity) occurs, or if the ablation showing no difference between verifier-guided and random trajectories indicates the generator is unnecessary.

## References

1. Scaling the Horizon, Not the Parameters: Reaching Trillion-Parameter Performance with a 35B Agent
2. Generalizing Verifiable Instruction Following
3. TÜLU 3: Pushing Frontiers in Open Language Model Post-Training
