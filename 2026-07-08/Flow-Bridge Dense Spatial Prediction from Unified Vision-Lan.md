# Flow-Bridge: Dense Spatial Prediction from Unified Vision-Language Models via Flow-Matching Decoding

## Motivation

Current unified vision-language models (e.g., SenseNova-Vision) treat dense prediction tasks as image generation or text description, losing metric precision and continuous accuracy. This limitation arises because their output space is restricted to discrete tokens or RGB pixels, which cannot directly represent continuous spatial quantities like depth or surface normals. As a result, task-specific heads or post-processing pipelines are required, breaking the unified paradigm and introducing extra complexity.

## Key Insight

Generative tokens from a pretrained unified VLM encode rich spatial semantics that can be decoded into continuous representations via a flow-matching process, eliminating the need for task-specific architectures while preserving metric fidelity.

## Method

the revised method string with specifications inserted

## Contribution

(1) A flow-matching adapter that attaches to any pretrained unified VLM's decoder hidden states, enabling direct generation of continuous spatial maps without task-specific heads or architectural modifications. (2) An empirical finding that generative tokens from VLMs contain sufficient spatial information to condition a flow-based decoder, achieving competitive performance on depth, normal, and segmentation tasks while preserving original language-image capabilities. (3) A training recipe that combines flow-matching loss with the VLM's original generative loss via a shared encoder, ensuring no catastrophic forgetting.

## Experiment

the revised experiment markdown string with table updated and paragraphs refined

## References

1. Vision as Unified Multimodal Generation
2. Scaling Spatial Intelligence with Multimodal Foundation Models
3. Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution
