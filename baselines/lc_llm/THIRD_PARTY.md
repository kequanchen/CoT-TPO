# Third-Party References and Notices for LC-LLM (Adapted)

## LC-LLM Article and Figure 3 Prompt

This implementation follows:

- Mingxing Peng, Xusen Guo, Xianda Chen, Kehua Chen, Meixin Zhu, Long Chen, and
  Fei-Yue Wang. **LC-LLM: Explainable lane-change intention and trajectory
  predictions with Large Language Models**. *Communications in Transportation
  Research*, 5(2), 100170, 2025.
  [Article](https://doi.org/10.1016/j.commtr.2025.100170) |
  [arXiv manuscript](https://arxiv.org/abs/2403.18344)

The final article is distributed under the
[Creative Commons Attribution 4.0 International license](https://creativecommons.org/licenses/by/4.0/).
The prompt structure and some task language in this directory are reconstructed
from Figure 3 and attributed to Peng et al. They have been modified for a
one-second observation, six dataset-specific neighbor roles, and a five-second,
50-point output. The Figure 3 image itself is not redistributed.

No author-released LC-LLM implementation, preprocessing program, adapter
weights, or machine-readable prompt file was available to this project when
this baseline was prepared. All source code in this directory was written
independently from the published method description. This is therefore a
paper-based reconstruction, not an exact reproduction.

## Llama 2

The paper uses Llama-2-13B-chat. Model files are not included. Downloading or
using Llama 2 requires accepting and complying with the
[Meta Llama 2 Community License](https://github.com/meta-llama/llama/blob/main/LICENSE)
and its acceptable-use policy. The configuration contains only a placeholder;
users must obtain lawful access and provide their own local path or model ID.

## LoRA and Software Dependencies

LoRA follows Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*.
Training and inference use independently installed PyTorch, Transformers, PEFT,
Accelerate, bitsandbytes, and safetensors packages. Their respective licenses
apply to those installations; none of those projects is vendored here.

## Data and Released Artifacts

This directory contains no highD data, private post-crash trajectories, crash
videos, prompts generated from real samples, model responses, model weights,
LoRA adapters, access tokens, or evaluation results. The public configuration
contains placeholders only.
