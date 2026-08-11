# Citation Guidance for LC-LLM (Adapted)

This directory is a paper-based reconstruction and domain adaptation. Cite the
final LC-LLM article, the Llama 2 model, and LoRA when using it:

```bibtex
@article{peng2025lcllm,
  title   = {LC-LLM: Explainable Lane-Change Intention and Trajectory
             Predictions with Large Language Models},
  author  = {Peng, Mingxing and Guo, Xusen and Chen, Xianda and Chen, Kehua and
             Zhu, Meixin and Chen, Long and Wang, Fei-Yue},
  journal = {Communications in Transportation Research},
  volume  = {5},
  number  = {2},
  pages   = {100170},
  year    = {2025},
  doi     = {10.1016/j.commtr.2025.100170}
}

@misc{touvron2023llama2,
  title         = {Llama 2: Open Foundation and Fine-Tuned Chat Models},
  author        = {Touvron, Hugo and Martin, Louis and Stone, Kevin and others},
  year          = {2023},
  eprint        = {2307.09288},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL}
}

@inproceedings{hu2022lora,
  title     = {LoRA: Low-Rank Adaptation of Large Language Models},
  author    = {Hu, Edward J. and Shen, Yelong and Wallis, Phillip and Allen-Zhu,
               Zeyuan and Li, Yuanzhi and Wang, Shean and Wang, Lu and Chen,
               Weizhu},
  booktitle = {International Conference on Learning Representations},
  year      = {2022}
}
```

Recommended implementation wording:

> LC-LLM (adapted) is an independent, paper-based reconstruction following Peng
> et al. The Figure 3 prompt structure, Llama-2-13B-chat backbone, joint
> reasoning-intention-trajectory language-modeling objective, and LoRA settings
> were retained where specified. The observation fields, neighbor roles,
> sampling rate, prediction horizon, labels, and evaluation protocol were
> adapted to the post-crash lane-changing data.

Do not describe this implementation as an exact reproduction or as code
released by the LC-LLM authors.
