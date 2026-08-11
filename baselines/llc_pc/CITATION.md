# Citation Guidance for LLC-PC

When using LLC-PC, cite the CoT-TP article associated with this repository and
the following method and architecture references. The LLC-PC implementation
should be described as a domain adaptation, not an exact reproduction.

```bibtex
@misc{zheng2024llmcontextmotion,
  title         = {Large Language Models Powered Context-aware Motion Prediction},
  author        = {Zheng, Xiaoji and Wu, Lixiu and Yan, Zhijie and Tang, Yuanrong and Zhao, Hao and Zhong, Chen and Chen, Bokui and Gong, Jiangtao},
  year          = {2024},
  eprint        = {2403.11057},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV}
}

@inproceedings{shi2022motiontransformer,
  title     = {Motion Transformer with Global Intention Localization and Local Movement Refinement},
  author    = {Shi, Shaoshuai and Jiang, Li and Dai, Dengxin and Schiele, Bernt},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2022}
}
```

Recommended implementation wording:

> LLC-PC is an independently implemented, domain-adapted baseline following the
> semantic-context conditioning design of Zheng et al. The data adapter, local
> context-map renderer, observation-based retrieval features, prompt, and
> compact MTR-style predictor were implemented for the post-crash lane-changing
> data used in this study.
