# Third-Party References for LLC-PC

LLC-PC is an independent, domain-adapted implementation. No source files,
prompt text, model weights, or data from the projects below are copied into this
directory.

## Method Reference

The semantic-context workflow is based on:

- Xiaoji Zheng, Lixiu Wu, Zhijie Yan, Yuanrong Tang, Hao Zhao, Chen Zhong,
  Bokui Chen, and Jiangtao Gong. **Large Language Models Powered Context-aware
  Motion Prediction**. arXiv:2403.11057, 2024.
  [Paper](https://arxiv.org/abs/2403.11057) ·
  [Reference repository](https://github.com/AIR-DISCOVER/LLM-Augmented-MTR)

The reference repository is used to understand the published method only. This
release does not redistribute its prompt or implementation.

## Motion Transformer Reference

The downstream query-decoding design follows the Motion Transformer (MTR)
architecture:

- Shaoshuai Shi, Li Jiang, Dengxin Dai, and Bernt Schiele. **Motion Transformer
  with Global Intention Localization and Local Movement Refinement**. NeurIPS,
  2022. [Paper](https://arxiv.org/abs/2209.13508) ·
  [Code](https://github.com/sshaoshuai/MTR)

The original MTR codebase is distributed under the Apache License 2.0. It is not
vendored here. If a user replaces the included independent MTR-style components
with upstream MTR code, that installation remains subject to the upstream
Apache-2.0 license and notice requirements.

## Scope of This Adaptation

The included implementation replaces WOMD-specific inputs and learned retrieval
embeddings with observed ego and six-neighbor histories, simplified local lane
geometry, and standardized flattened observation features. It retains the
17-dimensional action, affordance, and scenario interface. Semantic contexts
and intention-point anchors are projected independently and then combined into
the query tokens of a compact MTR-style decoder. Consequently, results should
be described as **LLC-PC, an independent post-crash lane-changing adaptation
following Zheng et al.**, not as an exact reproduction of LLM-Augmented-MTR.
