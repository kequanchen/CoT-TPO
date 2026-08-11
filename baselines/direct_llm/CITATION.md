# Citation Guidance for Direct LLM (Adapted)

The direct coordinate-generation baseline follows the zero-shot prompt paradigm
introduced by LMTraj-ZERO. Cite the CVPR paper:

```bibtex
@inproceedings{bae2024lmtrajectory,
  title     = {Can Language Beat Numerical Regression? Language-Based
               Multimodal Trajectory Prediction},
  author    = {Bae, Inhwan and Lee, Junoh and Jeon, Hae-Gon},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and
               Pattern Recognition},
  pages     = {753--766},
  year      = {2024},
  doi       = {10.1109/CVPR52733.2024.00078}
}
```

Official project code:

- <https://github.com/InhwanBae/LMTrajectory>

Recommended implementation wording:

> Direct LLM (adapted) is an independently implemented post-crash trajectory
> baseline following the zero-shot coordinate-sequence prompting paradigm of
> LMTraj-ZERO. The prompt wording, data adapter, vehicle context representation,
> output schema, retry handling, and evaluation pipeline were implemented for
> this study. One 50-point top-1 trajectory is generated for each sample under
> the same LLM setting used for the other LLM-based comparisons.

Do not describe this directory as a complete reproduction of LMTraj-ZERO or as
a modified copy of its official code.
