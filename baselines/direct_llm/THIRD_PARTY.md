# Third-Party References for Direct LLM (Adapted)

## Method and Official Implementation

The direct coordinate-sequence prompting paradigm follows:

- Inhwan Bae, Junoh Lee, and Hae-Gon Jeon. **Can Language Beat Numerical
  Regression? Language-Based Multimodal Trajectory Prediction**. CVPR, 2024,
  pp. 753-766. [CVPR paper](https://openaccess.thecvf.com/content/CVPR2024/html/Bae_Can_Language_Beat_Numerical_Regression_Language-Based_Multimodal_Trajectory_Prediction_CVPR_2024_paper.html)
  | [Official repository](https://github.com/InhwanBae/LMTrajectory)

Unlike LC-LLM, this method has an author-released implementation. The official
LMTrajectory repository is distributed under the
[Creative Commons Attribution-NonCommercial 4.0 International license](https://github.com/InhwanBae/LMTrajectory/blob/main/LICENSE).
That license permits covered upstream material only under its terms, including
the noncommercial restriction and attribution requirements.

No upstream source file, prompt string, dataset, response dump, or model asset
is copied into this directory. The code and prompt wording here were written
independently from the published method and publicly documented paradigm. The
official repository is linked for provenance and methodological verification,
not vendored as a dependency.

## Scope of the Adaptation

The official LMTraj-ZERO code targets ETH/UCY pedestrian trajectories and asks
for five future candidates in each of four conversational rounds. This release
instead predicts one post-crash vehicle trajectory from the available 10-frame
observation, optionally includes six anonymized neighboring trajectories, and
uses a strict 50-point JSON output. It is therefore not a numerical reproduction
of the official benchmark pipeline.

## LLM Service and SDK

Inference uses a user-configured OpenAI-compatible endpoint through the OpenAI
Python SDK. The example names environment variables but includes no endpoint,
credential, or account information. Users are responsible for the selected
model's access terms, API charges, data-handling policy, retention policy, and
regional requirements. The configured model name does not redistribute or
license model weights.

## Data and Artifacts

This directory includes no ETH/UCY data, private post-crash trajectory data,
crash video, generated prompt containing a real sample, LLM response, API key,
private endpoint, or evaluation output.
