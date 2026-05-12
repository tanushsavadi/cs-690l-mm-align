# Documentation Index

This folder explains the project in the way I would want a grader or future
reader to understand it. The code is still the source of truth, but these docs
summarize how the pieces fit together.

## Start Here

- [Project overview](project_overview.md): simple explanation of the research
  question, method, results, and final claim.
- [Reproducibility and Colab runbook](reproducibility.md): how the full
  experiment was run, why Colab GPUs were needed, and how to rerun or resume it.
- [Dashboard guide](dashboard.md): how to launch the local Streamlit dashboard
  and which pages to show in the final presentation.
- [Artifact guide](artifacts.md): what files each completed run should contain
  and how to validate them.
- [Final submission checklist](final_submission.md): what needs to go into the
  final zip file for CS 690L.

## Final Completed Runs

The final report and dashboard use these two run ids:

- `2026-04-08-standard_dpo-pilot-7`
- `2026-04-08-image_aware_dpo-pilot-7`

These are the source of truth for the final numbers in the report.

## Main Result

The clean reading is not that image-aware DPO wins everywhere. The honest
summary is:

- ChartQA improves from `0.3797` to `0.4063`.
- HallusionBench is basically tied, `0.5722` vs `0.5740`.
- POPE slightly favors standard DPO, `0.8733` vs `0.8718`.
- Image-aware DPO changes answers slightly more often under blank and
  mismatched image perturbations.

So the project supports a modest grounding-related gain, especially on ChartQA,
but not a complete solution to multimodal hallucination.
