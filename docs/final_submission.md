# Final Submission Checklist

The CS 690L final submission should be a zip file containing:

- `report.pdf`
- `video.mp4` or `video.webm`

The course allows code/results to be included too, but the required files are
the report and video.

## Current Report

The final report source is:

```text
reports/final_report.tex
```

The compiled report is:

```text
reports/final_report.pdf
```

The correctly named submission copy is:

```text
submission/report.pdf
```

The report uses a NeurIPS-style LaTeX format through:

```text
reports/neurips_2026.sty
```

It includes:

- abstract
- motivation and research question
- data and method
- experiment setup
- engineering/logistics section
- final benchmark results
- statistical evidence
- dashboard description
- limitations
- contribution and code availability section
- real citations and URLs

## Video

The video should be 3 to 5 minutes.

Use this story:

1. Problem: VLMs can hallucinate unsupported image details.
2. Method: compare standard DPO and image-aware DPO.
3. Setup: Qwen2.5-VL-3B, RLAIF-V, LoRA adapters, three benchmarks.
4. Evidence: ChartQA improves, HallusionBench nearly ties, POPE slightly favors
   standard DPO.
5. Dependence: image-aware DPO changes answers slightly more under blank and
   mismatched images.
6. Limitation: one seed, expensive Colab pipeline, modest result.
7. Takeaway: promising pilot, not a solved hallucination method.

The final video script is:

```text
reports/final_video_story.md
```

## Dashboard Recording Path

Recommended dashboard order:

```text
Story Map -> Evidence -> Dependence -> optional Examples
```

Do not spend time on `Preferences` in the video. Some raw images may not exist
locally, and it is not the strongest proof page.

## Build Final Zip

After placing the recorded video into `submission/`, run one of these commands.

For MP4:

```bash
cd "/Users/tanushsavadi/Documents/CS 690L/submission"
zip final_project_submission.zip report.pdf video.mp4
```

For WebM:

```bash
cd "/Users/tanushsavadi/Documents/CS 690L/submission"
zip final_project_submission.zip report.pdf video.webm
```

Before submitting, verify:

```bash
ls -lh final_project_submission.zip report.pdf video.*
```

## Last Sanity Checks

Run tests:

```bash
cd "/Users/tanushsavadi/Documents/CS 690L"
/Library/Frameworks/Python.framework/Versions/3.12/bin/python3 -m pytest
```

Launch dashboard:

```bash
bash scripts/run_dashboard.sh
```

Open the report PDF and quickly check:

- the title appears
- the author name appears
- the contribution section appears
- the GitHub URL appears
- references appear at the end
- the PDF is more than 3 pages
