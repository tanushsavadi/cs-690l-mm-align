import pandas as pd

from mm_align.eval.metrics import pope_metrics, relaxed_chartqa_match


def test_relaxed_chartqa_match_supports_numeric_tolerance() -> None:
    assert relaxed_chartqa_match("105", "100")
    assert not relaxed_chartqa_match("130", "100")


def test_pope_metrics_basic_case() -> None:
    frame = pd.DataFrame(
        {
            "prediction": ["yes", "no", "yes", "no"],
            "ground_truth": ["yes", "no", "no", "yes"],
        }
    )
    metrics = pope_metrics(frame)
    assert metrics["accuracy"] == 0.5
    assert round(metrics["precision"], 3) == 0.5
