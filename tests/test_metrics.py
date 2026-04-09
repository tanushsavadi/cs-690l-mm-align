import pandas as pd

from mm_align.eval.metrics import normalize_yes_no, pope_metrics, relaxed_chartqa_match


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


def test_normalize_yes_no_supports_natural_language_answers() -> None:
    assert normalize_yes_no("Yes, there is a snowboard in the image.") == "yes"
    assert normalize_yes_no("No, there is no dining table in the image.") == "no"
    assert normalize_yes_no("There is no car present in the image.") == "no"
    assert normalize_yes_no("There is a truck in the image.") == "yes"


def test_pope_metrics_supports_natural_language_predictions() -> None:
    frame = pd.DataFrame(
        {
            "prediction": [
                "Yes, there is a snowboard in the image.",
                "There is no car present in the image.",
                "No, there is no dining table in the image.",
                "There is a truck in the image.",
            ],
            "ground_truth": ["yes", "no", "no", "yes"],
        }
    )
    metrics = pope_metrics(frame)
    assert metrics["accuracy"] == 1.0
    assert metrics["yes_ratio"] == 0.5
