import pandas as pd

from mm_align.data.common import add_mismatch_paths


def test_add_mismatch_paths_is_deterministic() -> None:
    frame = pd.DataFrame({"image_path": ["a.png", "b.png", "c.png", "d.png"]})
    updated = add_mismatch_paths(frame)
    assert updated["mismatch_image_path"].tolist() == ["c.png", "d.png", "a.png", "b.png"]
