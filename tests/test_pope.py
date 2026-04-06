from pathlib import Path

from mm_align.data.pope import _load_pope_payload


def test_load_pope_payload_supports_jsonl(tmp_path: Path) -> None:
    source_path = tmp_path / "pope.json"
    source_path.write_text(
        "\n".join(
            [
                '{"question_id": 1, "image": "a.jpg", "text": "Is there a cat?", "label": "yes"}',
                '{"question_id": 2, "image": "b.jpg", "text": "Is there a dog?", "label": "no"}',
            ]
        ),
        encoding="utf-8",
    )

    payload = _load_pope_payload(source_path)
    assert len(payload) == 2
    assert payload[0]["question_id"] == 1
    assert payload[1]["label"] == "no"
