from pathlib import Path

from mm_align.config import load_config


def test_load_config_resolves_runtime_paths() -> None:
    config = load_config(Path("configs/smoke.yaml"), repo_root=Path.cwd())
    assert config.runtime.raw_dir.is_absolute()
    assert config.runtime.processed_dir.is_absolute()
    assert config.model.base_model_name == "Qwen/Qwen2.5-VL-3B-Instruct"


def test_load_config_applies_runtime_env_overrides(monkeypatch) -> None:
    monkeypatch.setenv("MM_ALIGN_RAW_DIR", "/tmp/mm-align/raw")
    monkeypatch.setenv("MM_ALIGN_PROCESSED_DIR", "/tmp/mm-align/processed")
    monkeypatch.setenv("MM_ALIGN_ARTIFACTS_DIR", "/tmp/mm-align/artifacts")

    config = load_config(Path("configs/smoke.yaml"), repo_root=Path.cwd())
    raw_root = Path("/tmp/mm-align/raw")
    processed_root = Path("/tmp/mm-align/processed")
    artifacts_root = Path("/tmp/mm-align/artifacts")
    assert config.runtime.raw_dir == raw_root
    assert config.runtime.processed_dir == processed_root
    assert config.runtime.artifacts_dir == artifacts_root
    assert Path(config.datasets.hallusionbench.path) == (raw_root / "hallusionbench" / "HallusionBench.json").resolve()
    assert Path(config.datasets.pope.path) == (raw_root / "pope").resolve()
    assert Path(config.datasets.chartqa.path) == (raw_root / "chartqa").resolve()
