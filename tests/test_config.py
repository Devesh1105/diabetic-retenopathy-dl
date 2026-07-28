"""Tests for config loading. These do not require TensorFlow."""

from __future__ import annotations

from pathlib import Path

import pytest

from dr.config import ExperimentConfig, PathConfig

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
CONFIG_FILES = sorted(CONFIG_DIR.glob("*.yaml"))


@pytest.mark.parametrize("config_path", CONFIG_FILES, ids=lambda p: p.stem)
def test_shipped_configs_load(config_path: Path) -> None:
    config = ExperimentConfig.from_yaml(config_path)

    assert config.name
    assert isinstance(config.paths, PathConfig)
    assert config.data.num_classes == len(config.data.class_names)
    assert config.data.augmentation_backend in {"keras", "albumentations"}
    assert config.training.num_epochs > 0
    assert config.data.batch_size > 0


@pytest.mark.parametrize("config_path", CONFIG_FILES, ids=lambda p: p.stem)
def test_config_roundtrips_through_yaml(config_path: Path, tmp_path: Path) -> None:
    original = ExperimentConfig.from_yaml(config_path)
    destination = tmp_path / "config.yaml"
    original.save(destination)

    assert ExperimentConfig.from_yaml(destination).to_dict() == original.to_dict()


def test_cli_overrides_win_over_file(tmp_path: Path) -> None:
    config = ExperimentConfig.from_yaml(
        CONFIG_DIR / "baseline.yaml",
        {"paths.dataset": str(tmp_path), "training.num_epochs": 3},
    )

    assert config.paths.dataset == tmp_path
    assert config.training.num_epochs == 3


def test_missing_dataset_path_is_rejected(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("DR_DATASET_PATH", raising=False)
    incomplete = tmp_path / "incomplete.yaml"
    incomplete.write_text("name: incomplete\n", encoding="utf-8")

    with pytest.raises(ValueError, match="dataset path"):
        ExperimentConfig.from_yaml(incomplete)


def test_dataset_path_falls_back_to_environment(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("DR_DATASET_PATH", str(tmp_path))
    minimal = tmp_path / "minimal.yaml"
    minimal.write_text("name: minimal\n", encoding="utf-8")

    assert ExperimentConfig.from_yaml(minimal).paths.dataset == tmp_path


def test_unknown_augmentation_backend_is_rejected(tmp_path: Path) -> None:
    invalid = tmp_path / "invalid.yaml"
    invalid.write_text(
        f"name: invalid\npaths:\n  dataset: {tmp_path}\n"
        "data:\n  augmentation_backend: torchvision\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="augmentation_backend"):
        ExperimentConfig.from_yaml(invalid)


def test_split_paths_are_derived_from_dataset_root(tmp_path: Path) -> None:
    paths = PathConfig(dataset=tmp_path)

    assert paths.split("train") == tmp_path / "train"
    assert paths.split("validation") == tmp_path / "validation"
