from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
import yaml


def load_config(config_path: str | Path) -> Dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Arquivo de configuração não encontrado: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_config(config: Dict[str, Any], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)


def resolve_dataset_paths(config: Dict[str, Any]) -> Dict[str, Path]:
    root = Path(config["paths"]["dataset_root"])
    return {
        "train_images": root / config["paths"]["train_images"],
        "train_masks": root / config["paths"]["train_masks"],
        "test_images": root / config["paths"]["test_images"],
        "test_masks": root / config["paths"]["test_masks"],
    }
