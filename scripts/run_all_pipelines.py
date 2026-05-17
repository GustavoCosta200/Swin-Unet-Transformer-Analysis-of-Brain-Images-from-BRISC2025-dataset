from __future__ import annotations

import argparse
import copy
import subprocess
import sys
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    base_path = Path(args.config)
    with base_path.open("r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)

    tmp_dir = Path("configs/generated")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    for pipeline in ["P2", "P3"]:
        cfg = copy.deepcopy(base_cfg)
        cfg["experiment"]["pipeline"] = pipeline
        cfg_path = tmp_dir / f"{pipeline}.yaml"
        with cfg_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

        print(f"\nExecutando pipeline {pipeline}")
        subprocess.run([sys.executable, "-m", "src.train", "--config", str(cfg_path)], check=True)


if __name__ == "__main__":
    main()
