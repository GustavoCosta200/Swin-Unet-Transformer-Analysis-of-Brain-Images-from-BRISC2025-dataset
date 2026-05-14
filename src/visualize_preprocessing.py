from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from .config import load_config, resolve_dataset_paths
from .dataset import collect_samples
from .preprocessing import preprocess_image_and_mask
from .utils import ensure_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--num-samples", type=int, default=5)
    args = parser.parse_args()

    config = load_config(args.config)
    paths = resolve_dataset_paths(config)
    samples = collect_samples(paths["train_images"], paths["train_masks"])
    out_dir = ensure_dir(Path(config["paths"]["output_dir"]) / "preprocessing_preview")

    for idx, sample in enumerate(samples[: args.num_samples]):
        image = cv2.imread(str(sample.image_path), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(sample.mask_path), cv2.IMREAD_GRAYSCALE)
        panels = []
        for pipeline in ["P0", "P1", "P2", "P3"]:
            processed, processed_mask = preprocess_image_and_mask(
                image=image.copy(),
                mask=mask.copy(),
                pipeline=pipeline,
                image_size=config["image"]["size"],
                preprocessing_cfg=config.get("preprocessing", {}),
            )
            title_bar = np.zeros((30, processed.shape[1]), dtype=np.uint8)
            cv2.putText(title_bar, pipeline, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2)
            mask_vis = (processed_mask * 255).astype(np.uint8)
            panels.append(np.concatenate([title_bar, processed, mask_vis], axis=0))
        grid = np.concatenate(panels, axis=1)
        cv2.imwrite(str(out_dir / f"preview_{idx:03d}.png"), grid)

    print(f"Pré-visualizações salvas em: {out_dir}")


if __name__ == "__main__":
    main()
