from __future__ import annotations

import argparse

import torch

from .config import load_config, resolve_dataset_paths
from .dataset import BriscSegmentationDataset, collect_samples, split_train_val
from .models import build_official_swin_unet
from .utils import get_device, set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config.get("seed", 42))
    paths = resolve_dataset_paths(config)
    samples = collect_samples(paths["train_images"], paths["train_masks"])
    train_samples, _ = split_train_val(samples, max_samples=8, seed=config.get("seed", 42))

    ds = BriscSegmentationDataset(
        train_samples,
        pipeline=config["experiment"]["pipeline"],
        image_size=config["image"]["size"],
        preprocessing_cfg=config.get("preprocessing", {}),
        augmentation_cfg=config.get("augmentation", {}),
        mean=config["image"].get("normalize_mean", 0.5),
        std=config["image"].get("normalize_std", 0.5),
        repeat_grayscale_to_rgb=config["image"].get("repeat_grayscale_to_rgb", False),
        is_train=True,
    )
    batch = torch.stack([ds[i]["image"] for i in range(min(2, len(ds)))])
    device = get_device(config.get("device", "cuda"))
    model = build_official_swin_unet(config["model"], config["image"]).to(device)
    with torch.no_grad():
        out = model(batch.to(device))
    print("Entrada:", tuple(batch.shape))
    print("Saída:", tuple(out.shape))


if __name__ == "__main__":
    main()
