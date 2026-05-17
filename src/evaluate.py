from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import load_config, resolve_dataset_paths
from .dataset import (
    BriscSegmentationDataset,
    balance_samples_by_class,
    collect_samples,
    print_class_distribution,
)
from .metrics import MetricAverager, segmentation_metrics
from .models import build_official_swin_unet
from .utils import ensure_dir, get_device, save_json, set_seed


def save_prediction_grid(image, mask, pred, out_path: Path) -> None:
    image = image.detach().cpu().numpy()
    if image.ndim == 3 and image.shape[0] == 3:
        image = image[0]
    else:
        image = image.squeeze()

    image = ((image * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)
    mask = (mask.squeeze().detach().cpu().numpy() * 255).astype(np.uint8)
    pred = (pred.squeeze().detach().cpu().numpy() * 255).astype(np.uint8)

    overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    overlay[pred > 0, 2] = 255

    grid = np.concatenate(
        [
            cv2.cvtColor(image, cv2.COLOR_GRAY2BGR),
            cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR),
            cv2.cvtColor(pred, cv2.COLOR_GRAY2BGR),
            overlay,
        ],
        axis=1,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), grid)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--save-images", type=int, default=12)
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config.get("seed", 42))

    device = get_device(config.get("device", "cuda"))
    pipeline = config["experiment"]["pipeline"]
    experiment_cfg = config.get("experiment", {})

    paths = resolve_dataset_paths(config)
    samples = collect_samples(
        paths["test_images"],
        paths["test_masks"],
        class_aliases=experiment_cfg.get("class_aliases"),
    )
    print_class_distribution(samples, "Distribuição original do teste")

    max_test = experiment_cfg.get("max_test_samples")
    if experiment_cfg.get("balance_classes", False):
        samples = balance_samples_by_class(
            samples,
            class_names=experiment_cfg.get("balance_class_names"),
            max_samples=max_test,
            seed=config.get("seed", 42),
        )
    elif max_test is not None:
        samples = samples[: int(max_test)]

    print_class_distribution(samples, "Distribuição final do teste")

    dataset = BriscSegmentationDataset(
        samples=samples,
        pipeline=pipeline,
        image_size=config["image"]["size"],
        preprocessing_cfg=config.get("preprocessing", {}),
        augmentation_cfg=config.get("augmentation", {}),
        mean=config["image"].get("normalize_mean", 0.5),
        std=config["image"].get("normalize_std", 0.5),
        repeat_grayscale_to_rgb=config["image"].get("repeat_grayscale_to_rgb", False),
        is_train=False,
    )

    loader = DataLoader(
        dataset,
        batch_size=config["train"]["batch_size"],
        shuffle=False,
        num_workers=experiment_cfg.get("num_workers", 0),
    )

    model = build_official_swin_unet(config["model"], config["image"]).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    output_dir = ensure_dir(Path(config["paths"]["output_dir"]) / pipeline / "test")
    pred_dir = ensure_dir(output_dir / "predictions")

    avg = MetricAverager()
    rows = []
    saved = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader)):
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            logits = model(images)
            metrics = segmentation_metrics(
                logits,
                masks,
                config["train"].get("threshold", 0.5),
            )

            avg.update(metrics, n=images.size(0))
            rows.append({"batch": batch_idx, **metrics})

            preds = (
                torch.sigmoid(logits) >= config["train"].get("threshold", 0.5)
            ).float()

            for i in range(images.size(0)):
                if saved >= args.save_images:
                    break
                save_prediction_grid(
                    images[i],
                    masks[i],
                    preds[i],
                    pred_dir / f"sample_{saved:03d}.png",
                )
                saved += 1

    final_metrics = avg.compute()
    save_json(final_metrics, output_dir / "test_metrics.json")
    pd.DataFrame(rows).to_csv(output_dir / "test_batches.csv", index=False)

    print("Métricas finais:", final_metrics)


if __name__ == "__main__":
    main()
