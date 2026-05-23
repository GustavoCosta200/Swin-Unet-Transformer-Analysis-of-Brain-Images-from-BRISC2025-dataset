from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import load_config, resolve_dataset_paths, save_config
from .dataset import (
    BriscSegmentationDataset,
    collect_samples,
    print_class_distribution,
    split_train_val,
)
from .losses import build_loss
from .metrics import MetricAverager, segmentation_metrics
from .models import build_official_swin_unet
from .utils import ensure_dir, get_device, save_json, set_seed

from .visualization import save_pipeline_debug_sample


def build_dataloaders(config: Dict) -> Tuple[DataLoader, DataLoader]:
    paths = resolve_dataset_paths(config)
    experiment_cfg = config.get("experiment", {})
    balance_classes = bool(experiment_cfg.get("balance_classes", False))
    class_names = experiment_cfg.get("balance_class_names")
    class_aliases = experiment_cfg.get("class_aliases")

    samples = collect_samples(
        paths["train_images"],
        paths["train_masks"],
        class_aliases=class_aliases,
    )
    print_class_distribution(samples, "Distribuição original do treino")

    train_samples, val_samples = split_train_val(
        samples,
        val_size=experiment_cfg.get("val_size", 0.2),
        seed=config.get("seed", 42),
        max_samples=experiment_cfg.get("max_train_samples"),
        balance_classes=balance_classes,
        class_names=class_names,
    )

    print_class_distribution(train_samples, "Distribuição final do conjunto de treino")
    print_class_distribution(val_samples, "Distribuição final do conjunto de validação")

    common_kwargs = dict(
        pipeline=experiment_cfg["pipeline"],
        image_size=config["image"]["size"],
        preprocessing_cfg=config.get("preprocessing", {}),
        augmentation_cfg=config.get("augmentation", {}),
        mean=config["image"].get("normalize_mean", 0.5),
        std=config["image"].get("normalize_std", 0.5),
        repeat_grayscale_to_rgb=config["image"].get("repeat_grayscale_to_rgb", False),
    )

    train_ds = BriscSegmentationDataset(train_samples, is_train=True, **common_kwargs)
    val_ds = BriscSegmentationDataset(val_samples, is_train=False, **common_kwargs)

    train_loader = DataLoader(
        train_ds,
        batch_size=config["train"]["batch_size"],
        shuffle=True,
        num_workers=experiment_cfg.get("num_workers", 0),
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config["train"]["batch_size"],
        shuffle=False,
        num_workers=experiment_cfg.get("num_workers", 0),
        pin_memory=torch.cuda.is_available(),
    )
    
    save_pipeline_debug_sample(
        dataset=train_ds,
        cfg=config,
        split_name="train",
        sample_index=0
    )

    return train_loader, val_loader


def run_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    threshold: float,
    train: bool,
) -> Dict[str, float]:
    model.train(train)

    avg = MetricAverager()
    loss_total = 0.0
    sample_count = 0

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for batch in tqdm(loader, leave=False):
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)

            if train:
                optimizer.zero_grad(set_to_none=True)

            logits = model(images)
            loss = criterion(logits, masks)

            if train:
                loss.backward()
                optimizer.step()

            n = images.size(0)
            loss_total += float(loss.detach().cpu()) * n
            sample_count += n
            avg.update(segmentation_metrics(logits, masks, threshold), n=n)

    metrics = avg.compute()
    metrics["loss"] = loss_total / max(sample_count, 1)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    pipeline = config["experiment"]["pipeline"]

    set_seed(config.get("seed", 42))
    device = get_device(config.get("device", "cuda"))

    output_dir = ensure_dir(Path(config["paths"]["output_dir"]) / pipeline)
    checkpoint_dir = ensure_dir(config["paths"]["checkpoint_dir"])

    save_config(config, output_dir / "used_config.yaml")

    train_loader, val_loader = build_dataloaders(config)

    model = build_official_swin_unet(config["model"], config["image"]).to(device)
    criterion = build_loss(config["train"].get("loss", "bce_dice"))

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["train"].get("learning_rate", 1e-4),
        weight_decay=config["train"].get("weight_decay", 1e-5),
    )

    best_dice = 0.0
    best_path = checkpoint_dir / f"{pipeline}_best.pt"
    patience = int(config["train"].get("early_stopping_patience", 15))
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, int(config["train"]["epochs"]) + 1):
        print(f"\nPipeline {pipeline} | Época {epoch}")

        train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            threshold=config["train"].get("threshold", 0.5),
            train=True,
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            criterion,
            optimizer,
            device,
            threshold=config["train"].get("threshold", 0.5),
            train=False,
        )

        row = {"epoch": epoch}
        row.update({f"train_{k}": v for k, v in train_metrics.items()})
        row.update({f"val_{k}": v for k, v in val_metrics.items()})
        history.append(row)

        pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)

        print(
            f"train_dice={train_metrics['dice']:.4f} |"
            f"train_loss={train_metrics['loss']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_dice={val_metrics['dice']:.4f} | "
            f"val_iou={val_metrics['iou']:.4f}"
        )

        if val_metrics['dice'] > best_dice:
            best_dice = val_metrics['dice']
            epochs_without_improvement = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "epoch": epoch,
                    "best_val_dice": best_dice,
                    "best_val_loss": val_metrics['loss'],
                    "best_iou": val_metrics['iou']
                },
                best_path,
            )

            save_json(
                {"best_epoch": epoch, "best_val_dice": best_dice, "best_val_loss": val_metrics['loss'],
                 "best_iou": val_metrics['iou']},
                output_dir / "best_metrics.json",
            )
            print(f"Novo melhor modelo salvo em: {best_path}")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print("Early stopping ativado.")
            break


if __name__ == "__main__":
    main()
