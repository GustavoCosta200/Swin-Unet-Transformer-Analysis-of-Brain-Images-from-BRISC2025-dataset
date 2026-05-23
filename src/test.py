from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import load_config, resolve_dataset_paths
from .dataset import (
    BriscSegmentationDataset,
    collect_samples,
    balance_samples_by_class,
    print_class_distribution,
)
from .metrics import MetricAverager, segmentation_metrics
from .models import build_official_swin_unet
from .utils import ensure_dir, get_device, save_json, set_seed


def get_configured_metrics(config: Dict) -> List[str]:
    """
    Lê as métricas definidas no YAML.

    Aceita qualquer uma destas formas:

    evaluation:
      metrics: [dice, iou, pixel_accuracy, precision, recall]

    ou:

    metrics: [dice, iou, pixel_accuracy]

    Caso não exista no YAML, usa as métricas já retornadas por src.metrics.segmentation_metrics.
    """

    if "evaluation" in config and "metrics" in config["evaluation"]:
        metrics = config["evaluation"]["metrics"]
    elif "metrics" in config:
        metrics = config["metrics"]
    else:
        metrics = ["dice", "iou", "pixel_accuracy", "precision", "recall"]

    return [str(metric).lower() for metric in metrics]


def get_threshold(config: Dict) -> float:
    """
    Usa o threshold do YAML.

    Prioridade:
    1. evaluation.threshold
    2. train.threshold
    3. 0.5
    """

    if "evaluation" in config and "threshold" in config["evaluation"]:
        return float(config["evaluation"]["threshold"])

    return float(config.get("train", {}).get("threshold", 0.5))


def filter_metrics(metrics: Dict[str, float], selected_metrics: List[str]) -> Dict[str, float]:
    """
    Mantém apenas as métricas selecionadas no YAML.
    """

    filtered = {}

    for metric in selected_metrics:
        if metric not in metrics:
            raise KeyError(
                f"A métrica '{metric}' foi solicitada no YAML, mas não é retornada por "
                f"segmentation_metrics. Métricas disponíveis: {list(metrics.keys())}"
            )

        filtered[metric] = metrics[metric]

    return filtered


def load_checkpoint(model: torch.nn.Module, checkpoint_path: str | Path, device: torch.device):
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint não encontrado: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        raise RuntimeError(f"Formato de checkpoint não suportado: {checkpoint_path}")

    clean_state_dict = {}

    for key, value in state_dict.items():
        clean_key = key.replace("module.", "")
        clean_state_dict[clean_key] = value

    missing, unexpected = model.load_state_dict(clean_state_dict, strict=False)

    if missing:
        print(f"Aviso: chaves ausentes no checkpoint: {missing[:10]}")
    if unexpected:
        print(f"Aviso: chaves inesperadas no checkpoint: {unexpected[:10]}")

    return checkpoint


def save_prediction_grid(
    image: torch.Tensor,
    mask: torch.Tensor,
    pred: torch.Tensor,
    out_path: Path,
) -> None:
    """
    Salva uma imagem com:
    imagem original | máscara real | máscara predita | overlay real | overlay predito
    """

    image_np = image.detach().cpu().numpy()

    if image_np.ndim == 3 and image_np.shape[0] == 3:
        image_np = image_np[0]
    else:
        image_np = np.squeeze(image_np)

    # Desnormaliza considerando mean=0.5 e std=0.5,
    # que é o padrão usado no YAML atual.
    image_np = ((image_np * 0.5 + 0.5) * 255.0).clip(0, 255).astype(np.uint8)

    mask_np = mask.squeeze().detach().cpu().numpy()
    pred_np = pred.squeeze().detach().cpu().numpy()

    mask_uint8 = (mask_np * 255).astype(np.uint8)
    pred_uint8 = (pred_np * 255).astype(np.uint8)

    image_rgb = cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR)

    gt_overlay = image_rgb.copy()
    pred_overlay = image_rgb.copy()

    # Verde: ground truth
    gt_overlay[mask_np > 0.5] = [0, 255, 0]

    # Vermelho: predição
    pred_overlay[pred_np > 0.5] = [0, 0, 255]

    grid = np.concatenate(
        [
            image_rgb,
            cv2.cvtColor(mask_uint8, cv2.COLOR_GRAY2BGR),
            cv2.cvtColor(pred_uint8, cv2.COLOR_GRAY2BGR),
            gt_overlay,
            pred_overlay,
        ],
        axis=1,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), grid)


def build_test_loader(config: Dict) -> DataLoader:
    paths = resolve_dataset_paths(config)

    experiment_cfg = config.get("experiment", {})
    image_cfg = config.get("image", {})

    class_aliases = experiment_cfg.get("class_aliases")
    balance_classes = bool(experiment_cfg.get("balance_classes", False))
    class_names = experiment_cfg.get("balance_class_names")
    max_test_samples = experiment_cfg.get("max_test_samples")

    samples = collect_samples(
        paths["test_images"],
        paths["test_masks"],
        class_aliases=class_aliases,
    )

    print_class_distribution(samples, "Distribuição original do teste")

    if balance_classes:
        samples = balance_samples_by_class(
            samples=samples,
            class_names=class_names,
            max_samples=max_test_samples,
            seed=config.get("seed", 42),
            drop_unknown=True,
        )

        print_class_distribution(samples, "Distribuição final do teste balanceado")

    elif max_test_samples is not None:
        samples = samples[: int(max_test_samples)]
        print_class_distribution(samples, "Distribuição final do teste limitado")

    dataset = BriscSegmentationDataset(
        samples=samples,
        pipeline=experiment_cfg["pipeline"],
        image_size=image_cfg["size"],
        preprocessing_cfg=config.get("preprocessing", {}),
        augmentation_cfg=config.get("augmentation", {}),
        mean=image_cfg.get("normalize_mean", 0.5),
        std=image_cfg.get("normalize_std", 0.5),
        repeat_grayscale_to_rgb=image_cfg.get("repeat_grayscale_to_rgb", False),
        is_train=False,
    )

    loader = DataLoader(
        dataset,
        batch_size=config["train"]["batch_size"],
        shuffle=False,
        num_workers=experiment_cfg.get("num_workers", 0),
        pin_memory=torch.cuda.is_available(),
    )

    return loader


@torch.no_grad()
def evaluate_on_test(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float,
    selected_metrics: List[str],
    output_dir: Path,
    save_images: int,
) -> Dict[str, float]:
    model.eval()

    avg = MetricAverager()
    rows_batch = []
    rows_image = []

    prediction_dir = ensure_dir(output_dir / "predictions")

    saved_images = 0
    total_inference_time = 0.0
    total_samples = 0

    for batch_idx, batch in enumerate(tqdm(loader, desc="Avaliando teste")):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        if device.type == "cuda":
            torch.cuda.synchronize()

        start_time = time.perf_counter()
        logits = model(images)

        if device.type == "cuda":
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - start_time
        total_inference_time += elapsed
        total_samples += images.size(0)

        if isinstance(logits, (tuple, list)):
            logits = logits[0]

        metrics_all = segmentation_metrics(logits, masks, threshold)
        metrics_selected = filter_metrics(metrics_all, selected_metrics)

        avg.update(metrics_selected, n=images.size(0))

        rows_batch.append(
            {
                "batch": batch_idx,
                "num_samples": images.size(0),
                "inference_time_seconds": elapsed,
                **metrics_selected,
            }
        )

        probs = torch.sigmoid(logits)
        preds = (probs >= threshold).float()

        image_paths = batch.get("image_path", [""] * images.size(0))
        mask_paths = batch.get("mask_path", [""] * images.size(0))
        tumor_classes = batch.get("tumor_class", ["unknown"] * images.size(0))

        for i in range(images.size(0)):
            single_metrics_all = segmentation_metrics(
                logits[i : i + 1],
                masks[i : i + 1],
                threshold,
            )
            single_metrics = filter_metrics(single_metrics_all, selected_metrics)

            rows_image.append(
                {
                    "image_path": image_paths[i],
                    "mask_path": mask_paths[i],
                    "tumor_class": tumor_classes[i],
                    **single_metrics,
                }
            )

            if saved_images < save_images:
                save_prediction_grid(
                    images[i],
                    masks[i],
                    preds[i],
                    prediction_dir / f"sample_{saved_images:03d}.png",
                )
                saved_images += 1

    final_metrics = avg.compute()

    final_metrics["num_samples"] = total_samples
    final_metrics["threshold"] = threshold
    final_metrics["mean_inference_time_seconds"] = (
        total_inference_time / max(total_samples, 1)
    )
    final_metrics["total_inference_time_seconds"] = total_inference_time

    pd.DataFrame(rows_batch).to_csv(output_dir / "test_batches.csv", index=False)
    pd.DataFrame(rows_image).to_csv(output_dir / "test_per_image.csv", index=False)
    save_json(final_metrics, output_dir / "test_metrics.json")

    return final_metrics


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Caminho do arquivo YAML de configuração.",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Caminho do checkpoint treinado. Ex: checkpoints/P0_best.pt",
    )

    parser.add_argument(
        "--save-images",
        type=int,
        default=12,
        help="Quantidade de exemplos qualitativos a salvar.",
    )

    args = parser.parse_args()

    config = load_config(args.config)

    set_seed(config.get("seed", 42))

    device = get_device(config.get("device", "cuda"))

    pipeline = config["experiment"]["pipeline"]
    selected_metrics = get_configured_metrics(config)
    threshold = get_threshold(config)

    print("=" * 80)
    print(f"Pipeline: {pipeline}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Dispositivo: {device}")
    print(f"Threshold: {threshold}")
    print(f"Métricas avaliadas: {selected_metrics}")
    print("=" * 80)

    output_dir = ensure_dir(
        Path(config["paths"]["output_dir"]) / pipeline / "test"
    )

    test_loader = build_test_loader(config)

    model = build_official_swin_unet(
        config["model"],
        config["image"],
    ).to(device)

    checkpoint = load_checkpoint(model, args.checkpoint, device)

    if isinstance(checkpoint, dict):
        if "epoch" in checkpoint:
            print(f"Checkpoint salvo na época: {checkpoint['epoch']}")
        if "best_val_loss" in checkpoint:
            print(f"Best val_loss: {checkpoint['best_val_loss']}")
        if "best_val_dice" in checkpoint:
            print(f"Best val_dice: {checkpoint['best_val_dice']}")

    final_metrics = evaluate_on_test(
        model=model,
        loader=test_loader,
        device=device,
        threshold=threshold,
        selected_metrics=selected_metrics,
        output_dir=output_dir,
        save_images=args.save_images,
    )

    print("\nMétricas finais no teste:")
    for key, value in final_metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.6f}")
        else:
            print(f"{key}: {value}")

    print(f"\nArquivos salvos em: {output_dir}")


if __name__ == "__main__":
    main()