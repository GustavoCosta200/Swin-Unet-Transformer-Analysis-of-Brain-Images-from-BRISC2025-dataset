#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Gera comparação visual entre imagem, máscara real e predições P0-P4.

Uso recomendado:
    python scripts/generate_qualitative_comparison.py --project-root . --out-dir report_figures/qualitative

O script procura:
    datasets/brisc2025/.../test/images
    datasets/brisc2025/.../test/masks
    outputs/P0/predictions
    outputs/P1/predictions
    ...
Também aceita caminhos explícitos:
    python scripts/generate_qualitative_comparison.py \
        --images-dir data/test/images \
        --masks-dir data/test/masks \
        --pred-root outputs \
        --out-dir report_figures/qualitative

Nomes de predição aceitos:
    mesmo_nome.png
    mesmo_nome_mask.png
    mesmo_nome_pred.png
    mesmo_nome_prediction.png

Importante:
    Antes de rodar este script, rode a avaliação salvando as predições.
    Exemplo esperado no projeto:
        python -m src.test --config configs/P0.yaml --checkpoint checkpoints/P0_best.pt --save-predictions outputs/P0/predictions
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


PIPELINES_DEFAULT = ["P0", "P1", "P2", "P3", "P4"]


def image_files(directory: Path) -> List[Path]:
    if not directory.exists():
        return []
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    return sorted([p for p in directory.iterdir() if p.suffix.lower() in exts])


def find_first_existing(paths: Iterable[Path]) -> Optional[Path]:
    for path in paths:
        if path.exists():
            return path
    return None


def infer_images_dir(project_root: Path) -> Optional[Path]:
    candidates = [
        project_root / "data" / "test" / "images",
        project_root / "dataset" / "test" / "images",
        project_root / "datasets" / "brisc2025" / "segmentation_task" / "test" / "images",
        project_root / "brisc2025" / "segmentation_task" / "test" / "images",
        project_root / "segmentation_task" / "test" / "images",
    ]
    candidates.extend(project_root.glob("**/test/images"))
    return find_first_existing(candidates)


def infer_masks_dir(project_root: Path) -> Optional[Path]:
    candidates = [
        project_root / "data" / "test" / "masks",
        project_root / "dataset" / "test" / "masks",
        project_root / "datasets" / "brisc2025" / "segmentation_task" / "test" / "masks",
        project_root / "brisc2025" / "segmentation_task" / "test" / "masks",
        project_root / "segmentation_task" / "test" / "masks",
    ]
    candidates.extend(project_root.glob("**/test/masks"))
    return find_first_existing(candidates)


def infer_pred_dir(project_root: Path, pred_root: Optional[Path], pipeline: str) -> Optional[Path]:
    roots = []
    if pred_root:
        roots.append(pred_root)

    roots.extend([
        project_root / "outputs",
        project_root / "output",
        project_root / "results",
        project_root / "resultados",
        project_root / "runs",
        project_root / "predictions",
    ])

    candidates: List[Path] = []
    for root in roots:
        candidates.extend([
            root / pipeline / "predictions",
            root / pipeline / "preds",
            root / pipeline / "test_predictions",
            root / f"{pipeline}_predictions",
            root / f"{pipeline}_preds",
        ])

    candidates.extend(project_root.glob(f"**/{pipeline}/predictions"))
    candidates.extend(project_root.glob(f"**/{pipeline}/preds"))
    candidates.extend(project_root.glob(f"**/{pipeline}*predictions*"))

    return find_first_existing(candidates)


def read_gray(path: Path, size: Optional[tuple[int, int]] = None) -> np.ndarray:
    img = Image.open(path).convert("L")
    if size is not None and img.size != size:
        img = img.resize(size, Image.NEAREST)
    arr = np.asarray(img).astype(np.float32)
    if arr.max() > 0:
        arr = arr / arr.max()
    return arr


def read_rgb(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img).astype(np.float32) / 255.0
    return arr


def mask_to_binary(arr: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    return (arr >= threshold).astype(np.float32)


def overlay_mask(image_rgb: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    mask = mask_to_binary(mask)
    overlay = image_rgb.copy()
    # Vermelho para máscara sem especificar colormap do matplotlib.
    overlay[..., 0] = np.where(mask > 0, (1 - alpha) * overlay[..., 0] + alpha * 1.0, overlay[..., 0])
    overlay[..., 1] = np.where(mask > 0, (1 - alpha) * overlay[..., 1], overlay[..., 1])
    overlay[..., 2] = np.where(mask > 0, (1 - alpha) * overlay[..., 2], overlay[..., 2])
    return overlay


def find_by_stem(directory: Path, stem: str) -> Optional[Path]:
    candidates = []
    for suffix in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]:
        candidates.extend([
            directory / f"{stem}{suffix}",
            directory / f"{stem}_mask{suffix}",
            directory / f"{stem}_pred{suffix}",
            directory / f"{stem}_prediction{suffix}",
            directory / f"{stem}.jpg{suffix}",
            directory / f"{stem}.png{suffix}",
        ])
    found = find_first_existing(candidates)
    if found:
        return found

    # fallback: busca por prefixo/stem aproximado
    for p in image_files(directory):
        if p.stem == stem or p.stem.startswith(stem):
            return p
    return None


def dice_score(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-7) -> float:
    pred = mask_to_binary(pred)
    gt = mask_to_binary(gt)
    inter = float((pred * gt).sum())
    return (2 * inter + eps) / (float(pred.sum() + gt.sum()) + eps)


def choose_samples(
    images: List[Path],
    masks_dir: Path,
    pred_dirs: Dict[str, Path],
    max_samples: int,
) -> List[Path]:
    scored = []
    primary_pred_dir = next(iter(pred_dirs.values()), None)

    for img_path in images:
        mask_path = find_by_stem(masks_dir, img_path.stem)
        if not mask_path:
            continue

        if primary_pred_dir:
            pred_path = find_by_stem(primary_pred_dir, img_path.stem)
            if pred_path:
                gt = read_gray(mask_path)
                pred = read_gray(pred_path, size=(gt.shape[1], gt.shape[0]))
                score = dice_score(pred, gt)
                # Ordena para pegar exemplos bons, médios e ruins.
                scored.append((score, img_path))
            else:
                scored.append((0.0, img_path))
        else:
            scored.append((0.0, img_path))

    if not scored:
        return images[:max_samples]

    scored = sorted(scored, key=lambda x: x[0])
    if len(scored) <= max_samples:
        return [p for _, p in scored]

    idxs = np.linspace(0, len(scored) - 1, max_samples).astype(int)
    return [scored[i][1] for i in idxs]


def make_grid(
    img_path: Path,
    masks_dir: Path,
    pred_dirs: Dict[str, Path],
    out_path: Path,
) -> None:
    image = read_rgb(img_path)
    size = (image.shape[1], image.shape[0])

    mask_path = find_by_stem(masks_dir, img_path.stem)
    if not mask_path:
        return

    gt = read_gray(mask_path, size=size)
    gt_overlay = overlay_mask(image, gt)

    columns = [("Imagem", image), ("Máscara real", gt), ("Real sobreposta", gt_overlay)]

    for pipeline, pred_dir in pred_dirs.items():
        pred_path = find_by_stem(pred_dir, img_path.stem)
        if not pred_path:
            continue
        pred = read_gray(pred_path, size=size)
        pred_overlay = overlay_mask(image, pred)
        columns.append((f"{pipeline} predição", pred))
        columns.append((f"{pipeline} sobreposta", pred_overlay))

    fig, axes = plt.subplots(1, len(columns), figsize=(3.0 * len(columns), 3.2))
    if len(columns) == 1:
        axes = [axes]

    for ax, (title, arr) in zip(axes, columns):
        if arr.ndim == 2:
            ax.imshow(arr, cmap="gray", vmin=0, vmax=1)
        else:
            ax.imshow(arr)
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    fig.suptitle(img_path.name, fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path("."), help="Raiz do projeto.")
    parser.add_argument("--images-dir", type=Path, default=None, help="Pasta de imagens de teste.")
    parser.add_argument("--masks-dir", type=Path, default=None, help="Pasta de máscaras reais de teste.")
    parser.add_argument("--pred-root", type=Path, default=None, help="Pasta raiz das predições por pipeline.")
    parser.add_argument("--pipelines", nargs="+", default=PIPELINES_DEFAULT, help="Pipelines a comparar.")
    parser.add_argument("--out-dir", type=Path, default=Path("report_figures/qualitative"), help="Pasta de saída.")
    parser.add_argument("--max-samples", type=int, default=6, help="Número de exemplos visuais.")
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    images_dir = args.images_dir.resolve() if args.images_dir else infer_images_dir(project_root)
    masks_dir = args.masks_dir.resolve() if args.masks_dir else infer_masks_dir(project_root)

    if not images_dir or not images_dir.exists():
        raise FileNotFoundError("Pasta de imagens de teste não encontrada. Use --images-dir.")
    if not masks_dir or not masks_dir.exists():
        raise FileNotFoundError("Pasta de máscaras de teste não encontrada. Use --masks-dir.")

    pred_root = args.pred_root.resolve() if args.pred_root else None
    pred_dirs: Dict[str, Path] = {}
    for pipeline in args.pipelines:
        d = infer_pred_dir(project_root, pred_root, pipeline)
        if d and d.exists():
            pred_dirs[pipeline] = d

    if not pred_dirs:
        raise FileNotFoundError(
            "Nenhuma pasta de predições foi encontrada. "
            "Rode a avaliação salvando as predições ou use --pred-root."
        )

    images = image_files(images_dir)
    samples = choose_samples(images, masks_dir, pred_dirs, args.max_samples)

    for i, img_path in enumerate(samples, start=1):
        make_grid(img_path, masks_dir, pred_dirs, out_dir / f"comparacao_visual_{i:02d}_{img_path.stem}.png")

    print(f"[OK] Comparações visuais salvas em: {out_dir}")
    print("[INFO] Predições encontradas:")
    for p, d in pred_dirs.items():
        print(f"  - {p}: {d}")


if __name__ == "__main__":
    main()
