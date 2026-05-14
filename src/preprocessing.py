from __future__ import annotations

from typing import Dict, Tuple
import cv2
import numpy as np

VALID_PIPELINES = {"P0", "P1", "P2", "P3"}


def _odd_kernel(value: int) -> int:
    value = int(value)
    if value < 3:
        value = 3
    return value if value % 2 == 1 else value + 1


def apply_gaussian(image: np.ndarray, kernel_size: int = 5, sigma: float = 0) -> np.ndarray:
    k = _odd_kernel(kernel_size)
    return cv2.GaussianBlur(image, (k, k), sigmaX=sigma)


def apply_median(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    k = _odd_kernel(kernel_size)
    return cv2.medianBlur(image, k)


def apply_clahe(image: np.ndarray, clip_limit: float = 2.0, tile_grid_size: int = 8) -> np.ndarray:
    clahe = cv2.createCLAHE(
        clipLimit=float(clip_limit),
        tileGridSize=(int(tile_grid_size), int(tile_grid_size)),
    )
    return clahe.apply(image)


def binarize_mask(mask: np.ndarray) -> np.ndarray:
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)
    return binary.astype(np.float32)


def preprocess_image_and_mask(
    image: np.ndarray,
    mask: np.ndarray,
    pipeline: str,
    image_size: int,
    preprocessing_cfg: Dict,
) -> Tuple[np.ndarray, np.ndarray]:
    """Aplica P0, P1, P2 ou P3 antes da normalização/augmentations.

    P3 recebe o mesmo pré-processamento da P2. O aumento de dados é aplicado
    posteriormente em `augmentations.py`, apenas no conjunto de treino.
    """
    if pipeline not in VALID_PIPELINES:
        raise ValueError(f"Pipeline inválido: {pipeline}. Use uma destas: {sorted(VALID_PIPELINES)}")

    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    if pipeline == "P1":
        image = apply_gaussian(
            image,
            kernel_size=preprocessing_cfg.get("gaussian_kernel", 5),
            sigma=preprocessing_cfg.get("gaussian_sigma", 0),
        )
    elif pipeline in {"P2", "P3"}:
        image = apply_median(image, kernel_size=preprocessing_cfg.get("median_kernel", 5))
        image = apply_clahe(
            image,
            clip_limit=preprocessing_cfg.get("clahe_clip_limit", 2.0),
            tile_grid_size=preprocessing_cfg.get("clahe_tile_grid_size", 8),
        )

    image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_AREA)
    mask = cv2.resize(mask, (image_size, image_size), interpolation=cv2.INTER_NEAREST)
    mask = binarize_mask(mask)
    return image, mask


def normalize_image(image: np.ndarray, mean: float = 0.5, std: float = 0.5) -> np.ndarray:
    image = image.astype(np.float32) / 255.0
    return (image - float(mean)) / float(std)
