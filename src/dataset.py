from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

from .augmentations import build_train_augmentation
from .preprocessing import normalize_image, preprocess_image_and_mask

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
MASK_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class SegmentationSample:
    image_path: Path
    mask_path: Path


def _index_files(folder: Path, extensions: set[str]) -> Dict[str, Path]:
    if not folder.exists():
        raise FileNotFoundError(f"Pasta não encontrada: {folder}")
    files = {}
    for path in folder.iterdir():
        if path.is_file() and path.suffix.lower() in extensions:
            files[path.stem] = path
    return files


def collect_samples(images_dir: str | Path, masks_dir: str | Path) -> List[SegmentationSample]:
    images = _index_files(Path(images_dir), IMAGE_EXTENSIONS)
    masks = _index_files(Path(masks_dir), MASK_EXTENSIONS)
    common = sorted(set(images.keys()) & set(masks.keys()))
    if not common:
        raise RuntimeError(
            f"Nenhum par imagem/máscara encontrado. Imagens: {images_dir}; Máscaras: {masks_dir}"
        )
    missing_masks = sorted(set(images.keys()) - set(masks.keys()))
    if missing_masks[:5]:
        print(f"Aviso: {len(missing_masks)} imagens sem máscara correspondente. Exemplos: {missing_masks[:5]}")
    return [SegmentationSample(images[k], masks[k]) for k in common]


def split_train_val(
    samples: Sequence[SegmentationSample],
    val_size: float = 0.2,
    seed: int = 42,
    max_samples: Optional[int] = None,
) -> Tuple[List[SegmentationSample], List[SegmentationSample]]:
    samples = list(samples)
    if max_samples is not None:
        samples = samples[: int(max_samples)]
    train_samples, val_samples = train_test_split(samples, test_size=val_size, random_state=seed, shuffle=True)
    return list(train_samples), list(val_samples)


class BriscSegmentationDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[SegmentationSample],
        pipeline: str,
        image_size: int,
        preprocessing_cfg: Dict,
        augmentation_cfg: Optional[Dict] = None,
        mean: float = 0.5,
        std: float = 0.5,
        repeat_grayscale_to_rgb: bool = False,
        is_train: bool = False,
    ) -> None:
        self.samples = list(samples)
        self.pipeline = pipeline
        self.image_size = int(image_size)
        self.preprocessing_cfg = preprocessing_cfg
        self.mean = float(mean)
        self.std = float(std)
        self.repeat_grayscale_to_rgb = bool(repeat_grayscale_to_rgb)
        self.is_train = is_train
        self.augmentation = build_train_augmentation(pipeline, augmentation_cfg) if is_train else None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str]:
        sample = self.samples[idx]
        image = cv2.imread(str(sample.image_path), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(sample.mask_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise RuntimeError(f"Falha ao ler imagem: {sample.image_path}")
        if mask is None:
            raise RuntimeError(f"Falha ao ler máscara: {sample.mask_path}")

        image, mask = preprocess_image_and_mask(
            image=image,
            mask=mask,
            pipeline=self.pipeline,
            image_size=self.image_size,
            preprocessing_cfg=self.preprocessing_cfg,
        )

        if self.augmentation is not None:
            augmented = self.augmentation(image=image, mask=mask)
            image, mask = augmented["image"], augmented["mask"]

        image = normalize_image(image, self.mean, self.std)
        mask = (mask > 0.5).astype(np.float32)

        image_tensor = torch.from_numpy(image).float().unsqueeze(0)
        if self.repeat_grayscale_to_rgb:
            image_tensor = image_tensor.repeat(3, 1, 1)
        mask_tensor = torch.from_numpy(mask).float().unsqueeze(0)

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "image_path": str(sample.image_path),
            "mask_path": str(sample.mask_path),
        }
