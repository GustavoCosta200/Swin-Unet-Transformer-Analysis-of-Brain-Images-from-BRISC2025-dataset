from __future__ import annotations

from typing import Dict, Optional
import albumentations as A


def build_train_augmentation(pipeline: str, cfg: Optional[Dict] = None) -> Optional[A.Compose]:
    """Retorna augmentation apenas para P3.

    As transformações seguem a metodologia: rotações leves, espelhamento,
    pequenas translações/escalas, brilho/contraste moderados e ruído gaussiano.
    """
    if pipeline != "P3":
        return None

    cfg = cfg or {}
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=cfg.get("shift_limit", 0.05),
            scale_limit=cfg.get("scale_limit", 0.10),
            rotate_limit=cfg.get("rotate_limit", 10),
            border_mode=0,
            value=0,
            mask_value=0,
            p=0.7,
        ),
        A.RandomBrightnessContrast(
            brightness_limit=cfg.get("brightness_limit", 0.10),
            contrast_limit=cfg.get("contrast_limit", 0.10),
            p=0.4,
        ),
        A.GaussNoise(var_limit=tuple(cfg.get("gaussian_noise_var_limit", [5.0, 20.0])), p=0.2),
    ])
