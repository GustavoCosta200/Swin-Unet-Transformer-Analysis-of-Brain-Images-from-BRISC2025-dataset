from __future__ import annotations

from typing import Dict, Optional

import albumentations as A


AUGMENTATION_PIPELINES = {"P3", "P4"}


def build_train_augmentation(pipeline: str, cfg: Optional[Dict] = None) -> Optional[A.Compose]:
    """Retorna augmentation para P3 e P4.

    P3: pré-processamento P2 + augmentation.
    P4: baseline P0 + augmentation, teste de ablação para isolar o efeito
    do aumento de dados sem filtros clássicos e sem CLAHE.
    """
    pipeline = str(pipeline).upper()
    if pipeline not in AUGMENTATION_PIPELINES:
        return None

    cfg = cfg or {}
    return A.Compose(
        [
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
        ]
    )
