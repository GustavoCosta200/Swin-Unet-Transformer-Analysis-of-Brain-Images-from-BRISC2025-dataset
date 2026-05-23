from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import torch


def _denormalize_image(
    image: torch.Tensor,
    mean: float = 0.5,
    std: float = 0.5,
) -> torch.Tensor:
    """
    Desfaz a normalização aplicada antes de enviar a imagem ao modelo.

    No seu default.yaml, a normalização está configurada como:
    normalize_mean: 0.5
    normalize_std: 0.5

    Assim, uma imagem normalizada para aproximadamente [-1, 1]
    volta para [0, 1].
    """
    image = image.detach().cpu().float()
    image = (image * std) + mean
    image = image.clamp(0.0, 1.0)
    return image


def _image_tensor_to_numpy(image: torch.Tensor):
    """
    Converte imagem em tensor para formato compatível com matplotlib.

    Entrada esperada:
    - C x H x W

    Saída:
    - H x W para imagem com 1 canal
    - H x W x C para imagem com 3 canais
    """
    if image.ndim == 2:
        return image.numpy()

    if image.ndim != 3:
        raise ValueError(f"Formato inesperado para imagem: {tuple(image.shape)}")

    if image.shape[0] == 1:
        return image.squeeze(0).numpy()

    return image.permute(1, 2, 0).numpy()


def _mask_tensor_to_numpy(mask: torch.Tensor):
    """
    Converte máscara em tensor para formato H x W.
    """
    mask = mask.detach().cpu().float()

    if mask.ndim == 3:
        mask = mask.squeeze(0)

    if mask.ndim != 2:
        raise ValueError(f"Formato inesperado para máscara: {tuple(mask.shape)}")

    return mask.numpy()


def save_pipeline_debug_sample(
    dataset,
    cfg: Dict[str, Any],
    split_name: str = "train",
    sample_index: int = 0,
) -> None:
    """
    Salva uma amostra visual exatamente como ela sai do Dataset
    e entra no DataLoader/modelo.

    Compatível com o train.py atual, no qual o Dataset retorna:
    {
        "image": tensor,
        "mask": tensor
    }

    A imagem é salva em:
    outputs/<pipeline>/debug_samples/
    """

    experiment_cfg = cfg.get("experiment", {})
    image_cfg = cfg.get("image", {})
    paths_cfg = cfg.get("paths", {})

    pipeline = experiment_cfg.get("pipeline", "P0")
    output_base = paths_cfg.get("output_dir", "./outputs")

    mean = image_cfg.get("normalize_mean", 0.5)
    std = image_cfg.get("normalize_std", 0.5)

    output_dir = Path(output_base) / pipeline / "debug_samples"
    output_dir.mkdir(parents=True, exist_ok=True)

    sample = dataset[sample_index]

    if not isinstance(sample, dict):
        raise TypeError(
            "O dataset deveria retornar um dicionário com as chaves "
            "'image' e 'mask', pois o train.py usa batch['image'] e batch['mask']."
        )

    if "image" not in sample or "mask" not in sample:
        raise KeyError(
            f"A amostra do dataset possui as chaves {list(sample.keys())}, "
            "mas eram esperadas as chaves 'image' e 'mask'."
        )

    image_tensor = sample["image"]
    mask_tensor = sample["mask"]

    image_denorm = _denormalize_image(image_tensor, mean=mean, std=std)

    image_np = _image_tensor_to_numpy(image_denorm)
    mask_np = _mask_tensor_to_numpy(mask_tensor)

    save_path = output_dir / f"{split_name}_sample_after_pipeline_{pipeline}.png"

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    if image_np.ndim == 2:
        plt.imshow(image_np, cmap="gray")
    else:
        plt.imshow(image_np)
    plt.title(f"Imagem após {pipeline}")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(mask_np, cmap="gray")
    plt.title("Máscara")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    if image_np.ndim == 2:
        plt.imshow(image_np, cmap="gray")
    else:
        plt.imshow(image_np)
    plt.imshow(mask_np, cmap="Reds", alpha=0.35)
    plt.title("Imagem + Máscara")
    plt.axis("off")

    plt.suptitle(
        f"Amostra enviada ao modelo | Pipeline {pipeline} | Split: {split_name}",
        fontsize=12,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"[DEBUG] Amostra da pipeline salva em: {save_path}")