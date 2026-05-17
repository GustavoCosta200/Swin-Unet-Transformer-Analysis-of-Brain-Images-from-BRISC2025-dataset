from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from .augmentations import build_train_augmentation
from .preprocessing import normalize_image, preprocess_image_and_mask


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
MASK_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class SegmentationSample:
    image_path: Path
    mask_path: Path
    tumor_class: Optional[str] = None


def _index_files(folder: Path, extensions: set[str]) -> Dict[str, Path]:
    if not folder.exists():
        raise FileNotFoundError(f"Pasta não encontrada: {folder}")

    files: Dict[str, Path] = {}
    for path in folder.iterdir():
        if path.is_file() and path.suffix.lower() in extensions:
            files[path.stem] = path
    return files


def _default_class_aliases() -> Dict[str, List[str]]:
    return {
        "glioma": ["glioma", "gl"],
        "meningioma": ["meningioma", "men", "me"],
        "pituitary": [
            "pituitary",
            "pituitario",
            "pituitária",
            "pituitaria",
            "pituitary_tumor",
            "pit",
            "pi",
        ],
    }


def _normalise_aliases(class_aliases: Optional[Dict]) -> Dict[str, List[str]]:
    aliases = _default_class_aliases()

    if class_aliases:
        for class_name, values in class_aliases.items():
            if values is None:
                aliases[str(class_name)] = []
            elif isinstance(values, str):
                aliases[str(class_name)] = [values]
            else:
                aliases[str(class_name)] = [str(v) for v in values]

    return {
        str(class_name).lower(): [str(alias).lower() for alias in values]
        for class_name, values in aliases.items()
    }


def infer_tumor_class(
    file_stem: str,
    class_aliases: Optional[Dict] = None,
) -> Optional[str]:
    """Infere a classe do tumor a partir do nome do arquivo.

    A base BRISC costuma codificar metadados no nome do arquivo, por exemplo:
    brisc2025_train_00015_gl_ax_t1.jpg -> classe "glioma".

    O método usa tokens separados por "_" e também verifica se algum alias aparece
    como substring. Isso permite funcionar tanto com nomes abreviados quanto com
    nomes por extenso.
    """

    stem = file_stem.lower()
    tokens = {token for token in stem.replace("-", "_").split("_") if token}
    aliases = _normalise_aliases(class_aliases)

    for class_name, class_alias_list in aliases.items():
        candidates = {class_name, *class_alias_list}

        if tokens & candidates:
            return class_name

        for alias in candidates:
            if alias and alias in stem:
                return class_name

    return None


def print_class_distribution(
    samples: Sequence[SegmentationSample],
    title: str = "Distribuição por classe",
) -> None:
    counter = Counter(sample.tumor_class or "unknown" for sample in samples)
    total = sum(counter.values())
    print(f"{title}: total={total}")
    for class_name, count in sorted(counter.items()):
        print(f"  - {class_name}: {count}")


def balance_samples_by_class(
    samples: Sequence[SegmentationSample],
    class_names: Optional[Sequence[str]] = None,
    max_samples: Optional[int] = None,
    seed: int = 42,
    drop_unknown: bool = True,
) -> List[SegmentationSample]:
    """Seleciona a mesma quantidade de amostras para cada classe.

    Se max_samples for informado, o total final também será balanceado entre as
    classes disponíveis. Ex.: max_samples=300 e 3 classes -> 100 imagens por classe.
    """

    rng = np.random.default_rng(seed)
    class_names_set = {str(c).lower() for c in class_names} if class_names else None

    grouped: Dict[str, List[SegmentationSample]] = defaultdict(list)
    for sample in samples:
        sample_class = (sample.tumor_class or "unknown").lower()

        if sample_class == "unknown" and drop_unknown:
            continue

        if class_names_set is not None and sample_class not in class_names_set:
            continue

        grouped[sample_class].append(sample)

    if not grouped:
        raise RuntimeError(
            "Nenhuma amostra com classe tumoral válida foi encontrada. "
            "Verifique se os nomes dos arquivos contêm aliases como gl, men/me ou pit/pi, "
            "ou ajuste experiment.class_aliases no arquivo de configuração."
        )

    min_per_class = min(len(items) for items in grouped.values())

    if max_samples is not None:
        max_samples = int(max_samples)
        if max_samples <= 0:
            raise ValueError("max_samples deve ser maior que zero.")
        min_per_class = min(min_per_class, max(1, max_samples // len(grouped)))

    balanced: List[SegmentationSample] = []
    for class_name in sorted(grouped):
        items = list(grouped[class_name])
        indices = rng.permutation(len(items))[:min_per_class]
        balanced.extend(items[int(i)] for i in indices)

    rng.shuffle(balanced)
    return balanced


def collect_samples(
    images_dir: str | Path,
    masks_dir: str | Path,
    class_aliases: Optional[Dict] = None,
) -> List[SegmentationSample]:
    images = _index_files(Path(images_dir), IMAGE_EXTENSIONS)
    masks = _index_files(Path(masks_dir), MASK_EXTENSIONS)

    common = sorted(set(images.keys()) & set(masks.keys()))
    if not common:
        raise RuntimeError(
            f"Nenhum par imagem/máscara encontrado.\n"
            f"Imagens: {images_dir}; Máscaras: {masks_dir}"
        )

    missing_masks = sorted(set(images.keys()) - set(masks.keys()))
    if missing_masks[:5]:
        print(
            f"Aviso: {len(missing_masks)} imagens sem máscara correspondente.\n"
            f"Exemplos: {missing_masks[:5]}"
        )

    return [
        SegmentationSample(
            image_path=images[k],
            mask_path=masks[k],
            tumor_class=infer_tumor_class(k, class_aliases=class_aliases),
        )
        for k in common
    ]


def split_train_val(
    samples: Sequence[SegmentationSample],
    val_size: float = 0.2,
    seed: int = 42,
    max_samples: Optional[int] = None,
    balance_classes: bool = False,
    class_names: Optional[Sequence[str]] = None,
) -> Tuple[List[SegmentationSample], List[SegmentationSample]]:
    samples = list(samples)

    if balance_classes:
        samples = balance_samples_by_class(
            samples,
            class_names=class_names,
            max_samples=max_samples,
            seed=seed,
        )
        print_class_distribution(samples, "Distribuição após balanceamento")
        stratify = [sample.tumor_class for sample in samples]
    else:
        if max_samples is not None:
            samples = samples[: int(max_samples)]
        stratify = None

    train_samples, val_samples = train_test_split(
        samples,
        test_size=val_size,
        random_state=seed,
        shuffle=True,
        stratify=stratify,
    )

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
        self.augmentation = (
            build_train_augmentation(pipeline, augmentation_cfg) if is_train else None
        )

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
            "tumor_class": sample.tumor_class or "unknown",
        }
