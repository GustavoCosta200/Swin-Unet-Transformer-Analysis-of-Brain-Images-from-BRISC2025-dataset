from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn as nn


def _as_path(value: str | Path) -> Path:
    return Path(value).expanduser().resolve()


def _require_official_repo(repo_dir: str | Path) -> Path:
    repo_path = _as_path(repo_dir)
    required = [
        repo_path / "config.py",
        repo_path / "networks" / "vision_transformer.py",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "A implementação oficial do Swin-Unet não foi encontrada. "
            f"Arquivos ausentes: {missing}. "
            "Execute `python scripts/setup_official_swin_unet.py` ou coloque o repositório "
            "HuCaoFighting/Swin-Unet em `external/Swin-Unet`."
        )
    return repo_path


def _import_official_modules(repo_path: Path):
    repo_str = str(repo_path)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)

    # Os nomes seguem a estrutura do repositório oficial HuCaoFighting/Swin-Unet.
    official_config_module = importlib.import_module("config")
    official_network_module = importlib.import_module("networks.vision_transformer")
    return official_config_module, official_network_module


def _build_official_args(model_cfg: Dict[str, Any], image_cfg: Dict[str, Any]) -> argparse.Namespace:
    """Cria os argumentos esperados pela função `get_config` do repositório oficial.

    O repositório oficial usa `argparse.Namespace` para carregar o YAML de configuração.
    Estes campos espelham o formato utilizado pelo projeto original.
    """
    official_config_path = model_cfg.get("official_config")
    if not official_config_path:
        raise ValueError("Defina `model.official_config` no YAML principal.")

    return argparse.Namespace(
        cfg=str(official_config_path),
        opts=None,
        batch_size=None,
        zip=False,
        cache_mode="part",
        resume=None,
        accumulation_steps=None,
        use_checkpoint=False,
        amp_opt_level="O1",
        tag=None,
        eval=False,
        throughput=False,
        img_size=int(image_cfg["size"]),
        num_classes=int(image_cfg["num_classes"]),
    )


def _override_if_present(obj: Any, dotted_path: str, value: Any) -> None:
    current = obj
    parts = dotted_path.split(".")
    for part in parts[:-1]:
        if not hasattr(current, part):
            return
        current = getattr(current, part)
    leaf = parts[-1]
    if hasattr(current, leaf):
        try:
            setattr(current, leaf, value)
        except Exception:
            pass


def build_official_swin_unet(model_cfg: Dict[str, Any], image_cfg: Dict[str, Any]) -> nn.Module:
    """Instancia o Swin-Unet oficial de Cao et al. usando o repositório original.

    Por padrão, o projeto usa imagens em escala de cinza replicadas em 3 canais, pois a
    configuração oficial e os pesos pré-treinados do Swin Transformer foram definidos para
    entrada RGB. A replicação é controlada em `image.repeat_grayscale_to_rgb`.
    """
    repo_path = _require_official_repo(model_cfg.get("official_repo_dir", "external/Swin-Unet"))
    official_config_module, official_network_module = _import_official_modules(repo_path)

    args = _build_official_args(model_cfg, image_cfg)
    config = official_config_module.get_config(args)

    # Mantém o modelo oficial e altera apenas parâmetros de compatibilidade experimental.
    _override_if_present(config, "DATA.IMG_SIZE", int(image_cfg["size"]))
    _override_if_present(config, "MODEL.NUM_CLASSES", int(image_cfg["num_classes"]))

    model = official_network_module.SwinUnet(
        config,
        img_size=int(image_cfg["size"]),
        num_classes=int(image_cfg["num_classes"]),
    )

    pretrained = model_cfg.get("pretrained_path")
    if pretrained:
        pretrained_path = _as_path(pretrained)
        if pretrained_path.exists() and hasattr(model, "load_from"):
            model.load_from(config)
        elif pretrained_path.exists():
            state = torch.load(pretrained_path, map_location="cpu")
            state_dict = state.get("model", state.get("state_dict", state))
            model.load_state_dict(state_dict, strict=False)
        else:
            raise FileNotFoundError(f"Pesos pré-treinados não encontrados: {pretrained_path}")

    return model
