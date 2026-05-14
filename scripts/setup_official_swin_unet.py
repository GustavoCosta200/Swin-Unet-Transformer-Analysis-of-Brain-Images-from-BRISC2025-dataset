from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

REPO_URL = "https://github.com/HuCaoFighting/Swin-Unet.git"


def main() -> None:
    parser = argparse.ArgumentParser(description="Baixa a implementação oficial do Swin-Unet.")
    parser.add_argument("--target", default="external/Swin-Unet", help="Pasta de destino do repositório oficial")
    args = parser.parse_args()

    target = Path(args.target)
    target.parent.mkdir(parents=True, exist_ok=True)

    if target.exists() and any(target.iterdir()):
        print(f"A pasta já existe e não está vazia: {target}")
        print("Nada foi alterado.")
        return

    subprocess.run(["git", "clone", REPO_URL, str(target)], check=True)
    print(f"Repositório oficial clonado em: {target}")


if __name__ == "__main__":
    main()
