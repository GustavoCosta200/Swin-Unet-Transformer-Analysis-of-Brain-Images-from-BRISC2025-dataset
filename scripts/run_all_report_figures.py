#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Executa a geração de gráficos quantitativos e, se houver predições salvas,
também gera comparação visual.

Uso:
    python scripts/run_all_report_figures.py --project-root . --out-dir report_figures
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]) -> int:
    print("[RUN]", " ".join(cmd))
    result = subprocess.run(cmd)
    return result.returncode


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--out-dir", type=Path, default=Path("report_figures"))
    parser.add_argument("--summary-csv", type=Path, default=None)
    parser.add_argument("--pipelines", nargs="+", default=["P0", "P1", "P2", "P3", "P4"])
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent

    graphs_cmd = [
        sys.executable,
        str(script_dir / "generate_report_graphs.py"),
        "--project-root",
        str(args.project_root),
        "--out-dir",
        str(args.out_dir),
        "--pipelines",
        *args.pipelines,
    ]
    if args.summary_csv:
        graphs_cmd.extend(["--summary-csv", str(args.summary_csv)])

    code = run(graphs_cmd)
    if code != 0:
        sys.exit(code)

    # Comparação visual é opcional, pois depende de predições salvas.
    qualitative_cmd = [
        sys.executable,
        str(script_dir / "generate_qualitative_comparison.py"),
        "--project-root",
        str(args.project_root),
        "--out-dir",
        str(args.out_dir / "qualitative"),
        "--pipelines",
        *args.pipelines,
    ]

    code = run(qualitative_cmd)
    if code != 0:
        print("[AVISO] Gráficos quantitativos foram gerados, mas a comparação visual não foi criada.")
        print("[AVISO] Verifique se existem pastas de predição como outputs/P0/predictions.")
        sys.exit(0)

    print("[OK] Todos os gráficos foram gerados.")


if __name__ == "__main__":
    main()
