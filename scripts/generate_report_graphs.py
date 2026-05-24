#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Gera gráficos e tabela final de ablação para as pipelines P0-P4.

Uso recomendado, a partir da raiz do projeto:
    python scripts/generate_report_graphs.py --project-root . --out-dir report_figures

O script procura automaticamente arquivos de métricas nos formatos mais comuns:
    outputs/P0/history.csv
    outputs/P0/test_metrics.json
    results/P0/history.csv
    results/P0/test_metrics.json
    logs/P0/history.csv
    checkpoints/P0_best_metrics.json
    P0_best_metrics.json

Também aceita um CSV manual:
    python scripts/generate_report_graphs.py --summary-csv report_metrics.csv

CSV manual esperado:
    pipeline,split,loss,dice,iou,pixel_accuracy,epoch
    P0,val,0.13,0.77,0.68,0.99,138
    P0,test,0.14,0.76,0.67,0.99,
    ...

Arquivos gerados:
    fig_bar_test_metrics.png
    fig_bar_val_metrics.png
    fig_loss_test.png
    fig_dice_gain_vs_p0.png
    fig_iou_gain_vs_p0.png
    fig_best_epoch.png
    fig_boxplot_dice_per_image.png, se houver métricas por imagem
    tabela_ablação.csv
    tabela_ablação.tex
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


PIPELINES_DEFAULT = ["P0", "P1", "P2", "P3", "P4"]


PIPELINE_DESCRIPTIONS = {
    "P0": "Baseline",
    "P1": "Gaussiano + Mediana",
    "P2": "P1 + CLAHE",
    "P3": "Pré-processamento + aumento de dados",
    "P4": "Aumento de dados sem pré-processamento",
}


METRIC_ALIASES = {
    "loss": ["loss", "test_loss", "val_loss", "valid_loss", "validation_loss", "dice_loss"],
    "dice": ["dice", "dice_score", "test_dice", "val_dice", "valid_dice", "dice_coefficient", "mean_dice"],
    "iou": ["iou", "jaccard", "jaccard_index", "test_iou", "val_iou", "valid_iou", "mean_iou"],
    "pixel_accuracy": [
        "pixel_accuracy", "pixel_acc", "accuracy", "acc",
        "test_pixel_accuracy", "val_pixel_accuracy", "valid_pixel_accuracy"
    ],
    "epoch": ["epoch", "best_epoch"],
}


def normalize_key(key: str) -> str:
    return key.strip().lower().replace(" ", "_").replace("-", "_")


def get_metric_value(data: Dict[str, Any], metric: str) -> Optional[float]:
    normalized = {normalize_key(k): v for k, v in data.items()}
    for alias in METRIC_ALIASES[metric]:
        if alias in normalized:
            value = normalized[alias]
            if value is None or value == "":
                return None
            try:
                return float(value)
            except (TypeError, ValueError):
                return None
    return None


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def candidate_dirs(project_root: Path, pipeline: str) -> List[Path]:
    return [
        project_root / "outputs" / pipeline,
        project_root / "output" / pipeline,
        project_root / "results" / pipeline,
        project_root / "resultados" / pipeline,
        project_root / "runs" / pipeline,
        project_root / "logs" / pipeline,
        project_root / "experiments" / pipeline,
        project_root / "checkpoints" / pipeline,
        project_root / pipeline,
    ]


def find_first_existing(paths: Iterable[Path]) -> Optional[Path]:
    for path in paths:
        if path.exists():
            return path
    return None


def find_history_file(project_root: Path, pipeline: str) -> Optional[Path]:
    names = [
        "history.csv", "metrics.csv", "train_history.csv", "training_history.csv",
        "log.csv", "logs.csv", "metrics_history.csv"
    ]
    candidates: List[Path] = []
    for d in candidate_dirs(project_root, pipeline):
        candidates.extend([d / name for name in names])
    candidates.extend(project_root.glob(f"**/{pipeline}*history*.csv"))
    candidates.extend(project_root.glob(f"**/{pipeline}*metrics*.csv"))
    return find_first_existing(candidates)


def find_metrics_json(project_root: Path, pipeline: str, split: str) -> Optional[Path]:
    names = {
        "test": [
            "test_metrics.json", "metrics_test.json", "test_results.json",
            f"{pipeline}_test_metrics.json", f"{pipeline}_metrics_test.json",
        ],
        "val": [
            "best_metrics.json", "val_metrics.json", "valid_metrics.json",
            "validation_metrics.json", f"{pipeline}_best_metrics.json",
            f"{pipeline}_val_metrics.json",
        ],
    }[split]

    candidates: List[Path] = []
    for d in candidate_dirs(project_root, pipeline):
        candidates.extend([d / name for name in names])

    if split == "test":
        candidates.extend(project_root.glob(f"**/{pipeline}*test*metrics*.json"))
        candidates.extend(project_root.glob(f"**/{pipeline}*test*results*.json"))
    else:
        candidates.extend(project_root.glob(f"**/{pipeline}*best*metrics*.json"))
        candidates.extend(project_root.glob(f"**/{pipeline}*val*metrics*.json"))

    return find_first_existing(candidates)


def load_best_from_history(path: Path, pipeline: str) -> Optional[Dict[str, Any]]:
    df = pd.read_csv(path)
    df.columns = [normalize_key(c) for c in df.columns]

    val_dice_col = next((c for c in ["val_dice", "valid_dice", "validation_dice", "dice_val"] if c in df.columns), None)
    val_loss_col = next((c for c in ["val_loss", "valid_loss", "validation_loss", "loss_val"] if c in df.columns), None)

    if val_dice_col:
        best_idx = df[val_dice_col].astype(float).idxmax()
    elif val_loss_col:
        best_idx = df[val_loss_col].astype(float).idxmin()
    else:
        return None

    row = df.loc[best_idx].to_dict()
    result = {
        "pipeline": pipeline,
        "split": "val",
        "source_file": str(path),
        "epoch": row.get("epoch", best_idx + 1),
        "loss": row.get(val_loss_col) if val_loss_col else None,
        "dice": row.get(val_dice_col) if val_dice_col else None,
        "iou": row.get(next((c for c in ["val_iou", "valid_iou", "validation_iou", "iou_val"] if c in df.columns), ""), None),
        "pixel_accuracy": row.get(next((c for c in ["val_pixel_accuracy", "valid_pixel_accuracy", "val_pixel_acc", "valid_pixel_acc"] if c in df.columns), ""), None),
    }
    return result


def load_split_metrics(project_root: Path, pipeline: str, split: str) -> Optional[Dict[str, Any]]:
    path = find_metrics_json(project_root, pipeline, split)
    if not path:
        return None

    data = read_json(path)
    result = {
        "pipeline": pipeline,
        "split": split,
        "source_file": str(path),
        "loss": get_metric_value(data, "loss"),
        "dice": get_metric_value(data, "dice"),
        "iou": get_metric_value(data, "iou"),
        "pixel_accuracy": get_metric_value(data, "pixel_accuracy"),
        "epoch": get_metric_value(data, "epoch"),
    }
    return result


def collect_summary(project_root: Path, pipelines: List[str]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for pipeline in pipelines:
        hist = find_history_file(project_root, pipeline)
        if hist:
            best = load_best_from_history(hist, pipeline)
            if best:
                rows.append(best)

        val_json = load_split_metrics(project_root, pipeline, "val")
        if val_json:
            # Se já veio do history, o JSON complementa/atualiza métricas ausentes.
            existing = next((r for r in rows if r["pipeline"] == pipeline and r["split"] == "val"), None)
            if existing:
                for k, v in val_json.items():
                    if existing.get(k) in [None, ""] and v not in [None, ""]:
                        existing[k] = v
            else:
                rows.append(val_json)

        test_json = load_split_metrics(project_root, pipeline, "test")
        if test_json:
            rows.append(test_json)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    for col in ["loss", "dice", "iou", "pixel_accuracy", "epoch"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["description"] = df["pipeline"].map(PIPELINE_DESCRIPTIONS).fillna("")
    return df.sort_values(["pipeline", "split"]).reset_index(drop=True)


def plot_grouped_bar(
    df: pd.DataFrame,
    split: str,
    metrics: List[str],
    title: str,
    output_path: Path,
    ylabel: str = "Valor da métrica",
    ylim: Optional[Tuple[float, float]] = None,
) -> None:
    sub = df[df["split"] == split].copy()
    sub = sub[sub["pipeline"].notna()]
    sub = sub.sort_values("pipeline")

    if sub.empty:
        return

    x = np.arange(len(sub))
    width = 0.8 / max(1, len(metrics))

    plt.figure(figsize=(10, 6))
    for i, metric in enumerate(metrics):
        if metric not in sub.columns or sub[metric].isna().all():
            continue
        offset = (i - (len(metrics) - 1) / 2) * width
        values = sub[metric].astype(float)
        plt.bar(x + offset, values, width, label=metric.upper().replace("_", " "))
        for xi, value in zip(x + offset, values):
            if not np.isnan(value):
                plt.text(xi, value + 0.004, f"{value:.3f}", ha="center", fontsize=8)

    plt.xticks(x, sub["pipeline"])
    plt.xlabel("Pipeline")
    plt.ylabel(ylabel)
    plt.title(title)
    if ylim:
        plt.ylim(*ylim)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_loss_bar(df: pd.DataFrame, split: str, output_path: Path) -> None:
    sub = df[df["split"] == split].sort_values("pipeline")
    if sub.empty or "loss" not in sub.columns or sub["loss"].isna().all():
        return

    plt.figure(figsize=(9, 5))
    plt.bar(sub["pipeline"], sub["loss"])
    plt.xlabel("Pipeline")
    plt.ylabel("Loss")
    plt.title(f"Loss no conjunto de {split}")
    for i, v in enumerate(sub["loss"]):
        if not np.isnan(v):
            plt.text(i, v + 0.002, f"{v:.3f}", ha="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_gain_vs_p0(df: pd.DataFrame, split: str, metric: str, output_path: Path) -> None:
    sub = df[df["split"] == split].sort_values("pipeline").copy()
    if sub.empty or metric not in sub.columns or sub[metric].isna().all():
        return

    p0 = sub.loc[sub["pipeline"] == "P0", metric]
    if p0.empty or pd.isna(p0.iloc[0]):
        return

    baseline = float(p0.iloc[0])
    sub[f"gain_{metric}"] = sub[metric] - baseline

    plt.figure(figsize=(9, 5))
    plt.bar(sub["pipeline"], sub[f"gain_{metric}"])
    plt.axhline(0, linewidth=0.8)
    plt.xlabel("Pipeline")
    plt.ylabel(f"Variação de {metric.upper()} em relação à P0")
    plt.title(f"Ganho de {metric.upper()} em relação à baseline P0 ({split})")
    for i, v in enumerate(sub[f"gain_{metric}"]):
        if not np.isnan(v):
            plt.text(i, v + (0.001 if v >= 0 else -0.003), f"{v:+.3f}", ha="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_best_epoch(df: pd.DataFrame, output_path: Path) -> None:
    sub = df[(df["split"] == "val") & df["epoch"].notna()].sort_values("pipeline")
    if sub.empty:
        return

    plt.figure(figsize=(9, 5))
    plt.plot(sub["pipeline"], sub["epoch"], marker="o")
    plt.xlabel("Pipeline")
    plt.ylabel("Época")
    plt.title("Época com melhor desempenho de validação")
    for x, y in zip(sub["pipeline"], sub["epoch"]):
        plt.text(x, y + 1, str(int(y)), ha="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def find_per_image_metrics(project_root: Path, pipeline: str) -> Optional[Path]:
    names = [
        "per_image_metrics.csv", "test_per_image_metrics.csv",
        "metrics_per_image.csv", "per_sample_metrics.csv",
    ]
    candidates: List[Path] = []
    for d in candidate_dirs(project_root, pipeline):
        candidates.extend([d / name for name in names])
    candidates.extend(project_root.glob(f"**/{pipeline}*per*image*metrics*.csv"))
    candidates.extend(project_root.glob(f"**/{pipeline}*per*sample*metrics*.csv"))
    return find_first_existing(candidates)


def plot_boxplot_per_image(project_root: Path, pipelines: List[str], out_path: Path) -> None:
    values = []
    labels = []

    for pipeline in pipelines:
        path = find_per_image_metrics(project_root, pipeline)
        if not path:
            continue

        df = pd.read_csv(path)
        df.columns = [normalize_key(c) for c in df.columns]
        dice_col = next((c for c in ["dice", "dice_score", "test_dice", "mean_dice"] if c in df.columns), None)
        if not dice_col:
            continue

        vals = pd.to_numeric(df[dice_col], errors="coerce").dropna()
        if not vals.empty:
            values.append(vals.values)
            labels.append(pipeline)

    if not values:
        return

    plt.figure(figsize=(9, 5))
    plt.boxplot(values, labels=labels, showmeans=True)
    plt.xlabel("Pipeline")
    plt.ylabel("Dice por imagem")
    plt.title("Distribuição do Dice por imagem no conjunto de teste")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def make_ablation_table(df: pd.DataFrame, out_dir: Path) -> None:
    if df.empty:
        return

    # Prioriza teste; se não houver, usa validação.
    test = df[df["split"] == "test"].copy()
    val = df[df["split"] == "val"].copy()

    rows = []
    for pipeline in sorted(df["pipeline"].dropna().unique()):
        row = {
            "Pipeline": pipeline,
            "Descrição": PIPELINE_DESCRIPTIONS.get(pipeline, ""),
        }

        v = val[val["pipeline"] == pipeline]
        t = test[test["pipeline"] == pipeline]

        if not v.empty:
            rv = v.iloc[0]
            row.update({
                "Val Dice": rv.get("dice", np.nan),
                "Val IoU": rv.get("iou", np.nan),
                "Val Loss": rv.get("loss", np.nan),
                "Melhor Época": rv.get("epoch", np.nan),
            })

        if not t.empty:
            rt = t.iloc[0]
            row.update({
                "Test Dice": rt.get("dice", np.nan),
                "Test IoU": rt.get("iou", np.nan),
                "Test Pixel Acc.": rt.get("pixel_accuracy", np.nan),
                "Test Loss": rt.get("loss", np.nan),
            })

        rows.append(row)

    table = pd.DataFrame(rows)

    # Arredondamento apenas para exportação.
    export = table.copy()
    for col in export.columns:
        if col not in ["Pipeline", "Descrição"]:
            export[col] = pd.to_numeric(export[col], errors="coerce").round(4)

    export.to_csv(out_dir / "tabela_ablacao.csv", index=False, encoding="utf-8-sig")

    latex = export.to_latex(
        index=False,
        escape=False,
        na_rep="--",
        caption="Tabela de ablação das pipelines avaliadas.",
        label="tab:ablacao_pipelines",
        float_format="%.4f",
    )
    (out_dir / "tabela_ablacao.tex").write_text(latex, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path("."), help="Raiz do projeto.")
    parser.add_argument("--out-dir", type=Path, default=Path("report_figures"), help="Pasta de saída.")
    parser.add_argument("--pipelines", nargs="+", default=PIPELINES_DEFAULT, help="Lista de pipelines.")
    parser.add_argument("--summary-csv", type=Path, default=None, help="CSV manual com métricas consolidadas.")
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.summary_csv:
        summary = pd.read_csv(args.summary_csv)
        summary.columns = [normalize_key(c) for c in summary.columns]
        if "pipeline" not in summary.columns:
            raise ValueError("O CSV precisa conter a coluna 'pipeline'.")
        if "split" not in summary.columns:
            raise ValueError("O CSV precisa conter a coluna 'split'.")
        for col in ["loss", "dice", "iou", "pixel_accuracy", "epoch"]:
            if col in summary.columns:
                summary[col] = pd.to_numeric(summary[col], errors="coerce")
        summary["description"] = summary["pipeline"].map(PIPELINE_DESCRIPTIONS).fillna("")
    else:
        summary = collect_summary(project_root, args.pipelines)

    if summary.empty:
        raise FileNotFoundError(
            "Nenhum arquivo de métricas foi encontrado. "
            "Use --summary-csv ou verifique se existem arquivos como outputs/P0/history.csv "
            "e outputs/P0/test_metrics.json."
        )

    summary.to_csv(out_dir / "metricas_consolidadas.csv", index=False, encoding="utf-8-sig")

    plot_grouped_bar(
        summary,
        split="test",
        metrics=["dice", "iou", "pixel_accuracy"],
        title="Comparação das métricas no conjunto de teste",
        output_path=out_dir / "fig_bar_test_metrics.png",
        ylim=(0, 1.05),
    )

    plot_grouped_bar(
        summary,
        split="val",
        metrics=["dice", "iou", "pixel_accuracy"],
        title="Comparação das métricas no conjunto de validação",
        output_path=out_dir / "fig_bar_val_metrics.png",
        ylim=(0, 1.05),
    )

    plot_loss_bar(summary, "test", out_dir / "fig_loss_test.png")
    plot_loss_bar(summary, "val", out_dir / "fig_loss_val.png")
    plot_gain_vs_p0(summary, "test", "dice", out_dir / "fig_dice_gain_vs_p0_test.png")
    plot_gain_vs_p0(summary, "val", "dice", out_dir / "fig_dice_gain_vs_p0_val.png")
    plot_gain_vs_p0(summary, "test", "iou", out_dir / "fig_iou_gain_vs_p0_test.png")
    plot_gain_vs_p0(summary, "val", "iou", out_dir / "fig_iou_gain_vs_p0_val.png")
    plot_best_epoch(summary, out_dir / "fig_best_epoch.png")
    plot_boxplot_per_image(project_root, args.pipelines, out_dir / "fig_boxplot_dice_per_image.png")
    make_ablation_table(summary, out_dir)

    print(f"[OK] Figuras e tabelas salvas em: {out_dir}")
    print("[OK] Arquivo consolidado:", out_dir / "metricas_consolidadas.csv")


if __name__ == "__main__":
    main()
