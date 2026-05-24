# Scripts para gráficos da monografia - P0 a P4

Copie os arquivos desta pasta para a pasta `scripts/` do projeto no GitHub.

## 1. Dependências

```bash
pip install pandas matplotlib pillow numpy
```

## 2. Gerar gráficos quantitativos

Na raiz do projeto:

```bash
python scripts/generate_report_graphs.py --project-root . --out-dir report_figures
```

O script tenta localizar automaticamente métricas em caminhos como:

```text
outputs/P0/history.csv
outputs/P0/test_metrics.json
outputs/P1/history.csv
outputs/P1/test_metrics.json
...
results/P0/history.csv
logs/P0/history.csv
checkpoints/P0_best_metrics.json
```

## 3. Gerar comparação visual

Antes, rode a avaliação salvando predições das pipelines. O script espera algo como:

```text
outputs/P0/predictions/
outputs/P1/predictions/
outputs/P2/predictions/
outputs/P3/predictions/
outputs/P4/predictions/
```

Depois:

```bash
python scripts/generate_qualitative_comparison.py --project-root . --out-dir report_figures/qualitative
```

Se o dataset estiver fora do padrão, informe os caminhos:

```bash
python scripts/generate_qualitative_comparison.py \
  --images-dir D:/TCC2/brisc2025/segmentation_task/test/images \
  --masks-dir D:/TCC2/brisc2025/segmentation_task/test/masks \
  --pred-root outputs \
  --out-dir report_figures/qualitative
```

## 4. Gerar tudo

```bash
python scripts/run_all_report_figures.py --project-root . --out-dir report_figures
```

## 5. Arquivos gerados

```text
report_figures/
  fig_bar_test_metrics.png
  fig_bar_val_metrics.png
  fig_loss_test.png
  fig_loss_val.png
  fig_dice_gain_vs_p0_test.png
  fig_dice_gain_vs_p0_val.png
  fig_iou_gain_vs_p0_test.png
  fig_iou_gain_vs_p0_val.png
  fig_best_epoch.png
  fig_boxplot_dice_per_image.png
  tabela_ablacao.csv
  tabela_ablacao.tex
  metricas_consolidadas.csv
  qualitative/
    comparacao_visual_01_...
```

## 6. CSV manual, caso seus arquivos tenham outro formato

Crie `report_metrics.csv`:

```csv
pipeline,split,loss,dice,iou,pixel_accuracy,epoch
P0,val,0.1317,0.7739,0.6823,,138
P1,val,0.1276,0.7797,0.6854,,116
P2,val,0.1295,0.7756,0.6804,,128
P3,val,0.1172,0.7972,0.7020,,134
P4,val,,,,,
P0,test,,,,,
P1,test,,,,,
P2,test,,,,,
P3,test,,,,,
P4,test,,,,,
```

E rode:

```bash
python scripts/generate_report_graphs.py --summary-csv report_metrics.csv --out-dir report_figures
```
