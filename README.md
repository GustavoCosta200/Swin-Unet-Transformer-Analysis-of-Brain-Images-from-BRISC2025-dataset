# TCC 2 - Segmentação de tumores cerebrais com Swin-Unet oficial

Este projeto implementa o desenho experimental descrito na metodologia da monografia, comparando quatro pipelines de entrada sobre **uma única arquitetura fixa**, o **Swin-Unet oficial de Cao et al.**:

- **P0 - Baseline:** redimensionamento, binarização da máscara e normalização;
- **P1 - Suavização gaussiana:** filtro gaussiano + P0;
- **P2 - Mediana e CLAHE:** filtro de mediana + CLAHE + P0;
- **P3 - Pré-processamento e aumento de dados:** P2 + augmentation aplicado apenas no treino.

Por rastreabilidade bibliográfica, a arquitetura não é reimplementada localmente. O projeto carrega diretamente a implementação oficial do artigo:

> Cao et al. (2022), *Swin-Unet: Unet-like Pure Transformer for Medical Image Segmentation*.

## 1. Estrutura esperada do dataset BRISC

```text
segmentation_task/
  train/
    images/
      arquivo.jpg
    masks/
      arquivo.png
  test/
    images/
      arquivo.jpg
    masks/
      arquivo.png
```

## 2. Instalação

```bash
pip install -r requirements.txt
```

## 3. Baixar a implementação oficial do Swin-Unet

```bash
python scripts/setup_official_swin_unet.py
```

O repositório será esperado em:

```text
external/Swin-Unet/
```

## 4. Configuração

Edite `configs/default.yaml` para ajustar:

- caminho do dataset;
- pipeline a ser executada (`P0`, `P1`, `P2`, `P3`);
- caminho do YAML oficial do Swin-Unet;
- pesos pré-treinados, caso sejam utilizados.

A configuração padrão usa imagens `224 x 224`, compatíveis com a configuração oficial do Swin-Unet. Como o BRISC é lido em tons de cinza, cada imagem é replicada em três canais para manter compatibilidade com a arquitetura oficial e com eventuais pesos pré-treinados.

## 5. Verificação inicial

```bash
python -m src.sanity_check --config configs/default.yaml
python -m src.visualize_preprocessing --config configs/default.yaml
```

## 6. Treinar um pipeline

Defina a pipeline em `configs/default.yaml` e execute:

```bash
python -m src.train --config configs/default.yaml
```

## 7. Avaliar o melhor modelo

```bash
python -m src.evaluate --config configs/default.yaml --checkpoint checkpoints/P0_best.pt
```

## 8. Rodar P0, P1, P2 e P3 em sequência

```bash
python scripts/run_all_pipelines.py --config configs/default.yaml
```

## 9. Observações metodológicas

- A arquitetura é mantida fixa entre os experimentos; variam apenas as pipelines de entrada.
- A mesma divisão treino/validação é preservada para assegurar comparabilidade.
- O aumento de dados é aplicado exclusivamente ao conjunto de treinamento, conforme a metodologia.
- As métricas calculadas são Dice, IoU, Pixel Accuracy, Precision e Recall; a monografia pode manter Dice, IoU e Pixel Accuracy como métricas principais e usar as demais como apoio interpretativo.

## 10. Referências

As referências BibTeX do artigo e do repositório oficial estão em:

```text
references/swin_unet.bib
```
