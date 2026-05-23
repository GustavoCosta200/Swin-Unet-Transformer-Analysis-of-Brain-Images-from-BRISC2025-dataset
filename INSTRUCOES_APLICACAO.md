# Como aplicar a atualização da pipeline P4

Copie os arquivos desta pasta para a raiz do seu projeto, substituindo os arquivos atuais de mesmo caminho:

```text
configs/default.yaml
src/preprocessing.py
src/augmentations.py
src/visualize_preprocessing.py
scripts/run_all_pipelines.py
README.md
```

Depois, execute:

```bash
python -m src.sanity_check --config configs/default.yaml
python -m src.visualize_preprocessing --config configs/default.yaml
python -m src.train --config configs/default.yaml
```

Para rodar somente P3 e P4, use:

```bash
python scripts/run_all_pipelines.py --config configs/default.yaml --pipelines P3 P4
```

Para avaliar a P4:

```bash
python -m src.evaluate --config configs/default.yaml --checkpoint checkpoints/P4_best.pt
```
