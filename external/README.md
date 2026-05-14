# Dependência externa: implementação oficial do Swin-Unet

Este projeto utiliza a implementação oficial do artigo:

- Cao et al. (2022), *Swin-Unet: Unet-like Pure Transformer for Medical Image Segmentation*.
- Repositório oficial: `HuCaoFighting/Swin-Unet`.

Por motivos de rastreabilidade bibliográfica, o código da arquitetura não foi reimplementado neste projeto.
A arquitetura é carregada diretamente do repositório oficial, esperado em:

```text
external/Swin-Unet/
```

Para obter o código na sua máquina:

```bash
python scripts/setup_official_swin_unet.py
```

Também é possível baixar/clonar manualmente o repositório oficial e colocá-lo nessa pasta.
