
<p align="center"><a href="https://github.com/VoxleOne/Auto-Annotate-BR#readme"><img src="https://github.com/VoxleOne/Auto-Annotate-BR/blob/main/asset/logos/auto-annotate-logo-transparent.png" alt="Auto-Annotate Logo" height="240"></a></p>
<h1 align="center">Auto-Annotate-BR</h1>
<p align="center">Anote imagens de todo um diretório, automaticamente, com um único comando. </p>

<p align="center"><img src="https://img.shields.io/badge/version-v3.0.0-brightgreen?style=plastic" alt="Auto-Annotate Version"> <img src="https://img.shields.io/github/repo-size/VoxleOne/Auto-Annotate-BR?style=plastic" alt="repo size"> <img src="https://img.shields.io/github/stars/VoxleOne/Auto-Annotate-BR?&style=social" alt="stars"> <img src="https://img.shields.io/badge/python-3.9%20%7C%203.10-blue?style=plastic" alt="Python Version"></p>




Para uma explicação mais detalhada e aplicação do código, consulte [este artigo no Medium](https://medium.com/analytics-vidhya/automated-image-annotation-using-auto-annotate-tool-f8fff8ea4900).<a href="https://medium.com/analytics-vidhya/automated-image-annotation-using-auto-annotate-tool-f8fff8ea4900">![](https://img.shields.io/badge/Medium-12100E?style=for-the-badge&logo=medium&logoColor=white) </a>


**Tão simples quanto dizer: "Anote todas as placas de rua (rótulo) no dataset (diretório) do carro autônomo".**
Toda e qualquer imagem no diretório do dataset que contenha uma placa de rua é então filtrada e a anotação de segmentação é executada em um único comando.

Auto-Annotate-BR fornece anotação automática por máscaras de segmentação baseada nos rótulos definidos para os objetos nas imagens de um diretório. A ferramenta é capaz de fornecer anotações automatizadas para os rótulos definidos no dataset COCO e também oferece suporte a rótulos personalizados. A partir da versão 3.0, ela usa **YOLOv8 (Ultralytics)** como backend padrão, substituindo a antiga arquitetura Mask R-CNN. O backend legado Mask R-CNN continua disponível via `--backend maskrcnn`.

![Working Sample: ANNOTATE CUSTOM](asset/AutoAnnotate-Working_LowRes.png)

A ferramenta de anotação automática funciona em dois modos - COCO e Personalizado.
* **Anotação de rótulo COCO** - Nenhum treinamento de modelo é necessário. Os pesos COCO são baixados automaticamente pelo YOLOv8. Aponte para o diretório correto e as anotações estão prontas.
* **Anotação de rótulo personalizado** - Treine o modelo para seu rótulo personalizado. Use os pesos e anote.

NOTA: Gentileza consultar o arquivo [knownIssues.md](knownIssues.md) no repositório, para checar os problemas conhecidos e sua resolução. Sinta-se à vontade para contribuir caso encontre erros/problemas durante a instalação e uso da ferramenta.

## Requisitos

* **Python**: 3.9 ou 3.10 (testado)
* **SO**: Linux (Ubuntu 20.04+), macOS, Windows 10+
* **PyTorch**: ≥ 2.0 (instalado automaticamente com `ultralytics`)
* **Docker**: Opcionalmente, use o `Dockerfile` incluído para ambiente reproduzível

### Backend legado (Mask R-CNN)

Se precisar usar o backend Mask R-CNN (`--backend maskrcnn`), instale também:
* **TensorFlow**: 2.10–2.15
* Veja `requirements-maskrcnn.txt`

## Backends de Detecção

| Backend | Flag | Dependência | Status |
|---------|------|------------|--------|
| **YOLOv8** | `--backend yolov8` (padrão) | `ultralytics` + PyTorch | ✅ Ativo |
| **Mask R-CNN** | `--backend maskrcnn` | TensorFlow 2.x | ⚠️ Depreciado |

### Tamanhos de modelo YOLOv8

Use `--model_size` para escolher o modelo:

| Tamanho | Flag | Modelo | Velocidade | Precisão |
|---------|------|--------|------------|----------|
| Nano | `--model_size nano` | yolov8n-seg.pt | ⚡ Mais rápido | Menor |
| Small | `--model_size small` | yolov8s-seg.pt | Rápido | Boa |
| Medium | `--model_size medium` | yolov8m-seg.pt | Equilibrado | Melhor |
| Large | `--model_size large` | yolov8l-seg.pt | Lento | Alta |
| XLarge | `--model_size xlarge` | yolov8x-seg.pt | Mais lento | Máxima |

## FORMATO JSON PARA ANOTAÇÃO

### Amostra JSON: 

```json
[
  {
    "filename": "bird_house_in_lawn.jpg",
    "id": 1,
    "label": "Bird House",
    "bbox": [ 111.5, 122.5, 73, 66 ],
    "segmentation": [
      [ 167, 188.5, 174, 185.5, 177.5, 181, 177.5, 157, 183.5, 154, 184.5, 149, 159, 124.5, 150, 122.5, 
        142, 124.5, 131, 131.5, 111.5, 150, 111.5, 156, 116.5, 162, 116.5, 184, 121, 188.5, 167, 188.5
      ]
    ]
  }
]
```

### Formato genérico JSON:
```
[
  {
    "filename": image_file_name,
    "id": id_of_the_image,
    "label": label_to_search_and_annotate,
    "bbox": [ x, y, w, h ], -- x,y coordinate of top left point of bounding box
                            -- w,h width and height of the bounding box
                            Format correspond to coco json response format for bounding box
    "segmentation": [
      [ x1, y1, x2, y2,...] -- For X,Y belonging to the pixel location of the mask segment
                            -- Format correspond to coco json response format for segmentation
    ]
  }
]
```

###
IMAGEM ORIGINAL            |  IMAGEM 'MASCARADA'
:-------------------------:|:-------------------------:
![](asset/bird_house_in_lawn.jpg)  |  ![](asset/bird_house_in_lawn_masked.jpg)


## Instalação
1. Clone este repositório.

2. Instale as dependências.
   ```bash
   pip3 install -r requirements.txt
   ```

   Se precisar do backend legado Mask R-CNN:
   ```bash
   pip3 install -r requirements.txt -r requirements-maskrcnn.txt
   ```

3. Execute os comandos abaixo conforme o modo de uso.

## Uso

### Anotando no MS COCO (YOLOv8 — padrão)

Os pesos COCO são baixados automaticamente. Basta passar `--weights=coco`:

```bash
# Anotar com YOLOv8 (padrão)
python3 annotate.py annotateCoco \
  --image_directory=/caminho/para/imagens/ \
  --label=person \
  --weights=coco

# Multi-rótulo com modelo grande e formato YOLO
python3 annotate.py annotateCoco \
  --image_directory=/caminho/para/imagens/ \
  --label=person,car,dog \
  --weights=coco \
  --model_size=large \
  --output_format=yolo \
  --min_confidence=0.8
```

### Anotando com backend legado (Mask R-CNN)

```bash
python3 annotate.py annotateCoco \
  --image_directory=/caminho/para/imagens/ \
  --label=person \
  --weights=/caminho/para/mask_rcnn_coco.h5 \
  --backend=maskrcnn
```

### Anotando em imagens personalizadas

```bash
# YOLOv8 com pesos personalizados
python3 annotate.py annotateCustom \
  --image_directory=/caminho/para/imagens/ \
  --label=meu_rotulo \
  --weights=/caminho/para/meus_pesos.pt

# Mask R-CNN com pesos personalizados (legado)
python3 annotate.py annotateCustom \
  --image_directory=/caminho/para/imagens/ \
  --label=meu_rotulo \
  --weights=/caminho/para/meus_pesos.h5 \
  --backend=maskrcnn
```

### Flags opcionais

| Flag | Descrição |
|------|-----------|
| `--backend yolov8\|maskrcnn` | Backend de detecção (padrão: yolov8) |
| `--model_size nano\|small\|medium\|large\|xlarge` | Tamanho do modelo YOLOv8 (padrão: medium) |
| `--output_format auto-annotate\|coco\|voc\|yolo` | Formato de saída (padrão: auto-annotate) |
| `--min_confidence 0.0-1.0` | Limiar de confiança (padrão: 0.7) |
| `--device cpu\|gpu` | Forçar uso de CPU ou GPU |
| `--no-overwrite` | Pular anotação se arquivo já existir |
| `--labels_file /caminho/labels.txt` | Carregar rótulos de arquivo |

### Treinando seu próprio dataset

#### YOLOv8 (recomendado)

```bash
# Treinar um novo modelo YOLOv8
python3 customTrain.py train \
  --dataset=/caminho/para/dataset/ \
  --weights=coco \
  --backend=yolov8 \
  --model_size=medium \
  --epochs=30 \
  --label=meu_rotulo
```

O dataset deve seguir a estrutura YOLO:
```
dataset/
├── train/
│   ├── images/
│   └── labels/
└── val/
    ├── images/
    └── labels/
```

#### Converter anotações VIA para formato YOLO

Se você tem anotações no formato VIA JSON (usado pelo Mask R-CNN), converta-as:

```bash
python3 customTrain.py convert --dataset=/caminho/para/dataset/
```

#### Mask R-CNN (legado)

```bash
python3 customTrain.py train \
  --dataset=/caminho/para/dataset/ \
  --weights=coco \
  --backend=maskrcnn
```

### Docker

```bash
# Build (YOLOv8)
docker build -t auto-annotate-br .

# Build com ambos os backends
docker build -f Dockerfile.maskrcnn -t auto-annotate-br-full .

# Executar
docker run --rm -v /caminho/para/imagens:/data auto-annotate-br \
  annotateCoco --image_directory=/data --label=person --weights=coco
```

### :clap: Apoiadores

### :twisted_rightwards_arrows: Forkers 
[![Forkers repo roster for @VoxleOne/Auto-Annotate-BR](https://reporoster.com/forks/dark/VoxleOne/Auto-Annotate-BR)](https://github.com/VoxleOne/Auto-Annotate-BR/network/members)

##

[🤝 NOSSO BLOG](https://www.voxleone.com/blog/)

##
