# MoE Medical Vision

Implementacion del proyecto de Mixture of Experts (MoE) para vision medica multimodal con 3 expertos 2D, 2 expertos 3D, un backbone ViT para routing y un dashboard en Streamlit.

## Contenido del repo

- `notebooks/`: flujo principal del proyecto, desde verificacion del pod hasta integracion end-to-end.
- `src/`: datasets, modelos, losses y utilidades de entrenamiento.
- `scripts/`: utilidades auxiliares para entrenamiento, precomputo, tests y recuperacion.
- `checkpoints/`: pesos entrenados y metricas.
- `embeddings/`: CLS tokens y artefactos del router.
- `app.py`: demo de inferencia en Streamlit.

## Requisitos

- Python 3.10 o 3.11.
- GPU NVIDIA recomendada para entrenamiento (Usada una 4090 por medio de una instacia de vast.ai).
- Estructura esperada en el pod: `/workspace/moe_medical_vision`.
- Datasets descomprimidos en `/workspace/moe_medical_vision/data/raw`.

Instalacion base:

```bash
pip install -r requirements.txt
```

Si tu entorno usa CUDA especifica, instala `torch`, `torchvision` y `torchaudio` con las ruedas compatibles antes o despues del `requirements.txt`, segun el selector oficial de PyTorch.

Si vas a descargar datasets desde Kaggle o crear repos en Hugging Face, exporta antes tus credenciales en el entorno:

```bash
export KAGGLE_USERNAME=tu_usuario
export KAGGLE_KEY=tu_api_key
export HF_TOKEN=tu_token
```

## Reproducibilidad

El proyecto usa semilla fija `42` en los notebooks principales y en el codigo fuente:

- `notebooks/01_train_experts_2D.ipynb`
- `notebooks/01B_train_experts_3D.ipynb`
- `notebooks/02_backbone_y_routers.ipynb`
- `notebooks/03_moe_system.ipynb`
- `src/config.py`
- `src/train/train_3d.py`

Para entrenamiento 3D se dejo `cudnn.deterministic = True` y `cudnn.benchmark = False` para que el comportamiento quede alineado con la rubric de reproducibilidad.

## Orden recomendado de ejecucion

1. `notebooks/00_setup_verificacion.ipynb`
2. `notebooks/01_train_experts_2D.ipynb`
3. `notebooks/01B_train_experts_3D.ipynb`
4. `notebooks/02_backbone_y_routers.ipynb`
5. `notebooks/03_moe_system.ipynb`

Notebooks auxiliares:

- `notebooks/download_datasets.ipynb`: descarga inicial de datasets.
- `notebooks/post_reboot.ipynb`: recuperacion rapida despues de reinicio del pod.

## Estructura de datos esperada

El codigo asume las siguientes rutas base:

```text
/workspace/moe_medical_vision/
|-- app.py
|-- checkpoints/
|-- embeddings/
|-- notebooks/
|-- scripts/
|-- src/
`-- data/
    `-- raw/
        |-- nih/
        |-- isic/
        |-- osteoporosis/
        |-- luna16/
        `-- pancreatic/
```

Las rutas exactas usadas por el proyecto se centralizan en `src/config.py`.

## Flujo resumido

### 1. Verificacion del pod

`00_setup_verificacion.ipynb` valida espacio en disco, imports, rutas y deteccion basica de datasets.

### 2. Entrenamiento de expertos

- `01_train_experts_2D.ipynb`: NIH, ISIC 2019 y osteoarthritis.
- `01B_train_experts_3D.ipynb`: LUNA16 y pancreatic.

Los mejores pesos quedan en `checkpoints/`.

### 3. Backbone y routers

`02_backbone_y_routers.ipynb` extrae embeddings CLS y entrena/compra routers lineal, GMM, Naive Bayes y FAISS k-NN.

Los artefactos intermedios quedan en `embeddings/`.

### 4. Integracion MoE

`03_moe_system.ipynb` integra expertos, router, auxiliary loss y fine-tuning del sistema global.

## Demo

Para levantar la app:

```bash
streamlit run app.py
```

La demo espera checkpoints y artefactos del router ya generados en `checkpoints/` y `embeddings/`.

Para visualizar lo de manera más sencilla, se puede abrir este link donde alojamos una version desplegada del dashboard: https://use-instrumental-sie-sie.trycloudflare.com/

## Notas 

- Ejecuta siempre `00_setup_verificacion.ipynb` antes de entrenar en un pod nuevo.
- Si cambias rutas de datos o checkpoints, hazlo desde `src/config.py`.
- `scripts/` contiene utilidades de apoyo, pero el flujo evaluable principal esta organizado en `notebooks/`.

---

Creado por [Sebastian Belalcazar](https://github.com/SEBASBELMOS) y [Manuel Gruezo](https://github.com/manuel-gruezo)